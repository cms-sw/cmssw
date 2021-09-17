/// Take:
//    - a PF candidate collection (which uses bugged HCAL respcorrs)
//    - respCorrs values from fixed GT and bugged GT
//  Produce:
//    - a new PFCandidate collection containing the recalibrated PFCandidates in HF and where the neutral had pointing to problematic cells are removed
//    - a second PFCandidate collection with just those discarded hadrons
//    - a ValueMap<reco::PFCandidateRef> that maps the old to the new, and vice-versa

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Association.h"
#include <iostream>

struct HEChannel {
  float eta;
  float phi;
  float ratio;
  HEChannel(float eta, float phi, float ratio) : eta(eta), phi(phi), ratio(ratio) {}
};
struct HFChannel {
  int ieta;
  int iphi;
  int depth;
  float ratio;
  HFChannel(int ieta, int iphi, int depth, float ratio) : ieta(ieta), iphi(iphi), depth(depth), ratio(ratio) {}
};

class PFCandidateRecalibrator : public edm::stream::EDProducer<> {
public:
  PFCandidateRecalibrator(const edm::ParameterSet&);
  ~PFCandidateRecalibrator() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginRun(const edm::Run& iRun, edm::EventSetup const& iSetup) override;
  void endRun(const edm::Run& iRun, edm::EventSetup const& iSetup) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

  edm::ESWatcher<HcalRecNumberingRecord> hcalDbWatcher_;
  edm::ESWatcher<HcalRespCorrsRcd> hcalRCWatcher_;

  edm::EDGetTokenT<reco::PFCandidateCollection> pfcandidates_;

  std::vector<HEChannel> badChHE_;
  std::vector<HFChannel> badChHF_;

  float shortFibreThr_;
  float longFibreThr_;
};

PFCandidateRecalibrator::PFCandidateRecalibrator(const edm::ParameterSet& iConfig)
    : pfcandidates_(consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("pfcandidates"))),
      shortFibreThr_(iConfig.getParameter<double>("shortFibreThr")),
      longFibreThr_(iConfig.getParameter<double>("longFibreThr")) {
  produces<reco::PFCandidateCollection>();
  produces<reco::PFCandidateCollection>("discarded");
  produces<edm::ValueMap<reco::PFCandidateRef>>();
}

void PFCandidateRecalibrator::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  if (hcalDbWatcher_.check(iSetup) || hcalRCWatcher_.check(iSetup)) {
    //Get Calib Constants from current GT
    edm::ESHandle<HcalDbService> gtCond;
    iSetup.get<HcalDbRecord>().get(gtCond);

    //Get Calib Constants from bugged tag
    edm::ESHandle<HcalTopology> htopo;
    iSetup.get<HcalRecNumberingRecord>().get(htopo);
    const HcalTopology* theHBHETopology = htopo.product();

    edm::ESHandle<HcalRespCorrs> buggedCond;
    iSetup.get<HcalRespCorrsRcd>().get("bugged", buggedCond);
    HcalRespCorrs buggedRespCorrs(*buggedCond.product());
    buggedRespCorrs.setTopo(theHBHETopology);

    //access calogeometry
    edm::ESHandle<CaloGeometry> calogeom;
    iSetup.get<CaloGeometryRecord>().get(calogeom);
    const CaloGeometry* cgeo = calogeom.product();
    const HcalGeometry* hgeom =
        static_cast<const HcalGeometry*>(cgeo->getSubdetectorGeometry(DetId::Hcal, HcalForward));

    //reset the bad channel containers
    badChHE_.clear();
    badChHF_.clear();

    //fill bad cells HE (use eta, phi)
    const std::vector<DetId>& cellsHE = hgeom->getValidDetIds(DetId::Detector::Hcal, HcalEndcap);
    for (auto id : cellsHE) {
      float currentRespCorr = gtCond->getHcalRespCorr(id)->getValue();
      float buggedRespCorr = buggedRespCorrs.getValues(id)->getValue();
      if (buggedRespCorr == 0.)
        continue;

      float ratio = currentRespCorr / buggedRespCorr;
      if (std::abs(ratio - 1.f) > 0.001) {
        GlobalPoint pos = hgeom->getPosition(id);
        badChHE_.push_back(HEChannel(pos.eta(), pos.phi(), ratio));
      }
    }

    //fill bad cells HF (use ieta, iphi)
    auto const& cellsHF = hgeom->getValidDetIds(DetId::Detector::Hcal, HcalForward);
    for (auto id : cellsHF) {
      float currentRespCorr = gtCond->getHcalRespCorr(id)->getValue();
      float buggedRespCorr = buggedRespCorrs.getValues(id)->getValue();
      if (buggedRespCorr == 0.)
        continue;

      float ratio = currentRespCorr / buggedRespCorr;
      if (std::abs(ratio - 1.f) > 0.001) {
        HcalDetId dummyId(id);
        badChHF_.push_back(HFChannel(dummyId.ieta(), dummyId.iphi(), dummyId.depth(), ratio));
      }
    }
  }
}

void PFCandidateRecalibrator::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {}

void PFCandidateRecalibrator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //access calogeometry
  edm::ESHandle<CaloGeometry> calogeom;
  iSetup.get<CaloGeometryRecord>().get(calogeom);
  const CaloGeometry* cgeo = calogeom.product();
  const HcalGeometry* hgeom = static_cast<const HcalGeometry*>(cgeo->getSubdetectorGeometry(DetId::Hcal, HcalForward));

  //access PFCandidates
  edm::Handle<reco::PFCandidateCollection> pfcandidates;
  iEvent.getByToken(pfcandidates_, pfcandidates);

  int nPfCand = pfcandidates->size();
  std::unique_ptr<reco::PFCandidateCollection> copy(new reco::PFCandidateCollection());
  std::unique_ptr<reco::PFCandidateCollection> discarded(new reco::PFCandidateCollection());
  copy->reserve(nPfCand);
  std::vector<int> oldToNew(nPfCand), newToOld, badToOld;
  newToOld.reserve(nPfCand);

  LogDebug("PFCandidateRecalibrator") << "NEW EV:";

  //loop over PFCandidates
  int i = -1;
  for (const reco::PFCandidate& pf : *pfcandidates) {
    ++i;
    float absEta = std::abs(pf.eta());

    //deal with HE
    if (pf.particleId() == reco::PFCandidate::ParticleType::h0 &&
        !badChHE_.empty() &&  //don't touch if no miscalibration is found
        absEta > 1.4 && absEta < 3.) {
      bool toKill = false;
      for (auto const& badIt : badChHE_)
        if (reco::deltaR2(pf.eta(), pf.phi(), badIt.eta, badIt.phi) < 0.07)
          toKill = true;

      if (toKill) {
        discarded->push_back(pf);
        oldToNew[i] = (-discarded->size());
        badToOld.push_back(i);
        continue;
      } else {
        copy->push_back(pf);
        oldToNew[i] = (copy->size());
        newToOld.push_back(i);
      }
    }
    //deal with HF
    else if ((pf.particleId() == reco::PFCandidate::ParticleType::h_HF ||
              pf.particleId() == reco::PFCandidate::ParticleType::egamma_HF) &&
             !badChHF_.empty() &&  //don't touch if no miscalibration is found
             absEta >= 3.) {
      const math::XYZPointF& ecalPoint = pf.positionAtECALEntrance();
      GlobalPoint ecalGPoint(ecalPoint.X(), ecalPoint.Y(), ecalPoint.Z());
      HcalDetId closestDetId(hgeom->getClosestCell(ecalGPoint));

      if (closestDetId.subdet() == HcalForward) {
        HcalDetId hDetId(closestDetId.subdet(), closestDetId.ieta(), closestDetId.iphi(), 1);  //depth1

        //raw*calEnergy() is the same as *calEnergy() - no corrections are done for HF
        float longE = pf.rawEcalEnergy() + pf.rawHcalEnergy() / 2.;  //depth1
        float shortE = pf.rawHcalEnergy() / 2.;                      //depth2

        float ecalEnergy = pf.rawEcalEnergy();
        float hcalEnergy = pf.rawHcalEnergy();
        float totEnergy = ecalEnergy + hcalEnergy;
        float totEnergyOrig = totEnergy;

        bool toKill = false;

        for (auto const& badIt : badChHF_) {
          if (hDetId.ieta() == badIt.ieta && hDetId.iphi() == badIt.iphi) {
            LogDebug("PFCandidateRecalibrator")
                << "==> orig en (tot,H,E): " << pf.energy() << " " << pf.rawHcalEnergy() << " " << pf.rawEcalEnergy();
            if (badIt.depth == 1)  //depth1
            {
              longE *= badIt.ratio;
              ecalEnergy = ((longE - shortE) > 0.) ? (longE - shortE) : 0.;
              totEnergy = ecalEnergy + hcalEnergy;
            } else  //depth2
            {
              shortE *= badIt.ratio;
              hcalEnergy = 2 * shortE;
              ecalEnergy = ((longE - shortE) > 0.) ? (longE - shortE) : 0.;
              totEnergy = ecalEnergy + hcalEnergy;
            }
            //kill candidate if goes below thr
            if ((pf.particleId() == reco::PFCandidate::ParticleType::h_HF && shortE < shortFibreThr_) ||
                (pf.particleId() == reco::PFCandidate::ParticleType::egamma_HF && longE < longFibreThr_))
              toKill = true;

            LogDebug("PFCandidateRecalibrator") << "====> ieta,iphi,depth: " << badIt.ieta << " " << badIt.iphi << " "
                                                << badIt.depth << " corr: " << badIt.ratio;
            LogDebug("PFCandidateRecalibrator")
                << "====> recal en (tot,H,E): " << totEnergy << " " << hcalEnergy << " " << ecalEnergy;
          }
        }

        if (toKill) {
          discarded->push_back(pf);
          oldToNew[i] = (-discarded->size());
          badToOld.push_back(i);

          LogDebug("PFCandidateRecalibrator") << "==> KILLED ";
        } else {
          copy->push_back(pf);
          oldToNew[i] = (copy->size());
          newToOld.push_back(i);

          copy->back().setHcalEnergy(hcalEnergy, hcalEnergy);
          copy->back().setEcalEnergy(ecalEnergy, ecalEnergy);

          float scalingFactor = totEnergy / totEnergyOrig;
          math::XYZTLorentzVector recalibP4 = pf.p4() * scalingFactor;
          copy->back().setP4(recalibP4);

          LogDebug("PFCandidateRecalibrator") << "====> stored en (tot,H,E): " << copy->back().energy() << " "
                                              << copy->back().hcalEnergy() << " " << copy->back().ecalEnergy();
        }
      } else {
        copy->push_back(pf);
        oldToNew[i] = (copy->size());
        newToOld.push_back(i);
      }
    } else {
      copy->push_back(pf);
      oldToNew[i] = (copy->size());
      newToOld.push_back(i);
    }
  }

  // Now we put things in the event
  edm::OrphanHandle<reco::PFCandidateCollection> newpf = iEvent.put(std::move(copy));
  edm::OrphanHandle<reco::PFCandidateCollection> badpf = iEvent.put(std::move(discarded), "discarded");

  std::unique_ptr<edm::ValueMap<reco::PFCandidateRef>> pf2pf(new edm::ValueMap<reco::PFCandidateRef>());
  edm::ValueMap<reco::PFCandidateRef>::Filler filler(*pf2pf);
  std::vector<reco::PFCandidateRef> refs;
  refs.reserve(nPfCand);

  // old to new
  for (auto iOldToNew : oldToNew) {
    if (iOldToNew > 0) {
      refs.push_back(reco::PFCandidateRef(newpf, iOldToNew - 1));
    } else {
      refs.push_back(reco::PFCandidateRef(badpf, -iOldToNew - 1));
    }
  }
  filler.insert(pfcandidates, refs.begin(), refs.end());
  // new good to old
  refs.clear();
  for (int i : newToOld) {
    refs.push_back(reco::PFCandidateRef(pfcandidates, i));
  }
  filler.insert(newpf, refs.begin(), refs.end());
  // new bad to old
  refs.clear();
  for (int i : badToOld) {
    refs.push_back(reco::PFCandidateRef(pfcandidates, i));
  }
  filler.insert(badpf, refs.begin(), refs.end());
  // done
  filler.fill();
  iEvent.put(std::move(pf2pf));
}

void PFCandidateRecalibrator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("pfcandidates", edm::InputTag("particleFlow"));
  desc.add<double>("shortFibreThr", 1.4);
  desc.add<double>("longFibreThr", 1.4);

  descriptions.add("pfCandidateRecalibrator", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFCandidateRecalibrator);
