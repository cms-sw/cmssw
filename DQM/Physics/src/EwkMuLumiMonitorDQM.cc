#include "DQM/Physics/src/EwkMuLumiMonitorDQM.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/RecoCandidate/interface/IsoDepositVetos.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/Math/interface/LorentzVector.h"

using namespace edm;
using namespace std;
using namespace reco;
using namespace isodeposit;

EwkMuLumiMonitorDQM::EwkMuLumiMonitorDQM(const ParameterSet& cfg)
    :  // Input collections
      trigTag_(cfg.getUntrackedParameter<edm::InputTag>("TrigTag", edm::InputTag("TriggerResults::HLT"))),
      trigToken_(consumes<edm::TriggerResults>(trigTag_)),
      trigEvToken_(consumes<trigger::TriggerEvent>(cfg.getUntrackedParameter<edm::InputTag>("triggerEvent"))),
      beamSpotToken_(consumes<reco::BeamSpot>(
          cfg.getUntrackedParameter<edm::InputTag>("offlineBeamSpot", edm::InputTag("offlineBeamSpot")))),
      muonToken_(consumes<edm::View<reco::Muon> >(cfg.getUntrackedParameter<edm::InputTag>("muons"))),
      trackToken_(consumes<reco::TrackCollection>(cfg.getUntrackedParameter<edm::InputTag>("tracks"))),
      caloTowerToken_(consumes<CaloTowerCollection>(cfg.getUntrackedParameter<edm::InputTag>("calotower"))),
      metToken_(consumes<edm::View<reco::MET> >(cfg.getUntrackedParameter<edm::InputTag>("metTag"))),
      metIncludesMuons_(cfg.getUntrackedParameter<bool>("METIncludesMuons")),
      // Main cuts
      // massMin_(cfg.getUntrackedParameter<double>("MtMin", 20.)),
      //   massMax_(cfg.getUntrackedParameter<double>("MtMax", 2000.))
      //  hltPath_(cfg.getUntrackedParameter<std::string> ("hltPath")) ,
      //  L3FilterName_(cfg.getUntrackedParameter<std::string>
      // ("L3FilterName")),
      ptMuCut_(cfg.getUntrackedParameter<double>("ptMuCut")),
      etaMuCut_(cfg.getUntrackedParameter<double>("etaMuCut")),
      isRelativeIso_(cfg.getUntrackedParameter<bool>("IsRelativeIso")),
      isCombinedIso_(cfg.getUntrackedParameter<bool>("IsCombinedIso")),
      isoCut03_(cfg.getUntrackedParameter<double>("IsoCut03")),
      //  deltaRTrk_(cfg.getUntrackedParameter<double>("deltaRTrk")),
      ptThreshold_(cfg.getUntrackedParameter<double>("ptThreshold")),
      // deltaRVetoTrk_(cfg.getUntrackedParameter<double>("deltaRVetoTrk")),
      maxDPtRel_(cfg.getUntrackedParameter<double>("maxDPtRel")),
      maxDeltaR_(cfg.getUntrackedParameter<double>("maxDeltaR")),
      mtMin_(cfg.getUntrackedParameter<double>("mtMin")),
      mtMax_(cfg.getUntrackedParameter<double>("mtMax")),
      acopCut_(cfg.getUntrackedParameter<double>("acopCut")),
      dxyCut_(cfg.getUntrackedParameter<double>("DxyCut")) {
  // just to initialize
  isValidHltConfig_ = false;
}

void EwkMuLumiMonitorDQM::dqmBeginRun(const Run& r, const EventSetup& iSetup) {
  nall = 0;
  nEvWithHighPtMu = 0;
  nInKinRange = 0;
  nsel = 0;
  niso = 0;
  nhlt = 0;
  n1hlt = 0;
  n2hlt = 0;
  nNotIso = 0;
  nGlbSta = 0;
  nGlbTrk = 0;
  nTMass = 0;
  nW = 0;

  // passed as parameter to HLTConfigProvider::init(), not yet used
  bool isConfigChanged = false;

  // isValidHltConfig_ used to short-circuit analyze() in case of problems
  isValidHltConfig_ = hltConfigProvider_.init(r, iSetup, trigTag_.process(), isConfigChanged);
  // std::cout << "hlt config trigger is valid??" << isValidHltConfig_ <<
  // std::endl;
}

void EwkMuLumiMonitorDQM::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  ibooker.setCurrentFolder("Physics/EwkMuLumiMonitorDQM");

  mass2HLT_ = ibooker.book1D("Z_2HLT_MASS", "Z mass [GeV/c^{2}]", 200, 0., 200.);
  mass1HLT_ = ibooker.book1D("Z_1HLT_MASS", "Z mass [GeV/c^{2}]", 200, 0., 200.);
  massNotIso_ = ibooker.book1D("Z_NOTISO_MASS", "Z mass [GeV/c^{2}]", 200, 0., 200.);
  massGlbSta_ = ibooker.book1D("Z_GLBSTA_MASS", "Z mass [GeV/c^{2}]", 200, 0., 200.);
  massGlbTrk_ = ibooker.book1D("Z_GLBTRK_MASS", "Z mass [GeV/c^{2}]", 200, 0., 200.);
  massIsBothGlbTrkThanW_ = ibooker.book1D("Z_ISBOTHGLBTRKTHANW_MASS", "Z mass [GeV/c^{2}]", 200, 0., 200.);

  highMass2HLT_ = ibooker.book1D("Z_2HLT_HIGHMASS", "Z high mass [GeV/c^{2}]", 2000, 0., 2000.);
  highMass1HLT_ = ibooker.book1D("Z_1HLT_HIGHMASS", "Z high mass [GeV/c^{2}]", 2000, 0., 2000.);
  highMassNotIso_ = ibooker.book1D("Z_NOTISO_HIGHMASS", "Z high mass [GeV/c^{2}]", 2000, 0., 2000.);
  highMassGlbSta_ = ibooker.book1D("Z_GLBSTA_HIGHMASS", "Z high mass [GeV/c^{2}]", 2000, 0., 2000.);
  highMassGlbTrk_ = ibooker.book1D("Z_GLBTRK_HIGHMASS", "Z high mass [GeV/c^{2}]", 2000, 0., 2000.);
  highMassIsBothGlbTrkThanW_ =
      ibooker.book1D("Z_ISBOTHGLBTRKTHANW_HIGHMASS", "Z high mass [GeV/c^{2}]", 2000, 0., 2000.);

  TMass_ = ibooker.book1D("TMASS", "Transverse mass [GeV]", 300, 0., 300.);
}

double EwkMuLumiMonitorDQM::muIso(const reco::Muon& mu) {
  double isovar = mu.isolationR03().sumPt;
  if (isCombinedIso_) {
    isovar += mu.isolationR03().emEt;
    isovar += mu.isolationR03().hadEt;
  }
  if (isRelativeIso_)
    isovar /= mu.pt();
  return isovar;
}

double EwkMuLumiMonitorDQM::tkIso(const reco::Track& tk,
                                  Handle<TrackCollection> tracks,
                                  Handle<CaloTowerCollection> calotower) {
  double ptSum = 0;
  for (size_t i = 0; i < tracks->size(); ++i) {
    const reco::Track& elem = tracks->at(i);
    double elemPt = elem.pt();
    // same parameter used for muIsolation: dR [0.01, IsoCut03_], |dZ|<0.2,
    // |d_r(xy)|<0.1
    double elemVx = elem.vx();
    double elemVy = elem.vy();
    double elemD0 = sqrt(elemVx * elemVx + elemVy * elemVy);
    if (elemD0 > 0.2)
      continue;
    double dz = fabs(elem.vz() - tk.vz());
    if (dz > 0.1)
      continue;
    // evaluate only for tracks with pt>ptTreshold
    if (elemPt < ptThreshold_)
      continue;
    double dR = deltaR(elem.eta(), elem.phi(), tk.eta(), tk.phi());
    // isolation in a cone with dR=0.3, and vetoing the track itself
    if ((dR < 0.01) || (dR > 0.3))
      continue;
    ptSum += elemPt;
  }
  if (isCombinedIso_) {
    // loop on clusters....
    for (CaloTowerCollection::const_iterator it = calotower->begin(); it != calotower->end(); ++it) {
      double dR = deltaR(it->eta(), it->phi(), tk.outerEta(), tk.outerPhi());
      // veto value is 0.1 for towers....
      if ((dR < 0.1) || (dR > 0.3))
        continue;
      ptSum += it->emEnergy();
      ptSum += it->hadEnergy();
    }
  }
  if (isRelativeIso_)
    ptSum /= tk.pt();
  return ptSum;
}

bool EwkMuLumiMonitorDQM::IsMuMatchedToHLTMu(const reco::Muon& mu,
                                             const std::vector<reco::Particle>& HLTMu,
                                             double DR,
                                             double DPtRel) {
  size_t dim = HLTMu.size();
  size_t nPass = 0;
  if (dim == 0)
    return false;
  for (size_t k = 0; k < dim; k++) {
    if ((deltaR(HLTMu[k], mu) < DR) && (fabs(HLTMu[k].pt() - mu.pt()) / HLTMu[k].pt() < DPtRel)) {
      nPass++;
    }
  }
  return (nPass > 0);
}

void EwkMuLumiMonitorDQM::analyze(const Event& ev, const EventSetup&) {
  nall++;
  bool hlt_sel = false;
  double iso1 = -1;
  double iso2 = -1;
  bool isMu1Iso = false;
  bool isMu2Iso = false;
  bool singleTrigFlag1 = false;
  bool singleTrigFlag2 = false;
  isZGolden1HLT_ = false;
  isZGolden2HLT_ = false;
  isZGoldenNoIso_ = false;
  isZGlbSta_ = false;
  isZGlbTrk_ = false;
  isW_ = false;
  // Trigger
  bool trigger_fired = false;

  Handle<TriggerResults> triggerResults;
  if (!ev.getByToken(trigToken_, triggerResults)) {
    // LogWarning("") << ">>> TRIGGER collection does not exist !!!";
    return;
  }

  ev.getByToken(trigToken_, triggerResults);
  /*
    const edm::TriggerNames & trigNames = ev.triggerNames(*triggerResults);


  for (size_t i=0; i<triggerResults->size(); i++) {
  std::string trigName = trigNames.triggerName(i);
  //std::cout << " trigName == " << trigName << std::endl;
    if ( trigName == hltPath_ && triggerResults->accept(i)) {
    trigger_fired = true;
    hlt_sel=true;
    nhlt++;
    }
    }
  */
  // see the trigger single muon which are present
  string lowestMuonUnprescaledTrig = "";
  bool lowestMuonUnprescaledTrigFound = false;
  const std::vector<std::string>& triggerNames = hltConfigProvider_.triggerNames();
  for (size_t ts = 0; ts < triggerNames.size(); ts++) {
    string trig = triggerNames[ts];
    size_t f = trig.find("HLT_Mu");
    if ((f != std::string::npos)) {
      // std::cout << "single muon trigger present: " << trig << std::endl;
      // See if the trigger is prescaled;
      /// number of prescale sets available
      bool prescaled = false;
      const unsigned int prescaleSize = hltConfigProvider_.prescaleSize();
      for (unsigned int ps = 0; ps < prescaleSize; ps++) {
        if (hltConfigProvider_.prescaleValue<double>(ps, trig) != 1)
          prescaled = true;
      }
      if (!prescaled) {
        // looking now for the lowest hlt path not prescaled, with name of the
        // form HLT_MuX or HLTMuX_vY
        for (unsigned int n = 9; n < 100; n++) {
          string lowestTrig = "HLT_Mu";
          string lowestTrigv0 = "copy";
          std::stringstream out;
          out << n;
          std::string s = out.str();
          lowestTrig.append(s);
          lowestTrigv0 = lowestTrig;
          for (unsigned int v = 1; v < 10; v++) {
            lowestTrig.append("_v");
            std::stringstream oout;
            oout << v;
            std::string ss = oout.str();
            lowestTrig.append(ss);
            if (trig == lowestTrig)
              lowestMuonUnprescaledTrig = trig;
            if (trig == lowestTrig)
              lowestMuonUnprescaledTrigFound = true;
            if (trig == lowestTrig)
              break;
          }
          if (lowestMuonUnprescaledTrigFound)
            break;

          lowestTrig = lowestTrigv0;
          if (trig == lowestTrig)
            lowestMuonUnprescaledTrig = trig;
          //      if (trig==lowestTrig) {std::cout << " before break, lowestTrig
          // lowest single muon trigger present unprescaled: " << lowestTrig <<
          // std::endl; }
          if (trig == lowestTrig)
            lowestMuonUnprescaledTrigFound = true;
          if (trig == lowestTrig)
            break;
        }
        if (lowestMuonUnprescaledTrigFound)
          break;
      }
    }
  }
  //  std::cout << "after break, lowest single muon trigger present unprescaled:
  // " << lowestMuonUnprescaledTrig << std::endl;
  unsigned int triggerIndex;  // index of trigger path

  // See if event passed trigger paths
  std::string hltPath_ = lowestMuonUnprescaledTrig;

  triggerIndex = hltConfigProvider_.triggerIndex(hltPath_);
  if (triggerIndex < triggerResults->size())
    trigger_fired = triggerResults->accept(triggerIndex);
  std::string L3FilterName_ = "";
  if (trigger_fired) {
    const std::vector<std::string>& moduleLabs = hltConfigProvider_.moduleLabels(hltPath_);
    /*for (size_t k =0; k < moduleLabs.size()-1 ; k++){
      std::cout << "moduleLabs[" << k << "] == " << moduleLabs[k] << std::endl;
    }
    */
    // the l3 filter name is just the last module....
    size_t moduleLabsSizeMinus2 = moduleLabs.size() - 2;
    //	std::cout<<"moduleLabs[" << moduleLabsSizeMinus2 << "]== "<<
    // moduleLabs[moduleLabsSizeMinus2] << std::endl;

    L3FilterName_ = moduleLabs[moduleLabsSizeMinus2];
  }

  edm::Handle<trigger::TriggerEvent> handleTriggerEvent;
  LogTrace("") << ">>> Trigger bit: " << trigger_fired << " (" << hltPath_ << ")";
  if (!ev.getByToken(trigEvToken_, handleTriggerEvent)) {
    // LogWarning( "errorTriggerEventValid" ) << "trigger::TriggerEvent product
    // with InputTag " << trigEv_.encode() << " not in event";
    return;
  }
  ev.getByToken(trigEvToken_, handleTriggerEvent);
  const trigger::TriggerObjectCollection& toc(handleTriggerEvent->getObjects());
  std::vector<reco::Particle> HLTMuMatched;
  for (size_t ia = 0; ia < handleTriggerEvent->sizeFilters(); ++ia) {
    std::string fullname = handleTriggerEvent->filterTag(ia).encode();
    // std::cout<< "fullname::== " << fullname<< std::endl;
    std::string name;
    size_t p = fullname.find_first_of(':');
    if (p != std::string::npos) {
      name = fullname.substr(0, p);
    } else {
      name = fullname;
    }
    if (!toc.empty()) {
      const trigger::Keys& k = handleTriggerEvent->filterKeys(ia);
      for (trigger::Keys::const_iterator ki = k.begin(); ki != k.end(); ++ki) {
        // looking at all the single muon l3 trigger present, for example
        // hltSingleMu15L3Filtered15.....
        if (name == L3FilterName_) {
          // trigger_fired = true;
          hlt_sel = true;
          nhlt++;
          HLTMuMatched.push_back(toc[*ki].particle());
        }
      }
    }
  }

  // Beam spot
  Handle<reco::BeamSpot> beamSpotHandle;
  if (!ev.getByToken(beamSpotToken_, beamSpotHandle)) {
    // LogWarning("") << ">>> No beam spot found !!!";
    return;
  }

  //  looping on muon....
  Handle<View<Muon> > muons;
  if (!ev.getByToken(muonToken_, muons)) {
    // LogError("") << ">>> muon collection does not exist !!!";
    return;
  }

  ev.getByToken(muonToken_, muons);
  // saving only muons with pt> ptMuCut and eta<etaMuCut, and dxy<dxyCut
  std::vector<reco::Muon> highPtGlbMuons;
  std::vector<reco::Muon> highPtStaMuons;

  for (size_t i = 0; i < muons->size(); i++) {
    const reco::Muon& mu = muons->at(i);
    double pt = mu.pt();
    double eta = mu.eta();
    if (pt > ptMuCut_ && fabs(eta) < etaMuCut_) {
      if (mu.isGlobalMuon()) {
        // check the dxy....
        double dxy = mu.innerTrack()->dxy(beamSpotHandle->position());
        if (fabs(dxy) > dxyCut_)
          continue;
        highPtGlbMuons.push_back(mu);
      }
      if (mu.isGlobalMuon())
        continue;
      // if is not, look if it is a standalone....
      if (mu.isStandAloneMuon())
        highPtStaMuons.push_back(mu);
      nEvWithHighPtMu++;
    }
  }
  size_t nHighPtGlbMu = highPtGlbMuons.size();
  size_t nHighPtStaMu = highPtStaMuons.size();
  if (hlt_sel && (nHighPtGlbMu > 0)) {
    // loop on high pt muons if there's at least two with opposite charge build
    // a Z, more then one z candidate is foreseen.........
    // stop the loop after 10 cicles....
    (nHighPtGlbMu > 10) ? nHighPtGlbMu = 10 : 1;
    // Z selection
    if (nHighPtGlbMu > 1) {
      for (unsigned int i = 0; i < nHighPtGlbMu; i++) {
        reco::Muon muon1 = highPtGlbMuons[i];
        const math::XYZTLorentzVector& mu1(muon1.p4());
        // double pt1= muon1.pt();
        for (unsigned int j = i + 1; j < nHighPtGlbMu; ++j) {
          reco::Muon muon2 = highPtGlbMuons[j];
          const math::XYZTLorentzVector& mu2(muon2.p4());
          // double pt2= muon2.pt();
          if (muon1.charge() == muon2.charge())
            continue;
          math::XYZTLorentzVector pair = mu1 + mu2;
          double mass = pair.M();
          // checking isolation and hlt maching
          iso1 = muIso(muon1);
          iso2 = muIso(muon2);
          isMu1Iso = (iso1 < isoCut03_);
          isMu2Iso = (iso2 < isoCut03_);
          singleTrigFlag1 = IsMuMatchedToHLTMu(muon1, HLTMuMatched, maxDeltaR_, maxDPtRel_);
          singleTrigFlag2 = IsMuMatchedToHLTMu(muon2, HLTMuMatched, maxDeltaR_, maxDPtRel_);
          if (singleTrigFlag1 && singleTrigFlag2)
            isZGolden2HLT_ = true;
          if ((singleTrigFlag1 && !singleTrigFlag2) || (!singleTrigFlag1 && singleTrigFlag2))
            isZGolden1HLT_ = true;
          // Z Golden passing all criteria, with 2 or 1 muon matched to an HLT
          // muon. Using the two cathegories as a control sample for the HLT
          // matching efficiency
          if (isZGolden2HLT_ || isZGolden1HLT_) {
            // a Z golden has been found, let's remove the two muons from the
            // list, dome for avoiding resolution effect enter muons in the
            // standalone and tracker collections.........
            highPtGlbMuons.erase(highPtGlbMuons.begin() + i);
            highPtGlbMuons.erase(highPtGlbMuons.begin() + j);
            // and updating the number of high pt muons....
            nHighPtGlbMu = highPtGlbMuons.size();

            if (isMu1Iso && isMu2Iso) {
              niso++;
              if (isZGolden2HLT_) {
                n2hlt++;
                mass2HLT_->Fill(mass);
                highMass2HLT_->Fill(mass);
                /*
                  if (pt1 > pt2) {
                  highest_mupt2HLT_ -> Fill (pt1);
                  lowest_mupt2HLT_ -> Fill (pt2);
                  } else {
                  highest_mupt2HLT_ -> Fill (pt2);
                  lowest_mupt2HLT_ -> Fill (pt1);
                  }
                */
              }
              if (isZGolden1HLT_) {
                n1hlt++;
                mass1HLT_->Fill(mass);
                highMass1HLT_->Fill(mass);
                /*
                  if (pt1 >pt2) {
                  highest_mupt1HLT_ -> Fill (pt1);
                  lowest_mupt1HLT_ -> Fill (pt2);
                } else {
                  highest_mupt1HLT_ -> Fill (pt2);
                  lowest_mupt1HLT_ -> Fill (pt1);
                }
                */
              }
            } else {
              // ZGlbGlb when at least one of the two muon is not isolated and
              // at least one HLT matched, used as control sample for the
              // isolation efficiency
              // filling events with one muon not isolated and both hlt mathced
              if (((isMu1Iso && !isMu2Iso) || (!isMu1Iso && isMu2Iso)) && (isZGolden2HLT_ && isZGolden1HLT_)) {
                isZGoldenNoIso_ = true;
                nNotIso++;
                massNotIso_->Fill(mass);
                highMassNotIso_->Fill(mass);
              }
              /*
                if (pt1 > pt2) {
                highest_muptNotIso_ -> Fill (pt1);
                lowest_muptNotIso_ -> Fill (pt2);
                } else {
                highest_muptNotIso_ -> Fill (pt2);
                lowest_muptNotIso_ -> Fill (pt1);
                }
              */
            }
          }
        }
      }
    }
    // W->MuNu selection (if at least one muon has been selected)
    // looking for a W if a Z is not found.... let's think if we prefer to
    // exclude zMuMuNotIso or zMuSta....
    if (!(isZGolden2HLT_ || isZGolden1HLT_)) {
      Handle<View<MET> > metCollection;
      if (!ev.getByToken(metToken_, metCollection)) {
        // LogError("") << ">>> MET collection does not exist !!!";
        return;
      }
      const MET& met = metCollection->at(0);
      nW = 0;
      for (unsigned int i = 0; i < nHighPtGlbMu; i++) {
        reco::Muon muon1 = highPtGlbMuons[i];
        /*
               quality cut not implemented
            Quality Cuts           double dxy =
gm->dxy(beamSpotHandle->position());
            Cut in 0.2           double trackerHits =
gm->hitPattern().numberOfValidTrackerHits();
            Cut in 11           bool quality = fabs(dxy)<dxyCut_  &&
muon::isGoodMuon(mu,muon::GlobalMuonPromptTight) && trackerHits>=trackerHitsCut_
&& mu.isTrackerMuon() ;
if (!quality) continue;
        */
        // isolation cut and hlt maching
        iso1 = muIso(muon1);
        isMu1Iso = (iso1 < isoCut03_);
        if (!isMu1Iso)
          continue;
        // checking if muon is matched to any HLT muon....
        singleTrigFlag1 = IsMuMatchedToHLTMu(muon1, HLTMuMatched, maxDeltaR_, maxDPtRel_);
        if (!singleTrigFlag1)
          continue;
        // std::cout << " is GlobMu macthecd to HLT: " << singleTrigFlag1 <<
        // std::endl;
        // MT cuts
        double met_px = met.px();
        double met_py = met.py();

        if (!metIncludesMuons_) {
          for (unsigned int i = 0; i < nHighPtGlbMu; i++) {
            reco::Muon muon1 = highPtGlbMuons[i];
            met_px -= muon1.px();
            met_py -= muon1.py();
          }
        }
        double met_et = met.pt();  // sqrt(met_px*met_px+met_py*met_py);
        LogTrace("") << ">>> MET, MET_px, MET_py: " << met_et << ", " << met_px << ", " << met_py << " [GeV]";
        double w_et = met_et + muon1.pt();
        double w_px = met_px + muon1.px();
        double w_py = met_py + muon1.py();
        double massT = w_et * w_et - w_px * w_px - w_py * w_py;
        massT = (massT > 0) ? sqrt(massT) : 0;
        // Acoplanarity cuts
        Geom::Phi<double> deltaphi(muon1.phi() - atan2(met_py, met_px));
        double acop = M_PI - fabs(deltaphi.value());
        // std::cout << " is acp of W candidate less then cut: " << (acop<
        // acopCut_) << std::endl;
        if (acop > acopCut_)
          continue;  // Cut in 2.0
        TMass_->Fill(massT);
        if (massT < mtMin_ || massT > mtMax_)
          continue;  // Cut in (50,200) GeV
        // we give the number of W only in the tMass selected but we have a look
        // at mass tails to check the QCD background
        isW_ = true;
        nW++;
        nTMass++;
      }
    }
    // if a zGlobal is not selected, look at the dimuonGlobalOneStandAlone and
    // dimuonGlobalOneTrack...., used as a control sample for the tracking
    // efficiency  and muon system efficiency
    if (!(isZGolden2HLT_ || isZGolden1HLT_ || isZGoldenNoIso_)) {
      for (unsigned int i = 0; i < nHighPtGlbMu; ++i) {
        reco::Muon glbMuon = highPtGlbMuons[i];
        const math::XYZTLorentzVector& mu1(glbMuon.p4());
        // double pt1= glbMuon.pt();
        // checking that the global muon is hlt matched otherwise skip the event
        singleTrigFlag1 = IsMuMatchedToHLTMu(glbMuon, HLTMuMatched, maxDeltaR_, maxDPtRel_);
        if (!singleTrigFlag1)
          continue;
        // checking that the global muon is isolated matched otherwise skip the
        // event
        iso1 = muIso(glbMuon);
        isMu1Iso = (iso1 < isoCut03_);
        if (!isMu1Iso)
          continue;
        // look at the standalone muon ....
        // stop the loop after 10 cicles....
        (nHighPtStaMu > 10) ? nHighPtStaMu = 10 : 1;
        for (unsigned int j = 0; j < nHighPtStaMu; ++j) {
          reco::Muon staMuon = highPtStaMuons[j];
          const math::XYZTLorentzVector& mu2(staMuon.p4());
          // double pt2= staMuon.pt();
          if (glbMuon.charge() == staMuon.charge())
            continue;
          math::XYZTLorentzVector pair = mu1 + mu2;
          double mass = pair.M();
          iso2 = muIso(staMuon);
          isMu2Iso = (iso2 < isoCut03_);
          LogTrace("") << "\t... isolation value" << iso1 << ", isolated? " << isMu1Iso;
          LogTrace("") << "\t... isolation value" << iso2 << ", isolated? " << isMu2Iso;
          // requiring theat the global mu is mathed to the HLT  and both are
          // isolated
          if (isMu2Iso) {
            isZGlbSta_ = true;
            nGlbSta++;
            massGlbSta_->Fill(mass);
            highMassGlbSta_->Fill(mass);
            /*
              if (pt1 > pt2) {
              highest_muptGlbSta_ -> Fill (pt1);
              lowest_muptGlbSta_ -> Fill (pt2);
            } else {
              highest_muptGlbSta_ -> Fill (pt2);
              lowest_muptGlbSta_ -> Fill (pt1);
            }
            */
          }
        }
        // look at the tracks....
        Handle<TrackCollection> tracks;
        if (!ev.getByToken(trackToken_, tracks)) {
          // LogError("") << ">>> track collection does not exist !!!";
          return;
        }
        ev.getByToken(trackToken_, tracks);
        Handle<CaloTowerCollection> calotower;
        if (!ev.getByToken(caloTowerToken_, calotower)) {
          // LogError("") << ">>> calotower collection does not exist !!!";
          return;
        }
        ev.getByToken(caloTowerToken_, calotower);
        // avoid to loop on more than 5000 trks
        size_t nTrk = tracks->size();
        (nTrk > 5000) ? nTrk = 5000 : 1;
        for (unsigned int j = 0; j < nTrk; j++) {
          const reco::Track& tk = tracks->at(j);
          if (glbMuon.charge() == tk.charge())
            continue;
          double pt2 = tk.pt();
          double eta = tk.eta();
          double dxy = tk.dxy(beamSpotHandle->position());
          if (pt2 < ptMuCut_ || fabs(eta) > etaMuCut_ || fabs(dxy) > dxyCut_)
            continue;
          // assuming that the track is a mu....
          math::XYZTLorentzVector mu2(tk.px(), tk.py(), tk.pz(), sqrt((tk.p() * tk.p()) + (0.10566 * 0.10566)));
          math::XYZTLorentzVector pair = mu1 + mu2;
          double mass = pair.M();
          // now requiring that the tracks is isolated.......
          iso2 = tkIso(tk, tracks, calotower);
          isMu2Iso = (iso2 < isoCut03_);
          //	      std::cout << "found a track with rel/comb iso: " << iso2
          //<< std::endl;
          if (isMu2Iso) {
            isZGlbTrk_ = true;
            nGlbTrk++;
            massGlbTrk_->Fill(mass);
            highMassGlbTrk_->Fill(mass);
            if (!isW_)
              continue;
            massIsBothGlbTrkThanW_->Fill(mass);
            highMassIsBothGlbTrkThanW_->Fill(mass);
            /*
              if (pt1 > pt2) {
              highest_muptGlbTrk_ -> Fill (pt1);
              lowest_muptGlbTrk_ -> Fill (pt2);
              } else {
              highest_muptGlbTrk_ -> Fill (pt2);
              lowest_muptGlbTrk_ -> Fill (pt1);
            }
            */
          }
        }
      }
    }
  }
  if ((hlt_sel || isZGolden2HLT_ || isZGolden1HLT_ || isZGoldenNoIso_ || isZGlbSta_ || isZGlbTrk_ || isW_)) {
    nsel++;
    LogTrace("") << ">>>> Event ACCEPTED";
  } else {
    LogTrace("") << ">>>> Event REJECTED";
  }
  return;
}

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
