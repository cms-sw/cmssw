// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/Common/interface/Association.h"
//Main File
#include "CommonTools/PileupAlgos/plugins/PuppiProducer.h"
#include "CommonTools/PileupAlgos/interface/PuppiCandidate.h"

// ------------------------------------------------------------------------------------------
PuppiProducer::PuppiProducer(const edm::ParameterSet& iConfig) {
  fPuppiDiagnostics = iConfig.getParameter<bool>("puppiDiagnostics");
  fPuppiForLeptons = iConfig.getParameter<bool>("puppiForLeptons");
  fUseFromPVLooseTight = iConfig.getParameter<bool>("UseFromPVLooseTight");
  fUseDZ = iConfig.getParameter<bool>("UseDeltaZCut");
  fDZCut = iConfig.getParameter<double>("DeltaZCut");
  fPtMaxCharged = iConfig.getParameter<double>("PtMaxCharged");
  fEtaMaxCharged = iConfig.getParameter<double>("EtaMaxCharged");
  fPtMaxPhotons = iConfig.getParameter<double>("PtMaxPhotons");
  fEtaMaxPhotons = iConfig.getParameter<double>("EtaMaxPhotons");
  fUseExistingWeights = iConfig.getParameter<bool>("useExistingWeights");
  fUseWeightsNoLep = iConfig.getParameter<bool>("useWeightsNoLep");
  fClonePackedCands = iConfig.getParameter<bool>("clonePackedCands");
  fVtxNdofCut = iConfig.getParameter<int>("vtxNdofCut");
  fVtxZCut = iConfig.getParameter<double>("vtxZCut");
  fPuppiContainer = std::unique_ptr<PuppiContainer>(new PuppiContainer(iConfig));

  tokenPFCandidates_ = consumes<CandidateView>(iConfig.getParameter<edm::InputTag>("candName"));
  tokenVertices_ = consumes<VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexName"));

  produces<edm::ValueMap<float>>();
  produces<edm::ValueMap<LorentzVector>>();
  produces<edm::ValueMap<reco::CandidatePtr>>();

  if (fUseExistingWeights || fClonePackedCands)
    produces<pat::PackedCandidateCollection>();
  else
    produces<reco::PFCandidateCollection>();

  if (fPuppiDiagnostics) {
    produces<double>("PuppiNAlgos");
    produces<std::vector<double>>("PuppiRawAlphas");
    produces<std::vector<double>>("PuppiAlphas");
    produces<std::vector<double>>("PuppiAlphasMed");
    produces<std::vector<double>>("PuppiAlphasRms");
  }
}
// ------------------------------------------------------------------------------------------
PuppiProducer::~PuppiProducer() {}
// ------------------------------------------------------------------------------------------
void PuppiProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Get PFCandidate Collection
  edm::Handle<CandidateView> hPFProduct;
  iEvent.getByToken(tokenPFCandidates_, hPFProduct);
  const CandidateView* pfCol = hPFProduct.product();

  // Get vertex collection w/PV as the first entry?
  edm::Handle<reco::VertexCollection> hVertexProduct;
  iEvent.getByToken(tokenVertices_, hVertexProduct);
  const reco::VertexCollection* pvCol = hVertexProduct.product();

  int npv = 0;
  const reco::VertexCollection::const_iterator vtxEnd = pvCol->end();
  for (reco::VertexCollection::const_iterator vtxIter = pvCol->begin(); vtxEnd != vtxIter; ++vtxIter) {
    if (!vtxIter->isFake() && vtxIter->ndof() >= fVtxNdofCut && std::abs(vtxIter->z()) <= fVtxZCut)
      npv++;
  }

  std::vector<double> lWeights;
  if (!fUseExistingWeights) {
    //Fill the reco objects
    fRecoObjCollection.clear();
    fRecoObjCollection.reserve(pfCol->size());
    for (auto const& aPF : *pfCol) {
      RecoObj pReco;
      pReco.pt = aPF.pt();
      pReco.eta = aPF.eta();
      pReco.phi = aPF.phi();
      pReco.m = aPF.mass();
      pReco.rapidity = aPF.rapidity();
      pReco.charge = aPF.charge();
      pReco.pdgId = aPF.pdgId();
      const reco::Vertex* closestVtx = nullptr;
      double pDZ = -9999;
      double pD0 = -9999;
      int pVtxId = -9999;
      bool lFirst = true;
      const pat::PackedCandidate* lPack = dynamic_cast<const pat::PackedCandidate*>(&aPF);
      if (lPack == nullptr) {
        const reco::PFCandidate* pPF = dynamic_cast<const reco::PFCandidate*>(&aPF);
        double curdz = 9999;
        int closestVtxForUnassociateds = -9999;
        const reco::TrackRef aTrackRef = pPF->trackRef();
        for (auto const& aV : *pvCol) {
          if (lFirst) {
            if (aTrackRef.isNonnull()) {
              pDZ = aTrackRef->dz(aV.position());
              pD0 = aTrackRef->d0();
            } else if (pPF->gsfTrackRef().isNonnull()) {
              pDZ = pPF->gsfTrackRef()->dz(aV.position());
              pD0 = pPF->gsfTrackRef()->d0();
            }
            lFirst = false;
            if (pDZ > -9999)
              pVtxId = 0;
          }
          if (aTrackRef.isNonnull() && aV.trackWeight(pPF->trackRef()) > 0) {
            closestVtx = &aV;
            break;
          }
          // in case it's unassocciated, keep more info
          double tmpdz = 99999;
          if (aTrackRef.isNonnull())
            tmpdz = aTrackRef->dz(aV.position());
          else if (pPF->gsfTrackRef().isNonnull())
            tmpdz = pPF->gsfTrackRef()->dz(aV.position());
          if (std::abs(tmpdz) < curdz) {
            curdz = std::abs(tmpdz);
            closestVtxForUnassociateds = pVtxId;
          }
          pVtxId++;
        }
        int tmpFromPV = 0;
        // mocking the miniAOD definitions
        if (std::abs(pReco.charge) > 0) {
          if (closestVtx != nullptr && pVtxId > 0)
            tmpFromPV = 0;
          if (closestVtx != nullptr && pVtxId == 0)
            tmpFromPV = 3;
          if (closestVtx == nullptr && closestVtxForUnassociateds == 0)
            tmpFromPV = 2;
          if (closestVtx == nullptr && closestVtxForUnassociateds != 0)
            tmpFromPV = 1;
        }
        pReco.dZ = pDZ;
        pReco.d0 = pD0;
        pReco.id = 0;
        if (std::abs(pReco.charge) == 0) {
          pReco.id = 0;
        } else {
          if (tmpFromPV == 0) {
            pReco.id = 2;
          }  // 0 is associated to PU vertex
          else if (tmpFromPV == 3) {
            pReco.id = 1;
          } else if (tmpFromPV == 1 || tmpFromPV == 2) {
            pReco.id = 0;
            if ((fPtMaxCharged > 0) and (pReco.pt > fPtMaxCharged))
              pReco.id = 1;
            else if (std::abs(pReco.eta) > fEtaMaxCharged)
              pReco.id = 1;
            else if (fUseDZ)
              pReco.id = (std::abs(pDZ) < fDZCut) ? 1 : 2;
            else if (fUseFromPVLooseTight && tmpFromPV == 1)
              pReco.id = 2;
            else if (fUseFromPVLooseTight && tmpFromPV == 2)
              pReco.id = 1;
          }
        }
      } else if (lPack->vertexRef().isNonnull()) {
        pDZ = lPack->dz();
        pD0 = lPack->dxy();
        pReco.dZ = pDZ;
        pReco.d0 = pD0;

        pReco.id = 0;
        if (std::abs(pReco.charge) == 0) {
          pReco.id = 0;
        }
        if (std::abs(pReco.charge) > 0) {
          if (lPack->fromPV() == 0) {
            pReco.id = 2;
          }  // 0 is associated to PU vertex
          else if (lPack->fromPV() == (pat::PackedCandidate::PVUsedInFit)) {
            pReco.id = 1;
          } else if (lPack->fromPV() == (pat::PackedCandidate::PVTight) ||
                     lPack->fromPV() == (pat::PackedCandidate::PVLoose)) {
            pReco.id = 0;
            if ((fPtMaxCharged > 0) and (pReco.pt > fPtMaxCharged))
              pReco.id = 1;
            else if (std::abs(pReco.eta) > fEtaMaxCharged)
              pReco.id = 1;
            else if (fUseDZ)
              pReco.id = (std::abs(pDZ) < fDZCut) ? 1 : 2;
            else if (fUseFromPVLooseTight && lPack->fromPV() == (pat::PackedCandidate::PVLoose))
              pReco.id = 2;
            else if (fUseFromPVLooseTight && lPack->fromPV() == (pat::PackedCandidate::PVTight))
              pReco.id = 1;
          }
        }
      }

      fRecoObjCollection.push_back(pReco);
    }

    fPuppiContainer->initialize(fRecoObjCollection);
    fPuppiContainer->setNPV(npv);

    //Compute the weights and get the particles
    lWeights = fPuppiContainer->puppiWeights();
  } else {
    //Use the existing weights
    int lPackCtr = 0;
    for (auto const& aPF : *pfCol) {
      const pat::PackedCandidate* lPack = dynamic_cast<const pat::PackedCandidate*>(&aPF);
      float curpupweight = -1.;
      if (lPack == nullptr) {
        // throw error
        throw edm::Exception(edm::errors::LogicError,
                             "PuppiProducer: cannot get weights since inputs are not PackedCandidates");
      } else {
        if (fUseWeightsNoLep) {
          curpupweight = lPack->puppiWeightNoLep();
        } else {
          curpupweight = lPack->puppiWeight();
        }
      }
      // Protect high pT photons (important for gamma to hadronic recoil balance)
      if ((fPtMaxPhotons > 0) && (lPack->pdgId() == 22) && (std::abs(lPack->eta()) < fEtaMaxPhotons) &&
          (lPack->pt() > fPtMaxPhotons))
        curpupweight = 1;
      lWeights.push_back(curpupweight);
      lPackCtr++;
    }
  }

  //Fill it into the event
  std::unique_ptr<edm::ValueMap<float>> lPupOut(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler lPupFiller(*lPupOut);
  lPupFiller.insert(hPFProduct, lWeights.begin(), lWeights.end());
  lPupFiller.fill();

  // This is a dummy to access the "translate" method which is a
  // non-static member function even though it doesn't need to be.
  // Will fix in the future.
  static const reco::PFCandidate dummySinceTranslateIsNotStatic;

  // Fill a new PF/Packed Candidate Collection and write out the ValueMap of the new p4s.
  // Since the size of the ValueMap must be equal to the input collection, we need
  // to search the "puppi" particles to find a match for each input. If none is found,
  // the input is set to have a four-vector of 0,0,0,0
  fPuppiCandidates.reset(new PFOutputCollection);
  fPackedPuppiCandidates.reset(new PackedOutputCollection);
  std::unique_ptr<edm::ValueMap<LorentzVector>> p4PupOut(new edm::ValueMap<LorentzVector>());
  LorentzVectorCollection puppiP4s;
  std::vector<reco::CandidatePtr> values(hPFProduct->size());

  int iCand = -1;
  for (auto const& aCand : *hPFProduct) {
    ++iCand;
    std::unique_ptr<pat::PackedCandidate> pCand;
    std::unique_ptr<reco::PFCandidate> pfCand;
    if (fUseExistingWeights || fClonePackedCands) {
      const pat::PackedCandidate* cand = dynamic_cast<const pat::PackedCandidate*>(&aCand);
      if (!cand)
        throw edm::Exception(edm::errors::LogicError, "PuppiProducer: inputs are not PackedCandidates");
      pCand.reset(new pat::PackedCandidate(*cand));
    } else {
      auto id = dummySinceTranslateIsNotStatic.translatePdgIdToType(aCand.pdgId());
      const reco::PFCandidate* cand = dynamic_cast<const reco::PFCandidate*>(&aCand);
      pfCand.reset(new reco::PFCandidate(cand ? *cand : reco::PFCandidate(aCand.charge(), aCand.p4(), id)));
    }

    if (fClonePackedCands && (!fUseExistingWeights)) {
      if (fPuppiForLeptons)
        pCand->setPuppiWeight(pCand->puppiWeight(), lWeights[iCand]);
      else
        pCand->setPuppiWeight(lWeights[iCand], pCand->puppiWeightNoLep());
    }

    puppiP4s.emplace_back(lWeights[iCand] * aCand.px(),
                          lWeights[iCand] * aCand.py(),
                          lWeights[iCand] * aCand.pz(),
                          lWeights[iCand] * aCand.energy());

    if (fUseExistingWeights || fClonePackedCands) {
      pCand->setP4(puppiP4s.back());
      pCand->setSourceCandidatePtr(aCand.sourceCandidatePtr(0));
      fPackedPuppiCandidates->push_back(*pCand);
    } else {
      pfCand->setP4(puppiP4s.back());
      pfCand->setSourceCandidatePtr(aCand.sourceCandidatePtr(0));
      fPuppiCandidates->push_back(*pfCand);
    }
  }

  //Compute the modified p4s
  edm::ValueMap<LorentzVector>::Filler p4PupFiller(*p4PupOut);
  p4PupFiller.insert(hPFProduct, puppiP4s.begin(), puppiP4s.end());
  p4PupFiller.fill();

  iEvent.put(std::move(lPupOut));
  iEvent.put(std::move(p4PupOut));
  if (fUseExistingWeights || fClonePackedCands) {
    edm::OrphanHandle<pat::PackedCandidateCollection> oh = iEvent.put(std::move(fPackedPuppiCandidates));
    for (unsigned int ic = 0, nc = oh->size(); ic < nc; ++ic) {
      reco::CandidatePtr pkref(oh, ic);
      values[ic] = pkref;
    }
  } else {
    edm::OrphanHandle<reco::PFCandidateCollection> oh = iEvent.put(std::move(fPuppiCandidates));
    for (unsigned int ic = 0, nc = oh->size(); ic < nc; ++ic) {
      reco::CandidatePtr pkref(oh, ic);
      values[ic] = pkref;
    }
  }
  std::unique_ptr<edm::ValueMap<reco::CandidatePtr>> pfMap_p(new edm::ValueMap<reco::CandidatePtr>());
  edm::ValueMap<reco::CandidatePtr>::Filler filler(*pfMap_p);
  filler.insert(hPFProduct, values.begin(), values.end());
  filler.fill();
  iEvent.put(std::move(pfMap_p));

  //////////////////////////////////////////////
  if (fPuppiDiagnostics && !fUseExistingWeights) {
    // all the different alphas per particle
    // THE alpha per particle
    std::unique_ptr<std::vector<double>> theAlphas(new std::vector<double>(fPuppiContainer->puppiAlphas()));
    std::unique_ptr<std::vector<double>> theAlphasMed(new std::vector<double>(fPuppiContainer->puppiAlphasMed()));
    std::unique_ptr<std::vector<double>> theAlphasRms(new std::vector<double>(fPuppiContainer->puppiAlphasRMS()));
    std::unique_ptr<std::vector<double>> alphas(new std::vector<double>(fPuppiContainer->puppiRawAlphas()));
    std::unique_ptr<double> nalgos(new double(fPuppiContainer->puppiNAlgos()));

    iEvent.put(std::move(alphas), "PuppiRawAlphas");
    iEvent.put(std::move(nalgos), "PuppiNAlgos");
    iEvent.put(std::move(theAlphas), "PuppiAlphas");
    iEvent.put(std::move(theAlphasMed), "PuppiAlphasMed");
    iEvent.put(std::move(theAlphasRms), "PuppiAlphasRms");
  }
}

// ------------------------------------------------------------------------------------------
void PuppiProducer::beginJob() {}
// ------------------------------------------------------------------------------------------
void PuppiProducer::endJob() {}
// ------------------------------------------------------------------------------------------
void PuppiProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("puppiDiagnostics", false);
  desc.add<bool>("puppiForLeptons", false);
  desc.add<bool>("UseFromPVLooseTight", false);
  desc.add<bool>("UseDeltaZCut", true);
  desc.add<double>("DeltaZCut", 0.3);
  desc.add<double>("PtMaxCharged", -1.);
  desc.add<double>("EtaMaxCharged", 99999.);
  desc.add<double>("PtMaxPhotons", -1.);
  desc.add<double>("EtaMaxPhotons", 2.5);
  desc.add<double>("PtMaxNeutrals", 200.);
  desc.add<double>("PtMaxNeutralsStartSlope", 0.);
  desc.add<bool>("useExistingWeights", false);
  desc.add<bool>("useWeightsNoLep", false);
  desc.add<bool>("clonePackedCands", false);
  desc.add<int>("vtxNdofCut", 4);
  desc.add<double>("vtxZCut", 24);
  desc.add<edm::InputTag>("candName", edm::InputTag("particleFlow"));
  desc.add<edm::InputTag>("vertexName", edm::InputTag("offlinePrimaryVertices"));
  desc.add<bool>("applyCHS", true);
  desc.add<bool>("invertPuppi", false);
  desc.add<bool>("useExp", false);
  desc.add<double>("MinPuppiWeight", .01);

  PuppiAlgo::fillDescriptionsPuppiAlgo(desc);

  descriptions.add("PuppiProducer", desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(PuppiProducer);
