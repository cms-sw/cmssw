#include "CommonTools/PileupAlgos/interface/PuppiAlgo.h"
#include "CommonTools/PileupAlgos/interface/PuppiContainer.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/PtEtaPhiMass.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>

// ------------------------------------------------------------------------------------------
class PuppiProducer : public edm::stream::EDProducer<> {
public:
  explicit PuppiProducer(const edm::ParameterSet&);
  ~PuppiProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  typedef math::XYZTLorentzVector LorentzVector;
  typedef std::vector<LorentzVector> LorentzVectorCollection;
  typedef reco::VertexCollection VertexCollection;
  typedef edm::View<reco::Candidate> CandidateView;
  typedef std::vector<reco::PFCandidate> PFInputCollection;
  typedef std::vector<reco::PFCandidate> PFOutputCollection;
  typedef std::vector<pat::PackedCandidate> PackedOutputCollection;
  typedef edm::View<reco::PFCandidate> PFView;
  typedef edm::Association<reco::VertexCollection> CandToVertex;

private:
  virtual void beginJob();
  void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob();

  edm::EDGetTokenT<CandidateView> tokenPFCandidates_;
  edm::EDGetTokenT<VertexCollection> tokenVertices_;
  edm::EDGetTokenT<CandToVertex> tokenVertexAssociation_;
  edm::EDGetTokenT<edm::ValueMap<int>> tokenVertexAssociationQuality_;
  edm::EDGetTokenT<PuppiContainer> tokenPuppiContainer_;
  edm::EDGetTokenT<PFOutputCollection> tokenPuppiCandidates_;
  edm::EDGetTokenT<PackedOutputCollection> tokenPackedPuppiCandidates_;
  edm::EDGetTokenT<double> puProxyValueToken_;
  edm::EDPutTokenT<edm::ValueMap<float>> ptokenPupOut_;
  edm::EDPutTokenT<edm::ValueMap<LorentzVector>> ptokenP4PupOut_;
  edm::EDPutTokenT<edm::ValueMap<reco::CandidatePtr>> ptokenValues_;
  edm::EDPutTokenT<pat::PackedCandidateCollection> ptokenPackedPuppiCandidates_;
  edm::EDPutTokenT<reco::PFCandidateCollection> ptokenPuppiCandidates_;
  edm::EDPutTokenT<double> ptokenNalgos_;
  edm::EDPutTokenT<std::vector<double>> ptokenRawAlphas_;
  edm::EDPutTokenT<std::vector<double>> ptokenAlphas_;
  edm::EDPutTokenT<std::vector<double>> ptokenAlphasMed_;
  edm::EDPutTokenT<std::vector<double>> ptokenAlphasRms_;
  std::string fPuppiName;
  std::string fPFName;
  std::string fPVName;
  bool fUseVertexAssociation;
  int vertexAssociationQuality_;
  bool fPuppiDiagnostics;
  bool fPuppiNoLep;
  bool fUseFromPVLooseTight;
  bool fUseDZ;
  bool fUseDZforPileup;
  double fDZCut;
  double fEtaMinUseDZ;
  double fPtMaxCharged;
  double fEtaMaxCharged;
  double fPtMaxPhotons;
  double fEtaMaxPhotons;
  uint fNumOfPUVtxsForCharged;
  double fDZCutForChargedFromPUVtxs;
  bool fUseExistingWeights;
  bool fApplyPhotonProtectionForExistingWeights;
  bool fClonePackedCands;
  bool fuseBugFix;
  int fVtxNdofCut;
  double fVtxZCut;
  bool fUsePUProxyValue;
  std::unique_ptr<PuppiContainer> fPuppiContainer;
  std::vector<RecoObj> fRecoObjCollection;
};

// ------------------------------------------------------------------------------------------
PuppiProducer::PuppiProducer(const edm::ParameterSet& iConfig) {
  fPuppiDiagnostics = iConfig.getParameter<bool>("puppiDiagnostics");
  fPuppiNoLep = iConfig.getParameter<bool>("puppiNoLep");
  fUseFromPVLooseTight = iConfig.getParameter<bool>("UseFromPVLooseTight");
  fUseDZ = iConfig.getParameter<bool>("UseDeltaZCut");
  fUseDZforPileup = iConfig.getParameter<bool>("UseDeltaZCutForPileup");
  fDZCut = iConfig.getParameter<double>("DeltaZCut");
  fEtaMinUseDZ = iConfig.getParameter<double>("EtaMinUseDeltaZ");
  fPtMaxCharged = iConfig.getParameter<double>("PtMaxCharged");
  fEtaMaxCharged = iConfig.getParameter<double>("EtaMaxCharged");
  fPtMaxPhotons = iConfig.getParameter<double>("PtMaxPhotons");
  fEtaMaxPhotons = iConfig.getParameter<double>("EtaMaxPhotons");
  fNumOfPUVtxsForCharged = iConfig.getParameter<uint>("NumOfPUVtxsForCharged");
  fDZCutForChargedFromPUVtxs = iConfig.getParameter<double>("DeltaZCutForChargedFromPUVtxs");
  fUseExistingWeights = iConfig.getParameter<bool>("useExistingWeights");
  fApplyPhotonProtectionForExistingWeights = iConfig.getParameter<bool>("applyPhotonProtectionForExistingWeights");
  fuseBugFix = iConfig.getParameter<bool>("useBugFix");
  fClonePackedCands = iConfig.getParameter<bool>("clonePackedCands");
  fVtxNdofCut = iConfig.getParameter<int>("vtxNdofCut");
  fVtxZCut = iConfig.getParameter<double>("vtxZCut");
  fPuppiContainer = std::make_unique<PuppiContainer>(iConfig);

  tokenPFCandidates_ = consumes<CandidateView>(iConfig.getParameter<edm::InputTag>("candName"));
  tokenVertices_ = consumes<VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexName"));
  fUseVertexAssociation = iConfig.getParameter<bool>("useVertexAssociation");
  vertexAssociationQuality_ = iConfig.getParameter<int>("vertexAssociationQuality");
  if (fUseVertexAssociation) {
    tokenVertexAssociation_ = consumes<CandToVertex>(iConfig.getParameter<edm::InputTag>("vertexAssociation"));
    tokenVertexAssociationQuality_ =
        consumes<edm::ValueMap<int>>(iConfig.getParameter<edm::InputTag>("vertexAssociation"));
  }

  fUsePUProxyValue = iConfig.getParameter<bool>("usePUProxyValue");

  if (fUsePUProxyValue) {
    puProxyValueToken_ = consumes<double>(iConfig.getParameter<edm::InputTag>("PUProxyValue"));
  }

  ptokenPupOut_ = produces<edm::ValueMap<float>>();
  ptokenP4PupOut_ = produces<edm::ValueMap<LorentzVector>>();
  ptokenValues_ = produces<edm::ValueMap<reco::CandidatePtr>>();

  if (fUseExistingWeights || fClonePackedCands)
    ptokenPackedPuppiCandidates_ = produces<pat::PackedCandidateCollection>();
  else {
    ptokenPuppiCandidates_ = produces<reco::PFCandidateCollection>();
  }

  if (fPuppiDiagnostics) {
    ptokenNalgos_ = produces<double>("PuppiNAlgos");
    ptokenRawAlphas_ = produces<std::vector<double>>("PuppiRawAlphas");
    ptokenAlphas_ = produces<std::vector<double>>("PuppiAlphas");
    ptokenAlphasMed_ = produces<std::vector<double>>("PuppiAlphasMed");
    ptokenAlphasRms_ = produces<std::vector<double>>("PuppiAlphasRms");
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

  edm::Association<reco::VertexCollection> associatedPV;
  edm::ValueMap<int> associationQuality;
  if ((fUseVertexAssociation) && (!fUseExistingWeights)) {
    associatedPV = iEvent.get(tokenVertexAssociation_);
    associationQuality = iEvent.get(tokenVertexAssociationQuality_);
  }

  double puProxyValue = 0.;
  if (fUsePUProxyValue) {
    puProxyValue = iEvent.get(puProxyValueToken_);
  } else {
    for (auto const& vtx : *pvCol) {
      if (!vtx.isFake() && vtx.ndof() >= fVtxNdofCut && std::abs(vtx.z()) <= fVtxZCut)
        ++puProxyValue;
    }
  }

  std::vector<double> lWeights;
  if (!fUseExistingWeights) {
    //Fill the reco objects
    fRecoObjCollection.clear();
    fRecoObjCollection.reserve(pfCol->size());
    int iCand = 0;
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
      uint pVtxId = 0;
      bool isLepton = ((std::abs(pReco.pdgId) == 11) || (std::abs(pReco.pdgId) == 13));
      const pat::PackedCandidate* lPack = dynamic_cast<const pat::PackedCandidate*>(&aPF);

      if (fUseVertexAssociation) {
        const reco::VertexRef& PVOrig = associatedPV[reco::CandidatePtr(hPFProduct, iCand)];
        int quality = associationQuality[reco::CandidatePtr(hPFProduct, iCand)];
        if (PVOrig.isNonnull() && (quality >= vertexAssociationQuality_)) {
          closestVtx = PVOrig.get();
          pVtxId = PVOrig.key();
        }
        if (std::abs(pReco.charge) == 0)
          pReco.id = 0;
        else if (fPuppiNoLep && isLepton)
          pReco.id = 3;
        else if (closestVtx != nullptr && pVtxId == 0)
          pReco.id = 1;  // Associated to main vertex
        else if (closestVtx != nullptr && pVtxId > 0)
          pReco.id = 2;  // Associated to PU
        else
          pReco.id = 0;  // Unassociated
      } else if (lPack == nullptr) {
        const reco::PFCandidate* pPF = dynamic_cast<const reco::PFCandidate*>(&aPF);
        double curdz = 9999;
        int closestVtxForUnassociateds = -9999;
        const reco::TrackRef aTrackRef = pPF->trackRef();
        bool lFirst = true;
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
          if (fPuppiNoLep && isLepton)
            pReco.id = 3;
          else if (tmpFromPV == 0) {
            pReco.id = 2;
            if (fNumOfPUVtxsForCharged > 0 and (pVtxId <= fNumOfPUVtxsForCharged) and
                (std::abs(pDZ) < fDZCutForChargedFromPUVtxs))
              pReco.id = 1;
          } else if (tmpFromPV == 3)
            pReco.id = 1;
          else if (tmpFromPV == 1 || tmpFromPV == 2) {
            pReco.id = 0;
            if ((fPtMaxCharged > 0) and (pReco.pt > fPtMaxCharged))
              pReco.id = 1;
            else if (std::abs(pReco.eta) > fEtaMaxCharged)
              pReco.id = 1;
            else if ((fUseDZ) && (std::abs(pReco.eta) >= fEtaMinUseDZ) && (std::abs(pDZ) < fDZCut))
              pReco.id = 1;
            else if ((fUseDZforPileup) && (std::abs(pReco.eta) >= fEtaMinUseDZ) && (std::abs(pDZ) >= fDZCut))
              pReco.id = 2;
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
          if (fPuppiNoLep && isLepton) {
            pReco.id = 3;
          } else if (lPack->fromPV() == 0) {
            pReco.id = 2;
            if ((fNumOfPUVtxsForCharged > 0) and (std::abs(pDZ) < fDZCutForChargedFromPUVtxs)) {
              for (size_t puVtx_idx = 1; puVtx_idx <= fNumOfPUVtxsForCharged && puVtx_idx < pvCol->size();
                   ++puVtx_idx) {
                if (lPack->fromPV(puVtx_idx) >= 2) {
                  pReco.id = 1;
                  break;
                }
              }
            }
          } else if (lPack->fromPV() == (pat::PackedCandidate::PVUsedInFit)) {
            pReco.id = 1;
          } else if (lPack->fromPV() == (pat::PackedCandidate::PVTight) ||
                     lPack->fromPV() == (pat::PackedCandidate::PVLoose)) {
            pReco.id = 0;
            if ((fPtMaxCharged > 0) and (pReco.pt > fPtMaxCharged))
              pReco.id = 1;
            else if (std::abs(pReco.eta) > fEtaMaxCharged)
              pReco.id = 1;
            else if ((fUseDZ) && (std::abs(pReco.eta) >= fEtaMinUseDZ) && (std::abs(pDZ) < fDZCut))
              pReco.id = 1;
            else if ((fUseDZforPileup) && (std::abs(pReco.eta) >= fEtaMinUseDZ) && (std::abs(pDZ) >= fDZCut))
              pReco.id = 2;
            else if (fUseFromPVLooseTight && lPack->fromPV() == (pat::PackedCandidate::PVLoose))
              pReco.id = 2;
            else if (fUseFromPVLooseTight && lPack->fromPV() == (pat::PackedCandidate::PVTight))
              pReco.id = 1;
          }
        }
      }

      fRecoObjCollection.push_back(pReco);
      iCand++;
    }

    fPuppiContainer->initialize(fRecoObjCollection);
    fPuppiContainer->setPUProxy(puProxyValue);

    //Compute the weights and get the particles
    lWeights = fPuppiContainer->puppiWeights();
  } else {
    //Use the existing weights
    int lPackCtr = 0;
    lWeights.reserve(pfCol->size());
    for (auto const& aPF : *pfCol) {
      const pat::PackedCandidate* lPack = dynamic_cast<const pat::PackedCandidate*>(&aPF);
      float curpupweight = -1.;
      if (lPack == nullptr) {
        // throw error
        throw edm::Exception(edm::errors::LogicError,
                             "PuppiProducer: cannot get weights since inputs are not PackedCandidates");
      } else {
        if (fPuppiNoLep) {
          curpupweight = lPack->puppiWeightNoLep();
        } else {
          curpupweight = lPack->puppiWeight();
        }
      }
      // Protect high pT photons (important for gamma to hadronic recoil balance)
      if ((fPtMaxPhotons > 0) && (lPack->pdgId() == 22) && (std::abs(lPack->eta()) < fEtaMaxPhotons) &&
          (lPack->pt() > fPtMaxPhotons) && fuseBugFix)
        curpupweight = 1;

      // Optional: Protect high pT photons (important for gamma to hadronic recoil balance) for existing weights.
      if (fApplyPhotonProtectionForExistingWeights && (fPtMaxPhotons > 0) && (lPack->pdgId() == 22) &&
          (std::abs(lPack->eta()) < fEtaMaxPhotons) && (lPack->pt() > fPtMaxPhotons))
        curpupweight = 1;
      lWeights.push_back(curpupweight);
      lPackCtr++;
    }
  }

  //Fill it into the event
  edm::ValueMap<float> lPupOut;
  edm::ValueMap<float>::Filler lPupFiller(lPupOut);
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
  PFOutputCollection fPuppiCandidates;
  PackedOutputCollection fPackedPuppiCandidates;

  edm::ValueMap<LorentzVector> p4PupOut;
  LorentzVectorCollection puppiP4s;
  std::vector<reco::CandidatePtr> values(hPFProduct->size());

  int iCand = -1;
  puppiP4s.reserve(hPFProduct->size());
  if (fUseExistingWeights || fClonePackedCands)
    fPackedPuppiCandidates.reserve(hPFProduct->size());
  else
    fPuppiCandidates.reserve(hPFProduct->size());
  for (auto const& aCand : *hPFProduct) {
    ++iCand;
    std::unique_ptr<pat::PackedCandidate> pCand;
    std::unique_ptr<reco::PFCandidate> pfCand;

    if (fUseExistingWeights || fClonePackedCands) {
      const pat::PackedCandidate* cand = dynamic_cast<const pat::PackedCandidate*>(&aCand);
      if (!cand)
        throw edm::Exception(edm::errors::LogicError, "PuppiProducer: inputs are not PackedCandidates");
      pCand = std::make_unique<pat::PackedCandidate>(*cand);
    } else {
      auto id = dummySinceTranslateIsNotStatic.translatePdgIdToType(aCand.pdgId());
      const reco::PFCandidate* cand = dynamic_cast<const reco::PFCandidate*>(&aCand);
      pfCand = std::make_unique<reco::PFCandidate>(cand ? *cand : reco::PFCandidate(aCand.charge(), aCand.p4(), id));
    }

    // Here, we are using new weights computed and putting them in the packed candidates.
    if (fClonePackedCands && (!fUseExistingWeights)) {
      if (fPuppiNoLep)
        pCand->setPuppiWeight(pCand->puppiWeight(), lWeights[iCand]);
      else
        pCand->setPuppiWeight(lWeights[iCand], pCand->puppiWeightNoLep());
    }

    puppiP4s.emplace_back(lWeights[iCand] * aCand.px(),
                          lWeights[iCand] * aCand.py(),
                          lWeights[iCand] * aCand.pz(),
                          lWeights[iCand] * aCand.energy());

    // Here, we are using existing weights, or we're using packed candidates.
    // That is, whether or not we recomputed the weights, we store the
    // source candidate appropriately, and set the p4 of the packed candidate.
    if (fUseExistingWeights || fClonePackedCands) {
      pCand->setP4(puppiP4s.back());
      pCand->setSourceCandidatePtr(aCand.sourceCandidatePtr(0));
      fPackedPuppiCandidates.push_back(*pCand);
    } else {
      pfCand->setP4(puppiP4s.back());
      pfCand->setSourceCandidatePtr(aCand.sourceCandidatePtr(0));
      fPuppiCandidates.push_back(*pfCand);
    }
  }

  //Compute the modified p4s
  edm::ValueMap<LorentzVector>::Filler p4PupFiller(p4PupOut);
  p4PupFiller.insert(hPFProduct, puppiP4s.begin(), puppiP4s.end());
  p4PupFiller.fill();

  iEvent.emplace(ptokenPupOut_, lPupOut);
  iEvent.emplace(ptokenP4PupOut_, p4PupOut);
  if (fUseExistingWeights || fClonePackedCands) {
    edm::OrphanHandle<pat::PackedCandidateCollection> oh =
        iEvent.emplace(ptokenPackedPuppiCandidates_, fPackedPuppiCandidates);
    for (unsigned int ic = 0, nc = oh->size(); ic < nc; ++ic) {
      reco::CandidatePtr pkref(oh, ic);
      values[ic] = pkref;
    }
  } else {
    edm::OrphanHandle<reco::PFCandidateCollection> oh = iEvent.emplace(ptokenPuppiCandidates_, fPuppiCandidates);
    for (unsigned int ic = 0, nc = oh->size(); ic < nc; ++ic) {
      reco::CandidatePtr pkref(oh, ic);
      values[ic] = pkref;
    }
  }
  edm::ValueMap<reco::CandidatePtr> pfMap_p;
  edm::ValueMap<reco::CandidatePtr>::Filler filler(pfMap_p);
  filler.insert(hPFProduct, values.begin(), values.end());
  filler.fill();
  iEvent.emplace(ptokenValues_, pfMap_p);

  //////////////////////////////////////////////
  if (fPuppiDiagnostics && !fUseExistingWeights) {
    // all the different alphas per particle
    // THE alpha per particle
    std::vector<double> theAlphas(fPuppiContainer->puppiAlphas());
    std::vector<double> theAlphasMed(fPuppiContainer->puppiAlphasMed());
    std::vector<double> theAlphasRms(fPuppiContainer->puppiAlphasRMS());
    std::vector<double> alphas(fPuppiContainer->puppiRawAlphas());
    double nalgos(fPuppiContainer->puppiNAlgos());

    iEvent.emplace(ptokenRawAlphas_, alphas);
    iEvent.emplace(ptokenNalgos_, nalgos);
    iEvent.emplace(ptokenAlphas_, theAlphas);
    iEvent.emplace(ptokenAlphasMed_, theAlphasMed);
    iEvent.emplace(ptokenAlphasRms_, theAlphasRms);
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
  desc.add<bool>("puppiNoLep", false);
  desc.add<bool>("UseFromPVLooseTight", false);
  desc.add<bool>("UseDeltaZCut", true);
  desc.add<bool>("UseDeltaZCutForPileup", true);
  desc.add<double>("DeltaZCut", 0.3);
  desc.add<double>("EtaMinUseDeltaZ", 0.);
  desc.add<double>("PtMaxCharged", -1.);
  desc.add<double>("EtaMaxCharged", 99999.);
  desc.add<double>("PtMaxPhotons", -1.);
  desc.add<double>("EtaMaxPhotons", 2.5);
  desc.add<double>("PtMaxNeutrals", 200.);
  desc.add<double>("PtMaxNeutralsStartSlope", 0.);
  desc.add<uint>("NumOfPUVtxsForCharged", 0);
  desc.add<double>("DeltaZCutForChargedFromPUVtxs", 0.2);
  desc.add<bool>("useExistingWeights", false);
  desc.add<bool>("applyPhotonProtectionForExistingWeights", false);
  desc.add<bool>("useBugFix",false);
  desc.add<bool>("clonePackedCands", false);
  desc.add<int>("vtxNdofCut", 4);
  desc.add<double>("vtxZCut", 24);
  desc.add<edm::InputTag>("candName", edm::InputTag("particleFlow"));
  desc.add<edm::InputTag>("vertexName", edm::InputTag("offlinePrimaryVertices"));
  desc.add<bool>("useVertexAssociation", false);
  desc.add<int>("vertexAssociationQuality", 0);
  desc.add<edm::InputTag>("vertexAssociation", edm::InputTag(""));
  desc.add<bool>("applyCHS", true);
  desc.add<bool>("invertPuppi", false);
  desc.add<bool>("useExp", false);
  desc.add<double>("MinPuppiWeight", .01);
  desc.add<bool>("usePUProxyValue", false);
  desc.add<edm::InputTag>("PUProxyValue", edm::InputTag(""));

  PuppiAlgo::fillDescriptionsPuppiAlgo(desc);

  descriptions.add("PuppiProducer", desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(PuppiProducer);
