
/** \class PFRecoTauDiscriminationByMVAIsolation2
 *
 * MVA based discriminator against jet -> tau fakes
 * 
 * \author Christian Veelken, LLR
 *
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameterAssociation.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "CondFormats/GBRForest/interface/GBRForest.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <TMath.h>
#include <TFile.h>

#include <iostream>

using namespace reco;

namespace {
  const GBRForest* loadMVAfromFile(const edm::FileInPath& inputFileName,
                                   const std::string& mvaName,
                                   std::vector<TFile*>& inputFilesToDelete) {
    if (inputFileName.location() == edm::FileInPath::Unknown)
      throw cms::Exception("PFRecoTauDiscriminationByIsolationMVA2::loadMVA")
          << " Failed to find File = " << inputFileName << " !!\n";
    TFile* inputFile = new TFile(inputFileName.fullPath().data());

    //const GBRForest* mva = dynamic_cast<GBRForest*>(inputFile->Get(mvaName.data())); // CV: dynamic_cast<GBRForest*> fails for some reason ?!
    const GBRForest* mva = (GBRForest*)inputFile->Get(mvaName.data());
    if (!mva)
      throw cms::Exception("PFRecoTauDiscriminationByIsolationMVA2::loadMVA")
          << " Failed to load MVA = " << mvaName.data() << " from file = " << inputFileName.fullPath().data()
          << " !!\n";

    inputFilesToDelete.push_back(inputFile);

    return mva;
  }

  const GBRForest* loadMVAfromDB(const edm::EventSetup& es, const std::string& mvaName) {
    edm::ESHandle<GBRForest> mva;
    es.get<GBRWrapperRcd>().get(mvaName, mva);
    return mva.product();
  }
}  // namespace

class PFRecoTauDiscriminationByIsolationMVA2 : public PFTauDiscriminationContainerProducerBase {
public:
  explicit PFRecoTauDiscriminationByIsolationMVA2(const edm::ParameterSet& cfg)
      : PFTauDiscriminationContainerProducerBase(cfg),
        moduleLabel_(cfg.getParameter<std::string>("@module_label")),
        mvaReader_(nullptr),
        mvaInput_(nullptr) {
    mvaName_ = cfg.getParameter<std::string>("mvaName");
    loadMVAfromDB_ = cfg.getParameter<bool>("loadMVAfromDB");
    if (!loadMVAfromDB_) {
      inputFileName_ = cfg.getParameter<edm::FileInPath>("inputFileName");
    }
    std::string mvaOpt_string = cfg.getParameter<std::string>("mvaOpt");
    if (mvaOpt_string == "oldDMwoLT")
      mvaOpt_ = kOldDMwoLT;
    else if (mvaOpt_string == "oldDMwLT")
      mvaOpt_ = kOldDMwLT;
    else if (mvaOpt_string == "newDMwoLT")
      mvaOpt_ = kNewDMwoLT;
    else if (mvaOpt_string == "newDMwLT")
      mvaOpt_ = kNewDMwLT;
    else
      throw cms::Exception("PFRecoTauDiscriminationByIsolationMVA2")
          << " Invalid Configuration Parameter 'mvaOpt' = " << mvaOpt_string << " !!\n";

    if (mvaOpt_ == kOldDMwoLT || mvaOpt_ == kNewDMwoLT)
      mvaInput_ = new float[6];
    else if (mvaOpt_ == kOldDMwLT || mvaOpt_ == kNewDMwLT)
      mvaInput_ = new float[12];
    else
      assert(0);

    tauTransverseImpactParameters_token_ =
        consumes<PFTauTIPAssociationByRef>(cfg.getParameter<edm::InputTag>("srcTauTransverseImpactParameters"));

    basicTauDiscriminators_token_ =
        consumes<reco::TauDiscriminatorContainer>(cfg.getParameter<edm::InputTag>("srcBasicTauDiscriminators"));
    chargedIsoPtSum_index_ = cfg.getParameter<int>("srcChargedIsoPtSumIndex");
    neutralIsoPtSum_index_ = cfg.getParameter<int>("srcNeutralIsoPtSumIndex");
    pucorrPtSum_index_ = cfg.getParameter<int>("srcPUcorrPtSumIndex");

    verbosity_ = cfg.getParameter<int>("verbosity");
  }

  void beginEvent(const edm::Event&, const edm::EventSetup&) override;

  reco::SingleTauDiscriminatorContainer discriminate(const PFTauRef&) const override;

  ~PFRecoTauDiscriminationByIsolationMVA2() override {
    if (!loadMVAfromDB_)
      delete mvaReader_;
    delete[] mvaInput_;
    for (std::vector<TFile*>::iterator it = inputFilesToDelete_.begin(); it != inputFilesToDelete_.end(); ++it) {
      delete (*it);
    }
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  std::string moduleLabel_;

  std::string mvaName_;
  bool loadMVAfromDB_;
  edm::FileInPath inputFileName_;
  const GBRForest* mvaReader_;
  enum { kOldDMwoLT, kOldDMwLT, kNewDMwoLT, kNewDMwLT };
  int mvaOpt_;
  float* mvaInput_;

  typedef edm::AssociationVector<reco::PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef> >
      PFTauTIPAssociationByRef;
  edm::EDGetTokenT<PFTauTIPAssociationByRef> tauTransverseImpactParameters_token_;
  edm::Handle<PFTauTIPAssociationByRef> tauLifetimeInfos_;

  edm::EDGetTokenT<reco::TauDiscriminatorContainer> basicTauDiscriminators_token_;
  edm::Handle<reco::TauDiscriminatorContainer> basicTauDiscriminators_;
  int chargedIsoPtSum_index_;
  int neutralIsoPtSum_index_;
  int pucorrPtSum_index_;

  edm::Handle<TauCollection> taus_;

  std::vector<TFile*> inputFilesToDelete_;

  int verbosity_;
};

void PFRecoTauDiscriminationByIsolationMVA2::beginEvent(const edm::Event& evt, const edm::EventSetup& es) {
  if (!mvaReader_) {
    if (loadMVAfromDB_) {
      mvaReader_ = loadMVAfromDB(es, mvaName_);
    } else {
      mvaReader_ = loadMVAfromFile(inputFileName_, mvaName_, inputFilesToDelete_);
    }
  }

  evt.getByToken(tauTransverseImpactParameters_token_, tauLifetimeInfos_);

  evt.getByToken(basicTauDiscriminators_token_, basicTauDiscriminators_);

  evt.getByToken(Tau_token, taus_);
}

reco::SingleTauDiscriminatorContainer PFRecoTauDiscriminationByIsolationMVA2::discriminate(const PFTauRef& tau) const {
  reco::SingleTauDiscriminatorContainer result;
  // CV: define dummy category index in order to use RecoTauDiscriminantCutMultiplexer module to apply WP cuts
  result.rawValues = {-1., 0.};

  // CV: computation of MVA value requires presence of leading charged hadron
  if (tau->leadChargedHadrCand().isNull())
    return 0.;

  int tauDecayMode = tau->decayMode();

  if (((mvaOpt_ == kOldDMwoLT || mvaOpt_ == kOldDMwLT) &&
       (tauDecayMode == 0 || tauDecayMode == 1 || tauDecayMode == 2 || tauDecayMode == 10)) ||
      ((mvaOpt_ == kNewDMwoLT || mvaOpt_ == kNewDMwLT) &&
       (tauDecayMode == 0 || tauDecayMode == 1 || tauDecayMode == 2 || tauDecayMode == 5 || tauDecayMode == 6 ||
        tauDecayMode == 10))) {
    double chargedIsoPtSum = (*basicTauDiscriminators_)[tau].rawValues.at(chargedIsoPtSum_index_);
    double neutralIsoPtSum = (*basicTauDiscriminators_)[tau].rawValues.at(neutralIsoPtSum_index_);
    double puCorrPtSum = (*basicTauDiscriminators_)[tau].rawValues.at(pucorrPtSum_index_);

    const reco::PFTauTransverseImpactParameter& tauLifetimeInfo = *(*tauLifetimeInfos_)[tau];

    double decayDistX = tauLifetimeInfo.flightLength().x();
    double decayDistY = tauLifetimeInfo.flightLength().y();
    double decayDistZ = tauLifetimeInfo.flightLength().z();
    double decayDistMag = TMath::Sqrt(decayDistX * decayDistX + decayDistY * decayDistY + decayDistZ * decayDistZ);

    if (mvaOpt_ == kOldDMwoLT || mvaOpt_ == kNewDMwoLT) {
      mvaInput_[0] = TMath::Log(TMath::Max(1., Double_t(tau->pt())));
      mvaInput_[1] = TMath::Abs(tau->eta());
      mvaInput_[2] = TMath::Log(TMath::Max(1.e-2, chargedIsoPtSum));
      mvaInput_[3] = TMath::Log(TMath::Max(1.e-2, neutralIsoPtSum - 0.125 * puCorrPtSum));
      mvaInput_[4] = TMath::Log(TMath::Max(1.e-2, puCorrPtSum));
      mvaInput_[5] = tauDecayMode;
    } else if (mvaOpt_ == kOldDMwLT || mvaOpt_ == kNewDMwLT) {
      mvaInput_[0] = TMath::Log(TMath::Max(1., Double_t(tau->pt())));
      mvaInput_[1] = TMath::Abs(tau->eta());
      mvaInput_[2] = TMath::Log(TMath::Max(1.e-2, chargedIsoPtSum));
      mvaInput_[3] = TMath::Log(TMath::Max(1.e-2, neutralIsoPtSum - 0.125 * puCorrPtSum));
      mvaInput_[4] = TMath::Log(TMath::Max(1.e-2, puCorrPtSum));
      mvaInput_[5] = tauDecayMode;
      mvaInput_[6] = TMath::Sign(+1., tauLifetimeInfo.dxy());
      mvaInput_[7] = TMath::Sqrt(TMath::Abs(TMath::Min(1., tauLifetimeInfo.dxy())));
      mvaInput_[8] = TMath::Min(10., TMath::Abs(tauLifetimeInfo.dxy_Sig()));
      mvaInput_[9] = (tauLifetimeInfo.hasSecondaryVertex()) ? 1. : 0.;
      mvaInput_[10] = TMath::Sqrt(decayDistMag);
      mvaInput_[11] = TMath::Min(10., tauLifetimeInfo.flightLengthSig());
    }

    double mvaValue = mvaReader_->GetClassifier(mvaInput_);
    if (verbosity_) {
      edm::LogPrint("PFTauDiscByMVAIsol2") << "<PFRecoTauDiscriminationByIsolationMVA2::discriminate>:";
      edm::LogPrint("PFTauDiscByMVAIsol2") << " tau: Pt = " << tau->pt() << ", eta = " << tau->eta();
      edm::LogPrint("PFTauDiscByMVAIsol2") << " isolation: charged = " << chargedIsoPtSum
                                           << ", neutral = " << neutralIsoPtSum << ", PUcorr = " << puCorrPtSum;
      edm::LogPrint("PFTauDiscByMVAIsol2") << " decay mode = " << tauDecayMode;
      edm::LogPrint("PFTauDiscByMVAIsol2") << " impact parameter: distance = " << tauLifetimeInfo.dxy()
                                           << ", significance = " << tauLifetimeInfo.dxy_Sig();
      edm::LogPrint("PFTauDiscByMVAIsol2")
          << " has decay vertex = " << tauLifetimeInfo.hasSecondaryVertex() << ":"
          << " distance = " << decayDistMag << ", significance = " << tauLifetimeInfo.flightLengthSig();
      edm::LogPrint("PFTauDiscByMVAIsol2") << "--> mvaValue = " << mvaValue;
    }
    result.rawValues.at(0) = mvaValue;
  }
  return result;
}

void PFRecoTauDiscriminationByIsolationMVA2::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pfRecoTauDiscriminationByIsolationMVA2
  edm::ParameterSetDescription desc;

  desc.add<std::string>("mvaName");
  desc.add<bool>("loadMVAfromDB");
  desc.addOptional<edm::FileInPath>("inputFileName");
  desc.add<std::string>("mvaOpt");

  desc.add<edm::InputTag>("srcTauTransverseImpactParameters");
  desc.add<edm::InputTag>("srcBasicTauDiscriminators");
  desc.add<int>("srcChargedIsoPtSumIndex");
  desc.add<int>("srcNeutralIsoPtSumIndex");
  desc.add<int>("srcPUcorrPtSumIndex");
  desc.add<int>("verbosity", 0);

  fillProducerDescriptions(desc);  // inherited from the base

  descriptions.add("pfRecoTauDiscriminationByIsolationMVA2", desc);
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationByIsolationMVA2);
