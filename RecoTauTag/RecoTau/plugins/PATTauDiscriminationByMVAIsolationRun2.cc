
/*
 * \class PATTauDiscriminationByMVAIsolationRun2
 * 
 * MVA based discriminator against jet -> tau fakes
 * 
 * Adopted from RecoTauTag/RecoTau/plugins/PFRecoTauDiscriminationByMVAIsolationRun2.cc
 * to enable computation of MVA isolation on MiniAOD
 * 
 * \author Alexander Nehrkorn, RWTH Aachen
 */

// todo 1: remove leadingTrackChi2 as input variable from:
//           - here
//           - TauPFEssential
//           - PFRecoTauDiscriminationByMVAIsolationRun2
//           - Training of BDT

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
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/PATTauDiscriminator.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "RecoTauTag/RecoTau/interface/PFRecoTauClusterVariables.h"

#include "CondFormats/GBRForest/interface/GBRForest.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <TFile.h>

#include <iostream>

using namespace pat;

namespace {
  const GBRForest* loadMVAfromFile(const edm::FileInPath& inputFileName,
                                   const std::string& mvaName,
                                   std::vector<TFile*>& inputFilesToDelete) {
    if (inputFileName.location() == edm::FileInPath::Unknown)
      throw cms::Exception("PATTauDiscriminationByIsolationMVARun2::loadMVA")
          << " Failed to find File = " << inputFileName << " !!\n";
    TFile* inputFile = new TFile(inputFileName.fullPath().data());

    //const GBRForest* mva = dynamic_cast<GBRForest*>(inputFile->Get(mvaName.data())); // CV: dynamic_cast<GBRForest*> fails for some reason ?!
    const GBRForest* mva = (GBRForest*)inputFile->Get(mvaName.data());
    if (!mva)
      throw cms::Exception("PATTauDiscriminationByIsolationMVARun2::loadMVA")
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

namespace reco {
  namespace tau {

    class PATTauDiscriminationByMVAIsolationRun2 : public PATTauDiscriminationContainerProducerBase {
    public:
      explicit PATTauDiscriminationByMVAIsolationRun2(const edm::ParameterSet& cfg)
          : PATTauDiscriminationContainerProducerBase(cfg),
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
        else if (mvaOpt_string == "DBoldDMwLT")
          mvaOpt_ = kDBoldDMwLT;
        else if (mvaOpt_string == "DBnewDMwLT")
          mvaOpt_ = kDBnewDMwLT;
        else if (mvaOpt_string == "PWoldDMwLT")
          mvaOpt_ = kPWoldDMwLT;
        else if (mvaOpt_string == "PWnewDMwLT")
          mvaOpt_ = kPWnewDMwLT;
        else if (mvaOpt_string == "DBoldDMwLTwGJ")
          mvaOpt_ = kDBoldDMwLTwGJ;
        else if (mvaOpt_string == "DBnewDMwLTwGJ")
          mvaOpt_ = kDBnewDMwLTwGJ;
        else if (mvaOpt_string == "DBnewDMwLTwGJPhase2")
          mvaOpt_ = kDBnewDMwLTwGJPhase2;
        else
          throw cms::Exception("PATTauDiscriminationByMVAIsolationRun2")
              << " Invalid Configuration Parameter 'mvaOpt' = " << mvaOpt_string << " !!\n";

        if (mvaOpt_ == kOldDMwoLT || mvaOpt_ == kNewDMwoLT)
          mvaInput_ = new float[6];
        else if (mvaOpt_ == kOldDMwLT || mvaOpt_ == kNewDMwLT)
          mvaInput_ = new float[12];
        else if (mvaOpt_ == kDBoldDMwLT || mvaOpt_ == kDBnewDMwLT || mvaOpt_ == kPWoldDMwLT || mvaOpt_ == kPWnewDMwLT ||
                 mvaOpt_ == kDBoldDMwLTwGJ || mvaOpt_ == kDBnewDMwLTwGJ)
          mvaInput_ = new float[23];
        else if (mvaOpt_ == kDBnewDMwLTwGJPhase2)
          mvaInput_ = new float[30];
        else
          assert(0);

        chargedIsoPtSums_ = cfg.getParameter<std::string>("srcChargedIsoPtSum");
        neutralIsoPtSums_ = cfg.getParameter<std::string>("srcNeutralIsoPtSum");
        puCorrPtSums_ = cfg.getParameter<std::string>("srcPUcorrPtSum");
        photonPtSumOutsideSignalCone_ = cfg.getParameter<std::string>("srcPhotonPtSumOutsideSignalCone");
        footprintCorrection_ = cfg.getParameter<std::string>("srcFootprintCorrection");

        verbosity_ = cfg.getParameter<int>("verbosity");
      }

      void beginEvent(const edm::Event&, const edm::EventSetup&) override;

      reco::SingleTauDiscriminatorContainer discriminate(const TauRef&) const override;

      ~PATTauDiscriminationByMVAIsolationRun2() override {
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
      int mvaOpt_;
      float* mvaInput_;

      std::string chargedIsoPtSums_;
      std::string neutralIsoPtSums_;
      std::string puCorrPtSums_;
      std::string photonPtSumOutsideSignalCone_;
      std::string footprintCorrection_;

      edm::Handle<TauCollection> taus_;
      std::vector<TFile*> inputFilesToDelete_;

      int verbosity_;
    };

    void PATTauDiscriminationByMVAIsolationRun2::beginEvent(const edm::Event& evt, const edm::EventSetup& es) {
      if (!mvaReader_) {
        if (loadMVAfromDB_) {
          mvaReader_ = loadMVAfromDB(es, mvaName_);
        } else {
          mvaReader_ = loadMVAfromFile(inputFileName_, mvaName_, inputFilesToDelete_);
        }
      }

      evt.getByToken(Tau_token, taus_);
    }

    reco::SingleTauDiscriminatorContainer PATTauDiscriminationByMVAIsolationRun2::discriminate(const TauRef& tau) const {
      // CV: define dummy category index in order to use RecoTauDiscriminantCutMultiplexer module to appy WP cuts
      reco::SingleTauDiscriminatorContainer result;
      result.rawValues = {-1.};

      // CV: computation of MVA value requires presence of leading charged hadron
      if (tau->leadChargedHadrCand().isNull()) {
        result.rawValues.at(0) = 0.;
        return result;
      }

      if (reco::tau::fillIsoMVARun2Inputs(mvaInput_,
                                          *tau,
                                          mvaOpt_,
                                          chargedIsoPtSums_,
                                          neutralIsoPtSums_,
                                          puCorrPtSums_,
                                          photonPtSumOutsideSignalCone_,
                                          footprintCorrection_)) {
        double mvaValue = mvaReader_->GetClassifier(mvaInput_);
        if (verbosity_) {
          edm::LogPrint("PATTauDiscByMVAIsolRun2") << "<PATTauDiscriminationByMVAIsolationRun2::discriminate>:";
          edm::LogPrint("PATTauDiscByMVAIsolRun2") << " tau: Pt = " << tau->pt() << ", eta = " << tau->eta();
          edm::LogPrint("PATTauDiscByMVAIsolRun2")
              << " isolation: charged = " << tau->tauID(chargedIsoPtSums_)
              << ", neutral = " << tau->tauID(neutralIsoPtSums_) << ", PUcorr = " << tau->tauID(puCorrPtSums_);
          edm::LogPrint("PATTauDiscByMVAIsolRun2") << " decay mode = " << tau->decayMode();
          edm::LogPrint("PATTauDiscByMVAIsolRun2")
              << " impact parameter: distance = " << tau->dxy() << ", significance = " << tau->dxy_Sig();
          edm::LogPrint("PATTauDiscByMVAIsolRun2") << " has decay vertex = " << tau->hasSecondaryVertex() << ":"
                                                   << ", significance = " << tau->flightLengthSig();
          edm::LogPrint("PATTauDiscByMVAIsolRun2") << "--> mvaValue = " << mvaValue;
        }
        result.rawValues.at(0) = mvaValue;
      }
      return result;
    }

    void PATTauDiscriminationByMVAIsolationRun2::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      // patTauDiscriminationByMVAIsolationRun2
      edm::ParameterSetDescription desc;

      desc.add<std::string>("mvaName");
      desc.add<bool>("loadMVAfromDB");
      desc.addOptional<edm::FileInPath>("inputFileName");
      desc.add<std::string>("mvaOpt");

      desc.add<std::string>("srcChargedIsoPtSum");
      desc.add<std::string>("srcNeutralIsoPtSum");
      desc.add<std::string>("srcPUcorrPtSum");
      desc.add<std::string>("srcPhotonPtSumOutsideSignalCone");
      desc.add<std::string>("srcFootprintCorrection");
      desc.add<int>("verbosity", 0);

      fillProducerDescriptions(desc);  // inherited from the base

      descriptions.add("patTauDiscriminationByMVAIsolationRun2", desc);
    }

    DEFINE_FWK_MODULE(PATTauDiscriminationByMVAIsolationRun2);

  }  // namespace tau
}  // namespace reco
