
/** \class PFRecoTauDiscriminationByMVAIsolation2
 *
 * MVA based discriminator against jet -> tau fakes
 * 
 * \author Christian Veelken, LLR
 *
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

#include "FWCore/Common/interface/Provenance.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameterAssociation.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "RecoTauTag/RecoTau/interface/PFRecoTauClusterVariables.h"

#include "CondFormats/GBRForest/interface/GBRForest.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

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

namespace reco {
  namespace tau {

    class PFRecoTauDiscriminationByMVAIsolationRun2 : public PFTauDiscriminationContainerProducerBase {
    public:
      explicit PFRecoTauDiscriminationByMVAIsolationRun2(const edm::ParameterSet& cfg)
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
        else
          throw cms::Exception("PFRecoTauDiscriminationByMVAIsolationRun2")
              << " Invalid Configuration Parameter 'mvaOpt' = " << mvaOpt_string << " !!\n";

        if (mvaOpt_ == kOldDMwoLT || mvaOpt_ == kNewDMwoLT)
          mvaInput_ = new float[6];
        else if (mvaOpt_ == kOldDMwLT || mvaOpt_ == kNewDMwLT)
          mvaInput_ = new float[12];
        else if (mvaOpt_ == kDBoldDMwLT || mvaOpt_ == kDBnewDMwLT || mvaOpt_ == kPWoldDMwLT || mvaOpt_ == kPWnewDMwLT ||
                 mvaOpt_ == kDBoldDMwLTwGJ || mvaOpt_ == kDBnewDMwLTwGJ)
          mvaInput_ = new float[23];
        else
          assert(0);

        TauTransverseImpactParameters_token =
            consumes<PFTauTIPAssociationByRef>(cfg.getParameter<edm::InputTag>("srcTauTransverseImpactParameters"));

        basicTauDiscriminators_token =
            consumes<reco::TauDiscriminatorContainer>(cfg.getParameter<edm::InputTag>("srcBasicTauDiscriminators"));
        input_id_name_suffix_ = cfg.getParameter<std::string>("inputIDNameSuffix");

        verbosity_ = cfg.getParameter<int>("verbosity");
      }

      void beginEvent(const edm::Event&, const edm::EventSetup&) override;

      reco::SingleTauDiscriminatorContainer discriminate(const PFTauRef&) const override;

      ~PFRecoTauDiscriminationByMVAIsolationRun2() override {
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

      typedef edm::AssociationVector<reco::PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef>>
          PFTauTIPAssociationByRef;
      edm::EDGetTokenT<PFTauTIPAssociationByRef> TauTransverseImpactParameters_token;
      edm::Handle<PFTauTIPAssociationByRef> tauLifetimeInfos;

      edm::EDGetTokenT<reco::TauDiscriminatorContainer> basicTauDiscriminators_token;
      edm::Handle<reco::TauDiscriminatorContainer> basicTauDiscriminators_;
      int chargedIsoPtSum_index_ = 0;
      int neutralIsoPtSum_index_ = 0;
      int pucorrPtSum_index_ = 0;
      int photonPtSumOutsideSignalCone_index_ = 0;
      int footprintCorrection_index_ = 0;
      std::string input_id_name_suffix_;
      edm::ProcessHistoryID phID_;

      edm::Handle<TauCollection> taus_;

      std::vector<TFile*> inputFilesToDelete_;

      int verbosity_;
    };

    void PFRecoTauDiscriminationByMVAIsolationRun2::beginEvent(const edm::Event& evt, const edm::EventSetup& es) {
      if (!mvaReader_) {
        if (loadMVAfromDB_) {
          mvaReader_ = loadMVAfromDB(es, mvaName_);
        } else {
          mvaReader_ = loadMVAfromFile(inputFileName_, mvaName_, inputFilesToDelete_);
        }
      }

      evt.getByToken(TauTransverseImpactParameters_token, tauLifetimeInfos);

      evt.getByToken(basicTauDiscriminators_token, basicTauDiscriminators_);

      evt.getByToken(Tau_token, taus_);

      // load indices from input provenance config if process history changed, in particular for the first event
      // skip missing IDs and leave treatment to produce/discriminate function
      if (evt.processHistoryID() != phID_ && basicTauDiscriminators_.isValid()) {
        phID_ = evt.processHistoryID();
        const edm::Provenance* prov = basicTauDiscriminators_.provenance();
        const std::vector<edm::ParameterSet> psetsFromProvenance =
            edm::parameterSet(prov->stable(), evt.processHistory())
                .getParameter<std::vector<edm::ParameterSet>>("IDdefinitions");
        for (uint i = 0; i < psetsFromProvenance.size(); i++) {
          if (psetsFromProvenance[i].getParameter<std::string>("IDname") == "ChargedIsoPtSum" + input_id_name_suffix_)
            chargedIsoPtSum_index_ = i;
          else if (psetsFromProvenance[i].getParameter<std::string>("IDname") ==
                   "NeutralIsoPtSum" + input_id_name_suffix_)
            neutralIsoPtSum_index_ = i;
          else if (psetsFromProvenance[i].getParameter<std::string>("IDname") == "PUcorrPtSum" + input_id_name_suffix_)
            pucorrPtSum_index_ = i;
          else if (psetsFromProvenance[i].getParameter<std::string>("IDname") ==
                   "PhotonPtSumOutsideSignalCone" + input_id_name_suffix_)
            photonPtSumOutsideSignalCone_index_ = i;
          else if (psetsFromProvenance[i].getParameter<std::string>("IDname") ==
                   "TauFootprintCorrection" + input_id_name_suffix_)
            footprintCorrection_index_ = i;
        }
      }
    }

    reco::SingleTauDiscriminatorContainer PFRecoTauDiscriminationByMVAIsolationRun2::discriminate(
        const PFTauRef& tau) const {
      reco::SingleTauDiscriminatorContainer result;
      result.rawValues = {-1.};

      // CV: computation of MVA value requires presence of leading charged hadron
      if (tau->leadChargedHadrCand().isNull())
        return 0.;

      int tauDecayMode = tau->decayMode();

      if (((mvaOpt_ == kOldDMwoLT || mvaOpt_ == kOldDMwLT || mvaOpt_ == kDBoldDMwLT || mvaOpt_ == kPWoldDMwLT ||
            mvaOpt_ == kDBoldDMwLTwGJ) &&
           (tauDecayMode == 0 || tauDecayMode == 1 || tauDecayMode == 2 || tauDecayMode == 10)) ||
          ((mvaOpt_ == kNewDMwoLT || mvaOpt_ == kNewDMwLT || mvaOpt_ == kDBnewDMwLT || mvaOpt_ == kPWnewDMwLT ||
            mvaOpt_ == kDBnewDMwLTwGJ) &&
           (tauDecayMode == 0 || tauDecayMode == 1 || tauDecayMode == 2 || tauDecayMode == 5 || tauDecayMode == 6 ||
            tauDecayMode == 10 || tauDecayMode == 11))) {
        auto const rawValues = (*basicTauDiscriminators_)[tau].rawValues;
        float chargedIsoPtSum = rawValues.at(chargedIsoPtSum_index_);
        float neutralIsoPtSum = rawValues.at(neutralIsoPtSum_index_);
        float puCorrPtSum = rawValues.at(pucorrPtSum_index_);
        float photonPtSumOutsideSignalCone = rawValues.at(photonPtSumOutsideSignalCone_index_);
        float footprintCorrection = rawValues.at(footprintCorrection_index_);

        const reco::PFTauTransverseImpactParameter& tauLifetimeInfo = *(*tauLifetimeInfos)[tau];

        float decayDistX = tauLifetimeInfo.flightLength().x();
        float decayDistY = tauLifetimeInfo.flightLength().y();
        float decayDistZ = tauLifetimeInfo.flightLength().z();
        float decayDistMag = std::sqrt(decayDistX * decayDistX + decayDistY * decayDistY + decayDistZ * decayDistZ);

        float nPhoton = (float)reco::tau::n_photons_total(*tau);
        float ptWeightedDetaStrip = reco::tau::pt_weighted_deta_strip(*tau, tauDecayMode);
        float ptWeightedDphiStrip = reco::tau::pt_weighted_dphi_strip(*tau, tauDecayMode);
        float ptWeightedDrSignal = reco::tau::pt_weighted_dr_signal(*tau, tauDecayMode);
        float ptWeightedDrIsolation = reco::tau::pt_weighted_dr_iso(*tau, tauDecayMode);
        float leadingTrackChi2 = reco::tau::lead_track_chi2(*tau);
        float eRatio = reco::tau::eratio(*tau);

        // Difference between measured and maximally allowed Gottfried-Jackson angle
        float gjAngleDiff = -999;
        if (tauDecayMode == 10) {
          double mTau = 1.77682;
          double mAOne = tau->p4().M();
          double pAOneMag = tau->p();
          double argumentThetaGJmax = (std::pow(mTau, 2) - std::pow(mAOne, 2)) / (2 * mTau * pAOneMag);
          double argumentThetaGJmeasured =
              (tau->p4().px() * decayDistX + tau->p4().py() * decayDistY + tau->p4().pz() * decayDistZ) /
              (pAOneMag * decayDistMag);
          if (std::abs(argumentThetaGJmax) <= 1. && std::abs(argumentThetaGJmeasured) <= 1.) {
            double thetaGJmax = std::asin(argumentThetaGJmax);
            double thetaGJmeasured = std::acos(argumentThetaGJmeasured);
            gjAngleDiff = thetaGJmeasured - thetaGJmax;
          }
        }

        if (mvaOpt_ == kOldDMwoLT || mvaOpt_ == kNewDMwoLT) {
          mvaInput_[0] = std::log(std::max(1.f, (float)tau->pt()));
          mvaInput_[1] = std::abs((float)tau->eta());
          mvaInput_[2] = std::log(std::max(1.e-2f, chargedIsoPtSum));
          mvaInput_[3] = std::log(std::max(1.e-2f, neutralIsoPtSum - 0.125f * puCorrPtSum));
          mvaInput_[4] = std::log(std::max(1.e-2f, puCorrPtSum));
          mvaInput_[5] = tauDecayMode;
        } else if (mvaOpt_ == kOldDMwLT || mvaOpt_ == kNewDMwLT) {
          mvaInput_[0] = std::log(std::max(1.f, (float)tau->pt()));
          mvaInput_[1] = std::abs((float)tau->eta());
          mvaInput_[2] = std::log(std::max(1.e-2f, chargedIsoPtSum));
          mvaInput_[3] = std::log(std::max(1.e-2f, neutralIsoPtSum - 0.125f * puCorrPtSum));
          mvaInput_[4] = std::log(std::max(1.e-2f, puCorrPtSum));
          mvaInput_[5] = tauDecayMode;
          mvaInput_[6] = std::copysign(+1.f, (float)tauLifetimeInfo.dxy());
          mvaInput_[7] = std::sqrt(std::abs(std::min(1.f, (float)tauLifetimeInfo.dxy())));
          mvaInput_[8] = std::min(10.f, std::abs((float)tauLifetimeInfo.dxy_Sig()));
          mvaInput_[9] = (tauLifetimeInfo.hasSecondaryVertex()) ? 1. : 0.;
          mvaInput_[10] = std::sqrt(decayDistMag);
          mvaInput_[11] = std::min(10.f, (float)tauLifetimeInfo.flightLengthSig());
        } else if (mvaOpt_ == kDBoldDMwLT || mvaOpt_ == kDBnewDMwLT) {
          mvaInput_[0] = std::log(std::max(1.f, (float)tau->pt()));
          mvaInput_[1] = std::abs((float)tau->eta());
          mvaInput_[2] = std::log(std::max(1.e-2f, chargedIsoPtSum));
          mvaInput_[3] = std::log(std::max(1.e-2f, neutralIsoPtSum));
          mvaInput_[4] = std::log(std::max(1.e-2f, puCorrPtSum));
          mvaInput_[5] = std::log(std::max(1.e-2f, photonPtSumOutsideSignalCone));
          mvaInput_[6] = tauDecayMode;
          mvaInput_[7] = std::min(30.f, nPhoton);
          mvaInput_[8] = std::min(0.5f, ptWeightedDetaStrip);
          mvaInput_[9] = std::min(0.5f, ptWeightedDphiStrip);
          mvaInput_[10] = std::min(0.5f, ptWeightedDrSignal);
          mvaInput_[11] = std::min(0.5f, ptWeightedDrIsolation);
          mvaInput_[12] = std::min(100.f, leadingTrackChi2);
          mvaInput_[13] = std::min(1.f, eRatio);
          mvaInput_[14] = std::copysign(+1.f, (float)tauLifetimeInfo.dxy());
          mvaInput_[15] = std::sqrt(std::min(1.f, std::abs((float)tauLifetimeInfo.dxy())));
          mvaInput_[16] = std::min(10.f, std::abs((float)tauLifetimeInfo.dxy_Sig()));
          mvaInput_[17] = std::copysign(+1.f, (float)tauLifetimeInfo.ip3d());
          mvaInput_[18] = std::sqrt(std::min(1.f, std::abs((float)tauLifetimeInfo.ip3d())));
          mvaInput_[19] = std::min(10.f, std::abs((float)tauLifetimeInfo.ip3d_Sig()));
          mvaInput_[20] = (tauLifetimeInfo.hasSecondaryVertex()) ? 1. : 0.;
          mvaInput_[21] = std::sqrt(decayDistMag);
          mvaInput_[22] = std::min(10.f, (float)tauLifetimeInfo.flightLengthSig());
        } else if (mvaOpt_ == kPWoldDMwLT || mvaOpt_ == kPWnewDMwLT) {
          mvaInput_[0] = std::log(std::max(1.f, (float)tau->pt()));
          mvaInput_[1] = std::abs((float)tau->eta());
          mvaInput_[2] = std::log(std::max(1.e-2f, chargedIsoPtSum));
          mvaInput_[3] = std::log(std::max(1.e-2f, neutralIsoPtSum));
          mvaInput_[4] = std::log(std::max(1.e-2f, footprintCorrection));
          mvaInput_[5] = std::log(std::max(1.e-2f, photonPtSumOutsideSignalCone));
          mvaInput_[6] = tauDecayMode;
          mvaInput_[7] = std::min(30.f, nPhoton);
          mvaInput_[8] = std::min(0.5f, ptWeightedDetaStrip);
          mvaInput_[9] = std::min(0.5f, ptWeightedDphiStrip);
          mvaInput_[10] = std::min(0.5f, ptWeightedDrSignal);
          mvaInput_[11] = std::min(0.5f, ptWeightedDrIsolation);
          mvaInput_[12] = std::min(100.f, leadingTrackChi2);
          mvaInput_[13] = std::min(1.f, eRatio);
          mvaInput_[14] = std::copysign(+1.f, (float)tauLifetimeInfo.dxy());
          mvaInput_[15] = std::sqrt(std::min(1.f, std::abs((float)tauLifetimeInfo.dxy())));
          mvaInput_[16] = std::min(10.f, std::abs((float)tauLifetimeInfo.dxy_Sig()));
          mvaInput_[17] = std::copysign(+1.f, (float)tauLifetimeInfo.ip3d());
          mvaInput_[18] = std::sqrt(std::min(1.f, std::abs((float)tauLifetimeInfo.ip3d())));
          mvaInput_[19] = std::min(10.f, std::abs((float)tauLifetimeInfo.ip3d_Sig()));
          mvaInput_[20] = (tauLifetimeInfo.hasSecondaryVertex()) ? 1. : 0.;
          mvaInput_[21] = std::sqrt(decayDistMag);
          mvaInput_[22] = std::min(10.f, (float)tauLifetimeInfo.flightLengthSig());
        } else if (mvaOpt_ == kDBoldDMwLTwGJ || mvaOpt_ == kDBnewDMwLTwGJ) {
          mvaInput_[0] = std::log(std::max(1.f, (float)tau->pt()));
          mvaInput_[1] = std::abs((float)tau->eta());
          mvaInput_[2] = std::log(std::max(1.e-2f, chargedIsoPtSum));
          mvaInput_[3] = std::log(std::max(1.e-2f, neutralIsoPtSum));
          mvaInput_[4] = std::log(std::max(1.e-2f, puCorrPtSum));
          mvaInput_[5] = std::log(std::max(1.e-2f, photonPtSumOutsideSignalCone));
          mvaInput_[6] = tauDecayMode;
          mvaInput_[7] = std::min(30.f, nPhoton);
          mvaInput_[8] = std::min(0.5f, ptWeightedDetaStrip);
          mvaInput_[9] = std::min(0.5f, ptWeightedDphiStrip);
          mvaInput_[10] = std::min(0.5f, ptWeightedDrSignal);
          mvaInput_[11] = std::min(0.5f, ptWeightedDrIsolation);
          mvaInput_[12] = std::min(1.f, eRatio);
          mvaInput_[13] = std::copysign(+1.f, (float)tauLifetimeInfo.dxy());
          mvaInput_[14] = std::sqrt(std::min(1.f, std::abs((float)tauLifetimeInfo.dxy())));
          mvaInput_[15] = std::min(10.f, std::abs((float)tauLifetimeInfo.dxy_Sig()));
          mvaInput_[16] = std::copysign(+1.f, (float)tauLifetimeInfo.ip3d());
          mvaInput_[17] = std::sqrt(std::min(1.f, std::abs((float)tauLifetimeInfo.ip3d())));
          mvaInput_[18] = std::min(10.f, std::abs((float)tauLifetimeInfo.ip3d_Sig()));
          mvaInput_[19] = (tauLifetimeInfo.hasSecondaryVertex()) ? 1. : 0.;
          mvaInput_[20] = std::sqrt(decayDistMag);
          mvaInput_[21] = std::min(10.f, (float)tauLifetimeInfo.flightLengthSig());
          mvaInput_[22] = std::max(-1.f, gjAngleDiff);
        }

        double mvaValue = mvaReader_->GetClassifier(mvaInput_);
        if (verbosity_) {
          edm::LogPrint("PFTauDiscByMVAIsol2") << "<PFRecoTauDiscriminationByMVAIsolationRun2::discriminate>:";
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

    void PFRecoTauDiscriminationByMVAIsolationRun2::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      // pfRecoTauDiscriminationByMVAIsolationRun2
      edm::ParameterSetDescription desc;

      desc.add<std::string>("mvaName", "tauIdMVAnewDMwLT");
      desc.add<bool>("loadMVAfromDB", true);
      desc.addOptional<edm::FileInPath>("inputFileName");
      desc.add<std::string>("mvaOpt", "newDMwLT");

      desc.add<edm::InputTag>("srcTauTransverseImpactParameters", edm::InputTag(""));
      desc.add<edm::InputTag>("srcBasicTauDiscriminators", edm::InputTag("hpsPFTauBasicDiscriminators"));
      desc.add<std::string>("inputIDNameSuffix", "");

      desc.add<int>("verbosity", 0);

      fillProducerDescriptions(desc);  // inherited from the base

      descriptions.add("pfRecoTauDiscriminationByMVAIsolationRun2", desc);
    }

    DEFINE_FWK_MODULE(PFRecoTauDiscriminationByMVAIsolationRun2);

  }  // namespace tau
}  // namespace reco
