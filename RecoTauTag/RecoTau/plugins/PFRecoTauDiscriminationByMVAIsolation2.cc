
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

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameterAssociation.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "CondFormats/EgammaObjects/interface/GBRForest.h"

#include <TMath.h>
#include <TFile.h>

#include <iostream>

using namespace reco;

namespace
{
  const GBRForest* loadMVAfromFile(const edm::FileInPath& inputFileName, const std::string& mvaName, std::vector<TFile*>& inputFilesToDelete)
  {
    if ( inputFileName.location() == edm::FileInPath::Unknown ) throw cms::Exception("PFRecoTauDiscriminationByIsolationMVA2::loadMVA") 
      << " Failed to find File = " << inputFileName << " !!\n";
    TFile* inputFile = new TFile(inputFileName.fullPath().data());
  
    //const GBRForest* mva = dynamic_cast<GBRForest*>(inputFile->Get(mvaName.data())); // CV: dynamic_cast<GBRForest*> fails for some reason ?!
    const GBRForest* mva = (GBRForest*)inputFile->Get(mvaName.data());
    if ( !mva )
      throw cms::Exception("PFRecoTauDiscriminationByIsolationMVA2::loadMVA")
        << " Failed to load MVA = " << mvaName.data() << " from file = " << inputFileName.fullPath().data() << " !!\n";

    inputFilesToDelete.push_back(inputFile);
    
    return mva;
  }
}

class PFRecoTauDiscriminationByIsolationMVA2 : public PFTauDiscriminationProducerBase  
{
 public:
  explicit PFRecoTauDiscriminationByIsolationMVA2(const edm::ParameterSet& cfg)
    : PFTauDiscriminationProducerBase(cfg),
      moduleLabel_(cfg.getParameter<std::string>("@module_label")),
      mvaReader_(0),
      mvaInput_(0),
      category_output_(0)
  {
    inputFileName_ = cfg.getParameter<edm::FileInPath>("inputFileName");
    mvaName_ = cfg.getParameter<std::string>("mvaName");
    mvaReader_ = loadMVAfromFile(inputFileName_, mvaName_, inputFilesToDelete_);
    std::string mvaOpt_string = cfg.getParameter<std::string>("mvaOpt");
    if      ( mvaOpt_string == "oldDMwoLT" ) mvaOpt_ = kOldDMwoLT;
    else if ( mvaOpt_string == "oldDMwLT"  ) mvaOpt_ = kOldDMwLT;
    else if ( mvaOpt_string == "newDMwoLT" ) mvaOpt_ = kNewDMwoLT;
    else if ( mvaOpt_string == "newDMwLT"  ) mvaOpt_ = kNewDMwLT;
    else throw cms::Exception("PFRecoTauDiscriminationByIsolationMVA2")
      << " Invalid Configuration Parameter 'mvaOpt' = " << mvaOpt_string << " !!\n";
    
    if      ( mvaOpt_ == kOldDMwoLT || mvaOpt_ == kNewDMwoLT ) mvaInput_ = new float[6];
    else if ( mvaOpt_ == kOldDMwLT  || mvaOpt_ == kNewDMwLT  ) mvaInput_ = new float[12];
    else assert(0);

    TauTransverseImpactParameters_token = consumes<PFTauTIPAssociationByRef>(cfg.getParameter<edm::InputTag>("srcTauTransverseImpactParameters"));
    
    ChargedIsoPtSum_token = consumes<reco::PFTauDiscriminator>(cfg.getParameter<edm::InputTag>("srcChargedIsoPtSum"));
    NeutralIsoPtSum_token = consumes<reco::PFTauDiscriminator>(cfg.getParameter<edm::InputTag>("srcNeutralIsoPtSum"));
    PUcorrPtSum_token = consumes<reco::PFTauDiscriminator>(cfg.getParameter<edm::InputTag>("srcPUcorrPtSum"));
  
    verbosity_ = ( cfg.exists("verbosity") ) ?
      cfg.getParameter<int>("verbosity") : 0;

    produces<PFTauDiscriminator>("category");
  }

  void beginEvent(const edm::Event&, const edm::EventSetup&);

  double discriminate(const PFTauRef&);

  void endEvent(edm::Event&);

  ~PFRecoTauDiscriminationByIsolationMVA2()
  {
    delete mvaReader_;
    delete[] mvaInput_;
    for ( std::vector<TFile*>::iterator it = inputFilesToDelete_.begin();
	  it != inputFilesToDelete_.end(); ++it ) {
      delete (*it);
    }
  }

 private:

  std::string moduleLabel_;

  edm::FileInPath inputFileName_;
  std::string mvaName_;
  const GBRForest* mvaReader_;
  enum { kOldDMwoLT, kOldDMwLT, kNewDMwoLT, kNewDMwLT };
  int mvaOpt_;
  float* mvaInput_;

  typedef edm::AssociationVector<reco::PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef> > PFTauTIPAssociationByRef;
  edm::EDGetTokenT<PFTauTIPAssociationByRef> TauTransverseImpactParameters_token;
  edm::Handle<PFTauTIPAssociationByRef> tauLifetimeInfos;

  edm::EDGetTokenT<reco::PFTauDiscriminator> ChargedIsoPtSum_token;
  edm::Handle<reco::PFTauDiscriminator> chargedIsoPtSums_;
  edm::EDGetTokenT<reco::PFTauDiscriminator> NeutralIsoPtSum_token;
  edm::Handle<reco::PFTauDiscriminator> neutralIsoPtSums_;
  edm::EDGetTokenT<reco::PFTauDiscriminator> PUcorrPtSum_token;
  edm::Handle<reco::PFTauDiscriminator> puCorrPtSums_;

  edm::Handle<TauCollection> taus_;
  std::auto_ptr<PFTauDiscriminator> category_output_;
  size_t tauIndex_;

  std::vector<TFile*> inputFilesToDelete_;

  int verbosity_;
};

void PFRecoTauDiscriminationByIsolationMVA2::beginEvent(const edm::Event& evt, const edm::EventSetup& es)
{
  evt.getByToken(TauTransverseImpactParameters_token, tauLifetimeInfos);

  evt.getByToken(ChargedIsoPtSum_token, chargedIsoPtSums_);
  evt.getByToken(NeutralIsoPtSum_token, neutralIsoPtSums_);
  evt.getByToken(PUcorrPtSum_token, puCorrPtSums_);

  evt.getByToken(Tau_token, taus_);
  category_output_.reset(new PFTauDiscriminator(TauRefProd(taus_)));
  tauIndex_ = 0;
}

double PFRecoTauDiscriminationByIsolationMVA2::discriminate(const PFTauRef& tau)
{
  // CV: define dummy category index in order to use RecoTauDiscriminantCutMultiplexer module to appy WP cuts
  double category = 0.; 
  category_output_->setValue(tauIndex_, category);
  ++tauIndex_;

  // CV: computation of MVA value requires presence of leading charged hadron
  if ( tau->leadPFChargedHadrCand().isNull() ) return 0.;

  int tauDecayMode = tau->decayMode();

  if ( ((mvaOpt_ == kOldDMwoLT || mvaOpt_ == kOldDMwLT) && (tauDecayMode == 0 || tauDecayMode == 1 || tauDecayMode == 2 || tauDecayMode == 10)) ||
       ((mvaOpt_ == kNewDMwoLT || mvaOpt_ == kNewDMwLT) && (tauDecayMode == 0 || tauDecayMode == 1 || tauDecayMode == 2 || tauDecayMode == 5 || tauDecayMode == 6 || tauDecayMode == 10)) ) {

    double chargedIsoPtSum = (*chargedIsoPtSums_)[tau];
    double neutralIsoPtSum = (*neutralIsoPtSums_)[tau];
    double puCorrPtSum     = (*puCorrPtSums_)[tau];
    
    const reco::PFTauTransverseImpactParameter& tauLifetimeInfo = *(*tauLifetimeInfos)[tau];
    
    double decayDistX = tauLifetimeInfo.flightLength().x();
    double decayDistY = tauLifetimeInfo.flightLength().y();
    double decayDistZ = tauLifetimeInfo.flightLength().z();
    double decayDistMag = TMath::Sqrt(decayDistX*decayDistX + decayDistY*decayDistY + decayDistZ*decayDistZ);
    
    if ( mvaOpt_ == kOldDMwoLT || mvaOpt_ == kNewDMwoLT ) {
      mvaInput_[0]  = TMath::Log(TMath::Max(1., Double_t(tau->pt())));
      mvaInput_[1]  = TMath::Abs(tau->eta());
      mvaInput_[2]  = TMath::Log(TMath::Max(1.e-2, chargedIsoPtSum));
      mvaInput_[3]  = TMath::Log(TMath::Max(1.e-2, neutralIsoPtSum - 0.125*puCorrPtSum));
      mvaInput_[4]  = TMath::Log(TMath::Max(1.e-2, puCorrPtSum));
      mvaInput_[5]  = tauDecayMode;
    } else if ( mvaOpt_ == kOldDMwLT || mvaOpt_ == kNewDMwLT  ) {
      mvaInput_[0]  = TMath::Log(TMath::Max(1., Double_t(tau->pt())));
      mvaInput_[1]  = TMath::Abs(tau->eta());
      mvaInput_[2]  = TMath::Log(TMath::Max(1.e-2, chargedIsoPtSum));
      mvaInput_[3]  = TMath::Log(TMath::Max(1.e-2, neutralIsoPtSum - 0.125*puCorrPtSum));
      mvaInput_[4]  = TMath::Log(TMath::Max(1.e-2, puCorrPtSum));
      mvaInput_[5]  = tauDecayMode;
      mvaInput_[6]  = TMath::Sign(+1., tauLifetimeInfo.dxy());
      mvaInput_[7]  = TMath::Sqrt(TMath::Abs(TMath::Min(1., tauLifetimeInfo.dxy())));
      mvaInput_[8]  = TMath::Min(10., TMath::Abs(tauLifetimeInfo.dxy_Sig()));
      mvaInput_[9]  = ( tauLifetimeInfo.hasSecondaryVertex() ) ? 1. : 0.;
      mvaInput_[10] = TMath::Sqrt(decayDistMag);
      mvaInput_[11] = TMath::Min(10., tauLifetimeInfo.flightLengthSig());
    }
        
    double mvaValue = mvaReader_->GetClassifier(mvaInput_);
    if ( verbosity_ ) {
      std::cout << "<PFRecoTauDiscriminationByIsolationMVA2::discriminate>:" << std::endl;
      std::cout << " tau: Pt = " << tau->pt() << ", eta = " << tau->eta() << std::endl;
      std::cout << " isolation: charged = " << chargedIsoPtSum << ", neutral = " << neutralIsoPtSum << ", PUcorr = " << puCorrPtSum << std::endl;
      std::cout << " decay mode = " << tauDecayMode << std::endl;
      std::cout << " impact parameter: distance = " << tauLifetimeInfo.dxy() << ", significance = " << tauLifetimeInfo.dxy_Sig() << std::endl;
      std::cout << " has decay vertex = " << tauLifetimeInfo.hasSecondaryVertex() << ":"
		<< " distance = " << decayDistMag << ", significance = " << tauLifetimeInfo.flightLengthSig() << std::endl;
      std::cout << "--> mvaValue = " << mvaValue << std::endl;
    }
    return mvaValue;
  } else {
    return -1.;
  }
}

void PFRecoTauDiscriminationByIsolationMVA2::endEvent(edm::Event& evt)
{
  // add all category indices to event
  evt.put(category_output_, "category");
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationByIsolationMVA2);
