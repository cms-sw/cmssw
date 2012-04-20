/* class PFRecoTauDiscriminationAgainstElectronMVA2
 * created : Apr 10 2012,
 * revised : ,
 * Authorss : Ivo Naranjo (LLR Ecole-Polytechnique)
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "RecoTauTag/RecoTau/interface/AntiElectronIDMVA2.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <TMath.h>

using namespace reco;

class PFRecoTauDiscriminationAgainstElectronMVA2 : public PFTauDiscriminationProducerBase  {
public:
  explicit PFRecoTauDiscriminationAgainstElectronMVA2(const edm::ParameterSet& iConfig)
    : PFTauDiscriminationProducerBase(iConfig),
      mva_(0)
  {    
    method_                                     = iConfig.getParameter<std::string>("method");
    inputFileName1prongNoEleMatchBL_            = iConfig.getParameter<edm::FileInPath>("inputFileName1prongNoEleMatchBL");
    inputFileName1prongBL_                      = iConfig.getParameter<edm::FileInPath>("inputFileName1prongBL");
    inputFileName1prongStripsWOgsfBL_           = iConfig.getParameter<edm::FileInPath>("inputFileName1prongStripsWOgsfBL");
    inputFileName1prongStripsWgsfWOpfEleMvaBL_  = iConfig.getParameter<edm::FileInPath>("inputFileName1prongStripsWgsfWOpfEleMvaBL");
    inputFileName1prongStripsWgsfWpfEleMvaBL_   = iConfig.getParameter<edm::FileInPath>("inputFileName1prongStripsWgsfWpfEleMvaBL");
    inputFileName1prongNoEleMatchEC_            = iConfig.getParameter<edm::FileInPath>("inputFileName1prongNoEleMatchEC");
    inputFileName1prongEC_                      = iConfig.getParameter<edm::FileInPath>("inputFileName1prongEC");
    inputFileName1prongStripsWOgsfEC_           = iConfig.getParameter<edm::FileInPath>("inputFileName1prongStripsWOgsfEC");
    inputFileName1prongStripsWgsfWOpfEleMvaEC_  = iConfig.getParameter<edm::FileInPath>("inputFileName1prongStripsWgsfWOpfEleMvaEC");
    inputFileName1prongStripsWgsfWpfEleMvaEC_   = iConfig.getParameter<edm::FileInPath>("inputFileName1prongStripsWgsfWpfEleMvaEC");

    returnMVA_                           = iConfig.getParameter<bool>("returnMVA");
    minMVA1prongNoEleMatchBL_            = iConfig.getParameter<double>("minMVA1prongNoEleMatchBL");
    minMVA1prongBL_                      = iConfig.getParameter<double>("minMVA1prongBL");
    minMVA1prongStripsWOgsfBL_           = iConfig.getParameter<double>("minMVA1prongStripsWOgsfBL");
    minMVA1prongStripsWgsfWOpfEleMvaBL_  = iConfig.getParameter<double>("minMVA1prongStripsWgsfWOpfEleMvaBL");
    minMVA1prongStripsWgsfWpfEleMvaBL_   = iConfig.getParameter<double>("minMVA1prongStripsWgsfWpfEleMvaBL");
    minMVA1prongNoEleMatchEC_            = iConfig.getParameter<double>("minMVA1prongNoEleMatchEC");
    minMVA1prongEC_                      = iConfig.getParameter<double>("minMVA1prongEC");
    minMVA1prongStripsWOgsfEC_           = iConfig.getParameter<double>("minMVA1prongStripsWOgsfEC");
    minMVA1prongStripsWgsfWOpfEleMvaEC_  = iConfig.getParameter<double>("minMVA1prongStripsWgsfWOpfEleMvaEC");
    minMVA1prongStripsWgsfWpfEleMvaEC_   = iConfig.getParameter<double>("minMVA1prongStripsWgsfWpfEleMvaEC");

    srcGsfElectrons_ = iConfig.getParameter<edm::InputTag>("srcGsfElectrons");

    mva_ = new AntiElectronIDMVA2();
    mva_->Initialize(method_,
		     inputFileName1prongNoEleMatchBL_.fullPath().data(),
		     inputFileName1prongBL_.fullPath().data(),
		     inputFileName1prongStripsWOgsfBL_.fullPath().data(),
		     inputFileName1prongStripsWgsfWOpfEleMvaBL_.fullPath().data(),
		     inputFileName1prongStripsWgsfWpfEleMvaBL_.fullPath().data(),
		     inputFileName1prongNoEleMatchEC_.fullPath().data(),
		     inputFileName1prongEC_.fullPath().data(),
		     inputFileName1prongStripsWOgsfEC_.fullPath().data(),
		     inputFileName1prongStripsWgsfWOpfEleMvaEC_.fullPath().data(),
		     inputFileName1prongStripsWgsfWpfEleMvaEC_.fullPath().data());
  }

  double discriminate(const PFTauRef& thePFTauRef);

  void beginEvent(const edm::Event& evt, const edm::EventSetup& es);

  ~PFRecoTauDiscriminationAgainstElectronMVA2(){ delete mva_;}

private:
  
  std::string method_ ;
  edm::FileInPath inputFileName1prongNoEleMatchBL_;
  edm::FileInPath inputFileName1prongBL_;
  edm::FileInPath inputFileName1prongStripsWOgsfBL_;
  edm::FileInPath inputFileName1prongStripsWgsfWOpfEleMvaBL_;
  edm::FileInPath inputFileName1prongStripsWgsfWpfEleMvaBL_;
  edm::FileInPath inputFileName1prongNoEleMatchEC_;
  edm::FileInPath inputFileName1prongEC_;
  edm::FileInPath inputFileName1prongStripsWOgsfEC_;
  edm::FileInPath inputFileName1prongStripsWgsfWOpfEleMvaEC_;
  edm::FileInPath inputFileName1prongStripsWgsfWpfEleMvaEC_;
  AntiElectronIDMVA2* mva_;
  bool returnMVA_ ;
  double minMVA1prongNoEleMatchBL_ ;
  double minMVA1prongBL_ ;
  double minMVA1prongStripsWOgsfBL_ ;
  double minMVA1prongStripsWgsfWOpfEleMvaBL_ ;
  double minMVA1prongStripsWgsfWpfEleMvaBL_ ;
  double minMVA1prongNoEleMatchEC_ ;
  double minMVA1prongEC_ ;
  double minMVA1prongStripsWOgsfEC_;
  double minMVA1prongStripsWgsfWOpfEleMvaEC_ ;
  double minMVA1prongStripsWgsfWpfEleMvaEC_ ;
  edm::InputTag srcGsfElectrons_;
  edm::Handle<reco::GsfElectronCollection> gsfElectrons_;
};

void PFRecoTauDiscriminationAgainstElectronMVA2::beginEvent(const edm::Event& evt, const edm::EventSetup& es)
{
  evt.getByLabel(srcGsfElectrons_, gsfElectrons_);
}

double PFRecoTauDiscriminationAgainstElectronMVA2::discriminate(const PFTauRef& thePFTauRef)
{
  double mva = 1.;
  double workingPoint = 1.;
  bool isGsfElectronMatched = false;
  if( (*thePFTauRef).leadPFChargedHadrCand().isNonnull()) {
    for ( reco::GsfElectronCollection::const_iterator theGsfElectron = gsfElectrons_->begin();
	  theGsfElectron != gsfElectrons_->end(); ++theGsfElectron ) {
      if ( theGsfElectron->pt() > 10. ) { // CV: only take electrons above some minimal energy/Pt into account...
	double deltaREleTau = deltaR(theGsfElectron->p4(), thePFTauRef->p4());
	if ( deltaREleTau < 0.3 ) {
	  mva_match = mva_->MVAValue(*thePFTauRef, *theGsfElectron);
	  
	  size_t numSignalPFGammaCands = thePFTauRef->signalPFGammaCands().size();
	  bool hasGsfTrack = thePFTauRef->leadPFChargedHadrCand()->gsfTrackRef().isNonnull();
	  bool isPFElectron = (theGsfElectron->mvaOutput().mva > -0.1);

	  if ( thePFTauRef->signalPFChargedHadrCands().size() == 1 ) {
	    double mvaCut = 999.;
	    if ( TMath::Abs(thePFTauRef->eta()) < 1.5 ) { // Barrel
	      if      ( numSignalPFGammaCands == 0                                  ) mvaCut = minMVA1prongBL_;
	      else if ( numSignalPFGammaCands >= 1 && !hasGsfTrack                  ) mvaCut = minMVA1prongStripsWOgsfBL_;
	      else if ( numSignalPFGammaCands >= 1 &&  hasGsfTrack && !isPFElectron ) mvaCut = minMVA1prongStripsWgsfWOpfEleMvaBL_;
	      else if ( numSignalPFGammaCands >= 1 &&  hasGsfTrack &&  isPFElectron ) mvaCut = minMVA1prongStripsWgsfWpfEleMvaBL_;
	    } else { // Endcap
	      if      ( numSignalPFGammaCands == 0                                  ) mvaCut = minMVA1prongEC_;
	      else if ( numSignalPFGammaCands >= 1 && !hasGsfTrack                  ) mvaCut = minMVA1prongStripsWOgsfEC_;
	      else if ( numSignalPFGammaCands >= 1 &&  hasGsfTrack && !isPFElectron ) mvaCut = minMVA1prongStripsWgsfWOpfEleMvaEC_;
	      else if ( numSignalPFGammaCands >= 1 &&  hasGsfTrack &&  isPFElectron ) mvaCut = minMVA1prongStripsWgsfWpfEleMvaEC_;
	    }
	    workingPoint_match = (mva_match > mvaCut);            
	  } else {
	    workingPoint_match = 1.;
	  } 

	  mva = TMath::Min(mva, mva_match);
          workingPoint = TMath::Min(workingPoint, workingPoint_match);
          isGsfElectronMatched = true;
	}
      }
    }
  }

  if ( !isGsfElectronMatched ) {
    mva = mva_->MVAValue(*thePFTauRef, *theGsfElectron);
    double mvaCut = 999.;
    if ( TMath::Abs(thePFTauRef->eta()) < 1.5 ) { // Barrel
      mvaCut = minMVA1prongNoEleMatchBL_;
    } else { // Endcap
      mvaCut = minMVA1prongNoEleMatchEC_;
    }
    workingPoint = (mva_match > mvaCut); 	
  }	

  //std::cout << "<PFRecoTauDiscriminationAgainstElectronMVA2::discriminate>:" << std::endl;
  //std::cout << " tau: Pt = " << thePFTauRef->pt() << ", eta = " << thePFTauRef->eta() << ", phi = " << thePFTauRef->phi() << std::endl;
  //std::cout << " mva = " << mva << ": workingPoint = " << workingPoint << std::endl;
  
  return ( returnMVA_ ? mva : workingPoint );
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstElectronMVA2);
