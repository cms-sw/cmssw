/* class PFRecoTauDiscriminationAgainstElectronMVA2
 * created : Apr 10 2012,
 * revised : ,
 * Authorss : Ivo Naranjo (LLR Ecole-Polytechnique)
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "RecoTauTag/RecoTau/interface/AntiElectronIDMVA2.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "PhysicsTools/MVAComputer/interface/zstream.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

#include <TMath.h>

#include <iostream>
#include <sstream>
#include <fstream>

using namespace reco;

class PFRecoTauDiscriminationAgainstElectronMVA2 : public PFTauDiscriminationProducerBase  
{
 public:
  explicit PFRecoTauDiscriminationAgainstElectronMVA2(const edm::ParameterSet& iConfig)
    : PFTauDiscriminationProducerBase(iConfig),
      moduleLabel_(iConfig.getParameter<std::string>("@module_label")),
      mva_(0),
      category_output_(0)
  {
    //std::cout << "<PFRecoTauDiscriminationAgainstElectronMVA2::PFRecoTauDiscriminationAgainstElectronMVA2>:" << std::endl;
    //std::cout << " moduleLabel = " << moduleLabel_ << std::endl;
 
    method_                                    = iConfig.getParameter<std::string>("method");
    inputFileName1prongNoEleMatchBL_           = iConfig.getParameter<edm::FileInPath>("inputFileName1prongNoEleMatchBL");
    inputFileName1prongBL_                     = iConfig.getParameter<edm::FileInPath>("inputFileName1prongBL");
    inputFileName1prongStripsWOgsfBL_          = iConfig.getParameter<edm::FileInPath>("inputFileName1prongStripsWOgsfBL");
    inputFileName1prongStripsWgsfWOpfEleMvaBL_ = iConfig.getParameter<edm::FileInPath>("inputFileName1prongStripsWgsfWOpfEleMvaBL");
    inputFileName1prongStripsWgsfWpfEleMvaBL_  = iConfig.getParameter<edm::FileInPath>("inputFileName1prongStripsWgsfWpfEleMvaBL");
    inputFileName1prongNoEleMatchEC_           = iConfig.getParameter<edm::FileInPath>("inputFileName1prongNoEleMatchEC");
    inputFileName1prongEC_                     = iConfig.getParameter<edm::FileInPath>("inputFileName1prongEC");
    inputFileName1prongStripsWOgsfEC_          = iConfig.getParameter<edm::FileInPath>("inputFileName1prongStripsWOgsfEC");
    inputFileName1prongStripsWgsfWOpfEleMvaEC_ = iConfig.getParameter<edm::FileInPath>("inputFileName1prongStripsWgsfWOpfEleMvaEC");
    inputFileName1prongStripsWgsfWpfEleMvaEC_  = iConfig.getParameter<edm::FileInPath>("inputFileName1prongStripsWgsfWpfEleMvaEC");

    returnMVA_                          = iConfig.getParameter<bool>("returnMVA");
    minMVA1prongNoEleMatchBL_           = iConfig.getParameter<double>("minMVA1prongNoEleMatchBL");
    minMVA1prongBL_                     = iConfig.getParameter<double>("minMVA1prongBL");
    minMVA1prongStripsWOgsfBL_          = iConfig.getParameter<double>("minMVA1prongStripsWOgsfBL");
    minMVA1prongStripsWgsfWOpfEleMvaBL_ = iConfig.getParameter<double>("minMVA1prongStripsWgsfWOpfEleMvaBL");
    minMVA1prongStripsWgsfWpfEleMvaBL_  = iConfig.getParameter<double>("minMVA1prongStripsWgsfWpfEleMvaBL");
    minMVA1prongNoEleMatchEC_           = iConfig.getParameter<double>("minMVA1prongNoEleMatchEC");
    minMVA1prongEC_                     = iConfig.getParameter<double>("minMVA1prongEC");
    minMVA1prongStripsWOgsfEC_          = iConfig.getParameter<double>("minMVA1prongStripsWOgsfEC");
    minMVA1prongStripsWgsfWOpfEleMvaEC_ = iConfig.getParameter<double>("minMVA1prongStripsWgsfWOpfEleMvaEC");
    minMVA1prongStripsWgsfWpfEleMvaEC_  = iConfig.getParameter<double>("minMVA1prongStripsWgsfWpfEleMvaEC");

    srcGsfElectrons_ = iConfig.getParameter<edm::InputTag>("srcGsfElectrons");

    mva_ = new AntiElectronIDMVA2();
    // CV: working version of file compression not implemented yet
    //mva_->Initialize_from_string(method_,
    //				   readZippedFile(inputFileName1prongNoEleMatchBL_.fullPath()),
    //				   readZippedFile(inputFileName1prongBL_.fullPath()),
    //				   readZippedFile(inputFileName1prongStripsWOgsfBL_.fullPath()),
    //				   readZippedFile(inputFileName1prongStripsWgsfWOpfEleMvaBL_.fullPath()),
    //				   readZippedFile(inputFileName1prongStripsWgsfWpfEleMvaBL_.fullPath()),
    //				   readZippedFile(inputFileName1prongNoEleMatchEC_.fullPath()),
    //				   readZippedFile(inputFileName1prongEC_.fullPath()),
    //				   readZippedFile(inputFileName1prongStripsWOgsfEC_.fullPath()),
    //				   readZippedFile(inputFileName1prongStripsWgsfWOpfEleMvaEC_.fullPath()),
    //				   readZippedFile(inputFileName1prongStripsWgsfWpfEleMvaEC_.fullPath()));
    mva_->Initialize_from_file(method_,
			       inputFileName1prongNoEleMatchBL_.fullPath(),
			       inputFileName1prongBL_.fullPath(),
			       inputFileName1prongStripsWOgsfBL_.fullPath(),
			       inputFileName1prongStripsWgsfWOpfEleMvaBL_.fullPath(),
			       inputFileName1prongStripsWgsfWpfEleMvaBL_.fullPath(),
			       inputFileName1prongNoEleMatchEC_.fullPath(),
			       inputFileName1prongEC_.fullPath(),
			       inputFileName1prongStripsWOgsfEC_.fullPath(),
			       inputFileName1prongStripsWgsfWOpfEleMvaEC_.fullPath(),
			       inputFileName1prongStripsWgsfWpfEleMvaEC_.fullPath());

    // add category index
    if ( returnMVA_ ) {
      produces<PFTauDiscriminator>("category");
    }
  }

  void beginEvent(const edm::Event&, const edm::EventSetup&);

  double discriminate(const PFTauRef&);

  void endEvent(edm::Event&);

  ~PFRecoTauDiscriminationAgainstElectronMVA2()
  {
    delete mva_;
  }

 private:

  std::string readZippedFile(const std::string& fileName)
  {
    //std::cout << "<PFRecoTauDiscriminationAgainstElectronMVA2::readZippedFile>:" << std::endl;
    //std::cout << " fileName = " << fileName << std::endl;
    // CV: code adapted from PhysicsTools/MVAComputer/src/MVAComputer.cc
    std::ifstream file;
    file.open(fileName.c_str());
    if ( !file.good() ) throw cms::Exception("InvalidFileState")
      << "Failed to open MVA file = " << fileName << " !!\n";
    std::ostringstream buffer_zipped;
    while ( file.good() ) {
      buffer_zipped << (char)file.get();
    }
    file.close();
    //std::cout << " buffer (zipped) = " << buffer_zipped.str() << std::endl;
    ext::izstream gunzip(&file);
    std::ostringstream buffer_unzipped;
    buffer_unzipped << gunzip.rdbuf();
    //std::cout << " buffer (unzipped) = " << buffer_unzipped.str() << std::endl;
    return buffer_unzipped.str();
  }

  std::string moduleLabel_;
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
  edm::Handle<TauCollection> taus_;
  std::auto_ptr<PFTauDiscriminator> category_output_;
  size_t tauIndex_;
};

void PFRecoTauDiscriminationAgainstElectronMVA2::beginEvent(const edm::Event& evt, const edm::EventSetup& es)
{
  if ( returnMVA_ ) {
    evt.getByLabel(TauProducer_, taus_);
    category_output_.reset(new PFTauDiscriminator(TauRefProd(taus_)));
    tauIndex_ = 0;
  }
  evt.getByLabel(srcGsfElectrons_, gsfElectrons_);
}

double PFRecoTauDiscriminationAgainstElectronMVA2::discriminate(const PFTauRef& thePFTauRef)
{
  double mva = 1.;
  double workingPoint = 1.;
  double category = -1.;
  bool isGsfElectronMatched = false;
  if( (*thePFTauRef).leadPFChargedHadrCand().isNonnull()) {
    for ( reco::GsfElectronCollection::const_iterator theGsfElectron = gsfElectrons_->begin();
	  theGsfElectron != gsfElectrons_->end(); ++theGsfElectron ) {
      if ( theGsfElectron->pt() > 10. ) { // CV: only take electrons above some minimal energy/Pt into account...
	double deltaREleTau = deltaR(theGsfElectron->p4(), thePFTauRef->p4());
	if ( deltaREleTau < 0.3 ) {
	  double mva_match = mva_->MVAValue(*thePFTauRef, *theGsfElectron);
          double workingPoint_match = 0.;

	  size_t numSignalPFGammaCands = thePFTauRef->signalPFGammaCands().size();
	  bool hasGsfTrack = thePFTauRef->leadPFChargedHadrCand()->gsfTrackRef().isNonnull();
	  bool isPFElectron = (theGsfElectron->mvaOutput().mva > -0.1);

	  if ( thePFTauRef->signalPFChargedHadrCands().size() == 1 ) {
	    double mvaCut = 999.;
	    if ( TMath::Abs(thePFTauRef->eta()) < 1.5 ) { // Barrel
	      if        ( numSignalPFGammaCands == 0                                  ) {
		category = 1.;
		mvaCut = minMVA1prongBL_;
	      } else if ( numSignalPFGammaCands >= 1 && !hasGsfTrack                  ) {
		category = 2.;
		mvaCut = minMVA1prongStripsWOgsfBL_;
	      } else if ( numSignalPFGammaCands >= 1 &&  hasGsfTrack && !isPFElectron ) {
		category = 3.;
		mvaCut = minMVA1prongStripsWgsfWOpfEleMvaBL_;
	      } else if ( numSignalPFGammaCands >= 1 &&  hasGsfTrack &&  isPFElectron ) {
		category = 4.;
		mvaCut = minMVA1prongStripsWgsfWpfEleMvaBL_;
	      }
	    } else { // Endcap
	      if        ( numSignalPFGammaCands == 0                                  ) {
		category = 6.;
		mvaCut = minMVA1prongEC_;
	      } else if ( numSignalPFGammaCands >= 1 && !hasGsfTrack                  ) {
		category = 7.;
		mvaCut = minMVA1prongStripsWOgsfEC_;
	      } else if ( numSignalPFGammaCands >= 1 &&  hasGsfTrack && !isPFElectron ) {
		category = 8.;
		mvaCut = minMVA1prongStripsWgsfWOpfEleMvaEC_;
	      } else if ( numSignalPFGammaCands >= 1 &&  hasGsfTrack &&  isPFElectron ) {
		category = 9.;
		mvaCut = minMVA1prongStripsWgsfWpfEleMvaEC_;
	      }
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
    mva = mva_->MVAValue(*thePFTauRef);
    double mvaCut = 999.;
    if ( TMath::Abs(thePFTauRef->eta()) < 1.5 ) { // Barrel
      category = 0.;
      mvaCut = minMVA1prongNoEleMatchBL_;
    } else { // Endcap
      category = 5.;
      mvaCut = minMVA1prongNoEleMatchEC_;
    }
    workingPoint = (mva > mvaCut);
  }

  //std::cout << "<PFRecoTauDiscriminationAgainstElectronMVA2::discriminate>:" << std::endl;
  //std::cout << " tau: Pt = " << thePFTauRef->pt() << ", eta = " << thePFTauRef->eta() << ", phi = " << thePFTauRef->phi() << std::endl;
  //std::cout << " mva = " << mva << ": workingPoint = " << workingPoint << std::endl;

  if ( returnMVA_ ) {
    // add category index
    category_output_->setValue(tauIndex_, category);
    ++tauIndex_;
    // return MVA output value
    return mva;
  } else {
    return workingPoint;
  }
}

void PFRecoTauDiscriminationAgainstElectronMVA2::endEvent(edm::Event& evt)
{
  // add all category indices to event
  if ( returnMVA_ ) {
    evt.put(category_output_, "category");
  }
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstElectronMVA2);
