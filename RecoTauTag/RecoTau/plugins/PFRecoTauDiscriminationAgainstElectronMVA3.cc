/* class PFRecoTauDiscriminationAgainstElectronMVA3
 * created : Oct 18 2012,
 * revised : ,
 * Authorss : Ivo Naranjo (LLR Ecole-Polytechnique)
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "RecoTauTag/RecoTau/interface/AntiElectronIDMVA3.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "PhysicsTools/MVAComputer/interface/zstream.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

#include <TMath.h>

#include <iostream>
#include <sstream>
#include <fstream>

using namespace reco;

class PFRecoTauDiscriminationAgainstElectronMVA3 : public PFTauDiscriminationProducerBase  {
public:
  explicit PFRecoTauDiscriminationAgainstElectronMVA3(const edm::ParameterSet& iConfig)
    : PFTauDiscriminationProducerBase(iConfig),
      mva_(0),
      category_output_(0)
  {
    method_                                    = iConfig.getParameter<std::string>("method");
    inputFileName1prongNoEleMatchWOgWOgsfBL_   = iConfig.getParameter<edm::FileInPath>("inputFileName1prongNoEleMatchWOgWOgsfBL");
    inputFileName1prongNoEleMatchWOgWgsfBL_    = iConfig.getParameter<edm::FileInPath>("inputFileName1prongNoEleMatchWOgWgsfBL");
    inputFileName1prongNoEleMatchWgWOgsfBL_    = iConfig.getParameter<edm::FileInPath>("inputFileName1prongNoEleMatchWgWOgsfBL");
    inputFileName1prongNoEleMatchWgWgsfBL_     = iConfig.getParameter<edm::FileInPath>("inputFileName1prongNoEleMatchWgWgsfBL");
    inputFileName1prongWOgWOgsfBL_             = iConfig.getParameter<edm::FileInPath>("inputFileName1prongWOgWOgsfBL");
    inputFileName1prongWOgWgsfBL_              = iConfig.getParameter<edm::FileInPath>("inputFileName1prongWOgWgsfBL");
    inputFileName1prongWgWOgsfBL_              = iConfig.getParameter<edm::FileInPath>("inputFileName1prongWgWOgsfBL");
    inputFileName1prongWgWgsfBL_               = iConfig.getParameter<edm::FileInPath>("inputFileName1prongWgWgsfBL");
    inputFileName1prongNoEleMatchWOgWOgsfEC_   = iConfig.getParameter<edm::FileInPath>("inputFileName1prongNoEleMatchWOgWOgsfEC");
    inputFileName1prongNoEleMatchWOgWgsfEC_    = iConfig.getParameter<edm::FileInPath>("inputFileName1prongNoEleMatchWOgWgsfEC");
    inputFileName1prongNoEleMatchWgWOgsfEC_    = iConfig.getParameter<edm::FileInPath>("inputFileName1prongNoEleMatchWgWOgsfEC");
    inputFileName1prongNoEleMatchWgWgsfEC_     = iConfig.getParameter<edm::FileInPath>("inputFileName1prongNoEleMatchWgWgsfEC");
    inputFileName1prongWOgWOgsfEC_             = iConfig.getParameter<edm::FileInPath>("inputFileName1prongWOgWOgsfEC");
    inputFileName1prongWOgWgsfEC_              = iConfig.getParameter<edm::FileInPath>("inputFileName1prongWOgWgsfEC");
    inputFileName1prongWgWOgsfEC_              = iConfig.getParameter<edm::FileInPath>("inputFileName1prongWgWOgsfEC");
    inputFileName1prongWgWgsfEC_               = iConfig.getParameter<edm::FileInPath>("inputFileName1prongWgWgsfEC");

    returnMVA_                          = iConfig.getParameter<bool>("returnMVA");
    minMVA1prongNoEleMatchWOgWOgsfBL_   = iConfig.getParameter<double>("minMVA1prongNoEleMatchWOgWOgsfBL");
    minMVA1prongNoEleMatchWOgWgsfBL_    = iConfig.getParameter<double>("minMVA1prongNoEleMatchWOgWgsfBL");
    minMVA1prongNoEleMatchWgWOgsfBL_    = iConfig.getParameter<double>("minMVA1prongNoEleMatchWgWOgsfBL");
    minMVA1prongNoEleMatchWgWgsfBL_     = iConfig.getParameter<double>("minMVA1prongNoEleMatchWgWgsfBL");
    minMVA1prongWOgWOgsfBL_             = iConfig.getParameter<double>("minMVA1prongWOgWOgsfBL");
    minMVA1prongWOgWgsfBL_              = iConfig.getParameter<double>("minMVA1prongWOgWgsfBL");
    minMVA1prongWgWOgsfBL_              = iConfig.getParameter<double>("minMVA1prongWgWOgsfBL");
    minMVA1prongWgWgsfBL_               = iConfig.getParameter<double>("minMVA1prongWgWgsfBL");
    minMVA1prongNoEleMatchWOgWOgsfEC_   = iConfig.getParameter<double>("minMVA1prongNoEleMatchWOgWOgsfEC");
    minMVA1prongNoEleMatchWOgWgsfEC_    = iConfig.getParameter<double>("minMVA1prongNoEleMatchWOgWgsfEC");
    minMVA1prongNoEleMatchWgWOgsfEC_    = iConfig.getParameter<double>("minMVA1prongNoEleMatchWgWOgsfEC");
    minMVA1prongNoEleMatchWgWgsfEC_     = iConfig.getParameter<double>("minMVA1prongNoEleMatchWgWgsfEC");
    minMVA1prongWOgWOgsfEC_             = iConfig.getParameter<double>("minMVA1prongWOgWOgsfEC");
    minMVA1prongWOgWgsfEC_              = iConfig.getParameter<double>("minMVA1prongWOgWgsfEC");
    minMVA1prongWgWOgsfEC_              = iConfig.getParameter<double>("minMVA1prongWgWOgsfEC");
    minMVA1prongWgWgsfEC_               = iConfig.getParameter<double>("minMVA1prongWgWgsfEC");
    minMVA3prongMatch_                  = iConfig.getParameter<double>("minMVA3prongMatch");
    minMVA3prongNoMatch_                = iConfig.getParameter<double>("minMVA3prongNoMatch");
  
    srcGsfElectrons_ = iConfig.getParameter<edm::InputTag>("srcGsfElectrons");

    mva_ = new AntiElectronIDMVA3();
    // CV: working version of file compression not implemented yet
//     mva_->Initialize_from_string(method_,
// 				 readZippedFile(inputFileName1prongNoEleMatchWOgWOgsfBL_.fullPath()),
// 				 readZippedFile(inputFileName1prongNoEleMatchWOgWgsfBL_.fullPath()),
// 				 readZippedFile(inputFileName1prongNoEleMatchWgWOgsfBL_.fullPath()),
// 				 readZippedFile(inputFileName1prongNoEleMatchWgWgsfBL_.fullPath()),
// 				 readZippedFile(inputFileName1prongWOgWOgsfBL_.fullPath()),
// 				 readZippedFile(inputFileName1prongWOgWgsfBL_.fullPath()),
// 				 readZippedFile(inputFileName1prongWgWOgsfBL_.fullPath()),
// 				 readZippedFile(inputFileName1prongWgWgsfBL_.fullPath()),
// 				 readZippedFile(inputFileName1prongNoEleMatchWOgWOgsfEC_.fullPath()),
// 				 readZippedFile(inputFileName1prongNoEleMatchWOgWgsfEC_.fullPath()),
// 				 readZippedFile(inputFileName1prongNoEleMatchWgWOgsfEC_.fullPath()),
// 				 readZippedFile(inputFileName1prongNoEleMatchWgWgsfEC_.fullPath()),
// 				 readZippedFile(inputFileName1prongWOgWOgsfEC_.fullPath()),
// 				 readZippedFile(inputFileName1prongWOgWgsfEC_.fullPath()),
// 				 readZippedFile(inputFileName1prongWgWOgsfEC_.fullPath()),
// 				 readZippedFile(inputFileName1prongWgWgsfEC_.fullPath()));
    mva_->Initialize_from_file(method_,
			       inputFileName1prongNoEleMatchWOgWOgsfBL_.fullPath(),
			       inputFileName1prongNoEleMatchWOgWgsfBL_.fullPath(),
			       inputFileName1prongNoEleMatchWgWOgsfBL_.fullPath(),
			       inputFileName1prongNoEleMatchWgWgsfBL_.fullPath(),
			       inputFileName1prongWOgWOgsfBL_.fullPath(),
			       inputFileName1prongWOgWgsfBL_.fullPath(),
			       inputFileName1prongWgWOgsfBL_.fullPath(),
			       inputFileName1prongWgWgsfBL_.fullPath(),
			       inputFileName1prongNoEleMatchWOgWOgsfEC_.fullPath(),
			       inputFileName1prongNoEleMatchWOgWgsfEC_.fullPath(),
			       inputFileName1prongNoEleMatchWgWOgsfEC_.fullPath(),
			       inputFileName1prongNoEleMatchWgWgsfEC_.fullPath(),
			       inputFileName1prongWOgWOgsfEC_.fullPath(),
			       inputFileName1prongWOgWgsfEC_.fullPath(),
			       inputFileName1prongWgWOgsfEC_.fullPath(),
			       inputFileName1prongWgWgsfEC_.fullPath());

    // add category index
    if ( returnMVA_ ) {
      produces<PFTauDiscriminator>("category");
    }
  }

  void beginEvent(const edm::Event&, const edm::EventSetup&);

  double discriminate(const PFTauRef&);

  void endEvent(edm::Event&);

  ~PFRecoTauDiscriminationAgainstElectronMVA3()
  {
    delete mva_;
  }

private:
  bool isInEcalCrack(double) const;
  std::string readZippedFile(const std::string& fileName)
  {
    //std::cout << "<PFRecoTauDiscriminationAgainstElectronMVA3::readZippedFile>:" << std::endl;
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

  std::string method_ ;
  edm::FileInPath inputFileName1prongNoEleMatchWOgWOgsfBL_;
  edm::FileInPath inputFileName1prongNoEleMatchWOgWgsfBL_;
  edm::FileInPath inputFileName1prongNoEleMatchWgWOgsfBL_;
  edm::FileInPath inputFileName1prongNoEleMatchWgWgsfBL_;
  edm::FileInPath inputFileName1prongWOgWOgsfBL_;
  edm::FileInPath inputFileName1prongWOgWgsfBL_;
  edm::FileInPath inputFileName1prongWgWOgsfBL_;
  edm::FileInPath inputFileName1prongWgWgsfBL_;
  edm::FileInPath inputFileName1prongNoEleMatchWOgWOgsfEC_;
  edm::FileInPath inputFileName1prongNoEleMatchWOgWgsfEC_;
  edm::FileInPath inputFileName1prongNoEleMatchWgWOgsfEC_;
  edm::FileInPath inputFileName1prongNoEleMatchWgWgsfEC_;
  edm::FileInPath inputFileName1prongWOgWOgsfEC_;
  edm::FileInPath inputFileName1prongWOgWgsfEC_;
  edm::FileInPath inputFileName1prongWgWOgsfEC_;
  edm::FileInPath inputFileName1prongWgWgsfEC_;
  AntiElectronIDMVA3* mva_;
  bool returnMVA_ ;
  double minMVA1prongNoEleMatchWOgWOgsfBL_ ;
  double minMVA1prongNoEleMatchWOgWgsfBL_ ;
  double minMVA1prongNoEleMatchWgWOgsfBL_ ;
  double minMVA1prongNoEleMatchWgWgsfBL_ ;
  double minMVA1prongWOgWOgsfBL_ ;
  double minMVA1prongWOgWgsfBL_ ;
  double minMVA1prongWgWOgsfBL_ ;
  double minMVA1prongWgWgsfBL_ ;
  double minMVA1prongNoEleMatchWOgWOgsfEC_ ;
  double minMVA1prongNoEleMatchWOgWgsfEC_ ;
  double minMVA1prongNoEleMatchWgWOgsfEC_ ;
  double minMVA1prongNoEleMatchWgWgsfEC_ ;
  double minMVA1prongWOgWOgsfEC_ ;
  double minMVA1prongWOgWgsfEC_ ;
  double minMVA1prongWgWOgsfEC_ ;
  double minMVA1prongWgWgsfEC_ ;
  double minMVA3prongMatch_ ;
  double minMVA3prongNoMatch_ ;
  edm::InputTag srcGsfElectrons_;
  edm::Handle<reco::GsfElectronCollection> gsfElectrons_;
  edm::Handle<TauCollection> taus_;
  std::auto_ptr<PFTauDiscriminator> category_output_;
  size_t tauIndex_;
};

void PFRecoTauDiscriminationAgainstElectronMVA3::beginEvent(const edm::Event& evt, const edm::EventSetup& es)
{
  if ( returnMVA_ ) {
    evt.getByLabel(TauProducer_, taus_);
    category_output_.reset(new PFTauDiscriminator(TauRefProd(taus_)));
    tauIndex_ = 0;
  }
  evt.getByLabel(srcGsfElectrons_, gsfElectrons_);
}

double PFRecoTauDiscriminationAgainstElectronMVA3::discriminate(const PFTauRef& thePFTauRef)
{
  double mva = 1.;
  double workingPoint = 1.;
  double category = -1.;
  bool isGsfElectronMatched = false;

  float deltaRTestigo = 9.9;
  float mvaCutTestigo = 999;

  if( (*thePFTauRef).leadPFChargedHadrCand().isNonnull()) {
    for ( reco::GsfElectronCollection::const_iterator theGsfElectron = gsfElectrons_->begin();
	  theGsfElectron != gsfElectrons_->end(); ++theGsfElectron ) {
      if ( theGsfElectron->pt() > 10. ) { // CV: only take electrons above some minimal energy/Pt into account...
	double deltaREleTau = deltaR(theGsfElectron->p4(), thePFTauRef->p4());
	deltaRTestigo = deltaREleTau;
	if ( deltaREleTau < 0.3 ) {
	  double mva_match = mva_->MVAValue(*thePFTauRef, *theGsfElectron);
	  double workingPoint_match = 0.;
	  size_t numSignalPFGammaCands = thePFTauRef->signalPFGammaCands().size();
	  bool hasGsfTrack = thePFTauRef->leadPFChargedHadrCand()->gsfTrackRef().isNonnull();
  
	  if ( thePFTauRef->signalPFChargedHadrCands().size() == 1 ) {
	    //// Veto taus that go to Ecal crack
	    Float_t TauEtaAtEcalEntrance = -99.;
	    float sumEtaTimesEnergy = 0;
	    float sumEnergy = 0;
	    for(unsigned int j = 0 ; j < ((*thePFTauRef).signalPFCands()).size() ; j++){
	      reco::PFCandidateRef pfcandidate = ((*thePFTauRef).signalPFCands()).at(j);
	      sumEtaTimesEnergy += pfcandidate->positionAtECALEntrance().eta()*pfcandidate->energy();
	      sumEnergy += pfcandidate->energy();
	    }
	    if(sumEnergy>0)TauEtaAtEcalEntrance = sumEtaTimesEnergy/sumEnergy;
	    
	    if (isInEcalCrack(TauEtaAtEcalEntrance)){
	      if ( returnMVA_ ) {
		// add category index
		category_output_->setValue(tauIndex_, category);
		++tauIndex_;
		// return MVA output value
		return -99;
	      } else {
		//return Workingpoint 0
		return 0;
	      }
	    }
	    //// Veto taus that go to Ecal crack

	    double mvaCut = 999.;
	    if ( TMath::Abs(thePFTauRef->eta()) < 1.5 ) { // Barrel
	      if        ( numSignalPFGammaCands == 0 && !hasGsfTrack ) {
		category = 4.;
		mvaCut = minMVA1prongWOgWOgsfBL_;
	      } else if ( numSignalPFGammaCands == 0 &&  hasGsfTrack ) {
		category = 5.;
		mvaCut = minMVA1prongWOgWgsfBL_;
	      } else if ( numSignalPFGammaCands >= 1 && !hasGsfTrack ) {
		category = 6.;
		mvaCut = minMVA1prongWgWOgsfBL_;
	      } else if ( numSignalPFGammaCands >= 1 &&  hasGsfTrack ) {
		category = 7.;
		mvaCut = minMVA1prongWgWgsfBL_;
	      }
	    } else { // Endcap
	      if       ( numSignalPFGammaCands == 0 && !hasGsfTrack ) {
		category = 12.;
		mvaCut = minMVA1prongWOgWOgsfEC_;
	      } else if ( numSignalPFGammaCands == 0 &&  hasGsfTrack ) {
		category = 13.;
		mvaCut = minMVA1prongWOgWgsfEC_;
	      } else if ( numSignalPFGammaCands >= 1 && !hasGsfTrack ) {
		category = 14.;
		mvaCut = minMVA1prongWgWOgsfEC_;
	      } else if ( numSignalPFGammaCands >= 1 &&  hasGsfTrack ) {
		category = 15.;
		mvaCut = minMVA1prongWgWgsfEC_;
	      }
	    }
	    workingPoint_match = (mva_match > mvaCut);
	    mvaCutTestigo = mvaCut;
	  } else {
	    category = 16.;
	    workingPoint_match = (mva_match > minMVA3prongMatch_);
	  }
	  mva = TMath::Min(mva, mva_match);
	  workingPoint = TMath::Min(workingPoint, workingPoint_match);
	  isGsfElectronMatched = true;
	}//deltaR<0.3
      }//electron pt>10
    }//loop electrons

    if ( !isGsfElectronMatched ) {
      mva = mva_->MVAValue(*thePFTauRef);
      size_t numSignalPFGammaCands = thePFTauRef->signalPFGammaCands().size();
      bool hasGsfTrack = thePFTauRef->leadPFChargedHadrCand()->gsfTrackRef().isNonnull();
      if ( thePFTauRef->signalPFChargedHadrCands().size() == 1 ) {
	//// Veto taus that go to Ecal crack
	Float_t TauEtaAtEcalEntrance = -99.;
	float sumEtaTimesEnergy = 0;
	float sumEnergy = 0;
	for(unsigned int j = 0 ; j < ((*thePFTauRef).signalPFCands()).size() ; j++){
	  reco::PFCandidateRef pfcandidate = ((*thePFTauRef).signalPFCands()).at(j);
	  sumEtaTimesEnergy += pfcandidate->positionAtECALEntrance().eta()*pfcandidate->energy();
	  sumEnergy += pfcandidate->energy();
	}
	if(sumEnergy>0)TauEtaAtEcalEntrance = sumEtaTimesEnergy/sumEnergy;
	
	if (isInEcalCrack(TauEtaAtEcalEntrance)){
	  if ( returnMVA_ ) {
	    // add category index
	    category_output_->setValue(tauIndex_, category);
	    ++tauIndex_;
	    // return MVA output value
	    return -99;
	  } else {
	    //return Workingpoint 0
	    return 0;
	  }
	}
	//// Veto taus that go to Ecal crack

	double mvaCut = 999.;
	if ( TMath::Abs(thePFTauRef->eta()) < 1.5 ) { // Barrel
	  if        ( numSignalPFGammaCands == 0 && !hasGsfTrack ) {
	    category = 0.;
	    mvaCut = minMVA1prongNoEleMatchWOgWOgsfBL_;
	  } else if ( numSignalPFGammaCands == 0 &&  hasGsfTrack ) {
	    category = 1.;
	    mvaCut = minMVA1prongNoEleMatchWOgWgsfBL_;
	  } else if ( numSignalPFGammaCands >= 1 && !hasGsfTrack ) {
	    category = 2.;
	    mvaCut = minMVA1prongNoEleMatchWgWOgsfBL_;
	  } else if ( numSignalPFGammaCands >= 1 &&  hasGsfTrack ) {
	    category = 3.;
	    mvaCut = minMVA1prongNoEleMatchWgWgsfBL_;
	  }
	} else { // Endcap
	  if        ( numSignalPFGammaCands == 0 && !hasGsfTrack ) {
	    category = 8.;
	    mvaCut = minMVA1prongNoEleMatchWOgWOgsfEC_;
	  } else if ( numSignalPFGammaCands == 0 &&  hasGsfTrack ) {
	    category = 9.;
	    mvaCut = minMVA1prongNoEleMatchWOgWgsfEC_;
	  } else if ( numSignalPFGammaCands >= 1 && !hasGsfTrack ) {
	    category = 10.;
	    mvaCut = minMVA1prongNoEleMatchWgWOgsfEC_;
	  } else if ( numSignalPFGammaCands >= 1 &&  hasGsfTrack ) {
	    category = 11.;
	    mvaCut = minMVA1prongNoEleMatchWgWgsfEC_;
	  }
	}
	workingPoint = (mva > mvaCut);
	mvaCutTestigo = mvaCut;
      } else {
	category = 17.; 
	workingPoint = (mva > minMVA3prongNoMatch_);
      }
    }
  }

//   std::cout<<" Taus : "<<TauProducer_<<std::endl;
//   std::cout << "<PFRecoTauDiscriminationAgainstElectronMVA3::discriminate>:" << std::endl;
//   std::cout << " tau: Pt = " << thePFTauRef->pt() << ", eta = " << thePFTauRef->eta() << ", phi = " << thePFTauRef->phi() << std::endl;
//   std::cout << " mva = " << mva <<  " mvaCut = " << mvaCutTestigo <<" isGsfElectronMatched = "<<isGsfElectronMatched<< std::endl;
//   std::cout << " category = " << category << " : workingPoint = " << workingPoint << std::endl;
//   std::cout << " deltaREleTau = " << deltaRTestigo << std::endl;
//   std::cout << " charged hadron in tau : "<<(*thePFTauRef).leadPFChargedHadrCand().isNonnull()<< std::endl;
//   std::cout << " Prongs in tau : "<<thePFTauRef->signalPFChargedHadrCands().size()<< std::endl;

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

void PFRecoTauDiscriminationAgainstElectronMVA3::endEvent(edm::Event& evt)
{
  // add all category indices to event
  if ( returnMVA_ ) {
    evt.put(category_output_, "category");
  }
}

bool
PFRecoTauDiscriminationAgainstElectronMVA3::isInEcalCrack(double eta) const
{
  eta = fabs(eta);
  return (eta>1.460 && eta<1.558);
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstElectronMVA3);
