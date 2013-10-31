/* class PFRecoTauDiscriminationAgainstElectronMVA4
 * created : Aug 13 2013,
 * revised : ,
 * Authorss : Ivo Naranjo (LLR Ecole-Polytechnique)
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "RecoTauTag/RecoTau/interface/AntiElectronIDMVA4GBR.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "PhysicsTools/MVAComputer/interface/zstream.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

#include <TMath.h>

#include <iostream>
#include <sstream>
#include <fstream>

using namespace reco;

class PFRecoTauDiscriminationAgainstElectronMVA4GBR : public PFTauDiscriminationProducerBase  {
public:
  explicit PFRecoTauDiscriminationAgainstElectronMVA4GBR(const edm::ParameterSet& iConfig)
    : PFTauDiscriminationProducerBase(iConfig),
      mva_(0),
      category_output_(0)
  {
    method_                                    = iConfig.getParameter<std::string>("method");
    gbrFile_                                   = iConfig.getParameter<edm::FileInPath>("gbrFile");
    if ( !gbrFile_.isLocal()) throw cms::Exception("PFRecoTauDiscriminationAgainstElectronMVA4GBR") 
      << " Failed to find File = " << gbrFile_ << " !!\n";

    returnMVA_                          = iConfig.getParameter<bool>("returnMVA");
    minMVANoEleMatchWOgWOgsfBL_   = iConfig.getParameter<double>("minMVANoEleMatchWOgWOgsfBL");
    minMVANoEleMatchWOgWgsfBL_    = iConfig.getParameter<double>("minMVANoEleMatchWOgWgsfBL");
    minMVANoEleMatchWgWOgsfBL_    = iConfig.getParameter<double>("minMVANoEleMatchWgWOgsfBL");
    minMVANoEleMatchWgWgsfBL_     = iConfig.getParameter<double>("minMVANoEleMatchWgWgsfBL");
    minMVAWOgWOgsfBL_             = iConfig.getParameter<double>("minMVAWOgWOgsfBL");
    minMVAWOgWgsfBL_              = iConfig.getParameter<double>("minMVAWOgWgsfBL");
    minMVAWgWOgsfBL_              = iConfig.getParameter<double>("minMVAWgWOgsfBL");
    minMVAWgWgsfBL_               = iConfig.getParameter<double>("minMVAWgWgsfBL");
    minMVANoEleMatchWOgWOgsfEC_   = iConfig.getParameter<double>("minMVANoEleMatchWOgWOgsfEC");
    minMVANoEleMatchWOgWgsfEC_    = iConfig.getParameter<double>("minMVANoEleMatchWOgWgsfEC");
    minMVANoEleMatchWgWOgsfEC_    = iConfig.getParameter<double>("minMVANoEleMatchWgWOgsfEC");
    minMVANoEleMatchWgWgsfEC_     = iConfig.getParameter<double>("minMVANoEleMatchWgWgsfEC");
    minMVAWOgWOgsfEC_             = iConfig.getParameter<double>("minMVAWOgWOgsfEC");
    minMVAWOgWgsfEC_              = iConfig.getParameter<double>("minMVAWOgWgsfEC");
    minMVAWgWOgsfEC_              = iConfig.getParameter<double>("minMVAWgWOgsfEC");
    minMVAWgWgsfEC_               = iConfig.getParameter<double>("minMVAWgWgsfEC");

    srcGsfElectrons_ = iConfig.getParameter<edm::InputTag>("srcGsfElectrons");

    mva_ = new AntiElectronIDMVA4GBR();
    // CV: working version of file compression not implemented yet
//     mva_->Initialize_from_string(method_,
// 				 readZippedFile(inputFileNameNoEleMatchWOgWOgsfBL_.fullPath()),
// 				 readZippedFile(inputFileNameNoEleMatchWOgWgsfBL_.fullPath()),
// 				 readZippedFile(inputFileNameNoEleMatchWgWOgsfBL_.fullPath()),
// 				 readZippedFile(inputFileNameNoEleMatchWgWgsfBL_.fullPath()),
// 				 readZippedFile(inputFileNameWOgWOgsfBL_.fullPath()),
// 				 readZippedFile(inputFileNameWOgWgsfBL_.fullPath()),
// 				 readZippedFile(inputFileNameWgWOgsfBL_.fullPath()),
// 				 readZippedFile(inputFileNameWgWgsfBL_.fullPath()),
// 				 readZippedFile(inputFileNameNoEleMatchWOgWOgsfEC_.fullPath()),
// 				 readZippedFile(inputFileNameNoEleMatchWOgWgsfEC_.fullPath()),
// 				 readZippedFile(inputFileNameNoEleMatchWgWOgsfEC_.fullPath()),
// 				 readZippedFile(inputFileNameNoEleMatchWgWgsfEC_.fullPath()),
// 				 readZippedFile(inputFileNameWOgWOgsfEC_.fullPath()),
// 				 readZippedFile(inputFileNameWOgWgsfEC_.fullPath()),
// 				 readZippedFile(inputFileNameWgWOgsfEC_.fullPath()),
// 				 readZippedFile(inputFileNameWgWgsfEC_.fullPath()));
    mva_->Initialize_from_file(method_,
			       gbrFile_.fullPath().data()
			       );

    // add category index
    if ( returnMVA_ ) {
      produces<PFTauDiscriminator>("category");
    }
  }

  void beginEvent(const edm::Event&, const edm::EventSetup&);

  double discriminate(const PFTauRef&);

  void endEvent(edm::Event&);

  ~PFRecoTauDiscriminationAgainstElectronMVA4GBR()
  {
    delete mva_;
  }

private:
  bool isInEcalCrack(double) const;
  std::string readZippedFile(const std::string& fileName)
  {
    //std::cout << "<PFRecoTauDiscriminationAgainstElectronMVA4GBR::readZippedFile>:" << std::endl;
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
  edm::FileInPath gbrFile_;
  AntiElectronIDMVA4GBR* mva_;
  bool returnMVA_ ;
  double minMVANoEleMatchWOgWOgsfBL_ ;
  double minMVANoEleMatchWOgWgsfBL_ ;
  double minMVANoEleMatchWgWOgsfBL_ ;
  double minMVANoEleMatchWgWgsfBL_ ;
  double minMVAWOgWOgsfBL_ ;
  double minMVAWOgWgsfBL_ ;
  double minMVAWgWOgsfBL_ ;
  double minMVAWgWgsfBL_ ;
  double minMVANoEleMatchWOgWOgsfEC_ ;
  double minMVANoEleMatchWOgWgsfEC_ ;
  double minMVANoEleMatchWgWOgsfEC_ ;
  double minMVANoEleMatchWgWgsfEC_ ;
  double minMVAWOgWOgsfEC_ ;
  double minMVAWOgWgsfEC_ ;
  double minMVAWgWOgsfEC_ ;
  double minMVAWgWgsfEC_ ;
  double minMVA3prongMatch_ ;
  double minMVA3prongNoMatch_ ;
  edm::InputTag srcGsfElectrons_;
  edm::Handle<reco::GsfElectronCollection> gsfElectrons_;
  edm::Handle<TauCollection> taus_;
  std::auto_ptr<PFTauDiscriminator> category_output_;
  size_t tauIndex_;
};

void PFRecoTauDiscriminationAgainstElectronMVA4GBR::beginEvent(const edm::Event& evt, const edm::EventSetup& es)
{
  if ( returnMVA_ ) {
    evt.getByLabel(TauProducer_, taus_);
    category_output_.reset(new PFTauDiscriminator(TauRefProd(taus_)));
    tauIndex_ = 0;
  }
  evt.getByLabel(srcGsfElectrons_, gsfElectrons_);
}

double PFRecoTauDiscriminationAgainstElectronMVA4GBR::discriminate(const PFTauRef& thePFTauRef)
{
  double mva = 1.;
  double workingPoint = 1.;
  double category = -1.;
  bool isGsfElectronMatched = false;

  // float deltaRDummy = 9.9;
  //  float mvaCutDummy = 999;

  if( (*thePFTauRef).leadPFChargedHadrCand().isNonnull()) {
    for ( reco::GsfElectronCollection::const_iterator theGsfElectron = gsfElectrons_->begin();
	  theGsfElectron != gsfElectrons_->end(); ++theGsfElectron ) {
      if ( theGsfElectron->pt() > 10. ) { // CV: only take electrons above some minimal energy/Pt into account...
	double deltaREleTau = deltaR(theGsfElectron->p4(), thePFTauRef->p4());
	//	deltaRDummy = deltaREleTau;
	if ( deltaREleTau < 0.3 ) {
	  double mva_match = mva_->MVAValue(*thePFTauRef, *theGsfElectron);
	  double workingPoint_match = 0.;
	  size_t numSignalPFGammaCands = thePFTauRef->signalPFGammaCands().size();
	  bool hasGsfTrack = thePFTauRef->leadPFChargedHadrCand()->gsfTrackRef().isNonnull();
  
	  Float_t TauEtaAtEcalEntrance = -99.;
	  float sumEtaTimesEnergy = 0;
	  float sumEnergy = 0;
	  for(unsigned int j = 0 ; j < ((*thePFTauRef).signalPFCands()).size() ; j++){
	    reco::PFCandidatePtr pfcandidate = ((*thePFTauRef).signalPFCands()).at(j);
	    sumEtaTimesEnergy += pfcandidate->positionAtECALEntrance().eta()*pfcandidate->energy();
	    sumEnergy += pfcandidate->energy();
	  }
	  if(sumEnergy>0)TauEtaAtEcalEntrance = sumEtaTimesEnergy/sumEnergy;
	    
	  //// Veto taus that go to Ecal crack
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
	  if ( TMath::Abs(TauEtaAtEcalEntrance) < 1.479 ) { // Barrel
	    if        ( numSignalPFGammaCands == 0 && !hasGsfTrack ) {
	      category = 4.;
	      mvaCut = minMVAWOgWOgsfBL_;
	    } else if ( numSignalPFGammaCands == 0 &&  hasGsfTrack ) {
	      category = 5.;
	      mvaCut = minMVAWOgWgsfBL_;
	    } else if ( numSignalPFGammaCands >= 1 && !hasGsfTrack ) {
	      category = 6.;
	      mvaCut = minMVAWgWOgsfBL_;
	    } else if ( numSignalPFGammaCands >= 1 &&  hasGsfTrack ) {
	      category = 7.;
	      mvaCut = minMVAWgWgsfBL_;
	    }
	  } else { // Endcap
	    if       ( numSignalPFGammaCands == 0 && !hasGsfTrack ) {
	      category = 12.;
	      mvaCut = minMVAWOgWOgsfEC_;
	    } else if ( numSignalPFGammaCands == 0 &&  hasGsfTrack ) {
	      category = 13.;
	      mvaCut = minMVAWOgWgsfEC_;
	    } else if ( numSignalPFGammaCands >= 1 && !hasGsfTrack ) {
	      category = 14.;
	      mvaCut = minMVAWgWOgsfEC_;
	    } else if ( numSignalPFGammaCands >= 1 &&  hasGsfTrack ) {
	      category = 15.;
	      mvaCut = minMVAWgWgsfEC_;
	    }
	  }
	  workingPoint_match = (mva_match > mvaCut);
	  // mvaCutDummy = mvaCut;

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
      Float_t TauEtaAtEcalEntrance = -99.;
      float sumEtaTimesEnergy = 0;
      float sumEnergy = 0;
      for(unsigned int j = 0 ; j < ((*thePFTauRef).signalPFCands()).size() ; j++){
	reco::PFCandidatePtr pfcandidate = ((*thePFTauRef).signalPFCands()).at(j);
	sumEtaTimesEnergy += pfcandidate->positionAtECALEntrance().eta()*pfcandidate->energy();
	sumEnergy += pfcandidate->energy();
      }
      if(sumEnergy>0)TauEtaAtEcalEntrance = sumEtaTimesEnergy/sumEnergy;
      
      //// Veto taus that go to Ecal crack
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
      if ( TMath::Abs(TauEtaAtEcalEntrance) < 1.479 ) { // Barrel
	if        ( numSignalPFGammaCands == 0 && !hasGsfTrack ) {
	  category = 0.;
	  mvaCut = minMVANoEleMatchWOgWOgsfBL_;
	} else if ( numSignalPFGammaCands == 0 &&  hasGsfTrack ) {
	  category = 1.;
	  mvaCut = minMVANoEleMatchWOgWgsfBL_;
	} else if ( numSignalPFGammaCands >= 1 && !hasGsfTrack ) {
	  category = 2.;
	  mvaCut = minMVANoEleMatchWgWOgsfBL_;
	} else if ( numSignalPFGammaCands >= 1 &&  hasGsfTrack ) {
	  category = 3.;
	  mvaCut = minMVANoEleMatchWgWgsfBL_;
	}
      } else { // Endcap
	if        ( numSignalPFGammaCands == 0 && !hasGsfTrack ) {
	  category = 8.;
	  mvaCut = minMVANoEleMatchWOgWOgsfEC_;
	} else if ( numSignalPFGammaCands == 0 &&  hasGsfTrack ) {
	  category = 9.;
	  mvaCut = minMVANoEleMatchWOgWgsfEC_;
	} else if ( numSignalPFGammaCands >= 1 && !hasGsfTrack ) {
	  category = 10.;
	  mvaCut = minMVANoEleMatchWgWOgsfEC_;
	} else if ( numSignalPFGammaCands >= 1 &&  hasGsfTrack ) {
	  category = 11.;
	  mvaCut = minMVANoEleMatchWgWgsfEC_;
	}
      }
      workingPoint = (mva > mvaCut);
      //   mvaCutDummy = mvaCut;
    }
  }

//   std::cout<<" Taus : "<<TauProducer_<<std::endl;
//   std::cout << "<PFRecoTauDiscriminationAgainstElectronMVA4::discriminate>:" << std::endl;
//   std::cout << " tau: Pt = " << thePFTauRef->pt() << ", eta = " << thePFTauRef->eta() << ", phi = " << thePFTauRef->phi() << std::endl;
//   std::cout << " mva = " << mva <<  " mvaCut = " << mvaCutDummy <<" isGsfElectronMatched = "<<isGsfElectronMatched<< std::endl;
//   std::cout << " category = " << category << " : workingPoint = " << workingPoint << std::endl;
//   std::cout << " deltaREleTau = " << deltaRDummy << std::endl;
//   std::cout << " charged hadron in tau : "<<(*thePFTauRef).leadPFChargedHadrCand().isNonnull()<< std::endl;
//   std::cout << " Prongs in tau : "<<thePFTauRef->signalPFChargedHadrCands().size()<< std::endl;
//    std::cout<<"MVA GBR :"<<mva<<std::endl;

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

void PFRecoTauDiscriminationAgainstElectronMVA4GBR::endEvent(edm::Event& evt)
{
  // add all category indices to event
  if ( returnMVA_ ) {
    evt.put(category_output_, "category");
  }
}

bool
PFRecoTauDiscriminationAgainstElectronMVA4GBR::isInEcalCrack(double eta) const
{
  eta = fabs(eta);
  return (eta>1.460 && eta<1.558);
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstElectronMVA4GBR);
