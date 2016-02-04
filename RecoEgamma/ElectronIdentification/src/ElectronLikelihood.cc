#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "RecoEgamma/ElectronIdentification/interface/ElectronLikelihood.h"
#include <iostream>


ElectronLikelihood::ElectronLikelihood (const ElectronLikelihoodCalibration *calibration,
					LikelihoodSwitches eleIDSwitches,
					std::string signalWeightSplitting,
					std::string backgroundWeightSplitting,
					bool splitSignalPdfs,
					bool splitBackgroundPdfs) :
  _EBlt15lh (new LikelihoodPdfProduct ("electronID_EB_ptLt15_likelihood",0,0)) ,
  _EElt15lh (new LikelihoodPdfProduct ("electronID_EE_ptLt15_likelihood",1,0)) ,
  _EBgt15lh (new LikelihoodPdfProduct ("electronID_EB_ptGt15_likelihood",0,1)) ,
  _EEgt15lh (new LikelihoodPdfProduct ("electronID_EE_ptGt15_likelihood",1,1)) ,
  m_eleIDSwitches (eleIDSwitches) ,
  m_signalWeightSplitting (signalWeightSplitting), 
  m_backgroundWeightSplitting (backgroundWeightSplitting),
  m_splitSignalPdfs (splitSignalPdfs), 
  m_splitBackgroundPdfs (splitBackgroundPdfs)  
{
  Setup (calibration,
	 signalWeightSplitting, backgroundWeightSplitting,
	 splitSignalPdfs, splitBackgroundPdfs) ;
}



// --------------------------------------------------------



ElectronLikelihood::~ElectronLikelihood () {
  delete _EBlt15lh ;
  delete _EElt15lh ;
  delete _EBgt15lh ;
  delete _EEgt15lh ;
}



// --------------------------------------------------------


void 
ElectronLikelihood::Setup (const ElectronLikelihoodCalibration *calibration,
			   std::string signalWeightSplitting,
			   std::string backgroundWeightSplitting,
			   bool splitSignalPdfs,
			   bool splitBackgroundPdfs) 
{

  // ECAL BARREL LIKELIHOOD - Pt < 15 GeV region
  _EBlt15lh->initFromDB (calibration) ;

  _EBlt15lh->addSpecies ("electrons") ;
  _EBlt15lh->addSpecies ("hadrons") ;

  if(signalWeightSplitting.compare("class")==0) {
    _EBlt15lh->setSplitFrac ("electrons", "class0") ;
    _EBlt15lh->setSplitFrac ("electrons", "class1") ;
  }
  else {
    throw cms::Exception("BadConfig") << "Only class (non-showering / showering)"
				      << " and fullclass (golden / bigbrem / narrow / showering)" 
				      << " splitting is implemented right now";
  }

  if (m_eleIDSwitches.m_useDeltaPhi)     _EBlt15lh->addPdf ("electrons", "dPhi",          splitSignalPdfs) ;   
  if (m_eleIDSwitches.m_useDeltaEta)     _EBlt15lh->addPdf ("electrons", "dEta",          splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useEoverP)       _EBlt15lh->addPdf ("electrons", "EoP",           splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useHoverE)       _EBlt15lh->addPdf ("electrons", "HoE",           splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta)  _EBlt15lh->addPdf ("electrons", "sigmaIEtaIEta", splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useSigmaPhiPhi)  _EBlt15lh->addPdf ("electrons", "sigmaIPhiIPhi", splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useFBrem)        _EBlt15lh->addPdf ("electrons", "fBrem",         splitSignalPdfs) ;

  if(backgroundWeightSplitting.compare("class")==0) {
    _EBlt15lh->setSplitFrac ("hadrons", "class0") ;
    _EBlt15lh->setSplitFrac ("hadrons", "class1") ;
  }
  else {
    throw cms::Exception("BadConfig") << "Only class (0 brem clusters / >=1 brem clusters)"
				      << " splitting is implemented right now";
  }

  if (m_eleIDSwitches.m_useDeltaPhi)     _EBlt15lh->addPdf ("hadrons", "dPhi",          splitBackgroundPdfs) ;   
  if (m_eleIDSwitches.m_useDeltaEta)     _EBlt15lh->addPdf ("hadrons", "dEta",          splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useEoverP)       _EBlt15lh->addPdf ("hadrons", "EoP",           splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useHoverE)       _EBlt15lh->addPdf ("hadrons", "HoE",           splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta)  _EBlt15lh->addPdf ("hadrons", "sigmaIEtaIEta", splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useSigmaPhiPhi)  _EBlt15lh->addPdf ("hadrons", "sigmaIPhiIPhi", splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useFBrem)        _EBlt15lh->addPdf ("hadrons", "fBrem",         splitBackgroundPdfs) ;

  // ECAL BARREL LIKELIHOOD - Pt >= 15 GeV region
  _EBgt15lh->initFromDB (calibration) ;

  _EBgt15lh->addSpecies ("electrons") ;  
  _EBgt15lh->addSpecies ("hadrons") ;

  if(signalWeightSplitting.compare("class")==0) {
    _EBgt15lh->setSplitFrac ("electrons", "class0") ;
    _EBgt15lh->setSplitFrac ("electrons", "class1") ;
  }
  else {
    throw cms::Exception("BadConfig") << "Only class (0 brem clusters / >=1 brem clusters)"
                                      << " splitting is implemented right now";
  }

  if (m_eleIDSwitches.m_useDeltaPhi)     _EBgt15lh->addPdf ("electrons", "dPhi",          splitSignalPdfs) ;   
  if (m_eleIDSwitches.m_useDeltaEta)     _EBgt15lh->addPdf ("electrons", "dEta",          splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useEoverP)       _EBgt15lh->addPdf ("electrons", "EoP",           splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useHoverE)       _EBgt15lh->addPdf ("electrons", "HoE",           splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta)  _EBgt15lh->addPdf ("electrons", "sigmaIEtaIEta", splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useSigmaPhiPhi)  _EBgt15lh->addPdf ("electrons", "sigmaIPhiIPhi", splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useFBrem)        _EBgt15lh->addPdf ("electrons", "fBrem",         splitSignalPdfs) ;

  if(backgroundWeightSplitting.compare("class")==0) {
    _EBgt15lh->setSplitFrac ("hadrons", "class0") ;
    _EBgt15lh->setSplitFrac ("hadrons", "class1") ;
  }
  else {
    throw cms::Exception("BadConfig") << "Only class (0 brem clusters / >=1 brem clusters)"
                                      << " splitting is implemented right now";
  }

  if (m_eleIDSwitches.m_useDeltaPhi)     _EBgt15lh->addPdf ("hadrons", "dPhi",          splitBackgroundPdfs) ;   
  if (m_eleIDSwitches.m_useDeltaEta)     _EBgt15lh->addPdf ("hadrons", "dEta",          splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useEoverP)       _EBgt15lh->addPdf ("hadrons", "EoP",           splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useHoverE)       _EBgt15lh->addPdf ("hadrons", "HoE",           splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta)  _EBgt15lh->addPdf ("hadrons", "sigmaIEtaIEta", splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useSigmaPhiPhi)  _EBgt15lh->addPdf ("hadrons", "sigmaIPhiIPhi", splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useFBrem)        _EBgt15lh->addPdf ("hadrons", "fBrem",         splitBackgroundPdfs) ;

  // ECAL ENDCAP LIKELIHOOD - Pt < 15 GeV
  _EElt15lh->initFromDB (calibration) ;

  _EElt15lh->addSpecies ("electrons") ;
  _EElt15lh->addSpecies ("hadrons") ;

  if(signalWeightSplitting.compare("class")==0) {
    _EElt15lh->setSplitFrac ("electrons", "class0") ;
    _EElt15lh->setSplitFrac ("electrons", "class1") ;
  }
  else {
    throw cms::Exception("BadConfig") << "Only class (0 brem clusters / >=1 brem clusters)"
                                      << " splitting is implemented right now";
  }

  if (m_eleIDSwitches.m_useDeltaPhi)     _EElt15lh->addPdf ("electrons", "dPhi",          splitSignalPdfs) ;   
  if (m_eleIDSwitches.m_useDeltaEta)     _EElt15lh->addPdf ("electrons", "dEta",          splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useEoverP)       _EElt15lh->addPdf ("electrons", "EoP",           splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useHoverE)       _EElt15lh->addPdf ("electrons", "HoE",           splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta)  _EElt15lh->addPdf ("electrons", "sigmaIEtaIEta", splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useSigmaPhiPhi)  _EElt15lh->addPdf ("electrons", "sigmaIPhiIPhi", splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useFBrem)        _EElt15lh->addPdf ("electrons", "fBrem",         splitSignalPdfs) ;

  if(backgroundWeightSplitting.compare("class")==0) {
    _EElt15lh->setSplitFrac ("hadrons", "class0") ;
    _EElt15lh->setSplitFrac ("hadrons", "class1") ;
  }
  else {
    throw cms::Exception("BadConfig") << "Only class (0 brem clusters / >=1 brem clusters)"
                                      << " splitting is implemented right now";
  }

  if (m_eleIDSwitches.m_useDeltaPhi)     _EElt15lh->addPdf ("hadrons", "dPhi",          splitBackgroundPdfs) ;   
  if (m_eleIDSwitches.m_useDeltaEta)     _EElt15lh->addPdf ("hadrons", "dEta",          splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useEoverP)       _EElt15lh->addPdf ("hadrons", "EoP",           splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useHoverE)       _EElt15lh->addPdf ("hadrons", "HoE",           splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta)  _EElt15lh->addPdf ("hadrons", "sigmaIEtaIEta", splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useSigmaPhiPhi)  _EElt15lh->addPdf ("hadrons", "sigmaIPhiIPhi", splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useFBrem)        _EElt15lh->addPdf ("hadrons", "fBrem",         splitBackgroundPdfs) ;

  // ECAL ENDCAP LIKELIHOOD - Pt >= 15 GeV
  _EEgt15lh->initFromDB (calibration) ;

  _EEgt15lh->addSpecies ("electrons") ;
  _EEgt15lh->addSpecies ("hadrons") ;

  if(signalWeightSplitting.compare("class")==0) {
    _EEgt15lh->setSplitFrac ("electrons", "class0") ;
    _EEgt15lh->setSplitFrac ("electrons", "class1") ;
  }
  else {
    throw cms::Exception("BadConfig") << "Only class (0 brem clusters / >=1 brem clusters)"
                                      << " splitting is implemented right now";
  }

  if (m_eleIDSwitches.m_useDeltaPhi)     _EEgt15lh->addPdf ("electrons", "dPhi",          splitSignalPdfs) ;   
  if (m_eleIDSwitches.m_useDeltaEta)     _EEgt15lh->addPdf ("electrons", "dEta",          splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useEoverP)       _EEgt15lh->addPdf ("electrons", "EoP",           splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useHoverE)       _EEgt15lh->addPdf ("electrons", "HoE",           splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta)  _EEgt15lh->addPdf ("electrons", "sigmaIEtaIEta", splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useSigmaPhiPhi)  _EEgt15lh->addPdf ("electrons", "sigmaIPhiIPhi", splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useFBrem)        _EEgt15lh->addPdf ("electrons", "fBrem",         splitSignalPdfs) ;

  if(backgroundWeightSplitting.compare("class")==0) {
    _EEgt15lh->setSplitFrac ("hadrons", "class0") ;
    _EEgt15lh->setSplitFrac ("hadrons", "class1") ;
  }
  else {
    throw cms::Exception("BadConfig") << "Only class (0 brem clusters / >=1 brem clusters)"
                                      << " splitting is implemented right now";
  }

  if (m_eleIDSwitches.m_useDeltaPhi)     _EEgt15lh->addPdf ("hadrons", "dPhi",          splitBackgroundPdfs) ;   
  if (m_eleIDSwitches.m_useDeltaEta)     _EEgt15lh->addPdf ("hadrons", "dEta",          splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useEoverP)       _EEgt15lh->addPdf ("hadrons", "EoP",           splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useHoverE)       _EEgt15lh->addPdf ("hadrons", "HoE",           splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta)  _EEgt15lh->addPdf ("hadrons", "sigmaIEtaIEta", splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useSigmaPhiPhi)  _EEgt15lh->addPdf ("hadrons", "sigmaIPhiIPhi", splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useFBrem)        _EEgt15lh->addPdf ("hadrons", "fBrem",         splitBackgroundPdfs) ;
}



// --------------------------------------------------------



void 
ElectronLikelihood::getInputVar (const reco::GsfElectron &electron, 
                                 std::vector<float> &measurements, 
                                 EcalClusterLazyTools myEcalCluster) const 
{

  // the variables entering the likelihood
  if (m_eleIDSwitches.m_useDeltaPhi) measurements.push_back ( electron.deltaPhiSuperClusterTrackAtVtx () ) ;
  if (m_eleIDSwitches.m_useDeltaEta) measurements.push_back ( electron.deltaEtaSuperClusterTrackAtVtx () ) ;
  if (m_eleIDSwitches.m_useEoverP) measurements.push_back ( electron.eSuperClusterOverP () ) ;
  if (m_eleIDSwitches.m_useHoverE) measurements.push_back ( electron.hadronicOverEm () ) ;
  std::vector<float> vCov = myEcalCluster.covariances(*(electron.superCluster()->seed())) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta) measurements.push_back ( sqrt (vCov[0]) );
  if (m_eleIDSwitches.m_useSigmaPhiPhi) measurements.push_back ( sqrt (vCov[2]) );
  if(m_eleIDSwitches.m_useFBrem) measurements.push_back( electron.fbrem() );

}



// --------------------------------------------------------



float 
ElectronLikelihood::result (const reco::GsfElectron &electron, 
                            EcalClusterLazyTools myEcalCluster) const 
{

  //=======================================================
  // used classification:
  // nbrem clusters = 0         =>  0
  // nbrem clusters >= 1        =>  1
  //=======================================================

  std::vector<float> measurements ;
  getInputVar (electron, measurements, myEcalCluster) ;

  // Split using only the number of brem clusters
  int bitVal=(electron.numberOfBrems()==0) ? 0 : 1 ;
  
  char className[20] ;
  if(m_signalWeightSplitting.compare("class")==0) {
    sprintf (className,"class%d",bitVal);
  } else {
    throw cms::Exception("BadConfig") << "Only class (0 brem clusters / >=1 brem clusters)"
				      << " splitting is implemented right now";
  }

  reco::SuperClusterRef sclusRef = electron.superCluster() ;
  EcalSubdetector subdet = EcalSubdetector (sclusRef->hitsAndFractions()[0].first.subdetId ()) ;
  float thisPt =  electron.pt();

  if (subdet==EcalBarrel && thisPt<15.)
    return _EBlt15lh->getRatio ("electrons",measurements,std::string (className)) ;
  else if (subdet==EcalBarrel && thisPt>=15.)
    return _EBgt15lh->getRatio ("electrons",measurements,std::string (className)) ;
  else if (subdet==EcalEndcap && thisPt<15.)
    return _EElt15lh->getRatio ("electrons",measurements,std::string (className)) ;
  else if (subdet==EcalEndcap && thisPt>=15.)
    return _EEgt15lh->getRatio ("electrons",measurements,std::string (className)) ;
  else return -999. ;
}

