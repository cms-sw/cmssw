#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

#include "RecoEgamma/ElectronIdentification/interface/ElectronLikelihood.h"
#include <iostream>


ElectronLikelihood::ElectronLikelihood (const ElectronLikelihoodCalibration *calibration,
					LikelihoodSwitches eleIDSwitches,
					std::string signalWeightSplitting,
					std::string backgroundWeightSplitting,
					bool splitSignalPdfs,
					bool splitBackgroundPdfs) :
  _EB0lt15lh (new LikelihoodPdfProduct ("electronID_EB0_ptLt15_likelihood",0,0)) ,
  _EB1lt15lh (new LikelihoodPdfProduct ("electronID_EB1_ptLt15_likelihood",1,0)) ,
  _EElt15lh (new LikelihoodPdfProduct ("electronID_EE_ptLt15_likelihood",2,0)) ,
  _EB0gt15lh (new LikelihoodPdfProduct ("electronID_EB0_ptGt15_likelihood",0,1)) ,
  _EB1gt15lh (new LikelihoodPdfProduct ("electronID_EB1_ptGt15_likelihood",1,1)) ,
  _EEgt15lh (new LikelihoodPdfProduct ("electronID_EE_ptGt15_likelihood",2,1)) ,
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
  delete _EB0lt15lh.get() ;
  delete _EB1lt15lh.get() ;
  delete _EElt15lh.get() ;
  delete _EB0gt15lh.get() ;
  delete _EB1gt15lh.get() ;
  delete _EEgt15lh.get() ;
}



// --------------------------------------------------------


void 
ElectronLikelihood::Setup (const ElectronLikelihoodCalibration *calibration,
			   std::string signalWeightSplitting,
			   std::string backgroundWeightSplitting,
			   bool splitSignalPdfs,
			   bool splitBackgroundPdfs) 
{

  // ECAL BARREL0 (|eta|<1.0) LIKELIHOOD - Pt < 15 GeV region
  _EB0lt15lh->initFromDB (calibration) ;

  _EB0lt15lh->addSpecies ("electrons") ;
  _EB0lt15lh->addSpecies ("hadrons") ;

  if(signalWeightSplitting.compare("class")==0) {
    _EB0lt15lh->setSplitFrac ("electrons", "class0") ;
    _EB0lt15lh->setSplitFrac ("electrons", "class1") ;
  }
  else {
    throw cms::Exception("BadConfig") << "Only class (non-showering / showering)"
				      << " and fullclass (golden / bigbrem / narrow / showering)" 
				      << " splitting is implemented right now";
  }

  if (m_eleIDSwitches.m_useDeltaPhi)     _EB0lt15lh->addPdf ("electrons", "dPhi",          splitSignalPdfs) ;   
  if (m_eleIDSwitches.m_useDeltaEta)     _EB0lt15lh->addPdf ("electrons", "dEta",          splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useEoverP)       _EB0lt15lh->addPdf ("electrons", "EoP",           splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useHoverE)       _EB0lt15lh->addPdf ("electrons", "HoE",           splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta)  _EB0lt15lh->addPdf ("electrons", "sigmaIEtaIEta", splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useSigmaPhiPhi)  _EB0lt15lh->addPdf ("electrons", "sigmaIPhiIPhi", splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useFBrem)        _EB0lt15lh->addPdf ("electrons", "fBrem",         splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useOneOverEMinusOneOverP)        _EB0lt15lh->addPdf ("electrons", "OneOverEMinusOneOverP",         splitSignalPdfs) ;

  if(backgroundWeightSplitting.compare("class")==0) {
    _EB0lt15lh->setSplitFrac ("hadrons", "class0") ;
    _EB0lt15lh->setSplitFrac ("hadrons", "class1") ;
  }
  else {
    throw cms::Exception("BadConfig") << "Only class (0 brem clusters / >=1 brem clusters)"
				      << " splitting is implemented right now";
  }

  if (m_eleIDSwitches.m_useDeltaPhi)     _EB0lt15lh->addPdf ("hadrons", "dPhi",          splitBackgroundPdfs) ;   
  if (m_eleIDSwitches.m_useDeltaEta)     _EB0lt15lh->addPdf ("hadrons", "dEta",          splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useEoverP)       _EB0lt15lh->addPdf ("hadrons", "EoP",           splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useHoverE)       _EB0lt15lh->addPdf ("hadrons", "HoE",           splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta)  _EB0lt15lh->addPdf ("hadrons", "sigmaIEtaIEta", splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useSigmaPhiPhi)  _EB0lt15lh->addPdf ("hadrons", "sigmaIPhiIPhi", splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useFBrem)        _EB0lt15lh->addPdf ("hadrons", "fBrem",         splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useOneOverEMinusOneOverP)        _EB0lt15lh->addPdf ("hadrons", "OneOverEMinusOneOverP",         splitBackgroundPdfs) ;

  // ECAL BARREL0 (|eta|<1.0) LIKELIHOOD - Pt >= 15 GeV region
  _EB0gt15lh->initFromDB (calibration) ;

  _EB0gt15lh->addSpecies ("electrons") ;  
  _EB0gt15lh->addSpecies ("hadrons") ;

  if(signalWeightSplitting.compare("class")==0) {
    _EB0gt15lh->setSplitFrac ("electrons", "class0") ;
    _EB0gt15lh->setSplitFrac ("electrons", "class1") ;
  }
  else {
    throw cms::Exception("BadConfig") << "Only class (0 brem clusters / >=1 brem clusters)"
                                      << " splitting is implemented right now";
  }

  if (m_eleIDSwitches.m_useDeltaPhi)     _EB0gt15lh->addPdf ("electrons", "dPhi",          splitSignalPdfs) ;   
  if (m_eleIDSwitches.m_useDeltaEta)     _EB0gt15lh->addPdf ("electrons", "dEta",          splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useEoverP)       _EB0gt15lh->addPdf ("electrons", "EoP",           splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useHoverE)       _EB0gt15lh->addPdf ("electrons", "HoE",           splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta)  _EB0gt15lh->addPdf ("electrons", "sigmaIEtaIEta", splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useSigmaPhiPhi)  _EB0gt15lh->addPdf ("electrons", "sigmaIPhiIPhi", splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useFBrem)        _EB0gt15lh->addPdf ("electrons", "fBrem",         splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useOneOverEMinusOneOverP)        _EB0gt15lh->addPdf ("electrons", "OneOverEMinusOneOverP",         splitSignalPdfs) ;

  if(backgroundWeightSplitting.compare("class")==0) {
    _EB0gt15lh->setSplitFrac ("hadrons", "class0") ;
    _EB0gt15lh->setSplitFrac ("hadrons", "class1") ;
  }
  else {
    throw cms::Exception("BadConfig") << "Only class (0 brem clusters / >=1 brem clusters)"
                                      << " splitting is implemented right now";
  }

  if (m_eleIDSwitches.m_useDeltaPhi)     _EB0gt15lh->addPdf ("hadrons", "dPhi",          splitBackgroundPdfs) ;   
  if (m_eleIDSwitches.m_useDeltaEta)     _EB0gt15lh->addPdf ("hadrons", "dEta",          splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useEoverP)       _EB0gt15lh->addPdf ("hadrons", "EoP",           splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useHoverE)       _EB0gt15lh->addPdf ("hadrons", "HoE",           splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta)  _EB0gt15lh->addPdf ("hadrons", "sigmaIEtaIEta", splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useSigmaPhiPhi)  _EB0gt15lh->addPdf ("hadrons", "sigmaIPhiIPhi", splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useFBrem)        _EB0gt15lh->addPdf ("hadrons", "fBrem",         splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useOneOverEMinusOneOverP)        _EB0gt15lh->addPdf ("hadrons", "OneOverEMinusOneOverP",         splitBackgroundPdfs) ;


  // ECAL BARREL1 (|eta|>1.0) LIKELIHOOD - Pt < 15 GeV region
  _EB1lt15lh->initFromDB (calibration) ;

  _EB1lt15lh->addSpecies ("electrons") ;
  _EB1lt15lh->addSpecies ("hadrons") ;

  if(signalWeightSplitting.compare("class")==0) {
    _EB1lt15lh->setSplitFrac ("electrons", "class0") ;
    _EB1lt15lh->setSplitFrac ("electrons", "class1") ;
  }
  else {
    throw cms::Exception("BadConfig") << "Only class (non-showering / showering)"
				      << " and fullclass (golden / bigbrem / narrow / showering)" 
				      << " splitting is implemented right now";
  }

  if (m_eleIDSwitches.m_useDeltaPhi)     _EB1lt15lh->addPdf ("electrons", "dPhi",          splitSignalPdfs) ;   
  if (m_eleIDSwitches.m_useDeltaEta)     _EB1lt15lh->addPdf ("electrons", "dEta",          splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useEoverP)       _EB1lt15lh->addPdf ("electrons", "EoP",           splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useHoverE)       _EB1lt15lh->addPdf ("electrons", "HoE",           splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta)  _EB1lt15lh->addPdf ("electrons", "sigmaIEtaIEta", splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useSigmaPhiPhi)  _EB1lt15lh->addPdf ("electrons", "sigmaIPhiIPhi", splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useFBrem)        _EB1lt15lh->addPdf ("electrons", "fBrem",         splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useOneOverEMinusOneOverP)        _EB1lt15lh->addPdf ("electrons", "OneOverEMinusOneOverP",         splitSignalPdfs) ;

  if(backgroundWeightSplitting.compare("class")==0) {
    _EB1lt15lh->setSplitFrac ("hadrons", "class0") ;
    _EB1lt15lh->setSplitFrac ("hadrons", "class1") ;
  }
  else {
    throw cms::Exception("BadConfig") << "Only class (0 brem clusters / >=1 brem clusters)"
				      << " splitting is implemented right now";
  }

  if (m_eleIDSwitches.m_useDeltaPhi)     _EB1lt15lh->addPdf ("hadrons", "dPhi",          splitBackgroundPdfs) ;   
  if (m_eleIDSwitches.m_useDeltaEta)     _EB1lt15lh->addPdf ("hadrons", "dEta",          splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useEoverP)       _EB1lt15lh->addPdf ("hadrons", "EoP",           splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useHoverE)       _EB1lt15lh->addPdf ("hadrons", "HoE",           splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta)  _EB1lt15lh->addPdf ("hadrons", "sigmaIEtaIEta", splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useSigmaPhiPhi)  _EB1lt15lh->addPdf ("hadrons", "sigmaIPhiIPhi", splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useFBrem)        _EB1lt15lh->addPdf ("hadrons", "fBrem",         splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useOneOverEMinusOneOverP)        _EB1lt15lh->addPdf ("hadrons", "OneOverEMinusOneOverP",         splitBackgroundPdfs) ;

  // ECAL BARREL1 (|eta|>1.0) LIKELIHOOD - Pt >= 15 GeV region
  _EB1gt15lh->initFromDB (calibration) ;

  _EB1gt15lh->addSpecies ("electrons") ;  
  _EB1gt15lh->addSpecies ("hadrons") ;

  if(signalWeightSplitting.compare("class")==0) {
    _EB1gt15lh->setSplitFrac ("electrons", "class0") ;
    _EB1gt15lh->setSplitFrac ("electrons", "class1") ;
  }
  else {
    throw cms::Exception("BadConfig") << "Only class (0 brem clusters / >=1 brem clusters)"
                                      << " splitting is implemented right now";
  }

  if (m_eleIDSwitches.m_useDeltaPhi)     _EB1gt15lh->addPdf ("electrons", "dPhi",          splitSignalPdfs) ;   
  if (m_eleIDSwitches.m_useDeltaEta)     _EB1gt15lh->addPdf ("electrons", "dEta",          splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useEoverP)       _EB1gt15lh->addPdf ("electrons", "EoP",           splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useHoverE)       _EB1gt15lh->addPdf ("electrons", "HoE",           splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta)  _EB1gt15lh->addPdf ("electrons", "sigmaIEtaIEta", splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useSigmaPhiPhi)  _EB1gt15lh->addPdf ("electrons", "sigmaIPhiIPhi", splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useFBrem)        _EB1gt15lh->addPdf ("electrons", "fBrem",         splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useOneOverEMinusOneOverP)        _EB1gt15lh->addPdf ("electrons", "OneOverEMinusOneOverP",         splitSignalPdfs) ;

  if(backgroundWeightSplitting.compare("class")==0) {
    _EB1gt15lh->setSplitFrac ("hadrons", "class0") ;
    _EB1gt15lh->setSplitFrac ("hadrons", "class1") ;
  }
  else {
    throw cms::Exception("BadConfig") << "Only class (0 brem clusters / >=1 brem clusters)"
                                      << " splitting is implemented right now";
  }

  if (m_eleIDSwitches.m_useDeltaPhi)     _EB1gt15lh->addPdf ("hadrons", "dPhi",          splitBackgroundPdfs) ;   
  if (m_eleIDSwitches.m_useDeltaEta)     _EB1gt15lh->addPdf ("hadrons", "dEta",          splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useEoverP)       _EB1gt15lh->addPdf ("hadrons", "EoP",           splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useHoverE)       _EB1gt15lh->addPdf ("hadrons", "HoE",           splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta)  _EB1gt15lh->addPdf ("hadrons", "sigmaIEtaIEta", splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useSigmaPhiPhi)  _EB1gt15lh->addPdf ("hadrons", "sigmaIPhiIPhi", splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useFBrem)        _EB1gt15lh->addPdf ("hadrons", "fBrem",         splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useOneOverEMinusOneOverP)        _EB1gt15lh->addPdf ("hadrons", "OneOverEMinusOneOverP",         splitBackgroundPdfs) ;

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
  if (m_eleIDSwitches.m_useOneOverEMinusOneOverP)        _EElt15lh->addPdf ("electrons", "OneOverEMinusOneOverP",         splitSignalPdfs) ;

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
  if (m_eleIDSwitches.m_useOneOverEMinusOneOverP)        _EElt15lh->addPdf ("hadrons", "OneOverEMinusOneOverP",         splitBackgroundPdfs) ;

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
  if (m_eleIDSwitches.m_useOneOverEMinusOneOverP)        _EEgt15lh->addPdf ("electrons", "OneOverEMinusOneOverP",         splitSignalPdfs) ;

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
  if (m_eleIDSwitches.m_useOneOverEMinusOneOverP)        _EEgt15lh->addPdf ("hadrons", "OneOverEMinusOneOverP",         splitBackgroundPdfs) ;
}



// --------------------------------------------------------



void 
ElectronLikelihood::getInputVar (const reco::GsfElectron &electron, 
                                 std::vector<float> &measurements, 
                                 const EcalClusterLazyTools& _myEcalCluster) const 
{
  EcalClusterLazyTools myEcalCluster = _myEcalCluster;
  // the variables entering the likelihood
  if (m_eleIDSwitches.m_useDeltaPhi) measurements.push_back ( electron.deltaPhiSuperClusterTrackAtVtx () ) ;
  if (m_eleIDSwitches.m_useDeltaEta) measurements.push_back ( electron.deltaEtaSuperClusterTrackAtVtx () ) ;
  if (m_eleIDSwitches.m_useEoverP) measurements.push_back ( electron.eSuperClusterOverP () ) ;
  if (m_eleIDSwitches.m_useHoverE) measurements.push_back ( electron.hadronicOverEm () ) ;
  std::vector<float> vCov = myEcalCluster.localCovariances(*(electron.superCluster()->seed())) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta) measurements.push_back ( sqrt (vCov[0]) );
  if (m_eleIDSwitches.m_useSigmaPhiPhi) measurements.push_back ( sqrt (vCov[2]) );
  if(m_eleIDSwitches.m_useFBrem) measurements.push_back( electron.fbrem() );
  // 1/E - 1/P calculated consistently with the variables used to make the PDFs
  reco::GsfTrackRef trkRef = electron.get<reco::GsfTrackRef>();
  float OneOverEMinusOneOverP = 1.0/(electron.eSuperClusterOverP() * trkRef->p()) - 1.0/trkRef->p();
  if(m_eleIDSwitches.m_useOneOverEMinusOneOverP) measurements.push_back( OneOverEMinusOneOverP );

}



// --------------------------------------------------------



float 
ElectronLikelihood::result (const reco::GsfElectron &electron, 
                            const EcalClusterLazyTools& myEcalCluster) const 
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
  
  char className[20];
  if(m_signalWeightSplitting.compare("class")==0) {
    snprintf(className, 20, "class%d", bitVal);
  } else {
    throw cms::Exception("BadConfig") << "Only class (0 brem clusters / >=1 brem clusters)"
				      << " splitting is implemented right now";
  }

  reco::SuperClusterRef sclusRef = electron.superCluster() ;
  EcalSubdetector subdet = EcalSubdetector (sclusRef->hitsAndFractions()[0].first.subdetId ()) ;
  float thisPt =  electron.pt();

  if (subdet==EcalBarrel && fabs(electron.eta())<=1.0 && thisPt<15.)
    return _EB0lt15lh->getRatio ("electrons",measurements,std::string (className)) ;
  else if (subdet==EcalBarrel && fabs(electron.eta())<=1.0 && thisPt>=15.)
    return _EB0gt15lh->getRatio ("electrons",measurements,std::string (className)) ;
  else if (subdet==EcalBarrel && fabs(electron.eta())>1.0 && thisPt<15.)
    return _EB1lt15lh->getRatio ("electrons",measurements,std::string (className)) ;
  else if (subdet==EcalBarrel && fabs(electron.eta())>1.0 && thisPt>=15.)
    return _EB1gt15lh->getRatio ("electrons",measurements,std::string (className)) ;
  else if (subdet==EcalEndcap && thisPt<15.)
    return _EElt15lh->getRatio ("electrons",measurements,std::string (className)) ;
  else if (subdet==EcalEndcap && thisPt>=15.)
    return _EEgt15lh->getRatio ("electrons",measurements,std::string (className)) ;
  else return -999. ;
}

float 
ElectronLikelihood::resultLog (const reco::GsfElectron &electron, 
                               const EcalClusterLazyTools& myEcalCluster) const 
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
  
  char className[20];
  if(m_signalWeightSplitting.compare("class")==0) {
    snprintf(className, 20, "class%d", bitVal);
  } else {
    throw cms::Exception("BadConfig") << "Only class (0 brem clusters / >=1 brem clusters)"
				      << " splitting is implemented right now";
  }

  reco::SuperClusterRef sclusRef = electron.superCluster() ;
  EcalSubdetector subdet = EcalSubdetector (sclusRef->hitsAndFractions()[0].first.subdetId ()) ;
  float thisPt =  electron.pt();

  float lh=-999.;

  if (subdet==EcalBarrel && fabs(electron.eta())<=1.0 && thisPt<15.)
    lh = _EB0lt15lh->getRatio ("electrons",measurements,std::string (className)) ;
  else if (subdet==EcalBarrel && fabs(electron.eta())<=1.0 && thisPt>=15.)
    lh = _EB0gt15lh->getRatio ("electrons",measurements,std::string (className)) ;
  else if (subdet==EcalBarrel && fabs(electron.eta())>1.0 && thisPt<15.)
    lh = _EB1lt15lh->getRatio ("electrons",measurements,std::string (className)) ;
  else if (subdet==EcalBarrel && fabs(electron.eta())>1.0 && thisPt>=15.)
    lh = _EB1gt15lh->getRatio ("electrons",measurements,std::string (className)) ;
  else if (subdet==EcalEndcap && thisPt<15.)
    lh = _EElt15lh->getRatio ("electrons",measurements,std::string (className)) ;
  else if (subdet==EcalEndcap && thisPt>=15.)
    lh = _EEgt15lh->getRatio ("electrons",measurements,std::string (className)) ;
  else lh = -999. ;

  if(lh<=0) return -20.;
  else if(lh==1) return 20.;
  else return log(lh/(1.0-lh));

}

