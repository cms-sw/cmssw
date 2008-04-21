#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "RecoEgamma/ElectronIdentification/interface/ElectronLikelihood.h"
#include <iostream>


ElectronLikelihood::ElectronLikelihood (const ElectronLikelihoodCalibration *calibration,
					const std::vector<double> & fisherEBLt15,
					const std::vector<double> & fisherEBGt15,
					const std::vector<double> & fisherEELt15,
					const std::vector<double> & fisherEEGt15,
					const std::vector<double> & eleFracsEBlt15,
					const std::vector<double> & piFracsEBlt15,
					const std::vector<double> & eleFracsEElt15,
					const std::vector<double> & piFracsEElt15,
					const std::vector<double> & eleFracsEBgt15,
					const std::vector<double> & piFracsEBgt15,
					const std::vector<double> & eleFracsEEgt15,
					const std::vector<double> & piFracsEEgt15,
					double eleWeight,
					double piWeight,
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
	 fisherEBLt15, fisherEBGt15,
	 fisherEELt15, fisherEEGt15,
	 eleFracsEBlt15, piFracsEBlt15,
	 eleFracsEElt15, piFracsEElt15,
	 eleFracsEBgt15, piFracsEBgt15,
	 eleFracsEEgt15, piFracsEEgt15,
	 eleWeight, piWeight,
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
			   const std::vector<double> & fisherEBLt15,
			   const std::vector<double> & fisherEBGt15,
			   const std::vector<double> & fisherEELt15,
			   const std::vector<double> & fisherEEGt15,
			   const std::vector<double> & eleFracsEBlt15,
			   const std::vector<double> & piFracsEBlt15,
			   const std::vector<double> & eleFracsEElt15,
			   const std::vector<double> & piFracsEElt15,
			   const std::vector<double> & eleFracsEBgt15,
			   const std::vector<double> & piFracsEBgt15,
			   const std::vector<double> & eleFracsEEgt15,
			   const std::vector<double> & piFracsEEgt15,
			   double eleWeight,
			   double piWeight,
			   std::string signalWeightSplitting,
			   std::string backgroundWeightSplitting,
			   bool splitSignalPdfs,
			   bool splitBackgroundPdfs) 
{

  // ECAL BARREL LIKELIHOOD - Pt < 15 GeV region
  _EBlt15lh->initFromDB (calibration) ;

  _EBlt15lh->addSpecies ("electrons",eleWeight) ;
  _EBlt15lh->addSpecies ("hadrons",  piWeight) ;

  if(signalWeightSplitting.compare("fullclass")==0) {
    _EBlt15lh->setSplitFrac ("electrons", "fullclass0", eleFracsEBlt15[0]) ;
    _EBlt15lh->setSplitFrac ("electrons", "fullclass1", eleFracsEBlt15[1]) ;
    _EBlt15lh->setSplitFrac ("electrons", "fullclass2", eleFracsEBlt15[2]) ;
    _EBlt15lh->setSplitFrac ("electrons", "fullclass3", eleFracsEBlt15[3]) ;
  }
  else if(signalWeightSplitting.compare("class")==0) {
    _EBlt15lh->setSplitFrac ("electrons", "class0", eleFracsEBlt15[0]+eleFracsEBlt15[1]+eleFracsEBlt15[2]) ;
    _EBlt15lh->setSplitFrac ("electrons", "class1", eleFracsEBlt15[3]) ;
  }
  else {
    throw cms::Exception("BadConfig") << "Only class (non-showering / showering)"
				      << " and fullclass (golden / bigbrem / narrow / showering)" 
				      << " splitting is implemented right now";
  }

  if (m_eleIDSwitches.m_useDeltaPhiIn)   _EBlt15lh->addPdf ("electrons", "dPhiVtx",     splitSignalPdfs) ;   
  if (m_eleIDSwitches.m_useDeltaEtaCalo) _EBlt15lh->addPdf ("electrons", "dEtaCalo",    splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useEoverPOut)    _EBlt15lh->addPdf ("electrons", "EoPout",      splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useHoverE)       _EBlt15lh->addPdf ("electrons", "HoE",         splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useShapeFisher)  _EBlt15lh->addPdf ("electrons", "shapeFisher", splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta)  _EBlt15lh->addPdf ("electrons", "sigmaEtaEta", splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useE9overE25)    _EBlt15lh->addPdf ("electrons", "s9s25",       splitSignalPdfs) ;

  if(backgroundWeightSplitting.compare("fullclass")==0) {
    _EBlt15lh->setSplitFrac ("hadrons", "fullclass0", piFracsEBlt15[0]) ;
    _EBlt15lh->setSplitFrac ("hadrons", "fullclass1", piFracsEBlt15[1]) ;
    _EBlt15lh->setSplitFrac ("hadrons", "fullclass2", piFracsEBlt15[2]) ;
    _EBlt15lh->setSplitFrac ("hadrons", "fullclass3", piFracsEBlt15[3]) ;
  }
  else if(backgroundWeightSplitting.compare("class")==0) {
    _EBlt15lh->setSplitFrac ("hadrons", "class0", piFracsEBlt15[0]+piFracsEBlt15[1]+piFracsEBlt15[2]) ;
    _EBlt15lh->setSplitFrac ("hadrons", "class1", piFracsEBlt15[3]) ;
  }
  else {
    throw cms::Exception("BadConfig") << "Only class (non-showering / showering)"
				      << " and fullclass (golden / bigbrem / narrow / showering)" 
				      << " splitting is implemented right now";
  }

  if (m_eleIDSwitches.m_useDeltaPhiIn)   _EBlt15lh->addPdf ("hadrons", "dPhiVtx",       splitBackgroundPdfs) ;  
  if (m_eleIDSwitches.m_useDeltaEtaCalo) _EBlt15lh->addPdf ("hadrons", "dEtaCalo",      splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useEoverPOut)    _EBlt15lh->addPdf ("hadrons", "EoPout",        splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useHoverE)       _EBlt15lh->addPdf ("hadrons", "HoE",           splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useShapeFisher)  _EBlt15lh->addPdf ("hadrons", "shapeFisher",   splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta)  _EBlt15lh->addPdf ("hadrons", "sigmaEtaEta",   splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useE9overE25)    _EBlt15lh->addPdf ("hadrons", "s9s25",         splitBackgroundPdfs) ;


  // ECAL BARREL LIKELIHOOD - Pt >= 15 GeV region
  _EBgt15lh->initFromDB (calibration) ;

  _EBgt15lh->addSpecies ("electrons",eleWeight) ;  
  _EBgt15lh->addSpecies ("hadrons",piWeight) ;

  if(signalWeightSplitting.compare("fullclass")==0) {
    _EBgt15lh->setSplitFrac ("electrons", "fullclass0", eleFracsEBgt15[0]) ;
    _EBgt15lh->setSplitFrac ("electrons", "fullclass1", eleFracsEBgt15[1]) ;
    _EBgt15lh->setSplitFrac ("electrons", "fullclass2", eleFracsEBgt15[2]) ;
    _EBgt15lh->setSplitFrac ("electrons", "fullclass3", eleFracsEBgt15[3]) ;
  }
  else if(signalWeightSplitting.compare("class")==0) {
    _EBgt15lh->setSplitFrac ("electrons", "class0", eleFracsEBgt15[0]+eleFracsEBgt15[1]+eleFracsEBgt15[2]) ;
    _EBgt15lh->setSplitFrac ("electrons", "class1", eleFracsEBgt15[3]) ;
  }
  else {
    throw cms::Exception("BadConfig") << "Only class (non-showering / showering)"
				      << " and fullclass (golden / bigbrem / narrow / showering)" 
				      << " splitting is implemented right now";
  }

  if (m_eleIDSwitches.m_useDeltaPhiIn)   _EBgt15lh->addPdf ("electrons", "dPhiVtx",     splitSignalPdfs) ;  
  if (m_eleIDSwitches.m_useDeltaEtaCalo) _EBgt15lh->addPdf ("electrons", "dEtaCalo",    splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useEoverPOut)    _EBgt15lh->addPdf ("electrons", "EoPout",      splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useHoverE)       _EBgt15lh->addPdf ("electrons", "HoE",         splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useShapeFisher)  _EBgt15lh->addPdf ("electrons", "shapeFisher", splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta)  _EBgt15lh->addPdf ("electrons", "sigmaEtaEta", splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useE9overE25)    _EBgt15lh->addPdf ("electrons", "s9s25",       splitSignalPdfs) ;


  if(backgroundWeightSplitting.compare("fullclass")==0) {
    _EBgt15lh->setSplitFrac ("hadrons", "fullclass0", piFracsEBgt15[0]) ;
    _EBgt15lh->setSplitFrac ("hadrons", "fullclass1", piFracsEBgt15[1]) ;
    _EBgt15lh->setSplitFrac ("hadrons", "fullclass2", piFracsEBgt15[2]) ;
    _EBgt15lh->setSplitFrac ("hadrons", "fullclass3", piFracsEBgt15[3]) ;
  }
  else if(backgroundWeightSplitting.compare("class")==0) {
    _EBgt15lh->setSplitFrac ("hadrons", "class0", piFracsEBgt15[0]+piFracsEBgt15[1]+ piFracsEBgt15[2]) ;
    _EBgt15lh->setSplitFrac ("hadrons", "class1", piFracsEBgt15[3]) ;
  }
  else {
    throw cms::Exception("BadConfig") << "Only class (non-showering / showering)"
				      << " and fullclass (golden / bigbrem / narrow / showering)" 
				      << " splitting is implemented right now";
  }

  if (m_eleIDSwitches.m_useDeltaPhiIn)   _EBgt15lh->addPdf ("hadrons", "dPhiVtx",       splitBackgroundPdfs) ; 
  if (m_eleIDSwitches.m_useDeltaEtaCalo) _EBgt15lh->addPdf ("hadrons", "dEtaCalo",      splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useEoverPOut)    _EBgt15lh->addPdf ("hadrons", "EoPout",        splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useHoverE)       _EBgt15lh->addPdf ("hadrons", "HoE",           splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useShapeFisher)  _EBgt15lh->addPdf ("hadrons", "shapeFisher",   splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta)  _EBgt15lh->addPdf ("hadrons", "sigmaEtaEta",   splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useE9overE25)    _EBgt15lh->addPdf ("hadrons", "s9s25",         splitBackgroundPdfs) ;

  // ECAL ENDCAP LIKELIHOOD - Pt < 15 GeV
  _EElt15lh->initFromDB (calibration) ;

  _EElt15lh->addSpecies ("electrons",eleWeight) ;
  _EElt15lh->addSpecies ("hadrons",piWeight) ;

  if(signalWeightSplitting.compare("fullclass")==0) {
    _EElt15lh->setSplitFrac ("electrons", "fullclass0", eleFracsEElt15[0]) ;
    _EElt15lh->setSplitFrac ("electrons", "fullclass1", eleFracsEElt15[1]) ;
    _EElt15lh->setSplitFrac ("electrons", "fullclass2", eleFracsEElt15[2]) ;
    _EElt15lh->setSplitFrac ("electrons", "fullclass3", eleFracsEElt15[3]) ;
  }
  else if(signalWeightSplitting.compare("class")==0) {
    _EElt15lh->setSplitFrac ("electrons", "class0", eleFracsEElt15[0]+eleFracsEElt15[1]+eleFracsEElt15[2]) ;
    _EElt15lh->setSplitFrac ("electrons", "class1", eleFracsEElt15[3]) ;
  }
  else {
    throw cms::Exception("BadConfig") << "Only class (non-showering / showering)"
				      << " and fullclass (golden / bigbrem / narrow / showering)" 
				      << " splitting is implemented right now";
  }

  if (m_eleIDSwitches.m_useDeltaPhiIn)   _EElt15lh->addPdf ("electrons", "dPhiVtx",     splitSignalPdfs) ;  
  if (m_eleIDSwitches.m_useDeltaEtaCalo) _EElt15lh->addPdf ("electrons", "dEtaCalo",    splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useEoverPOut)    _EElt15lh->addPdf ("electrons", "EoPout",      splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useHoverE)       _EElt15lh->addPdf ("electrons", "HoE",         splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useShapeFisher)  _EElt15lh->addPdf ("electrons", "shapeFisher", splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta)  _EElt15lh->addPdf ("electrons", "sigmaEtaEta", splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useE9overE25)    _EElt15lh->addPdf ("electrons", "s9s25",       splitSignalPdfs) ;

  if(backgroundWeightSplitting.compare("fullclass")==0) {
    _EElt15lh->setSplitFrac ("hadrons", "fullclass0", piFracsEElt15[0]) ;
    _EElt15lh->setSplitFrac ("hadrons", "fullclass1", piFracsEElt15[1]) ;
    _EElt15lh->setSplitFrac ("hadrons", "fullclass2", piFracsEElt15[2]) ;
    _EElt15lh->setSplitFrac ("hadrons", "fullclass3", piFracsEElt15[3]) ;
  }
  else if(backgroundWeightSplitting.compare("class")==0) {
    _EElt15lh->setSplitFrac ("hadrons", "class0", piFracsEElt15[0]+piFracsEElt15[1]+piFracsEElt15[2]) ;
    _EElt15lh->setSplitFrac ("hadrons", "class1", piFracsEElt15[3]) ;
  }
  else {
    throw cms::Exception("BadConfig") << "Only class (non-showering / showering)"
				      << " and fullclass (golden / bigbrem / narrow / showering)" 
				      << " splitting is implemented right now";
  }


  if (m_eleIDSwitches.m_useDeltaPhiIn)   _EElt15lh->addPdf ("hadrons", "dPhiVtx",     splitBackgroundPdfs) ;  
  if (m_eleIDSwitches.m_useDeltaEtaCalo) _EElt15lh->addPdf ("hadrons", "dEtaCalo",    splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useEoverPOut)    _EElt15lh->addPdf ("hadrons", "EoPout",      splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useHoverE)       _EElt15lh->addPdf ("hadrons", "HoE",         splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useShapeFisher)  _EElt15lh->addPdf ("hadrons", "shapeFisher", splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta)  _EElt15lh->addPdf ("hadrons", "sigmaEtaEta", splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useE9overE25)    _EElt15lh->addPdf ("hadrons", "s9s25",       splitBackgroundPdfs) ;


  // ECAL ENDCAP LIKELIHOOD - Pt >= 15 GeV
  _EEgt15lh->initFromDB (calibration) ;

  _EEgt15lh->addSpecies ("electrons",eleWeight) ;
  _EEgt15lh->addSpecies ("hadrons",piWeight) ;

  if(signalWeightSplitting.compare("fullclass")==0) {
    _EEgt15lh->setSplitFrac ("electrons", "fullclass0", eleFracsEEgt15[0]) ;
    _EEgt15lh->setSplitFrac ("electrons", "fullclass1", eleFracsEEgt15[1]) ;
    _EEgt15lh->setSplitFrac ("electrons", "fullclass2", eleFracsEEgt15[2]) ;
    _EEgt15lh->setSplitFrac ("electrons", "fullclass3", eleFracsEEgt15[3]) ;
  }
  else if(signalWeightSplitting.compare("class")==0) {
    _EEgt15lh->setSplitFrac ("electrons", "class0", eleFracsEEgt15[0]+eleFracsEEgt15[1]+eleFracsEEgt15[2]) ;
    _EEgt15lh->setSplitFrac ("electrons", "class1", eleFracsEEgt15[3]) ;
  }
  else {
    throw cms::Exception("BadConfig") << "Only class (non-showering / showering)"
				      << " and fullclass (golden / bigbrem / narrow / showering)" 
				      << " splitting is implemented right now";
  }

  if (m_eleIDSwitches.m_useDeltaPhiIn)   _EEgt15lh->addPdf ("electrons", "dPhiVtx",     splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useDeltaEtaCalo) _EEgt15lh->addPdf ("electrons", "dEtaCalo",    splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useEoverPOut)    _EEgt15lh->addPdf ("electrons", "EoPout",      splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useHoverE)       _EEgt15lh->addPdf ("electrons", "HoE",         splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useShapeFisher)  _EEgt15lh->addPdf ("electrons", "shapeFisher", splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta)  _EEgt15lh->addPdf ("electrons", "sigmaEtaEta", splitSignalPdfs) ;
  if (m_eleIDSwitches.m_useE9overE25)    _EEgt15lh->addPdf ("electrons", "s9s25",       splitSignalPdfs) ;

  if(backgroundWeightSplitting.compare("fullclass")==0) {
    _EEgt15lh->setSplitFrac ("hadrons", "fullclass0", piFracsEEgt15[0]) ;
    _EEgt15lh->setSplitFrac ("hadrons", "fullclass1", piFracsEEgt15[1]) ;
    _EEgt15lh->setSplitFrac ("hadrons", "fullclass2", piFracsEEgt15[2]) ;
    _EEgt15lh->setSplitFrac ("hadrons", "fullclass3", piFracsEEgt15[3]) ;
  }
  else if(backgroundWeightSplitting.compare("class")==0) {
    _EEgt15lh->setSplitFrac ("hadrons", "class0", piFracsEEgt15[0]+piFracsEEgt15[1]+piFracsEEgt15[2]) ;
    _EEgt15lh->setSplitFrac ("hadrons", "class1", piFracsEEgt15[3]) ;
  }
  else {
    throw cms::Exception("BadConfig") << "Only class (non-showering / showering)"
				      << " and fullclass (golden / bigbrem / narrow / showering)" 
				      << " splitting is implemented right now";
  }

  // initialise the fisher coefficients from cfi
  m_fisherEBLt15 = fisherEBLt15;
  m_fisherEBGt15 = fisherEBGt15;
  m_fisherEELt15 = fisherEELt15;
  m_fisherEEGt15 = fisherEEGt15;

  if (m_eleIDSwitches.m_useDeltaPhiIn)   _EEgt15lh->addPdf ("hadrons", "dPhiVtx",     splitBackgroundPdfs) ; 
  if (m_eleIDSwitches.m_useDeltaEtaCalo) _EEgt15lh->addPdf ("hadrons", "dEtaCalo",    splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useEoverPOut)    _EEgt15lh->addPdf ("hadrons", "EoPout",      splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useHoverE)       _EEgt15lh->addPdf ("hadrons", "HoE",         splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useShapeFisher)  _EEgt15lh->addPdf ("hadrons", "shapeFisher", splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta)  _EEgt15lh->addPdf ("hadrons", "sigmaEtaEta", splitBackgroundPdfs) ;
  if (m_eleIDSwitches.m_useE9overE25)    _EEgt15lh->addPdf ("hadrons", "s9s25",       splitBackgroundPdfs) ;
}



// --------------------------------------------------------



void 
ElectronLikelihood::getInputVar (const reco::GsfElectron &electron, 
                                 std::vector<float> &measurements, 
                                 const reco::ClusterShape &sClShape) const 
{

  // the variables entering the likelihood
  if (m_eleIDSwitches.m_useDeltaPhiIn) measurements.push_back ( electron.deltaPhiSuperClusterTrackAtVtx () ) ;
  if (m_eleIDSwitches.m_useDeltaEtaCalo) measurements.push_back ( electron.deltaEtaSeedClusterTrackAtCalo () ) ;
  if (m_eleIDSwitches.m_useEoverPOut) measurements.push_back ( electron.eSeedClusterOverPout () ) ;
  if (m_eleIDSwitches.m_useHoverE) measurements.push_back ( electron.hadronicOverEm () ) ;
  if (m_eleIDSwitches.m_useShapeFisher) measurements.push_back ( CalculateFisher(electron, sClShape) ) ;
  if (m_eleIDSwitches.m_useSigmaEtaEta) measurements.push_back ( sqrt (sClShape.covEtaEta ()) );
  if (m_eleIDSwitches.m_useE9overE25)   measurements.push_back ( sClShape.e3x3 ()/sClShape.e5x5 ()) ;
}



// --------------------------------------------------------



double 
ElectronLikelihood::CalculateFisher(const reco::GsfElectron &electron,
				    const reco::ClusterShape& sClShape) const
{

  // the variables entering the shape fisher
  double s9s25, sigmaEtaEta, etaLat, a20;

  s9s25=sClShape.e3x3 ()/sClShape.e5x5 () ;
  sigmaEtaEta=sqrt (sClShape.covEtaEta ()) ;
  etaLat=sClShape.etaLat() ;
  a20=sClShape.zernike20 () ;


  // evaluate the Fisher discriminant
  double clusterShapeFisher;
  std::vector<DetId> vecId=electron.superCluster()->getHitsByDetId () ;
  EcalSubdetector subdet = EcalSubdetector (vecId[0].subdetId ()) ;
  
  if (subdet==EcalBarrel) {
    if (electron.pt()<15.) {
      clusterShapeFisher = m_fisherEBLt15[0] + 
	m_fisherEBLt15[1] * sigmaEtaEta +
	m_fisherEBLt15[2] * s9s25 +
	m_fisherEBLt15[3] * etaLat +
	m_fisherEBLt15[4] * a20;
    }
    else {
      clusterShapeFisher = m_fisherEBGt15[0] + 
	m_fisherEBGt15[1] * sigmaEtaEta +
	m_fisherEBGt15[2] * s9s25 +
	m_fisherEBGt15[3] * etaLat +
	m_fisherEBGt15[4] * a20;
    }
  }
  else if (subdet==EcalEndcap) {
    if (electron.pt()<15.) {
      clusterShapeFisher = m_fisherEELt15[0] + 
	m_fisherEELt15[1] * sigmaEtaEta +
	m_fisherEELt15[2] * s9s25 +
	m_fisherEELt15[3] * etaLat +
	m_fisherEELt15[4] * a20;
    }
    else {
      clusterShapeFisher = m_fisherEEGt15[0] + 
	m_fisherEEGt15[1] * sigmaEtaEta +
	m_fisherEEGt15[2] * s9s25 +
	m_fisherEEGt15[3] * etaLat +
	m_fisherEEGt15[4] * a20;
    }
  }
  else {
    clusterShapeFisher = -999 ;
    edm::LogWarning ("ElectronLikelihood") << "Undefined electron, eta = " << electron.eta () << "!" ;
  }
  return clusterShapeFisher;
}



// --------------------------------------------------------



float 
ElectronLikelihood::result (const reco::GsfElectron &electron, 
                            const reco::ClusterShape &sClShape) const 
{

  //=======================================================
  // used classification:
  // golden                     =>  0
  // big brem                   => 10
  // narrow                     => 20
  // showering nbrem 0          => 30
  // showering nbrem 1          => 31
  // showering nbrem 2          => 32
  // showering nbrem 3          => 33
  // showering nbrem 4 ou plus  => 34
  // cracks                     => 40
  // endcap                     => barrel + 100
  //=======================================================

  std::vector<float> measurements ;
  getInputVar (electron, measurements, sClShape) ;

  // Split using only the 10^1 bit (golden/big brem/narrow/showering)
  int bitVal=-1 ;
  int gsfclass=electron.classification () ;
  if (gsfclass<99) 
    bitVal=int (gsfclass)/10 ; 
  else
    bitVal=int (int (gsfclass)%100)/10 ;
  
  // temporary: crack electrons goes into Class3 (showering)
  if (bitVal==4) bitVal=3 ;
  if (bitVal<0 || bitVal>3)
    throw cms::Exception ("ElectronLikelihood") << "ERROR! electron class " << gsfclass << " is not accepted!\n" ;

  char className[20] ;
  if(m_signalWeightSplitting.compare("fullclass")==0) {
    sprintf (className,"fullclass%d",bitVal) ;
  }
  else if(m_signalWeightSplitting.compare("class")==0) {
    int classVal = (bitVal<3) ? 0 : 1;
    sprintf (className,"class%d",classVal);
  }
  else {
    throw cms::Exception("BadConfig") << "Only class (non-showering / showering)"
				      << " and fullclass (golden / bigbrem / narrow / showering)" 
				      << " splitting is implemented right now";
  }

  reco::SuperClusterRef sclusRef = electron.superCluster() ;
  std::vector<DetId> vecId=sclusRef->getHitsByDetId () ;
  EcalSubdetector subdet = EcalSubdetector (vecId[0].subdetId ()) ;
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

