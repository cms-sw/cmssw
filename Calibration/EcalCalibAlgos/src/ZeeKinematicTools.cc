
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "Calibration/Tools/interface/calibXMLwriter.h"
#include "Calibration/Tools/interface/CalibrationCluster.h"
#include "Calibration/Tools/interface/HouseholderDecomposition.h"
#include "Calibration/Tools/interface/MinL3Algorithm.h"
#include "Calibration/EcalCalibAlgos/interface/ZeeKinematicTools.h"
#include "Calibration/Tools/interface/ZIterativeAlgorithmWithFit.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TRandom.h"


#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

ZeeKinematicTools::ZeeKinematicTools(){}

ZeeKinematicTools::~ZeeKinematicTools(){}


//--------------------------------------------


 float ZeeKinematicTools::cosThetaElectrons_SC( const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate, float ele1EnergyCorrection, float ele2EnergyCorrection ){
  
  float theta1 = 2. * atan( exp(- aZCandidate.first->getRecoElectron()->superCluster()->eta()) );
  float phi1 = aZCandidate.first->getRecoElectron()->superCluster()->phi();

  float x1 = aZCandidate.first->getRecoElectron()->superCluster()->energy() * sin(theta1) * cos ( phi1 );
  float y1 = aZCandidate.first->getRecoElectron()->superCluster()->energy() * sin(theta1) * sin ( phi1 );
  float z1 = aZCandidate.first->getRecoElectron()->superCluster()->energy() * cos(theta1);
  float mod1 = sqrt( x1*x1 + y1*y1 + z1*z1 );

  float theta2 = 2. * atan( exp(- aZCandidate.second->getRecoElectron()->superCluster()->eta()) );
  float phi2 = aZCandidate.second->getRecoElectron()->superCluster()->phi();

  float x2 = aZCandidate.second->getRecoElectron()->superCluster()->energy() * sin(theta2) * cos ( phi2 );
  float y2 = aZCandidate.second->getRecoElectron()->superCluster()->energy() * sin(theta2) * sin ( phi2 );
  float z2 = aZCandidate.second->getRecoElectron()->superCluster()->energy() * cos(theta2);
  float mod2 = sqrt( x2*x2 + y2*y2 + z2*z2 );

  return (x1*x2 + y1*y2 + z1*z2)/( mod1* mod2 );

  
}

//--------------------------------------------

 float ZeeKinematicTools::cosThetaElectrons_TK( const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate, float ele1EnergyCorrection, float ele2EnergyCorrection ){
  
  float theta1 = 2. * atan( exp(- aZCandidate.first->getRecoElectron()->eta()) );
  float phi1 = aZCandidate.first->getRecoElectron()->phi();
  
  float x1 = aZCandidate.first->getRecoElectron()->superCluster()->energy() * sin(theta1) * cos ( phi1 ); 
  float y1 = aZCandidate.first->getRecoElectron()->superCluster()->energy() * sin(theta1) * sin ( phi1 );
  float z1 = aZCandidate.first->getRecoElectron()->superCluster()->energy() * cos(theta1);
  float mod1 = sqrt( x1*x1 + y1*y1 + z1*z1 );

  float theta2 = 2. * atan( exp(- aZCandidate.second->getRecoElectron()->eta()) );
  float phi2 = aZCandidate.second->getRecoElectron()->phi();

  float x2 = aZCandidate.second->getRecoElectron()->superCluster()->energy() * sin(theta2) * cos ( phi2 );
  float y2 = aZCandidate.second->getRecoElectron()->superCluster()->energy() * sin(theta2) * sin ( phi2 );
  float z2 = aZCandidate.second->getRecoElectron()->superCluster()->energy() * cos(theta2);
  float mod2 = sqrt( x2*x2 + y2*y2 + z2*z2 );

  return (x1*x2 + y1*y2 + z1*z2)/( mod1* mod2 );
}

//--------------------------------------------

 float ZeeKinematicTools::calculateZMassWithCorrectedElectrons_noTK(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate, float ele1EnergyCorrection, float ele2EnergyCorrection)
{
  return ZIterativeAlgorithmWithFit::invMassCalc(aZCandidate.first->getRecoElectron()->superCluster()->energy() / ele1EnergyCorrection, aZCandidate.first->getRecoElectron()->superCluster()->eta(), aZCandidate.first->getRecoElectron()->superCluster()->phi(), aZCandidate.second->getRecoElectron()->superCluster()->energy() / ele2EnergyCorrection, aZCandidate.second->getRecoElectron()->superCluster()->eta(), aZCandidate.second->getRecoElectron()->superCluster()->phi());

}

//--------------------------------------------

 float ZeeKinematicTools::calculateZMassWithCorrectedElectrons_withTK(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate, float ele1EnergyCorrection, float ele2EnergyCorrection)
{
  return ZIterativeAlgorithmWithFit::invMassCalc(aZCandidate.first->getRecoElectron()->superCluster()->energy() / ele1EnergyCorrection, aZCandidate.first->getRecoElectron()->eta(), aZCandidate.first->getRecoElectron()->phi(), aZCandidate.second->getRecoElectron()->superCluster()->energy() / ele2EnergyCorrection, aZCandidate.second->getRecoElectron()->eta(), aZCandidate.second->getRecoElectron()->phi());

}

//--------------------------------------------

 float ZeeKinematicTools::calculateZMass_noTK(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate)
{

  return  ZIterativeAlgorithmWithFit::invMassCalc(aZCandidate.first->getRecoElectron()->superCluster()->energy(), aZCandidate.first->getRecoElectron()->superCluster()->eta(), aZCandidate.first->getRecoElectron()->superCluster()->phi(), aZCandidate.second->getRecoElectron()->superCluster()->energy(), aZCandidate.second->getRecoElectron()->superCluster()->eta(), aZCandidate.second->getRecoElectron()->superCluster()->phi());  

}

//--------------------------------------------

 float ZeeKinematicTools::calculateZMass_withTK(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate)
{

  return  ZIterativeAlgorithmWithFit::invMassCalc(aZCandidate.first->getRecoElectron()->superCluster()->energy(), aZCandidate.first->getRecoElectron()->eta(), aZCandidate.first->getRecoElectron()->phi(), aZCandidate.second->getRecoElectron()->superCluster()->energy(), aZCandidate.second->getRecoElectron()->eta(), aZCandidate.second->getRecoElectron()->phi());  

}

//--------------------------------------------

 float ZeeKinematicTools::calculateZRapidity(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate)
{

  TLorentzVector ele1LV( aZCandidate.first->getRecoElectron()->px(), aZCandidate.first->getRecoElectron()->py(), aZCandidate.first->getRecoElectron()->pz(), aZCandidate.first->getRecoElectron()->superCluster()->energy());

  TLorentzVector ele2LV( aZCandidate.second->getRecoElectron()->px(), aZCandidate.second->getRecoElectron()->py(), aZCandidate.second->getRecoElectron()->pz(), aZCandidate.second->getRecoElectron()->superCluster()->energy());
  

  return  (ele1LV + ele2LV).Rapidity();

}

//--------------------------------------------

 float ZeeKinematicTools::calculateZEta(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate)
{

  TLorentzVector ele1LV( aZCandidate.first->getRecoElectron()->px(), aZCandidate.first->getRecoElectron()->py(), aZCandidate.first->getRecoElectron()->pz(), aZCandidate.first->getRecoElectron()->superCluster()->energy());

  TLorentzVector ele2LV( aZCandidate.second->getRecoElectron()->px(), aZCandidate.second->getRecoElectron()->py(), aZCandidate.second->getRecoElectron()->pz(), aZCandidate.second->getRecoElectron()->superCluster()->energy());
  
  return  (ele1LV + ele2LV).Eta();
  
}

//--------------------------------------------

 float ZeeKinematicTools::calculateZTheta(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate)
{

  TLorentzVector ele1LV( aZCandidate.first->getRecoElectron()->px(), aZCandidate.first->getRecoElectron()->py(), aZCandidate.first->getRecoElectron()->pz(), aZCandidate.first->getRecoElectron()->superCluster()->energy());

  TLorentzVector ele2LV( aZCandidate.second->getRecoElectron()->px(), aZCandidate.second->getRecoElectron()->py(), aZCandidate.second->getRecoElectron()->pz(), aZCandidate.second->getRecoElectron()->superCluster()->energy());
  
  return  (ele1LV + ele2LV).Theta();
  
}

//--------------------------------------------

 float ZeeKinematicTools::calculateZPhi(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate)
{

  TLorentzVector ele1LV( aZCandidate.first->getRecoElectron()->px(), aZCandidate.first->getRecoElectron()->py(), aZCandidate.first->getRecoElectron()->pz(), aZCandidate.first->getRecoElectron()->superCluster()->energy());

  TLorentzVector ele2LV( aZCandidate.second->getRecoElectron()->px(), aZCandidate.second->getRecoElectron()->py(), aZCandidate.second->getRecoElectron()->pz(), aZCandidate.second->getRecoElectron()->superCluster()->energy());
  
  return  (ele1LV + ele2LV).Phi();

}

//--------------------------------------------

 float ZeeKinematicTools::calculateZPt(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate)
{

  TLorentzVector ele1LV( aZCandidate.first->getRecoElectron()->px(), aZCandidate.first->getRecoElectron()->py(), aZCandidate.first->getRecoElectron()->pz(), aZCandidate.first->getRecoElectron()->superCluster()->energy());

  TLorentzVector ele2LV( aZCandidate.second->getRecoElectron()->px(), aZCandidate.second->getRecoElectron()->py(), aZCandidate.second->getRecoElectron()->pz(), aZCandidate.second->getRecoElectron()->superCluster()->energy());
  
  return  (ele1LV + ele2LV).Pt();

}



