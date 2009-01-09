#ifndef ZEEKINEMATICTOOLS_H
#define  ZEEKINEMATICTOOLS_H

#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "Calibration/Tools/interface/HouseholderDecomposition.h"
#include "Calibration/Tools/interface/MinL3Algorithm.h"
#include "Calibration/Tools/interface/CalibrationCluster.h"
#include "Calibration/Tools/interface/CalibElectron.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"


// class declaration
//

class ZeeKinematicTools {

   public:

      ZeeKinematicTools();
      ~ZeeKinematicTools();

      static float calculateZMass_noTK(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate);
      static float calculateZMass_withTK(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate);
      static float calculateZEta(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate);
      static float calculateZTheta(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate);
      static float calculateZRapidity(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate);
      static float calculateZPhi(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate);
      static float calculateZPt(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate);

      static float calculateZMassWithCorrectedElectrons_noTK(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate, float ele1EnergyCorrection, float ele2EnergyCorrection);
      static float calculateZMassWithCorrectedElectrons_withTK(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate, float ele1EnergyCorrection, float ele2EnergyCorrection);

      static float cosThetaElectrons_SC( const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate, float ele1EnergyCorrection, float ele2EnergyCorrection );
      static float cosThetaElectrons_TK( const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate, float ele1EnergyCorrection, float ele2EnergyCorrection );



};
#endif
