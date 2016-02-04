//
// $Id: CaloIsolationEnergy.h,v 1.3 2008/03/05 14:51:02 fronga Exp $
//

#ifndef PhysicsTools_PatUtils_CaloIsolationEnergy_h
#define PhysicsTools_PatUtils_CaloIsolationEnergy_h

/**
  \class    pat::CaloIsolationEnergy CaloIsolationEnergy.h "PhysicsTools/PatUtils/interface/CaloIsolationEnergy.h"
  \brief    Calculates a lepton's calorimetric isolation energy

   CaloIsolationEnergy calculates a calorimetric isolation energy in
   a half-cone (dependent on the lepton's charge) around the lepton's impact
   position on the ECAL surface, as defined in CMS Note 2006/024

  \author   Steven Lowette
  \version  $Id: CaloIsolationEnergy.h,v 1.3 2008/03/05 14:51:02 fronga Exp $
*/
#include <vector>

class MagneticField;
class TrackToEcalPropagator;
class CaloTower;

namespace reco {
  class Track; 
}

namespace pat {
  class Muon;
  class Electron;

  class CaloIsolationEnergy {
    public:
      CaloIsolationEnergy();
      virtual ~CaloIsolationEnergy();

      float calculate(const Electron & anElectron, const std::vector<CaloTower> & theTowers, float isoConeElectron = 0.3) const;
      float calculate(const Muon & aMuon, const std::vector<CaloTower> & theTowers, float isoConeMuon = 0.3) const;

    private:
      float calculate(const reco::Track & track, const float leptonEnergy, const std::vector<CaloTower> & theTowers, float isoCone) const;
  };

}

#endif

