//
// $Id: CaloIsolationEnergy.h,v 1.1 2008/01/07 11:48:26 lowette Exp $
//

#ifndef PhysicsTools_PatUtils_CaloIsolationEnergy_h
#define PhysicsTools_PatUtils_CaloIsolationEnergy_h

/**
  \class    CaloIsolationEnergy CaloIsolationEnergy.h "PhysicsTools/PatUtils/interface/CaloIsolationEnergy.h"
  \brief    Calculates a lepton's calorimetric isolation energy

   CaloIsolationEnergy calculates a calorimetric isolation energy in
   a half-cone (dependent on the lepton's charge) around the lepton's impact
   position on the ECAL surface, as defined in CMS Note 2006/024

  \author   Steven Lowette
  \version  $Id: CaloIsolationEnergy.h,v 1.1 2008/01/07 11:48:26 lowette Exp $
*/


#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

#include "DataFormats/TrackReco/interface/Track.h"


class MagneticField;
class TrackToEcalPropagator;


namespace pat {


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

