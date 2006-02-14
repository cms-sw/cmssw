#ifndef RecoJets_JetableObjectHelper_h
#define RecoJets_JetableObjectHelper_h
/** \class JetableObjectHelper
 *
 * Class to provide methods to access calorimetry information,
 * primarily for jet algorithms.
 *
 * \author Robert M Harris, Fermilab
 *
 * \version   1st Version Feb. 28, 2005  Methods needed by jet algorithm for EDM Demo.
 *                   RMH, Mar. 24, 2005 Added unpackTowerIndex and caloTowerGrid and
 *                                      medthod getTower to fetch towers using eta-phi grid,
 *                                      getTowerCenter as temporary method of getting the eta
 *                                      and phi value of the tower centers on the eta-phi grid. 
 *                  RMH, Apr 20, 2005   Add getNearestTower method.
 *                  RMH, Oct 19, 2005   Modified to work with real CaloTowers from Jeremy Mans
 *
 ************************************************************/
 //
 //
#include "PhysicsTools/Candidate/interface/CandidateFwd.h"
#include "FWCore/Framework/interface/Handle.h"
#include "PhysicsTools/Candidate/interface/CandidateFwd.h"

class JetableObjectHelper {
public:
  typedef edm::Handle<aod::CandidateCollection> Handle;
  /// Constructor takes a const pointer to the CaloTowerCollection we will help with.
  JetableObjectHelper( const Handle & ctcp ) : caloTowerCollPointer(ctcp) { }  
  /// etOrderedCaloTowers returns an Et order list of pointers to CaloTowers with Et>etTreshold
  aod::CandidateRefs etOrderedCaloTowers(double etThreshold) const;
  
  /// towersWithinCone returns a list of pointers to CaloTowers with Et>etThreshold within coneRadius
  /// in eta-phi space of the coneEta and conePhi.
  aod::CandidateRefs towersWithinCone(double coneEta, double conePhi, double coneRadius, double etEthreshold);
  
  /// phidif calculates the difference between phi1 and phi2 taking into account the 2pi issue.
  double phidif(double phi1, double phi2);
  
private:
   Handle caloTowerCollPointer;   

};
#endif
