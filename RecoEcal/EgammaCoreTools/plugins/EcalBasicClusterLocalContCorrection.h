#ifndef RecoEcal_EgammaCoreTools_EcalBasicClusterLocalContCorrection_h
#define RecoEcal_EgammaCoreTools_EcalBasicClusterLocalContCorrection_h

/** \class EcalBasicClusterLocalContCorrection
  *  Function to correct em object energy for energy not contained in a 5x5 crystal area in the calorimeter
  *
  *  $Id: EcalBasicClusterLocalContCorrection.h
  *  $Date:
  *  $Revision:
  *  \author Federico Ferri, CEA Saclay, November 2008
  */

#include "RecoEcal/EgammaCoreTools/plugins/EcalClusterLocalContCorrectionBaseClass.h"

class EcalBasicClusterLocalContCorrection : public EcalClusterLocalContCorrectionBaseClass {
 public:
  EcalBasicClusterLocalContCorrection( const edm::ParameterSet &) {};
  // compute the correction
  //virtual float getValue( const reco::BasicCluster &, const EcalRecHitCollection & ) const;
  //virtual float getValue( const reco::BasicCluster & basicCluster) const;
  virtual float getValue( const reco::BasicCluster &, const EcalRecHitCollection & ) const;
  virtual float getValue( const reco::SuperCluster &, const int mode ) const;
 private:	
  int getEcalModule(DetId id) const;
    

};

#endif
