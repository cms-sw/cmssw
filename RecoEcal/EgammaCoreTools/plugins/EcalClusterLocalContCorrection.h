#ifndef RecoEcal_EgammaCoreTools_EcalClusterLocalContCorrection_h
#define RecoEcal_EgammaCoreTools_EcalClusterLocalContCorrection_h

/** \class EcalClusterLocalContCorrection
  *  Function to correct em object energy for energy not contained in a 5x5 crystal area in the calorimeter
  *
  *  $Id: EcalClusterLocalContCorrection.h
  *  $Date:
  *  $Revision:
  *  \author Federico Ferri, CEA Saclay, November 2008
  */

#include "RecoEcal/EgammaCoreTools/plugins/EcalClusterLocalContCorrectionBaseClass.h"

class EcalClusterLocalContCorrection : public EcalClusterLocalContCorrectionBaseClass {
        public:
                EcalClusterLocalContCorrection( const edm::ParameterSet &) {};
                // compute the correction
                virtual float getValue( const reco::BasicCluster &, const EcalRecHitCollection & ) const;
                virtual float getValue( const reco::SuperCluster &, const int mode ) const;
};

#endif
