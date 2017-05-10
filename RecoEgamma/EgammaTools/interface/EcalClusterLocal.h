#ifndef EGAMMATOOLS_EcalClusterLocal_h
#define EGAMMATOOLS_EcalClusterLocal_h

/** \class EcalClusterLocal
  *  Function to compute local coordinates of Ecal clusters
  *  (adapted from RecoEcal/EgammaCoreTools/plugins/EcalClusterLocal)
  *  $Id: EcalClusterLocal.h
  *  $Date:
  *  $Revision:
  *  \author Josh Bendavid, MIT, 2011
  */


//#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
//#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

class CaloGeometry; 

class EcalClusterLocal {
        public:
          EcalClusterLocal();
          ~EcalClusterLocal();
          
          void localCoordsEB( const reco::CaloCluster &bclus, const edm::EventSetup &es, float &etacry, float &phicry, int &ieta, int &iphi, float &thetatilt, float &phitilt) const;
          void localCoordsEE( const reco::CaloCluster &bclus, const edm::EventSetup &es, float &xcry, float &ycry, int &ix, int &iy, float &thetatilt, float &phitilt) const;
	  
	  void localCoordsEB( const reco::CaloCluster &bclus, const CaloGeometry & geom, float &etacry, float &phicry, int &ieta, int &iphi, float &thetatilt, float &phitilt) const;
          void localCoordsEE( const reco::CaloCluster &bclus, const CaloGeometry & geom, float &xcry, float &ycry, int &ix, int &iy, float &thetatilt, float &phitilt) const;

		 
};

#endif
