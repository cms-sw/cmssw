#ifndef PhotonEnergyCorrector_H
#define PhotonEnergyCorrector_H
/** \class PhotonEnergyCorrector
 **  
 **
 **  $Id: PhotonEnergyCorrector.h,v 1.1 2011/11/02 19:03:08 nancy Exp $ 
 **  $Date: 2011/11/02 19:03:08 $ 
 **  $Revision: 1.1 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "RecoEgamma/EgammaTools/interface/EGEnergyCorrector.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h" 
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h"


#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"



class PhotonEnergyCorrector
 {
  public:

   PhotonEnergyCorrector(const edm::ParameterSet& config);
   ~PhotonEnergyCorrector();

   void init(const edm::EventSetup& theEventSetup );
   void calculate( reco::Photon &, int subdet ) ;


  private:
   bool weightsfromDB_;
   std::string w_file_;
   std::string w_db_;
   std::string candidateP4type_; 
   EGEnergyCorrector*       regressionCorrector_;
   EcalClusterFunctionBaseClass * scEnergyFunction_;
   EcalClusterFunctionBaseClass * scCrackEnergyFunction_;
   EcalClusterFunctionBaseClass * scEnergyErrorFunction_;
   EcalClusterFunctionBaseClass * photonEcalEnergyCorrFunction_;
   double minR9Barrel_;
   double minR9Endcap_;
   edm::ESHandle<CaloGeometry> theCaloGeom_; 
   
 } ;

#endif




