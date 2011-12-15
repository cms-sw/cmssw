#ifndef PhotonEnergyCorrector_H
#define PhotonEnergyCorrector_H
/** \class PhotonEnergyCorrector
 **  
 **
 **  $Id: PhotonEnergyCorrector.h,v 1.2 2011/11/24 18:13:56 nancy Exp $ 
 **  $Date: 2011/11/24 18:13:56 $ 
 **  $Revision: 1.2 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/
#include "FWCore/Framework/interface/Event.h"
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
   void calculate( edm::Event& evt, reco::Photon &, int subdet,const reco::VertexCollection& vtxcol,const edm::EventSetup& iSetup) ;


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




