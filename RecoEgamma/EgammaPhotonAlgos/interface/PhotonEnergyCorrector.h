#ifndef PhotonEnergyCorrector_H
#define PhotonEnergyCorrector_H
/** \class PhotonEnergyCorrector
 **  
 **
 **  $Id: PhotonEnergyCorrector.h,v 1.7 2012/03/26 14:38:40 nancy Exp $ 
 **  $Date: 2012/03/26 14:38:40 $ 
 **  $Revision: 1.7 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "RecoEgamma/EgammaTools/interface/EGEnergyCorrector.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h" 
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "RecoEgamma/EgammaPhotonAlgos/interface/EnergyUncertaintyPhotonSpecific.h"
#include <iostream>

class PhotonEnergyCorrector
 {
  public:

   PhotonEnergyCorrector(const edm::ParameterSet& config);
   ~PhotonEnergyCorrector();

   void init(const edm::EventSetup& theEventSetup );
   void calculate( edm::Event& evt, reco::Photon &, int subdet,const reco::VertexCollection& vtxcol,const edm::EventSetup& iSetup) ;
   double applyCrackCorrection(const reco::SuperCluster &cl, EcalClusterFunctionBaseClass* crackCorrectionFunction);

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
   edm::InputTag barrelEcalHits_;
   edm::InputTag endcapEcalHits_;

   EnergyUncertaintyPhotonSpecific* photonUncertaintyCalculator_;
   
 } ;

#endif




