#ifndef PhotonEnergyCorrector_H
#define PhotonEnergyCorrector_H
/** \class PhotonEnergyCorrector
 **  
 **
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

#include "RecoEgamma/EgammaTools/interface/BaselinePFSCRegression.h"

class PhotonEnergyCorrector
 {
  public:

   PhotonEnergyCorrector(const edm::ParameterSet& config, edm::ConsumesCollector && iC);
   ~PhotonEnergyCorrector();

   std::unique_ptr<PFSCRegressionCalc>& gedRegression() 
     { return gedRegression_; }
   
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
   std::unique_ptr<PFSCRegressionCalc> gedRegression_;
   double minR9Barrel_;
   double minR9Endcap_;
   edm::ESHandle<CaloGeometry> theCaloGeom_;
   edm::InputTag barrelEcalHits_;
   edm::InputTag endcapEcalHits_;
   edm::EDGetTokenT<EcalRecHitCollection> barrelEcalHitsToken_;
   edm::EDGetTokenT<EcalRecHitCollection> endcapEcalHitsToken_;

   EnergyUncertaintyPhotonSpecific* photonUncertaintyCalculator_;
   
 } ;

#endif




