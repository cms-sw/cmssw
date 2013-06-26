#ifndef EnergyUncertaintyPhotonSpecific_H
#define EnergyUncertaintyPhotonSpecific_H

/** \class EnergyUncertaintyPhotonSpecific
 **  
 **  \author Nicolas Chanon, ETH Zurich, Switzerland
 **
 ***/

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class EnergyUncertaintyPhotonSpecific
 {
  public:

   EnergyUncertaintyPhotonSpecific( const edm::ParameterSet& config);
   ~EnergyUncertaintyPhotonSpecific();

   void init(const edm::EventSetup& theEventSetup );
   //void calculate( edm::Event& evt, reco::Photon &, int subdet,const reco::VertexCollection& vtxcol,const edm::EventSetup& iSetup) ;
   //double applyCrackCorrection(const reco::SuperCluster &cl, EcalClusterFunctionBaseClass* crackCorrectionFunction);

   double computePhotonEnergyUncertainty_lowR9(double eta, double brem, double energy);
   double computePhotonEnergyUncertainty_highR9(double eta, double brem, double energy);
   

  private:
 
   
 } ;

#endif
