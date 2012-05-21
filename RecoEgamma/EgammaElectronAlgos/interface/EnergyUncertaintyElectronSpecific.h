#ifndef EnergyUncertaintyElectronSpecific_H
#define EnergyUncertaintyElectronSpecific_H

/** \class EnergyUncertaintyElectronSpecific
 **
 **  \author Anne-Fleur Barfuss, Kansas State University
 **
 ***/

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

class EnergyUncertaintyElectronSpecific
 {
  public:

   //EnergyUncertaintyElectronSpecific( const edm::ParameterSet& config);
   EnergyUncertaintyElectronSpecific();
   ~EnergyUncertaintyElectronSpecific();

   void init(const edm::EventSetup& theEventSetup );
   //void calculate( edm::Event& evt, reco::Electron &, int subdet,const reco::VertexCollection& vtxcol,const edm::EventSetup& iSetup) ;
   //double applyCrackCorrection(const reco::SuperCluster &cl, EcalClusterFunctionBaseClass* crackCorrectionFunction);

   double computeElectronEnergyUncertainty( reco::GsfElectron::Classification c, double eta, double brem, double energy);

  private:

   double computeElectronEnergyUncertainty_golden(double eta, double brem, double energy);
   double computeElectronEnergyUncertainty_bigbrem(double eta, double brem, double energy);
   double computeElectronEnergyUncertainty_showering(double eta, double brem, double energy);
   double computeElectronEnergyUncertainty_cracks(double eta, double brem, double energy);
   double computeElectronEnergyUncertainty_badtrack(double eta, double brem, double energy);


 } ;

#endif
