#ifndef ElectronIDSelectorLikelihood_h
#define ElectronIDSelectorLikelihood_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoEgamma/ElectronIdentification/interface/ElectronLikelihood.h"

class ElectronIDSelectorLikelihood
{
 public:

  explicit ElectronIDSelectorLikelihood (const edm::ParameterSet& conf) ;
  virtual ~ElectronIDSelectorLikelihood () ;

  void newEvent (const edm::Event& e, const edm::EventSetup& c) ;
  double operator() (const reco::GsfElectron & electron, const edm::Event & event) ;
   
 private:
  
  edm::ESHandle<ElectronLikelihood> likelihoodAlgo_ ;
  
  edm::ParameterSet conf_;
  
  edm::InputTag barrelClusterShapeAssociation_;
  edm::InputTag endcapClusterShapeAssociation_;

  bool doLikelihood_;

};

#endif
