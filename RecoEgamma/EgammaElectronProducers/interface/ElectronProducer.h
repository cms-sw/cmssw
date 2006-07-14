#ifndef RecoEgamma_EgammaElectronProducers_ElectronProducer_h
#define RecoEgamma_EgammaElectronProducers_ElectronProducer_h
/** \class ElectronProducer
 **  
 **
 **  $Id: ElectronProducer.h,v 1.2 2006/07/10 18:01:13 nancy Exp $ 
 **  $Date: 2006/07/10 18:01:13 $ 
 **  $Revision: 1.2 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "MagneticField/Engine/interface/MagneticField.h"



// ElectronProducer inherits from EDProducer, so it can be a module:
class ElectronProducer : public edm::EDProducer {

 public:

  ElectronProducer (const edm::ParameterSet& ps);
  ~ElectronProducer();


  virtual void beginJob (edm::EventSetup const & es);
  virtual void produce(edm::Event& evt, const edm::EventSetup& es);

 private:

  

  
  std::string ElectronCollection_;
  std::string scBarrelProducer_;
  std::string scEndcapProducer_;
  std::string scBarrelCollection_;
  std::string scEndcapCollection_;
  edm::ParameterSet conf_;


  edm::ESHandle<MagneticField> theMF_;


};
#endif
