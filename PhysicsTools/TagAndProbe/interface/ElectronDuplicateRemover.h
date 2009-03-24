#ifndef PhysicsTools_TagAndProbe_ElectronDuplicateRemover_h
#define PhysicsTools_TagAndProbe_ElectronDuplicateRemover_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


// forward declarations

class ElectronDuplicateRemover : public edm::EDProducer 
{
 public:
  explicit ElectronDuplicateRemover(const edm::ParameterSet&);
  ~ElectronDuplicateRemover();

 private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
      
  // ----------member data ---------------------------
      
  std::string _inputProducer;
  double _BarrelMaxEta;
  double _EndcapMinEta;
  double _EndcapMaxEta;
  double _ptMin;
  double _ptMax;
};

#endif
