#ifndef PhysicsTools_TagAndProbe_gsfCategoryProducer_h
#define PhysicsTools_TagAndProbe_gsfCategoryProducer_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


// forward declarations

class gsfCategoryProducer : public edm::EDProducer 
{
 public:
  explicit gsfCategoryProducer(const edm::ParameterSet&);
  ~gsfCategoryProducer();

 private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
      
  // ----------member data ---------------------------
      
  std::string _inputProducer;
  std::string _gsfCategory;
  bool _isInBarrel;
  bool _isInEndCap;
  bool _isInCrack;
  int componentsBit;
  enum Component { Golden = 1,
		   Bigbrem =2,
		   Narrow = 4,
		   Showering = 8}; 
};

#endif
