#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

class VertexFilter : public edm::EDProducer {
 public:
  explicit VertexFilter(const edm::ParameterSet&);
  ~VertexFilter();
  
 private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  // ----------member data ---------------------------
 private:
  //edm::ParameterSet conf_;
  float dist;
  edm::InputTag tkTag; 
  edm::InputTag vtxTag;
  
  unsigned minhits;
  float distz;
  float distrho;
  float chi_cut;

  
};
