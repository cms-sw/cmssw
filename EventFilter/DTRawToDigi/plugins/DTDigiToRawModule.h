#ifndef EventFilter_DTDigiToRawModule_h
#define EventFilter_DTDigiToRawModule_h


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"



class DTDigiToRaw;

class DTDigiToRawModule : public edm::EDProducer {
public:
  /// Constructor
  DTDigiToRawModule(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTDigiToRawModule();

  // Operations
  virtual void produce( edm::Event&, const edm::EventSetup& );

private:
  DTDigiToRaw * packer;
  
  int dduID;
  bool debug;
  edm::InputTag digicoll;
  
  bool useStandardFEDid_;
  int minFEDid_;
  int maxFEDid_;

};
#endif

