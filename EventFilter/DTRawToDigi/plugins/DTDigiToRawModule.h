#ifndef EventFilter_DTDigiToRawModule_h
#define EventFilter_DTDigiToRawModule_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"


class DTDigiToRaw;

class DTDigiToRawModule : public edm::stream::EDProducer<> {
public:
  /// Constructor
  DTDigiToRawModule(const edm::ParameterSet& pset);

  /// Destructor
  ~DTDigiToRawModule() override;

  // Operations
  void produce( edm::Event&, const edm::EventSetup& ) override;

private:
  DTDigiToRaw * packer;
  
  int dduID;
  bool debug;
  edm::EDGetTokenT<DTDigiCollection> digicoll;
  
  bool useStandardFEDid_;
  int minFEDid_;
  int maxFEDid_;

};
#endif

