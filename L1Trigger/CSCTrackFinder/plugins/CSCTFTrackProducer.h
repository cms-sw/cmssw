#ifndef CSCTrackFinder_CSCTFTrackProducer_h
#define CSCTrackFinder_CSCTFTrackProducer_h

#include <string>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "L1Trigger/CSCTrackFinder/src/CSCTFDTReceiver.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

class CSCTFTrackBuilder;
class L1MuDTChambPhContainer;
template<typename T> class CSCTriggerContainer;
namespace csctf {
  class TrackStub;
}

class CSCTFTrackProducer : public edm::EDProducer
{
 public:
  CSCTFTrackProducer(const edm::ParameterSet&);
  virtual ~CSCTFTrackProducer();
  void produce(edm::Event & e, const edm::EventSetup& c);
  void beginJob();

 private:
  CSCTFDTReceiver* my_dtrc;
  bool useDT, TMB07, readDtDirect;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> input_module;
  edm::EDGetTokenT<L1MuDTChambPhContainer> dt_producer;
  edm::EDGetTokenT<CSCTriggerContainer<csctf::TrackStub> >  directProd;
  edm::ParameterSet sp_pset ;
  unsigned long long m_scalesCacheID ;
  unsigned long long m_ptScaleCacheID ;
  CSCTFTrackBuilder* my_builder;
};

#endif
