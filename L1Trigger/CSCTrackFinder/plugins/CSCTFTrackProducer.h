#ifndef CSCTrackFinder_CSCTFTrackProducer_h
#define CSCTrackFinder_CSCTFTrackProducer_h

#include <string>

#include <FWCore/Framework/interface/EDProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/ParameterSet/interface/InputTag.h>
#include <FWCore/Framework/interface/EventSetup.h>

class CSCTFTrackBuilder;

class CSCTFTrackProducer : public edm::EDProducer
{
 public:

  CSCTFTrackProducer(const edm::ParameterSet&);

  virtual ~CSCTFTrackProducer();

  void produce(edm::Event & e, const edm::EventSetup& c);

  void beginJob(const edm::EventSetup& es);

 private:

  bool useDT, TMB07;
  edm::InputTag input_module, dt_producer;
  edm::ParameterSet sp_pset ;
  unsigned long long m_scalesCacheID ;
  unsigned long long m_ptScaleCacheID ;
  CSCTFTrackBuilder* my_builder;
};

#endif
