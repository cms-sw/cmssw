#ifndef CSCTrackFinder_CSCTFCandidateProducer_h
#define CSCTrackFinder_CSCTFCandidateProducer_h

#include <string>

#include <FWCore/Framework/interface/EDProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Utilities/interface/InputTag.h>
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"

class CSCTFCandidateBuilder;

class CSCTFCandidateProducer : public edm::EDProducer
{
 public:

  CSCTFCandidateProducer(const edm::ParameterSet&);

  virtual ~CSCTFCandidateProducer();

  void produce(edm::Event & e, const edm::EventSetup& c);

 private:
  edm::EDGetTokenT<L1CSCTrackCollection> input_module;
  CSCTFCandidateBuilder* my_builder;
};

#endif
