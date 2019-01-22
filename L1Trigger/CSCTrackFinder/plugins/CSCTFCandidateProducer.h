#ifndef CSCTrackFinder_CSCTFCandidateProducer_h
#define CSCTrackFinder_CSCTFCandidateProducer_h

#include <string>
#include <vector>

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "L1Trigger/CSCTrackFinder/src/CSCTFCandidateBuilder.h"

class L1MuRegionalCand;

class CSCTFCandidateProducer : public edm::global::EDProducer<>
{
 public:

  explicit CSCTFCandidateProducer(const edm::ParameterSet&);

  void produce(edm::StreamID, edm::Event & e, const edm::EventSetup& c) const override;

 private:
  const edm::EDGetTokenT<L1CSCTrackCollection> input_module;
  const edm::EDPutTokenT<std::vector<L1MuRegionalCand>> putToken_;
  const CSCTFCandidateBuilder my_builder;
};

#endif
