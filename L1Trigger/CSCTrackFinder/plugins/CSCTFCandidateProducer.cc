#include "CSCTFCandidateProducer.h"

#include <vector>
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

CSCTFCandidateProducer::CSCTFCandidateProducer(const edm::ParameterSet& pset):
  input_module{consumes<L1CSCTrackCollection>(pset.getUntrackedParameter<edm::InputTag>("CSCTrackProducer"))},
  putToken_{produces<std::vector<L1MuRegionalCand> >("CSC")},
  my_builder{pset.getParameter<edm::ParameterSet>("MuonSorter")}
{
}

void CSCTFCandidateProducer::produce(edm::StreamID, edm::Event & e, const edm::EventSetup& c) const
{
  edm::Handle<L1CSCTrackCollection> tracks;
  std::vector<L1MuRegionalCand> cand_product;

  e.getByToken(input_module, tracks);

  my_builder.buildCandidates(tracks.product(), &cand_product);

  e.emplace(putToken_, std::move(cand_product));
}
