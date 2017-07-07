#include "CSCTFCandidateProducer.h"
#include "L1Trigger/CSCTrackFinder/src/CSCTFCandidateBuilder.h"

#include <vector>
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

CSCTFCandidateProducer::CSCTFCandidateProducer(const edm::ParameterSet& pset)
{
  edm::ParameterSet mu_sorter_pset = pset.getParameter<edm::ParameterSet>("MuonSorter");
  my_builder = new CSCTFCandidateBuilder(mu_sorter_pset);
  input_module = consumes<L1CSCTrackCollection>(pset.getUntrackedParameter<edm::InputTag>("CSCTrackProducer"));
  produces<std::vector<L1MuRegionalCand> >("CSC");
}

CSCTFCandidateProducer::~CSCTFCandidateProducer()
{
  delete my_builder;
  my_builder = nullptr;
}

void CSCTFCandidateProducer::produce(edm::Event & e, const edm::EventSetup& c)
{
  edm::Handle<L1CSCTrackCollection> tracks;
  std::unique_ptr<std::vector<L1MuRegionalCand> > cand_product(new std::vector<L1MuRegionalCand>);

  e.getByToken(input_module, tracks);

  my_builder->buildCandidates(tracks.product(), cand_product.get());

  e.put(std::move(cand_product),"CSC");
}
