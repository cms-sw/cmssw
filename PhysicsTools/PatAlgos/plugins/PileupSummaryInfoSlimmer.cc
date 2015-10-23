#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

#include <iostream>

class PileupSummaryInfoSlimmer : public edm::global::EDProducer<> {
public:
  PileupSummaryInfoSlimmer(const edm::ParameterSet& conf) :
    src_(consumes<std::vector<PileupSummaryInfo> >(conf.getParameter<edm::InputTag>("src"))),
    keepDetailedInfoFor_(conf.getParameter<std::vector<int32_t> >("keepDetailedInfoFor")) {
    produces<std::vector<PileupSummaryInfo> >();
  }

  void produce(edm::StreamID, edm::Event &, edm::EventSetup const &) const override final;

private:
  const edm::EDGetTokenT<std::vector<PileupSummaryInfo> > src_;
  const std::vector<int> keepDetailedInfoFor_;
};

void PileupSummaryInfoSlimmer::produce(edm::StreamID, 
                                       edm::Event& evt,
                                       const edm::EventSetup& es ) const {
  edm::Handle<std::vector<PileupSummaryInfo> > input;
  std::auto_ptr<std::vector<PileupSummaryInfo> > output( new std::vector<PileupSummaryInfo> );

  evt.getByToken(src_,input);
  
  for( const auto& psu : *input ) {
    const int bunchCrossing = psu.getBunchCrossing();
    const int bunchSpacing = psu.getBunchSpacing();
    const int num_PU_vertices = psu.getPU_NumInteractions();
    const float TrueNumInteractions = psu.getTrueNumInteractions();
    
    std::vector<float> zpositions;
    std::vector<float> sumpT_lowpT;
    std::vector<float> sumpT_highpT;
    std::vector<int> ntrks_lowpT;
    std::vector<int> ntrks_highpT;
    std::vector<edm::EventID> eventInfo;
    std::vector<float> pT_hats;

    const bool keep_details = std::find(keepDetailedInfoFor_.begin(),
                                        keepDetailedInfoFor_.end(),
                                        bunchCrossing) != keepDetailedInfoFor_.end();
    
    if( keep_details ) {
      zpositions   = psu.getPU_zpositions();
      sumpT_lowpT  = psu.getPU_sumpT_lowpT();
      sumpT_highpT = psu.getPU_sumpT_highpT();
      ntrks_lowpT  = psu.getPU_ntrks_lowpT();
      ntrks_highpT = psu.getPU_ntrks_highpT();
      eventInfo    = psu.getPU_EventID();
      pT_hats      = psu.getPU_pT_hats();
    }
    // insert the slimmed vertex info
    output->emplace_back(num_PU_vertices,
                         zpositions,
                         sumpT_lowpT, sumpT_highpT,
                         ntrks_lowpT, ntrks_highpT,
                         eventInfo,
                         pT_hats,
                         bunchCrossing,
                         TrueNumInteractions,
                         bunchSpacing);
  }

  evt.put(output);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PileupSummaryInfoSlimmer);
