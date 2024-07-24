#include "L1Trigger/DTTriggerPhase2/interface/ShowerGrouping.h"
#include "L1Trigger/DTTriggerPhase2/interface/ShowerBuffer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace edm;
using namespace std;
using namespace cmsdt;

// ============================================================================
// Constructors and destructor
// ============================================================================
ShowerGrouping::ShowerGrouping(const ParameterSet& pset, edm::ConsumesCollector& iC) : 
    debug_(pset.getUntrackedParameter<bool>("debug")), 
    nHits_per_bx(pset.getParameter<int>("nHits_per_bx")), 
    threshold_for_shower(pset.getParameter<int>("threshold_for_shower")), 
    showerTaggingAlgo_(pset.getParameter<int>("showerTaggingAlgo")), 
    currentBaseChannel_(-1) {
    
    // Initialise channelIn array
    chInDummy_.push_back(DTPrimitive());
    for (int lay = 0; lay < NUM_LAYERS_2SL; lay++) {
        for (int ch = 0; ch < NUM_CH_PER_LAYER; ch++) {
            channelIn_[lay][ch] = {chInDummy_};
            channelIn_[lay][ch].clear();
        }
    }
}

ShowerGrouping::~ShowerGrouping() {}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void ShowerGrouping::initialise(const edm::EventSetup& iEventSetup) {}

void ShowerGrouping::run(Event& iEvent,
                         const EventSetup& iEventSetup,
                         const DTDigiCollection& digis,
                         ShowerBufferPtr &showerBuffer) {
    /* This function returns the analyzable shower collection */

    // Clear channels from previous event
    clearChannels();

    // Set the incoming hits in the channels
    setInChannels(&digis);

    // Now sort them by time
    sortHits();
    if (debug_) std::cout << "      + Going to study " << all_hits.size() << " hits" << std::endl;
    auto showerBuf = std::make_shared<ShowerBuffer>();
  
    // Create a buffer 
    for (auto &hit : all_hits) {
      // Standalone mode: just save hits in buffer 
      showerBuf->addHit(hit); 
      
      // TBD: add OBDT delays
    }

    if (triggerShower(showerBuf)) {

      if (debug_) std::cout << "        o Shower found with " << all_hits.size() << " hits" << std::endl;
      showerBuf->flag(); 
    }
    showerBuffer = std::move(showerBuf);
    //std::cout << "Shower has " << showerBuf->getNhits() << std::endl;
}

bool ShowerGrouping::triggerShower(const ShowerBufferPtr& showerBuf) {
  // Method to apply shower tagging logic
  
  auto nHits = showerBuf->getNhits();
  bool tagged_shower = false;

  if (showerTaggingAlgo_ == 1) {
    // Method v1: pure counting is over a given threshold
    if (nHits >= threshold_for_shower) {
        tagged_shower = true;
    }
  }
  if (showerTaggingAlgo_ == 2) {
    // Method v2: correlate hits based on distance
    // TO BE IMPLEMENTED
    return tagged_shower;
  }
  return tagged_shower;
}

void ShowerGrouping::clearChannels() {
  // This function clears channelIn and allHits collections. 
  for (int lay = 0; lay < NUM_LAYERS_2SL; lay++) {
    if (debug_) std::cout << "        o Clearing hits in layer " << lay << std::endl;
    for (int ch = 0; ch < NUM_CH_PER_LAYER; ch++) {
      channelIn_[lay][ch].clear();
    }
  }
  all_hits.clear();
}

void ShowerGrouping::setInChannels(const DTDigiCollection *digis) {

  for (const auto &dtLayerId_It : *digis) {
    const DTLayerId dtLId = dtLayerId_It.first;

    // Now iterate over the digis
    for (DTDigiCollection::const_iterator digiIt = (dtLayerId_It.second).first; digiIt != (dtLayerId_It.second).second;
         ++digiIt) {
      int layer = dtLId.layer() - 1;
      int wire = (*digiIt).wire() - 1;
      int digiTIME = (*digiIt).time();
      int digiTIMEPhase2 = digiTIME;

      auto dtpAux = DTPrimitive();
      dtpAux.setTDCTimeStamp(digiTIMEPhase2);
      dtpAux.setChannelId(wire);
      dtpAux.setLayerId(layer);    //  L=0,1,2,3
      dtpAux.setSuperLayerId(dtLId.superlayer());  // SL=0,1,2
      dtpAux.setCameraId(dtLId.rawId());
      channelIn_[layer][wire].push_back(dtpAux);
      all_hits.push_back(dtpAux);
    }
  }
}

void ShowerGrouping::sortHits() {
  // Sort everything by the time of the hits, so it has the same behaviour as the fw
  for (int lay = 0; lay < NUM_LAYERS_2SL; lay++) {
    for (int ch = 0; ch < NUM_CH_PER_LAYER; ch++) {
      std::stable_sort(channelIn_[lay][ch].begin(), channelIn_[lay][ch].end(), hitTimeSort_shower);
    }
  }
  std::stable_sort(all_hits.begin(), all_hits.end(), hitTimeSort_shower);
}

void ShowerGrouping::finish(){};
