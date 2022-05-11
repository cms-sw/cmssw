#include <memory>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/DTTriggerPhase2/interface/InitialGrouping.h"

using namespace edm;
using namespace std;
using namespace cmsdt;
using namespace dtamgrouping;
// ============================================================================
// Constructors and destructor
// ============================================================================
InitialGrouping::InitialGrouping(const ParameterSet &pset, edm::ConsumesCollector &iC)
    : MotherGrouping(pset, iC), debug_(pset.getUntrackedParameter<bool>("debug")), currentBaseChannel_(-1) {
  // Obtention of parameters
  if (debug_)
    LogDebug("InitialGrouping") << "InitialGrouping: constructor";

  // Initialisation of channelIn array
  chInDummy_.push_back(DTPrimitive());
  for (int lay = 0; lay < NUM_LAYERS; lay++) {
    for (int ch = 0; ch < NUM_CH_PER_LAYER; ch++) {
      channelIn_[lay][ch] = {chInDummy_};
      channelIn_[lay][ch].clear();
    }
  }
}

InitialGrouping::~InitialGrouping() {
  if (debug_)
    LogDebug("InitialGrouping") << "InitialGrouping: destructor";
}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void InitialGrouping::initialise(const edm::EventSetup &iEventSetup) {
  if (debug_)
    LogDebug("InitialGrouping") << "InitialGrouping::initialiase";
}

void InitialGrouping::run(Event &iEvent,
                          const EventSetup &iEventSetup,
                          const DTDigiCollection &digis,
                          MuonPathPtrs &mpaths) {
  //   This function returns the analyzable mpath collection back to the the main function
  //   so it can be fitted. This is in fact doing the so-called grouping.

  for (int supLayer = 0; supLayer < NUM_SUPERLAYERS; supLayer++) {  // for each SL:
    if (debug_)
      LogDebug("InitialGrouping") << "InitialGrouping::run Reading SL" << supLayer;
    setInChannels(&digis, supLayer);

    for (int baseCh = 0; baseCh < TOTAL_BTI; baseCh++) {
      currentBaseChannel_ = baseCh;
      selectInChannels(currentBaseChannel_);  //map a number of wires for a given base channel
      if (notEnoughDataInChannels())
        continue;

      if (debug_)
        LogDebug("InitialGrouping") << "InitialGrouping::run --> now check pathId";
      for (int pathId = 0; pathId < 8; pathId++) {
        resetPrvTDCTStamp();
        if (debug_)
          LogDebug("InitialGrouping") << "[InitialGrouping::run] mixChannels calling";
        mixChannels(supLayer, pathId, mpaths);
        if (debug_)
          LogDebug("InitialGrouping") << "[InitialGrouping::run] mixChannels end";
      }
    }
  }
  if (debug_)
    LogDebug("InitialGrouping") << "[InitialGrouping::run] end";
}

void InitialGrouping::finish() { return; };

// ============================================================================
// Other methods
// ============================================================================
void InitialGrouping::setInChannels(const DTDigiCollection *digis, int sl) {
  //   before setting channels we need to clear
  for (int lay = 0; lay < NUM_LAYERS; lay++) {
    for (int ch = 0; ch < NUM_CH_PER_LAYER; ch++) {
      channelIn_[lay][ch].clear();
    }
  }

  // now fill with those primitives that makes sense:
  for (const auto &dtLayerId_It : *digis) {
    const DTLayerId dtLId = dtLayerId_It.first;
    if (dtLId.superlayer() != sl + 1)
      continue;  //skip digis not in SL...

    for (DTDigiCollection::const_iterator digiIt = (dtLayerId_It.second).first; digiIt != (dtLayerId_It.second).second;
         ++digiIt) {
      int layer = dtLId.layer() - 1;
      int wire = (*digiIt).wire() - 1;
      int digiTIME = (*digiIt).time();
      int digiTIMEPhase2 = digiTIME;

      if (debug_)
        LogDebug("InitialGrouping") << "[InitialGrouping::setInChannels] SL" << sl << " L" << layer << " : " << wire
                                    << " " << digiTIMEPhase2;
      auto dtpAux = DTPrimitive();
      dtpAux.setTDCTimeStamp(digiTIMEPhase2);
      dtpAux.setChannelId(wire);
      dtpAux.setLayerId(layer);    //  L=0,1,2,3
      dtpAux.setSuperLayerId(sl);  // SL=0,1,2
      dtpAux.setCameraId(dtLId.rawId());
      channelIn_[layer][wire].push_back(dtpAux);
    }
  }
}

void InitialGrouping::selectInChannels(int baseChannel) {
  // Channels are labeled following next schema:
  // Input Muxer Indexes
  // ---------------------------------
  // |   6   |   7   |   8   |   9   |
  // ---------------------------------
  // |   3   |   4   |   5   |
  // -------------------------
  // |   1   |   2   |
  // -----------------
  // |   0   |
  // ---------

  // ****** LAYER 0 ******
  muxInChannels_[0] = channelIn_[0][baseChannel];

  // ****** LAYER 1 ******
  muxInChannels_[1] = channelIn_[1][baseChannel];

  if (baseChannel + 1 < NUM_CH_PER_LAYER)
    muxInChannels_[2] = channelIn_[1][baseChannel + 1];
  else
    muxInChannels_[2] = chInDummy_;

  // ****** LAYER 2 ******
  if (baseChannel - 1 >= 0)
    muxInChannels_[3] = channelIn_[2][baseChannel - 1];
  else
    muxInChannels_[3] = chInDummy_;

  muxInChannels_[4] = channelIn_[2][baseChannel];

  if (baseChannel + 1 < NUM_CH_PER_LAYER)
    muxInChannels_[5] = channelIn_[2][baseChannel + 1];
  else
    muxInChannels_[5] = chInDummy_;

  // ****** LAYER 3 ******
  if (baseChannel - 1 >= 0)
    muxInChannels_[6] = channelIn_[3][baseChannel - 1];
  else
    muxInChannels_[6] = chInDummy_;

  muxInChannels_[7] = channelIn_[3][baseChannel];

  if (baseChannel + 1 < NUM_CH_PER_LAYER)
    muxInChannels_[8] = channelIn_[3][baseChannel + 1];
  else
    muxInChannels_[8] = chInDummy_;

  if (baseChannel + 2 < NUM_CH_PER_LAYER)
    muxInChannels_[9] = channelIn_[3][baseChannel + 2];
  else
    muxInChannels_[9] = chInDummy_;
}

bool InitialGrouping::notEnoughDataInChannels(void) {
  // Empty layer indicators
  bool lEmpty[4];

  lEmpty[0] = muxInChannels_[0].empty();

  lEmpty[1] = muxInChannels_[1].empty() && muxInChannels_[2].empty();

  lEmpty[2] = muxInChannels_[3].empty() && muxInChannels_[4].empty() && muxInChannels_[5].empty();

  lEmpty[3] =
      muxInChannels_[6].empty() && muxInChannels_[7].empty() && muxInChannels_[8].empty() && muxInChannels_[9].empty();

  // If there are at least two empty layers, you cannot link it to a possible trace
  if ((lEmpty[0] && lEmpty[1]) or (lEmpty[0] && lEmpty[2]) or (lEmpty[0] && lEmpty[3]) or (lEmpty[1] && lEmpty[2]) or
      (lEmpty[1] && lEmpty[3]) or (lEmpty[2] && lEmpty[3])) {
    return true;
  } else {
    return false;
  }
}

void InitialGrouping::resetPrvTDCTStamp(void) {
  for (int i = 0; i < NUM_LAYERS; i++)
    prevTDCTimeStamps_[i] = -1;
}

bool InitialGrouping::isEqualComb2Previous(DTPrimitives &dtPrims) {
  bool answer = true;

  for (int i = 0; i < NUM_LAYERS; i++) {
    if (prevTDCTimeStamps_[i] != dtPrims[i].tdcTimeStamp()) {
      answer = false;
      for (int j = 0; j < NUM_LAYERS; j++) {
        prevTDCTimeStamps_[j] = dtPrims[j].tdcTimeStamp();
      }
      break;
    }
  }
  return answer;
}

void InitialGrouping::mixChannels(int supLayer, int pathId, MuonPathPtrs &outMuonPath) {
  if (debug_)
    LogDebug("InitialGrouping") << "[InitialGrouping::mixChannel] begin";
  DTPrimitives data[4];

  // Real amount of values extracted from each channel.
  int numPrimsPerLayer[4] = {0, 0, 0, 0};
  unsigned int canal;
  int channelEmptyCnt = 0;
  for (int layer = 0; layer <= 3; layer++) {
    canal = CHANNELS_PATH_ARRANGEMENTS[pathId][layer];
    if (muxInChannels_[canal].empty())
      channelEmptyCnt++;
  }

  if (channelEmptyCnt >= 2)
    return;
  //

  // We extract the number of elements necesary from each channel as the combination requires
  for (int layer = 0; layer < NUM_LAYERS; layer++) {
    canal = CHANNELS_PATH_ARRANGEMENTS[pathId][layer];
    unsigned int maxPrimsToBeRetrieved = muxInChannels_[canal].size();
    /*
    If the number of primitives is zero, in order to avoid that only one
    empty channel avoids mixing data from the other three, we, at least,
    consider one dummy element from this channel.
    In other cases, where two or more channels has zero elements, the final
    combination will be not analyzable (the condition for being analyzable is
    that it has at least three good TDC time values, not dummy), so it will
    be discarded and not sent to the analyzer.
  */
    if (maxPrimsToBeRetrieved == 0)
      maxPrimsToBeRetrieved = 1;

    for (unsigned int items = 0; items < maxPrimsToBeRetrieved; items++) {
      auto dtpAux = DTPrimitive();
      if (!muxInChannels_[canal].empty())
        dtpAux = DTPrimitive(&(muxInChannels_[canal].at(items)));

      /*
	I won't allow a whole loop cycle. When a DTPrimitive has an invalid
	time-stamp (TDC value = -1) it means that the buffer is empty or the
	buffer has reached the last element within the configurable time window.
	In this case the loop is broken, but only if there is, at least, one
	DTPrim (even invalid) on the outgoing array. This is mandatory to cope
        with the idea explained in the previous comment block
      */
      if (dtpAux.tdcTimeStamp() < 0 && items > 0)
        break;

      // In this new schema, if the hit corresponds with the SL over which
      // you are doing the mixings, it is sent to the intermediate mixing
      // buffer. In the opposite case, a blank and invalid copy is sent to
      // allow them mixing to be complete, as it was done in the one SL case.

      // This is a kind of quick solution in which there will be no few cases
      // where you will have invalid mixings. Because of that, the verification
      // that is done later, where the segment is analysed to check whether it
      // can be analysed is essential.
      if (dtpAux.superLayerId() == supLayer)
        data[layer].push_back(dtpAux);  // values are 0, 1, 2
      else
        data[layer].push_back(DTPrimitive());
      numPrimsPerLayer[layer]++;
    }
  }

  if (debug_)
    LogDebug("InitialGrouping") << "[InitialGrouping::mixChannels] filled data";

  // Here we do the different combinations and send them to the output FIFO.
  DTPrimitives ptrPrimitive;
  int chIdx[4];
  for (chIdx[0] = 0; chIdx[0] < numPrimsPerLayer[0]; chIdx[0]++) {
    for (chIdx[1] = 0; chIdx[1] < numPrimsPerLayer[1]; chIdx[1]++) {
      for (chIdx[2] = 0; chIdx[2] < numPrimsPerLayer[2]; chIdx[2]++) {
        for (chIdx[3] = 0; chIdx[3] < numPrimsPerLayer[3]; chIdx[3]++) {
          // We build a copy of the object so that we can manipulate each one
          // in each thread of the process independently, allowing us also to
          // delete them whenever it is necessary, without relying upon a
          // unique reference all over the code.

          for (int i = 0; i < NUM_LAYERS; i++) {
            ptrPrimitive.push_back((data[i])[chIdx[i]]);
            if (debug_)
              LogDebug("InitialGrouping")
                  << "[InitialGrouping::mixChannels] reading " << ptrPrimitive[i].tdcTimeStamp();
          }

          auto ptrMuonPath = std::make_shared<MuonPath>(ptrPrimitive);
          ptrMuonPath->setCellHorizontalLayout(CELL_HORIZONTAL_LAYOUTS[pathId]);

          /*
            This new version of this code is redundant with PathAnalyzer code,
            where every MuonPath not analyzable is discarded.
            I insert this discarding mechanism here, as well, to avoid inserting
            not-analyzable MuonPath into the candidate FIFO.
            Equivalent code must be removed in the future from PathAnalyzer, but
            it the mean time, at least during the testing state, I'll preserve
            both.
            Code in the PathAnalyzer should be doing nothing now.
          */
          if (debug_)
            LogDebug("InitialGrouping") << "[InitialGrouping::mixChannels] muonPath is analyzable? " << ptrMuonPath;
          if (ptrMuonPath->isAnalyzable()) {
            if (debug_)
              LogDebug("InitialGrouping") << "[InitialGrouping::mixChannels] YES";
            /*
            This is a very simple filter because, during the tests, it has been
            detected that many consecutive MuonPaths are duplicated mainly due
            to buffers empty (or dummy) that give a TDC time-stamp = -1
            With this filter, I'm removing those consecutive identical
            combinations.
            
            If duplicated combinations are not consecutive, they won't be
            detected here
	    */
            if (!isEqualComb2Previous(ptrPrimitive)) {
              if (debug_)
                LogDebug("InitialGrouping") << "[InitialGrouping::mixChannels] isNOT equal to previous";
              ptrMuonPath->setBaseChannelId(currentBaseChannel_);
              outMuonPath.push_back(std::move(ptrMuonPath));
            }
            ptrPrimitive.clear();
          }
        }
      }
    }
  }
  for (int layer = 0; layer < NUM_LAYERS; layer++) {
    data[layer].clear();
  }
}
