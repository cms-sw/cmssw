#include "L1Trigger/TrackFindingTracklet/interface/ChannelAssignment.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

#include <vector>
#include <array>
#include <set>
#include <algorithm>
#include <numeric>
#include <iterator>

using namespace std;
using namespace edm;
using namespace tt;

namespace trklet {

  ChannelAssignment::ChannelAssignment(const edm::ParameterSet& iConfig, const Setup* setup)
      : setup_(setup),
        pSetDRin_(iConfig.getParameter<ParameterSet>("DRin")),
        widthLayerId_(pSetDRin_.getParameter<int>("WidthLayerId")),
        widthStubId_(pSetDRin_.getParameter<int>("WidthStubId")),
        widthSeedStubId_(pSetDRin_.getParameter<int>("WidthSeedStubId")),
        widthPSTilt_(pSetDRin_.getParameter<int>("WidthPSTilt")),
        depthMemory_(pSetDRin_.getParameter<int>("DepthMemory")),
        ptBoundaries_(pSetDRin_.getParameter<vector<double>>("PtBoundaries")),
        pSetDR_(iConfig.getParameter<ParameterSet>("DR")),
        numComparisonModules_(pSetDR_.getParameter<int>("NumComparisonModules")),
        minIdenticalStubs_(pSetDR_.getParameter<int>("MinIdenticalStubs")),
        numNodesDR_(2 * (ptBoundaries_.size() + 1)),
        seedTypeNames_(iConfig.getParameter<vector<string>>("SeedTypes")),
        numSeedTypes_(seedTypeNames_.size()),
        numChannelsTrack_(numSeedTypes_),
        channelEncoding_(iConfig.getParameter<vector<int>>("IRChannelsIn")) {
    const ParameterSet& pSetSeedTypesSeedLayers = iConfig.getParameter<ParameterSet>("SeedTypesSeedLayers");
    const ParameterSet& pSetSeedTypesProjectionLayers = iConfig.getParameter<ParameterSet>("SeedTypesProjectionLayers");
    seedTypesSeedLayers_.reserve(numSeedTypes_);
    seedTypesProjectionLayers_.reserve(numSeedTypes_);
    for (const string& s : seedTypeNames_) {
      seedTypesSeedLayers_.emplace_back(pSetSeedTypesSeedLayers.getParameter<vector<int>>(s));
      seedTypesProjectionLayers_.emplace_back(pSetSeedTypesProjectionLayers.getParameter<vector<int>>(s));
    }
    // consistency check
    const int offsetBarrel = setup_->offsetLayerId();
    const int numBarrelLayer = setup_->numBarrelLayer();
    const int offsetDisk = setup_->offsetLayerDisks() + offsetBarrel;
    static constexpr int invalidSeedLayer = -1;
    static constexpr int invalidLayerDisk = 0;
    const Settings settings;
    const auto& seedlayers = settings.seedlayers();
    const auto& projlayers = settings.projlayers();
    const auto& projdisks = settings.projdisks();
    // convert Seetings layer ids into ChannelAssignment layer ids
    vector<set<int>> allSeedingLayer(seedlayers.size());
    vector<set<int>> allProjectionLayer(seedlayers.size());
    for (int iSeed = 0; iSeed < (int)seedlayers.size(); iSeed++)
      for (const auto& layer : seedlayers[iSeed])
        if (layer != invalidSeedLayer)
          allSeedingLayer[iSeed].insert(layer < numBarrelLayer ? layer + offsetBarrel
                                                               : layer + offsetDisk - numBarrelLayer);
    for (int iSeed = 0; iSeed < (int)projlayers.size(); iSeed++)
      for (const auto& layer : projlayers[iSeed])
        if (layer != invalidLayerDisk)
          allProjectionLayer[iSeed].insert(layer);
    for (int iSeed = 0; iSeed < (int)projdisks.size(); iSeed++)
      for (const auto& disk : projdisks[iSeed])
        if (disk != invalidLayerDisk)
          allProjectionLayer[iSeed].insert(disk - offsetBarrel + offsetDisk);
    // check if ChannelAssignment seedTypesSeedLayers_ and seedTypesProjectionLayers_ are subsets of Settings pendants
    for (int iSubSeed = 0; iSubSeed < numSeedTypes_; iSubSeed++) {
      const vector<int>& seedLayers = seedTypesSeedLayers_[iSubSeed];
      bool subset(false);
      for (int iAllSeed = 0; iAllSeed < (int)seedlayers.size(); iAllSeed++) {
        // compare seed layer
        const set<int>& asl = allSeedingLayer[iAllSeed];
        set<int> sl(seedLayers.begin(), seedLayers.end());
        set<int> intersect;
        set_intersection(sl.begin(), sl.end(), asl.begin(), asl.end(), inserter(intersect, intersect.begin()));
        if (intersect == sl) {
          subset = true;
          // compare projection layer
          const vector<int>& projectionLayers = seedTypesProjectionLayers_[iSubSeed];
          const set<int>& apl = allProjectionLayer[iAllSeed];
          set<int> pl(projectionLayers.begin(), projectionLayers.end());
          set<int> intersect;
          set_intersection(pl.begin(), pl.end(), apl.begin(), apl.end(), inserter(intersect, intersect.begin()));
          if (intersect == pl)
            break;
          set<int> difference;
          set_difference(
              pl.begin(), pl.end(), intersect.begin(), intersect.end(), inserter(difference, difference.begin()));
          cms::Exception exception("LogicError.");
          exception << "ProjectionLayers ( ";
          for (int layer : difference)
            exception << layer << " ";
          exception << ") are not supported with seed type ( ";
          for (int layer : seedLayers)
            exception << layer << " ";
          exception << ")";
          exception.addContext("trklet::ChannelAssignment::ChannelAssignment");
          throw exception;
        }
      }
      if (subset)
        continue;
      cms::Exception exception("LogicError.");
      exception << "SeedLayers ( ";
      for (int layer : seedLayers)
        exception << layer << " ";
      exception << ") are not supported";
      exception.addContext("trklet::ChannelAssignment::ChannelAssignment");
      throw exception;
    }
    auto bigger = [](const vector<int>& lhs, const vector<int>& rhs) { return lhs.size() < rhs.size(); };
    numSeedingLayers_ = max_element(seedTypesSeedLayers_.begin(), seedTypesSeedLayers_.end(), bigger)->size();
    maxNumProjectionLayers_ =
        max_element(seedTypesProjectionLayers_.begin(), seedTypesProjectionLayers_.end(), bigger)->size();
    auto acc = [](int sum, vector<int> ints) { return sum += (int)ints.size(); };
    offsetsStubs_.reserve(numSeedTypes_);
    numChannelsStub_ = accumulate(
        seedTypesProjectionLayers_.begin(), seedTypesProjectionLayers_.end(), numSeedingLayers_ * numSeedTypes_, acc);
    for (int seed = 0; seed < numSeedTypes_; seed++) {
      const auto it = next(seedTypesProjectionLayers_.begin(), seed);
      offsetsStubs_.emplace_back(accumulate(seedTypesProjectionLayers_.begin(), it, numSeedingLayers_ * seed, acc));
    }
  }

  // returns channelId of given TTTrackRef
  int ChannelAssignment::channelId(const TTTrackRef& ttTrackRef) const {
    const int seedType = ttTrackRef->trackSeedType();
    if (seedType >= numSeedTypes_) {
      cms::Exception exception("logic_error");
      exception << "TTTracks form seed type" << seedType << " not in supported list: (";
      for (const auto& s : seedTypeNames_)
        exception << s << " ";
      exception << ").";
      exception.addContext("trklet:ChannelAssignment:channelId");
      throw exception;
    }
    return ttTrackRef->phiSector() * numSeedTypes_ + seedType;
  }

  // sets layerId of given TTStubRef and seedType, returns false if seeed stub
  bool ChannelAssignment::layerId(int seedType, const TTStubRef& ttStubRef, int& layerId) const {
    layerId = -1;
    if (seedType < 0 || seedType >= numSeedTypes_) {
      cms::Exception exception("logic_error");
      exception.addContext("trklet::ChannelAssignment::layerId");
      exception << "TTTracks with with seed type " << seedType << " not supported.";
      throw exception;
    }
    const int layer = setup_->layerId(ttStubRef);
    const vector<int>& seedingLayers = seedTypesSeedLayers_[seedType];
    if (find(seedingLayers.begin(), seedingLayers.end(), layer) != seedingLayers.end())
      return false;
    const vector<int>& projectingLayers = seedTypesProjectionLayers_[seedType];
    const auto pos = find(projectingLayers.begin(), projectingLayers.end(), layer);
    if (pos == projectingLayers.end()) {
      const string& name = seedTypeNames_[seedType];
      cms::Exception exception("logic_error");
      exception.addContext("trklet::ChannelAssignment::layerId");
      exception << "TTStub from layer " << layer << " (barrel: 1-6; discs: 11-15) from seed type " << name
                << " not supported.";
      throw exception;
    }
    layerId = distance(projectingLayers.begin(), pos);
    return true;
  }

  // index of first stub channel belonging to given track channel
  int ChannelAssignment::offsetStub(int channelTrack) const {
    const int region = channelTrack / numChannelsTrack_;
    const int channel = channelTrack % numChannelsTrack_;
    return region * numChannelsStub_ + offsetsStubs_[channel];
  }

  // returns TBout channel Id
  int ChannelAssignment::channelId(int seedType, int layerId) const {
    const vector<int>& projections = seedTypesProjectionLayers_.at(seedType);
    const auto itp = find(projections.begin(), projections.end(), layerId);
    if (itp != projections.end())
      return distance(projections.begin(), itp);
    const vector<int>& seeds = seedTypesSeedLayers_.at(seedType);
    const auto its = find(seeds.begin(), seeds.end(), layerId);
    if (its != seeds.end())
      return (int)projections.size() + distance(seeds.begin(), its);
    return -1;
  }

  // return DR node for given ttTrackRef
  int ChannelAssignment::nodeDR(const TTTrackRef& ttTrackRef) const {
    const double pt = ttTrackRef->momentum().perp();
    int bin(0);
    for (double b : ptBoundaries_) {
      if (pt < b)
        break;
      bin++;
    }
    if (ttTrackRef->rInv() >= 0.)
      bin += numNodesDR_ / 2;
    else
      bin = numNodesDR_ / 2 - 1 - bin;
    return bin;
  }

  // layers a seed types can project to using default layer id [barrel: 1-6, discs: 11-15]
  int ChannelAssignment::layerId(int seedType, int channel) const {
    if (channel < numProjectionLayers(seedType))
      return seedTypesProjectionLayers_.at(seedType).at(channel);
    return seedTypesSeedLayers_.at(seedType).at(channel - numProjectionLayers(seedType));
  }

}  // namespace trklet
