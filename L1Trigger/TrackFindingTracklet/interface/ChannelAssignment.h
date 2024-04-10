#ifndef L1Trigger_TrackFindingTracklet_ChannelAssignment_h
#define L1Trigger_TrackFindingTracklet_ChannelAssignment_h

#include "FWCore/Framework/interface/data_default_record_trait.h"
#include "L1Trigger/TrackFindingTracklet/interface/ChannelAssignmentRcd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"

#include <vector>

namespace trklet {

  /*! \class  trklet::ChannelAssignment
   *  \brief  Class to assign tracklet tracks and stubs to output channel
   *          based on their Pt or seed type as well as DTC stubs to input channel
   *  \author Thomas Schuh
   *  \date   2020, Nov; updated 2021 Oct
   */
  class ChannelAssignment {
  public:
    ChannelAssignment() {}
    ChannelAssignment(const edm::ParameterSet& iConfig, const tt::Setup* setup);
    ~ChannelAssignment() {}
    // returns channelId of given TTTrackRef from TrackBuilder
    int channelId(const TTTrackRef& ttTrackRef) const;
    // number of used TB channels for tracks
    int numChannelsTrack() const { return numChannelsTrack_; }
    // number of used TB channels for stubs
    int numChannelsStub() const { return numChannelsStub_; }
    // number of bits used to represent layer id [barrel: 0-5, discs: 6-10]
    int widthLayerId() const { return widthLayerId_; }
    // number of bits used to represent stub id for projected stubs
    int widthStubId() const { return widthStubId_; }
    // number of bits used to represent stub id for seed stubs
    int widthSeedStubId() const { return widthSeedStubId_; }
    // number of bits used to distinguish between tilted and untilded barrel modules or 2S and PS endcap modules
    int widthPSTilt() const { return widthPSTilt_; }
    // depth of fifos within systolic array
    int depthMemory() const { return depthMemory_; }
    // number of comparison modules used in each DR node
    int numComparisonModules() const { return numComparisonModules_; }
    // min number of shared stubs to identify duplicates
    int minIdenticalStubs() const { return minIdenticalStubs_; }
    // number of DR nodes
    int numNodesDR() const { return numNodesDR_; }
    // number of used seed types in tracklet algorithm
    int numSeedTypes() const { return numSeedTypes_; }
    // sets layerId (0-7 in sequence the seed type projects to) of given TTStubRef and seedType, returns false if seeed stub
    bool layerId(int seedType, const TTStubRef& ttStubRef, int& layerId) const;
    // number layers a given seed type projects to
    int numProjectionLayers(int seedType) const { return (int)seedTypesProjectionLayers_.at(seedType).size(); }
    // max. no. layers that any seed type projects to
    int maxNumProjectionLayers() const { return maxNumProjectionLayers_; }
    // map of used DTC tfp channels in InputRouter
    const std::vector<int>& channelEncoding() const { return channelEncoding_; }
    // index of first stub channel belonging to given track channel
    int offsetStub(int channelTrack) const;
    // seed layers for given seed type id
    const std::vector<int>& seedingLayers(int seedType) const { return seedTypesSeedLayers_.at(seedType); }
    // returns SensorModule::Type for given TTStubRef
    tt::SensorModule::Type type(const TTStubRef& ttStubRef) const { return setup_->type(ttStubRef); }
    // layers a seed types can project to using default layer id [barrel: 1-6, discs: 11-15]
    int layerId(int seedType, int channel) const;
    // returns TBout channel Id for given seed type and default layer id [barrel: 1-6, discs: 11-15], returns -1 if layerId and seedType are inconsistent
    int channelId(int seedType, int layerId) const;
    // max number of seeding layers
    int numSeedingLayers() const { return numSeedingLayers_; }
    // return DR node for given ttTrackRef
    int nodeDR(const TTTrackRef& ttTrackRef) const;

  private:
    // helper class to store configurations
    const tt::Setup* setup_;
    // DRin parameter
    edm::ParameterSet pSetDRin_;
    // number of bits used to represent layer id [barrel: 0-5, discs: 6-10]
    int widthLayerId_;
    // number of bits used to represent stub id for projected stubs
    int widthStubId_;
    // number of bits used to represent stub id for seed stubs
    int widthSeedStubId_;
    // number of bits used to distinguish between tilted and untilded barrel modules or 2S and PS endcap modules
    int widthPSTilt_;
    // depth of fifos within systolic array
    int depthMemory_;
    // positive pt Boundaries in GeV (symmetric negatives are assumed), first boundary is pt cut, last boundary is infinity, defining ot bins used by DR
    std::vector<double> ptBoundaries_;
    // DRin parameter
    edm::ParameterSet pSetDR_;
    // number of comparison modules used in each DR node
    int numComparisonModules_;
    // min number of shared stubs to identify duplicates [default: 3]
    int minIdenticalStubs_;
    // number of DR nodes
    int numNodesDR_;
    // seed type names
    std::vector<std::string> seedTypeNames_;
    // number of used seed types in tracklet algorithm
    int numSeedTypes_;
    // number of used TB channels for tracks
    int numChannelsTrack_;
    // number of used TB channels for stubs
    int numChannelsStub_;
    // seeding layers of seed types using default layer id [barrel: 1-6, discs: 11-15]
    std::vector<std::vector<int>> seedTypesSeedLayers_;
    // layers a seed types can project to using default layer id [barrel: 1-6, discs: 11-15]
    std::vector<std::vector<int>> seedTypesProjectionLayers_;
    // max. number of layers to which any seed type projects
    int maxNumProjectionLayers_;
    // map of used DTC tfp channels in InputRouter
    std::vector<int> channelEncoding_;
    // accumulated number of projections layer from seed 0 to vector index
    std::vector<int> offsetsStubs_;
    // max number of seeding layers
    int numSeedingLayers_;
  };

}  // namespace trklet

EVENTSETUP_DATA_DEFAULT_RECORD(trklet::ChannelAssignment, trklet::ChannelAssignmentRcd);

#endif
