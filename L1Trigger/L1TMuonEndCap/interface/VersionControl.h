#ifndef L1TMuonEndCap_VersionControl_h
#define L1TMuonEndCap_VersionControl_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SectorProcessor;

class VersionControl {
public:
  explicit VersionControl(const edm::ParameterSet& iConfig);
  ~VersionControl();

  // Modify the configurables based on FW version
  void configure_by_fw_version(unsigned fw_version);

  // Getters
  const edm::ParameterSet& getConfig() const { return config_; }
  int verbose() const { return verbose_; }
  bool useO2O() const { return useO2O_; }
  std::string era() const { return era_; }

  friend class SectorProcessor;  // allow access to private memebers

private:
  // All the configurables from python/simEmtfDigis_cfi.py must be visible to this class, except InputTags.
  const edm::ParameterSet config_;

  int verbose_;
  bool useO2O_;
  std::string era_;

  // Trigger primitives & BX settings
  bool useDT_, useCSC_, useRPC_, useIRPC_, useCPPF_, useGEM_, useME0_;
  int minBX_, maxBX_, bxWindow_, bxShiftCSC_, bxShiftRPC_, bxShiftGEM_, bxShiftME0_;

  // For primitive conversion
  std::vector<int> zoneBoundaries_;
  int zoneOverlap_;
  bool includeNeighbor_, duplicateTheta_, fixZonePhi_, useNewZones_, fixME11Edges_;

  // For pattern recognition
  std::vector<std::string> pattDefinitions_, symPattDefinitions_;
  bool useSymPatterns_;

  // For track building
  int thetaWindow_, thetaWindowZone0_;
  bool useSingleHits_;
  bool bugSt2PhDiff_, bugME11Dupes_, bugAmbigThetaWin_, twoStationSameBX_;

  // For ghost cancellation
  int maxRoadsPerZone_, maxTracks_;
  bool useSecondEarliest_;
  bool bugSameSectorPt0_;

  // For pt assignment
  bool readPtLUTFile_, fixMode15HighPt_;
  bool bug9BitDPhi_, bugMode7CLCT_, bugNegPt_, bugGMTPhi_, promoteMode7_;
  int modeQualVer_;
};

#endif
