#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentSeedSelector_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentSeedSelector_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include <vector>

namespace edm {
  class Event;
}

class AlignmentSeedSelector {
public:
  typedef std::vector<const TrajectorySeed*> Seeds;

  /// constructor
  AlignmentSeedSelector(const edm::ParameterSet& cfg);

  /// destructor
  ~AlignmentSeedSelector();

  /// select tracks
  Seeds select(const Seeds& seeds, const edm::Event& evt) const;

  static void fillPSetDescription(edm::ParameterSetDescription& desc);

private:
  /// private data members
  bool applySeedNumber;
  int minNSeeds, maxNSeeds;
};

#endif
