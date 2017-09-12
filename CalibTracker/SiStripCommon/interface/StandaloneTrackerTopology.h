#ifndef TRACKER_TOPOLOGY_STANDALONE_H
#define TRACKER_TOPOLOGY_STANDALONE_H

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

namespace StandaloneTrackerTopology {
  /**
   * Construct a TrackerTopology from a trackerParameters.xml file
   *
   * WARNING: this method has been introduced to construct a TrackerTopology
   * object only for the rare cases where it cannot be retrieved from an
   * edm::EventSetup (e.g. ROOT macros).
   */
  TrackerTopology fromTrackerParametersXML(const std::string& xmlFileName);
};

#endif // TRACKER_TOPOLOGY_STANDALONE_H
