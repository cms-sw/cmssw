#ifndef TRACKER_TOPOLOGY_STANDALONE_H
#define TRACKER_TOPOLOGY_STANDALONE_H

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

namespace StandaloneTrackerTopology {
  /**
   * Construct a TrackerTopology from a trackerParameters.xml file, from the name of the file
   *
   * WARNING: this method has been introduced to construct a TrackerTopology
   * object only for the rare cases where it cannot be retrieved from an
   * edm::EventSetup (e.g. ROOT macros).
   */
  TrackerTopology fromTrackerParametersXMLFile(const std::string& xmlFileName);

  /**
   * Construct a TrackerTopology from a trackerParameters.xml file, from the contents read into a std::string
   *
   * WARNING: this method has been introduced to construct a TrackerTopology
   * object only for the rare cases where it cannot be retrieved from an
   * edm::EventSetup (e.g. ROOT macros).
   */
  TrackerTopology fromTrackerParametersXMLString(const std::string& xmlContent);
};

#endif // TRACKER_TOPOLOGY_STANDALONE_H
