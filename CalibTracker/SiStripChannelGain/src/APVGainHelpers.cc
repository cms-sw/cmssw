#include "CalibTracker/SiStripChannelGain/interface/APVGainHelpers.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

/** Brief Extract from the DetId the subdetector type.
 * Return an integer which is associated to the subdetector type. The integer
 * coding for phase0/phase1 geometry follows:
 *
 *  3 - TIB
 *  4 - TID
 *  5 - TOB
 *  6 - TEC
 */
int APVGain::subdetectorId(uint32_t det_id) { return DetId(det_id).subdetId(); };

/** Brief Extract from a char * the subdetector type.
 * Return an integer whioch is associated to the subdetector type. The integer
 * coding follows:
 *
 * 3 - TIB
 * 4 - TID
 * 5 - TOB
 * 6 - TEC
 *
 * The char * string is expected to have a 3 char descriptor of the subdetector
 * type in front.
 */
int APVGain::subdetectorId(const std::string& tag) {
  std::string d = tag.substr(0, 3);
  if (d == "TIB")
    return 3;
  if (d == "TID")
    return 4;
  if (d == "TOB")
    return 5;
  if (d == "TEC")
    return 6;
  return 0;
};

/** Brief Extract the subdetector side from the Det Id
 * Return and integer whose coding is
 *   0 - no side description can be applied
 *   1 - for negative side
 *   2 - for positive side
 */
int APVGain::subdetectorSide(uint32_t det_id, const TrackerTopology* topo) { return topo->side(det_id); }

/** Brief Extract the subdetector side from a char * descriptor
 * Return and integer whose coding is
 *   0 - no side description can be applied
 *   1 - for negative side
 *   2 - for positive side
 *
 *   The char * descriptor is expected to have either "minus" or "plus"
 *   string to specify the sign. If no sign spec is found 0 is returned.
 */
int APVGain::subdetectorSide(const std::string& tag) {
  std::size_t m = tag.find("minus");
  std::size_t p = tag.find("plus");
  if (m != std::string::npos)
    return 1;
  if (p != std::string::npos)
    return 2;
  return 0;
}

/** Brief Extract the sensor thickness from the Det Id
 * Return and integer whose coding is
 *   0 - no thickness can be determined
 *   1 - for thin sensors
 *   2 - for thick sensors
 */
int APVGain::thickness(uint32_t det_id) {
  if (APVGain::subdetectorId(det_id) >= SiStripDetId::TIB) {
    SiStripDetId siStripDetId(det_id);
    if (siStripDetId.subdetId() == SiStripDetId::TOB) {
      return 2;  // so it is TOB (thick)
    }
    if (siStripDetId.moduleGeometry() == SiStripModuleGeometry::W5 ||
        siStripDetId.moduleGeometry() == SiStripModuleGeometry::W6 ||
        siStripDetId.moduleGeometry() == SiStripModuleGeometry::W7) {
      return 2;  // so it is TEC ring 5-7 (thick)
    }
    return 1;  // so it is TEC ring 1-4 or TIB or TID (thin)
  } else {
    return 0;
  }
}

/** Brief Extract the thickness from a char * descriptor
 * Return and integer whose coding is
 *   0 - no thicnkness can be determined
 *   1 - for thin sensors
 *   2 - for thick sensors
 *
 *   The char * descriptor is expected to have either "thin" or "thick"
 *   string to specify the thickness. If no sign spec is found 0 is returned.
 */
int APVGain::thickness(const std::string& tag) {
  std::size_t thin = tag.find("thin");
  std::size_t thick = tag.find("thick");
  if (thin != std::string::npos)
    return 1;
  if (thick != std::string::npos)
    return 2;
  return 0;
}

/** Brief Extract the detector plane position from a DetId.
 * Return an integer that represent the detector plane where the module sits.
 * For the barrel detectors (TIB and TOB) the detector plane is the layer, e.g.
 * ranging from 1 to 4 in the TIB and from 1 to 6 in the TOB. For the endcap 
 * detectors the detector plane is the wheel number with a sign in front to 
 * tell in which side the wheel is sitting.
 */
int APVGain::subdetectorPlane(uint32_t det_id, const TrackerTopology* topo) {
  if (topo) {
    if (APVGain::subdetectorId(det_id) == StripSubdetector::TIB)
      return topo->tibLayer(det_id);
    else if (APVGain::subdetectorId(det_id) == StripSubdetector::TID)
      return (2 * topo->tidSide(det_id) - 3) * topo->tidWheel(det_id);
    else if (APVGain::subdetectorId(det_id) == StripSubdetector::TOB)
      return topo->tobLayer(det_id);
    else if (APVGain::subdetectorId(det_id) == StripSubdetector::TEC)
      return (2 * topo->tecSide(det_id) - 3) * topo->tecWheel(det_id);
  }
  return 0;
};

/** Brief Extract from a char * the subdetector type.
 * Return an integer whioch is the detector plane where the module sits.
 * The char * string is expected to have the subdetector plane put at its
 * end after an "_" char.
 */
int APVGain::subdetectorPlane(const std::string& tag) {
  std::size_t p = (tag.find("layer") != std::string::npos) ? tag.find("layer") : tag.find("wheel");
  if (p != std::string::npos) {
    std::size_t start = tag.find('_', p + 1) + 1;
    std::size_t stop = tag.find('_', start);
    std::string plane = tag.substr(start, stop - start);
    return atoi(plane.c_str());
  }
  return 0;
};

/** Brief Fetch the Monitor Element corresponding to a DetId.
 *  */
std::vector<APVGain::MonitorElement*> APVGain::FetchMonitor(std::vector<APVGain::APVmon> histos,
                                                            uint32_t det_id,
                                                            const TrackerTopology* topo) {
  std::vector<MonitorElement*> found = std::vector<MonitorElement*>();
  int sThick = APVGain::thickness(det_id);
  int sId = APVGain::subdetectorId(det_id);
  int sPlane = APVGain::subdetectorPlane(det_id, topo);
  int sSide = APVGain::subdetectorSide(det_id, topo);
  auto it = histos.begin();

  LogDebug("APVGainHelpers") << "sId: " << sId << " sPlane: " << sPlane << " sSide: " << sSide << std::endl;

  while (it != histos.end()) {
    std::string tag = (*it).getMonitor()->getName();
    int subdetectorThickness = (*it).getThickness();
    int subdetectorId = (*it).getSubdetectorId();
    int subdetectorSide = (*it).getSubdetectorSide();
    int subdetectorPlane = (*it).getSubdetectorPlane();

    bool match = (subdetectorId == 0 || subdetectorId == sId) &&
                 (subdetectorPlane == 0 || subdetectorPlane == sPlane) &&
                 (subdetectorSide == 0 || subdetectorSide == sSide) &&
                 (subdetectorThickness == 0 || subdetectorThickness == sThick);

    if (match) {
      found.emplace_back((*it).getMonitor());
      LogDebug("APVGainHelpers") << det_id << " found: " << tag << std::endl;
      (*it).printAll();
    }
    it++;
  }
  return found;
}

/** Brief Fetch the Monitor Element index corresponding to a DetId.
 *  */
std::vector<unsigned int> APVGain::FetchIndices(std::map<unsigned int, APVloc> theMap,
                                                uint32_t det_id,
                                                const TrackerTopology* topo) {
  std::vector<unsigned int> found_indices = std::vector<unsigned int>();

  int sThick = APVGain::thickness(det_id);
  int sId = APVGain::subdetectorId(det_id);
  int sPlane = APVGain::subdetectorPlane(det_id, topo);
  int sSide = APVGain::subdetectorSide(det_id, topo);

  for (auto& element : theMap) {
    int subdetectorThickness = element.second.m_thickness;
    int subdetectorId = element.second.m_subdetectorId;
    int subdetectorSide = element.second.m_subdetectorSide;
    int subdetectorPlane = element.second.m_subdetectorPlane;

    bool match = (subdetectorId == 0 || subdetectorId == sId) &&
                 (subdetectorPlane == 0 || subdetectorPlane == sPlane) &&
                 (subdetectorSide == 0 || subdetectorSide == sSide) &&
                 (subdetectorThickness == 0 || subdetectorThickness == sThick);

    if (match) {
      found_indices.push_back(element.first);
    }
  }
  return found_indices;
}

std::vector<std::pair<std::string, std::string>> APVGain::monHnames(std::vector<std::string> VH,
                                                                    bool allPlanes,
                                                                    const char* tag) {
  std::vector<std::pair<std::string, std::string>> out;

  // total number of measurement layers/wheels in the Strips Tracker
  // 4(TIB) + 6(TOB) + 3(TID+) + 3(TID-) + 9(TEC+) + 9(TEC-)
  constexpr int countOfPlanes = 34;

  int re = (allPlanes) ? countOfPlanes + VH.size() : VH.size();
  out.reserve(re);

  std::string Tag = tag;
  if (Tag.length())
    Tag = "__" + Tag;

  std::string h_tag = "";
  std::string h_tit = "";

  if (allPlanes) {
    // Names of monitoring histogram for TIB layers
    constexpr int TIBlayers = 4;  //number of TIB layers.
    for (int i = 1; i <= TIBlayers; i++) {
      h_tag = "TIB_layer_" + std::to_string(i) + Tag;
      h_tit = h_tag;
      std::replace(h_tit.begin(), h_tit.end(), '_', ' ');
      out.push_back(std::pair<std::string, std::string>(h_tag, h_tit));
    }
    // Names of monitoring histogram for TOB layers
    constexpr int TOBlayers = 6;  //number of TOB layers
    for (int i = 1; i <= TOBlayers; i++) {
      h_tag = "TOB_layer_" + std::to_string(i) + Tag;
      h_tit = h_tag;
      std::replace(h_tit.begin(), h_tit.end(), '_', ' ');
      out.push_back(std::pair<std::string, std::string>(h_tag, h_tit));
    }
    // Names of monitoring histogram for TID wheels
    constexpr int TIDwheels = 3;  //number of TID wheels
    for (int i = -TIDwheels; i <= TIDwheels; i++) {
      if (i == 0)
        continue;
      if (i < 0)
        h_tag = "TIDminus_wheel_" + std::to_string(i) + Tag;
      else
        h_tag = "TIDplus_wheel_" + std::to_string(i) + Tag;
      h_tit = h_tag;
      std::replace(h_tit.begin(), h_tit.end(), '_', ' ');
      out.push_back(std::pair<std::string, std::string>(h_tag, h_tit));
    }
    // Names of monitoring histogram for TEC wheels
    constexpr int TECwheels = 9;  //number of TEC wheels
    for (int i = -TECwheels; i <= TECwheels; i++) {
      if (i == 0)
        continue;
      if (i < 0)
        h_tag = "TECminus_wheel_" + std::to_string(i) + Tag;
      else
        h_tag = "TECplus_wheel_" + std::to_string(i) + Tag;
      h_tit = h_tag;
      std::replace(h_tit.begin(), h_tit.end(), '_', ' ');
      out.push_back(std::pair<std::string, std::string>(h_tag, h_tit));
    }
  }

  for (unsigned int i = 0; i < VH.size(); i++) {
    h_tag = VH[i] + Tag;
    h_tit = h_tag;
    std::replace(h_tit.begin(), h_tit.end(), '_', ' ');
    out.push_back(std::pair<std::string, std::string>(h_tag, h_tit));
  }

  return out;
}
