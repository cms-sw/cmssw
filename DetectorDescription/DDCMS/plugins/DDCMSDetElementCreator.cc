#include "DD4hep/VolumeProcessor.h"
#include "DD4hep/detail/DetectorInterna.h"
#include "DD4hep/DetFactoryHelper.h"
#include "DD4hep/DetectorHelper.h"
#include "DD4hep/Printout.h"
#include "DetectorDescription/DDCMS/interface/DDAlgoArguments.h"

#include <sstream>

using namespace std;
using namespace cms;
using namespace dd4hep;

namespace cms {

  // Heuristically assign DetElement structures
  // to the sensitive volume pathes
  //
  class DDCMSDetElementCreator : public dd4hep::PlacedVolumeProcessor {
  public:
    DDCMSDetElementCreator(dd4hep::Detector&);
    ~DDCMSDetElementCreator() override;

    /// Callback to output PlacedVolume information of an single Placement
    int operator()(dd4hep::PlacedVolume volume, int level) override;
    /// Callback to output PlacedVolume information of an entire Placement
    int process(dd4hep::PlacedVolume volume, int level, bool recursive) override;

  private:
    dd4hep::DetElement addSubdetector(const std::string& nam, dd4hep::PlacedVolume volume, bool valid);
    dd4hep::DetElement createElement(const char* debugTag, dd4hep::PlacedVolume volume, int id);
    void createTopLevelDetectors(dd4hep::PlacedVolume volume);

    struct Data {
      Data() = default;
      Data(dd4hep::PlacedVolume v) : volume(v) {}
      Data(const Data& d) = default;
      Data& operator=(const Data& d) = default;

      dd4hep::PlacedVolume volume{nullptr};
      dd4hep::DetElement element{};
      bool sensitive = false;
      bool hasSensitive = false;
      int volumeCount = 0;
      int daughterCount = 0;
      int sensitiveCount = 0;
    };

    struct Count {
      Count() = default;
      Count(const Count&) = default;
      Count& operator=(const Count&) = default;

      int elements = 0;
      int volumes = 0;
      int sensitives = 0;
    };

    using Detectors = std::map<std::string, dd4hep::DetElement>;
    using Counters = std::map<dd4hep::DetElement, Count>;
    using LeafCount = std::map<std::pair<dd4hep::DetElement, int>, std::pair<int, int> >;
    using VolumeStack = std::vector<Data>;

    std::map<dd4hep::PlacedVolume, std::pair<int, int> > m_allPlacements;

    Counters m_counters;
    LeafCount m_leafCount;
    VolumeStack m_stack;
    Detectors m_subdetectors;
    dd4hep::DetElement m_tracker, m_currentDetector;
    dd4hep::SensitiveDetector m_currentSensitive;
    dd4hep::Detector& m_description;
    dd4hep::Atom m_silicon;
  };

  std::string detElementName(dd4hep::PlacedVolume volume);
}  // namespace cms

std::string cms::detElementName(dd4hep::PlacedVolume volume) {
  if (volume.isValid()) {
    std::string name = volume.name();
    std::string nnam = name.substr(name.find(NAMESPACE_SEP) + 1);
    return nnam;
  }
  except("DD4CMS", "++ Cannot deduce name from invalid PlacedVolume handle!");
  return std::string();
}

DDCMSDetElementCreator::DDCMSDetElementCreator(dd4hep::Detector& desc) : m_description(desc) {
  dd4hep::DetectorHelper helper(m_description);
  m_silicon = helper.element("SI");
  if (!m_silicon.isValid()) {
    except("DDCMSDetElementCreator", "++ Failed to extract SILICON from the element table.");
  }
  m_stack.reserve(32);
}

DDCMSDetElementCreator::~DDCMSDetElementCreator() {
  Count total;
  stringstream str, id_str;

  printout(INFO, "DDCMSDetElementCreator", "+++++++++++++++ Summary of sensitve elements  ++++++++++++++++++++++++");
  for (const auto& c : m_counters) {
    printout(INFO,
             "DDCMSDetElementCreator",
             "++ Summary: SD: %-24s %7d DetElements %7d sensitives out of %7d volumes",
             (c.first.name() + string(":")).c_str(),
             c.second.elements,
             c.second.sensitives,
             c.second.volumes);
    total.elements += c.second.elements;
    total.sensitives += c.second.sensitives;
    total.volumes += c.second.volumes;
  }
  printout(INFO,
           "DDCMSDetElementCreator",
           "++ Summary:     %-24s %7d DetElements %7d sensitives out of %7d volumes",
           "Grand Total:",
           total.elements,
           total.sensitives,
           total.volumes);
  printout(INFO, "DDCMSDetElementCreator", "+++++++++++++++ Summary of geometry depth analysis  ++++++++++++++++++");
  int totalCount = 0, totalDepth = 0;
  map<dd4hep::DetElement, vector<pair<int, int> > > fields;
  for (const auto& l : m_leafCount) {
    dd4hep::DetElement de = l.first.first;
    printout(INFO,
             "DDCMSDetElementCreator",
             "++ Summary: SD: %-24s system:%04X Lvl:%3d Sensitives: %6d [Max: %6d].",
             (de.name() + string(":")).c_str(),
             de.id(),
             l.first.second,
             l.second.second,
             l.second.first);
    fields[de].push_back(make_pair(l.first.second, l.second.first));
    totalDepth += l.second.second;
    ++totalCount;
  }
  printout(INFO, "DDCMSDetElementCreator", "++ Summary:     %-24s  %d.", "Total DetElements:", totalCount);
  printout(INFO, "DDCMSDetElementCreator", "+++++++++++++++ Readout structure generation  ++++++++++++++++++++++++");
  str << endl;
  for (const auto& f : fields) {
    string roName = f.first.name() + string("Hits");
    int num_bits = 8;
    id_str.str("");
    id_str << "system:" << num_bits;
    for (const auto& q : f.second) {
      int bits = 0;
      if (q.second < 1 << 0)
        bits = 1;
      else if (q.second < 1 << 1)
        bits = 1;
      else if (q.second < 1 << 2)
        bits = 2;
      else if (q.second < 1 << 3)
        bits = 3;
      else if (q.second < 1 << 4)
        bits = 4;
      else if (q.second < 1 << 5)
        bits = 5;
      else if (q.second < 1 << 6)
        bits = 6;
      else if (q.second < 1 << 7)
        bits = 7;
      else if (q.second < 1 << 8)
        bits = 8;
      else if (q.second < 1 << 9)
        bits = 9;
      else if (q.second < 1 << 10)
        bits = 10;
      else if (q.second < 1 << 11)
        bits = 11;
      else if (q.second < 1 << 12)
        bits = 12;
      else if (q.second < 1 << 13)
        bits = 13;
      else if (q.second < 1 << 14)
        bits = 14;
      else if (q.second < 1 << 15)
        bits = 15;
      bits += 1;
      id_str << ",Lv" << q.first << ":" << bits;
      num_bits += bits;
    }
    string idspec = id_str.str();
    str << "<readout name=\"" << roName << "\">" << endl
        << "\t<id>" << idspec << "</id>  <!-- Number of bits: " << num_bits << " -->" << endl
        << "</readout>" << endl;

    /// Create ID Descriptors and readout configurations
    IDDescriptor dsc(roName, idspec);
    m_description.addIDSpecification(dsc);
    Readout ro(roName);
    ro.setIDDescriptor(dsc);
    m_description.addReadout(ro);
    dd4hep::SensitiveDetector sd = m_description.sensitiveDetector(f.first.name());
    sd.setHitsCollection(ro.name());
    sd.setReadout(ro);
    printout(INFO,
             "DDCMSDetElementCreator",
             "++ Setting up readout for subdetector:%-24s id:%04X",
             f.first.name(),
             f.first.id());
  }
  printout(INFO, "DDCMSDetElementCreator", "+++++++++++++++ ID Descriptor generation  ++++++++++++++++++++++++++++");
  printout(INFO, "", str.str().c_str());
  char volId[32];
  for (auto& p : m_allPlacements) {
    dd4hep::PlacedVolume place = p.first;
    dd4hep::Volume volume = place.volume();
    ::snprintf(volId, sizeof(volId), "Lv%d", p.second.first);
    printout(DEBUG,
             "DDCMSDetElementCreator",
             "++ Set volid (%-24s): %-6s = %3d  -> %s  (%p)",
             volume.isSensitive() ? volume.sensitiveDetector().name() : "Not Sensitive",
             volId,
             p.second.second,
             place.name(),
             place.ptr());
    place.addPhysVolID(volId, p.second.second);
  }
  printout(ALWAYS,
           "DDCMSDetElementCreator",
           "++ Instrumented %ld subdetectors with %d DetElements %d sensitives out of %d volumes and %ld sensitive "
           "placements.",
           fields.size(),
           total.elements,
           total.sensitives,
           total.volumes,
           m_allPlacements.size());
}

dd4hep::DetElement DDCMSDetElementCreator::createElement(const char*, PlacedVolume volume, int id) {
  string name = detElementName(volume);
  dd4hep::DetElement det(name, id);
  det.setPlacement(volume);
  return det;
}
void DDCMSDetElementCreator::createTopLevelDetectors(PlacedVolume volume) {
  auto& data = m_stack.back();
  if (m_stack.size() == 2) {  // Main subssystem: tracker:Tracker
    data.element = m_tracker = addSubdetector(cms::detElementName(volume), volume, false);
    m_tracker->SetTitle("compound");
  } else if (m_stack.size() == 3) {  // Main subsystem detector: TIB, TEC, ....
    data.element = m_currentDetector = addSubdetector(cms::detElementName(volume), volume, true);
  }
}

dd4hep::DetElement DDCMSDetElementCreator::addSubdetector(const std::string& nam,
                                                          dd4hep::PlacedVolume volume,
                                                          bool valid) {
  auto idet = m_subdetectors.find(nam);
  if (idet == m_subdetectors.end()) {
    dd4hep::DetElement det(nam, m_subdetectors.size() + 1);
    det.setPlacement(volume);
    if (valid) {
      det.placement().addPhysVolID("system", det.id());
    }
    idet = m_subdetectors.insert(make_pair(nam, det)).first;
    m_description.add(det);
  }
  return idet->second;
}

int DDCMSDetElementCreator::operator()(dd4hep::PlacedVolume volume, int volumeLevel) {
  double fracSi = volume.volume().material().fraction(m_silicon);
  if (fracSi > 90e-2) {
    Data& data = m_stack.back();
    data.sensitive = true;
    data.hasSensitive = true;
    ++data.volumeCount;
    int idx = volume->GetMotherVolume()->GetIndex(volume.ptr()) + 1;
    auto& cnt = m_leafCount[make_pair(m_currentDetector, volumeLevel)];
    cnt.first = std::max(cnt.first, idx);
    ++cnt.second;
    m_allPlacements[volume] = make_pair(volumeLevel, idx);
    return 1;
  }
  return 0;
}

int DDCMSDetElementCreator::process(dd4hep::PlacedVolume volume, int level, bool recursive) {
  m_stack.push_back(Data(volume));
  if (m_stack.size() <= 3) {
    createTopLevelDetectors(volume);
  }
  int ret = dd4hep::PlacedVolumeProcessor::process(volume, level, recursive);

  /// Complete structures if the m_stack size is > 3!
  if (m_stack.size() > 3) {
    // Note: short-cuts to entries in the m_stack MUST be local and
    // initialized AFTER the call to "process"! The vector may be resized!
    auto& data = m_stack.back();
    auto& parent = m_stack[m_stack.size() - 2];
    auto& counts = m_counters[m_currentDetector];
    if (data.sensitive) {
      /// If this volume is sensitve, we must attach a sensitive detector handle
      if (!m_currentSensitive.isValid()) {
        dd4hep::SensitiveDetector sd = m_description.sensitiveDetector(m_currentDetector.name());
        if (!sd.isValid()) {
          sd = dd4hep::SensitiveDetector(m_currentDetector.name(), "tracker");
          m_currentDetector->flag |= DetElement::Object::HAVE_SENSITIVE_DETECTOR;
          m_description.add(sd);
        }
        m_currentSensitive = sd;
      }
      volume.volume().setSensitiveDetector(m_currentSensitive);
      ++counts.sensitives;
    }
    ++counts.volumes;
    bool added = false;
    if (data.volumeCount > 0) {
      parent.daughterCount += data.volumeCount;
      parent.daughterCount += data.daughterCount;
      data.hasSensitive = true;
    } else {
      parent.daughterCount += data.daughterCount;
      data.hasSensitive = (data.daughterCount > 0);
    }

    if (data.hasSensitive) {
      // If we have sensitive elements at this level or below,
      // we must complete the DetElement hierarchy
      if (!data.element.isValid()) {
        data.element = createElement("Element", data.volume, m_currentDetector.id());
        ++counts.elements;
      }
      if (!parent.element.isValid()) {
        parent.element = createElement("Parent ", parent.volume, m_currentDetector.id());
        ++counts.elements;
      }
      printout(DEBUG,
               "DDCMSDetElementCreator",
               "++ Assign detector element: %s (%p, %ld children) to %s (%p) with %ld vols",
               data.element.name(),
               data.element.ptr(),
               data.element.children().size(),
               parent.element.name(),
               parent.element.ptr(),
               data.volumeCount);

      // Trickle up the tree only for sensitive pathes. Forget the passive rest
      // This should automatically omit non-sensitive pathes
      parent.hasSensitive = true;
      parent.element.add(data.element);
      added = true;
      // It is simpler to collect the volumes and later assign the volids
      // rather than checking if the volid already exists.
      int volumeLevel = level;
      int idx = data.volume->GetMotherVolume()->GetIndex(data.volume.ptr()) + 1;
      m_allPlacements[data.volume] = make_pair(volumeLevel, idx);  // 1...n
      // Update counters
      auto& cnt_det = m_leafCount[make_pair(m_currentDetector, volumeLevel)];
      cnt_det.first = std::max(cnt_det.first, idx);
      cnt_det.second += 1;
    }
    if (!added && data.element.isValid()) {
      printout(WARNING,
               "MEMORY-LEAK",
               "Level:%3d Orpahaned DetElement:%s Daugthers:%d Parent:%s",
               int(m_stack.size()),
               data.element.name(),
               data.volumeCount,
               parent.volume.name());
    }
  }
  /// Now the cleanup kicks in....
  if (m_stack.size() == 3) {
    m_currentSensitive = SensitiveDetector();
    m_currentDetector = DetElement();
    ret = 0;
  }
  m_stack.pop_back();
  return ret;
}

static void* createObject(dd4hep::Detector& description, int /* argc */, char** /* argv */) {
  dd4hep::PlacedVolumeProcessor* proc = new DDCMSDetElementCreator(description);
  return (void*)proc;
}

// first argument is the type from the xml file
DECLARE_DD4HEP_CONSTRUCTOR(DDCMS_DetElementCreator, createObject);
