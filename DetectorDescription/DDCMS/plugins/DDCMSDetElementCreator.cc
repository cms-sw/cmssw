#include "DD4hep/VolumeProcessor.h"

namespace dd4hep {
  
  /// DD4hep DetElement creator for the CMS geometry.
  /*  Heuristically assign DetElement structures to the sensitive volume pathes.
   *
   *  \author  M.Frank
   *  \version 1.0
   *  \ingroup DD4HEP_CORE
   */
  class DDCMSDetElementCreator : public PlacedVolumeProcessor  {
    Detector&    description;
    Atom         silicon;
    struct Data {
      PlacedVolume pv {0};
      DetElement   element {};
      bool         sensitive = false;
      bool         has_sensitive = false;
      int          vol_count = 0;
      int          daughter_count = 0;
      int          sensitive_count = 0;

      Data() = default;
      Data(PlacedVolume v) : pv(v) {}
      Data(const Data& d) = default;
      Data& operator=(const Data& d) = default;
    };
    struct Count {
      int elements = 0;
      int volumes = 0;
      int sensitives = 0;
      Count() = default;
      Count(const Count&) = default;
      Count& operator=(const Count&) = default;
    };
    typedef std::vector<Data> VolumeStack;
    typedef std::map<std::string,dd4hep::DetElement> Detectors;
    typedef std::map<dd4hep::DetElement,Count> Counters;
    typedef std::map<std::pair<dd4hep::DetElement,int>, std::pair<int,int> > LeafCount;

    Counters          counters;
    LeafCount         leafCount;
    VolumeStack       stack;
    Detectors         subdetectors;
    DetElement        tracker, current_detector;
    SensitiveDetector current_sensitive;
    std::map<PlacedVolume, std::pair<int,int> > all_placements;
    
    /// Add new subdetector to the detector description
    DetElement addSubdetector(const std::string& nam, PlacedVolume pv, bool volid);
    /// Create a new detector element
    DetElement createElement(const char* debug_tag, PlacedVolume pv, int id);
    /// Create the top level detectors
    void createTopLevelDetectors(PlacedVolume pv);

  public:
    /// Initializing constructor
    DDCMSDetElementCreator(Detector& desc);
    /// Default destructor
    virtual ~DDCMSDetElementCreator();
    /// Callback to output PlacedVolume information of an single Placement
    virtual int operator()(PlacedVolume pv, int level);
    /// Callback to output PlacedVolume information of an entire Placement
    virtual int process(PlacedVolume pv, int level, bool recursive);
  };
}


#include "DD4hep/detail/DetectorInterna.h"
#include "DD4hep/DetFactoryHelper.h"
#include "DD4hep/DetectorHelper.h"
#include "DD4hep/Printout.h"
#include "DDCMS/DDCMS.h"

#include <sstream>

using namespace std;
using namespace dd4hep;

/// Initializing constructor
DDCMSDetElementCreator::DDCMSDetElementCreator(Detector& desc)
  : description(desc)
{
  DetectorHelper helper(description);
  silicon = helper.element("SI");
  if ( !silicon.isValid() )   {
    except("DDCMSDetElementCreator",
           "++ Failed to extract SILICON from the element table.");
  }
  stack.reserve(32);
}

/// Default destructor
DDCMSDetElementCreator::~DDCMSDetElementCreator()   {
  Count total;
  stringstream str, id_str;

  printout(INFO,"DDCMSDetElementCreator","+++++++++++++++ Summary of sensitve elements  ++++++++++++++++++++++++");
  for ( const auto& c : counters )  {
    printout(INFO,"DDCMSDetElementCreator","++ Summary: SD: %-24s %7d DetElements %7d sensitives out of %7d volumes",
             (c.first.name()+string(":")).c_str(), c.second.elements, c.second.sensitives, c.second.volumes);
    total.elements   += c.second.elements;
    total.sensitives += c.second.sensitives;
    total.volumes    += c.second.volumes;
  }
  printout(INFO,"DDCMSDetElementCreator",  "++ Summary:     %-24s %7d DetElements %7d sensitives out of %7d volumes",
           "Grand Total:",total.elements,total.sensitives,total.volumes);
  printout(INFO,"DDCMSDetElementCreator","+++++++++++++++ Summary of geometry depth analysis  ++++++++++++++++++");
  int total_cnt = 0, total_depth = 0;
  map<DetElement, vector<pair<int,int> > > fields;
  for ( const auto& l : leafCount )  {
    DetElement de = l.first.first;
    printout(INFO,"DDCMSDetElementCreator","++ Summary: SD: %-24s system:%04X Lvl:%3d Sensitives: %6d [Max: %6d].",
             (de.name()+string(":")).c_str(), de.id(),
             l.first.second, l.second.second, l.second.first);
    fields[de].push_back(make_pair(l.first.second,l.second.first));
    total_depth += l.second.second;
    ++total_cnt;
  }
  printout(INFO,"DDCMSDetElementCreator","++ Summary:     %-24s  %d.","Total DetElements:",total_cnt);
  printout(INFO,"DDCMSDetElementCreator","+++++++++++++++ Readout structure generation  ++++++++++++++++++++++++");
  str << endl;
  for( const auto& f : fields )   {
    string ro_name = f.first.name() + string("Hits");
    int num_bits = 8;
    id_str.str("");
    id_str << "system:" << num_bits;
    for( const auto& q : f.second )   {
      int bits = 0;
      if      ( q.second < 1<<0  ) bits = 1;
      else if ( q.second < 1<<1  ) bits = 1;
      else if ( q.second < 1<<2  ) bits = 2;
      else if ( q.second < 1<<3  ) bits = 3;
      else if ( q.second < 1<<4  ) bits = 4;
      else if ( q.second < 1<<5  ) bits = 5;
      else if ( q.second < 1<<6  ) bits = 6;
      else if ( q.second < 1<<7  ) bits = 7;
      else if ( q.second < 1<<8  ) bits = 8;
      else if ( q.second < 1<<9  ) bits = 9;
      else if ( q.second < 1<<10 ) bits = 10;
      else if ( q.second < 1<<11 ) bits = 11;
      else if ( q.second < 1<<12 ) bits = 12;
      else if ( q.second < 1<<13 ) bits = 13;
      else if ( q.second < 1<<14 ) bits = 14;
      else if ( q.second < 1<<15 ) bits = 15;
      bits += 1;
      id_str << ",Lv" << q.first << ":" << bits;
      num_bits += bits;
    }
    string idspec = id_str.str();
    str << "<readout name=\"" << ro_name << "\">" << endl
        << "\t<id>"
        << idspec
        << "</id>  <!-- Number of bits: " << num_bits << " -->" << endl
        << "</readout>" << endl;
    /// Create ID Descriptors and readout configurations
    IDDescriptor dsc(ro_name,idspec);
    description.addIDSpecification(dsc);
    Readout ro(ro_name);
    ro.setIDDescriptor(dsc);
    description.addReadout(ro);
    SensitiveDetector sd = description.sensitiveDetector(f.first.name());
    sd.setHitsCollection(ro.name());
    sd.setReadout(ro);
    printout(INFO,"DDCMSDetElementCreator",
             "++ Setting up readout for subdetector:%-24s id:%04X",
             f.first.name(), f.first.id());
  }
  printout(INFO,"DDCMSDetElementCreator","+++++++++++++++ ID Descriptor generation  ++++++++++++++++++++++++++++");
  printout(INFO,"",str.str().c_str());
  char volid[32];
  for(auto& p : all_placements )  {
    PlacedVolume place = p.first;
    Volume vol = place.volume();
    ::snprintf(volid,sizeof(volid),"Lv%d", p.second.first);
    printout(DEBUG,"DDCMSDetElementCreator",
             "++ Set volid (%-24s): %-6s = %3d  -> %s  (%p)",
             vol.isSensitive() ? vol.sensitiveDetector().name() : "Not Sensitive",
             volid, p.second.second, place.name(), place.ptr());
    place.addPhysVolID(volid, p.second.second);
  }
  printout(ALWAYS,"DDCMSDetElementCreator",
           "++ Instrumented %ld subdetectors with %d DetElements %d sensitives out of %d volumes and %ld sensitive placements.",
           fields.size(),total.elements,total.sensitives,total.volumes,all_placements.size());
}

/// Create a new detector element
DetElement DDCMSDetElementCreator::createElement(const char* /* debug_tag */, PlacedVolume pv, int id) {
  string     name = cms::detElementName(pv);
  DetElement det(name, id);
  det.setPlacement(pv);
  /*
    printout(INFO,"DDCMSDetElementCreator","++ Created detector element [%s]: %s (%s)  %p",
    debug_tag, det.name(), name.c_str(), det.ptr());
  */
  return det;
}

/// Create the top level detectors
void DDCMSDetElementCreator::createTopLevelDetectors(PlacedVolume pv)   {
  auto& data = stack.back();
  if ( stack.size() == 2 )    {      // Main subssystem: tracker:Tracker
    data.element = tracker = addSubdetector(cms::detElementName(pv), pv, false);
    tracker->SetTitle("compound");
  }
  else if ( stack.size() == 3 )    { // Main subsystem detector: TIB, TEC, ....
    data.element = current_detector = addSubdetector(cms::detElementName(pv), pv, true);
  }
}

/// Add new subdetector to the detector description
DetElement DDCMSDetElementCreator::addSubdetector(const std::string& nam, PlacedVolume pv, bool volid)  {
  Detectors::iterator idet = subdetectors.find(nam);
  if ( idet == subdetectors.end() )   {
    DetElement det(nam, subdetectors.size()+1);
    det.setPlacement(pv);
    if ( volid )  {
      det.placement().addPhysVolID("system",det.id());
    }
    idet = subdetectors.insert(make_pair(nam,det)).first;
    description.add(det);
  }
  return idet->second;
}

/// Callback to output PlacedVolume information of an single Placement
int DDCMSDetElementCreator::operator()(PlacedVolume pv, int vol_level)   {
  double frac_si = pv.volume().material().fraction(silicon);
  if ( frac_si > 90e-2 )  {
    Data& data = stack.back();
    data.sensitive     = true;
    data.has_sensitive = true;
    ++data.vol_count;
    int   idx   = pv->GetMotherVolume()->GetIndex(pv.ptr())+1;
    auto& cnt   = leafCount[make_pair(current_detector,vol_level)];
    cnt.first   = std::max(cnt.first,idx);
    ++cnt.second;
    all_placements[pv] = make_pair(vol_level,idx);
    return 1;
  }
  return 0;
}

/// Callback to output PlacedVolume information of an entire Placement
int DDCMSDetElementCreator::process(PlacedVolume pv, int level, bool recursive)   {
  stack.push_back(Data(pv));
  if ( stack.size() <= 3 )   {
    createTopLevelDetectors(pv);
  }
  int ret = PlacedVolumeProcessor::process(pv,level,recursive);

  /// Complete structures if the stack size is > 3!
  if ( stack.size() > 3 )   {
    // Note: short-cuts to entries in the stack MUST be local and
    // initialized AFTER the call to "process"! The vector may be resized!
    auto& data = stack.back();
    auto& parent = stack[stack.size()-2];
    auto& counts = counters[current_detector];
    if ( data.sensitive )   {
      /// If this volume is sensitve, we must attach a sensitive detector handle
      if ( !current_sensitive.isValid() )  {
        SensitiveDetector sd = description.sensitiveDetector(current_detector.name());
        if ( !sd.isValid() )  {
          sd = SensitiveDetector(current_detector.name(),"tracker");
          current_detector->flag |= DetElement::Object::HAVE_SENSITIVE_DETECTOR;
          description.add(sd);
        }
        current_sensitive = sd;
      }
      pv.volume().setSensitiveDetector(current_sensitive);
      ++counts.sensitives;
    }
    ++counts.volumes;
    bool added = false;
    if ( data.vol_count > 0 )   {
      parent.daughter_count  += data.vol_count;
      parent.daughter_count  += data.daughter_count;
      data.has_sensitive      = true;
    }
    else   {
      parent.daughter_count  += data.daughter_count;
      data.has_sensitive      = (data.daughter_count>0);
    }

    if ( data.has_sensitive )  {
      // If we have sensitive elements at this level or below,
      // we must complete the DetElement hierarchy
      if ( !data.element.isValid() )  {
        data.element = createElement("Element", data.pv, current_detector.id());
        ++counts.elements;
      }
      if ( !parent.element.isValid() )  {
        parent.element = createElement("Parent ", parent.pv, current_detector.id());
        ++counts.elements;
      }
      printout(DEBUG,"DDCMSDetElementCreator",
               "++ Assign detector element: %s (%p, %ld children) to %s (%p) with %ld vols",
               data.element.name(), data.element.ptr(), data.element.children().size(),
               parent.element.name(), parent.element.ptr(),
               data.vol_count);

      // Trickle up the tree only for sensitive pathes. Forget the passive rest
      // This should automatically omit non-sensitive pathes
      parent.has_sensitive = true;
      parent.element.add(data.element);
      added = true;
      // It is simpler to collect the volumes and later assign the volids
      // rather than checking if the volid already exists.
      int vol_level = level;
      int idx = data.pv->GetMotherVolume()->GetIndex(data.pv.ptr())+1;
      all_placements[data.pv] = make_pair(vol_level,idx); // 1...n
      // Update counters
      auto& cnt_det   = leafCount[make_pair(current_detector,vol_level)];
      cnt_det.first   = std::max(cnt_det.first,idx);
      cnt_det.second += 1;
    }
    if ( !added && data.element.isValid() )  {
      printout(WARNING,"MEMORY-LEAK","Level:%3d Orpahaned DetElement:%s Daugthers:%d Parent:%s",
               int(stack.size()), data.element.name(), data.vol_count, parent.pv.name());
    }
  }
  /// Now the cleanup kicks in....
  if ( stack.size() == 3 )  {
    current_sensitive = SensitiveDetector();
    current_detector = DetElement();
    ret = 0;
  }
  stack.pop_back();
  return ret;
}

static void* create_object(Detector& description, int /* argc */, char** /* argv */)   {
  PlacedVolumeProcessor* proc = new DDCMSDetElementCreator(description);
  return (void*)proc;
}

// first argument is the type from the xml file
DECLARE_DD4HEP_CONSTRUCTOR(DDCMS_DetElementCreator,create_object)

