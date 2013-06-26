#if !defined(__CINT__) && !defined(__MAKECINT__)
#define CondFormats_Alignment_Definitions_H // avoid including this header

namespace align { typedef unsigned int ID; }

#include "Alignment/TrackerAlignment/interface/TPBNameSpace.h"
#include "Alignment/TrackerAlignment/interface/TPENameSpace.h"
#include "Alignment/TrackerAlignment/interface/TIBNameSpace.h"
#include "Alignment/TrackerAlignment/interface/TIDNameSpace.h"
#include "Alignment/TrackerAlignment/interface/TOBNameSpace.h"
#include "Alignment/TrackerAlignment/interface/TECNameSpace.h"
#endif

#include <sstream>
#include <string>
#include <vector>
#include <utility>

namespace align
{
  namespace trk
  {
    enum { TPB = 1, TPE = 2, TIB = 3, TID = 4, TOB = 5, TEC = 6, MAX = 7 };

    unsigned int subdetId(unsigned int id)
    {
      return (id >> 25) & 0x7;
    }
  }
}

// Extract the id from histogram's title

unsigned int getId(const std::string& title)
{
  unsigned int id;

  std::stringstream ss(title.substr(title.find_last_of(' ')));

  ss >> id;

  return id;
}

typedef unsigned int (*Counter)(unsigned int);
typedef std::pair<Counter, std::string> CounterName;
typedef std::vector<CounterName> CounterNames;

class Helper
{
  public:

  inline static const CounterNames& counterNames(unsigned int subdetId);

  static std::string name(unsigned int id);

  private:

  Helper(); // init counters and names

  CounterNames theCounterNames[align::trk::MAX];
};

const CounterNames& Helper::counterNames(unsigned int subdetId)
{
  static const Helper helper;

  return helper.theCounterNames[subdetId];
}

std::string Helper::name(unsigned int id)
{
  std::ostringstream os;

  const CounterNames& cn = counterNames(align::trk::subdetId(id));

  for (unsigned int i = 0; i < cn.size(); ++i)
  {
    os << cn[i].second << ' ' << cn[i].first(id) << ' ';
  }

  return os.str();
}

Helper::Helper()
{
  using namespace align;

  // Barrel Pixel
  theCounterNames[trk::TPB].push_back(std::make_pair(trk::          subdetId, "TPB"         ));
  theCounterNames[trk::TPB].push_back(std::make_pair(tpb::  halfBarrelNumber, "HalfBarrel"  ));
  theCounterNames[trk::TPB].push_back(std::make_pair(tpb::       layerNumber, "Layer"       ));
  theCounterNames[trk::TPB].push_back(std::make_pair(tpb::      ladderNumber, "Ladder"      ));
  theCounterNames[trk::TPB].push_back(std::make_pair(tpb::      moduleNumber, "Module"      ));

  // Forward Pixel
  theCounterNames[trk::TPE].push_back(std::make_pair(trk::          subdetId, "TPE"         ));
  theCounterNames[trk::TPE].push_back(std::make_pair(tpe::      endcapNumber, "Endcap"      ));
  theCounterNames[trk::TPE].push_back(std::make_pair(tpe::halfCylinderNumber, "HalfCylinder"));
  theCounterNames[trk::TPE].push_back(std::make_pair(tpe::    halfDiskNumber, "HalfDisk"    ));
  theCounterNames[trk::TPE].push_back(std::make_pair(tpe::       bladeNumber, "Blade"       ));
  theCounterNames[trk::TPE].push_back(std::make_pair(tpe::       panelNumber, "Panel"       ));
  theCounterNames[trk::TPE].push_back(std::make_pair(tpe::      moduleNumber, "Module"      ));

  // Tracker Inner Barrel
  theCounterNames[trk::TIB].push_back(std::make_pair(trk::          subdetId, "TIB"         ));
  theCounterNames[trk::TIB].push_back(std::make_pair(tib::  halfBarrelNumber, "HalfBarrel"  ));
  theCounterNames[trk::TIB].push_back(std::make_pair(tib::       layerNumber, "Layer"       ));
  theCounterNames[trk::TIB].push_back(std::make_pair(tib::   halfShellNumber, "HalfShell"   ));
  theCounterNames[trk::TIB].push_back(std::make_pair(tib::     surfaceNumber, "Surface"     ));
  theCounterNames[trk::TIB].push_back(std::make_pair(tib::      stringNumber, "String"      ));
  theCounterNames[trk::TIB].push_back(std::make_pair(tib::      moduleNumber, "Module"      ));

  // Tracker Inner Disks
  theCounterNames[trk::TID].push_back(std::make_pair(trk::          subdetId, "TID"         ));
  theCounterNames[trk::TID].push_back(std::make_pair(tid::      endcapNumber, "Endcap"      ));
  theCounterNames[trk::TID].push_back(std::make_pair(tid::        diskNumber, "Disk"        ));
  theCounterNames[trk::TID].push_back(std::make_pair(tid::        ringNumber, "Ring"        ));
  theCounterNames[trk::TID].push_back(std::make_pair(tid::        sideNumber, "Side"        ));
  theCounterNames[trk::TID].push_back(std::make_pair(tid::      moduleNumber, "Module"      ));

  // Tracker Outer Barrel
  theCounterNames[trk::TOB].push_back(std::make_pair(trk::          subdetId, "TOB"         ));
  theCounterNames[trk::TOB].push_back(std::make_pair(tob::  halfBarrelNumber, "HalfBarrel"  ));
  theCounterNames[trk::TOB].push_back(std::make_pair(tob::       layerNumber, "Layer"       ));
  theCounterNames[trk::TOB].push_back(std::make_pair(tob::         rodNumber, "Rod"         ));
  theCounterNames[trk::TOB].push_back(std::make_pair(tob::      moduleNumber, "Module"      ));

  // Tracker Endcaps
  theCounterNames[trk::TEC].push_back(std::make_pair(trk::          subdetId, "TEC"         ));
  theCounterNames[trk::TEC].push_back(std::make_pair(tec::      endcapNumber, "Endcap"      ));
  theCounterNames[trk::TEC].push_back(std::make_pair(tec::        diskNumber, "Disk"        ));
  theCounterNames[trk::TEC].push_back(std::make_pair(tec::        sideNumber, "Side"        ));
  theCounterNames[trk::TEC].push_back(std::make_pair(tec::       petalNumber, "Petal"       ));
  theCounterNames[trk::TEC].push_back(std::make_pair(tec::        ringNumber, "Ring"        ));
  theCounterNames[trk::TEC].push_back(std::make_pair(tec::      moduleNumber, "Module"      ));
}
