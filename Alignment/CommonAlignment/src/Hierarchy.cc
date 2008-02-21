#include "Alignment/CommonAlignment/interface/MuonNameSpace.h"
#include "Alignment/CommonAlignment/interface/TRKNameSpace.h"
#include "Alignment/TrackerAlignment/interface/TECNameSpace.h"
#include "Alignment/TrackerAlignment/interface/TIBNameSpace.h"
#include "Alignment/TrackerAlignment/interface/TIDNameSpace.h"
#include "Alignment/TrackerAlignment/interface/TOBNameSpace.h"
#include "Alignment/TrackerAlignment/interface/TPBNameSpace.h"
#include "Alignment/TrackerAlignment/interface/TPENameSpace.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "Alignment/CommonAlignment/interface/Hierarchy.h"

std::string Hierarchy::theDetectors[];
std::vector<std::string> Hierarchy::theSubdetectors[];

Hierarchy::NameCounters Hierarchy::theNameCounters[][maxSubdetector];

Hierarchy::Hierarchy()
{
  using namespace align;

  if ("Unknown" == theDetectors[0]) return; // already initialised

  theDetectors[0] = "Unknown";

  theDetectors[DetId::Tracker] = "Tracker";
  theDetectors[DetId::Muon   ] = "Muon";
  theDetectors[DetId::Ecal   ] = "Ecal";
  theDetectors[DetId::Hcal   ] = "Hcal";
  theDetectors[DetId::Calo   ] = "Calo";

  theSubdetectors[DetId::Tracker].resize(trk::MAX);
  theSubdetectors[DetId::Tracker][0] = "Unknown";
  theSubdetectors[DetId::Tracker][trk::TPB] = "TPB";
  theSubdetectors[DetId::Tracker][trk::TPE] = "TPE";
  theSubdetectors[DetId::Tracker][trk::TIB] = "TIB";
  theSubdetectors[DetId::Tracker][trk::TID] = "TID";
  theSubdetectors[DetId::Tracker][trk::TOB] = "TOB";
  theSubdetectors[DetId::Tracker][trk::TEC] = "TEC";

  theSubdetectors[DetId::Muon].resize(muon::MAX);
  theSubdetectors[DetId::Muon][0] = "Unknown";
  theSubdetectors[DetId::Muon][muon::DT ] = "DT";
  theSubdetectors[DetId::Muon][muon::CSC] = "CSC";
  theSubdetectors[DetId::Muon][muon::RPC] = "RPC";

  theNameCounters[DetId::Tracker][trk::TPB].push_back( std::make_pair("Module"      , tpb::      moduleNumber) );
  theNameCounters[DetId::Tracker][trk::TPB].push_back( std::make_pair("Ladder"      , tpb::      ladderNumber) );
  theNameCounters[DetId::Tracker][trk::TPB].push_back( std::make_pair("Layer"       , tpb::       layerNumber) );
  theNameCounters[DetId::Tracker][trk::TPB].push_back( std::make_pair("HalfBarrel"  , tpb::  halfBarrelNumber) );

  theNameCounters[DetId::Tracker][trk::TPE].push_back( std::make_pair("Module"      , tpe::      moduleNumber) );
  theNameCounters[DetId::Tracker][trk::TPE].push_back( std::make_pair("Panel"       , tpe::       panelNumber) );
  theNameCounters[DetId::Tracker][trk::TPE].push_back( std::make_pair("Blade"       , tpe::       bladeNumber) );
  theNameCounters[DetId::Tracker][trk::TPE].push_back( std::make_pair("HalfDisk"    , tpe::    halfDiskNumber) );
  theNameCounters[DetId::Tracker][trk::TPE].push_back( std::make_pair("HalfCylinder", tpe::halfCylinderNumber) );
  theNameCounters[DetId::Tracker][trk::TPE].push_back( std::make_pair("Endcap"      , tpe::      endcapNumber) );

  theNameCounters[DetId::Tracker][trk::TIB].push_back( std::make_pair("Module"      , tib::      moduleNumber) );
  theNameCounters[DetId::Tracker][trk::TIB].push_back( std::make_pair("String"      , tib::      stringNumber) );
  theNameCounters[DetId::Tracker][trk::TIB].push_back( std::make_pair("Surface"     , tib::     surfaceNumber) );
  theNameCounters[DetId::Tracker][trk::TIB].push_back( std::make_pair("HalfShell"   , tib::   halfShellNumber) );
  theNameCounters[DetId::Tracker][trk::TIB].push_back( std::make_pair("Layer"       , tib::       layerNumber) );
  theNameCounters[DetId::Tracker][trk::TIB].push_back( std::make_pair("HalfBarrel"  , tib::  halfBarrelNumber) );

  theNameCounters[DetId::Tracker][trk::TID].push_back( std::make_pair("Module"      , tid::      moduleNumber) );
  theNameCounters[DetId::Tracker][trk::TID].push_back( std::make_pair("Side"        , tid::        sideNumber) );
  theNameCounters[DetId::Tracker][trk::TID].push_back( std::make_pair("Ring"        , tid::        ringNumber) );
  theNameCounters[DetId::Tracker][trk::TID].push_back( std::make_pair("Disk"        , tid::        diskNumber) );
  theNameCounters[DetId::Tracker][trk::TID].push_back( std::make_pair("Endcap"      , tid::      endcapNumber) );

  theNameCounters[DetId::Tracker][trk::TOB].push_back( std::make_pair("Module"      , tob::      moduleNumber) );
  theNameCounters[DetId::Tracker][trk::TOB].push_back( std::make_pair("Rod"         , tob::         rodNumber) );
  theNameCounters[DetId::Tracker][trk::TOB].push_back( std::make_pair("Layer"       , tob::       layerNumber) );
  theNameCounters[DetId::Tracker][trk::TOB].push_back( std::make_pair("HalfBarrel"  , tob::  halfBarrelNumber) );

  theNameCounters[DetId::Tracker][trk::TEC].push_back( std::make_pair("Module"      , tec::      moduleNumber) );
  theNameCounters[DetId::Tracker][trk::TEC].push_back( std::make_pair("Ring"        , tec::        ringNumber) );
  theNameCounters[DetId::Tracker][trk::TEC].push_back( std::make_pair("Petal"       , tec::       petalNumber) );
  theNameCounters[DetId::Tracker][trk::TEC].push_back( std::make_pair("Side"        , tec::        sideNumber) );
  theNameCounters[DetId::Tracker][trk::TEC].push_back( std::make_pair("Disk"        , tec::        diskNumber) );
  theNameCounters[DetId::Tracker][trk::TEC].push_back( std::make_pair("Endcap"      , tec::      endcapNumber) );

//   theNameCounters[DetId::Tracker][muon::DT ].push_back( std::make_pair("Layer"     ,  dt::     layerNumber) );
//   theNameCounters[DetId::Tracker][muon::DT ].push_back( std::make_pair("SuperLayer",  dt::superLayerNumber) );
//   theNameCounters[DetId::Tracker][muon::DT ].push_back( std::make_pair("Chamber"   ,  dt::   chamberNumber) );
//   theNameCounters[DetId::Tracker][muon::DT ].push_back( std::make_pair("Station"   ,  dt::   stationNumber) );
//   theNameCounters[DetId::Tracker][muon::DT ].push_back( std::make_pair("Wheel"     ,  dt::     wheelNumber) );
//   theNameCounters[DetId::Tracker][muon::DT ].push_back( std::make_pair("Barrel"    ,  dt::    barrelNumber) );
// 
//   theNameCounters[DetId::Tracker][muon::CSC].push_back( std::make_pair("Layer"     , csc::     layerNumber) );
//   theNameCounters[DetId::Tracker][muon::CSC].push_back( std::make_pair("Chamber"   , csc::   chamberNumber) );
//   theNameCounters[DetId::Tracker][muon::CSC].push_back( std::make_pair("Ring"      , csc::      ringNumber) );
//   theNameCounters[DetId::Tracker][muon::CSC].push_back( std::make_pair("Station"   , csc::   stationNumber) );
//   theNameCounters[DetId::Tracker][muon::CSC].push_back( std::make_pair("Endcap"    , csc::    endcapNumber) );
}

Hierarchy::Hierarchy(align::ID id)
{
  static Hierarchy hierarchy; // init names and counters in hierarchy

  DetId detId(id);

  theDet = detId.det();

  if (theDet >= maxDetector || theDet < 0)
  {
    throw cms::Exception("HierarchyError")
      << "Invalid detector number from Alignable ID " << id;
  }

  theSubdet = detId.subdetId();

  if (theSubdet >= theSubdetectors[theDet].size() || theSubdet < 0)
  {
    throw cms::Exception("HierarchyError")
      << "Invalid subdetector number from Alignable ID " << id;
  }
}
