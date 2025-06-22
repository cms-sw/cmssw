// -*- C++ -*-
//
// Package:     SiStripCommon
// Class  :     SiStripFolderOrganizer
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  dkcira
//         Created:  Thu Jan 26 23:52:43 CET 2006

// system includes
#include <cstring>  // For strlen
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

// user includes
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#define CONTROL_FOLDER_NAME "ControlView"
#define MECHANICAL_FOLDER_NAME "MechanicalView"
#define SEP "/"

SiStripFolderOrganizer::SiStripFolderOrganizer() {
  TopFolderName = "SiStrip";
  // get a pointer to DQMStore
  dbe_ = edm::Service<DQMStore>().operator->();
}

SiStripFolderOrganizer::~SiStripFolderOrganizer() {}

void SiStripFolderOrganizer::setSiStripFolderName(std::string name) { TopFolderName = name; }

std::string SiStripFolderOrganizer::getSiStripFolder() { return TopFolderName; }

void SiStripFolderOrganizer::setSiStripFolder() {
  dbe_->setCurrentFolder(TopFolderName);
  return;
}

std::string SiStripFolderOrganizer::getSiStripTopControlFolder() {
  std::string lokal_folder = TopFolderName + CONTROL_FOLDER_NAME;
  return lokal_folder;
}

void SiStripFolderOrganizer::setSiStripTopControlFolder() {
  std::string lokal_folder = TopFolderName + CONTROL_FOLDER_NAME;
  dbe_->setCurrentFolder(lokal_folder);
  return;
}

std::string SiStripFolderOrganizer::getSiStripControlFolder(
    // unsigned short crate,
    unsigned short slot,
    unsigned short ring,
    unsigned short addr,
    unsigned short chan
    // unsigned short i2c
) {
  std::stringstream lokal_folder;
  lokal_folder << getSiStripTopControlFolder();
  //   if ( crate != all_ ) {// if ==all_ then remain in top control folder
  //     lokal_folder << SEP << "FecCrate" << crate;
  if (slot != all_) {
    lokal_folder << SEP << "FecSlot" << slot;
    if (ring != all_) {
      lokal_folder << SEP << "FecRing" << ring;
      if (addr != all_) {
        lokal_folder << SEP << "CcuAddr" << addr;
        if (chan != all_) {
          lokal_folder << SEP << "CcuChan" << chan;
          // 	    if ( i2c != all_ ) {
          // 	      lokal_folder << SEP << "I2cAddr" << i2c;
          // 	    }
        }
      }
    }
  }
  //   }
  std::string folder_name = lokal_folder.str();
  return folder_name;
}

void SiStripFolderOrganizer::setSiStripControlFolder(
    // unsigned short crate,
    unsigned short slot,
    unsigned short ring,
    unsigned short addr,
    unsigned short chan
    // unsigned short i2c
) {
  std::string lokal_folder = getSiStripControlFolder(slot, ring, addr, chan);
  dbe_->setCurrentFolder(lokal_folder);
  return;
}

std::pair<std::string, int32_t> SiStripFolderOrganizer::GetSubDetAndLayer(const uint32_t& detid,
                                                                          const TrackerTopology* tTopo,
                                                                          bool ring_flag) {
  std::string cSubDet;
  int32_t layer = 0;
  switch (StripSubdetector::SubDetector(StripSubdetector(detid).subdetId())) {
    case StripSubdetector::TIB:
      cSubDet = "TIB";
      layer = tTopo->tibLayer(detid);
      break;
    case StripSubdetector::TOB:
      cSubDet = "TOB";
      layer = tTopo->tobLayer(detid);
      break;
    case StripSubdetector::TID:
      cSubDet = "TID";
      if (ring_flag)
        layer = tTopo->tidRing(detid) * (tTopo->tidSide(detid) == 1 ? -1 : +1);
      else
        layer = tTopo->tidWheel(detid) * (tTopo->tidSide(detid) == 1 ? -1 : +1);
      break;
    case StripSubdetector::TEC:
      cSubDet = "TEC";
      if (ring_flag)
        layer = tTopo->tecRing(detid) * (tTopo->tecSide(detid) == 1 ? -1 : +1);
      else
        layer = tTopo->tecWheel(detid) * (tTopo->tecSide(detid) == 1 ? -1 : +1);
      break;
    default:
      edm::LogWarning("SiStripMonitorTrack") << "WARNING!!! this detid does not belong to tracker" << std::endl;
  }
  return std::make_pair(cSubDet, layer);
}

std::pair<std::string, int32_t> SiStripFolderOrganizer::GetSubDetAndLayerThickness(const uint32_t& detid,
                                                                                   const TrackerTopology* tTopo,
                                                                                   std::string& cThickness) {
  std::string cSubDet;
  int32_t layer = 0;
  int32_t ring = 0;
  switch (StripSubdetector::SubDetector(StripSubdetector(detid).subdetId())) {
    case StripSubdetector::TIB:
      cSubDet = "TIB";
      layer = tTopo->tibLayer(detid);
      cThickness = "THIN";
      break;
    case StripSubdetector::TOB:
      cSubDet = "TOB";
      layer = tTopo->tobLayer(detid);
      cThickness = "THICK";
      break;
    case StripSubdetector::TID:
      cSubDet = "TID";
      layer = tTopo->tidWheel(detid) * (tTopo->tidSide(detid) == 1 ? -1 : +1);
      cThickness = "THIN";
      break;
    case StripSubdetector::TEC:
      cSubDet = "TEC";
      layer = tTopo->tecWheel(detid) * (tTopo->tecSide(detid) == 1 ? -1 : +1);
      ring = tTopo->tecRing(detid) * (tTopo->tecSide(detid) == 1 ? -1 : +1);
      if (ring >= 1 && ring <= 4)
        cThickness = "THIN";
      else
        cThickness = "THICK";
      break;
    default:
      edm::LogWarning("SiStripMonitorTrack") << "WARNING!!! this detid does not belong to tracker" << std::endl;
  }
  return std::make_pair(cSubDet, layer);
}

std::pair<std::string, int32_t> SiStripFolderOrganizer::GetSubDetAndRing(const uint32_t& detid,
                                                                         const TrackerTopology* tTopo) {
  std::string cSubDet;
  int32_t ring = 0;
  switch (StripSubdetector::SubDetector(StripSubdetector(detid).subdetId())) {
    case StripSubdetector::TIB:
      cSubDet = "TIB";
      break;
    case StripSubdetector::TOB:
      cSubDet = "TOB";
      break;
    case StripSubdetector::TID:
      cSubDet = "TID";
      ring = tTopo->tidRing(detid) * (tTopo->tidSide(detid) == 1 ? -1 : +1);
      break;
    case StripSubdetector::TEC:
      cSubDet = "TEC";
      ring = tTopo->tecRing(detid) * (tTopo->tecSide(detid) == 1 ? -1 : +1);
      break;
    default:
      edm::LogWarning("SiStripMonitorTrack") << "WARNING!!! this detid does not belong to tracker" << std::endl;
  }
  return std::make_pair(cSubDet, ring);
}

void SiStripFolderOrganizer::setDetectorFolder(uint32_t rawdetid, const TrackerTopology* tTopo) {
  std::string folder_name;
  getFolderName(rawdetid, tTopo, folder_name);
  dbe_->setCurrentFolder(folder_name);
}

void SiStripFolderOrganizer::getSubDetLayerFolderName(std::stringstream& ss,
                                                      SiStripDetId::SubDetector subDet,
                                                      uint32_t layer,
                                                      uint32_t side) {
  //  std::cout << "[SiStripFolderOrganizer::getSubDetLayerFolderName] TopFolderName: " << TopFolderName << std::endl;
  ss << TopFolderName << SEP << MECHANICAL_FOLDER_NAME;

  std::stringstream sside;
  if (side == 1) {
    sside << "MINUS";
  } else if (side == 2) {
    sside << "PLUS";
  }

  if (subDet == SiStripDetId::TIB) {
    ss << SEP << "TIB" << SEP << "layer_" << layer << SEP;
  } else if (subDet == SiStripDetId::TID) {
    ss << SEP << "TID" << SEP << sside.str() << SEP << "wheel_" << layer << SEP;
  } else if (subDet == SiStripDetId::TOB) {
    ss << SEP << "TOB" << SEP << "layer_" << layer << SEP;
  } else if (subDet == SiStripDetId::TEC) {
    ss << SEP << "TEC" << SEP << sside.str() << SEP << "wheel_" << layer << SEP;
  } else {
    // ---------------------------  ???  --------------------------- //
    edm::LogWarning("SiStripTkDQM|WrongInput") << "no such SubDet :" << subDet << " no folder set!" << std::endl;
  }
}

void SiStripFolderOrganizer::getFolderName(int32_t rawdetid, const TrackerTopology* tTopo, std::string& lokal_folder) {
  lokal_folder = "";
  if (rawdetid == 0) {  // just top MechanicalFolder if rawdetid==0;
    return;
  }
  std::stringstream rest;
  SiStripDetId stripdet = SiStripDetId(rawdetid);

  if (stripdet.subDetector() == SiStripDetId::TIB) {
    // ---------------------------  TIB  --------------------------- //

    getSubDetLayerFolderName(rest, stripdet.subDetector(), tTopo->tibLayer(rawdetid));

    if (tTopo->tibIsZMinusSide(rawdetid))
      rest << "backward_strings" << SEP;
    else
      rest << "forward_strings" << SEP;
    if (tTopo->tibIsExternalString(rawdetid))
      rest << "external_strings" << SEP;
    else
      rest << "internal_strings" << SEP;
    rest << "string_" << tTopo->tibString(rawdetid) << SEP << "module_" << rawdetid;
  } else if (stripdet.subDetector() == SiStripDetId::TID) {
    // ---------------------------  TID  --------------------------- //

    getSubDetLayerFolderName(rest, stripdet.subDetector(), tTopo->tidWheel(rawdetid), tTopo->tidSide(rawdetid));
    rest << "ring_" << tTopo->tidRing(rawdetid) << SEP;

    if (tTopo->tidIsStereo(rawdetid))
      rest << "stereo_modules" << SEP;
    else
      rest << "mono_modules" << SEP;
    rest << "module_" << rawdetid;
  } else if (stripdet.subDetector() == SiStripDetId::TOB) {
    // ---------------------------  TOB  --------------------------- //

    getSubDetLayerFolderName(rest, stripdet.subDetector(), tTopo->tobLayer(rawdetid));
    if (tTopo->tobIsZMinusSide(rawdetid))
      rest << "backward_rods" << SEP;
    else
      rest << "forward_rods" << SEP;
    rest << "rod_" << tTopo->tobRod(rawdetid) << SEP << "module_" << rawdetid;
  } else if (stripdet.subDetector() == SiStripDetId::TEC) {
    // ---------------------------  TEC  --------------------------- //

    getSubDetLayerFolderName(rest, stripdet.subDetector(), tTopo->tecWheel(rawdetid), tTopo->tecSide(rawdetid));
    if (tTopo->tecIsBackPetal(rawdetid))
      rest << "backward_petals" << SEP;
    else
      rest << "forward_petals" << SEP;

    rest << "petal_" << tTopo->tecPetalNumber(rawdetid) << SEP << "ring_" << tTopo->tecRing(rawdetid) << SEP;

    if (tTopo->tecIsStereo(rawdetid))
      rest << "stereo_modules" << SEP;
    else
      rest << "mono_modules" << SEP;

    rest << "module_" << rawdetid;
  } else {
    // ---------------------------  ???  --------------------------- //
    edm::LogWarning("SiStripTkDQM|WrongInput")
        << "no such subdetector type :" << stripdet.subDetector() << " no folder set!" << std::endl;
    return;
  }
  lokal_folder += rest.str();
}

void SiStripFolderOrganizer::setLayerFolder(uint32_t rawdetid,
                                            const TrackerTopology* tTopo,
                                            int32_t layer,
                                            bool ring_flag) {
  std::string lokal_folder = TopFolderName + SEP + MECHANICAL_FOLDER_NAME;
  if (rawdetid == 0) {  // just top MechanicalFolder if rawdetid==0;
    dbe_->setCurrentFolder(lokal_folder);
    return;
  }

  std::ostringstream rest;
  SiStripDetId stripdet = SiStripDetId(rawdetid);
  if (stripdet.subDetector() == SiStripDetId::TIB) {
    // ---------------------------  TIB  --------------------------- //

    int tib_layer = tTopo->tibLayer(rawdetid);
    if (abs(layer) != tib_layer) {
      edm::LogWarning("SiStripTkDQM|Layer mismatch!!!")
          << " expect " << abs(layer) << " but getting " << tTopo->tibLayer(rawdetid) << std::endl;
      return;
    }
    rest << SEP << "TIB" << SEP << "layer_" << tTopo->tibLayer(rawdetid);
  } else if (stripdet.subDetector() == SiStripDetId::TID) {
    // ---------------------------  TID  --------------------------- //

    int tid_ring = tTopo->tidRing(rawdetid);

    // side
    uint32_t side = tTopo->tidSide(rawdetid);
    std::stringstream sside;
    if (side == 1) {
      sside << "MINUS";
    } else if (side == 2) {
      sside << "PLUS";
    }

    if (ring_flag) {
      if (abs(layer) != tid_ring) {
        edm::LogWarning("SiStripTkDQM|Layer mismatch!!!")
            << " expect " << abs(layer) << " but getting " << tTopo->tidRing(rawdetid) << std::endl;
        return;
      }
      rest << SEP << "TID" << SEP << sside.str() << SEP << "ring_" << tTopo->tidRing(rawdetid);
    } else {
      int tid_wheel = tTopo->tidWheel(rawdetid);
      if (abs(layer) != tid_wheel) {
        edm::LogWarning("SiStripTkDQM|Layer mismatch!!!")
            << " expect " << abs(layer) << " but getting " << tTopo->tidWheel(rawdetid) << std::endl;
        return;
      }
      rest << SEP << "TID" << SEP << sside.str() << SEP << "wheel_" << tTopo->tidWheel(rawdetid);
    }
  } else if (stripdet.subDetector() == SiStripDetId::TOB) {
    // ---------------------------  TOB  --------------------------- //

    int tob_layer = tTopo->tobLayer(rawdetid);
    if (abs(layer) != tob_layer) {
      edm::LogWarning("SiStripTkDQM|Layer mismatch!!!")
          << " expect " << abs(layer) << " but getting " << tTopo->tobLayer(rawdetid) << std::endl;
      return;
    }
    rest << SEP << "TOB" << SEP << "layer_" << tTopo->tobLayer(rawdetid);
  } else if (stripdet.subDetector() == SiStripDetId::TEC) {
    // ---------------------------  TEC  --------------------------- //

    // side
    uint32_t side = tTopo->tecSide(rawdetid);
    std::stringstream sside;
    if (side == 1) {
      sside << "MINUS";
    } else if (side == 2) {
      sside << "PLUS";
    }

    if (ring_flag) {
      int tec_ring = tTopo->tecRing(rawdetid);
      if (abs(layer) != tec_ring) {
        edm::LogWarning("SiStripTkDQM|Layer mismatch!!!")
            << " expect " << abs(layer) << " but getting " << tTopo->tecRing(rawdetid) << std::endl;
        return;
      }
      rest << SEP << "TEC" << SEP << sside.str() << SEP << "ring_" << tTopo->tecRing(rawdetid);
    } else {
      int tec_wheel = tTopo->tecWheel(rawdetid);
      if (abs(layer) != tec_wheel) {
        edm::LogWarning("SiStripTkDQM|Layer mismatch!!!")
            << " expect " << abs(layer) << " but getting " << tTopo->tecWheel(rawdetid) << std::endl;
        return;
      }
      rest << SEP << "TEC" << SEP << sside.str() << SEP << "wheel_" << tTopo->tecWheel(rawdetid);
    }
  } else {
    // ---------------------------  ???  --------------------------- //
    edm::LogWarning("SiStripTkDQM|WrongInput")
        << "no such subdetector type :" << stripdet.subDetector() << " no folder set!" << std::endl;
    return;
  }

  lokal_folder += rest.str();
  dbe_->setCurrentFolder(lokal_folder);
}

void SiStripFolderOrganizer::getSubDetFolder(const uint32_t& detid,
                                             const TrackerTopology* tTopo,
                                             std::string& folder_name) {
  auto subdet_and_tag = getSubDetFolderAndTag(detid, tTopo);
  folder_name = subdet_and_tag.first;
}
//
// -- Get the name of Subdetector Layer folder
//
void SiStripFolderOrganizer::getLayerFolderName(std::stringstream& ss,
                                                uint32_t rawdetid,
                                                const TrackerTopology* tTopo,
                                                bool ring_flag) {
  ss << TopFolderName + SEP + MECHANICAL_FOLDER_NAME;
  if (rawdetid == 0) {  // just top MechanicalFolder if rawdetid==0;
    return;
  }

  SiStripDetId stripdet = SiStripDetId(rawdetid);
  if (stripdet.subDetector() == SiStripDetId::TIB) {
    // ---------------------------  TIB  --------------------------- //

    ss << SEP << "TIB" << SEP << "layer_" << tTopo->tibLayer(rawdetid);
  } else if (stripdet.subDetector() == SiStripDetId::TID) {
    // ---------------------------  TID  --------------------------- //

    uint32_t side = tTopo->tidSide(rawdetid);
    std::stringstream sside;
    if (side == 1) {
      sside << "MINUS";
    } else if (side == 2) {
      sside << "PLUS";
    }

    if (ring_flag) {
      ss << SEP << "TID" << SEP << sside.str() << SEP << "ring_" << tTopo->tidRing(rawdetid);
    } else {
      ss << SEP << "TID" << SEP << sside.str() << SEP << "wheel_" << tTopo->tidWheel(rawdetid);
    }
  } else if (stripdet.subDetector() == SiStripDetId::TOB) {
    // ---------------------------  TOB  --------------------------- //

    ss << SEP << "TOB" << SEP << "layer_" << tTopo->tobLayer(rawdetid);
  } else if (stripdet.subDetector() == SiStripDetId::TEC) {
    // ---------------------------  TEC  --------------------------- //

    uint32_t side = tTopo->tecSide(rawdetid);
    std::stringstream sside;
    if (side == 1) {
      sside << "MINUS";
    } else if (side == 2) {
      sside << "PLUS";
    }

    if (ring_flag) {
      ss << SEP << "TEC" << SEP << sside.str() << SEP << "ring_" << tTopo->tecRing(rawdetid);
    } else {
      ss << SEP << "TEC" << SEP << sside.str() << SEP << "wheel_" << tTopo->tecWheel(rawdetid);
    }
  } else {
    // ---------------------------  ???  --------------------------- //
    edm::LogWarning("SiStripTkDQM|WrongInput")
        << "no such subdetector type :" << stripdet.subDetector() << " no folder set!" << std::endl;
    return;
  }
}

using namespace std::literals::string_view_literals;

std::pair<std::string_view, std::string_view> SiStripFolderOrganizer::getSubdetStrings(const uint32_t& detid,
                                                                                       const TrackerTopology* tTopo) {
  using std::string_view;
  switch (StripSubdetector::SubDetector(StripSubdetector(detid).subdetId())) {
    case StripSubdetector::TIB:
      return {"TIB", "TIB"};
    case StripSubdetector::TOB:
      return {"TOB", "TOB"};
    case StripSubdetector::TID:
      return (tTopo->tidSide(detid) == 2) ? std::make_pair("TID/PLUS"sv, "TID__PLUS"sv)
                                          : std::make_pair("TID/MINUS"sv, "TID__MINUS"sv);
    case StripSubdetector::TEC:
      return (tTopo->tecSide(detid) == 2) ? std::make_pair("TEC/PLUS"sv, "TEC__PLUS"sv)
                                          : std::make_pair("TEC/MINUS"sv, "TEC__MINUS"sv);
    default:
      edm::LogWarning("SiStripCommon") << "WARNING!!! this detid does not belong to tracker" << std::endl;
      return {"", ""};
  }
}

const std::string_view SiStripFolderOrganizer::getSubDetTag(const uint32_t& detid, const TrackerTopology* tTopo) {
  return getSubdetStrings(detid, tTopo).second;
}

std::pair<std::string, std::string_view> SiStripFolderOrganizer::getSubDetFolderAndTag(const uint32_t& detid,
                                                                                       const TrackerTopology* tTopo) {
  auto [folder_component, tag] = getSubdetStrings(detid, tTopo);
  std::string folder = TopFolderName + SEP + MECHANICAL_FOLDER_NAME + SEP + std::string(folder_component);
  return {std::move(folder), tag};
}
