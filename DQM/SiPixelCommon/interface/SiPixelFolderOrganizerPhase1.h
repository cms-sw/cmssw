#ifndef SiPixelCommon_SiPixelFolderOrganizerPhase1_h
#define SiPixelCommon_SiPixelFolderOrganizerPhase1_h
// -*- C++ -*-
//
// Package:     SiPixelCommon
// Class  :     SiPixelFolderOrganizerPhase1
// 
/**\class SiPixelFolderOrganizerPhase1 SiPixelFolderOrganizerPhase1.h DQM/SiPixelCommon/interface/SiPixelFolderOrganizerPhase1.h
   
Description: <Organizes the folders for the monitoring elements of the Pixel detector. Its methods return strings with names of folders to be created and used.>

Usage:
<usage>

*/
//
// Original Author:  chiochia
//         Created:  Thu Jan 26 23:49:46 CET 2006
#include "DQMServices/Core/interface/DQMStore.h"
#include <boost/cstdint.hpp>
#include <string>

class SiPixelFolderOrganizerPhase1 {
  
 public:

  /// Constructor - getStore should be called false from multi-thread DQM applications
  SiPixelFolderOrganizerPhase1(bool getStore = true);

  /// Destructor
  virtual ~SiPixelFolderOrganizerPhase1();
  
  /// Set folder name for a module or plaquette
  //type is: BPIX  mod=0, lad=1, lay=2, phi=3, 
  //         FPIX  mod=0, blade=4, disc=5, ring=6
  bool setModuleFolder(const uint32_t& rawdetid=0, int type=0);
  bool setModuleFolder(DQMStore::IBooker&, const uint32_t& rawdetid=0, int type=0);
  void getModuleFolder(const uint32_t& rawdetid, std::string& path);

  /// Set folder name for a FED (used in the case of errors without detId)
  bool setFedFolder(const uint32_t FedId);
  bool setFedFolder(DQMStore::IBooker&, const uint32_t FedId);

  
 private:

  std::string rootFolder;
  DQMStore* dbe_;
};
#endif
