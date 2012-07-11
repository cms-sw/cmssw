#ifndef SiStripCommon_SiStripFolderOrganizer_h
#define SiStripCommon_SiStripFolderOrganizer_h
// -*- C++ -*-
//
// Package:     SiStripCommon
// Class  :     SiStripFolderOrganizer
// 
/**\class SiStripFolderOrganizer SiStripFolderOrganizer.h DQM/SiStripCommon/interface/SiStripFolderOrganizer.h

 Description: <Organizes the folders for the monitoring elements of the SiStrip Tracker. Its methods return strings with names of folders to be created and used.>

 Usage:
    <usage>

*/
//
// Original Author:  dkcira
//         Created:  Thu Jan 26 23:49:46 CET 2006

// $Id: SiStripFolderOrganizer.h,v 1.14 2010/05/06 06:55:40 dutta Exp $

//
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

#include <string>

class DQMStore;

class SiStripFolderOrganizer
{

   public:
      static unsigned short const all_ = 65535;

      SiStripFolderOrganizer();
      virtual ~SiStripFolderOrganizer();

      // top folder
      void setSiStripFolderName(std::string name);
      std::string getSiStripFolder();
      void setSiStripFolder();

      // control folder
      std::string getSiStripTopControlFolder();
      void setSiStripTopControlFolder();
      std::string getSiStripControlFolder(
              // unsigned short crate,
              unsigned short slot = all_,
              unsigned short ring = all_,
              unsigned short addr = all_,
              unsigned short chan = all_
              // unsigned short i2c
      );
      void setSiStripControlFolder(
              // unsigned short crate,
              unsigned short slot = all_,
              unsigned short ring = all_,
              unsigned short addr = all_,
              unsigned short chan = all_
              // unsigned short i2c
      );

      std::pair<std::string,int32_t> GetSubDetAndLayer(const uint32_t& detid, bool ring_flag = 0);
      // detector folders
      void setDetectorFolder(uint32_t rawdetid=0);
      void getFolderName(int32_t rawdetid, std::string& lokal_folder);

      // layer folders
      void setLayerFolder(uint32_t rawdetid=0,int32_t layer=0,bool ring_flag = 0);
      void getLayerFolderName(std::stringstream& ss, uint32_t rawdetid,bool ring_flag = 0);
      void getSubDetLayerFolderName(std::stringstream& ss, SiStripDetId::SubDetector subDet, uint32_t layer, uint32_t side=0);
      // SubDetector Folder
      void getSubDetFolder(const uint32_t& detid, std::string& folder_name);
      std::pair<std::string, std::string> getSubDetFolderAndTag(const uint32_t& detid);
   private:
      SiStripFolderOrganizer(const SiStripFolderOrganizer&); // stop default
      const SiStripFolderOrganizer& operator=(const SiStripFolderOrganizer&); // stop default

   private:
      std::string TopFolderName;
      DQMStore* dbe_;
};
#endif
