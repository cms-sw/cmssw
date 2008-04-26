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

// $Id: SiStripFolderOrganizer.h,v 1.7 2008/03/03 11:54:03 maborgia Exp $

//

#include <string>

class DQMStore;

class SiStripFolderOrganizer
{

   public:
      static unsigned short const all_ = 65535;

      SiStripFolderOrganizer();
      virtual ~SiStripFolderOrganizer();

      // top folder
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

      std::pair<std::string,int32_t> GetSubDetAndLayer(const uint32_t& detid);
      // detector folders
      void setDetectorFolder(uint32_t rawdetid=0);
      void getFolderName(int32_t rawdetid, std::string& lokal_folder);
      // layer folders
      void setLayerFolder(uint32_t rawdetid=0,int32_t layer=0);
   private:
      SiStripFolderOrganizer(const SiStripFolderOrganizer&); // stop default
      const SiStripFolderOrganizer& operator=(const SiStripFolderOrganizer&); // stop default

   private:
      std::string TopFolderName;
      std::string MechanicalFolderName;
      std::string ReadoutFolderName;
      std::string ControlFolderName;
      std::string sep;
      DQMStore* dbe_;
};
#endif
