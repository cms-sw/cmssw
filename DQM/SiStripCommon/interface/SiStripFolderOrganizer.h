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
// $Id$
//

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include <string>

class SiStripFolderOrganizer
{

   public:
      SiStripFolderOrganizer();
      virtual ~SiStripFolderOrganizer();

      // get top folder
      std::string getSiStripFolder();
      // get Readout view top folder or, if fed_id!= 99999, to corresponding Fed folder
      std::string getReadoutFolder(unsigned int fed_id = 99999);
      // get TOB folders (specifying first parameters means going to underlying folder)
      std::string getTOBFolder();
      std::string getTIBFolder();

      //new classes
      void setDetectorFolder(uint32_t rawdetid=0);

   private:
      SiStripFolderOrganizer(const SiStripFolderOrganizer&); // stop default
      const SiStripFolderOrganizer& operator=(const SiStripFolderOrganizer&); // stop default

   private:
      std::string TopFolderName;
      std::string MechanicalFolderName;
      std::string ReadoutFolderName;
      std::string ControlFolderName;
      std::string sep;
      DaqMonitorBEInterface* dbe_;

};

#endif
