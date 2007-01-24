#ifndef SiStripObjects_SiStripDetCabling_h
#define SiStripObjects_SiStripDetCabling_h
// -*- C++ -*-
//
// Package:     SiStripObjects
// Class  :     SiStripDetCabling
// 
/**\class SiStripDetCabling SiStripDetCabling.h CalibFormats/SiStripObjects/interface/SiStripDetCabling.h

 Description: give detector view for the cabling classes

 Usage:
    <usage>

*/
//
// Original Author:  dkcira / bainbrid
//         Created:  Wed Mar 22 12:24:20 CET 2006
// $Id: SiStripDetCabling.h,v 1.2 2006/05/15 08:50:44 dkcira Exp $
//

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include <boost/cstdint.hpp>
#include <vector>
#include <map>


class SiStripDetCabling
{

   public:
      SiStripDetCabling();
      virtual ~SiStripDetCabling();
      SiStripDetCabling(const SiStripFedCabling &);

      void addDevices(const FedChannelConnection &, std::map< uint32_t, std::vector<FedChannelConnection> >&);
      void addDevices(const FedChannelConnection &); // special case of above addDevices

      // getters
      inline const  std::map< uint32_t, std::vector<FedChannelConnection> >& getDetCabling() const { return connected_; } // return cabling for connected detids

      void  addConnectedDetectorsRawIds(std::vector<uint32_t> &) const;  // add detectors seen from FECs and FEDs
      void  addActiveDetectorsRawIds(std::vector<uint32_t> &) const;     // same method as addConnectedDetectorsRawIds
      void  addDetectedDetectorsRawIds(std::vector<uint32_t> &) const;   // add detectors seen from FECs but not from FEDs
      void  addUnDetectedDetectorsRawIds(std::vector<uint32_t> &) const; // add detectors seen neither from FECS or FEDs

      const std::vector<FedChannelConnection>& getConnections( uint32_t det_id ) const;
      const FedChannelConnection& getConnection( uint32_t det_id, unsigned short apv_pair ) const;
      const unsigned int getDcuId( uint32_t det_id ) const;

   private:
      SiStripDetCabling(const SiStripDetCabling&); // stop default

      const SiStripDetCabling& operator=(const SiStripDetCabling&); // stop default

      // ---------- member data --------------------------------
      // map of KEY=detid DATA=vector<FedChannelConnection> 
      std::map< uint32_t, std::vector<FedChannelConnection> > connected_; // seen from FECs and FEDs
      std::map< uint32_t, std::vector<FedChannelConnection> > detected_; // seen from FECs but not from FEDs
      std::map< uint32_t, std::vector<FedChannelConnection> > undetected_; // seen from neither FECs or FEDs, DetIds inferred from static Look-Up-Table in the configuration database
};

#endif
