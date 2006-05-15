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
// $Id: SiStripDetCabling.h,v 1.1 2006/03/22 18:14:00 dkcira Exp $
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
      void addDevices(const FedChannelConnection &);

      // getters
      inline const  std::map< uint32_t, std::vector<FedChannelConnection> >& getDetCabling() const { return detCabling_; }
      const std::vector<uint32_t> & getActiveDetectorsRawIds() const;
      void  addActiveDetectorsRawIds(std::vector<uint32_t> &) const; // add detectors to given reference - avoid using static
      const std::vector<FedChannelConnection>& getConnections( uint32_t det_id ) const;
      const FedChannelConnection& getConnection( uint32_t det_id, unsigned short apv_pair ) const;
      const unsigned int getDcuId( uint32_t det_id ) const;

   private:
      SiStripDetCabling(const SiStripDetCabling&); // stop default

      const SiStripDetCabling& operator=(const SiStripDetCabling&); // stop default

      // ---------- member data --------------------------------
      // map of KEY=detid DATA=vector<FedChannelConnection> 
      std::map< uint32_t, std::vector<FedChannelConnection> > detCabling_;

};

#endif
