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
// Original Author:  dkcira
//         Created:  Wed Mar 22 12:24:20 CET 2006
// $Id: SiStripDetCabling.h,v 1.3 2007/01/24 14:45:00 dkcira Exp $
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
    inline const  std::map< uint32_t, std::vector<FedChannelConnection> >& getDetCabling() const { return fullcabling_; }

    // for DQM use: all detectors that have at least one connected APV
    void  addActiveDetectorsRawIds(std::vector<uint32_t> &) const;     // same method as addConnectedDetectorsRawIds

    // for RECO use
    void  addConnected ( std::map<uint32_t, std::vector<int> > &) const; // map of detector to list of APVs for APVs seen from FECs and FEDs
    void  addDetected  ( std::map<uint32_t, std::vector<int> > &) const; // map of detector to list of APVs for APVs seen from FECs but not from FEDs
    void  addUnDetected( std::map<uint32_t, std::vector<int> > &) const; // map of detector to list of APVs for APVs seen neither from FECS or FEDs
    void  addNotConnectedAPVs( std::map<uint32_t, std::vector<int> > &) const; // map of detector to list of APVs that are not connected - combination of addDetected and addUnDetected

    // other
    const std::vector<FedChannelConnection>& getConnections( uint32_t det_id ) const;
    const FedChannelConnection& getConnection( uint32_t det_id, unsigned short apv_pair ) const;
    const unsigned int getDcuId( uint32_t det_id ) const;
    const uint16_t nApvPairs(uint32_t det_id) const; // maximal nr. of apvpairs a detector can have (2 or 3)

  private:
    SiStripDetCabling(const SiStripDetCabling&); // stop default
    const SiStripDetCabling& operator=(const SiStripDetCabling&); // stop default
    void addFromSpecificConnection( std::map<uint32_t, std::vector<int> > & , const std::map< uint32_t, std::vector<int> >  &) const;

    // ---------- member data --------------------------------
    // map of KEY=detid DATA=vector<FedChannelConnection> 
    std::map< uint32_t, std::vector<FedChannelConnection> > fullcabling_;
    // map of KEY=detid DATA=vector of apvs, maximum 6 APVs per detector module : 0,1,2,3,4,5
    std::map< uint32_t, std::vector<int> > connected_; // seen from FECs and FEDs
    std::map< uint32_t, std::vector<int> > detected_; // seen from FECs but not from FEDs
    std::map< uint32_t, std::vector<int> > undetected_; // seen from neither FECs or FEDs, DetIds inferred from static Look-Up-Table in the configuration database
};

#endif
