// Last commit: $Id: SiStripFec.h,v 1.3 2007/03/28 09:11:51 bainbrid Exp $

#ifndef CalibFormats_SiStripObjects_SiStripFec_H
#define CalibFormats_SiStripObjects_SiStripFec_H

#include "CalibFormats/SiStripObjects/interface/SiStripRing.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include <boost/cstdint.hpp>
#include <vector>

/** 
    \class SiStripFec
    \author R.Bainbridge
*/
class SiStripFec {

 public: 

  /** */
  SiStripFec( const FedChannelConnection& conn );
  
  /** */
  ~SiStripFec() {;}
  
  /** */
  inline const std::vector<SiStripRing>& rings() const;

  /** */
  inline const uint16_t& fecSlot() const;

  /** */
  void addDevices( const FedChannelConnection& conn );
  
 private:

  /** */
  SiStripFec() {;}

  /** */
  uint16_t fecSlot_;

  /** */
  std::vector<SiStripRing> rings_;

};

// ---------- inline methods ----------

const std::vector<SiStripRing>& SiStripFec::rings() const { return rings_; }
const uint16_t& SiStripFec::fecSlot() const { return fecSlot_; }

#endif // CalibTracker_SiStripObjects_SiStripFec_H


