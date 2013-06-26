// Last commit: $Id: SiStripRing.h,v 1.3 2007/03/28 09:11:51 bainbrid Exp $

#ifndef CalibFormats_SiStripObjects_SiStripRing_H
#define CalibFormats_SiStripObjects_SiStripRing_H

#include "CalibFormats/SiStripObjects/interface/SiStripCcu.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include <boost/cstdint.hpp>
#include <vector>

/** 
    \class SiStripRing
    \author R.Bainbridge
*/
class SiStripRing {
  
 public: 
  
  /** */
  SiStripRing( const FedChannelConnection& conn );
  
  /** */
  ~SiStripRing() {;}

  /** */
  inline const std::vector<SiStripCcu>& ccus() const;

  /** */
  inline const uint16_t& fecRing() const;
  
  /** */
  void addDevices( const FedChannelConnection& conn );
  
 private:

  /** */
  SiStripRing() {;}

  /** */
  uint16_t fecRing_;

  /** */
  std::vector<SiStripCcu> ccus_;

};

// ---------- inline methods ----------

const std::vector<SiStripCcu>& SiStripRing::ccus() const { return ccus_; }
const uint16_t& SiStripRing::fecRing() const { return fecRing_; }

#endif // CalibTracker_SiStripObjects_SiStripRing_H


