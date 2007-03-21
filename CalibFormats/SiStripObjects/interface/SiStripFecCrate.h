// Last commit: $Id: $

#ifndef CalibFormats_SiStripObjects_SiStripFecCrate_H
#define CalibFormats_SiStripObjects_SiStripFecCrate_H

#include "CalibFormats/SiStripObjects/interface/SiStripFec.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include <boost/cstdint.hpp>
#include <vector>

/** 
    \class SiStripFecCrate
    \author R.Bainbridge
*/
class SiStripFecCrate {

 public: 

  /** */
  SiStripFecCrate( const FedChannelConnection& conn )
    : fecCrate_( conn.fecCrate() ), fecs_() { addDevices( conn ); }
  
  /** */
  ~SiStripFecCrate() {;}
  
  /** */
  inline const std::vector<SiStripFec>& fecs() const;
  
  /** */
  inline const uint16_t& fecCrate() const;

  /** */
  void addDevices( const FedChannelConnection& conn );
  
 private:

  /** */
  SiStripFecCrate() {;}

  /** */
  uint16_t fecCrate_;

  /** */
  std::vector<SiStripFec> fecs_;

};

// ---------- inline methods ----------

const std::vector<SiStripFec>& SiStripFecCrate::fecs() const { return fecs_; }
const uint16_t& SiStripFecCrate::fecCrate() const { return fecCrate_; }

#endif // CalibTracker_SiStripObjects_SiStripFecCrate_H


