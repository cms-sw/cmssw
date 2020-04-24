
#ifndef CalibFormats_SiStripObjects_SiStripCcu_H
#define CalibFormats_SiStripObjects_SiStripCcu_H

#include "CalibFormats/SiStripObjects/interface/SiStripModule.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include <boost/cstdint.hpp>
#include <vector>

/** 
    \class SiStripCcu
    \author R.Bainbridge
*/
class SiStripCcu {
  
 public: 
  
  /** */
  SiStripCcu( const FedChannelConnection& conn );
  
  /** */
  ~SiStripCcu() {;}
 
   /** */
  inline const std::vector<SiStripModule>& modules() const;
  inline std::vector<SiStripModule>& modules();

  /** */
  inline const uint16_t& ccuAddr() const;

  /** */
  void addDevices( const FedChannelConnection& conn );
  
 private:

  /** */
  SiStripCcu() {;}

  /** */
  uint16_t ccuAddr_;

  /** */
  std::vector<SiStripModule> modules_;

};

// ---------- inline methods ----------

const std::vector<SiStripModule>& SiStripCcu::modules() const { return modules_; }
std::vector<SiStripModule>& SiStripCcu::modules() { return modules_; }
const uint16_t& SiStripCcu::ccuAddr() const { return ccuAddr_; }

#endif // CalibTracker_SiStripObjects_SiStripCcu_H


