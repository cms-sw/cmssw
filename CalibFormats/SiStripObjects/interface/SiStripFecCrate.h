
#ifndef CalibFormats_SiStripObjects_SiStripFecCrate_H
#define CalibFormats_SiStripObjects_SiStripFecCrate_H

#include "CalibFormats/SiStripObjects/interface/SiStripFec.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include <vector>
#include <cstdint>

/**
    \class SiStripFecCrate
    \author R.Bainbridge
*/
class SiStripFecCrate {
public:
  /** */
  SiStripFecCrate(const FedChannelConnection &conn);

  /** */
  ~SiStripFecCrate() { ; }

  /** */
  inline const std::vector<SiStripFec> &fecs() const;
  inline std::vector<SiStripFec> &fecs();

  /** */
  inline const uint16_t &fecCrate() const;

  /** */
  void addDevices(const FedChannelConnection &conn);

private:
  /** */
  SiStripFecCrate() { ; }

  /** */
  uint16_t fecCrate_;

  /** */
  std::vector<SiStripFec> fecs_;
};

// ---------- inline methods ----------

const std::vector<SiStripFec> &SiStripFecCrate::fecs() const { return fecs_; }
std::vector<SiStripFec> &SiStripFecCrate::fecs() { return fecs_; }
const uint16_t &SiStripFecCrate::fecCrate() const { return fecCrate_; }

#endif  // CalibTracker_SiStripObjects_SiStripFecCrate_H
