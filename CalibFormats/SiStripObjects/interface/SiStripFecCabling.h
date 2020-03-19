
#ifndef CalibFormats_SiStripObjects_SiStripFecCabling_H
#define CalibFormats_SiStripObjects_SiStripFecCabling_H

#include "CalibFormats/SiStripObjects/interface/NumberOfDevices.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCrate.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include <ostream>
#include <sstream>
#include <vector>
#include <cstdint>

class SiStripFecCabling;

/** Debug info for SiStripFecCabling class. */
std::ostream &operator<<(std::ostream &, const SiStripFecCabling &);

/*
   @class SiStripFecCabling
   @author R.Bainbridge
   @brief FEC cabling object for the strip tracker.
*/
class SiStripFecCabling {
public:
  // ---------- Constructors, destructors ----------

  /** */
  SiStripFecCabling(const SiStripFedCabling &);
  /** */
  SiStripFecCabling() { ; }
  /** */
  ~SiStripFecCabling() { ; }  //@@ needs implementation!!

  // ---------- Methods to retrieve connection info ----------

  /** */
  inline const std::vector<SiStripFecCrate> &crates() const;
  inline std::vector<SiStripFecCrate> &crates();
  /** */
  inline const std::vector<SiStripFec> &fecs() const;  //@@ TEMPORARY: to maintain backward compatibility!
  /** */
  void connections(std::vector<FedChannelConnection> &) const;
  /** */
  const SiStripModule &module(const FedChannelConnection &conn) const;
  SiStripModule *module(const FedChannelConnection &conn);
  /** */
  const SiStripModule &module(const uint32_t &dcu_id) const;
  /** */
  NumberOfDevices countDevices() const;
  /** */
  void print(std::stringstream &) const;
  /** */
  void terse(std::stringstream &) const;

  // ---------- Methods used to build FEC cabling ----------

  /** */
  void buildFecCabling(const SiStripFedCabling &);
  /** */
  void addDevices(const FedChannelConnection &conn);
  /** */
  inline void dcuId(const FedChannelConnection &conn);
  /** */
  inline void detId(const FedChannelConnection &conn);
  /** */
  inline void nApvPairs(const FedChannelConnection &conn);

private:
  /** */
  std::vector<SiStripFecCrate> crates_;
};

// ---------- Inline methods ----------

const std::vector<SiStripFecCrate> &SiStripFecCabling::crates() const { return crates_; }
std::vector<SiStripFecCrate> &SiStripFecCabling::crates() { return crates_; }

// TEMPORARY method to maintain backward compatibility!
const std::vector<SiStripFec> &SiStripFecCabling::fecs() const {
  const static std::vector<SiStripFec> my_fecs;
  if (!crates_.empty()) {
    return crates_[0].fecs();
  } else {
    return my_fecs;
  }
}

void SiStripFecCabling::dcuId(const FedChannelConnection &conn) {
  auto m = module(conn);
  if (m) {
    m->dcuId(conn.dcuId());
  }
}

void SiStripFecCabling::detId(const FedChannelConnection &conn) {
  auto m = module(conn);
  if (m) {
    m->detId(conn.detId());
  }
}

void SiStripFecCabling::nApvPairs(const FedChannelConnection &conn) {
  auto m = module(conn);
  if (m) {
    m->nApvPairs(conn.nApvPairs());
  }
}

#endif  // CalibTracker_SiStripObjects_SiStripFecCabling_H

/*

inline void fedCh( const FedChannelConnection& conn ); //@@ needs to be
implemented

void SiStripFecCabling::fedCh( const FedChannelConnection& conn ) {
  module(conn).detId(conn.fedId());
  module(conn).detId(conn.fedCh());
}

*/
