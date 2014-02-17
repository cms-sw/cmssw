// Last commit: $Id: SiStripFecCabling.h,v 1.18 2008/02/06 16:40:42 bainbrid Exp $

#ifndef CalibFormats_SiStripObjects_SiStripFecCabling_H
#define CalibFormats_SiStripObjects_SiStripFecCabling_H

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CalibFormats/SiStripObjects/interface/NumberOfDevices.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCrate.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include <boost/cstdint.hpp>
#include <sstream>
#include <ostream>
#include <vector>

class SiStripFecCabling;

/** Debug info for SiStripFecCabling class. */
std::ostream& operator<< ( std::ostream&, const SiStripFecCabling& );

/* 
   @class SiStripFecCabling
   @author R.Bainbridge
   @brief FEC cabling object for the strip tracker.
*/
class SiStripFecCabling {
  
 public:

  // ---------- Constructors, destructors ----------
  
  /** */
  SiStripFecCabling( const SiStripFedCabling& );
  /** */
  SiStripFecCabling() {;}
  /** */
  ~SiStripFecCabling() {;} //@@ needs implementation!!
  
  // ---------- Methods to retrieve connection info ----------
  
  /** */
  inline const std::vector<SiStripFecCrate>& crates() const;
  /** */
  inline const std::vector<SiStripFec>& fecs() const; //@@ TEMPORARY: to maintain backward compatibility!
  /** */
  void connections( std::vector<FedChannelConnection>& ) const;
  /** */
  const SiStripModule& module( const FedChannelConnection& conn ) const;
  /** */
  const SiStripModule& module( const uint32_t& dcu_id ) const;
  /** */
  NumberOfDevices countDevices() const;
  /** */
  void print( std::stringstream& ) const;
  /** */
  void terse( std::stringstream& ) const;

  // ---------- Methods used to build FEC cabling ----------

  /** */
  void buildFecCabling( const SiStripFedCabling& ); 
  /** */
  void addDevices( const FedChannelConnection& conn );
  /** */
  inline void dcuId( const FedChannelConnection& conn );
  /** */
  inline void detId( const FedChannelConnection& conn );
  /** */
  inline void nApvPairs( const FedChannelConnection& conn );
  
 private:

  /** */
  std::vector<SiStripFecCrate> crates_;
  
};

// ---------- Inline methods ----------

const std::vector<SiStripFecCrate>& SiStripFecCabling::crates() const { return crates_; }

// TEMPORARY method to maintain backward compatibility!
const std::vector<SiStripFec>& SiStripFecCabling::fecs() const { 
  static std::vector<SiStripFec> my_fecs;
  if ( !crates_.empty() ) { return crates_[0].fecs(); }
  else { return my_fecs; }
}

void SiStripFecCabling::dcuId( const FedChannelConnection& conn ) { 
  const_cast<SiStripModule&>(module(conn)).dcuId(conn.dcuId()); 
}

void SiStripFecCabling::detId( const FedChannelConnection& conn ) { 
  const_cast<SiStripModule&>(module(conn)).detId(conn.detId()); 
}

void SiStripFecCabling::nApvPairs( const FedChannelConnection& conn ) { 
  const_cast<SiStripModule&>(module(conn)).nApvPairs(conn.nApvPairs()); 
}

#endif // CalibTracker_SiStripObjects_SiStripFecCabling_H

/*

inline void fedCh( const FedChannelConnection& conn ); //@@ needs to be implemented

void SiStripFecCabling::fedCh( const FedChannelConnection& conn ) {  
  module(conn).detId(conn.fedId());  
  module(conn).detId(conn.fedCh());  
}

*/
