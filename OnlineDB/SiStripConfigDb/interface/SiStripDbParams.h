// Last commit: $Id: SiStripDbParams.h,v 1.13 2009/02/20 10:01:15 alinn Exp $

#ifndef OnlineDB_SiStripConfigDb_SiStripDbParams_h
#define OnlineDB_SiStripConfigDb_SiStripDbParams_h

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripPartition.h"
#include "boost/cstdint.hpp"
#include "boost/range/iterator_range.hpp"
#include <ostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

class SiStripDbParams;

/** Debug printout for SiStripDbParams class. */
std::ostream& operator<< ( std::ostream&, const SiStripDbParams& );

/** 
    @class SiStripDbParams
    @brief Container class for database connection parameters. 
    @author R.Bainbridge
*/
class SiStripDbParams { 
  
 public:

  // ---------- cons(de)structors ----------
  
  SiStripDbParams();
  
  SiStripDbParams( const SiStripDbParams& );
  
  SiStripDbParams& operator= ( const SiStripDbParams& );
  
  bool operator== ( const SiStripDbParams& ) const;
  
  bool operator!= ( const SiStripDbParams& ) const;
  
  ~SiStripDbParams();

  // ---------- typedefs ----------
  
  typedef std::map< std::string, SiStripPartition > SiStripPartitions;

  typedef SiStripPartitions::size_type size_type;
  
  typedef boost::iterator_range<SiStripPartitions::const_iterator> const_iterator_range;

  typedef boost::iterator_range<SiStripPartitions::iterator> iterator_range;

  // ---------- database-related ----------

  bool usingDb() const;
  
  std::string confdb() const;

  std::string user() const;

  std::string passwd() const;

  std::string path() const;

  bool usingDbCache() const;

  std::string sharedMemory() const;

  std::string tnsAdmin() const;

  // ---------- partition-related ----------

  /** Returns pair of const iterators to partitions objects. */
  const_iterator_range partitions() const;

  /** Returns pair of iterators to partitions objects. */
  iterator_range partitions();

  /** Returns const iterator to partition object. */
  SiStripPartitions::const_iterator partition( std::string partition_name ) const;
  
  /** Returns iterator to partition object. */
  SiStripPartitions::iterator partition( std::string partition_name );
  
  /** */
  void clearPartitions();

  /** */
  void addPartition( const SiStripPartition& );

  /** Extract (non-zero) partition names from partition objects. */
  std::vector<std::string> partitionNames() const;
  
  /** Extract (non-zero) partition names from string. */
  std::vector<std::string> partitionNames( std::string ) const;
  
  /** Construct string from (non-zero) partition names. */
  std::string partitionNames( const std::vector<std::string>& ) const;

  /** Return the number of partitions. */
  size_type partitionsSize() const;
  
  // ---------- setters ----------

  void usingDb( bool );
  
  void usingDbCache( bool );

  void sharedMemory( std::string );

  void pset( const edm::ParameterSet& );
  
  void confdb( const std::string& );
  
  void confdb( const std::string& user,
	       const std::string& passwd,
	       const std::string& path );
  
  void reset(); 
  
  // ---------- xml file names ----------

  std::vector<std::string> inputModuleXmlFiles() const;

  std::vector<std::string> inputDcuInfoXmlFiles() const;

  std::vector<std::string> inputFecXmlFiles() const;

  std::vector<std::string> inputFedXmlFiles() const;

  std::string outputModuleXml() const;

  std::string outputDcuInfoXml() const;

  std::string outputFecXml() const;

  std::string outputFedXml() const;

  void print( std::stringstream& ) const; 

  // ---------- private member data ---------- 
  
 private:
  
  bool usingDb_;

  std::string confdb_;

  std::string user_;

  std::string passwd_;

  std::string path_;

  bool usingDbCache_;

  std::string sharedMemory_;

  std::string tnsAdmin_;

  SiStripPartitions partitions_; 

  std::string outputModuleXml_;

  std::string outputDcuInfoXml_;

  std::string outputFecXml_;

  std::string outputFedXml_;

};

// ---------- inline methods ----------

inline bool SiStripDbParams::usingDb() const { return usingDb_; }
inline std::string SiStripDbParams::confdb() const { return confdb_; }
inline std::string SiStripDbParams::user() const { return  user_; }
inline std::string SiStripDbParams::passwd() const { return passwd_; }
inline std::string SiStripDbParams::path() const { return path_; }
inline bool SiStripDbParams::usingDbCache() const { return usingDbCache_; }
inline std::string SiStripDbParams::sharedMemory() const { return sharedMemory_; }
inline std::string SiStripDbParams::tnsAdmin() const { return tnsAdmin_; }

inline SiStripDbParams::const_iterator_range SiStripDbParams::partitions() const { return const_iterator_range( partitions_.begin(), 
														partitions_.end() ); }
inline SiStripDbParams::iterator_range SiStripDbParams::partitions() { return iterator_range( partitions_.begin(), 
											      partitions_.end() ); }

inline SiStripDbParams::size_type SiStripDbParams::partitionsSize() const { return partitions_.size(); }

inline std::string SiStripDbParams::outputModuleXml() const { return outputModuleXml_; }
inline std::string SiStripDbParams::outputDcuInfoXml() const { return outputDcuInfoXml_; }
inline std::string SiStripDbParams::outputFecXml() const { return outputFecXml_; }
inline std::string SiStripDbParams::outputFedXml() const { return outputFedXml_; }

inline void SiStripDbParams::clearPartitions() { partitions_.clear(); }
inline void SiStripDbParams::usingDb( bool using_db ) { usingDb_ = using_db; }
inline void SiStripDbParams::usingDbCache( bool using_cache ) { usingDbCache_ = using_cache; }
inline void SiStripDbParams::sharedMemory( std::string name ) { sharedMemory_ = name; }

#endif // OnlineDB_SiStripConfigDb_SiStripDbParams_h
