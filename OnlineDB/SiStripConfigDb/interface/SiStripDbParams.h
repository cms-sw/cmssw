// Last commit: $Id: SiStripDbParams.h,v 1.8 2008/05/26 13:35:51 giordano Exp $

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

  ~SiStripDbParams();

  void reset(); 

  // ---------- typedefs ----------
  
  typedef std::map< std::string, SiStripPartition > SiStripPartitions;
  
  typedef boost::iterator_range<SiStripPartitions::const_iterator> const_iterator_range;

  typedef boost::iterator_range<SiStripPartitions::iterator> iterator_range;

  // ---------- database-related ----------

  inline bool usingDb() const;
  
  inline std::string confdb() const;

  inline std::string user() const;

  inline std::string passwd() const;

  inline std::string path() const;

  inline bool usingDbCache() const;

  inline std::string sharedMemory() const;

  inline std::string tnsAdmin() const;

  // ---------- partition-related ----------

  /** Returns pair of const iterators to partitions objects. */
  inline const_iterator_range partitions() const;

  /** Returns pair of iterators to partitions objects. */
  inline iterator_range partitions();

  /** Returns const iterator to partition object. */
  SiStripPartitions::const_iterator partition( std::string partition_name ) const;
  
  /** Returns iterator to partition object. */
  SiStripPartitions::iterator partition( std::string partition_name );
  
  /** */
  inline void clearPartitions();

  /** */
  void addPartition( const SiStripPartition& );

  /** Extract (non-zero) partition names from partition objects. */
  std::vector<std::string> partitionNames() const;
  
  /** Extract (non-zero) partition names from string. */
  std::vector<std::string> partitionNames( std::string ) const;
  
  /** Construct string from (non-zero) partition names. */
  std::string partitionNames( const std::vector<std::string>& ) const;

  // ---------- setters ----------

  inline void usingDb( bool );
  
  inline void usingDbCache( bool );

  inline void sharedMemory( std::string );

  void pset( const edm::ParameterSet& );
  
  void confdb( const std::string& );
  
  void confdb( const std::string& user,
	       const std::string& passwd,
	       const std::string& path );
  
  // ---------- xml file names ----------

  std::vector<std::string> inputModuleXmlFiles() const;

  std::vector<std::string> inputDcuInfoXmlFiles() const;

  std::vector<std::string> inputFecXmlFiles() const;

  std::vector<std::string> inputFedXmlFiles() const;

  inline std::string outputModuleXml() const;

  inline std::string outputDcuInfoXml() const;

  inline std::string outputFecXml() const;

  inline std::string outputFedXml() const;

  void print( std::stringstream& ) const; 

  bool operator == ( const SiStripDbParams& other );

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

bool SiStripDbParams::usingDb() const { return usingDb_; }
std::string SiStripDbParams::confdb() const { return confdb_; }
std::string SiStripDbParams::user() const { return  user_; }
std::string SiStripDbParams::passwd() const { return passwd_; }
std::string SiStripDbParams::path() const { return path_; }
bool SiStripDbParams::usingDbCache() const { return usingDbCache_; }
std::string SiStripDbParams::sharedMemory() const { return sharedMemory_; }
std::string SiStripDbParams::tnsAdmin() const { return tnsAdmin_; }

SiStripDbParams::const_iterator_range SiStripDbParams::partitions() const { return const_iterator_range( partitions_.begin(), 
													 partitions_.end() ); }
SiStripDbParams::iterator_range SiStripDbParams::partitions() { return iterator_range( partitions_.begin(), 
										       partitions_.end() ); }

std::string SiStripDbParams::outputModuleXml() const { return outputModuleXml_; }
std::string SiStripDbParams::outputDcuInfoXml() const { return outputDcuInfoXml_; }
std::string SiStripDbParams::outputFecXml() const { return outputFecXml_; }
std::string SiStripDbParams::outputFedXml() const { return outputFedXml_; }

void SiStripDbParams::clearPartitions() { partitions_.clear(); }
void SiStripDbParams::usingDb( bool using_db ) { usingDb_ = using_db; }
void SiStripDbParams::usingDbCache( bool using_cache ) { usingDbCache_ = using_cache; }
void SiStripDbParams::sharedMemory( std::string name ) { sharedMemory_ = name; }

#endif // OnlineDB_SiStripConfigDb_SiStripDbParams_h
