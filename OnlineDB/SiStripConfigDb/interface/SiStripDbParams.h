// Last commit: $Id: SiStripDbParams.h,v 1.3 2008/04/11 13:27:33 bainbrid Exp $

#ifndef OnlineDB_SiStripConfigDb_SiStripDbParams_h
#define OnlineDB_SiStripConfigDb_SiStripDbParams_h

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripPartition.h"
#include "boost/cstdint.hpp"
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

  typedef std::map<std::string,SiStripPartition> SiStripPartitions;
  
  SiStripDbParams();

  ~SiStripDbParams();

  void reset(); 
  
  void setParams( const edm::ParameterSet& );
  
  void confdb( const std::string& );
  
  void confdb( const std::string& user,
	       const std::string& passwd,
	       const std::string& path );

  /** Extract (non-zero) partition names from partition objects. */
  std::vector<std::string> partitions() const;
  
  /** Extract (non-zero) partition names from string. */
  std::vector<std::string> partitions( std::string ) const;
  
  /** Construct string from (non-zero) partition names. */
  std::string partitions( const std::vector<std::string>& ) const;

  std::vector<std::string> inputModuleXmlFiles() const;

  std::vector<std::string> inputDcuInfoXmlFiles() const;

  std::vector<std::string> inputFecXmlFiles() const;

  std::vector<std::string> inputFedXmlFiles() const;

  void print( std::stringstream& ) const; 
  
  // ---------- PUBLIC member data ----------

 public:
  
  bool usingDb_;

  std::string confdb_;

  std::string user_;

  std::string passwd_;

  std::string path_;

  bool usingDbCache_;

  std::string sharedMemory_;

  std::string tnsAdmin_;

  SiStripPartitions partitions_; 

  // output xml files

  std::string outputModuleXml_;

  std::string outputDcuInfoXml_;

  std::string outputFecXml_;

  std::string outputFedXml_;

};

#endif // OnlineDB_SiStripConfigDb_SiStripDbParams_h
