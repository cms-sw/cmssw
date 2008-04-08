// Last commit: $Id: $

#ifndef OnlineDB_SiStripConfigDb_SiStripDbParams_h
#define OnlineDB_SiStripConfigDb_SiStripDbParams_h

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "boost/cstdint.hpp"
#include <vector>
#include <string>
#include <ostream>
#include <sstream>

/** Container class for database connection parameters. */
class SiStripDbParams { 
  
 public:
    
  // Constructor and methods
  SiStripDbParams();

  ~SiStripDbParams();

  void print( std::stringstream& ) const; 

  void reset(); 

  void setParams( const edm::ParameterSet& );

  void confdb( const std::string& );

  void confdb( const std::string& user,
	       const std::string& passwd,
	       const std::string& path );

  std::string partitions() const;

  std::vector<std::string> partitions( std::string ) const;

  // Public member data 

 public:
  
  bool usingDb_;

  std::string confdb_;

  std::string user_;

  std::string passwd_;

  std::string path_;

  std::vector<std::string> partitions_; 

  bool usingDbCache_;

  std::string sharedMemory_;

  uint32_t runNumber_;

  uint32_t cabMajor_;

  uint32_t cabMinor_;

  uint32_t fedMajor_;

  uint32_t fedMinor_;

  uint32_t fecMajor_;

  uint32_t fecMinor_;

  uint32_t calMajor_;

  uint32_t calMinor_;

  uint32_t dcuMajor_;

  uint32_t dcuMinor_;

  sistrip::RunType runType_;

  bool force_;

  std::string inputModuleXml_;

  std::string inputDcuInfoXml_;

  std::vector<std::string> inputFecXml_;

  std::vector<std::string> inputFedXml_;

  std::string inputDcuConvXml_;

  std::string outputModuleXml_;

  std::string outputDcuInfoXml_;

  std::string outputFecXml_;

  std::string outputFedXml_;

  std::string tnsAdmin_;

};

/** Debug printout for SiStripDbParams class. */
std::ostream& operator<< ( std::ostream&, const SiStripDbParams& );

#endif // OnlineDB_SiStripConfigDb_SiStripDbParams_h
