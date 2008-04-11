// Last commit: $Id: $

#ifndef OnlineDB_SiStripConfigDb_SiStripPartition_h
#define OnlineDB_SiStripConfigDb_SiStripPartition_h

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "boost/cstdint.hpp"
#include <vector>
#include <string>
#include <ostream>
#include <sstream>

class SiStripPartition;

/** Debug printout for SiStripPartition class. */
std::ostream& operator<< ( std::ostream&, const SiStripPartition& );

/** 
    @class SiStripPartition
    @brief Container class for database partition parameters. 
    @author R.Bainbridge
*/
class SiStripPartition { 
  
 public:
    
  SiStripPartition();

  ~SiStripPartition();
  
  void reset(); 

  void setParams( const edm::ParameterSet& );
  
  void print( std::stringstream&, bool using_db = false ) const; 
  
  // ---------- PUBLIC member data ----------

 public:
  
  // partition and run information
  
  std::string partitionName_; 
  
  uint32_t runNumber_;
  
  sistrip::RunType runType_;
  
  // description versions

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

  bool forceVersions_;

  // input xml files

  std::string inputModuleXml_;

  std::string inputDcuInfoXml_;

  std::vector<std::string> inputFecXml_;

  std::vector<std::string> inputFedXml_;

};

#endif // OnlineDB_SiStripConfigDb_SiStripPartition_h
