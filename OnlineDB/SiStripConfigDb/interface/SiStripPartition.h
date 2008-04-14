// Last commit: $Id: SiStripPartition.h,v 1.1 2008/04/11 13:27:33 bainbrid Exp $

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

  typedef std::pair<uint32_t,uint32_t> Versions; 
  
  void reset(); 

  void setParams( const edm::ParameterSet& );
  
  Versions versions( std::vector<unsigned int> );
  
  void print( std::stringstream&, bool using_db = false ) const; 
  
  // ---------- PUBLIC member data ----------

 public:
  
  // partition and run information
  
  std::string partitionName_; 
  
  uint32_t runNumber_;
  
  sistrip::RunType runType_;
  
  // description versions
  
  Versions cabVersion_;

  Versions fedVersion_;

  Versions fecVersion_;

  Versions calVersion_;

  Versions dcuVersion_;

  bool forceVersions_;

  // input xml files

  std::string inputModuleXml_;

  std::string inputDcuInfoXml_;

  std::vector<std::string> inputFecXml_;

  std::vector<std::string> inputFedXml_;

};

#endif // OnlineDB_SiStripConfigDb_SiStripPartition_h
