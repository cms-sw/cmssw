// Last commit: $Id: SiStripPartition.h,v 1.2 2008/04/14 05:44:33 bainbrid Exp $

#ifndef OnlineDB_SiStripConfigDb_SiStripPartition_h
#define OnlineDB_SiStripConfigDb_SiStripPartition_h

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "boost/cstdint.hpp"
#include <vector>
#include <string>
#include <ostream>
#include <sstream>

class SiStripConfigDb;
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
  
  void pset( const edm::ParameterSet& );
  
  void update( const SiStripConfigDb* const );
  
  void print( std::stringstream&, bool using_db = false ) const; 
  
  // partition and run information

  inline std::string partitionName() const; 
  
  inline uint32_t runNumber() const;
  
  inline sistrip::RunType runType() const;

  // description versions
  
  inline Versions cabVersion() const;

  inline Versions fedVersion() const;

  inline Versions fecVersion() const;

  inline Versions calVersion() const;

  inline Versions dcuVersion() const;

  inline bool forceVersions() const;

  inline bool forceCurrentState() const;

  // input xml files

  inline std::string inputModuleXml() const;

  inline std::string inputDcuInfoXml() const;

  inline std::vector<std::string> inputFecXml() const;

  inline std::vector<std::string> inputFedXml() const;

 private:

  Versions versions( std::vector<unsigned int> );

 private:

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

  bool forceCurrentState_;

  // input xml files

  std::string inputModuleXml_;

  std::string inputDcuInfoXml_;

  std::vector<std::string> inputFecXml_;

  std::vector<std::string> inputFedXml_;

};

// ---------- Inline methods ----------

std::string SiStripPartition::partitionName() const { return partitionName_; } 
uint32_t SiStripPartition::runNumber() const { return runNumber_; }
sistrip::RunType SiStripPartition::runType() const { return runType_; }
SiStripPartition::Versions SiStripPartition::cabVersion() const { return ( forceCurrentState_ ? Versions(0,0) : cabVersion_ ); }
SiStripPartition::Versions SiStripPartition::fedVersion() const { return ( forceCurrentState_ ? Versions(0,0) : fedVersion_ ); }
SiStripPartition::Versions SiStripPartition::fecVersion() const { return ( forceCurrentState_ ? Versions(0,0) : fecVersion_ ); }
SiStripPartition::Versions SiStripPartition::calVersion() const { return ( forceCurrentState_ ? Versions(0,0) : calVersion_ ); }
SiStripPartition::Versions SiStripPartition::dcuVersion() const { return ( forceCurrentState_ ? Versions(0,0) : dcuVersion_ ); }
bool SiStripPartition::forceVersions() const { return forceVersions_; }
bool SiStripPartition::forceCurrentState() const { return forceCurrentState_; }
std::string SiStripPartition::inputModuleXml() const { return inputModuleXml_; }
std::string SiStripPartition::inputDcuInfoXml() const { return inputDcuInfoXml_; }
std::vector<std::string> SiStripPartition::inputFecXml() const { return inputFecXml_; }
std::vector<std::string> SiStripPartition::inputFedXml() const { return inputFedXml_; }

#endif // OnlineDB_SiStripConfigDb_SiStripPartition_h
