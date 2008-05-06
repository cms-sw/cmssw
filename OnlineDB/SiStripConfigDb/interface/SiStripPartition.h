// Last commit: $Id: SiStripPartition.h,v 1.4 2008/04/29 11:57:04 bainbrid Exp $

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
    
  SiStripPartition( std::string partition_name );

  SiStripPartition();

  ~SiStripPartition();

  typedef std::pair<uint32_t,uint32_t> Versions; 
  
  void reset(); 
  
  void pset( const edm::ParameterSet& );
  
  void update( const SiStripConfigDb* const );
  
  void print( std::stringstream&, bool using_db = false ) const; 
  
  // partition, run and version information
  
  inline std::string partitionName() const; 
  
  inline uint32_t runNumber() const;
  
  inline sistrip::RunType runType() const;
  
  inline bool forceVersions() const;

  inline bool forceCurrentState() const;

  // description versions
  
  inline Versions cabVersion() const;

  inline Versions fedVersion() const;

  inline Versions fecVersion() const;

  inline Versions dcuVersion() const;

  inline Versions psuVersion() const;

  inline uint32_t globalAnalysisVersion() const;

  inline Versions fastCablingVersion() const;

  inline Versions apvTimingVersion() const;

  inline Versions optoScanVersion() const;

  inline Versions vpspScanVersion() const;

  inline Versions apvCalibVersion() const;

  inline Versions pedestalsVersion() const;

  inline Versions apvLatencyVersion() const;

  inline Versions fineDelayVersion() const;

  // input xml files

  inline std::string inputModuleXml() const;

  inline std::string inputDcuInfoXml() const;

  inline std::vector<std::string> inputFecXml() const;

  inline std::vector<std::string> inputFedXml() const;

  // setters
  
  inline void partitionName( std::string ); 
  
  inline void runNumber( uint32_t );
  
  inline void forceVersions( bool );

  inline void forceCurrentState( bool );

 private:
  
  Versions versions( std::vector<uint32_t> );

 private:

  std::string partitionName_; 
  
  uint32_t runNumber_;
  
  sistrip::RunType runType_;

  bool forceVersions_;

  bool forceCurrentState_;
  
  // device description versions
  
  Versions cabVersion_;

  Versions fedVersion_;

  Versions fecVersion_;

  Versions dcuVersion_;

  Versions psuVersion_;

  // analysis description versions

  uint32_t globalAnalysisV_;

  Versions runTableVersion_;

  Versions fastCablingV_;
  
  Versions apvTimingV_;

  Versions optoScanV_;

  Versions vpspScanV_;

  Versions apvCalibV_;

  Versions pedestalsV_;

  Versions apvLatencyV_;

  Versions fineDelayV_;

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
bool SiStripPartition::forceVersions() const { return forceVersions_; }
bool SiStripPartition::forceCurrentState() const { return forceCurrentState_; }

SiStripPartition::Versions SiStripPartition::cabVersion() const { return cabVersion_; }
SiStripPartition::Versions SiStripPartition::fedVersion() const { return fedVersion_; }
SiStripPartition::Versions SiStripPartition::fecVersion() const { return fecVersion_; }
SiStripPartition::Versions SiStripPartition::dcuVersion() const { return dcuVersion_; }
SiStripPartition::Versions SiStripPartition::psuVersion() const { return psuVersion_; }

uint32_t SiStripPartition::globalAnalysisVersion() const { return globalAnalysisV_; } 
SiStripPartition::Versions SiStripPartition::fastCablingVersion() const { return fastCablingV_; }
SiStripPartition::Versions SiStripPartition::apvTimingVersion() const { return apvTimingV_; }
SiStripPartition::Versions SiStripPartition::optoScanVersion() const { return optoScanV_; }
SiStripPartition::Versions SiStripPartition::vpspScanVersion() const { return vpspScanV_; }
SiStripPartition::Versions SiStripPartition::apvCalibVersion() const { return apvCalibV_; }
SiStripPartition::Versions SiStripPartition::pedestalsVersion() const { return pedestalsV_; }
SiStripPartition::Versions SiStripPartition::apvLatencyVersion() const { return apvLatencyV_; }
SiStripPartition::Versions SiStripPartition::fineDelayVersion() const { return fineDelayV_; }

std::string SiStripPartition::inputModuleXml() const { return inputModuleXml_; }
std::string SiStripPartition::inputDcuInfoXml() const { return inputDcuInfoXml_; }
std::vector<std::string> SiStripPartition::inputFecXml() const { return inputFecXml_; }
std::vector<std::string> SiStripPartition::inputFedXml() const { return inputFedXml_; }

void SiStripPartition::partitionName( std::string name ) { partitionName_ = name ; } 
void SiStripPartition::runNumber( uint32_t run ) { runNumber_ = run; }
void SiStripPartition::forceVersions( bool force ) { forceVersions_ = force; }
void SiStripPartition::forceCurrentState( bool force ) { forceCurrentState_ = force; }

#endif // OnlineDB_SiStripConfigDb_SiStripPartition_h
