
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
    
  SiStripPartition( std::string partition_name );

  SiStripPartition( const SiStripPartition& );

  SiStripPartition& operator= ( const SiStripPartition& );
  
  bool operator== ( const SiStripPartition& ) const;
  
  bool operator!= ( const SiStripPartition& ) const;
  
  ~SiStripPartition();

  static const std::string defaultPartitionName_;

  typedef std::pair<uint32_t,uint32_t> Versions; 

  void reset(); 
  
  void pset( const edm::ParameterSet& );
  
  void update( const SiStripConfigDb* const );
  
  void print( std::stringstream&, bool using_db = false ) const; 
  
  // partition, run and version information
  
  std::string partitionName() const; 
  
  uint32_t runNumber() const;
  
  sistrip::RunType runType() const;
  
  bool forceVersions() const;

  bool forceCurrentState() const;

  // description versions
  
  Versions cabVersion() const;

  Versions fedVersion() const;

  Versions fecVersion() const;

  Versions dcuVersion() const;

  Versions psuVersion() const;

//#ifdef USING_DATABASE_MASKING // define anyway, otherwise I get into a mess with includes
  Versions maskVersion() const;
//#endif

  uint32_t globalAnalysisVersion() const;

  Versions runTableVersion() const;

  Versions fastCablingVersion() const;

  Versions apvTimingVersion() const;

  Versions optoScanVersion() const;

  Versions vpspScanVersion() const;

  Versions apvCalibVersion() const;

  Versions pedestalsVersion() const;

  Versions apvLatencyVersion() const;

  Versions fineDelayVersion() const;

  // input xml files

  std::string inputModuleXml() const;

  std::string inputDcuInfoXml() const;

  std::vector<std::string> inputFecXml() const;

  std::vector<std::string> inputFedXml() const;

  // setters
  
  void partitionName( std::string ); 
  
  void runNumber( uint32_t );
  
  void forceVersions( bool );

  void forceCurrentState( bool );

 private:
  
  Versions versions( const std::vector<uint32_t>& );

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

//#ifdef USING_DATABASE_MASKING // define anyway, otherwise I get into a mess with includes
  Versions maskVersion_;
//#endif

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

inline std::string SiStripPartition::partitionName() const { return partitionName_; } 
inline uint32_t SiStripPartition::runNumber() const { return runNumber_; }
inline sistrip::RunType SiStripPartition::runType() const { return runType_; }
inline bool SiStripPartition::forceVersions() const { return forceVersions_; }
inline bool SiStripPartition::forceCurrentState() const { return forceCurrentState_; }

inline SiStripPartition::Versions SiStripPartition::cabVersion() const { return cabVersion_; }
inline SiStripPartition::Versions SiStripPartition::fedVersion() const { return fedVersion_; }
inline SiStripPartition::Versions SiStripPartition::fecVersion() const { return fecVersion_; }
inline SiStripPartition::Versions SiStripPartition::dcuVersion() const { return dcuVersion_; }
inline SiStripPartition::Versions SiStripPartition::psuVersion() const { return psuVersion_; }
//#ifdef USING_DATABASE_MASKING // define anyway, otherwise I get into a mess with includes
inline SiStripPartition::Versions SiStripPartition::maskVersion() const { return maskVersion_; }
//#endif

inline uint32_t SiStripPartition::globalAnalysisVersion() const { return globalAnalysisV_; } 
inline SiStripPartition::Versions SiStripPartition::runTableVersion() const { return runTableVersion_; }
inline SiStripPartition::Versions SiStripPartition::fastCablingVersion() const { return fastCablingV_; }
inline SiStripPartition::Versions SiStripPartition::apvTimingVersion() const { return apvTimingV_; }
inline SiStripPartition::Versions SiStripPartition::optoScanVersion() const { return optoScanV_; }
inline SiStripPartition::Versions SiStripPartition::vpspScanVersion() const { return vpspScanV_; }
inline SiStripPartition::Versions SiStripPartition::apvCalibVersion() const { return apvCalibV_; }
inline SiStripPartition::Versions SiStripPartition::pedestalsVersion() const { return pedestalsV_; }
inline SiStripPartition::Versions SiStripPartition::apvLatencyVersion() const { return apvLatencyV_; }
inline SiStripPartition::Versions SiStripPartition::fineDelayVersion() const { return fineDelayV_; }

inline std::string SiStripPartition::inputModuleXml() const { return inputModuleXml_; }
inline std::string SiStripPartition::inputDcuInfoXml() const { return inputDcuInfoXml_; }
inline std::vector<std::string> SiStripPartition::inputFecXml() const { return inputFecXml_; }
inline std::vector<std::string> SiStripPartition::inputFedXml() const { return inputFedXml_; }

inline void SiStripPartition::partitionName( std::string name ) { partitionName_ = name ; } 
inline void SiStripPartition::runNumber( uint32_t run ) { runNumber_ = run; }
inline void SiStripPartition::forceVersions( bool force ) { forceVersions_ = force; }
inline void SiStripPartition::forceCurrentState( bool force ) { forceCurrentState_ = force; }

#endif // OnlineDB_SiStripConfigDb_SiStripPartition_h
