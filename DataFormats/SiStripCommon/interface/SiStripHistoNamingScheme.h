#ifndef DataFormats_SiStripCommon_SiStripHistoNamingScheme_H
#define DataFormats_SiStripCommon_SiStripHistoNamingScheme_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumeratedTypes.h"
#include "DataFormats/SiStripCommon/interface/HistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "boost/cstdint.hpp"
#include <string>

/** */
class SiStripHistoNamingScheme {
  
 public:


  // ---------- Conversion between enums and strings ----------

  
  static std::string view( const sistrip::View& );
  static sistrip::View view( const std::string& directory );

  static std::string task( const sistrip::Task& );
  static sistrip::Task task( const std::string& task );

  static std::string keyType( const sistrip::KeyType& );
  static sistrip::KeyType keyType( const std::string& key_type );

  static std::string granularity( const sistrip::Granularity& );
  static sistrip::Granularity granularity( const std::string& granularity );

  static std::string monitorable( const sistrip::Monitorable& );
  static sistrip::Monitorable monitorable( const std::string& histo_monitorable );
  
  static std::string presentation( const sistrip::Presentation& );
  static sistrip::Presentation presentation( const std::string& histo_presentation );
  
  static std::string cablingSource( const sistrip::CablingSource& );
  static sistrip::CablingSource cablingSource( const std::string& cabling_source );

  
  // ---------- Formulation of histogram titles ----------


  /** Contructs histogram name based on a general histogram name, a
      histogram key and a channel id. */
  static std::string histoTitle( const HistoTitle& title );

  /** Extracts various parameters from histogram name and returns the
      values in the form of a HistoTitle struct. */
  static HistoTitle histoTitle( const std::string& histo_title );
  

  // ---------- Formulation of directory paths ----------

  
  /** Returns control directory path in the form of a string. */ 
  static std::string controlPath( const SiStripFecKey::Path& );
  
  /** Returns control path based on directory name. */
  static SiStripFecKey::Path controlPath( const std::string& path );
  
  /** Returns readout directory path in the form of a string. */ 
  static std::string readoutPath( const SiStripFedKey::Path& );
  
  /** Returns readout path based on directory path. */
  static SiStripFedKey::Path readoutPath( const std::string& path );
  
};

#endif // DataFormats_SiStripCommon_SiStripHistoNamingScheme_H


