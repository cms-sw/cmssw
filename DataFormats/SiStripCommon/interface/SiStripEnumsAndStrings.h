// Last commit: $Id: SiStripEnumsAndStrings.h,v 1.4 2007/11/29 17:08:03 bainbrid Exp $

#ifndef DataFormats_SiStripCommon_SiStripEnumsAndStrings_H
#define DataFormats_SiStripCommon_SiStripEnumsAndStrings_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <string>

/** */
class SiStripEnumsAndStrings {
  
 public:

  static std::string view( const sistrip::View& );
  static sistrip::View view( const std::string& directory );
  
  static std::string runType( const sistrip::RunType& );
  static sistrip::RunType runType( const std::string& run_type );

  static sistrip::RunType runType( const uint16_t& );
  
  static std::string keyType( const sistrip::KeyType& );
  static sistrip::KeyType keyType( const std::string& key_type );
  
  static std::string granularity( const sistrip::Granularity& );
  static sistrip::Granularity granularity( const std::string& granularity );

  static std::string apvReadoutMode( const sistrip::ApvReadoutMode& );
  static sistrip::ApvReadoutMode apvReadoutMode( const std::string& apv_readout_mode );

  static std::string fedReadoutMode( const sistrip::FedReadoutMode& );
  static sistrip::FedReadoutMode fedReadoutMode( const std::string& fed_readout_mode );
  
  static std::string histoType( const sistrip::HistoType& );
  static sistrip::HistoType histoType( const std::string& histo_type );
  
  static std::string monitorable( const sistrip::Monitorable& );
  static sistrip::Monitorable monitorable( const std::string& histo_monitorable );
  
  static std::string presentation( const sistrip::Presentation& );
  static sistrip::Presentation presentation( const std::string& histo_presentation );
  
  static std::string cablingSource( const sistrip::CablingSource& );
  static sistrip::CablingSource cablingSource( const std::string& cabling_source );
  
};

#endif // DataFormats_SiStripCommon_SiStripEnumsAndStrings_H


