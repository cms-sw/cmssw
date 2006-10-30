#ifndef DataFormats_SiStripCommon_HistoTitle_H
#define DataFormats_SiStripCommon_HistoTitle_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumeratedTypes.h"
#include "boost/cstdint.hpp"
#include <sstream>
#include <string>

class HistoTitle;

/** Debug info for FedChannelConnection class. */
std::ostream& operator<< ( std::ostream&, const HistoTitle& );

/** Simple class to hold components of histogram title. */
class HistoTitle {

 public: // public data members 

  sistrip::Task        task_;
  sistrip::KeyType     keyType_;
  uint32_t             keyValue_;
  sistrip::Granularity granularity_;
  uint16_t             channel_;
  std::string          extraInfo_;

  /** Constructor */
  HistoTitle( sistrip::Task task, 
	      sistrip::KeyType type,
	      uint32_t value,
	      sistrip::Granularity gran,
	      uint16_t channel,
	      std::string info = "" ) :
    task_(task), keyType_(type), 
    keyValue_(value), granularity_(gran), 
    channel_(channel), extraInfo_(info) {;}
    
  /** Default constructor. */
  HistoTitle() :
    task_(sistrip::UNDEFINED_TASK), keyType_(sistrip::UNDEFINED_KEY),
    keyValue_(0), granularity_(sistrip::UNDEFINED_GRAN),
    channel_(0), extraInfo_("") {;}
  
  /** Some debug. */
  void print( std::stringstream& ) const;

};

#endif // DataFormats_SiStripCommon_HistoTitle_H


