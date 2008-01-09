#include "CalibFormats/SiStripObjects/interface/SiStripHashedDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>
#include <sstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripHashedDetId::SiStripHashedDetId( const std::vector<uint32_t>& raw_ids ) 
  : detIds_(),
    id_(0),
    iter_(detIds_.begin())
{
  LogTrace(mlCabling_)
    << "[SiStripHashedDetId::" << __func__ << "]"
    << " Constructing object...";
  init(raw_ids);
}

// -----------------------------------------------------------------------------
//
SiStripHashedDetId::SiStripHashedDetId( const std::vector<DetId>& det_ids ) 
  : detIds_(),
    id_(0),
    iter_(detIds_.begin())
{
  LogTrace(mlCabling_)
    << "[SiStripHashedDetId::" << __func__ << "]"
    << " Constructing object...";
  detIds_.clear();
  detIds_.reserve(16000);
  std::vector<DetId>::const_iterator iter = det_ids.begin();
  for ( ; iter != det_ids.end(); ++iter ) {
    detIds_.push_back( iter->rawId() );
  }
  init(detIds_);
}

// -----------------------------------------------------------------------------
//
SiStripHashedDetId::SiStripHashedDetId( const SiStripHashedDetId& input ) 
  : detIds_(),
    id_(0),
    iter_(detIds_.begin())
{
  LogTrace(mlCabling_)
    << "[SiStripHashedDetId::" << __func__ << "]"
    << " Constructing object...";
  detIds_.reserve( input.end() - input.begin() );
  std::copy( input.begin(), input.end(), detIds_.begin() );
}

// -----------------------------------------------------------------------------
//
SiStripHashedDetId::SiStripHashedDetId() 
  : detIds_(),
    id_(0),
    iter_(detIds_.begin())
{
  LogTrace(mlCabling_) 
    << "[SiStripHashedDetId::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
//
SiStripHashedDetId::~SiStripHashedDetId() {
  LogTrace(mlCabling_)
    << "[SiStripHashedDetId::" << __func__ << "]"
    << " Destructing object...";
  detIds_.clear();
}

// -----------------------------------------------------------------------------
//
void SiStripHashedDetId::init( const std::vector<uint32_t>& raw_ids ) {
  detIds_.clear();
  detIds_.reserve(16000);
  const_iterator iter = raw_ids.begin();
  for ( ; iter != raw_ids.end(); ++iter ) {
    SiStripDetId detid(*iter);
    if ( *iter != sistrip::invalid32_ && 
	 *iter != sistrip::invalid_ && 
	 detid.det() == DetId::Tracker && 
	 ( detid.subDetector() == SiStripDetId::TID || 
	   detid.subDetector() == SiStripDetId::TIB || 
	   detid.subDetector() == SiStripDetId::TOB || 
	   detid.subDetector() == SiStripDetId::TEC ) ) { 
      detIds_.push_back(*iter);
    } else {
      edm::LogWarning(mlCabling_)
	<< "[SiStripHashedDetId::" << __func__ << "]"
	<< " DetId 0x" 
	<< std::hex << std::setw(8) << std::setfill('0') << *iter
	<< " is not from the strip tracker!";
    }
  }
  if ( !detIds_.empty() ) {
    std::sort( detIds_.begin(), detIds_.end() );
    id_ = detIds_.front();
    iter_ = detIds_.begin();
  }
}

// -----------------------------------------------------------------------------
//
std::ostream& operator<< ( std::ostream& os, const SiStripHashedDetId& input ) {
  std::stringstream ss;
  ss << "[SiStripHashedDetId::" << __func__ << "]"
     << " Found " << input.end() - input.begin()
     << " entries in DetId hash map:"
     << std::endl;
  SiStripHashedDetId::const_iterator iter = input.begin();
  for ( ; iter != input.end(); ++iter ) {
    ss << " Index: "
       << std::dec << std::setw(5) << std::setfill(' ')
       << iter - input.begin() 
       << "  DetId: 0x"
       << std::hex << std::setw(8) << std::setfill('0')
       << *iter << std::endl;
  }
  os << ss.str();
  return os;
}

// -----------------------------------------------------------------------------
//
EVENTSETUP_DATA_REG(SiStripHashedDetId);
