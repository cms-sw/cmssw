#include "CalibTracker/SiStripCommon/interface/SiStripFedIdListReader.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>
#include <sstream>

// -----------------------------------------------------------------------------
//
SiStripFedIdListReader::SiStripFedIdListReader( std::string filePath ) {
  fedIds_.clear();
  inputFile_.open( filePath.c_str() );

  if ( inputFile_.is_open() ) {
    
    for(;;) {
      
      uint32_t fed_id; 
      inputFile_ >> fed_id;

      if ( !( inputFile_.eof() || inputFile_.fail() ) ) {
	
	std::vector<uint16_t>::const_iterator it = find( fedIds_.begin(), fedIds_.end(), fed_id );
	if( it == fedIds_.end() ) { fedIds_.push_back(fed_id); }
	else {
	  edm::LogWarning("Unknown") 
	    << "[SiStripFedIdListReader::" << __func__ << "]"
	    << " FedId " << fed_id << " has already been found in file!" << std::endl;
	  continue;
	}

      } else if ( inputFile_.eof() ) {
	edm::LogVerbatim("Unknown") 
	  << "[SiStripFedIdListReader::" << __func__ << "]"
	  << " End of file reached! Found " << fedIds_.size() 
	  << " valid FedIds!" << std::endl;
	break;
      } else if ( inputFile_.fail() ) {
	edm::LogVerbatim("Unknown") 
	  << "[SiStripFedIdListReader::" << __func__ << "]"
	  << " Error while reading file \"" << filePath << "\"!" << std::endl;
	break;
      }
    }

    inputFile_.close();
  } else {
    edm::LogVerbatim("Unknown") 
      << "[SiStripFedIdListReader::" << __func__ << "]"
      << " Unable to open file \"" << filePath << "\"!" << std::endl;
    return;
    
  }

}

// -----------------------------------------------------------------------------
//
SiStripFedIdListReader::SiStripFedIdListReader( const SiStripFedIdListReader& copy ) {
  edm::LogVerbatim("Unknown") 
    << "[SiStripFedIdListReader::" << __func__ << "]";
  fedIds_ = copy.fedIds_;
}

// -----------------------------------------------------------------------------
//
SiStripFedIdListReader& SiStripFedIdListReader::operator=( const SiStripFedIdListReader& copy ) {
  edm::LogVerbatim("Unknown") 
    << "[SiStripFedIdListReader::" << __func__ << "]";
  fedIds_ = copy.fedIds_;
  return *this;  
}

// -----------------------------------------------------------------------------
//
SiStripFedIdListReader::~SiStripFedIdListReader(){
  edm::LogVerbatim("Unknown") 
    << "[SiStripFedIdListReader::" << __func__ << "]";
}


// -----------------------------------------------------------------------------
//
std::ostream& operator<< ( std::ostream& os, const SiStripFedIdListReader& in ) {
  std::vector<uint16_t> fed_ids = in.fedIds();
  std::stringstream ss;
  ss << "[SiStripFedIdListReader::" << __func__ << "]"
     << " Found " << fed_ids.size() << " valid FED ids with values: ";
  std::vector<uint16_t>::const_iterator iter = fed_ids.begin();
  for ( ; iter != fed_ids.end(); ++iter ) { ss << *iter << " "; }
  os << ss.str();
  return os;
}

