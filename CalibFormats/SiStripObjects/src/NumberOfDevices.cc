// Last commit: $Id: NumberOfDevices.cc,v 1.10 2007/12/19 17:51:54 bainbrid Exp $

#include "CalibFormats/SiStripObjects/interface/NumberOfDevices.h"
#include <iomanip>

// -----------------------------------------------------------------------------
//
void NumberOfDevices::clear() {
  nFecCrates_ = 0;
  nFecSlots_ = 0;
  nFecRings_ = 0;
  nCcuAddrs_ = 0;
  nCcuChans_ = 0;
  nApvs_ = 0;
  nDcuIds_ = 0;
  nDetIds_ = 0;
  nApvPairs_ = 0;
  nApvPairs0_ = 0; 
  nApvPairs1_ = 0; 
  nApvPairs2_ = 0; 
  nApvPairs3_ = 0;
  nApvPairsX_ = 0;
  nFedCrates_ = 0;
  nFedSlots_ = 0;
  nFedIds_ = 0;
  nFedChans_ = 0;
  nDcus_ = 0;
  nMuxes_ = 0;
  nPlls_ = 0;
  nLlds_ = 0;
}

// -----------------------------------------------------------------------------
//
void NumberOfDevices::print( std::stringstream& ss ) const {
  ss << "  FEC crates   : " << nFecCrates_ << std::endl
     << "  FEC slots    : " << nFecSlots_ << std::endl
     << "  FEC rings    : " << nFecRings_ << std::endl
     << "  CCU addrs    : " << nCcuAddrs_ << std::endl
     << "  CCU chans    : " << nCcuChans_ << std::endl
     << "  DCU ids      : " << nDcuIds_ << std::endl
     << "  DCUs         : " << nDcus_ << std::endl
     << "  MUXes        : " << nMuxes_ << std::endl
     << "  PLLs         : " << nPlls_ << std::endl
     << "  LLDs         : " << nLlds_ << std::endl
     << "  DET ids      : " << nDetIds_ << std::endl
     << "  APV pairs    : " << nApvPairs_ << std::endl
     << "  APVs         : " << nApvs_ << std::endl
     << "  FED crates   : " << nFedCrates_ << std::endl
     << "  FED slots    : " << nFedSlots_ << std::endl
     << "  FED ids      : " << nFedIds_ << std::endl
     << "  FED channels : " << nFedChans_ << std::endl
     << "  Number of APV pairs (0/1/2/3/>3) per module     : " 
     << nApvPairs0_ << "/" 
     << nApvPairs1_ << "/"
     << nApvPairs2_ << "/"
     << nApvPairs3_ << "/"
     << nApvPairsX_ << std::endl
     << "  Total number of modules/channels (nApvPairs<=3) : " 
     << ( nApvPairs0_ + nApvPairs1_ + nApvPairs2_ + nApvPairs3_ ) << "/"
     << ( 0*nApvPairs0_ + 1*nApvPairs1_ + 2*nApvPairs2_ + 3*nApvPairs3_ );
}
  
// -----------------------------------------------------------------------------
//
std::ostream& operator<< ( std::ostream& os, const NumberOfDevices& devs ) {
  std::stringstream ss;
  devs.print(ss);
  os << ss.str();
  return os;
}
