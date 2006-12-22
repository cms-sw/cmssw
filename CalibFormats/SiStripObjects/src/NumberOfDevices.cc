#include "CalibFormats/SiStripObjects/interface/NumberOfDevices.h"
#include <iomanip>

using namespace std;

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
  nFedIds_ = 0;
  nFedChans_ = 0;
  nDcus_ = 0;
  nMuxes_ = 0;
  nPlls_ = 0;
  nLlds_ = 0;
}

// -----------------------------------------------------------------------------
//
void NumberOfDevices::print( stringstream& ss ) const {
  ss << "  FEC crates   : " << nFecCrates_ << endl
     << "  FEC slots    : " << nFecSlots_ << endl
     << "  FEC rings    : " << nFecRings_ << endl
     << "  CCU addrs    : " << nCcuAddrs_ << endl
     << "  CCU chans    : " << nCcuChans_ << endl
     << "  DCU ids      : " << nDcuIds_ << endl
     << "  DCUs         : " << nDcus_ << endl
     << "  MUXes        : " << nMuxes_ << endl
     << "  PLLs         : " << nPlls_ << endl
     << "  LLDs         : " << nLlds_ << endl
     << "  DET ids      : " << nDetIds_ << endl
     << "  APV pairs    : " << nApvPairs_ << endl
     << "  APVs         : " << nApvs_ << endl
     << "  FED channels : " << nFedChans_ << endl
     << "  Number of APV pairs per module (0/1/2/3/other)    : " 
     << nApvPairs0_ << " / " 
     << nApvPairs1_ << " / "
     << nApvPairs2_ << " / "
     << nApvPairs3_ << " / "
     << nApvPairsX_ << endl
     << "  Total number of modules / channels (nApvPairs<=3) : " 
     << ( nApvPairs0_ + nApvPairs1_ + nApvPairs2_ + nApvPairs3_ ) << " / "
     << ( 0*nApvPairs0_ + 1*nApvPairs1_ + 2*nApvPairs2_ + 3*nApvPairs3_ );
}
  
// -----------------------------------------------------------------------------
//
ostream& operator<< ( ostream& os, const NumberOfDevices& devs ) {
  stringstream ss;
  devs.print(ss);
  os << ss.str();
  return os;
}
