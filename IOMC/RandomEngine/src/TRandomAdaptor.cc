#include "CLHEP/Random/defs.h"
#include "CLHEP/Random/Random.h"
#include "CLHEP/Random/engineIDulong.h"
#include "IOMC/RandomEngine/src/TRandomAdaptor.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "TBufferFile.h"

#include <string>
#include <cstddef>
#include <cmath>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace edm {

TRandomAdaptor::TRandomAdaptor(std::istream& is) {
  Grumble(std::string("Cannot instantiate a TRandom engine from an istream"));
}

TRandomAdaptor::~TRandomAdaptor() {
}

std::ostream & TRandomAdaptor::put (std::ostream& os) const {
  Grumble(std::string("put(std::ostream) not available for TRandom engines"));
  return os;
}

std::vector<unsigned long> TRandomAdaptor::put () const {
  int32_t itemSize = sizeof(unsigned long);
  std::vector<unsigned long> v;
  v.push_back (CLHEP::engineIDulong<TRandomAdaptor>());
  TBufferFile buffer(TBuffer::kWrite,1024*itemSize);
  trand_->Streamer(buffer);
  buffer.SetReadMode();
  char* bufferPtr = buffer.Buffer();
  int32_t numItems = (buffer.Length()+itemSize-1)/itemSize;
  for( int i=0; i<(int)numItems; ++i)  {
    v.push_back(*(unsigned long*)(bufferPtr+i*itemSize));
  }
  return v;
}

std::istream &  TRandomAdaptor::get (std::istream& is) {
  Grumble(std::string("get(std::istream) not available for TRandom engines"));
  return getState(is);
}

std::istream &  TRandomAdaptor::getState (std::istream& is) {
  Grumble(std::string("getState(std::istream) not available for TRandom engines"));
  return is;
}

bool TRandomAdaptor::get (std::vector<unsigned long> const& v) {
  if(v.empty())  return false;
  if(v[0] != CLHEP::engineIDulong<TRandomAdaptor>()) return false;
  int32_t numItems = v.size()-1;
  int32_t itemSize = sizeof(unsigned long);
  TBufferFile buffer(TBuffer::kRead,numItems*itemSize+1024);
  char* bufferPtr = buffer.Buffer();
  for(int i=0; i<(int)numItems; ++i) {
    *(unsigned long*)(bufferPtr+i*itemSize) = v[i+1];
  }
  trand_->Streamer(buffer);
  return true;
}

void TRandomAdaptor::Grumble(std::string const& errortext) const {

// Throw an edm::Exception for unimplemented functions
   std::ostringstream sstr;
   sstr << "Unimplemented Feature: " << errortext << '\n';
   edm::Exception except(edm::errors::UnimplementedFeature, sstr.str());
   throw except;
}

}  // namespace edm
