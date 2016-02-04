#include "IOMC/RandomEngine/src/TRandomAdaptor.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "CLHEP/Random/engineIDulong.h"

#include "TBufferFile.h"

#include <string>
#include <cstddef>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <stdint.h>

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
  std::vector<unsigned long> v;

  int32_t itemSize = sizeof(uint32_t);
  TBufferFile buffer(TBuffer::kWrite, 2048 * itemSize);
  trand_->Streamer(buffer);
  buffer.SetReadMode();
  char* bufferPtr = buffer.Buffer();
  int32_t numItems = (buffer.Length() + itemSize - 1) / itemSize;
  v.reserve(numItems + 1);
  v.push_back (CLHEP::engineIDulong<TRandomAdaptor>());
  for( int i = 0; i < numItems; ++i) {

    // Here we do some ugly manipulations to the data to match the format
    // of the output of the CLHEP put function (the whole point of this adaptor
    // is to make TRandom3 work through the CLHEP interface as if it were a
    // a CLHEP engine ...).  In CLHEP, the vector returned by put contains
    // unsigned long's, but these always contain only 32 bits of information.
    // In the case of a 64 bit build the top 32 bits is only padding (all 0's).

    // Get the next 32 bits of data from the buffer
    uint32_t value32 = *reinterpret_cast<uint32_t *>(bufferPtr + i * itemSize);

    if (i == numItems - 1) {
      int nBytes = buffer.Length() % itemSize;
      if (nBytes == 1) value32 &= 0xffu;
      else if (nBytes == 2) value32 &= 0xffffu;
      else if (nBytes == 3) value32 &= 0xffffffu;
    }

    // Push it into the vector in an unsigned long which may be 32 or 64 bits
    v.push_back(static_cast<unsigned long>(value32));
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

  int32_t itemSize = sizeof(uint32_t);
  TBufferFile buffer(TBuffer::kRead, numItems * itemSize + 1024);
  char* bufferPtr = buffer.Buffer();
  for(int32_t i = 0; i < numItems; ++i) {

    *reinterpret_cast<uint32_t *>(bufferPtr + i * itemSize) = static_cast<uint32_t>(v[i+1] & 0xffffffff);
  }

  // Note that this will fail if the TRandom3 version (specifically the TStreamerInfo)
  // has changed between the time the state was saved and the time the following call
  // is made.  Because we are manually calling the Streamer function, the original
  // TStreamerInfo is not saved anywhere. Normally ROOT saves the TStreamerInfo
  // automatically.
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
