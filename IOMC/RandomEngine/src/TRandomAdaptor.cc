#include "CLHEP/Random/defs.h"
#include "CLHEP/Random/Random.h"
#include "CLHEP/Random/engineIDulong.h"
#include "IOMC/RandomEngine/src/TRandomAdaptor.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "RVersion.h"
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,15,0)
#include "TBufferFile.h"
#else
#include "TBuffer.h"
#endif

#include <string>
#include <cstddef>
#include <cmath>
#include <sstream>
#include <iostream>
#include <iomanip>

using namespace std;

namespace edm {

TRandomAdaptor::TRandomAdaptor( std::istream& is )  
{
  Grumble(std::string("Cannot instantiate a TRandom engine from an istream"));
}

std::ostream & TRandomAdaptor::put ( std::ostream& os ) const
{
  Grumble(std::string("put(std::ostream) not available for TRandom engines"));
  return os;
}

std::vector<unsigned long> TRandomAdaptor::put () const {
  UInt_t itemSize = sizeof(UInt_t);
  std::vector<unsigned long> v;
  v.push_back (CLHEP::engineIDulong<TRandomAdaptor>());
  TBuffer buffer(TBuffer::kWrite,1024*itemSize);
  trand_->Streamer(buffer);
  buffer.SetReadMode();
  char* bufferPtr = buffer.Buffer();
  UInt_t numItems = (buffer.Length()+itemSize-1)/itemSize+1;
  for( int i=0; i<(int)numItems; ++i)  {
    v.push_back(*(unsigned long*)(bufferPtr+i*itemSize));
  }
  return v;
}

std::istream &  TRandomAdaptor::get ( std::istream& is )
{
  Grumble(std::string("get(std::istream) not available for TRandom engines"));
  return getState(is);
}

std::istream &  TRandomAdaptor::getState ( std::istream& is )
{
  Grumble(std::string("getState(std::istream) not available for TRandom engines"));
  return is;
}

bool TRandomAdaptor::get (const std::vector<unsigned long> & v) {
  if(v.empty())  return false;
  if(v[0] != CLHEP::engineIDulong<TRandomAdaptor>()) return false;
  size_t numItems = v.size()-1;
  int32_t itemSize = sizeof(UInt_t);
  TBuffer buffer(TBuffer::kWrite,numItems*itemSize+1024);
  char* bufferPtr = buffer.Buffer();
  for(int i=0; i<(int)numItems; ++i) {
    *(unsigned int*)(bufferPtr+i*itemSize) = v[i+1];
  }
  trand_->Streamer(buffer);
  return true;
}

void TRandomAdaptor::Grumble( std::string errortext ) const {

// Throw an edm::Exception for unimplemented functions
   std::ostringstream sstr;
   sstr << "Unimplemented Feature: " << errortext << '\n';
   edm::Exception except(edm::errors::UnimplementedFeature, sstr.str());
   throw except;
}

}  // namespace edm
