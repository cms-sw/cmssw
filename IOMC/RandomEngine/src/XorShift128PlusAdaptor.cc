#include "IOMC/RandomEngine/src/XorShift128PlusAdaptor.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "CLHEP/Random/engineIDulong.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TBufferFile.h"

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdint.h>
#include <string>
#include <time.h>
#include <limits>

namespace edm {

XorShift128PlusAdaptor::XorShift128PlusAdaptor() {
  srand(time(NULL));
  for( auto& seed : seeds ) {
    seed = rand();
  }
}

XorShift128PlusAdaptor::XorShift128PlusAdaptor( const std::array<uint64_t,2>& inseeds ) {  
  if( inseeds[0] == 0 && inseeds[1] == 0 ) {
    Grumble(std::string("XorShift128Plus cannot be seeded with all zeroes!"));
  }

  for( unsigned i = 0; i < inseeds.size(); ++i ) {
    seeds[2*i] = inseeds[i]&0xffffffff;
    seeds[2*i+1] = (inseeds[i]>>32)&0xffffffff;    
  }
  
}

XorShift128PlusAdaptor::XorShift128PlusAdaptor( const std::array<uint32_t,4>& inseeds ) {
  if( inseeds[0] == 0 && inseeds[1] == 0  && 
      inseeds[2] == 0 && inseeds[3] == 0 ) {
    Grumble(std::string("XorShift128Plus cannot be seeded with all zeroes!"));
  }

  for( unsigned i = 0; i < inseeds.size(); ++i ) {
    seeds[i] = inseeds[i];
  }  
}

XorShift128PlusAdaptor::XorShift128PlusAdaptor(std::istream&) {
  Grumble(std::string("Cannot instantiate a TRandom engine from an istream"));
}

XorShift128PlusAdaptor::~XorShift128PlusAdaptor() {
}

std::ostream& XorShift128PlusAdaptor::put(std::ostream& os) const {
  Grumble(std::string("put(std::ostream) not available for TRandom engines"));
  return os;
}

std::vector<unsigned long> XorShift128PlusAdaptor::put() const {
  std::vector<unsigned long> v;
  v.reserve(5);
  v.push_back(CLHEP::engineIDulong<XorShift128PlusAdaptor>());
  for(int i = 0; i < 4; ++i) v.push_back(seeds[i]);
  return v;
}

uint64_t  XorShift128PlusAdaptor::getNumber() {
  uint64_t s1 = (uint64_t)seeds[0] + (((uint64_t)seeds[1])<<32);
  const uint64_t s0 = (uint64_t)seeds[2] + (((uint64_t)seeds[3])<<32);
  seeds[0] = s0&0xffffffff;
  seeds[1] = (s0>>32)&0xffffffff;
  s1 ^= s1 << 23; // a
  s1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);// b, c
  seeds[2] = s1&0xffffffff; 
  seeds[3] = (s1>>32)&0xffffffff;
  return s1 + s0; 
}

double XorShift128PlusAdaptor::flat() { 
  return ((double)getNumber())/((double)std::numeric_limits<uint64_t>::max()); 
}

void XorShift128PlusAdaptor::flatArray(int const size, double* vect) {
  for( int i = 0; i < size; ++i ) {
    vect[i] = flat();
  }
}

void XorShift128PlusAdaptor::setSeed(long seed, int idx) { 
  if(idx > 2) Grumble(std::string("XorShift128Plus only has two seeds, setting to %2 of idx!"));
  const int realidx = idx%4;
  seeds[realidx]  = (uint32_t)seed;
}

// Sets the state of the algorithm according to the zero terminated
// array of seeds. It is allowed to ignore one or many seeds in this array.
void XorShift128PlusAdaptor::setSeeds(long const* inseeds, int) { 
  for(unsigned i = 0; i < 4; ++i ) {
    if( seeds[i] == 0 ) break;
    seeds[i] = (uint32_t)inseeds[i];
  }
}

// Saves the current engine status in the named file
void XorShift128PlusAdaptor::saveStatus(char const filename[]) const {
  std::ofstream outFile( filename, std::ios::app ) ;  
  if (!outFile.bad()) {
    outFile << "Uvec\n";
    std::vector<unsigned long> v = put();
    for (unsigned int i=0; i<v.size(); ++i) {
      outFile << v[i] << "\n";
    }
  }
}

// Reads from named file the the last saved engine status and restores it.
void XorShift128PlusAdaptor::restoreStatus(char const filename[]) {
  std::ifstream inFile( filename, std::ios::in);
  if (!checkFile ( inFile, filename, engineName(), "restoreStatus" )) {
    edm::LogVerbatim("XorShift128Plus") << "  -- Engine state remains unchanged\n";
    return;
  }
  if ( CLHEP::possibleKeywordInput ( inFile, "Uvec", theSeed ) ) {
    std::vector<unsigned long> v;
    unsigned long xin;
    for (unsigned int ivec=0; ivec < VECTOR_STATE_SIZE; ++ivec) {
      inFile >> xin;
      if (!inFile) {
        inFile.clear(std::ios::badbit | inFile.rdstate());
        edm::LogVerbatim("XorShift128Plus") << "\nXorShift128Plus state (vector) description improper."
                                            << "\nrestoreStatus has failed."
                                            << "\nInput stream is probably mispositioned now." << std::endl;
        return;
      }
      v.push_back(xin);
    }
    getState(v);
    return;
  }
  
  if (!inFile.bad() && !inFile.eof()) {
    for (int i=0; i<4; ++i)
      inFile >> seeds[i];
  }
}

void XorShift128PlusAdaptor::showStatus() const {
  edm::LogVerbatim("XorShift128Plus") << std::endl;
  edm::LogVerbatim("XorShift128Plus") << "--------- XorShift128Plus engine status ---------" << std::endl;
  edm::LogVerbatim("XorShift128Plus") << " Current seeds = "
                                      << seeds[0] << ", "
                                      << seeds[1] << ", " 
                                      << seeds[2] << ", "
                                      << seeds[3] << std::endl;
  edm::LogVerbatim("XorShift128Plus") << "----------------------------------------" << std::endl;
}

std::istream& XorShift128PlusAdaptor::get(std::istream& is) {
  Grumble(std::string("get(std::istream) not available for TRandom engines"));
  return getState(is);
}

std::istream& XorShift128PlusAdaptor::getState(std::istream& is) {
  Grumble(std::string("getState(std::istream) not available for TRandom engines"));
  return is;
}

bool XorShift128PlusAdaptor::get(std::vector<unsigned long> const& v) {
  if(v.empty() || v.size() != VECTOR_STATE_SIZE)  return false;
  if(v[0] != CLHEP::engineIDulong<XorShift128PlusAdaptor>()) return false;
  
  for(unsigned i = 0; i < 4; ++i ) {
    seeds[i] = v[i+1];
  }

  return true;
}

void XorShift128PlusAdaptor::Grumble(std::string const& errortext) const {

// Throw an edm::Exception for unimplemented functions
   std::ostringstream sstr;
   sstr << "Unimplemented Feature: " << errortext << '\n';
   edm::Exception except(edm::errors::UnimplementedFeature, sstr.str());
   throw except;
}

}  // namespace edm
