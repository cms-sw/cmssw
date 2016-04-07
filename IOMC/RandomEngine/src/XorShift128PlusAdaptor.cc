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

namespace {
  constexpr double uint64_norm = 1.0/((double)std::numeric_limits<uint64_t>::max());
}

namespace edm {

XorShift128PlusAdaptor::XorShift128PlusAdaptor() {
  
  srand(time(NULL));
  for( auto& seed : seeds ) {
    seed = rand();
  }
}

XorShift128PlusAdaptor::XorShift128PlusAdaptor( const std::array<uint64_t,2>& inseeds ) {  
  if( inseeds.size() < 2 ) {
    Grumble(std::string("XorShift128Plus needs two 64 bit seeds!"));
  }
  
  if( inseeds[0] == 0 && inseeds[1] == 0 ) {
    Grumble(std::string("XorShift128Plus cannot be seeded with all zeroes!"));
  }
  seeds = inseeds;
  theSeeds = reinterpret_cast<const int64_t*>(seeds.data());
}

XorShift128PlusAdaptor::XorShift128PlusAdaptor( const std::array<uint32_t,4>& inseeds ) {  
  if( inseeds.size() < 4 ) {
    Grumble(std::string("XorShift128Plus needs four 32 bit seeds!"));
  }
  
  if( inseeds[0] == 0 && inseeds[1] == 0 && 
      inseeds[2] == 0 && inseeds[2] == 0    ) {
    Grumble(std::string("XorShift128Plus cannot be seeded with all zeroes!"));
  }

  seeds.fill(0ULL);
  for( size_t i = 0; i < inseeds.size() && i < 4; ++i ) {
    uint64_t temp = inseeds[i];
    seeds[i/2] += ( temp <<( i%2 ? 0 : 32 ) ); 
  }
  
  theSeeds = reinterpret_cast<const int64_t*>(seeds.data());
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
  for(int i = 0; i < 2; ++i) {
    v.push_back(seeds[i]&uint_max);
    v.push_back((seeds[i]>>32)&uint_max);
  }
  return v;
}

uint64_t  XorShift128PlusAdaptor::getNumber() {
  uint64_t x = seeds[0];
  const uint64_t y = seeds[1];
  seeds[0] = y;
  x ^= x << 23; // a
  seeds[1] = x ^ y ^ (x >> 17) ^ (y >> 26); // b, c
  return seeds[1] + y;
}

double XorShift128PlusAdaptor::flat() { 
  return uint64_norm*getNumber(); 
}

void XorShift128PlusAdaptor::flatArray(int const size, double* vect) {
  for( int i = 0; i < size; ++i ) {
    vect[i] = flat();
  }
}

void XorShift128PlusAdaptor::setSeed(long seed, int idx) { 
  if(idx >= 4) Grumble(std::string("XorShift128Plus only has four seeds!"));
  const int shift   = 32*(idx%2);
  const int realidx = idx/2;
  seeds[realidx] = seeds[realidx] & ~((uint64_t)uint_max << shift*32);
  seeds[realidx] += (uint64_t)seed << (shift*32);
  theSeeds = reinterpret_cast<const int64_t*>(seeds.data());
}

// Sets the state of the algorithm according to the zero terminated
// array of seeds. It is allowed to ignore one or many seeds in this array.
void XorShift128PlusAdaptor::setSeeds(long const* inseeds, int) { 
  seeds.fill(0ULL);
  for(unsigned i = 0; i < 4; ++i ) {
    const int shift   = 32*(i%2);
    const int realidx = i/2;
    seeds[realidx] += (uint64_t)inseeds[i] << shift;
  }
  theSeeds = reinterpret_cast<const int64_t*>(seeds.data());
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
    for (int i=0; i<2; ++i)
      inFile >> seeds[i];
  }
}

void XorShift128PlusAdaptor::showStatus() const {
  edm::LogVerbatim("XorShift128Plus") << std::endl;
  edm::LogVerbatim("XorShift128Plus") << "--------- XorShift128Plus engine status ---------" << std::endl;
  edm::LogVerbatim("XorShift128Plus") << " Current seeds = "
                                      << seeds[0] << ", "
                                      << seeds[1] << ", "
                                      << std::endl;
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
  
  seeds.fill(0ULL);
  for(unsigned i = 0; i < 4; ++i ) {
    const int shift   = 32*(i%2);
    const int realidx = i/2;
    seeds[realidx] += (uint64_t)(v[i+1]) << shift;
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
