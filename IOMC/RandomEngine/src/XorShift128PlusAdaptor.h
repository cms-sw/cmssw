#ifndef IOMC_RandomEngine_XorShift128PlusAdaptor_h
#define IOMC_RandomEngine_XorShift128PlusAdaptor_h

#include "CLHEP/Random/RandomEngine.h"

#include <cmath>
#include <cstdint>
#include <atomic>
#include <array>
#include <limits>

// This implements the "xorshift128plus" randomnumber generator from
// http://xorshift.di.unimi.it/xorshift128plus.c
// as a CLHEP::RandomEngine
// its period is 2^128 - 1
// This is one of the fastest PRNGs available that actually generates 
// passes standard tests.

namespace edm {

  class XorShift128PlusAdaptor : public CLHEP::HepRandomEngine {

  public:
    // Constructors and destructor.
    XorShift128PlusAdaptor();
    XorShift128PlusAdaptor( const std::array<uint64_t,2>& inseeds );
    XorShift128PlusAdaptor( const std::array<uint32_t,4>& inseeds );
    XorShift128PlusAdaptor( std::istream & is );
    virtual ~XorShift128PlusAdaptor();

    // Returns a pseudo random number in ]0,1[ (i. e., excluding the end points).
    double flat();

    // Fills an array "vect" of specified size with flat random values.
    void flatArray(int const size, double* vect);

    // Sets the state of the algorithm according to seed.
    void setSeed(long seed, int);

    // Sets the state of the algorithm according to the zero terminated
    // array of seeds. It is allowed to ignore one or many seeds in this array.
    void setSeeds(long const* seeds, int);

    // Saves the current engine status in the named file
    void saveStatus(char const filename[] = "TRandom.conf") const;

    // Reads from named file the the last saved engine status and restores it.
    void restoreStatus(char const filename[] = "TRandom.conf" );

    // Dumps the current engine status on the screen.
    void showStatus() const;

    // Returns a float flat ]0,1[
    operator float() { return (float)flat(); }

    // Returns an unsigned int (32-bit) flat 
    operator unsigned int() { return (unsigned int)(getNumber()&0xffffffff); }
    
    // returns a uint64_t
    operator uint64_t() { return getNumber(); } 

    virtual std::ostream & put (std::ostream & os) const;
    virtual std::istream & get (std::istream & is);
    std::string beginTag ( ) { return engineName()+std::string("-begin"); }
    virtual std::istream & getState ( std::istream & is );

    // Returns the engine name as a string
    std::string name() const { return engineName(); }
    static std::string engineName() { return std::string("XorShift128Plus"); }

    std::vector<unsigned long> put () const;
    bool get (std::vector<unsigned long> const& v);
    bool getState (std::vector<unsigned long> const& v) { return get(v); }
    
  private:

    uint64_t getNumber();

    void Grumble(std::string const& errortext) const;
    
    std::array<uint64_t, 2> seeds;

    static const unsigned int VECTOR_STATE_SIZE = 5; //convert from 64 to 32 bit
    static constexpr uint32_t uint_max = std::numeric_limits<uint32_t>::max();
    static constexpr double   uint_norm = 1.0/(double)uint_max;
  }; // XorShift128PlusAdaptor

}  // namespace edm

#endif // IOMC_RandomEngine_XorShift128PlusAdaptor_h
