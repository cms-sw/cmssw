#ifndef IOMC_RandomEngine_TRandomAdaptor_h
#define IOMC_RandomEngine_TRandomAdaptor_h

#include "CLHEP/Random/RandomEngine.h"

#include "TRandom3.h"

#include <cmath>
#include "FWCore/Utilities/interface/value_ptr.h"

namespace edm {

  class TRandomAdaptor : public CLHEP::HepRandomEngine {

  public:
    typedef value_ptr<TRandom3> TRandom3Ptr;

    // Constructors and destructor.
    TRandomAdaptor();
    TRandomAdaptor( long seed );
    TRandomAdaptor( int rowIndex, int colIndex );
    TRandomAdaptor( std::istream & is );
    virtual ~TRandomAdaptor();

    // Returns a pseudo random number in ]0,1[ (i. e., excluding the end points).
    double flat() override { return trand_->Rndm(); }

    // Fills an array "vect" of specified size with flat random values.
    void flatArray(int const size, double* vect) override { trand_->RndmArray(size,vect); }

    // Sets the state of the algorithm according to seed.
    void setSeed(long seed, int) override;

    // Sets the state of the algorithm according to the zero terminated
    // array of seeds. It is allowed to ignore one or many seeds in this array.
    void setSeeds(long const* seeds, int) override;

    // Saves the current engine status in the named file
    void saveStatus(char const filename[] = "TRandom.conf") const override { trand_->WriteRandom(filename); }

    // Reads from named file the the last saved engine status and restores it.
    void restoreStatus(char const filename[] = "TRandom.conf" ) override { trand_->ReadRandom(filename); }

    // Dumps the current engine status on the screen.
    void showStatus() const override { trand_->Dump(); }

    // Returns a float flat ]0,1[
    operator float() override { return (float)(trand_->Rndm()); }

    // Returns an unsigned int (32-bit) flat 
    operator unsigned int() override { return (unsigned int)((trand_->Rndm())*exponent_bit_32()); }

    virtual std::ostream & put(std::ostream & os) const override;
    virtual std::istream & get(std::istream & is) override;
    std::string beginTag ( ) { return std::string(trand_->GetName())+std::string("-begin"); }
    virtual std::istream & getState ( std::istream & is ) override;

    // Returns the engine name as a string
    std::string name() const override { return std::string("T")+std::string(trand_->GetName()); }
    static std::string engineName() { return std::string("TRandomAdaptor"); }

    virtual std::vector<unsigned long> put () const override;
    bool get (std::vector<unsigned long> const& v) override;
    bool getState (std::vector<unsigned long> const& v) override { return get(v); }

    // In case all else fails, let the user talk directly to the engine
    TRandom3* getRootEngine() { return trand_.operator->(); }

  private:

    void Grumble(std::string const& errortext) const;

    mutable TRandom3Ptr trand_;

  }; // TRandomAdaptor

}  // namespace edm

#endif // IOMC_RandomEngine_TRandomAdaptor_h
