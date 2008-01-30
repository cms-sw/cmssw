#ifndef FWCore_Utilities_TRandomAdaptor_h
#define FWCore_Utilities_TRandomAdaptor_h

#include "CLHEP/Random/defs.h"
#include "CLHEP/Random/RandomEngine.h"

#include "TRandom3.h"
#include "TFile.h"
#include "TTree.h"

#include <cmath>

class TRandomAdaptor : public CLHEP::HepRandomEngine {

public:

  // Constructors and destructor.
  TRandomAdaptor() : trand_(new TRandom3()) {}
  TRandomAdaptor( long seed ) : trand_(new TRandom3(seed)) {}
  TRandomAdaptor( int rowIndex, int colIndex ) : trand_(new TRandom3(rowIndex*colIndex-1)) {}
  TRandomAdaptor( std::istream & is );
  virtual ~TRandomAdaptor() { delete trand_; }

  // Copy constructor and operator=.
  TRandomAdaptor( const TRandomAdaptor & p ) { TRandom3* tmp = new TRandom3();
                                                                   *tmp = *(p.trand_);
                                                                   this->trand_ = tmp; }

  TRandomAdaptor & operator=( const TRandomAdaptor & p ) { TRandom3* tmp = new TRandom3();
                                                                   *tmp = *(p.trand_);
                                                                   this->trand_ = tmp;
                                                                   return *this; }

  // Returns a pseudo random number in ]0,1[ (i. e., excluding the end points).
  double flat() { return trand_->Rndm(); }

  // Fills an array "vect" of specified size with flat random values.
  void flatArray(const int size, double* vect) { trand_->RndmArray(size,vect); }

  // Sets the state of the algorithm according to seed.
  void setSeed(long seed, int) { trand_->SetSeed(seed); }

  // Sets the state of the algorithm according to the zero terminated
  // array of seeds. It is allowed to ignore one or many seeds in this array.
  void setSeeds(const long * seeds, int) { trand_->SetSeed(seeds[0]); }

  // Saves the current engine status in the named file
  void saveStatus( const char filename[] = "TRandom.conf") const { trand_->WriteRandom(filename); }

  // Reads from named file the the last saved engine status and restores it.
  void restoreStatus( const char filename[] = "TRandom.conf" ) { trand_->ReadRandom(filename); }

  // Dumps the current engine status on the screen.
  void showStatus() const { trand_->Dump(); }

  // Returns a float flat ]0,1[
  operator float() { return (float)(trand_->Rndm()); }

  // Returns an unsigned int (32-bit) flat 
  operator unsigned int() { return (unsigned int)((trand_->Rndm())*exponent_bit_32); }

  virtual std::ostream & put (std::ostream & os) const;
  virtual std::istream & get (std::istream & is);
  std::string beginTag ( ) { return std::string(trand_->GetName())+std::string("-begin"); }
  virtual std::istream & getState ( std::istream & is );

  // Returns the engine name as a string
  std::string name() const { return std::string("T")+std::string(trand_->GetName()); }
  static std::string engineName() { return std::string("TRandomAdaptor"); }

  std::vector<unsigned long> put () const;
  bool get (const std::vector<unsigned long> & v);
  bool getState (const std::vector<unsigned long> & v) { return get(v); }

  // In case all else fails, let the user talk directly to the engine
  TRandom3* getRootEngine() { return trand_; }

private:

  void Grumble(std::string errortext) const;

  TRandom3* trand_;

}; // TRandomAdaptor

#endif // FWCore_Utilities_TRandomAdaptor_h
