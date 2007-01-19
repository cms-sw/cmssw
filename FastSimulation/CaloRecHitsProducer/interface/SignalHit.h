#ifndef FastSimulation__SignalHit__h
#define FastSimulation__SignalHit__h

#include <boost/cstdint.hpp>

// Created 16.01.06.  F. Beaudette
// When doing the RecHits, the noise is injected. 
// It isn't injected everywhere, the number of cells above the threshold
// is computed. And noise is injected in randomly determined cells 
// provided they don't contain signal.
// One has however to be a little bit careful and not inject noise in 
// cells containing signal but which have been removed because the 
// energy was below the threshold
// This class is aimed at saving this information


class SignalHit
{
 public:
  SignalHit(uint32_t val ,bool killed=false):val_(val),killed_(killed)
    {;
    }
    ~SignalHit() {;};
    inline uint32_t operator()() const {return val_;};	
    inline uint32_t id() const {return val_;}
    inline bool operator<(const SignalHit& comp) const {return val_<comp.id();}
    inline bool operator==(const SignalHit& comp) const {return val_==comp.id();}
    inline bool killed() const {return killed_;}
 private:
    uint32_t val_;
    bool killed_;
};
#endif
