#ifndef Input_HepMCFileReader_h
#define Input_HepMCFileReader_h

/** \class HepMCFileReader
* 
*  This class is used by the implementation of DaqEventFactory present 
*  in this package to read in the full event raw data from a flat 
*  binary file. 
*  WARNING: If you want to use this class for other purposes you must 
*  always invoke the method initialize before starting using the interface
*  it exposes.
*
*  \author G. Bruno - CERN, EP Division
*/

#include <vector>
#include <map>

#include "FWCore/Utilities/interface/get_underlying_safe.h"

namespace HepMC {
  class IO_BaseClass;
  class GenEvent;
  class GenParticle;
}  // namespace HepMC

class HepMCFileReader {
protected:
  HepMCFileReader();

public:
  virtual ~HepMCFileReader();
  virtual void initialize(const std::string &filename);
  inline bool isInitialized() const;

  virtual bool setEvent(int event);
  virtual bool readCurrentEvent();
  virtual bool printHepMcEvent() const;
  HepMC::GenEvent *fillCurrentEventData();
  //  virtual bool fillEventData(HepMC::GenEvent *event);
  // this method prints the event information as
  // obtained by the input file in HepEvt style
  void printEvent() const;
  // get all the 'integer' properties of a particle
  // like mother, daughter, pid and status
  // 'j' is the number of the particle in the HepMc
  virtual void getStatsFromTuple(int &mo1, int &mo2, int &da1, int &da2, int &status, int &pid, int j) const;
  virtual void ReadStats();

  static HepMCFileReader *instance();

private:
  HepMC::IO_BaseClass const *input() const { return get_underlying_safe(input_); }
  HepMC::IO_BaseClass *&input() { return get_underlying_safe(input_); }

  // current  HepMC evt
  edm::propagate_const<HepMC::GenEvent *> evt_;
  edm::propagate_const<HepMC::IO_BaseClass *> input_;

  static HepMCFileReader *instance_;

  int rdstate() const;
  //maps to convert HepMC::GenParticle to particles # and vice versa
  // -> needed for HepEvt like output
  std::vector<HepMC::GenParticle *> index_to_particle;
  std::map<HepMC::GenParticle *, int> particle_to_index;
  // find index to HepMC::GenParticle* p in map m
  int find_in_map(const std::map<HepMC::GenParticle *, int> &m, HepMC::GenParticle *p) const;
};

bool HepMCFileReader::isInitialized() const { return input_ != nullptr; }

#endif
