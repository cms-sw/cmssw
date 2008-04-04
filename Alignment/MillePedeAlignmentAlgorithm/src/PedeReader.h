#ifndef MILLEPEDEPEDEREADER_H
#define MILLEPEDEPEDEREADER_H

/**
 * \class PedeReader
 *
 * read in result from pede text file
 *
 *  \author    : Gero Flucke
 *  date       : November 2006
 *  $Revision: 1.1 $
 *  $Date: 2006/11/14 08:45:05 $
 *  (last update by $Author: flucke $)
 */

#include <fstream>
#include <vector>

class PedeSteerer;
class Alignable;
class AlignmentParameters;

namespace edm {
  class ParameterSet;
}

/***************************************
****************************************/
class PedeReader
{
 public:
  PedeReader(const edm::ParameterSet &config, const PedeSteerer &steerer);
  /// non virtual destructor: do not inherit from this class
  ~PedeReader() {}
  bool read(std::vector<Alignable*> &alignables);
  /// true if 'outValue' could be read via operator >> from the current line (!) of aStream,
  /// false otherwise
  template<class T>
    bool readIfSameLine(std::ifstream &aStream, T &outValue) const;

  Alignable* setParameter(unsigned int paramLabel, unsigned int bufLength, float *buf) const;
  /// returns parameters of alignable (creates - including MillePedeUser - if not yet existing)
  AlignmentParameters* checkAliParams(Alignable *alignable) const;
 private:
  //  PedeReader() {} // no default ctr.

  std::ifstream      myPedeResult;
  const PedeSteerer &mySteerer;

  static const unsigned int myMaxNumValPerParam;
};

#endif
