#ifndef MILLEPEDEPEDEREADER_H
#define MILLEPEDEPEDEREADER_H

/**
 * \class PedeReader
 *
 * read in result from pede text file
 *
 *  \author    : Gero Flucke
 *  date       : November 2006
 *  $Revision: 1.2 $
 *  $Date: 2007/03/16 17:06:54 $
 *  (last update by $Author: flucke $)
 */

#include <fstream>
#include <vector>

class PedeSteerer;
class PedeLabeler;
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
  PedeReader(const edm::ParameterSet &config, const PedeSteerer &steerer,
	     const PedeLabeler &labels);
  /// non virtual destructor: do not inherit from this class
  ~PedeReader() {}
  /// Read pede output into AlignmentParameters attached to 'alignables',
  /// if (setUserVars == true) also care about MillePedeVariables.
  bool read(std::vector<Alignable*> &alignables, bool setUserVars);
  /// true if 'outValue' could be read via operator >> from the current line (!) of aStream,
  /// false otherwise
  template<class T>
    bool readIfSameLine(std::ifstream &aStream, T &outValue) const;
  /// Set pede results stored in 'buf' to AlignmentParameters 
  /// and (if setUserVars == true) to MillePedeVariables, return corresponding Alignable.
  Alignable* setParameter(unsigned int paramLabel, unsigned int bufLength, float *buf,
			  bool setUserVars) const;
  /// returns parameters of alignable (creates if not yet existing, but MillePedeVariables
  /// are only created if createUserVars == true)
  AlignmentParameters* checkAliParams(Alignable *alignable, bool createUserVars) const;
 private:
  //  PedeReader() {} // no default ctr.

  std::ifstream      myPedeResult;
  const PedeSteerer &mySteerer;
  const PedeLabeler &myLabels;

  static const unsigned int myMaxNumValPerParam;
};

#endif
