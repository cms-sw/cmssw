#ifndef PEDESTEERER_H
#define PEDESTEERER_H

/**
 * \class PedeSteerer
 *
 * provides steering for pede according to configuration
 *
 * \author    : Gero Flucke
 * date       : October 2006
 * $Date$
 * $Revision$
 * (last update by $Author$)
 */

#include <fstream>
#include <vector>
#include <map> 

class Alignable;
class AlignableTracker;
class AlignmentParameterStore;
namespace edm {
  class ParameterSet;
}

/***************************************
****************************************/
class PedeSteerer
{
 public:
  PedeSteerer(AlignableTracker *tracker, AlignmentParameterStore *store,
	      const edm::ParameterSet &config);
  /** non-virtual destructor: do not inherit from this class */
  ~PedeSteerer();
    
  /// uniqueId of Alignable, 0 if alignable not known
  /// between this ID and the next there is enough 'space' to add parameter
  /// numbers 0...nPar-1 to make unique IDs for the labels of active parameters
  unsigned int alignableLabel(const Alignable *alignable) const;
  unsigned int parameterLabel(unsigned int aliLabel, unsigned int parNum) const;

 private:
  typedef std::map <const Alignable*, unsigned int> AlignableToIdMap;
  typedef std::pair<const Alignable*, unsigned int> AlignableToIdPair;

  unsigned int buildMap(Alignable *highestLevelAli);
  unsigned int fixParameters(AlignmentParameterStore *store, const edm::ParameterSet &config);
  bool insideRanges(double value, const std::vector<double> &ranges) const;

  std::ofstream     mySteerFile; // text file
  AlignableToIdMap  myAlignableToIdMap; 

};

#endif
