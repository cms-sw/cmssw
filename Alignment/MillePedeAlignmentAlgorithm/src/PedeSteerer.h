#ifndef PEDESTEERER_H
#define PEDESTEERER_H

/**
 * \class PedeSteerer
 *
 * provides steering for pede according to configuration
 *
 * \author    : Gero Flucke
 * date       : October 2006
 * $Date: 2007/02/13 12:03:11 $
 * $Revision: 1.7 $
 * (last update by $Author: flucke $)
 */

#include <vector>
#include <map> 
#include <string>
// forward ofstream:
#include <iosfwd> 

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class Alignable;
class AlignmentParameterStore;

/***************************************
****************************************/
class PedeSteerer
{
 public:
  /// constructor from e.g. AlignableTracker and the AlignmentParameterStore
  PedeSteerer(Alignable *highestLevelAlignable, AlignmentParameterStore *store,
	      const edm::ParameterSet &config);
  /** non-virtual destructor: do not inherit from this class */
  ~PedeSteerer();
    
  /// uniqueId of Alignable, 0 if alignable not known
  /// between this ID and the next there is enough 'space' to add parameter
  /// numbers 0...nPar-1 to make unique IDs for the labels of active parameters
  unsigned int alignableLabel(Alignable *alignable) const;
  unsigned int parameterLabel(unsigned int aliLabel, unsigned int parNum) const;
  
  /// parameter number, 0 <= .. < theMaxNumParam, belonging to unique parameter label
  unsigned int paramNumFromLabel(unsigned int paramLabel) const;
  /// alignable label from parameter label (works also for alignable label...)
  unsigned int alignableLabelFromLabel(unsigned int label) const;
  /// alignable from alignable or parameter label
  Alignable* alignableFromLabel(unsigned int label) const;

  /// construct (and return name of) master steering file from config, binaryFiles etc.
  std::string buildMasterSteer(const std::vector<std::string> &binaryFiles);
  /// run pede, masterSteer should be as returned from buildMasterSteer(...)
  bool runPede(const std::string &masterSteer) const;
  double cmsToPedeFactor(unsigned int parNum) const;
  /// results from pede (and start values for pede) might need a sign flip
  double parameterSign() const;
  /// directory from configuration, '/' is attached if needed
  std::string directory() const;

 private:
  typedef std::map <Alignable*, unsigned int> AlignableToIdMap;
  typedef std::pair<Alignable*, unsigned int> AlignableToIdPair;

  typedef std::map <unsigned int, Alignable*> IdToAlignableMap;
  typedef std::pair<unsigned int, Alignable*> IdToAlignablePair;

  unsigned int buildMap(Alignable *highestLevelAli);
  unsigned int buildReverseMap();

  std::pair<unsigned int, unsigned int> fixParameters(const std::vector<Alignable*> &alignables,
						      const std::string &file);
  int fixParameter(Alignable *ali, unsigned int iParam, char selector, std::ofstream &file) const;
  unsigned int hierarchyConstraints(const std::vector<Alignable*> &alis,
				    const std::string &file);
  void hierarchyConstraint(const Alignable *ali, const std::vector<Alignable*> &components,
			   std::ofstream &file) const;

  /// full name with directory and 'idenitfier'
  std::string fileName(const std::string &addendum) const;
  /// create and open file with name, if (addToList) append to mySteeringFiles
  std::ofstream* createSteerFile(const std::string &name, bool addToList);

  const AlignmentParameterStore *myParameterStore;
  edm::ParameterSet myConfig;
  std::vector<std::string> mySteeringFiles; /// keeps track of created 'secondary' steering files
  AlignableToIdMap  myAlignableToIdMap; /// providing unique ID for alignable, space for param IDs
  IdToAlignableMap  myIdToAlignableMap; /// reverse map

  static const unsigned int theMaxNumParam;
  static const unsigned int theMinLabel;
};

#endif
