#ifndef PEDESTEERER_H
#define PEDESTEERER_H

/**
 * \class PedeSteerer
 *
 * provides steering for pede according to configuration
 *
 * \author    : Gero Flucke
 * date       : October 2006
 * $Date: 2007/05/11 16:14:43 $
 * $Revision: 1.8.2.1 $
 * (last update by $Author: flucke $)
 */

#include <vector>
#include <map> 
#include <string>
// forward ofstream:
#include <iosfwd> 

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class Alignable;
class AlignableTracker;
class AlignableMuon;
class AlignmentParameterStore;

/***************************************
****************************************/
class PedeSteerer
{
 public:
  /// constructor from e.g. AlignableTracker/AlignableMuon and the AlignmentParameterStore
  PedeSteerer(AlignableTracker *aliTracker, AlignableMuon *aliMuon,
	      AlignmentParameterStore *store, const edm::ParameterSet &config,
	      const std::string &defaultDir = "");
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
  /// If some parameters Alignable* were chosen to be excluded as subcomponent of a hierarchical
  /// parameterisation, return vector indicating these parameters (all but '0' mean excluded).
  /// Otherwise empty return value.
  const std::vector<char>& noHieraParamSel(const Alignable* ali) const;

  /// construct (and return name of) master steering file from config, binaryFiles etc.
  std::string buildMasterSteer(const std::vector<std::string> &binaryFiles);
  /// run pede, masterSteer should be as returned from buildMasterSteer(...)
  bool runPede(const std::string &masterSteer) const;
  double cmsToPedeFactor(unsigned int parNum) const;
  /// results from pede (and start values for pede) might need a sign flip
  double parameterSign() const;
  /// directory from constructor input, '/' is attached if needed
  const std::string& directory() const { return myDirectory;}

 private:
  typedef std::map <Alignable*, unsigned int> AlignableToIdMap;
  typedef AlignableToIdMap::value_type AlignableToIdPair;
  typedef std::map <unsigned int, Alignable*> IdToAlignableMap;
  typedef std::map<const Alignable*, std::vector<char> > AlignableSelVecMap;

  unsigned int buildMap(Alignable *highestLevelAli1, Alignable *highestLevelAli2);
  unsigned int buildReverseMap();
  void buildNoHierarchyMap(AlignableTracker *aliTracker, AlignableMuon *aliMuon,
			   const edm::ParameterSet &selPSet);

  std::pair<unsigned int, unsigned int> fixParameters(const std::vector<Alignable*> &alignables,
						      const std::string &file);
  int fixParameter(Alignable *ali, unsigned int iParam, char selector, std::ofstream &file) const;
  unsigned int hierarchyConstraints(const std::vector<Alignable*> &alis, const std::string &file);
  void hierarchyConstraint(const Alignable *ali, const std::vector<Alignable*> &components,
			   std::ofstream &file) const;

  /// full name with directory and 'idenitfier'
  std::string fileName(const std::string &addendum) const;
  /// create and open file with name, if (addToList) append to mySteeringFiles
  std::ofstream* createSteerFile(const std::string &name, bool addToList);

  const AlignmentParameterStore *myParameterStore;
  edm::ParameterSet myConfig;
  std::string myDirectory; /// directory of all files

  std::vector<std::string> mySteeringFiles; /// keeps track of created 'secondary' steering files
  AlignableToIdMap  myAlignableToIdMap; /// providing unique ID for alignable, space for param IDs
  IdToAlignableMap  myIdToAlignableMap; /// reverse map

  AlignableSelVecMap myNoHierarchyMap; /// Alignable(Params) deselected for hierarchy constraints

  static const unsigned int theMaxNumParam;
  static const unsigned int theMinLabel;
};

#endif
