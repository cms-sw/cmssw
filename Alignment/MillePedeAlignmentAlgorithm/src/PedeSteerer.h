#ifndef PEDESTEERER_H
#define PEDESTEERER_H

/**
 * \class PedeSteerer
 *
 * provides steering for pede according to configuration
 *
 * \author    : Gero Flucke
 * date       : October 2006
 * $Date: 2007/06/21 12:43:03 $
 * $Revision: 1.8.2.4 $
 * (last update by $Author: flucke $)
 */

#include <vector>
#include <map> 
#include <set> 
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
  /// True if 'ali' was deselected from hierarchy and any ancestor (e.g. mother) has parameters.
  bool isNoHiera(const Alignable* ali) const;

  /// construct (and return name of) master steering file from config, binaryFiles etc.
  std::string buildMasterSteer(const std::vector<std::string> &binaryFiles);
  /// run pede, masterSteer should be as returned from buildMasterSteer(...)
  bool runPede(const std::string &masterSteer) const;
  /// If reference alignables have been configured, shift everything such that mean
  /// position and orientation of dets in these alignables are zero.
  void correctToReferenceSystem();

  double cmsToPedeFactor(unsigned int parNum) const;
  /// results from pede (and start values for pede) might need a sign flip
  int parameterSign() const { return myParameterSign; }
  /// directory from constructor input, '/' is attached if needed
  const std::string& directory() const { return myDirectory;}

 private:
  typedef std::map <Alignable*, unsigned int> AlignableToIdMap;
  typedef AlignableToIdMap::value_type AlignableToIdPair;
  typedef std::map <unsigned int, Alignable*> IdToAlignableMap;

  unsigned int buildMap(Alignable *highestLevelAli1, Alignable *highestLevelAli2);
  unsigned int buildReverseMap();
  /// Store Alignables that have SelectionUserVariables attached to their AlignmentParameters
  /// (these must exist) that indicate removal from hierarchy, i.e. make it 'top level'.
  unsigned int buildNoHierarchyCollection(const std::vector<Alignable*> &alis);
  /// Checks whether 'alignables' have SelectionUserVariables attached to their AlignmentParameters
  /// (these must exist) that indicate fixation of a parameter, a steering 'file'
  /// is created accordingly.
  /// Returns number of parameters fixed at 0 and at 'nominal truth'.
  std::pair<unsigned int, unsigned int> fixParameters(const std::vector<Alignable*> &alignables,
						      const std::string &file);
  /// If 'selector' means fixing, create corresponding steering file line in file pointed to
  /// by 'filePtr'. If 'filePtr == 0' create file with name 'fileName'
  /// (and return pointer via reference).
  int fixParameter(Alignable *ali, unsigned int iParam, char selector, std::ofstream* &filePtr,
		   const std::string &fileName);
  /// Return 'alignables' that have SelectionUserVariables attached to their AlignmentParameters
  /// (these must exist) that indicate a definition of a coordinate system.
  /// Throws if ill defined reference objects.
  std::vector<Alignable*> selectCoordinateAlis(const std::vector<Alignable*> &alignables) const;
  /// Create steering file with constraints defining coordinate system via hierarchy constraints
  /// between 'aliMaster' and 'alis'; 'aliMaster' must not have parameters: would not make sense!
  void defineCoordinates(const std::vector<Alignable*> &alis, Alignable *aliMaster,
			 const std::string &fileName);

  unsigned int hierarchyConstraints(const std::vector<Alignable*> &alis, const std::string &file);
  void hierarchyConstraint(const Alignable *ali, const std::vector<Alignable*> &components,
			   std::ofstream &file) const;

  /// full name with directory and 'idenitfier'
  std::string fileName(const std::string &addendum) const;
  /// create and open file with name, if (addToList) append to mySteeringFiles
  std::ofstream* createSteerFile(const std::string &name, bool addToList);

  // data members
  const AlignmentParameterStore *myParameterStore;
  edm::ParameterSet myConfig;
  std::string myDirectory; /// directory of all files
  int myParameterSign; /// old pede versions (before May '07) need a sign flip...

  std::vector<std::string> mySteeringFiles; /// keeps track of created 'secondary' steering files
  AlignableToIdMap  myAlignableToIdMap; /// providing unique ID for alignable, space for param IDs
  IdToAlignableMap  myIdToAlignableMap; /// reverse map

  std::set<const Alignable*> myNoHieraCollection; /// Alignables deselected for hierarchy constr.
  Alignable *theCoordMaster;                      /// master coordinates, must (?) be global frame
  std::vector<Alignable*> theCoordDefiners;      /// Alignables selected to define coordinates
  
  static const unsigned int theMaxNumParam;
  static const unsigned int theMinLabel;
};

#endif
