#ifndef PEDESTEERER_H
#define PEDESTEERER_H

/**
 * \class PedeSteerer
 *
 * provides steering for pede according to configuration
 *
 * \author    : Gero Flucke
 * date       : October 2006
 * $Date: 2013/06/18 13:31:29 $
 * $Revision: 1.29 $
 * (last update by $Author: jbehr $)
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
class AlignableExtras;
class AlignmentParameterStore;
class PedeLabelerBase;
class PedeSteererWeakModeConstraints;

/***************************************
****************************************/
class PedeSteerer
{
 public:
  /// constructor from AlignableTracker/AlignableMuon, their AlignmentParameterStore and the labeler
  /// (NOTE: The latter two must live longer than the constructed PedeSteerer!)
  PedeSteerer(AlignableTracker *aliTracker, AlignableMuon *aliMuon, AlignableExtras *aliExtras,
	      AlignmentParameterStore *store,
	      const PedeLabelerBase *labels, const edm::ParameterSet &config,
	      const std::string &defaultDir, bool noSteerFiles);
  /** non-virtual destructor: do not inherit from this class **/
  ~PedeSteerer();
    
  /// True if 'ali' was deselected from hierarchy and any ancestor (e.g. mother) has parameters.
  bool isNoHiera(const Alignable* ali) const;

  /// construct steering files about hierarchy, fixing etc. an keep track of their names
  void buildSubSteer(AlignableTracker *aliTracker, AlignableMuon *aliMuon, AlignableExtras *aliExtras);
  /// construct (and return name of) master steering file from config, binaryFiles etc.
  std::string buildMasterSteer(const std::vector<std::string> &binaryFiles);
  /// run pede, masterSteer should be as returned from buildMasterSteer(...)
  int runPede(const std::string &masterSteer) const;
  /// If reference alignables have been configured, shift everything such that mean
  /// position and orientation of dets in these alignables are zero.
  void correctToReferenceSystem();
  bool isCorrectToRefSystem(const std::vector<Alignable*> &coordDefiners) const;


  double cmsToPedeFactor(unsigned int parNum) const;
  /// results from pede (and start values for pede) might need a sign flip
  int parameterSign() const { return myParameterSign; }
  /// directory from constructor input, '/' is attached if needed
  const std::string& directory() const { return myDirectory;}

 private:
  typedef std::map<const Alignable*,std::vector<float> > AlignablePresigmasMap;

  /// Checks whether SelectionUserVariables that might be attached to alis' AlignmentParameters
  /// (these must exist) are all known.
  bool checkParameterChoices(const std::vector<Alignable*> &alis) const;
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
  int fixParameter(Alignable *ali, unsigned int iRunRange,
		   unsigned int iParam, char selector, std::ofstream* &filePtr,
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

  /// interprete content of presigma VPSet 'cffPresi' and call presigmasFile
  unsigned int presigmas(const std::vector<edm::ParameterSet> &cffPresi,
			 const std::string &fileName, const std::vector<Alignable*> &alis,
			 AlignableTracker *aliTracker, AlignableMuon *aliMuon, AlignableExtras *aliExtras);
  /// look for active 'alis' in map of presigma values and create steering file 
  unsigned int presigmasFile(const std::string &fileName, const std::vector<Alignable*> &alis,
			     const AlignablePresigmasMap &aliPresisMap); 
  /// full name with directory and 'idenitfier'
  std::string fileName(const std::string &addendum) const;
  /// create and open file with name, if (addToList) append to mySteeringFiles
  std::ofstream* createSteerFile(const std::string &name, bool addToList);

  // data members
  const AlignmentParameterStore *myParameterStore; /// not the owner!
  const PedeLabelerBase         *myLabels; /// pointer to labeler (not the owner)

  edm::ParameterSet myConfig;
  std::string myDirectory; /// directory of all files
  bool myNoSteerFiles; /// flag to write steering files to /dev/null
  bool myIsSteerFileDebug; /// whether or not to fill pede steering files with debug info
  int myParameterSign; /// old pede versions (before May '07) need a sign flip...
  double theMinHieraConstrCoeff; /// min absolute value of coefficients in hierarchy constraints
  unsigned int theMinHieraParPerConstr; /// hierarchy constraints with less params are ignored
  unsigned int theConstrPrecision; /// precision for writing constraints to text file
  
  std::vector<std::string> mySteeringFiles; /// keeps track of created 'secondary' steering files

  std::set<const Alignable*> myNoHieraCollection; /// Alignables deselected for hierarchy constr.
  Alignable *theCoordMaster;                      /// master coordinates, must (?) be global frame
  std::vector<Alignable*> theCoordDefiners;      /// Alignables selected to define coordinates
};

#endif
