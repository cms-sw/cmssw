#ifndef PEDESTEERERWEAKMODECONSTRAINTS_H
#define PEDESTEERERWEAKMODECONSTRAINTS_H

/**
 * \class PedeSteererWeakModeConstraints
 *
 * Provides steering of weak mode constraints for Pede according to configuration
 *
 * \author    : Joerg Behr
 * date       : February 2013
 */

#include <list>
#include <vector>
#include <map>
#include <set>
#include <string>
// forward ofstream:
#include <iosfwd>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Alignment/MillePedeAlignmentAlgorithm/src/PedeSteerer.h"

#include <DataFormats/GeometryVector/interface/GlobalPoint.h>
#include <CondFormats/Alignment/interface/Definitions.h>

class Alignable;
class PedeLabelerBase;

/***************************************
****************************************/

//FIXME: move GeometryConstraintConfigData to PedeSteererWeakModeConstraints?
class GeometryConstraintConfigData {
 public:
  GeometryConstraintConfigData(const std::vector<double>& co,
                               const std::string& c,
                               const std::vector<std::pair<Alignable*,std::string> >& alisFile,
                               const int sd,
                               const std::vector<Alignable*>& ex,
                               const int instance,
			       const bool downToLowestLevel
                               );
  const std::vector<double> coefficients_;
  const std::string constraintName_;
  const std::vector<std::pair<Alignable*, std::string> > levelsFilenames_;
  const std::vector<Alignable*> excludedAlignables_;
  std::map<std::string, std::ofstream*> mapFileName_;
  std::list<std::pair<Alignable*, std::list<Alignable*> > > HLSsubdets_; //first pointer to HLS object, second list is the list of pointers to the lowest components
  const int sysdeformation_;
  const int instance_;
  const bool downToLowestLevel_;
};

class PedeSteererWeakModeConstraints {
 public:
  ~PedeSteererWeakModeConstraints();
  PedeSteererWeakModeConstraints(AlignableTracker *aliTracker,
                                 const PedeLabelerBase *labels,
                                 const std::vector<edm::ParameterSet> &config,
                                 std::string sf
                                 );

  //FIXME: split the code of the method into smaller pieces/submethods
  // Main method that configures everything and calculates also the constraints
  unsigned int constructConstraints(const std::vector<Alignable*> &alis);

  // Returns a references to the container in which the configuration is stored
  std::list<GeometryConstraintConfigData>& getConfigData() { return ConstraintsConfigContainer_; }

 private:
  // Method creates the data structures with the full configuration
  unsigned int createAlignablesDataStructure();

  // Write the calculated constraints to the output files
  void writeOutput(const std::list<std::pair<unsigned int,double> > &output,
                   const GeometryConstraintConfigData &it, const Alignable* iHLS, double sum_xi_x0);

  // find the out file stream for a given constraint and high-level structure
  std::ofstream* getFile(const GeometryConstraintConfigData &it, const Alignable* iHLS) const;

  // Close the output files
  void closeOutputfiles();

  // Checks whether lowleveldet is a daugther of HLS
  bool checkMother(const Alignable * const lowleveldet, const Alignable * const HLS) const;

  std::pair<align::GlobalPoint, align::GlobalPoint> getDoubleSensorPosition(const Alignable *ali) const;

  double getPhase(const std::vector<double> &coefficients) const;

  // The function for the geometry deformation is defined as f(x).
  // The methods returns x depending on the type of deformation
  double getX(const int sysdeformation, const align::GlobalPoint &pos, const double phase) const;

  double getX0(const std::pair<Alignable*, std::list<Alignable*> > &iHLS,
               const GeometryConstraintConfigData &it) const;

  // Calculates and returns the coefficient for alignment parameter iParameter
  // for an alignable at position pos.
  double getCoefficient(const int sysdeformation,
                        const align::GlobalPoint &pos,
                        const GlobalPoint gUDirection,
                        const GlobalPoint gVDirection,
                        const GlobalPoint gWDirection,
                        const int iParameter, const double &x0,
                        const std::vector<double> &constraintparameters) const;

  //returns true if iParameter of Alignable is selected in configuration file
  bool checkSelectionShiftParameter(const Alignable *ali, unsigned int iParameter) const;

  // Method used to test the provided configuration for unknown parameters
  void verifyParameterNames(const edm::ParameterSet &pset, unsigned int psetnr) const;

  // Method which creates the associative map between levels and coefficient file names
  const std::vector<std::pair<Alignable*, std::string> > makeLevelsFilenames(
                                                                             std::set<std::string> &steerFilePrefixContainer,
                                                                             const std::vector<Alignable*> &alis,
                                                                             const std::string &steerFilePrefix
                                                                             ) const;

  // Verify that the name of the configured deformation is known and that the number of coefficients has been correctly configured
  int verifyDeformationName(const std::string &name, const std::vector<double> &coefficients) const;

  //list of dead modules which are not used in any constraint
  std::list<align::ID> deadmodules_;

  //the data structure that holds all needed informations for the constraint configuration
  std::list<GeometryConstraintConfigData> ConstraintsConfigContainer_;

  const PedeLabelerBase *myLabels_; //PedeLabeler needed to get for the alignables the corresponding Pede label

  const std::vector<edm::ParameterSet> myConfig_; //the VPSet with the configurations for all constraints

  const std::string steerFile_; // the name of the PedeSteerer steering file

  enum SystematicDeformations {
    kUnknown = 0,
    kTwist,
    kZexpansion,
    kSagitta,
    kRadial,
    kTelescope,
    kLayerRotation,
    kElliptical,
    kBowing,
    kSkew
  };
};

#endif
