#ifndef PEDESTEERERWEAKMODECONSTRAINTS_H
#define PEDESTEERERWEAKMODECONSTRAINTS_H

/**
 * \class PedeSteererWeakModeConstraints
 *
 * Provides steering of weak mode constraints for Pede according to configuration
 *
 * \author    : Joerg Behr
 * date       : February 2013
 * $Date:  $
 * $Revision:  $
 * (last update by $Author: jbehr $)
 */

#include <list>
#include <vector>
#include <map> 
#include <set> 
#include <sstream>
#include <string>
// forward ofstream:
#include <iosfwd> 

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PedeSteerer.h"
#include "boost/shared_ptr.hpp"

#include <DataFormats/GeometryVector/interface/GlobalPoint.h>
#include <CondFormats/Alignment/interface/Definitions.h>

class Alignable;
class AlignableTracker;
class AlignableMuon;
class AlignableExtras;
class AlignmentParameterStore;
class PedeLabelerBase;

/***************************************
****************************************/
class GeometryConstraintConfigData {
 public:
  GeometryConstraintConfigData(const std::vector<double> co,
                               const std::string c,
                               const std::vector<std::pair<Alignable*,std::string> > &alisFile,
                               const int sd,
                               const std::vector<Alignable*> ex
                               );
  const std::vector<double> coefficients;
  const std::string constraintName;
  const std::vector<std::pair<Alignable*, std::string> > levelsFilenames;
  const std::vector<Alignable*> excludedAlignables;
  std::map<std::string, std::ofstream*> mapFileName;
  std::list<std::pair<Alignable*, std::list<Alignable*> > > HLSsubdets; //first pointer to HLS object, second list is the list of pointers to the lowest components
  int sysdeformation;
};

class PedeSteererWeakModeConstraints {
 public:
  ~PedeSteererWeakModeConstraints();
  PedeSteererWeakModeConstraints(AlignableTracker *aliTracker,
                                 AlignableMuon *aliMuon,
                                 AlignableExtras *aliExtras,
                                 const PedeLabelerBase *labels,
                                 const std::vector<edm::ParameterSet> &config,
                                 const bool apply,
                                 std::string sf
                                 );
  unsigned int createAlignablesDataStructure();
  unsigned int ConstructConstraints(const std::vector<Alignable*> &alis, PedeSteerer *thePedeSteerer);
  bool checkMother(const Alignable * const lowleveldet, const Alignable * const HLS) const;
  std::pair<align::GlobalPoint, align::GlobalPoint> getDoubleSensorPosition(const Alignable *ali);
  double getX(const int sysdeformation, const align::GlobalPoint &pos);
  double getCoefficient(const int sysdeformation,
                        const align::GlobalPoint &pos,
                        const GlobalPoint gUDirection,
                        const GlobalPoint gVDirection,
                        const GlobalPoint gWDirection,
                        const int iParameter, const double &x0,
                        const std::vector<double> &constraintparameters);
  //returns true if iParameter of Alignable is selected in configuration file
  bool checkSelectionShiftParameter(const Alignable *ali, int iParameter);
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
    
 private:
  std::list<align::ID> deadmodules;
  std::list<GeometryConstraintConfigData> ConstraintsConfigContainer;
  const PedeLabelerBase *myLabels;
  const std::vector<edm::ParameterSet> myConfig;
  const bool applyconstraints;
  const std::string steerFile;
};
 
#endif
