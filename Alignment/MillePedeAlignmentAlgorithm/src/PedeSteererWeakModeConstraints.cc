/**
 * \file PedeSteererWeakModeConstraints.cc
 *
 *  \author    : Joerg Behr
 *  date       : February 2013
 */

#include "Alignment/MillePedeAlignmentAlgorithm/src/PedeSteererWeakModeConstraints.h"
#include "Alignment/MillePedeAlignmentAlgorithm/src/PedeSteerer.h"

#include "Alignment/MillePedeAlignmentAlgorithm/interface/PedeLabelerBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterSelector.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/SelectionUserVariables.h"
#include "Alignment/CommonAlignmentParametrization/interface/AlignmentParametersFactory.h"
#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"
#include "Alignment/CommonAlignmentParametrization/interface/BowedSurfaceAlignmentDerivatives.h"
#include "Alignment/CommonAlignmentParametrization/interface/TwoBowedSurfacesAlignmentParameters.h"

#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"
// for 'type identification' as Alignable
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Alignment/CommonAlignment/interface/AlignableExtras.h"
// GF doubts the need of these includes from include checker campaign:
#include <FWCore/Framework/interface/EventSetup.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <Geometry/CommonDetUnit/interface/GeomDetType.h>
#include <DataFormats/GeometrySurface/interface/LocalError.h>
#include <Geometry/DTGeometry/interface/DTLayer.h>
// end of doubt

#include <DataFormats/GeometryVector/interface/GlobalPoint.h>

#include <fstream>
#include <sstream>
#include <algorithm>

// from ROOT
#include <TSystem.h>
#include <TMath.h>

#include <iostream>

GeometryConstraintConfigData::GeometryConstraintConfigData(
    const std::vector<double>& co,
    const std::string& c,
    const std::vector<std::pair<Alignable*, std::string> >& alisFile,
    const int sd,
    const align::Alignables& ex,
    const int instance,
    const bool downToLowestLevel)
    : coefficients_(co),
      constraintName_(c),
      levelsFilenames_(alisFile),
      excludedAlignables_(ex),
      sysdeformation_(sd),
      instance_(instance),
      downToLowestLevel_(downToLowestLevel) {}

//_________________________________________________________________________
PedeSteererWeakModeConstraints::PedeSteererWeakModeConstraints(AlignableTracker* aliTracker,
                                                               const PedeLabelerBase* labels,
                                                               const std::vector<edm::ParameterSet>& config,
                                                               std::string sf)
    : myLabels_(labels),
      myConfig_(config),
      steerFile_(sf),
      alignableObjectId_{AlignableObjectId::commonObjectIdProvider(aliTracker, nullptr)} {
  unsigned int psetnr = 0;
  std::set<std::string> steerFilePrefixContainer;
  for (const auto& pset : myConfig_) {
    this->verifyParameterNames(pset, psetnr);
    psetnr++;

    const auto coefficients = pset.getParameter<std::vector<double> >("coefficients");
    const auto dm = pset.exists("deadmodules") ? pset.getParameter<std::vector<unsigned int> >("deadmodules")
                                               : std::vector<unsigned int>();
    std::string name = pset.getParameter<std::string>("constraint");
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);
    const auto ignoredInstances = pset.exists("ignoredInstances")
                                      ? pset.getUntrackedParameter<std::vector<unsigned int> >("ignoredInstances")
                                      : std::vector<unsigned int>();

    const auto downToLowestLevel =
        pset.exists("downToLowestLevel") ? pset.getUntrackedParameter<bool>("downToLowestLevel") : false;

    AlignmentParameterSelector selector(aliTracker, nullptr, nullptr);
    selector.clear();
    selector.addSelections(pset.getParameter<edm::ParameterSet>("levels"));

    const auto& alis = selector.selectedAlignables();

    AlignmentParameterSelector selector_excludedalignables(aliTracker, nullptr, nullptr);
    selector_excludedalignables.clear();
    if (pset.exists("excludedAlignables")) {
      selector_excludedalignables.addSelections(pset.getParameter<edm::ParameterSet>("excludedAlignables"));
    }
    const auto& excluded_alis = selector_excludedalignables.selectedAlignables();

    //check that the name of the deformation is known and that the number
    //of provided parameter is right.
    auto sysdeformation = this->verifyDeformationName(name, coefficients);

    if (deadmodules_.empty()) {  //fill the list of dead modules only once
      edm::LogInfo("Alignment") << "@SUB=PedeSteererWeakModeConstraints"
                                << "Load list of dead modules (size = " << dm.size() << ").";
      for (const auto& it : dm)
        deadmodules_.push_back(it);
    }

    // loop over all IOVs/momentum ranges
    for (unsigned int instance = 0; instance < myLabels_->maxNumberOfParameterInstances(); ++instance) {
      // check if this IOV/momentum range is to be ignored:
      if (std::find(ignoredInstances.begin(), ignoredInstances.end(), instance) != ignoredInstances.end()) {
        continue;
      }
      std::stringstream defaultsteerfileprefix;
      defaultsteerfileprefix << "autosteerFilePrefix_" << name << "_" << psetnr << "_" << instance;

      const auto steerFilePrefix = pset.exists("steerFilePrefix") ? pset.getParameter<std::string>("steerFilePrefix") +
                                                                        "_" + std::to_string(instance)
                                                                  : defaultsteerfileprefix.str();

      const auto levelsFilenames = this->makeLevelsFilenames(steerFilePrefixContainer, alis, steerFilePrefix);

      //Add the configuration data for this constraint to the container of config data
      ConstraintsConfigContainer_.emplace_back(GeometryConstraintConfigData(
          coefficients, name, levelsFilenames, sysdeformation, excluded_alis, instance, downToLowestLevel));
    }
  }
}

//_________________________________________________________________________
std::pair<align::GlobalPoint, align::GlobalPoint> PedeSteererWeakModeConstraints::getDoubleSensorPosition(
    const Alignable* ali) const {
  const auto aliPar = dynamic_cast<TwoBowedSurfacesAlignmentParameters*>(ali->alignmentParameters());
  if (aliPar) {
    const auto ySplit = aliPar->ySplit();
    const auto halfLength = 0.5 * ali->surface().length();
    const auto yM1 = 0.5 * (ySplit - halfLength);  // y_mean of surface 1
    const auto yM2 = yM1 + halfLength;             // y_mean of surface 2
    const auto pos_sensor0(ali->surface().toGlobal(align::LocalPoint(0., yM1, 0.)));
    const auto pos_sensor1(ali->surface().toGlobal(align::LocalPoint(0., yM2, 0.)));
    return std::make_pair(pos_sensor0, pos_sensor1);
  } else {
    throw cms::Exception("Alignment") << "[PedeSteererWeakModeConstraints::getDoubleSensorPosition]"
                                      << " Dynamic cast to double sensor parameters failed.";
    return std::make_pair(align::GlobalPoint(0.0, 0.0, 0.0), align::GlobalPoint(0.0, 0.0, 0.0));
  }
}

//_________________________________________________________________________
unsigned int PedeSteererWeakModeConstraints::createAlignablesDataStructure() {
  unsigned int nConstraints = 0;
  for (auto& iC : ConstraintsConfigContainer_) {
    //loop over all HLS for which the constraint is to be determined
    for (const auto& iHLS : iC.levelsFilenames_) {
      //determine next active sub-alignables for iHLS
      align::Alignables aliDaughts;
      if (iC.downToLowestLevel_) {
        if (!iHLS.first->lastCompsWithParams(aliDaughts)) {
          edm::LogWarning("Alignment") << "@SUB=PedeSteererWeakModeConstraints::createAlignablesDataStructure"
                                       << "Some but not all component branches "
                                       << alignableObjectId_.idToString(iHLS.first->alignableObjectId())
                                       << " with params!";
        }
      } else {
        if (!iHLS.first->firstCompsWithParams(aliDaughts)) {
          edm::LogWarning("Alignment") << "@SUB=PedeSteererWeakModeConstraints::createAlignablesDataStructure"
                                       << "Some but not all daughters of "
                                       << alignableObjectId_.idToString(iHLS.first->alignableObjectId())
                                       << " with params!";
        }
      }
      ++nConstraints;

      std::list<Alignable*> usedinconstraint;
      for (const auto& iD : aliDaughts) {
        bool isNOTdead = true;
        for (const auto& iDeadmodules : deadmodules_) {
          if ((iD->alignableObjectId() == align::AlignableDetUnit || iD->alignableObjectId() == align::AlignableDet) &&
              iD->geomDetId().rawId() == iDeadmodules) {
            isNOTdead = false;
            break;
          }
        }
        //check if the module is excluded
        for (const auto& iEx : iC.excludedAlignables_) {
          if (iD->id() == iEx->id() && iD->alignableObjectId() == iEx->alignableObjectId()) {
            //if(iD->geomDetId().rawId() == (*iEx)->geomDetId().rawId()) {
            isNOTdead = false;
            break;
          }
        }
        const bool issubcomponent = this->checkMother(iD, iHLS.first);
        if (issubcomponent) {
          if (isNOTdead) {
            usedinconstraint.push_back(iD);
          }
        } else {
          //sanity check
          throw cms::Exception("Alignment") << "[PedeSteererWeakModeConstraints::createAlignablesDataStructure]"
                                            << " Sanity check failed. Alignable defined as active sub-component, "
                                            << " but in fact its not a daugther of "
                                            << alignableObjectId_.idToString(iHLS.first->alignableObjectId());
        }
      }

      if (!usedinconstraint.empty()) {
        iC.HLSsubdets_.push_back(std::make_pair(iHLS.first, usedinconstraint));
      } else {
        edm::LogInfo("Alignment") << "@SUB=PedeSteererWeakModeConstraints"
                                  << "No sub-components for "
                                  << alignableObjectId_.idToString(iHLS.first->alignableObjectId()) << " at ("
                                  << iHLS.first->globalPosition().x() << "," << iHLS.first->globalPosition().y() << ","
                                  << iHLS.first->globalPosition().z() << ") selected. Skip constraint";
      }
      if (aliDaughts.empty()) {
        edm::LogWarning("Alignment") << "@SUB=PedeSteererWeakModeConstraints::createAlignablesDataStructure"
                                     << "No active sub-alignables found for "
                                     << alignableObjectId_.idToString(iHLS.first->alignableObjectId()) << " at ("
                                     << iHLS.first->globalPosition().x() << "," << iHLS.first->globalPosition().y()
                                     << "," << iHLS.first->globalPosition().z() << ").";
      }
    }
  }
  return nConstraints;
}

//_________________________________________________________________________
double PedeSteererWeakModeConstraints::getX(const int sysdeformation,
                                            const align::GlobalPoint& pos,
                                            const double phase) const {
  double x = 0.0;

  const double r = TMath::Sqrt(pos.x() * pos.x() + pos.y() * pos.y());

  switch (sysdeformation) {
    case SystematicDeformations::kTwist:
    case SystematicDeformations::kZexpansion:
      x = pos.z();
      break;
    case SystematicDeformations::kSagitta:
    case SystematicDeformations::kRadial:
    case SystematicDeformations::kTelescope:
    case SystematicDeformations::kLayerRotation:
      x = r;
      break;
    case SystematicDeformations::kBowing:
      x = pos.z() * pos.z();  //TMath::Abs(pos.z());
      break;
    case SystematicDeformations::kElliptical:
      x = r * TMath::Cos(2.0 * pos.phi() + phase);
      break;
    case SystematicDeformations::kSkew:
      x = TMath::Cos(pos.phi() + phase);
      break;
  };

  return x;
}

//_________________________________________________________________________
double PedeSteererWeakModeConstraints::getCoefficient(const int sysdeformation,
                                                      const align::GlobalPoint& pos,
                                                      const GlobalPoint gUDirection,
                                                      const GlobalPoint gVDirection,
                                                      const GlobalPoint gWDirection,
                                                      const int iParameter,
                                                      const double& x0,
                                                      const std::vector<double>& constraintparameters) const {
  if (iParameter < 0 || iParameter > 2) {
    throw cms::Exception("Alignment") << "[PedeSteererWeakModeConstraints::getCoefficient]"
                                      << " iParameter has to be in the range [0,2] but"
                                      << " it is equal to " << iParameter << ".";
  }

  //global vectors pointing in u,v,w direction
  const std::vector<double> vec_u = {pos.x() - gUDirection.x(), pos.y() - gUDirection.y(), pos.z() - gUDirection.z()};
  const std::vector<double> vec_v = {pos.x() - gVDirection.x(), pos.y() - gVDirection.y(), pos.z() - gVDirection.z()};
  const std::vector<double> vec_w = {pos.x() - gWDirection.x(), pos.y() - gWDirection.y(), pos.z() - gWDirection.z()};

  //FIXME: how to make inner vectors const?
  const std::vector<std::vector<double> > global_vecs = {vec_u, vec_v, vec_w};

  const double n = TMath::Sqrt(global_vecs.at(iParameter).at(0) * global_vecs.at(iParameter).at(0) +
                               global_vecs.at(iParameter).at(1) * global_vecs.at(iParameter).at(1) +
                               global_vecs.at(iParameter).at(2) * global_vecs.at(iParameter).at(2));
  const double r = TMath::Sqrt(pos.x() * pos.x() + pos.y() * pos.y());

  const double phase = this->getPhase(constraintparameters);
  //const double radial_direction[3] = {TMath::Sin(phase), TMath::Cos(phase), 0.0};
  const std::vector<double> radial_direction = {TMath::Sin(phase), TMath::Cos(phase), 0.0};
  //is equal to unity by construction ...
  const double norm_radial_direction =
      TMath::Sqrt(radial_direction.at(0) * radial_direction.at(0) + radial_direction.at(1) * radial_direction.at(1) +
                  radial_direction.at(2) * radial_direction.at(2));

  //const double phi_direction[3] = { -1.0 * pos.y(), pos.x(), 0.0};
  const std::vector<double> phi_direction = {-1.0 * pos.y(), pos.x(), 0.0};
  const double norm_phi_direction =
      TMath::Sqrt(phi_direction.at(0) * phi_direction.at(0) + phi_direction.at(1) * phi_direction.at(1) +
                  phi_direction.at(2) * phi_direction.at(2));

  //const double z_direction[3] = {0.0, 0.0, 1.0};
  static const std::vector<double> z_direction = {0.0, 0.0, 1.0};
  const double norm_z_direction =
      TMath::Sqrt(z_direction.at(0) * z_direction.at(0) + z_direction.at(1) * z_direction.at(1) +
                  z_direction.at(2) * z_direction.at(2));

  //unit vector pointing from the origin to the module position in the transverse plane
  const std::vector<double> rDirection = {pos.x(), pos.y(), 0.0};
  const double norm_rDirection = TMath::Sqrt(rDirection.at(0) * rDirection.at(0) + rDirection.at(1) * rDirection.at(1) +
                                             rDirection.at(2) * rDirection.at(2));

  double coeff = 0.0;
  double dot_product = 0.0;
  double normalisation_factor = 1.0;

  //see https://indico.cern.ch/getFile.py/access?contribId=15&sessionId=1&resId=0&materialId=slides&confId=127126
  switch (sysdeformation) {
    case SystematicDeformations::kTwist:
    case SystematicDeformations::kLayerRotation:
      dot_product = phi_direction.at(0) * global_vecs.at(iParameter).at(0) +
                    phi_direction.at(1) * global_vecs.at(iParameter).at(1) +
                    phi_direction.at(2) * global_vecs.at(iParameter).at(2);
      normalisation_factor = r * n * norm_phi_direction;
      break;
    case SystematicDeformations::kZexpansion:
    case SystematicDeformations::kTelescope:
    case SystematicDeformations::kSkew:
      dot_product = global_vecs.at(iParameter).at(0) * z_direction.at(0) +
                    global_vecs.at(iParameter).at(1) * z_direction.at(1) +
                    global_vecs.at(iParameter).at(2) * z_direction.at(2);
      normalisation_factor = (n * norm_z_direction);
      break;
    case SystematicDeformations::kRadial:
    case SystematicDeformations::kBowing:
    case SystematicDeformations::kElliptical:
      dot_product = global_vecs.at(iParameter).at(0) * rDirection.at(0) +
                    global_vecs.at(iParameter).at(1) * rDirection.at(1) +
                    global_vecs.at(iParameter).at(2) * rDirection.at(2);
      normalisation_factor = (n * norm_rDirection);
      break;
    case SystematicDeformations::kSagitta:
      dot_product = global_vecs.at(iParameter).at(0) * radial_direction.at(0) +
                    global_vecs.at(iParameter).at(1) * radial_direction.at(1) +
                    global_vecs.at(iParameter).at(2) * radial_direction.at(2);
      normalisation_factor = (n * norm_radial_direction);
      break;
    default:
      break;
  }

  if (TMath::Abs(normalisation_factor) > 0.0) {
    coeff = dot_product * (this->getX(sysdeformation, pos, phase) - x0) / normalisation_factor;
  } else {
    throw cms::Exception("Alignment") << "[PedeSteererWeakModeConstraints::getCoefficient]"
                                      << " Normalisation factor"
                                      << "for coefficient calculation equal to zero! Misconfiguration?";
  }
  return coeff;
}

//_________________________________________________________________________
bool PedeSteererWeakModeConstraints::checkSelectionShiftParameter(const Alignable* ali, unsigned int iParameter) const {
  bool isselected = false;
  const std::vector<bool>& aliSel = ali->alignmentParameters()->selector();
  //exclude non-shift parameters
  if ((iParameter <= 2) || (iParameter >= 9 && iParameter <= 11)) {
    if (!aliSel.at(iParameter)) {
      isselected = false;
    } else {
      auto params = ali->alignmentParameters();
      auto selVar = dynamic_cast<SelectionUserVariables*>(params->userVariables());
      if (selVar) {
        if (selVar->fullSelection().size() <= (iParameter + 1)) {
          throw cms::Exception("Alignment")
              << "[PedeSteererWeakModeConstraints::checkSelectionShiftParameter]"
              << " Can not access selected alignment variables of alignable "
              << alignableObjectId_.idToString(ali->alignableObjectId()) << "at (" << ali->globalPosition().x() << ","
              << ali->globalPosition().y() << "," << ali->globalPosition().z() << ") "
              << "for parameter number " << (iParameter + 1) << ".";
        }
      }
      const char selChar = (selVar ? selVar->fullSelection().at(iParameter) : '1');
      // if(selChar == '1') { //FIXME??? what about 'r'?
      if (selChar == '1' || selChar == 'r') {
        isselected = true;
      } else {
        isselected = false;
      }
    }
  }
  return isselected;
}

//_________________________________________________________________________
void PedeSteererWeakModeConstraints::closeOutputfiles() {
  //'delete' output files which means: close them
  for (auto& it : ConstraintsConfigContainer_) {
    for (auto& iFile : it.mapFileName_) {
      if (iFile.second) {
        delete iFile.second;
        iFile.second = nullptr;
      } else {
        throw cms::Exception("FileCloseProblem") << "[PedeSteererWeakModeConstraints]"
                                                 << " can not close file " << iFile.first << ".";
      }
    }
  }
}

//_________________________________________________________________________
void PedeSteererWeakModeConstraints::writeOutput(const std::list<std::pair<unsigned int, double> >& output,
                                                 const GeometryConstraintConfigData& it,
                                                 const Alignable* iHLS,
                                                 double sum_xi_x0) {
  std::ofstream* ofile = getFile(it, iHLS);

  if (ofile == nullptr) {
    throw cms::Exception("FileFindError") << "[PedeSteererWeakModeConstraints] Cannot find output file.";
  } else {
    if (!output.empty()) {
      const double constr = sum_xi_x0 * it.coefficients_.front();
      (*ofile) << "Constraint " << std::scientific << constr << std::endl;
      for (const auto& ioutput : output) {
        (*ofile) << std::fixed << ioutput.first << " " << std::scientific << ioutput.second << std::endl;
      }
    }
  }
}

//_________________________________________________________________________
std::ofstream* PedeSteererWeakModeConstraints::getFile(const GeometryConstraintConfigData& it,
                                                       const Alignable* iHLS) const {
  std::ofstream* file = nullptr;

  for (const auto& ilevelsFilename : it.levelsFilenames_) {
    if (ilevelsFilename.first->id() == iHLS->id() &&
        ilevelsFilename.first->alignableObjectId() == iHLS->alignableObjectId()) {
      const auto iFile = it.mapFileName_.find(ilevelsFilename.second);
      if (iFile != it.mapFileName_.end()) {
        file = iFile->second;
      }
    }
  }

  return file;
}

//_________________________________________________________________________
double PedeSteererWeakModeConstraints::getX0(const std::pair<Alignable*, std::list<Alignable*> >& iHLS,
                                             const GeometryConstraintConfigData& it) const {
  double nmodules = 0.0;
  double x0 = 0.0;

  for (const auto& ali : iHLS.second) {
    align::PositionType pos = ali->globalPosition();
    bool alignableIsFloating = false;  //means: true=alignable is able to move in at least one direction

    //test whether at least one variable has been selected in the configuration
    for (unsigned int iParameter = 0; static_cast<int>(iParameter) < ali->alignmentParameters()->size(); iParameter++) {
      if (this->checkSelectionShiftParameter(ali, iParameter)) {
        alignableIsFloating = true;
        // //verify that alignable has just one label -- meaning no IOV-dependence etc
        // const unsigned int nInstances = myLabels_->numberOfParameterInstances(ali, iParameter);
        // if(nInstances > 1) {
        //   throw cms::Exception("PedeSteererWeakModeConstraints")
        //     << "@SUB=PedeSteererWeakModeConstraints::ConstructConstraints"
        //     << " Weak mode constraints are only supported for alignables which have"
        //     << " just one label. However, e.g. alignable"
        //     << " " << alignableObjectId_.idToString(ali->alignableObjectId())
        //     << "at (" << ali->globalPosition().x() << ","<< ali->globalPosition().y() << "," << ali->globalPosition().z()<< "), "
        //     << " was configured to have >1 label. Remove e.g. IOV-dependence for this (and other) alignables which are used in the constraint.";
        // }
        break;
      }
    }
    //at least one parameter of the alignable can be changed in the alignment
    if (alignableIsFloating) {
      const auto phase = this->getPhase(it.coefficients_);
      if (ali->alignmentParameters()->type() != AlignmentParametersFactory::kTwoBowedSurfaces) {
        x0 += this->getX(it.sysdeformation_, pos, phase);
        nmodules++;
      } else {
        std::pair<align::GlobalPoint, align::GlobalPoint> sensorpositions = this->getDoubleSensorPosition(ali);
        x0 += this->getX(it.sysdeformation_, sensorpositions.first, phase) +
              this->getX(it.sysdeformation_, sensorpositions.second, phase);
        nmodules++;
        nmodules++;
      }
    }
  }
  if (nmodules > 0) {
    x0 = x0 / nmodules;
  } else {
    throw cms::Exception("Alignment") << "@SUB=PedeSteererWeakModeConstraints::ConstructConstraints"
                                      << " Number of selected modules equal to zero. Check configuration!";
    x0 = 1.0;
  }
  return x0;
}

//_________________________________________________________________________
unsigned int PedeSteererWeakModeConstraints::constructConstraints(const align::Alignables& alis) {
  //FIXME: split the code of the method into smaller pieces/submethods

  //create the data structures that store the alignables
  //for which the constraints need to be calculated and
  //their association to high-level structures
  const auto nConstraints = this->createAlignablesDataStructure();

  std::vector<std::list<std::pair<unsigned int, double> > > createdConstraints;

  //calculate constraints
  //loop over all constraints
  for (const auto& it : ConstraintsConfigContainer_) {
    //loop over all subdets for which constraints are determined
    for (const auto& iHLS : it.HLSsubdets_) {
      double sum_xi_x0 = 0.0;
      std::list<std::pair<unsigned int, double> > output;

      const double x0 = this->getX0(iHLS, it);

      for (std::list<Alignable*>::const_iterator iAlignables = iHLS.second.begin(); iAlignables != iHLS.second.end();
           iAlignables++) {
        const Alignable* ali = (*iAlignables);
        const auto aliLabel =
            myLabels_->alignableLabelFromParamAndInstance(const_cast<Alignable*>(ali), 0, it.instance_);
        const AlignableSurface& surface = ali->surface();

        const LocalPoint lUDirection(1., 0., 0.), lVDirection(0., 1., 0.), lWDirection(0., 0., 1.);

        GlobalPoint gUDirection = surface.toGlobal(lUDirection), gVDirection = surface.toGlobal(lVDirection),
                    gWDirection = surface.toGlobal(lWDirection);

        const bool isDoubleSensor = ali->alignmentParameters()->type() == AlignmentParametersFactory::kTwoBowedSurfaces;

        const auto sensorpositions = isDoubleSensor ? this->getDoubleSensorPosition(ali)
                                                    : std::make_pair(ali->globalPosition(), align::PositionType());

        const auto& pos_sensor0 = sensorpositions.first;
        const auto& pos_sensor1 = sensorpositions.second;
        const auto phase = this->getPhase(it.coefficients_);
        const auto x_sensor0 = this->getX(it.sysdeformation_, pos_sensor0, phase);
        const auto x_sensor1 = isDoubleSensor ? this->getX(it.sysdeformation_, pos_sensor1, phase) : 0.0;

        sum_xi_x0 += (x_sensor0 - x0) * (x_sensor0 - x0);
        if (isDoubleSensor) {
          sum_xi_x0 += (x_sensor1 - x0) * (x_sensor1 - x0);
        }
        const int numparameterlimit = ali->alignmentParameters()->size();  //isDoubleSensor ? 18 : 3;

        for (int iParameter = 0; iParameter < numparameterlimit; iParameter++) {
          int localindex = 0;
          if (iParameter == 0 || iParameter == 9)
            localindex = 0;
          if (iParameter == 1 || iParameter == 10)
            localindex = 1;
          if (iParameter == 2 || iParameter == 11)
            localindex = 2;

          if ((iParameter >= 0 && iParameter <= 2) || (iParameter >= 9 && iParameter <= 11)) {
          } else {
            continue;
          }
          if (!this->checkSelectionShiftParameter(ali, iParameter)) {
            continue;
          }
          //do it for each 'instance' separately? -> IOV-dependence, no
          const auto paramLabel = myLabels_->parameterLabel(aliLabel, iParameter);

          const auto& pos = (iParameter <= 2) ? pos_sensor0 : pos_sensor1;
          //select only u,v,w
          if (iParameter == 0 || iParameter == 1 || iParameter == 2 || iParameter == 9 || iParameter == 10 ||
              iParameter == 11) {
            const double coeff = this->getCoefficient(
                it.sysdeformation_, pos, gUDirection, gVDirection, gWDirection, localindex, x0, it.coefficients_);
            if (TMath::Abs(coeff) > 0.0) {
              //nothing
            } else {
              edm::LogWarning("PedeSteererWeakModeConstraints")
                  << "@SUB=PedeSteererWeakModeConstraints::getCoefficient"
                  << "Coefficient of alignable " << alignableObjectId_.idToString(ali->alignableObjectId()) << " at ("
                  << ali->globalPosition().x() << "," << ali->globalPosition().y() << "," << ali->globalPosition().z()
                  << ") "
                  << " in subdet " << alignableObjectId_.idToString(iHLS.first->alignableObjectId())
                  << " for parameter " << localindex << " equal to zero. This alignable is used in the constraint"
                  << " '" << it.constraintName_
                  << "'. The id is: alignable->geomDetId().rawId() = " << ali->geomDetId().rawId() << ".";
            }
            output.push_back(std::make_pair(paramLabel, coeff));
          }
        }
      }

      if (std::find(createdConstraints.begin(), createdConstraints.end(), output) != createdConstraints.end()) {
        // check if linearly dependent constraint exists already:
        auto outFile = getFile(it, iHLS.first);
        if (outFile == nullptr) {
          throw cms::Exception("FileFindError") << "[PedeSteererWeakModeConstraints] Cannot find output file.";
        } else {
          *outFile << "! The constraint for this IOV/momentum range" << std::endl
                   << "! has been removed because the used parameters" << std::endl
                   << "! are not IOV or momentum-range dependent." << std::endl;
        }
        continue;
      }
      this->writeOutput(output, it, iHLS.first, sum_xi_x0);
      createdConstraints.push_back(output);
    }
  }
  this->closeOutputfiles();

  return nConstraints;
}

//_________________________________________________________________________
bool PedeSteererWeakModeConstraints::checkMother(const Alignable* const lowleveldet, const Alignable* const HLS) const {
  if (lowleveldet->id() == HLS->id() && lowleveldet->alignableObjectId() == HLS->alignableObjectId()) {
    return true;
  } else {
    if (lowleveldet->mother() == nullptr)
      return false;
    else
      return this->checkMother(lowleveldet->mother(), HLS);
  }
}

//_________________________________________________________________________
void PedeSteererWeakModeConstraints::verifyParameterNames(const edm::ParameterSet& pset, unsigned int psetnr) const {
  const auto parameterNames = pset.getParameterNames();
  for (const auto& name : parameterNames) {
    if (name != "coefficients" && name != "deadmodules" && name != "constraint" && name != "steerFilePrefix" &&
        name != "levels" && name != "excludedAlignables" && name != "ignoredInstances" && name != "downToLowestLevel") {
      throw cms::Exception("BadConfig") << "@SUB=PedeSteererWeakModeConstraints::verifyParameterNames:"
                                        << " Unknown parameter name '" << name << "' in PSet number " << psetnr
                                        << ". Maybe a typo?";
    }
  }
}

//_________________________________________________________________________
const std::vector<std::pair<Alignable*, std::string> > PedeSteererWeakModeConstraints::makeLevelsFilenames(
    std::set<std::string>& steerFilePrefixContainer,
    const align::Alignables& alis,
    const std::string& steerFilePrefix) const {
  //check whether the prefix is unique
  if (steerFilePrefixContainer.find(steerFilePrefix) != steerFilePrefixContainer.end()) {
    throw cms::Exception("BadConfig") << "[PedeSteererWeakModeConstraints] Steering file"
                                      << " prefix '" << steerFilePrefix << "' already exists. Specify unique names!";
  } else {
    steerFilePrefixContainer.insert(steerFilePrefix);
  }

  std::vector<std::pair<Alignable*, std::string> > levelsFilenames;
  for (const auto& ali : alis) {
    std::stringstream n;
    n << steerFile_ << "_" << steerFilePrefix  //<< "_" << name
      << "_" << alignableObjectId_.idToString(ali->alignableObjectId()) << "_" << ali->id() << "_"
      << ali->alignableObjectId() << ".txt";

    levelsFilenames.push_back(std::make_pair(ali, n.str()));
  }
  return levelsFilenames;
}

//_________________________________________________________________________
int PedeSteererWeakModeConstraints::verifyDeformationName(const std::string& name,
                                                          const std::vector<double>& coefficients) const {
  int sysdeformation = SystematicDeformations::kUnknown;

  if (name == "twist") {
    sysdeformation = SystematicDeformations::kTwist;
  } else if (name == "zexpansion") {
    sysdeformation = SystematicDeformations::kZexpansion;
  } else if (name == "sagitta") {
    sysdeformation = SystematicDeformations::kSagitta;
  } else if (name == "radial") {
    sysdeformation = SystematicDeformations::kRadial;
  } else if (name == "telescope") {
    sysdeformation = SystematicDeformations::kTelescope;
  } else if (name == "layerrotation") {
    sysdeformation = SystematicDeformations::kLayerRotation;
  } else if (name == "bowing") {
    sysdeformation = SystematicDeformations::kBowing;
  } else if (name == "skew") {
    sysdeformation = SystematicDeformations::kSkew;
  } else if (name == "elliptical") {
    sysdeformation = SystematicDeformations::kElliptical;
  }

  if (sysdeformation == SystematicDeformations::kUnknown) {
    throw cms::Exception("BadConfig") << "[PedeSteererWeakModeConstraints]"
                                      << " specified configuration option '" << name << "' not known.";
  }
  if ((sysdeformation == SystematicDeformations::kSagitta || sysdeformation == SystematicDeformations::kElliptical ||
       sysdeformation == SystematicDeformations::kSkew) &&
      coefficients.size() != 2) {
    throw cms::Exception("BadConfig") << "[PedeSteererWeakModeConstraints]"
                                      << " Excactly two parameters using the coefficient"
                                      << " variable have to be provided for the " << name << " constraint.";
  }
  if ((sysdeformation == SystematicDeformations::kTwist || sysdeformation == SystematicDeformations::kZexpansion ||
       sysdeformation == SystematicDeformations::kTelescope ||
       sysdeformation == SystematicDeformations::kLayerRotation || sysdeformation == SystematicDeformations::kRadial ||
       sysdeformation == SystematicDeformations::kBowing) &&
      coefficients.size() != 1) {
    throw cms::Exception("BadConfig") << "[PedeSteererWeakModeConstraints]"
                                      << " Excactly ONE parameter using the coefficient"
                                      << " variable have to be provided for the " << name << " constraint.";
  }

  if (coefficients.empty()) {
    throw cms::Exception("BadConfig") << "[PedeSteererWeakModeConstraints]"
                                      << " At least one coefficient has to be specified.";
  }
  return sysdeformation;
}

//_________________________________________________________________________
double PedeSteererWeakModeConstraints::getPhase(const std::vector<double>& coefficients) const {
  return coefficients.size() == 2 ? coefficients.at(1) : 0.0;  //treat second parameter as phase otherwise return 0
}

//_________________________________________________________________________
PedeSteererWeakModeConstraints::~PedeSteererWeakModeConstraints() = default;
