/**
 * \file PedeSteerer.cc
 *
 *  \author    : Gero Flucke
 *  date       : October 2006
 *  $Revision: 1.42 $
 *  $Date: 2013/06/18 15:47:55 $
 *  (last update by $Author: jbehr $)
 */

#include "PedeSteerer.h"
#include "PedeSteererWeakModeConstraints.h"

#include "Alignment/MillePedeAlignmentAlgorithm/interface/PedeLabelerBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include <boost/cstdint.hpp> 
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterSelector.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/SelectionUserVariables.h"
#include "Alignment/CommonAlignmentParametrization/interface/AlignmentParametersFactory.h"
#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"

#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"
// for 'type identification' as Alignable
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Alignment/CommonAlignment/interface/AlignableExtras.h"
// GF doubts the need of these includes from include checker campaign:
#include <FWCore/Framework/interface/EventSetup.h> 
#include <Geometry/CommonDetUnit/interface/GeomDetUnit.h> 
#include <Geometry/CommonDetUnit/interface/GeomDetType.h> 
#include <DataFormats/GeometrySurface/interface/LocalError.h> 
#include <Geometry/DTGeometry/interface/DTLayer.h> 
// end of doubt

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <fstream>
#include <sstream>
#include <algorithm>

// from ROOT
#include <TSystem.h>
#include <TMath.h>

#include <iostream>

//_________________________________________________________________________
PedeSteerer::PedeSteerer(AlignableTracker *aliTracker, AlignableMuon *aliMuon, AlignableExtras *aliExtras,
			 AlignmentParameterStore *store, const PedeLabelerBase *labels,
                         const edm::ParameterSet &config, const std::string &defaultDir,
			 bool noSteerFiles) :
  myParameterStore(store), myLabels(labels), myConfig(config),
  myDirectory(myConfig.getUntrackedParameter<std::string>("fileDir")),
  myNoSteerFiles(noSteerFiles),
  myIsSteerFileDebug(myConfig.getUntrackedParameter<bool>("steerFileDebug")),
  myParameterSign(myConfig.getUntrackedParameter<int>("parameterSign")),
  theMinHieraConstrCoeff(myConfig.getParameter<double>("minHieraConstrCoeff")),
  theMinHieraParPerConstr(myConfig.getParameter<unsigned int>("minHieraParPerConstr")),
  theConstrPrecision(myConfig.getParameter<unsigned int>("constrPrecision")),
  theCoordMaster(0)
{
  if (myParameterSign != 1 && myParameterSign != -1) {
    cms::Exception("BadConfig") << "Expect PedeSteerer.parameterSign = +/-1, "
				<< "found " << myParameterSign << ".";
  }

  // Correct directory, needed before asking for fileName(..):
  if (myDirectory.empty()) myDirectory = defaultDir;
  if (!myDirectory.empty() && myDirectory.find_last_of('/') != myDirectory.size() - 1) {
    myDirectory += '/'; // directory may need '/'
  }

  const std::vector<Alignable*> &alis = myParameterStore->alignables();
  if (!this->checkParameterChoices(alis)) {} // anyway thrown exception

  // Coordinate system selection and correction before everything
  theCoordDefiners = this->selectCoordinateAlis(alis);
  if (!theCoordDefiners.empty()) { // Create steering with constraints to define coordinate system:
    // OK, some hacks:
    // - we want a composite with global coordinates where tracker and muon are components
    //   (to call RigidBodyAl.Param.->globalParameters() in correctToReferenceSystem(..))
    // - so we create a AlignableComposite and add tracker and muon
    // - but the addComponent(..) method is so cute that it calculates position from 
    //   daughters' deepComponents()
    // - so we want to move it back to (0,0,0), but ali->move(..) would move daughters as well
    //   => cheat with a const_cast and move only the surface back
    // - this hacked master object does not have a label for its parameters
    //   => some warnings if debug output selected in pedeSteer files
    // - we must not delete our new master (little mem. leak...) since that would delete
    //   the daughters as well!
    if (aliTracker) {
      theCoordMaster = new AlignableComposite(aliTracker->id(), align::invalid);
      theCoordMaster->addComponent(aliTracker);
    } else if (aliMuon) {
      theCoordMaster = new AlignableComposite(aliMuon->id(), align::invalid);
    } else {
      throw cms::Exception("BadConfig")
	<< "[PedeSteerer]" << "Cannot define global coordinate system "
	<< "with neither tracker nor muon!";
    }
    if (aliMuon) theCoordMaster->addComponent(aliMuon); // tracker is already added if existing
    if (aliExtras) { // tracker and/or muon are already added if existing
      align::Alignables allExtras = aliExtras->components();
      for ( std::vector<Alignable*>::iterator it = allExtras.begin(); it != allExtras.end(); ++it ) {
	theCoordMaster->addComponent(*it);
      }
    }

    const Alignable::PositionType &tmpPos = theCoordMaster->globalPosition();
    AlignableSurface & masterSurf = const_cast<AlignableSurface&>(theCoordMaster->surface());
    masterSurf.move(align::GlobalVector(-tmpPos.x(),-tmpPos.y(),-tmpPos.z()));

    if (this->isCorrectToRefSystem(theCoordDefiners)) { // defined by 's' (MC): 'correct' misalignment
      this->correctToReferenceSystem(); // really before 'defineCoordinates'?
    }
  } 

}

//___________________________________________________________________________
PedeSteerer::~PedeSteerer()
{
  // delete theCoordMaster; NO, see above
}

//_________________________________________________________________________
bool PedeSteerer::isNoHiera(const Alignable* ali) const
{
  return (myNoHieraCollection.find(ali) != myNoHieraCollection.end());
}

//_________________________________________________________________________
double PedeSteerer::cmsToPedeFactor(unsigned int parNum) const
{
  return 1.; // mmh, otherwise would need to FIXME hierarchyConstraint...

  switch (parNum) {
  case RigidBodyAlignmentParameters::dx:
  case RigidBodyAlignmentParameters::dy:
    return 1000.; // cm to mum *1/10 to get smaller values
  case RigidBodyAlignmentParameters::dz:
    return 2500.;   // cm to mum *1/4 
  case RigidBodyAlignmentParameters::dalpha:
  case RigidBodyAlignmentParameters::dbeta:
    return 1000.; // rad to mrad (no first guess for sensitivity yet)
  case RigidBodyAlignmentParameters::dgamma:
    return 10000.; // rad to mrad *10 to get larger values
  default:
    return 1.;
  }
}

//_________________________________________________________________________
unsigned int PedeSteerer::buildNoHierarchyCollection(const std::vector<Alignable*> &alis)
{
  myNoHieraCollection.clear();  // just in case of re-use...

  for (std::vector<Alignable*>::const_iterator iAli = alis.begin() ; iAli != alis.end(); ++iAli) {
    AlignmentParameters *params = (*iAli)->alignmentParameters();
    SelectionUserVariables *selVar = dynamic_cast<SelectionUserVariables*>(params->userVariables());
    if (!selVar) continue;
    // Now check whether taking out of hierarchy is selected - must be consistent!
    unsigned int numNoHieraPar = 0;
    unsigned int numHieraPar = 0;
    for (unsigned int iParam = 0; static_cast<int>(iParam) < params->size(); ++iParam) {
      const char selector = selVar->fullSelection()[iParam];
      if (selector == 'C' || selector == 'F' || selector == 'H') {
	++numNoHieraPar;
      } else if (selector == 'c' || selector == 'f' || selector == '1' || selector == 'r'
                 || selector == 's') {
	++numHieraPar;
      } // else ... accept '0' as undetermined
    }
    if (numNoHieraPar) { // Selected to be taken out.
      if (numHieraPar) { // Inconsistent: Some parameters still in hierarchy ==> exception!
	throw cms::Exception("BadConfig") 
	  << "[PedeSteerer::buildNoHierarchyCollection] All active parameters of alignables to be "
	  << " taken out of the hierarchy must be marked with capital letters 'C', 'F' or 'H'!";
      }
      bool isInHiera = false; // Check whether Alignable is really part of hierarchy:
      Alignable *mother = *iAli;
      while ((mother = mother->mother())) {
	if (mother->alignmentParameters()) isInHiera = true; // could 'break;', but loop is short
      }
      // Complain, but keep collection short if not in hierarchy:
      if (isInHiera) myNoHieraCollection.insert(*iAli);
      else edm::LogWarning("Alignment") << "@SUB=PedeSteerer::buildNoHierarchyCollection"
					<< "Alignable not in hierarchy, no need to remove it!";
    }
  } // end loop on alignables

  return myNoHieraCollection.size();
}

//_________________________________________________________________________
bool PedeSteerer::checkParameterChoices(const std::vector<Alignable*> &alis) const
{
  for (std::vector<Alignable*>::const_iterator iAli = alis.begin() ; iAli != alis.end(); ++iAli) {
    AlignmentParameters *paras = (*iAli)->alignmentParameters();
    SelectionUserVariables *selVar = dynamic_cast<SelectionUserVariables*>(paras->userVariables());
    if (!selVar) continue;
    for (unsigned int iParam = 0; static_cast<int>(iParam) < paras->size(); ++iParam) {
      const char sel = selVar->fullSelection()[iParam];
      if (sel != 'f' && sel != 'F' && sel != 'c' && sel != 'C' &&
	  sel != '0' && sel != '1' && sel != 'H' && sel != 'r' && sel != 's') {
	throw cms::Exception("BadConfig")
	  << "[PedeSteerer::unknownParameterChoices] " 
	  << "Unexpected parameter selector '" << sel
	  << "', use \n'f/F' (fix),\n'c/C' (fix at correct pos.),\n'1/H' (free),\n"
	  << "'r/s' (free, but defining reference system, trying to correct misalignment if 's')"
          << " or \n'0' (ignore).\n"
	  << "Capital letters mean that the Alignable is taken out of a possible hierarchy,\n"
	  << "but must be used consistently for all its parameters.";
	return false; // unreached
      }
    }
  }

  return true;
}

//_________________________________________________________________________
std::pair<unsigned int, unsigned int>
PedeSteerer::fixParameters(const std::vector<Alignable*> &alis, const std::string &fileName)
{
  // return number of parameters fixed at 0. and fixed at original position 
  std::pair<unsigned int, unsigned int> numFixNumFixCor(0, 0);

  std::ofstream *filePtr = 0;

  for (std::vector<Alignable*>::const_iterator iAli = alis.begin() ; iAli != alis.end(); ++iAli) {

    AlignmentParameters *params = (*iAli)->alignmentParameters();
    SelectionUserVariables *selVar = dynamic_cast<SelectionUserVariables*>(params->userVariables());
    if (!selVar) continue;
    
    for (unsigned int iParam = 0; static_cast<int>(iParam) < params->size(); ++iParam) {
      const unsigned int nInstances = myLabels->numberOfParameterInstances(*iAli, iParam);
      for (unsigned int iInstance=0;iInstance<nInstances;++iInstance) {
	int whichFix = this->fixParameter(*iAli, iInstance, iParam,
					  selVar->fullSelection()[iParam], filePtr,
					  fileName);
	if (whichFix == 1) {
	  ++(numFixNumFixCor.first);
	} else if (whichFix == -1) {
	  ++(numFixNumFixCor.second);
	}
      }
    }
  }

  delete filePtr; // automatically flushes, no problem if NULL ptr.   

  return numFixNumFixCor;
}

//_________________________________________________________________________
int PedeSteerer::fixParameter(Alignable *ali, unsigned int iInstance,
			      unsigned int iParam, char selector,
                              std::ofstream* &filePtr, const std::string &fileName)
{
  int result = 0;
  float fixAt = 0.;
  if (selector == 'c' || selector == 'C') {
    if (ali->alignmentParameters()->type() != AlignmentParametersFactory::kRigidBody) {
      throw cms::Exception("BadConfig") 
	<< "PedeSteerer::fixParameter: correction (c/C) possible only for RigidBodyParameters";
    }
    fixAt = -this->parameterSign() * RigidBodyAlignmentParameters(ali, true).parameters()[iParam];
    result = -1;
  } else if (selector == 'f' || selector == 'F') {
    result = 1;
  }

  if (result) {
    if (!filePtr) {
      filePtr = this->createSteerFile(fileName, true);
      (*filePtr) << "Parameter\n";
    }
    std::ofstream &file = *filePtr;

    const unsigned int aliLabel = myLabels->alignableLabelFromParamAndInstance(ali, iParam, iInstance);
    file << myLabels->parameterLabel(aliLabel, iParam) << "  " 
         << fixAt * this->cmsToPedeFactor(iParam) << " -1.0";
    if (myIsSteerFileDebug) { // debug
      const GlobalPoint position(ali->globalPosition());
      file << "  * id " << ali->id() << ", eta " << position.eta() << ", z " << position.z()
           << ", r " << position.perp() << ", phi " << position.phi();
    }
    file << "\n";
  }

  return result;
}

//_________________________________________________________________________
std::vector<Alignable*> PedeSteerer::selectCoordinateAlis(const std::vector<Alignable*> &alis) const
{
  std::vector<Alignable*> coordAlis;

  for (std::vector<Alignable*>::const_iterator iAli = alis.begin() ; iAli != alis.end(); ++iAli) {
    AlignmentParameters *params = (*iAli)->alignmentParameters();
    SelectionUserVariables *selVar = dynamic_cast<SelectionUserVariables*>(params->userVariables());
    if (!selVar) continue;
    unsigned int refParam = 0;
    unsigned int nonRefParam = 0;
    for (unsigned int iParam = 0; static_cast<int>(iParam) < params->size(); ++iParam) {
      const char selector = selVar->fullSelection()[iParam];
      if (selector == 'r' || selector == 's') {
	++refParam;
      } else if (selector != '0' && selector != 'f') { // allow also 'c'?
	++nonRefParam;
      }
    }
    // Check whether some 'r/s' selection string. If yes and selection makes sense, add to result:
    if (refParam) {
      if (nonRefParam) {
	throw cms::Exception("BadConfig") 
	  << "[PedeSteerer::selectCoordinateAlis] All active parameters of alignables defining "
	  << "the coordinate system must be marked with 'r/s' (or fixed, 'f')!";
      } else {
	Alignable *mother = *iAli;
	while ((mother = mother->mother())) {
	  if (mother->alignmentParameters()) {
	    throw cms::Exception("BadConfig") << "[PedeSteerer::selectCoordinateAlis] "
					      << "Alignables defining the coordinate system must "
					      << "be highest level!";
	  }
	}
	coordAlis.push_back(*iAli);
      }
    }
  } // end loop on alignables

  return coordAlis;
}


//_________________________________________________________________________
void PedeSteerer::defineCoordinates(const std::vector<Alignable*> &alis, Alignable *aliMaster,
				    const std::string &fileName)
{
  std::ofstream *filePtr = this->createSteerFile(fileName, true);
  (*filePtr) << "* Constraints to define coordinate system:\n";
  if (!aliMaster || aliMaster->alignmentParameters()) {
    throw cms::Exception("BadConfig")
      << "[PedeSteerer::defineCoordinates] " << "No master alignable or it has parameters!";
  }
  if (myIsSteerFileDebug) { // See constructor comments about hack:
    edm::LogError("Alignment") << "@SUB=PedeSteerer::defineCoordinates"
			       << "Ignore following LogicErrors from PedeLabeler.";
  }
  AlignmentParameters *par = new RigidBodyAlignmentParameters(aliMaster, false);
  aliMaster->setAlignmentParameters(par); // hierarchyConstraint needs parameters
  this->hierarchyConstraint(aliMaster, alis, *filePtr);
  aliMaster->setAlignmentParameters(0); // erase dummy parameters

  delete filePtr; // automatically flushes, no problem if NULL ptr.   
}

//_________________________________________________________________________
bool PedeSteerer::isCorrectToRefSystem(const std::vector<Alignable*> &coordDefiners) const
{
  bool doCorrect = false;
  bool doNotCorrect = false;
  for (std::vector<Alignable*>::const_iterator it = coordDefiners.begin(), iE=coordDefiners.end();
       it != iE; ++it) {
    SelectionUserVariables *selVar = 
      ((*it)->alignmentParameters() ? 
       dynamic_cast<SelectionUserVariables*>((*it)->alignmentParameters()->userVariables()) : 0);
    if (!selVar) continue;  // is an error!?

    for (unsigned int i = 0; i < selVar->fullSelection().size(); ++i) {
      if (selVar->fullSelection()[i] == 'r') doNotCorrect = true;
      else if (selVar->fullSelection()[i] == 's') doCorrect = true;
    }
  }

  if (doCorrect && doNotCorrect) {
    throw cms::Exception("BadConfig")
      << "[PedeSteerer::doCorrectToRefSystem]: Parameter selection 's' and 'r' must not coexist!";
  }

  return doCorrect;
}

//_________________________________________________________________________
void PedeSteerer::correctToReferenceSystem()
{
  typedef RigidBodyAlignmentParameters RbPars;
  if (!theCoordMaster || theCoordDefiners.empty()) return; // nothing was defined

  std::vector<Alignable*> definerDets; // or ...DetUnits
  for (std::vector<Alignable*>::iterator it = theCoordDefiners.begin(), iE = theCoordDefiners.end();
       it != iE; ++it) {// find lowest level objects of alignables that define the coordinate system
    const std::vector<Alignable*> &comp = (*it)->deepComponents();
    definerDets.insert(definerDets.end(), comp.begin(), comp.end());
  }

  for (unsigned int iLoop = 0; ; ++iLoop) { // iterate: shifts and rotations are not independent
    AlgebraicVector meanPars(RbPars::N_PARAM);
    for (std::vector<Alignable*>::iterator it = definerDets.begin(), iE = definerDets.end();
	 it != iE; ++it) { // sum up mean displacements/misrotations:
      meanPars += RbPars(*it, true).globalParameters();// requires theCoordMaster has global frame
    }
    meanPars /= definerDets.size();
    const align::Scalar squareSum = meanPars.normsq();

    if (squareSum < 1.e-20) break; // sqrt(1.e-20)=1.e-10: close enough to stop iterating
    if (iLoop == 0) {
      edm::LogInfo("Alignment") << "@SUB=PedeSteerer::correctToReferenceSystem"
				<< "Loop " << iLoop << " "
				<< "Mean misalignment of dets of defined coordinate system"
				<< (squareSum < 1.e-20 ? ":" :
				    " (will be iteratively corrected to < 1.e-10):") << meanPars;
    }
    if (iLoop >=5) { // 3 iterations should be safe, use 5 for 'more' safety...
      edm::LogError("Alignment") << "@SUB=PedeSteerer::correctToReferenceSystem"
                                 << "No convergence in " << iLoop << " iterations, " 
                                 << "remaining misalignment: " << meanPars;
      break;
    }

    const GlobalVector globalShift(meanPars[RbPars::dx],meanPars[RbPars::dy],meanPars[RbPars::dz]);
    theCoordMaster->move(-globalShift); // sign to revert
    align::EulerAngles globalAngles(3);
    globalAngles[0] = meanPars[RbPars::dalpha];
    globalAngles[1] = meanPars[RbPars::dbeta];
    globalAngles[2] = meanPars[RbPars::dgamma];
    theCoordMaster->rotateInGlobalFrame(align::toMatrix(-globalAngles)); // sign to revert
  }
  
}

//_________________________________________________________________________
unsigned int PedeSteerer::hierarchyConstraints(const std::vector<Alignable*> &alis,
					       const std::string &fileName)
{
  std::ofstream *filePtr = 0;

  unsigned int nConstraints = 0;
  std::vector<Alignable*> aliDaughts;
  for (std::vector<Alignable*>::const_iterator iA = alis.begin(), iEnd = alis.end();
       iA != iEnd; ++iA) {
    aliDaughts.clear();
    if (!(*iA)->firstCompsWithParams(aliDaughts)) {
      edm::LogWarning("Alignment") << "@SUB=PedeSteerer::hierarchyConstraints"
				   << "Some but not all daughters of "
				   << AlignableObjectId::idToString((*iA)->alignableObjectId())
				   << " with params!";
    }
    //     edm::LogInfo("Alignment") << "@SUB=PedeSteerer::hierarchyConstraints"
    // 			      << aliDaughts.size() << " ali param components";
    if (aliDaughts.empty()) continue;
    //     edm::LogInfo("Alignment") << "@SUB=PedeSteerer::hierarchyConstraints"
    // 			      << aliDaughts.size() << " alignable components ("
    // 			      << (*iA)->size() << " in total) for " 
    // 			      << aliId.alignableTypeName(*iA) 
    // 			      << ", layer " << aliId.typeAndLayerFromAlignable(*iA).second
    // 			      << ", position " << (*iA)->globalPosition()
    // 			      << ", r = " << (*iA)->globalPosition().perp();
    if (!filePtr) filePtr = this->createSteerFile(fileName, true);
    ++nConstraints;
    this->hierarchyConstraint(*iA, aliDaughts, *filePtr);
  }

  delete filePtr; // automatically flushes, no problem if NULL ptr.   

  return nConstraints;
}
//_________________________________________________________________________
void PedeSteerer::hierarchyConstraint(const Alignable *ali,
                                      const std::vector<Alignable*> &components,
				      std::ofstream &file) const
{
  typedef AlignmentParameterStore::ParameterId ParameterId;

  std::vector<std::vector<ParameterId> > paramIdsVec;
  std::vector<std::vector<double> > factorsVec;
  const bool allConstr = false; // true; // make configurable?
  static bool first = true;
  if (allConstr && first) {
    edm::LogWarning("Alignment") << "@SUB=PedeSteerer::hierarchyConstraint"
				 << "changed to use all 6 constraints";
    first = false;
  }
  if (!myParameterStore->hierarchyConstraints(ali, components, paramIdsVec, factorsVec, allConstr,
					      theMinHieraConstrCoeff)){
    edm::LogWarning("Alignment") << "@SUB=PedeSteerer::hierarchyConstraint"
				 << "Problems from store.";
  }

  for (unsigned int iConstr = 0; iConstr < paramIdsVec.size(); ++iConstr) {
    std::ostringstream aConstr;

    const std::vector<ParameterId> &parIds = paramIdsVec[iConstr];
    const std::vector<double> &factors = factorsVec[iConstr];
    unsigned int nParPerConstr = 0; // keep track of used factor/parId pair
    // parIds.size() == factors.size() granted by myParameterStore->hierarchyConstraints
    for (unsigned int iParam = 0; iParam < parIds.size(); ++iParam) {
      Alignable *aliSubComp = parIds[iParam].first;
      const unsigned int compParNum = parIds[iParam].second;
      if (this->isNoHiera(aliSubComp)) {
	if (myIsSteerFileDebug) aConstr << "* Taken out of hierarchy: ";
	continue;
      }
      const unsigned int aliLabel = myLabels->alignableLabel(aliSubComp);
      const unsigned int paramLabel = myLabels->parameterLabel(aliLabel, compParNum);
      // FIXME: multiply by cmsToPedeFactor(subcomponent)/cmsToPedeFactor(mother) (or vice a versa?)
      if (theConstrPrecision > 0)
        aConstr << paramLabel << "    " << std::setprecision(theConstrPrecision) << factors[iParam];
      else
        aConstr << paramLabel << "    " << factors[iParam];
      if (myIsSteerFileDebug) { // debug
	aConstr << "   ! for param " << compParNum << " of a " 
		<< AlignableObjectId::idToString(aliSubComp->alignableObjectId()) << " at " 
		<< aliSubComp->globalPosition() << ", r=" << aliSubComp->globalPosition().perp();
      }
      aConstr << "\n";
      ++nParPerConstr; // OK, we used one.
    } // end loop on params

    // 
    if (nParPerConstr && nParPerConstr >= theMinHieraParPerConstr) { // Enough to make sense?
      if (myIsSteerFileDebug) { //debug
	file << "\n* Nr. " << iConstr << " of a '"
	     << AlignableObjectId::idToString(ali->alignableObjectId()) << "' (label "
	     << myLabels->alignableLabel(const_cast<Alignable*>(ali)) // ugly cast: FIXME!
	     << "), position " << ali->globalPosition()
	     << ", r = " << ali->globalPosition().perp();
      }
      file << "\nConstraint   0.\n" << aConstr.str(); // in future 'Wconstraint'?
    } else if (nParPerConstr > 0) { // no warning for trivial case...
      edm::LogWarning("Alignment") << "@SUB=PedeSteerer::hierarchyConstraint"
				   << "Skip constraint on " << nParPerConstr
				   << " parameter(s):\n" << aConstr.str();
    }
  } // end loop on constraints
}

//_________________________________________________________________________
unsigned int PedeSteerer::presigmas(const std::vector<edm::ParameterSet> &cffPresi,
                                    const std::string &fileName,
                                    const std::vector<Alignable*> &alis,
                                    AlignableTracker *aliTracker, AlignableMuon *aliMuon, AlignableExtras *aliExtras)
{
  // We loop on given PSet's, each containing a parameter selection and the presigma value
  // The resulting presigmas are stored in a map with Alignable* as key.
  // This map, 'fileName' and 'alis' are passed further to create the steering file.

  AlignmentParameterSelector selector(aliTracker, aliMuon, aliExtras);
  AlignablePresigmasMap aliPresiMap; // map to store alis with presigmas of their parameters 
  for (std::vector<edm::ParameterSet>::const_iterator iSet = cffPresi.begin(), iE = cffPresi.end();
       iSet != iE; ++iSet) { // loop on individual PSets defining ali-params with their presigma
    selector.clear();
    selector.addSelections((*iSet).getParameter<edm::ParameterSet>("Selector"));
    const std::vector<Alignable*> &alis = selector.selectedAlignables();
    const std::vector<std::vector<char> > &sels =  selector.selectedParameters();
    const float presigma = (*iSet).getParameter<double>("presigma");
    if (presigma <= 0.) { // given presigma > 0., 0. later used if not (yet) chosen for parameter
      throw cms::Exception("BadConfig")
        << "[PedeSteerer::presigmas]: Pre-sigma must be > 0., but is " << presigma << ".";
    }
    // now loop on alis of present selection
    for (unsigned int iAli = 0; iAli < alis.size(); ++iAli) {
      std::vector<float> &presigmas = aliPresiMap[alis[iAli]]; // existing or empty, so ensure length:
      if (presigmas.size() < sels[iAli].size()) presigmas.resize(sels[iAli].size(), 0.);
      for (unsigned int iParam = 0; iParam < sels[iAli].size(); ++iParam) { // loop on parameters
        if (sels[iAli][iParam] != '0') { // all but '0' means to apply the chosen presigma
          if (presigmas[iParam] != 0.) { // reset forbidden (would make it order dependent!)
            throw cms::Exception("BadConfig")
              << "[PedeSteerer::presigmas]: Try to set pre-sigma " << presigma << ", but already "
              << "set " << presigmas[iParam] << " (for a " 
              << AlignableObjectId::idToString(alis[iAli]->alignableObjectId()) << ").";
          }
          presigmas[iParam] = presigma;
        } // end if selected for presigma
      } // end loop on params
    } // end loop on alignables for given selection and presigma
  } // end loop on PSets 

  if (aliPresiMap.empty()) return 0;
  else return this->presigmasFile(fileName, alis, aliPresiMap);
}

//_________________________________________________________________________
unsigned int PedeSteerer::presigmasFile(const std::string &fileName,
                                        const std::vector<Alignable*> &alis,
                                        const AlignablePresigmasMap &aliPresiMap)
{
  // Check if 'alis' are in aliPresiMap, 
  // if yes apply presigma - but NOT if parameter is fixed!
  std::ofstream *filePtr = 0;

  unsigned int nPresiParam = 0;
  for (std::vector<Alignable*>::const_iterator iAli = alis.begin(), iAliE = alis.end();
       iAli != iAliE; ++iAli) {
    // Any presigma chosen for alignable?
    AlignablePresigmasMap::const_iterator presigmasIt = aliPresiMap.find(*iAli);
    if (presigmasIt == aliPresiMap.end()) continue; // no presigma chosen for alignable

    // Why does the following not work? It does with CMSSW_1_3_X on SLC3...
    // const AlignablePresigmasMap::data_type &presigmas = presigmasIt->second;
    const std::vector<float> &presigmas = presigmasIt->second; // I want to hide float or double...
    for (unsigned int iParam = 0; iParam < presigmas.size(); ++iParam) {
      // Now check whether a presigma value > 0. chosen: 
      if (presigmas[iParam] <= 0.) continue; // must be positive, '<' checked above
      // Do not apply presigma to inactive or fixed values.
      if (!(*iAli)->alignmentParameters()->selector()[iParam]) continue;
      SelectionUserVariables *selVar 
        = dynamic_cast<SelectionUserVariables*>((*iAli)->alignmentParameters()->userVariables());
      const char selChar = (selVar ? selVar->fullSelection()[iParam] : '1');
      if (selChar == 'f' || selChar == 'F' || selChar == 'c' || selChar == 'C') continue;
      // Finally create and write steering file:
      if (!filePtr) {
        filePtr = this->createSteerFile(fileName, true);
        (*filePtr) << "* Presigma values for active parameters: \nParameter\n";
      }
      const unsigned int aliLabel = myLabels->alignableLabel(*iAli);
      (*filePtr) << myLabels->parameterLabel(aliLabel, iParam) << "   0.   " 
                 << presigmas[iParam] * fabs(this->cmsToPedeFactor(iParam));
      if (myIsSteerFileDebug) {
	(*filePtr) << "  ! for a " << AlignableObjectId::idToString((*iAli)->alignableObjectId());
      }
      (*filePtr) << '\n';

      ++nPresiParam;
    } // end loop on parameters for alignables with chosen presigmas
  } // end loop on alignables

  delete filePtr; // close properly file
  return nPresiParam;
}

//_________________________________________________________________________
std::ofstream* PedeSteerer::createSteerFile(const std::string &name, bool addToList)
{
  const std::string realName(myNoSteerFiles ? "/dev/null" : name.c_str());

  std::ofstream *result = new std::ofstream(realName.c_str(), std::ios::out);
  if (!result || !result->is_open()) {
    delete result; // needed before exception in case just open failed
    throw cms::Exception("FileOpenProblem") << "[PedeSteerer::createSteerFile]" 
					    << "Could not open " << realName 
					    << " as output file.";
  } else if (addToList) {
    mySteeringFiles.push_back(realName); // keep track
  }

  return result;
}


//_________________________________________________________________________
std::string PedeSteerer::fileName(const std::string &addendum) const
{

  std::string name(myDirectory);
  name += myConfig.getParameter<std::string>("steerFile");
  name += addendum;
  name += ".txt";

  return name;
}

//___________________________________________________________________________
void PedeSteerer::buildSubSteer(AlignableTracker *aliTracker, AlignableMuon *aliMuon, AlignableExtras *aliExtras)
{
  const std::vector<Alignable*> &alis = myParameterStore->alignables();

  if (theCoordMaster && !theCoordDefiners.empty()) {
    const std::string nameCoordFile(this->fileName("Coord"));
    this->defineCoordinates(theCoordDefiners, theCoordMaster, nameCoordFile);
    edm::LogInfo("Alignment") << "@SUB=PedeSteerer::buildSubSteer" 
                              << theCoordDefiners.size() << " highest level objects define the "
			      << "coordinate system, steering file " << nameCoordFile << ".";
  }

  const std::string nameFixFile(this->fileName("FixPara"));
  const std::pair<unsigned int, unsigned int> nFixFixCor(this->fixParameters(alis, nameFixFile));
  if (nFixFixCor.first != 0 || nFixFixCor.second != 0) {
    edm::LogInfo("Alignment") << "@SUB=PedeSteerer::buildSubSteer" 
                              << nFixFixCor.first << " parameters fixed at 0. and "
                              << nFixFixCor.second << " at 'original' position, "
                              << "steering file " << nameFixFile << ".";
  } 

  if (this->buildNoHierarchyCollection(alis)) { // before hierarchyConstraints(..)
    edm::LogInfo("Alignment") << "@SUB=PedeSteerer::buildSubSteer"
			      << myNoHieraCollection.size()<<" alignables taken out of hierarchy.";
  }

  const std::string nameHierarchyFile(this->fileName("Hierarchy"));
  unsigned int nConstraint = this->hierarchyConstraints(alis, nameHierarchyFile);
  if (nConstraint) {
    edm::LogInfo("Alignment") << "@SUB=PedeSteerer::buildSubSteer" 
                              << "Hierarchy constraints for " << nConstraint << " alignables, "
                              << "steering file " << nameHierarchyFile << ".";
  }
  
  //construct the systematic geometry deformations
  if((myConfig.getParameter<std::vector<edm::ParameterSet> >("constraints")).size() > 0) {
    PedeSteererWeakModeConstraints GeometryConstraints(aliTracker,
                                                       myLabels,
                                                       myConfig.getParameter<std::vector<edm::ParameterSet> >("constraints"),
                                                       myConfig.getParameter<std::string>("steerFile"));
    
    //prepare the output files
    //Get the data structure in which the configuration data are stored.
    //The relation between the ostream* and the corresponding file name needs to be filled
    auto& ConstraintsConfigContainer = GeometryConstraints.getConfigData();
    
    //loop over all configured constraints
    for(auto& it: ConstraintsConfigContainer) {
      //each level has its own constraint which means the output is stored in a separate file
      for(const auto& ilevelsFilename: it.levelsFilenames_) {
        it.mapFileName_.insert(std::make_pair
			       (ilevelsFilename.second,this->createSteerFile(ilevelsFilename.second,true))
			       );
      }
    }
    
    unsigned int nGeometryConstraint = GeometryConstraints.constructConstraints(alis);
    if (nGeometryConstraint) {
      edm::LogInfo("Alignment") << "@SUB=PedeSteerer::buildSubSteer" 
                                << "Geometry constraints for " << nGeometryConstraint << " alignables.";
    }
  }
  
  const std::string namePresigmaFile(this->fileName("Presigma"));
  unsigned int nPresigma = 
    this->presigmas(myConfig.getParameter<std::vector<edm::ParameterSet> >("Presigmas"),
                    namePresigmaFile, alis, aliTracker, aliMuon, aliExtras);
  if (nPresigma) {
    edm::LogInfo("Alignment") << "@SUB=PedeSteerer::buildSubSteer" 
                              << "Presigma values set for " << nPresigma << " parameters, "
                              << "steering file " << namePresigmaFile << ".";
  }

  // Delete all SelectionUserVariables now? They will anyway be overwritten by MillePedeVariables...
}

//_________________________________________________________________________
std::string PedeSteerer::buildMasterSteer(const std::vector<std::string> &binaryFiles)
{
  const std::string nameMasterSteer(this->fileName("Master"));
  std::ofstream *mainSteerPtr = this->createSteerFile(nameMasterSteer, false);
  if (!mainSteerPtr) return "";

  // add external steering files, if any
  std::vector<std::string> addfiles =  myConfig.getParameter<std::vector<std::string> >("additionalSteerFiles");
  mySteeringFiles.insert(mySteeringFiles.end(),
                         addfiles.begin(),
                         addfiles.end());

  // add steering files to master steering file
  std::ofstream &mainSteerRef = *mainSteerPtr;
  for (unsigned int iFile = 0; iFile < mySteeringFiles.size(); ++iFile) {
    mainSteerRef << mySteeringFiles[iFile] << "\n";
  }

  // add binary files to master steering file
  mainSteerRef << "\nCfiles\n";
  for (unsigned int iFile = 0; iFile < binaryFiles.size(); ++iFile) {
    mainSteerRef << binaryFiles[iFile] << "\n";
  }

  // add method
  mainSteerRef << "\nmethod  " << myConfig.getParameter<std::string>("method") << "\n";

  // add further options
  const std::vector<std::string> opt(myConfig.getParameter<std::vector<std::string> >("options"));
  mainSteerRef << "\n* Outlier treatment and other options \n";
  for (unsigned int i = 0; i < opt.size(); ++i) {
    mainSteerRef << opt[i] << "\n";
  }

  delete mainSteerPtr;  // close (and flush) again

  return nameMasterSteer;
}

//_________________________________________________________________________
int PedeSteerer::runPede(const std::string &masterSteer) const
{
  if (masterSteer.empty()) {
    edm::LogError("Alignment") << "@SUB=PedeSteerer::runPede" << "Empty master steer file, stop";
    return 0; //false;
  }

  std::string command(myConfig.getUntrackedParameter<std::string>("pedeCommand"));
  (command += " ") += masterSteer;
  const std::string dump(myConfig.getUntrackedParameter<std::string>("pedeDump"));
  if (!dump.empty()) {
    command += " > ";
    (command += myDirectory) += dump;
  }

  edm::LogInfo("Alignment") << "@SUB=PedeSteerer::runPede" << "Start running " << command;
  // FIXME: Recommended interface to system commands?
  int shellReturn = gSystem->Exec(command.c_str());
  edm::LogInfo("Alignment") << "@SUB=PedeSteerer::runPede" << "Command returns " << shellReturn;

  return shellReturn;
}
