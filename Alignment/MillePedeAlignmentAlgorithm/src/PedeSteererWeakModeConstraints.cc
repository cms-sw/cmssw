/**
 * \file PedeSteererWeakModeConstraints.cc
 *
 *  \author    : Joerg Behr
 *  date       : February 2013
 *  $Revision: 1.14 $
 *  $Date: 2013/06/18 15:47:55 $
 *  (last update by $Author: jbehr $)
 */

#include "Alignment/MillePedeAlignmentAlgorithm/src/PedeSteererWeakModeConstraints.h"
#include "Alignment/MillePedeAlignmentAlgorithm/src/PedeSteerer.h"

#include "Alignment/MillePedeAlignmentAlgorithm/interface/PedeLabelerBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include <boost/cstdint.hpp> 
#include <boost/assign/list_of.hpp>
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
#include <Geometry/CommonDetUnit/interface/GeomDetUnit.h> 
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





GeometryConstraintConfigData::GeometryConstraintConfigData(const std::vector<double> co,
                                                           const std::string c,
                                                           const std::vector<std::pair<Alignable*,std::string> > &alisFile,
                                                           const int sd,
                                                           const std::vector<Alignable*> ex
                                                           ) :
  coefficients_(co),
  constraintName_(c),
  levelsFilenames_(alisFile),
  excludedAlignables_(ex),
  sysdeformation_(sd)
{
}

//_________________________________________________________________________
PedeSteererWeakModeConstraints::PedeSteererWeakModeConstraints(AlignableTracker *aliTracker,
                                                               const PedeLabelerBase *labels,
                                                               const std::vector<edm::ParameterSet> &config,
                                                               std::string sf
                                                               ) :
  myLabels_(labels),
  myConfig_(config),
  steerFile_(sf)
{
  unsigned int psetnr = 0;
  std::set<std::string> steerFilePrefixContainer;
  for(std::vector<edm::ParameterSet>::const_iterator pset = myConfig_.begin();
      pset != myConfig_.end();
      ++pset) {
    this->verifyParameterNames((*pset),psetnr);
    psetnr++;
 
    const std::vector<double> coefficients = pset->getParameter<std::vector<double> > ("coefficients");
    const std::vector<unsigned int> dm = pset->exists("deadmodules") ?
      pset->getParameter<std::vector<unsigned int> >("deadmodules") : std::vector<unsigned int>();
    std::string name = pset->getParameter<std::string> ("constraint");
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);

    std::stringstream defaultsteerfileprefix;
    defaultsteerfileprefix << "autosteerFilePrefix_" << name << "_" << psetnr;

    std::string steerFilePrefix = pset->exists("steerFilePrefix") ?
      pset->getParameter<std::string> ("steerFilePrefix") : defaultsteerfileprefix.str();

                
    AlignmentParameterSelector selector(aliTracker, NULL, NULL);
    selector.clear();
    selector.addSelections(pset->getParameter<edm::ParameterSet> ("levels"));
        
    const std::vector<Alignable*> &alis = selector.selectedAlignables();
        
    AlignmentParameterSelector selector_excludedalignables(aliTracker, NULL, NULL);
    selector_excludedalignables.clear();
    if(pset->exists("excludedAlignables")) {
      selector_excludedalignables.addSelections(pset->getParameter<edm::ParameterSet> ("excludedAlignables"));
    }
    const std::vector<Alignable*> &excluded_alis = selector_excludedalignables.selectedAlignables();
      
    const std::vector<std::pair<Alignable*, std::string> > levelsFilenames = this->makeLevelsFilenames(steerFilePrefixContainer,
                                                                                                       alis,
                                                                                                       steerFilePrefix);
    //check that the name of the deformation is known and that the number 
    //of provided parameter is right.
    int sysdeformation = this->verifyDeformationName(name,coefficients);
      
    //Add the configuration data for this constraint to the container of config data
    ConstraintsConfigContainer_.push_back(GeometryConstraintConfigData(coefficients,
                                                                       name,
                                                                       levelsFilenames,
                                                                       sysdeformation,
                                                                       excluded_alis));
    if(deadmodules_.size() == 0) { //fill the list of dead modules only once
      edm::LogInfo("Alignment") << "@SUB=PedeSteererWeakModeConstraints"
                                << "Load list of dead modules (size = " << dm.size()<< ").";
      for(std::vector<unsigned int>::const_iterator it = dm.begin(); it != dm.end(); it++) {
        deadmodules_.push_back((*it));
      }
    }
      
  }
}

//_________________________________________________________________________
std::pair<align::GlobalPoint, align::GlobalPoint> PedeSteererWeakModeConstraints::getDoubleSensorPosition(const Alignable *ali) const
{
  const TwoBowedSurfacesAlignmentParameters *aliPar = 
    dynamic_cast<TwoBowedSurfacesAlignmentParameters*>(ali->alignmentParameters());
  if(aliPar) {
    const double ySplit = aliPar->ySplit();
    //const double halfWidth   = 0.5 * ali->surface().width();
    const double halfLength  = 0.5 * ali->surface().length();
    //const double halfLength1 = 0.5 * (halfLength + ySplit);
    //const double halfLength2 = 0.5 * (halfLength - ySplit);
    const double yM1 = 0.5 * (ySplit - halfLength); // y_mean of surface 1
    const double yM2 = yM1 + halfLength;            // y_mean of surface 2
    const align::GlobalPoint pos_sensor0(ali->surface().toGlobal(align::LocalPoint(0.,yM1,0.)));
    const align::GlobalPoint pos_sensor1(ali->surface().toGlobal(align::LocalPoint(0.,yM2,0.)));
    return std::make_pair(pos_sensor0, pos_sensor1);
  } else {
    throw cms::Exception("Alignment")
      << "[PedeSteererWeakModeConstraints::getDoubleSensorPosition]"
      << " Dynamic cast to double sensor parameters failed.";
    return std::make_pair( align::GlobalPoint(0.0, 0.0, 0.0), align::GlobalPoint(0.0, 0.0, 0.0));
  }
}

//_________________________________________________________________________
unsigned int PedeSteererWeakModeConstraints::createAlignablesDataStructure() 
{
  unsigned int nConstraints = 0;
  for(std::list<GeometryConstraintConfigData>::iterator iC = ConstraintsConfigContainer_.begin();
      iC != ConstraintsConfigContainer_.end(); ++iC) {
    //loop over all HLS for which the constraint is to be determined
    for(std::vector<std::pair<Alignable*,std::string> >::const_iterator iHLS = iC->levelsFilenames_.begin();
        iHLS != iC->levelsFilenames_.end(); ++iHLS) {
      //determine next active sub-alignables for iHLS
      std::vector<Alignable*> aliDaughts;
      if (!(*iHLS).first->firstCompsWithParams(aliDaughts)) {
        edm::LogWarning("Alignment") << "@SUB=PedeSteererWeakModeConstraints::createAlignablesDataStructure"
                                     << "Some but not all daughters of "
                                     << AlignableObjectId::idToString((*iHLS).first->alignableObjectId())
                                     << " with params!";
      }
      ++nConstraints;
  
      std::list<Alignable*> usedinconstraint;
      for (std::vector<Alignable*>::const_iterator iD = aliDaughts.begin();
           iD != aliDaughts.end(); ++iD) {
        bool isNOTdead = true;
        for(std::list<unsigned int>::const_iterator iDeadmodules = deadmodules_.begin();
            iDeadmodules != deadmodules_.end(); iDeadmodules++) {
          if( ((*iD)->alignableObjectId() == align::AlignableDetUnit 
               || (*iD)->alignableObjectId() == align::AlignableDet)
              && (*iD)->geomDetId().rawId() == (*iDeadmodules)) {
            isNOTdead = false;
            break;
          }
        }
        //check if the module is excluded
        for(std::vector<Alignable*>::const_iterator iEx = iC->excludedAlignables_.begin();
            iEx != iC->excludedAlignables_.end(); iEx++) {
          if((*iD)->id() == (*iEx)->id() &&  (*iD)->alignableObjectId() == (*iEx)->alignableObjectId() ) {
            //if((*iD)->geomDetId().rawId() == (*iEx)->geomDetId().rawId()) {
            isNOTdead = false;
            break;
          }
        }
        const bool issubcomponent = this->checkMother((*iD),(*iHLS).first);
        if(issubcomponent) {
          if(isNOTdead) {
            usedinconstraint.push_back((*iD));
          }
        } else {
          //sanity check
          throw cms::Exception("Alignment")
            << "[PedeSteererWeakModeConstraints::createAlignablesDataStructure]" 
            << " Sanity check failed. Alignable defined as active sub-component, "
            << " but in fact its not a daugther of " << AlignableObjectId::idToString((*iHLS).first->alignableObjectId());
        }
      }

      if( usedinconstraint.size() > 0){
        (*iC).HLSsubdets_.push_back(std::make_pair((*iHLS).first, usedinconstraint));
      } else {
        edm::LogInfo("Alignment") << "@SUB=PedeSteererWeakModeConstraints"
                                  << "No sub-components for " 
                                  << AlignableObjectId::idToString((*iHLS).first->alignableObjectId())
                                  << "at (" << (*iHLS).first->globalPosition().x() << ","<< (*iHLS).first->globalPosition().y() << "," << (*iHLS).first->globalPosition().z() << ") "
                                  << "selected. Skip constraint";
      }
      if(aliDaughts.size() == 0) {
        edm::LogWarning("Alignment") << "@SUB=PedeSteererWeakModeConstraints::createAlignablesDataStructure"
                                     << "No active sub-alignables found for "
                                     << AlignableObjectId::idToString((*iHLS).first->alignableObjectId())
                                     << " at (" << (*iHLS).first->globalPosition().x() << ","<< (*iHLS).first->globalPosition().y() << "," << (*iHLS).first->globalPosition().z() << ").";
      }
         
       
    }
  }
  return nConstraints;
}

//_________________________________________________________________________
double PedeSteererWeakModeConstraints::getX(const int sysdeformation, const align::GlobalPoint &pos, const double phase) const
{
  double x = 0.0;
  
  const double r = TMath::Sqrt(pos.x() * pos.x() + pos.y() * pos.y());
  
  switch(sysdeformation) {
  case SystematicDeformations::kTwist:
    x = pos.z();
    break;
  case SystematicDeformations::kZexpansion:
    x = pos.z();
    break;
  case SystematicDeformations::kSagitta:
    x = r; 
    break;
  case SystematicDeformations::kRadial:
    x = r; 
    break;
  case SystematicDeformations::kTelescope:
    x = r; 
    break;
  case SystematicDeformations::kLayerRotation:
    x = r; 
    break;
  case SystematicDeformations::kBowing:
    x = pos.z() * pos.z(); //TMath::Abs(pos.z()); 
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
                                                      const align::GlobalPoint &pos,
                                                      const GlobalPoint gUDirection,
                                                      const GlobalPoint gVDirection,
                                                      const GlobalPoint gWDirection,
                                                      const int iParameter, const double &x0,
                                                      const std::vector<double> &constraintparameters) const
{


  if(iParameter < 0 || iParameter > 2) {
    throw cms::Exception("Alignment")
      << "[PedeSteererWeakModeConstraints::getCoefficient]" << " iParameter has to be in the range [0,2] but"
      << " it is equal to " << iParameter << ".";
  }
  

  //global vectors pointing in u,v,w direction
  const std::vector<double> vec_u = boost::assign::list_of(pos.x() - gUDirection.x())(pos.y() - gUDirection.y())(pos.z() - gUDirection.z());
  const std::vector<double> vec_v = boost::assign::list_of(pos.x() - gVDirection.x())(pos.y() - gVDirection.y())(pos.z() - gVDirection.z());
  const std::vector<double> vec_w = boost::assign::list_of(pos.x() - gWDirection.x())(pos.y() - gWDirection.y())(pos.z() - gWDirection.z());

  //FIXME: how to make inner vectors const?
  const std::vector<std::vector<double> > global_vecs = boost::assign::list_of(vec_u)(vec_v)(vec_w);

  const double n = TMath::Sqrt( global_vecs.at(iParameter).at(0) * global_vecs.at(iParameter).at(0) 
                                + global_vecs.at(iParameter).at(1) * global_vecs.at(iParameter).at(1) 
                                + global_vecs.at(iParameter).at(2) * global_vecs.at(iParameter).at(2) );
  const double r = TMath::Sqrt( pos.x() * pos.x() + pos.y() * pos.y() );
  
  const double phase = this->getPhase(constraintparameters);
  //const double radial_direction[3] = {TMath::Sin(phase), TMath::Cos(phase), 0.0};
  const std::vector<double> radial_direction = boost::assign::list_of(TMath::Sin(phase))(TMath::Cos(phase))(0.0);
  //is equal to unity by construction ...
  const double norm_radial_direction =  TMath::Sqrt(radial_direction.at(0) * radial_direction.at(0)
                                                    + radial_direction.at(1) * radial_direction.at(1) 
                                                    + radial_direction.at(2) * radial_direction.at(2));
  
  //const double phi_direction[3] = { -1.0 * pos.y(), pos.x(), 0.0};
  const std::vector<double> phi_direction = boost::assign::list_of(-1.0 * pos.y())(pos.x())(0.0);
  const double norm_phi_direction = TMath::Sqrt(phi_direction.at(0) * phi_direction.at(0)
                                                + phi_direction.at(1) * phi_direction.at(1) 
                                                + phi_direction.at(2) * phi_direction.at(2));
  
  //const double z_direction[3] = {0.0, 0.0, 1.0};
  static const std::vector<double> z_direction = boost::assign::list_of(0.0)(0.0)(1.0);
  const double norm_z_direction = TMath::Sqrt(z_direction.at(0)*z_direction.at(0)
                                              + z_direction.at(1)*z_direction.at(1) 
                                              + z_direction.at(2)*z_direction.at(2));

  //unit vector pointing from the origin to the module position in the transverse plane
  const std::vector<double> rDirection = boost::assign::list_of(pos.x())(pos.y())(0.0);
  const double norm_rDirection =  TMath::Sqrt(rDirection.at(0) * rDirection.at(0)
                                              + rDirection.at(1) * rDirection.at(1) 
                                              + rDirection.at(2) * rDirection.at(2));

  double coeff = 0.0;
  double dot_product = 0.0;
  double normalisation_factor = 1.0;

  //see https://indico.cern.ch/getFile.py/access?contribId=15&sessionId=1&resId=0&materialId=slides&confId=127126
  if(sysdeformation == SystematicDeformations::kTwist 
     || sysdeformation == SystematicDeformations::kLayerRotation) {
    dot_product = phi_direction.at(0) * global_vecs.at(iParameter).at(0) 
      + phi_direction.at(1) * global_vecs.at(iParameter).at(1)
      + phi_direction.at(2) * global_vecs.at(iParameter).at(2);
    normalisation_factor = r * n * norm_phi_direction;
  } else if(sysdeformation == SystematicDeformations::kZexpansion 
            || sysdeformation == SystematicDeformations::kTelescope 
            || sysdeformation == SystematicDeformations::kSkew) {
    dot_product = global_vecs.at(iParameter).at(0) * z_direction.at(0) 
      + global_vecs.at(iParameter).at(1) * z_direction.at(1) 
      + global_vecs.at(iParameter).at(2) * z_direction.at(2);
    normalisation_factor = ( n * norm_z_direction);
  } else if(sysdeformation == SystematicDeformations::kRadial 
            || sysdeformation == SystematicDeformations::kBowing 
            || sysdeformation == SystematicDeformations::kElliptical) {
    dot_product = global_vecs.at(iParameter).at(0) * rDirection.at(0) 
      + global_vecs.at(iParameter).at(1) * rDirection.at(1) 
      + global_vecs.at(iParameter).at(2) * rDirection.at(2);
    normalisation_factor = ( n * norm_rDirection);
  } else if(sysdeformation == SystematicDeformations::kSagitta) {
    dot_product = global_vecs.at(iParameter).at(0) * radial_direction.at(0) 
      + global_vecs.at(iParameter).at(1) * radial_direction.at(1) 
      + global_vecs.at(iParameter).at(2) * radial_direction.at(2);
    normalisation_factor = ( n * norm_radial_direction);
  }
  
  if(TMath::Abs(normalisation_factor) > 0.0) {
    coeff = dot_product * ( this->getX(sysdeformation,pos,phase) - x0 ) / normalisation_factor;
  } else {
    throw cms::Exception("Alignment")
      << "[PedeSteererWeakModeConstraints::getCoefficient]" << " Normalisation factor"
      << "for coefficient calculation equal to zero! Misconfiguration?";
  }
  return coeff;
}

//_________________________________________________________________________
bool PedeSteererWeakModeConstraints::checkSelectionShiftParameter(const Alignable *ali, unsigned int iParameter) const
{
  bool isselected = false;
  const std::vector<bool> &aliSel= ali->alignmentParameters()->selector();
  //exclude non-shift parameters
  if((iParameter <= 2) 
     || (iParameter >= 9 && iParameter <=11)) {
    if(!aliSel.at(iParameter)) {
      isselected = false;
    } else {
      AlignmentParameters *params = ali->alignmentParameters();
      SelectionUserVariables *selVar = dynamic_cast<SelectionUserVariables*>(params->userVariables());
      if (selVar) {
        if(selVar->fullSelection().size() <= (iParameter+1)) {
          throw cms::Exception("Alignment") 
            << "[PedeSteererWeakModeConstraints::checkSelectionShiftParameter]" 
            << " Can not access selected alignment variables of alignable "
            <<  AlignableObjectId::idToString(ali->alignableObjectId())
            << "at (" << ali->globalPosition().x() << ","<< ali->globalPosition().y() << "," << ali->globalPosition().z()<< ") "
            << "for parameter number " << (iParameter+1) << ".";
        }
      }
      const char selChar = (selVar ? selVar->fullSelection().at(iParameter) : '1');
      if(selChar == '1') { //FIXME??? what about 'r'?
        isselected = true;
      } else {
        isselected = false;
      }
    }
  }
  return isselected;
}
//_________________________________________________________________________
void PedeSteererWeakModeConstraints::closeOutputfiles()
{
   //'delete' output files which means: close them
  for(std::list<GeometryConstraintConfigData>::iterator it = ConstraintsConfigContainer_.begin();
      it != ConstraintsConfigContainer_.end(); it++) {
    for(std::map<std::string, std::ofstream*>::iterator iFile = it->mapFileName_.begin();
        iFile != it->mapFileName_.end(); iFile++) {
      if(iFile->second)
        delete iFile->second;
      else {
        throw cms::Exception("FileCloseProblem")
          << "[PedeSteererWeakModeConstraints]" << " can not close file " << iFile->first << ".";
      }
    }
  }
}

//_________________________________________________________________________
void PedeSteererWeakModeConstraints::writeOutput(const std::list<std::pair<unsigned int,double> > &output,
                                                 const std::list<GeometryConstraintConfigData>::const_iterator &it, Alignable* iHLS, double sum_xi_x0)
{
  
  //write output to file
  std::ofstream* ofile = NULL;

  for(std::vector<std::pair<Alignable*, std::string> >::const_iterator ilevelsFilename = it->levelsFilenames_.begin();
      ilevelsFilename != it->levelsFilenames_.end(); ilevelsFilename++) {
    if((*ilevelsFilename).first->id() == iHLS->id() && (*ilevelsFilename).first->alignableObjectId() == iHLS->alignableObjectId()) {

      std::map<std::string, std::ofstream*>::const_iterator iFile = it->mapFileName_.find((*ilevelsFilename).second);
      if(iFile != it->mapFileName_.end()) {
        ofile = (*iFile).second; 
      }
    }
  }

  if(ofile == NULL) {
    throw cms::Exception("FileFindError")
      << "[PedeSteererWeakModeConstraints]" << " Can not find output file.";
  } else {
    if(output.size() > 0) {
      const double constr = sum_xi_x0 * it->coefficients_.front();
      (*ofile) << "Constraint " << std::scientific << constr << std::endl;
      for(std::list<std::pair<unsigned int,double> >::const_iterator ioutput = output.begin();
          ioutput != output.end(); ioutput++) {
        (*ofile) << std::fixed << ioutput->first << " " << std::scientific << ioutput->second << std::endl;
      }
    }
  }
}

//_________________________________________________________________________
double PedeSteererWeakModeConstraints::getX0(std::list<std::pair<Alignable*, std::list<Alignable*> > >::iterator &iHLS,
                                             std::list<GeometryConstraintConfigData>::iterator &it)
{
  double nmodules = 0.0;
  double x0 =0.0;

  for(std::list<Alignable*>::iterator iAlignables = iHLS->second.begin();
      iAlignables != iHLS->second.end(); iAlignables++) {
        
    Alignable *ali = (*iAlignables);
    align::PositionType pos = ali->globalPosition();
    bool alignableIsFloating = false; //means: true=alignable is able to move in at least one direction
       
    //test whether at least one variable has been selected in the configuration
    for(unsigned int iParameter = 0; 
        static_cast<int>(iParameter) < ali->alignmentParameters()->size(); iParameter++) {
      if(this->checkSelectionShiftParameter(ali,iParameter) ) {
        alignableIsFloating = true;
        //verify that alignable has just one label -- meaning no IOV-dependence etc
        const unsigned int nInstances = myLabels_->numberOfParameterInstances(ali, iParameter);
        if(nInstances > 1) {
          throw cms::Exception("PedeSteererWeakModeConstraints")
            << "@SUB=PedeSteererWeakModeConstraints::ConstructConstraints"
            << " Weak mode constraints are only supported for alignables which have"
            << " just one label. However, e.g. alignable" 
            << " " << AlignableObjectId::idToString(ali->alignableObjectId())
            << "at (" << ali->globalPosition().x() << ","<< ali->globalPosition().y() << "," << ali->globalPosition().z()<< "), "
            << " was configured to have >1 label. Remove e.g. IOV-dependence for this (and other) alignables which are used in the constraint.";
        }
        break;
      }
    }
    //at least one parameter of the alignable can be changed in the alignment 
    if(alignableIsFloating) {
      const double phase = this->getPhase(it->coefficients_);
      if(ali->alignmentParameters()->type() != AlignmentParametersFactory::kTwoBowedSurfaces ) {
        x0 += this->getX(it->sysdeformation_,pos,phase);
        nmodules++;
      } else {
        std::pair<align::GlobalPoint, align::GlobalPoint> sensorpositions = this->getDoubleSensorPosition(ali);
        x0 += this->getX(it->sysdeformation_,sensorpositions.first,phase) + this->getX(it->sysdeformation_,sensorpositions.second,phase);
        nmodules++;
        nmodules++;
      }
    }
  }
  if(nmodules>0) {
    x0 = x0 / nmodules;
  } else {
    throw cms::Exception("Alignment") << "@SUB=PedeSteererWeakModeConstraints::ConstructConstraints"
                                      << " Number of selected modules equal to zero. Check configuration!";
    x0 = 1.0;
  }
  return x0;
}

//_________________________________________________________________________
unsigned int PedeSteererWeakModeConstraints::constructConstraints(const std::vector<Alignable*> &alis)
{
  //FIXME: split the code of the method into smaller pieces/submethods
  
  //create the data structures that store the alignables 
  //for which the constraints need to be calculated and
  //their association to high-level structures
  const unsigned int nConstraints = this->createAlignablesDataStructure();

  //calculate constraints
  //loop over all constraints
  for(std::list<GeometryConstraintConfigData>::iterator it = ConstraintsConfigContainer_.begin();
      it != ConstraintsConfigContainer_.end(); it++) {
     
    //loop over all subdets for which constraints are determined
    for(std::list<std::pair<Alignable*, std::list<Alignable*> > >::iterator iHLS = it->HLSsubdets_.begin();
        iHLS != it->HLSsubdets_.end(); iHLS++) {
      double sum_xi_x0 = 0.0;
      std::list<std::pair<unsigned int,double> > output;
      
      const double x0 = this->getX0(iHLS, it);
      
      for(std::list<Alignable*>::const_iterator iAlignables = iHLS->second.begin();
          iAlignables != iHLS->second.end(); iAlignables++) {
        const Alignable *ali = (*iAlignables);
        const unsigned int aliLabel = myLabels_->alignableLabel(const_cast<Alignable*>(ali));
        const AlignableSurface &surface = ali->surface();

        const LocalPoint  lUDirection(1.,0.,0.),
          lVDirection(0.,1.,0.),
          lWDirection(0.,0.,1.);
        
        GlobalPoint gUDirection = surface.toGlobal(lUDirection),
          gVDirection = surface.toGlobal(lVDirection),
          gWDirection = surface.toGlobal(lWDirection);
        
        const bool isDoubleSensor = ali->alignmentParameters()->type() == AlignmentParametersFactory::kTwoBowedSurfaces ? true : false;
       
        const std::pair<align::GlobalPoint, align::GlobalPoint> sensorpositions =
          isDoubleSensor ? this->getDoubleSensorPosition(ali) : std::make_pair(ali->globalPosition(), align::PositionType ());

        const align::GlobalPoint &pos_sensor0 = sensorpositions.first;
        const align::GlobalPoint &pos_sensor1 = sensorpositions.second;
        const double phase = this->getPhase(it->coefficients_);
        const double x_sensor0 = this->getX(it->sysdeformation_,pos_sensor0,phase);
        const double x_sensor1 = isDoubleSensor ? this->getX(it->sysdeformation_,pos_sensor1,phase) : 0.0;
            
        sum_xi_x0 += ( x_sensor0 - x0 ) * ( x_sensor0 - x0 );
        if(isDoubleSensor) {
          sum_xi_x0 += ( x_sensor1 - x0 ) * ( x_sensor1 - x0 );
        }
        const int numparameterlimit = ali->alignmentParameters()->size(); //isDoubleSensor ? 18 : 3;
        
        for(int iParameter = 0; iParameter < numparameterlimit; iParameter++) {
          int localindex = 0;
          if(iParameter == 0 || iParameter == 9)
            localindex = 0;
          if(iParameter == 1 || iParameter == 10)
            localindex = 1;
          if(iParameter == 2 || iParameter == 11)
            localindex = 2;

          if((iParameter >= 0 && iParameter <= 2) 
             || (iParameter >= 9 && iParameter <=11)) {
          } else {
            continue;
          }
          if(! this->checkSelectionShiftParameter(ali,iParameter) ) {
            continue;
          }
          //do it for each 'instance' separately? -> IOV-dependence, no
          const unsigned int paramLabel = myLabels_->parameterLabel(aliLabel, iParameter);
                  
          const align::GlobalPoint &pos = (iParameter <= 2) ? pos_sensor0 : pos_sensor1;
          //select only u,v,w
          if(iParameter == 0 || iParameter == 1 || iParameter == 2 
             || iParameter == 9 || iParameter == 10 || iParameter == 11) {
            const double coeff = this->getCoefficient(it->sysdeformation_, 
                                                      pos, 
                                                      gUDirection,
                                                      gVDirection,
                                                      gWDirection,
                                                      localindex,
                                                      x0,
                                                      it->coefficients_);
            if(TMath::Abs(coeff) > 0.0) {
              //nothing
            } else {
              edm::LogWarning("PedeSteererWeakModeConstraints")
                << "@SUB=PedeSteererWeakModeConstraints::getCoefficient"
                << "Coefficient of alignable " 
                <<  AlignableObjectId::idToString(ali->alignableObjectId())
                << " at (" << ali->globalPosition().x() << ","<< ali->globalPosition().y() << "," << ali->globalPosition().z()<< ") "
                << " in subdet " << AlignableObjectId::idToString((*iHLS).first->alignableObjectId())
                << " for parameter " << localindex << " equal to zero. This alignable is used in the constraint"
                << " '" << it->constraintName_ << "'. The id is: alignable->geomDetId().rawId() = "
                << ali->geomDetId().rawId() << ".";
            }
            output.push_back(std::make_pair (paramLabel, coeff));
          }
        }
        
        
      }

      this->writeOutput(output, it, (*iHLS).first, sum_xi_x0);
    }
  }
  this->closeOutputfiles();
 
  return nConstraints;
}

//_________________________________________________________________________
bool PedeSteererWeakModeConstraints::checkMother(const Alignable * const lowleveldet, const Alignable * const HLS) const
{
  if(lowleveldet->id() == HLS->id() && lowleveldet->alignableObjectId() == HLS->alignableObjectId()) {
    return true;
  } else {
    if(lowleveldet->mother() == NULL) 
      return false;
    else
      return this->checkMother(lowleveldet->mother(),HLS);
  }
}

//_________________________________________________________________________
void PedeSteererWeakModeConstraints::verifyParameterNames(const edm::ParameterSet &pset, unsigned int psetnr) const
{
  std::vector<std::string> parameterNames = pset.getParameterNames();
  for ( std::vector<std::string>::const_iterator iParam = parameterNames.begin(); 
        iParam != parameterNames.end(); ++iParam) {
    const std::string name = (*iParam);
    if(
       name != "coefficients"
       && name != "deadmodules" && name != "constraint"
       && name != "steerFilePrefix" && name != "levels"
       && name != "excludedAlignables"
       ) {
      throw cms::Exception("BadConfig")
        << "@SUB=PedeSteererWeakModeConstraints::verifyParameterNames:"
        << " Unknown parameter name '" << name << "' in PSet number " << psetnr << ". Maybe a typo?";
    }
  }
}

//_________________________________________________________________________
const std::vector<std::pair<Alignable*, std::string> > PedeSteererWeakModeConstraints::makeLevelsFilenames(
                                                                                                           std::set<std::string> &steerFilePrefixContainer,
                                                                                                           const std::vector<Alignable*> &alis,
                                                                                                           const std::string &steerFilePrefix
                                                                                                           ) const
{
  //check whether the prefix is unique
  if(steerFilePrefixContainer.find(steerFilePrefix) != steerFilePrefixContainer.end()) {
    throw cms::Exception("BadConfig") << "[PedeSteererWeakModeConstraints] Steering file"
                                      << " prefix '" << steerFilePrefix << "' already exists. Specify unique names!";
  } else {
    steerFilePrefixContainer.insert(steerFilePrefix);
  }
        
  std::vector<std::pair<Alignable*, std::string> > levelsFilenames;
  for(std::vector<Alignable*>::const_iterator it = alis.begin(); it != alis.end(); ++it) {
    std::stringstream n;
    n << steerFile_ << "_" << steerFilePrefix //<< "_" << name 
      << "_" << AlignableObjectId::idToString((*it)->alignableObjectId())
      << "_" << (*it)->id() << "_" << (*it)->alignableObjectId() << ".txt";
            
    levelsFilenames.push_back(std::make_pair((*it),n.str()));
  }
  return levelsFilenames;
}

//_________________________________________________________________________
int PedeSteererWeakModeConstraints::verifyDeformationName(const std::string &name, const std::vector<double> &coefficients) const
{
  int sysdeformation = SystematicDeformations::kUnknown;

  if(name == "twist") {
    sysdeformation = SystematicDeformations::kTwist;
  }  else if(name == "zexpansion") {
    sysdeformation = SystematicDeformations::kZexpansion;
  } else if(name == "sagitta") {
    sysdeformation = SystematicDeformations::kSagitta;
  } else if(name == "radial") {
    sysdeformation = SystematicDeformations::kRadial;
  } else if(name == "telescope") {
    sysdeformation = SystematicDeformations::kTelescope;
  } else if(name == "layerrotation") {
    sysdeformation = SystematicDeformations::kLayerRotation;
  } else if(name == "bowing") {
    sysdeformation = SystematicDeformations::kBowing;
  } else if(name == "skew") {
    sysdeformation = SystematicDeformations::kSkew;
  } else if(name == "elliptical") {
    sysdeformation = SystematicDeformations::kElliptical;
  }
        
  if(sysdeformation == SystematicDeformations::kUnknown) {
    throw cms::Exception("BadConfig")
      << "[PedeSteererWeakModeConstraints]" << " specified configuration option '"
      << name << "' not known.";
  }
  if((sysdeformation == SystematicDeformations::kSagitta 
      || sysdeformation == SystematicDeformations::kElliptical 
      || sysdeformation == SystematicDeformations::kSkew) && coefficients.size() != 2) {
    throw cms::Exception("BadConfig")
      << "[PedeSteererWeakModeConstraints]" << " Excactly two parameters using the coefficient"
      << " variable have to be provided for the " << name << " constraint.";
  }
  if((sysdeformation == SystematicDeformations::kTwist 
      || sysdeformation == SystematicDeformations::kZexpansion
      || sysdeformation == SystematicDeformations::kTelescope
      || sysdeformation == SystematicDeformations::kLayerRotation
      || sysdeformation == SystematicDeformations::kRadial
      || sysdeformation == SystematicDeformations::kBowing) && coefficients.size() != 1) {
    throw cms::Exception("BadConfig")
      << "[PedeSteererWeakModeConstraints]" << " Excactly ONE parameter using the coefficient"
      << " variable have to be provided for the " << name << " constraint.";
  }

  if(coefficients.size() == 0) {
    throw cms::Exception("BadConfig")
      << "[PedeSteererWeakModeConstraints]" << " At least one coefficient has to be specified.";
  }
  return sysdeformation;
}
 

//_________________________________________________________________________
double PedeSteererWeakModeConstraints::getPhase(const std::vector<double> &coefficients) const
{
  return coefficients.size() == 2 ? coefficients.at(1) : 0.0; //treat second parameter as phase otherwise return 0
}

//_________________________________________________________________________
PedeSteererWeakModeConstraints::~PedeSteererWeakModeConstraints()
{
 
}
