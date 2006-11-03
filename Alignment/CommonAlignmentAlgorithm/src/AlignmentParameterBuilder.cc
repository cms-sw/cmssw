/** \file AlignableParameterBuilder.cc
 *
 *  $Date: 2006/11/03 11:00:55 $
 *  $Revision: 1.6 $
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"

#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"
#include "Alignment/CommonAlignmentParametrization/interface/CompositeRigidBodyAlignmentParameters.h"

//#include "Alignment/TrackerAlignment/interface/AlignableTracker.h" not needed since only forwarded

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignableSelector.h"

// This class's header

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterBuilder.h"


//__________________________________________________________________________________________________
AlignmentParameterBuilder::AlignmentParameterBuilder(AlignableTracker* alignableTracker) :
  theAlignables(), theAlignableTracker(alignableTracker)
{
}

//__________________________________________________________________________________________________
AlignmentParameterBuilder::AlignmentParameterBuilder(AlignableTracker* alignableTracker,
						     const edm::ParameterSet &pSet) :
  theAlignables(), theAlignableTracker(alignableTracker)
{
  this->addSelections(pSet);
}

//__________________________________________________________________________________________________
unsigned int AlignmentParameterBuilder::addSelections(const edm::ParameterSet &pSet)
{

  const char *setName = "alignableParamSelector";
  const std::vector<std::string> selections = pSet.getParameter<std::vector<std::string> >(setName);

   unsigned int addedSets = 0;
   AlignableSelector selector(theAlignableTracker);
   // loop via index instead of iterator due to possible enlargement inside loop
   for (unsigned int iSel = 0; iSel < selections.size(); ++iSel) {

     std::vector<std::string> decompSel(this->decompose(selections[iSel], ','));
     if (decompSel.empty()) continue; // edm::LogError or even cms::Exception??

     selector.clear();

//      // special scenarios mixing alignable and parameter selection
//      const std::string geoSelSpecial(decompSel.size() > 1 ? "," + decompSel[1] : "");
//      if (decompSel[0] == "ScenarioA") {
//        selections.push_back(std::string("PixelHalfBarrelDets,111000") += geoSelSpecial);
//        selections.push_back(std::string("BarrelDSRods,111000") += geoSelSpecial);
//        selections.push_back(std::string("BarrelSSRods,101000") += geoSelSpecial);
//        continue;
//      } else if (decompSel[0] == "ScenarioB") {
//        selections.push_back(std::string("PixelHalfBarrelLadders,111000") += geoSelSpecial);
//        selections.push_back(std::string("BarrelDSLayers,111000") += geoSelSpecial);
//        selections.push_back(std::string("BarrelSSLayers,101000") += geoSelSpecial);
//        continue;
//      } else if (decompSel[0] == "CustomStripLayers") {
//        selections.push_back(std::string("BarrelDSLayers,111000") += geoSelSpecial);
//        selections.push_back(std::string("BarrelSSLayers,110000") += geoSelSpecial);
//        selections.push_back(std::string("TIDLayers,111000") += geoSelSpecial);
//        selections.push_back(std::string("TECLayers,110000") += geoSelSpecial);
//        continue;
//      } else if (decompSel[0] == "CustomStripRods") {
//        selections.push_back(std::string("BarrelDSRods,111000") += geoSelSpecial);
//        selections.push_back(std::string("BarrelSSRods,101000") += geoSelSpecial);
//        selections.push_back(std::string("TIDRings,111000") += geoSelSpecial);
//        selections.push_back(std::string("TECPetals,110000") += geoSelSpecial);
//        continue;
//      } else if (decompSel[0] == "CSA06Selection") {
//        selections.push_back(std::string("TOBDSRods,111111") += geoSelSpecial);
//        selections.push_back(std::string("TOBSSRods15,100111") += geoSelSpecial);
//        selections.push_back(std::string("TIBDSDets,111111") += geoSelSpecial);
//        selections.push_back(std::string("TIBSSDets,100111") += geoSelSpecial);
//        continue;
//      }

     if (decompSel.size() < 2) {
       throw cms::Exception("BadConfig") << "@SUB=AlignmentParameterBuilder::addSelections"
                                         << selections[iSel]<<" from alignableParamSelector: "
                                         << " should have at least 2 ','-separated parts";
     } else if (decompSel.size() > 2) {
       const edm::ParameterSet geoSel(pSet.getParameter<edm::ParameterSet>(decompSel[2].c_str()));
       selector.addSelection(decompSel[0], geoSel);
     } else {
       selector.addSelection(decompSel[0]); // previous selection already cleared above
     }

     this->add(selector.selectedAlignables(), this->decodeParamSel(decompSel[1]));

     ++addedSets;
   }

   edm::LogInfo("Alignment") << "@SUB=AlignmentParameterBuilder::addSelections"
                             << " added " << addedSets << " sets of alignables"
                             << " from PSet " << setName;
   return addedSets;
}

//__________________________________________________________________________________________________
std::vector<bool> AlignmentParameterBuilder::decodeParamSel(const std::string &selString) const
{

  std::vector<bool> result(RigidBodyAlignmentParameters::N_PARAM, false);
  if (selString.length() != RigidBodyAlignmentParameters::N_PARAM) {
    throw cms::Exception("BadConfig") <<"@SUB=AlignmentParameterBuilder::decodeSelections"
                                      << selString << " has wrong size != "
                                      << RigidBodyAlignmentParameters::N_PARAM;
  } else {
    // shifts
    if (selString.substr(0,1)=="1") result[RigidBodyAlignmentParameters::dx] = true;
    if (selString.substr(1,1)=="1") result[RigidBodyAlignmentParameters::dy] = true;
    if (selString.substr(2,1)=="1") result[RigidBodyAlignmentParameters::dz] = true;
    // rotations
    if (selString.substr(3,1)=="1") result[RigidBodyAlignmentParameters::dalpha] = true;
    if (selString.substr(4,1)=="1") result[RigidBodyAlignmentParameters::dbeta] = true;
    if (selString.substr(5,1)=="1") result[RigidBodyAlignmentParameters::dgamma] = true;
  }

  return result;
}


//__________________________________________________________________________________________________
std::vector<std::string> 
AlignmentParameterBuilder::decompose(const std::string &s, std::string::value_type delimiter) const
{

  std::vector<std::string> result;

  std::string::size_type previousPos = 0;
  while (true) {
    const std::string::size_type delimiterPos = s.find(delimiter, previousPos);
    if (delimiterPos == std::string::npos) {
      result.push_back(s.substr(previousPos)); // until end
      break;
    }
    result.push_back(s.substr(previousPos, delimiterPos - previousPos));
    previousPos = delimiterPos + 1; // +1: skip delimiter
  }

  return result;
}

//__________________________________________________________________________________________________
void AlignmentParameterBuilder::add(const std::vector<Alignable*> &alignables,
				    const std::vector<bool> &sel)
{

  int num_adu = 0;
  int num_det = 0;
  int num_hlo = 0;

  // loop on Alignable objects
  for ( std::vector<Alignable*>::const_iterator ia=alignables.begin();
        ia!=alignables.end();  ia++ ) {
    Alignable* ali=(*ia);

//     // select on single/double sided barrel layers
// 	std::pair<int,int> tl=theTrackerAlignableId->typeAndLayerFromAlignable( ali );
//     int type = tl.first;
//     int layer = tl.second;
//
//     bool keep=true;
//     if (theOnlySS) // only single sided
//       if ( (abs(type)==3 || abs(type)==5) && layer<=2 ) 
// 		keep=false;
//
//     if (theOnlyDS) // only double sided
//       if ( (abs(type)==3 || abs(type)==5) && layer>2 )
// 		keep=false;
//
//     // reject layers
//     if ( theSelLayers && (layer<theMinLayer || layer>theMaxLayer) )  
// 	  keep=false;
//
//
//     if (keep) {
    AlgebraicVector par(6,0);
    AlgebraicSymMatrix cov(6,0);

    AlignableDet* alidet = dynamic_cast<AlignableDet*>(ali);
    if (alidet !=0) { // alignable Det
      RigidBodyAlignmentParameters* dap = 
        new RigidBodyAlignmentParameters(ali,par,cov,sel);
      ali->setAlignmentParameters(dap);
      num_det++;
    } else { // higher level object
      CompositeRigidBodyAlignmentParameters* dap = 
        new CompositeRigidBodyAlignmentParameters(ali,par,cov,sel);
      ali->setAlignmentParameters(dap);
      num_hlo++;
    }
    
    theAlignables.push_back(ali);
    num_adu++;
//     }
  }

  edm::LogInfo("Alignment") << "@SUB=AlignmentParameterBuilder::add"
                            << "Added " << num_adu 
                            << " Alignables, of which " << num_det << " are Dets and "
                            << num_hlo << " are higher level.";
}


//__________________________________________________________________________________________________
void AlignmentParameterBuilder::fixAlignables(int n)
{

  if (n<1 || n>3) {
    edm::LogError("BadArgument") << " n = " << n << " is not in [1,3]";
    return;
  }

  std::vector<Alignable*> theNewAlignables;
  int i=0;
  int imax = theAlignables.size();
  for ( std::vector<Alignable*>::const_iterator ia=theAlignables.begin();
        ia!=theAlignables.end();  ia++ ) 
	{
	  i++;
	  if ( n==1 && i>1 ) 
		theNewAlignables.push_back(*ia);
	  else if ( n==2 && i>1 && i<imax ) 
		theNewAlignables.push_back(*ia);
	  else if ( n==3 && i>2 && i<imax) 
		theNewAlignables.push_back(*ia);
	}

  theAlignables = theNewAlignables;

  edm::LogWarning("Alignment") << "removing " << n 
			       << " alignables, so that " << theAlignables.size() 
			       << " alignables left";
  
}

