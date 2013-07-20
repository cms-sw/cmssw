/** \file AlignmentParameterSelector.cc
 *  \author Gero Flucke, Nov. 2006
 *
 *  $Date: 2013/01/07 20:56:25 $
 *  $Revision: 1.21 $
 *  (last update by $Author: wmtan $)
 */

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterSelector.h"
#include "Alignment/CommonAlignment/interface/AlignableExtras.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"  // for enums TID/TIB/etc.
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

//________________________________________________________________________________
AlignmentParameterSelector::AlignmentParameterSelector(AlignableTracker *aliTracker, AlignableMuon* aliMuon,
						       AlignableExtras *aliExtras) :
  theTracker(aliTracker), theMuon(aliMuon), theExtras(aliExtras), theSelectedAlignables(), 
  theRangesEta(), theRangesPhi(), theRangesR(), theRangesX(), theRangesY(), theRangesZ()
{
  this->setSpecials(""); // init theOnlyDS, theOnlySS, theSelLayers, theMinLayer, theMaxLayer, theRphiOrStereoDetUnit
}

//________________________________________________________________________________
void AlignmentParameterSelector::clear()
{
  theSelectedAlignables.clear();
  theSelectedParameters.clear();
  this->clearGeometryCuts();
}

//________________________________________________________________________________
void AlignmentParameterSelector::clearGeometryCuts()
{
  theRangesEta.clear();
  theRangesPhi.clear();
  theRangesR.clear();
  theRangesX.clear();
  theRangesY.clear();
  theRangesZ.clear();

  thePXBDetIdRanges.clear();
  thePXFDetIdRanges.clear();
  theTIBDetIdRanges.clear();
  theTIDDetIdRanges.clear();
  theTOBDetIdRanges.clear();
  theTECDetIdRanges.clear();
}

const AlignableTracker* AlignmentParameterSelector::alignableTracker() const
{
  return theTracker;
}

//__________________________________________________________________________________________________
unsigned int AlignmentParameterSelector::addSelections(const edm::ParameterSet &pSet)
{

  const std::vector<std::string> selections
    = pSet.getParameter<std::vector<std::string> >("alignParams");
  
  unsigned int addedSets = 0;

  for (unsigned int iSel = 0; iSel < selections.size(); ++iSel) {
    std::vector<std::string> decompSel(this->decompose(selections[iSel], ','));
    if (decompSel.empty()) continue; // edm::LogError or even cms::Exception??

    if (decompSel.size() < 2) {
      throw cms::Exception("BadConfig") << "@SUB=AlignmentParameterSelector::addSelections"
                                        << selections[iSel]<<" from alignableParamSelector: "
                                        << " should have at least 2 ','-separated parts";
    } else if (decompSel.size() > 2) {
      const edm::ParameterSet geoSel(pSet.getParameter<edm::ParameterSet>(decompSel[2].c_str()));
      this->addSelection(decompSel[0], this->convertParamSel(decompSel[1]), geoSel);
    } else {
      this->clearGeometryCuts();
      this->addSelection(decompSel[0], this->convertParamSel(decompSel[1]));
    }
    
    ++addedSets;
  }

  return addedSets;
}

//________________________________________________________________________________
void AlignmentParameterSelector::setGeometryCuts(const edm::ParameterSet &pSet)
{
  // Allow non-specified arrays to be interpreted as empty (i.e. no selection),
  // but take care that nothing unknown is configured (to fetch typos!). 

  this->clearGeometryCuts();
  const std::vector<std::string> parameterNames(pSet.getParameterNames());
  for (std::vector<std::string>::const_iterator iParam = parameterNames.begin(),
	 iEnd = parameterNames.end(); iParam != iEnd; ++iParam) {

    // Calling swap is more efficient than assignment:
    if (*iParam == "etaRanges") {
      pSet.getParameter<std::vector<double> >(*iParam).swap(theRangesEta);
    } else if (*iParam == "phiRanges") {
      pSet.getParameter<std::vector<double> >(*iParam).swap(theRangesPhi);
    } else if (*iParam == "rRanges") {
      pSet.getParameter<std::vector<double> >(*iParam).swap(theRangesR);
    } else if (*iParam == "xRanges") {
      pSet.getParameter<std::vector<double> >(*iParam).swap(theRangesX);
    } else if (*iParam == "yRanges") {
      pSet.getParameter<std::vector<double> >(*iParam).swap(theRangesY);
    } else if (*iParam == "zRanges") {
      pSet.getParameter<std::vector<double> >(*iParam).swap(theRangesZ);
    } else if (*iParam == "detIds") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(theDetIds);
    } else if (*iParam == "detIdRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(theDetIdRanges);
    } else if (*iParam == "excludedDetIds") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(theExcludedDetIds);
    } else if (*iParam == "excludedDetIdRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(theExcludedDetIdRanges);
    } else if (*iParam == "pxbDetId") {
      const edm::ParameterSet & pxbDetIdPSet = pSet.getParameterSet(*iParam);
      this->setPXBDetIdCuts(pxbDetIdPSet);
    } else if (*iParam == "pxfDetId") {
      const edm::ParameterSet & pxfDetIdPSet = pSet.getParameterSet(*iParam);
      this->setPXFDetIdCuts(pxfDetIdPSet);
    } else if (*iParam == "tibDetId") {
      const edm::ParameterSet & tibDetIdPSet = pSet.getParameterSet(*iParam);
      this->setTIBDetIdCuts(tibDetIdPSet);
    } else if (*iParam == "tidDetId") {
      const edm::ParameterSet & tidDetIdPSet = pSet.getParameterSet(*iParam);
      this->setTIDDetIdCuts(tidDetIdPSet);
    } else if (*iParam == "tobDetId") {
      const edm::ParameterSet & tobDetIdPSet = pSet.getParameterSet(*iParam);
      this->setTOBDetIdCuts(tobDetIdPSet);
    } else if (*iParam == "tecDetId") {
      const edm::ParameterSet & tecDetIdPSet = pSet.getParameterSet(*iParam);
      this->setTECDetIdCuts(tecDetIdPSet);
    } else {
      throw cms::Exception("BadConfig") << "[AlignmentParameterSelector::setGeometryCuts] "
					<< "Unknown parameter '" << *iParam << "'.\n";
    }
  }
}

//________________________________________________________________________________
void AlignmentParameterSelector::setPXBDetIdCuts(const edm::ParameterSet &pSet)
{
  // Allow non-specified arrays to be interpreted as empty (i.e. no selection),
  // but take care that nothing unknown is configured (to fetch typos!). 

  const std::vector<std::string> parameterNames(pSet.getParameterNames());
  for (std::vector<std::string>::const_iterator iParam = parameterNames.begin(),
	 iEnd = parameterNames.end(); iParam != iEnd; ++iParam) {

    // Calling swap is more efficient than assignment:
    if (*iParam == "ladderRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(thePXBDetIdRanges.theLadderRanges);
    } else if (*iParam == "layerRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(thePXBDetIdRanges.theLayerRanges);
    } else if (*iParam == "moduleRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(thePXBDetIdRanges.theModuleRanges);
    } else {
      throw cms::Exception("BadConfig") << "[AlignmentParameterSelector::setPXBDetIdCuts] "
					<< "Unknown parameter '" << *iParam << "'.\n";
    }
  }
}

//________________________________________________________________________________
void AlignmentParameterSelector::setPXFDetIdCuts(const edm::ParameterSet &pSet)
{
  // Allow non-specified arrays to be interpreted as empty (i.e. no selection),
  // but take care that nothing unknown is configured (to fetch typos!). 

  const std::vector<std::string> parameterNames(pSet.getParameterNames());
  for (std::vector<std::string>::const_iterator iParam = parameterNames.begin(),
	 iEnd = parameterNames.end(); iParam != iEnd; ++iParam) {

    // Calling swap is more efficient than assignment:
    if (*iParam == "bladeRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(thePXFDetIdRanges.theBladeRanges);
    } else if (*iParam == "diskRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(thePXFDetIdRanges.theDiskRanges);
    } else if (*iParam == "moduleRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(thePXFDetIdRanges.theModuleRanges);
    } else if (*iParam == "panelRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(thePXFDetIdRanges.thePanelRanges);
    } else if (*iParam == "sideRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(thePXFDetIdRanges.theSideRanges);
    } else {
      throw cms::Exception("BadConfig") << "[AlignmentParameterSelector::setPXFDetIdCuts] "
					<< "Unknown parameter '" << *iParam << "'.\n";
    }
  }
}

//________________________________________________________________________________
void AlignmentParameterSelector::setTIBDetIdCuts(const edm::ParameterSet &pSet)
{
  // Allow non-specified arrays to be interpreted as empty (i.e. no selection),
  // but take care that nothing unknown is configured (to fetch typos!). 

  const std::vector<std::string> parameterNames(pSet.getParameterNames());
  for (std::vector<std::string>::const_iterator iParam = parameterNames.begin(),
	 iEnd = parameterNames.end(); iParam != iEnd; ++iParam) {

    // Calling swap is more efficient than assignment:
    if (*iParam == "layerRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(theTIBDetIdRanges.theLayerRanges);
    } else if (*iParam == "moduleRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(theTIBDetIdRanges.theModuleRanges);
    } else if (*iParam == "stringRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(theTIBDetIdRanges.theStringRanges);
    } else if (*iParam == "sideRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(theTIBDetIdRanges.theSideRanges);
    } else {
      throw cms::Exception("BadConfig") << "[AlignmentParameterSelector::setTIBDetIdCuts] "
					<< "Unknown parameter '" << *iParam << "'.\n";
    }
  }
}

//________________________________________________________________________________
void AlignmentParameterSelector::setTIDDetIdCuts(const edm::ParameterSet &pSet)
{
  // Allow non-specified arrays to be interpreted as empty (i.e. no selection),
  // but take care that nothing unknown is configured (to fetch typos!). 

  const std::vector<std::string> parameterNames(pSet.getParameterNames());
  for (std::vector<std::string>::const_iterator iParam = parameterNames.begin(),
	 iEnd = parameterNames.end(); iParam != iEnd; ++iParam) {

    // Calling swap is more efficient than assignment:
    if (*iParam == "diskRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(theTIDDetIdRanges.theDiskRanges);
    } else if (*iParam == "moduleRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(theTIDDetIdRanges.theModuleRanges);
    } else if (*iParam == "ringRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(theTIDDetIdRanges.theRingRanges);
    } else if (*iParam == "sideRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(theTIDDetIdRanges.theSideRanges);
    } else {
      throw cms::Exception("BadConfig") << "[AlignmentParameterSelector::setTIDDetIdCuts] "
					<< "Unknown parameter '" << *iParam << "'.\n";
    }
  }
}

//________________________________________________________________________________
void AlignmentParameterSelector::setTOBDetIdCuts(const edm::ParameterSet &pSet)
{
  // Allow non-specified arrays to be interpreted as empty (i.e. no selection),
  // but take care that nothing unknown is configured (to fetch typos!). 

  const std::vector<std::string> parameterNames(pSet.getParameterNames());
  for (std::vector<std::string>::const_iterator iParam = parameterNames.begin(),
	 iEnd = parameterNames.end(); iParam != iEnd; ++iParam) {

    // Calling swap is more efficient than assignment:
    if (*iParam == "layerRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(theTOBDetIdRanges.theLayerRanges);
    } else if (*iParam == "moduleRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(theTOBDetIdRanges.theModuleRanges);
    } else if (*iParam == "sideRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(theTOBDetIdRanges.theSideRanges);
    } else if (*iParam == "rodRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(theTOBDetIdRanges.theRodRanges);
    } else {
      throw cms::Exception("BadConfig") << "[AlignmentParameterSelector::setTOBDetIdCuts] "
					<< "Unknown parameter '" << *iParam << "'.\n";
    }
  }
}

//________________________________________________________________________________
void AlignmentParameterSelector::setTECDetIdCuts(const edm::ParameterSet &pSet)
{
  // Allow non-specified arrays to be interpreted as empty (i.e. no selection),
  // but take care that nothing unknown is configured (to fetch typos!). 

  const std::vector<std::string> parameterNames(pSet.getParameterNames());
  for (std::vector<std::string>::const_iterator iParam = parameterNames.begin(),
	 iEnd = parameterNames.end(); iParam != iEnd; ++iParam) {

    // Calling swap is more efficient than assignment:
    if (*iParam == "wheelRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(theTECDetIdRanges.theWheelRanges);
    } else if (*iParam == "petalRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(theTECDetIdRanges.thePetalRanges);
    } else if (*iParam == "moduleRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(theTECDetIdRanges.theModuleRanges);
    } else if (*iParam == "ringRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(theTECDetIdRanges.theRingRanges);
    } else if (*iParam == "sideRanges") {
      pSet.getParameter<std::vector<int> >(*iParam).swap(theTECDetIdRanges.theSideRanges);
    } else {
      throw cms::Exception("BadConfig") << "[AlignmentParameterSelector::setTECDetIdCuts] "
					<< "Unknown parameter '" << *iParam << "'.\n";
    }
  }
}

//________________________________________________________________________________
unsigned int AlignmentParameterSelector::addSelection(const std::string &name,
                                                      const std::vector<char> &paramSel,
                                                      const edm::ParameterSet &pSet)
{
  this->setGeometryCuts(pSet);
  return this->addSelection(name, paramSel);
}

//________________________________________________________________________________
unsigned int AlignmentParameterSelector::addSelection(const std::string &nameInput, 
                                                      const std::vector<char> &paramSel)
{
  const std::string name(this->setSpecials(nameInput)); // possibly changing name

  unsigned int numAli = 0;

  ////////////////////////////////////
  // Generic Tracker Section
  ////////////////////////////////////
  if (name.find("Tracker") == 0) { // string starts with "Tracker"
    if (!theTracker) {
      throw cms::Exception("BadConfig") << "[AlignmentParameterSelector::addSelection] "
					<< "Configuration requires access to AlignableTracker"
					<< " (for " << name << ") that is not initialized";
    }
    const std::string substructName(name, 7); // erase "Tracker" at the beginning
    numAli += this->add(theTracker->subStructures(substructName), paramSel);
  }
  ////////////////////////////////////
  // Old hardcoded (i.e. deprecated) tracker section (NOTE: no check on theTracker != 0)
  ////////////////////////////////////
  else if (name == "AllDets")       numAli += this->addAllDets(paramSel);
  else if (name == "AllRods")       numAli += this->addAllRods(paramSel);
  else if (name == "AllLayers")     numAli += this->addAllLayers(paramSel);
  else if (name == "AllComponents") numAli += this->add(theTracker->components(), paramSel);
  else if (name == "AllAlignables") numAli += this->addAllAlignables(paramSel);
  //
  // TIB+TOB
  //
  else if (name == "BarrelRods")   numAli += this->add(theTracker->barrelRods(), paramSel);
  else if (name == "BarrelDets")   numAli += this->add(theTracker->barrelGeomDets(), paramSel);
  else if (name == "BarrelLayers") numAli += this->add(theTracker->barrelLayers(), paramSel);
  else if (name == "TOBDets")      numAli += this->add(theTracker->outerBarrelGeomDets(), paramSel);
  else if (name == "TOBRods")      numAli += this->add(theTracker->outerBarrelRods(), paramSel);
  else if (name == "TOBLayers")    numAli += this->add(theTracker->outerBarrelLayers(), paramSel);
  else if (name == "TOBHalfBarrels") numAli += this->add(theTracker->outerHalfBarrels(), paramSel);
  else if (name == "TIBDets")      numAli += this->add(theTracker->innerBarrelGeomDets(), paramSel);
  else if (name == "TIBRods")      numAli += this->add(theTracker->innerBarrelRods(), paramSel);
  else if (name == "TIBLayers")    numAli += this->add(theTracker->innerBarrelLayers(), paramSel);
  else if (name == "TIBHalfBarrels") numAli += this->add(theTracker->innerHalfBarrels(), paramSel);
  //
  // PXBarrel
  //
  else if (name == "PixelHalfBarrelDets") {
    numAli += this->add(theTracker->pixelHalfBarrelGeomDets(), paramSel);
  } else if (name == "PixelHalfBarrelLadders") {
    numAli += this->add(theTracker->pixelHalfBarrelLadders(), paramSel);
  } else if (name == "PixelHalfBarrelLayers") {
    numAli += this->add(theTracker->pixelHalfBarrelLayers(), paramSel);
  } else if (name == "PixelHalfBarrels") {
    numAli += this->add(theTracker->pixelHalfBarrels(), paramSel);
  }
  //
  // PXEndcap
  //
  else if (name == "PXECDets") numAli += this->add(theTracker->pixelEndcapGeomDets(), paramSel);
  else if (name == "PXECPetals") numAli += this->add(theTracker->pixelEndcapPetals(), paramSel);
  else if (name == "PXECLayers") numAli += this->add(theTracker->pixelEndcapLayers(), paramSel);
  else if (name == "PXEndCaps") numAli += this->add(theTracker->pixelEndCaps(), paramSel);
  //
  // Pixel Barrel+endcap
  //
  else if (name == "PixelDets") {
    numAli += this->add(theTracker->pixelHalfBarrelGeomDets(), paramSel);
    numAli += this->add(theTracker->pixelEndcapGeomDets(), paramSel);
  } else if (name == "PixelRods") {
    numAli += this->add(theTracker->pixelHalfBarrelLadders(), paramSel);
    numAli += this->add(theTracker->pixelEndcapPetals(), paramSel);
  } else if (name == "PixelLayers") {
    numAli += this->add(theTracker->pixelHalfBarrelLayers(), paramSel);
    numAli += this->add(theTracker->pixelEndcapLayers(), paramSel);
  }
  //
  // TID
  //
  else if (name == "TIDs")          numAli += this->add(theTracker->TIDs(), paramSel);
  else if (name == "TIDLayers")     numAli += this->add(theTracker->TIDLayers(), paramSel);
  else if (name == "TIDRings")      numAli += this->add(theTracker->TIDRings(), paramSel);
  else if (name == "TIDDets")       numAli += this->add(theTracker->TIDGeomDets(), paramSel);
  //
  // TEC
  //
  else if (name == "TECDets")       numAli += this->add(theTracker->endcapGeomDets(), paramSel);
  else if (name == "TECPetals")     numAli += this->add(theTracker->endcapPetals(), paramSel);
  else if (name == "TECLayers")     numAli += this->add(theTracker->endcapLayers(), paramSel);
  else if (name == "TECs")          numAli += this->add(theTracker->endCaps(), paramSel);
  //
  // StripEndcap (TID+TEC)
  //
  else if (name == "EndcapDets") {
    numAli += this->add(theTracker->TIDGeomDets(), paramSel);
    numAli += this->add(theTracker->endcapGeomDets(), paramSel); 
  } else if (name == "EndcapPetals") {
    numAli += this->add(theTracker->TIDRings(), paramSel);
    numAli += this->add(theTracker->endcapPetals(), paramSel);
  } else if (name == "EndcapLayers") {
    numAli += this->add(theTracker->TIDLayers(), paramSel);
    numAli += this->add(theTracker->endcapLayers(), paramSel);
  }
  //
  // Strip Barrel+endcap
  //
  else if (name == "StripDets") {
    numAli += this->add(theTracker->barrelGeomDets(), paramSel);
    numAli += this->add(theTracker->TIDGeomDets(), paramSel);
    numAli += this->add(theTracker->endcapGeomDets(), paramSel); 
  } else if (name == "StripRods") {
    numAli += this->add(theTracker->barrelRods(), paramSel);
    numAli += this->add(theTracker->TIDRings(), paramSel);
    numAli += this->add(theTracker->endcapPetals(), paramSel);
  } else if (name == "StripLayers") {
    numAli += this->add(theTracker->barrelLayers(), paramSel);
    numAli += this->add(theTracker->TIDLayers(), paramSel);
    numAli += this->add(theTracker->endcapLayers(), paramSel);
  }
  ////////////////////////////////////
  // Muon selection
  ////////////////////////////////////
  // Check if name contains muon and react if alignable muon not initialized
  else if (name.find("Muon") != std::string::npos) {
    if  (!theMuon) {
      throw cms::Exception("BadConfig") << "[AlignmentParameterSelector::addSelection] "
					<< "Configuration requires access to AlignableMuon"
					<< " which is not initialized";
    }
    else if (name == "MuonDTLayers")             add(theMuon->DTLayers(), paramSel);
    else if (name == "MuonDTSuperLayers")        add(theMuon->DTSuperLayers(), paramSel);
    else if (name == "MuonDTChambers")  	 add(theMuon->DTChambers(), paramSel);
    else if (name == "MuonDTStations")  	 add(theMuon->DTStations(), paramSel);
    else if (name == "MuonDTWheels")    	 add(theMuon->DTWheels(), paramSel);
    else if (name == "MuonBarrel")      	 add(theMuon->DTBarrel(), paramSel);
    else if (name == "MuonCSCLayers")   	 add(theMuon->CSCLayers(), paramSel);
    else if (name == "MuonCSCRings")             add(theMuon->CSCRings(), paramSel);
    else if (name == "MuonCSCChambers") 	 add(theMuon->CSCChambers(), paramSel);
    else if (name == "MuonCSCStations") 	 add(theMuon->CSCStations(), paramSel);
    else if (name == "MuonEndcaps")     	 add(theMuon->CSCEndcaps(), paramSel);

    ////////////////////////////////////
    // not found, but Muon
    ////////////////////////////////////
    else {
      throw cms::Exception("BadConfig") <<"[AlignmentParameterSelector::addSelection]"
					<< ": Selection '" << name << "' invalid!";
    }
  }

  ////////////////////////////////////
  // Generic Extra Alignable Section
  ////////////////////////////////////
  else if (name.find("Extras") == 0) { // string starts with "Extras"
    if (!theExtras) {
      throw cms::Exception("BadConfig") << "[AlignmentParameterSelector::addSelection] "
					<< "Configuration requires access to AlignableExtras"
					<< " (for " << name << ") that is not initialized";
    }
    const std::string substructName(name, 6); // erase "Extras" at the beginning
    numAli += this->add(theExtras->subStructures(substructName), paramSel);
  }
  // end of "name.find("Extras") != std::string::npos"

  else {
    throw cms::Exception("BadConfig") <<"[AlignmentParameterSelector::addSelection]"
				      << ": Selection '" << name << "' invalid!";
  }
  
  this->setSpecials(""); // reset

  return numAli;
}

//________________________________________________________________________________
unsigned int AlignmentParameterSelector::add(const align::Alignables &alignables,
                                             const std::vector<char> &paramSel)
{
  unsigned int numAli = 0;

  // loop on Alignable objects
  for (align::Alignables::const_iterator iAli = alignables.begin();
       iAli != alignables.end(); ++iAli) {

    if (!this->layerDeselected(*iAli)             // check layers
	&& !this->detUnitDeselected(*iAli)        // check detunit selection
	&& !this->outsideGeometricalRanges(*iAli) // check geometrical ranges
	&& !this->outsideDetIdRanges(*iAli)) {    // check DetId ranges
      // all fine, so add to output arrays
      theSelectedAlignables.push_back(*iAli);
      theSelectedParameters.push_back(paramSel);
      ++numAli;
    }
  }

  return numAli;
}

//_________________________________________________________________________
bool AlignmentParameterSelector::layerDeselected(const Alignable *ali) const
{
  if (theOnlySS || theOnlyDS || theSelLayers) {
    TrackerAlignableId idProvider;
    std::pair<int,int> typeLayer = idProvider.typeAndLayerFromDetId(ali->id(), alignableTracker()->trackerTopology());
    int type  = typeLayer.first;
    int layer = typeLayer.second;
    
    // select on single/double sided barrel layers in TIB/TOB
    if (theOnlySS // only single sided
	&& (std::abs(type) == SiStripDetId::TIB || std::abs(type) == SiStripDetId::TOB)
	&& layer <= 2) {
      return true;
    }
    if (theOnlyDS // only double sided
	&& (std::abs(type) == SiStripDetId::TIB || std::abs(type) == SiStripDetId::TOB)
	&& layer > 2) {
      return true;
    }
    
    // reject layers
    if (theSelLayers && (layer < theMinLayer || layer > theMaxLayer)) {
      return true;
    }
  }
  
  return false; // do not deselect...
}

//_________________________________________________________________________
bool AlignmentParameterSelector::detUnitDeselected(const Alignable *ali) const
{

  if (theRphiOrStereoDetUnit != Both && ali->alignableObjectId() == align::AlignableDetUnit) {
    const SiStripDetId detId(ali->geomDetId()); // do not know yet whether right type...
    if (detId.det() == DetId::Tracker 
	&& (detId.subdetId() == SiStripDetId::TIB || detId.subdetId() == SiStripDetId::TID ||
	    detId.subdetId() == SiStripDetId::TOB || detId.subdetId() == SiStripDetId::TEC)) {
      // We have a DetUnit in strip, so check for a selection of stereo/rphi (DetUnits in 1D layers are rphi):
      if ((theRphiOrStereoDetUnit == Stereo && !detId.stereo())
	  || (theRphiOrStereoDetUnit == Rphi   &&  detId.stereo())) {
	return true;
      }
    }
  }

  return false; // do not deselect...
}

//_________________________________________________________________________
bool AlignmentParameterSelector::outsideGeometricalRanges(const Alignable *alignable) const
{
  const align::PositionType& position(alignable->globalPosition());

  if (!theRangesEta.empty() && !this->insideRanges<double>((position.eta()),  theRangesEta)) return true;
  if (!theRangesPhi.empty() && !this->insideRanges<double>((position.phi()),  theRangesPhi, true))return true;
  if (!theRangesR.empty()   && !this->insideRanges<double>((position.perp()), theRangesR)) return true;
  if (!theRangesX.empty()   && !this->insideRanges<double>((position.x()),    theRangesX)) return true;
  if (!theRangesY.empty()   && !this->insideRanges<double>((position.y()),    theRangesY)) return true;
  if (!theRangesZ.empty()   && !this->insideRanges<double>((position.z()),    theRangesZ)) return true;
  
  return false;
}

//_________________________________________________________________________
bool AlignmentParameterSelector::outsideDetIdRanges(const Alignable *alignable) const
{
  //const DetId detId(alignable->geomDetId());
  const DetId detId(alignable->id());
  const int subdetId = detId.subdetId();
  
  const TrackerTopology* tTopo = alignableTracker()->trackerTopology();

  if (!theDetIds.empty() &&
      !this->isMemberOfVector((detId.rawId()), theDetIds)) return true;
  if (!theDetIdRanges.empty() &&
      !this->insideRanges<int>((detId.rawId()), theDetIdRanges)) return true;
  if (!theExcludedDetIds.empty() &&
      this->isMemberOfVector((detId.rawId()), theExcludedDetIds)) return true;
  if (!theExcludedDetIdRanges.empty() &&
      this->insideRanges<int>((detId.rawId()), theExcludedDetIdRanges)) return true;

  if (detId.det()==DetId::Tracker) {
    
    if (subdetId==static_cast<int>(PixelSubdetector::PixelBarrel)) {
      if (!thePXBDetIdRanges.theLadderRanges.empty() && 
	  !this->insideRanges<int>(tTopo->pxbLadder(detId), thePXBDetIdRanges.theLadderRanges)) return true;
      if (!thePXBDetIdRanges.theLayerRanges.empty() && 
	  !this->insideRanges<int>(tTopo->pxbLayer(detId), thePXBDetIdRanges.theLayerRanges)) return true;
      if (!thePXBDetIdRanges.theModuleRanges.empty() && 
	  !this->insideRanges<int>(tTopo->pxbModule(detId), thePXBDetIdRanges.theModuleRanges)) return true;
    }
    
    if (subdetId==static_cast<int>(PixelSubdetector::PixelEndcap)) {
      if (!thePXFDetIdRanges.theBladeRanges.empty() && 
	  !this->insideRanges<int>(tTopo->pxfBlade(detId), thePXFDetIdRanges.theBladeRanges)) return true;
      if (!thePXFDetIdRanges.theDiskRanges.empty() && 
	  !this->insideRanges<int>(tTopo->pxfDisk(detId), thePXFDetIdRanges.theDiskRanges)) return true;
      if (!thePXFDetIdRanges.theModuleRanges.empty() && 
	  !this->insideRanges<int>(tTopo->pxfModule(detId), thePXFDetIdRanges.theModuleRanges)) return true;
      if (!thePXFDetIdRanges.thePanelRanges.empty() && 
	  !this->insideRanges<int>(tTopo->pxfPanel(detId), thePXFDetIdRanges.thePanelRanges)) return true;
      if (!thePXFDetIdRanges.theSideRanges.empty() && 
	  !this->insideRanges<int>(tTopo->pxfSide(detId), thePXFDetIdRanges.theSideRanges)) return true;
    }
    
    if (subdetId==static_cast<int>(SiStripDetId::TIB)) {
      if (!theTIBDetIdRanges.theLayerRanges.empty() && 
	  !this->insideRanges<int>(tTopo->tibLayer(detId), theTIBDetIdRanges.theLayerRanges)) return true;
      if (!theTIBDetIdRanges.theModuleRanges.empty() && 
	  !this->insideRanges<int>(tTopo->tibModule(detId), theTIBDetIdRanges.theModuleRanges)) return true;
      if (!theTIBDetIdRanges.theSideRanges.empty() && 
	  !this->insideRanges<int>(tTopo->tibSide(detId), theTIBDetIdRanges.theSideRanges)) return true;
      if (!theTIBDetIdRanges.theStringRanges.empty() && 
	  !this->insideRanges<int>(tTopo->tibString(detId), theTIBDetIdRanges.theStringRanges)) return true;
    }
    
    if (subdetId==static_cast<int>(SiStripDetId::TID)) {
      if (!theTIDDetIdRanges.theDiskRanges.empty() && 
	  !this->insideRanges<int>(tTopo->tidWheel(detId), theTIDDetIdRanges.theDiskRanges)) return true;
      if (!theTIDDetIdRanges.theModuleRanges.empty() && 
	  !this->insideRanges<int>(tTopo->tidModule(detId), theTIDDetIdRanges.theModuleRanges)) return true;
      if (!theTIDDetIdRanges.theRingRanges.empty() && 
	  !this->insideRanges<int>(tTopo->tidRing(detId), theTIDDetIdRanges.theRingRanges)) return true;
      if (!theTIDDetIdRanges.theSideRanges.empty() && 
	  !this->insideRanges<int>(tTopo->tidSide(detId), theTIDDetIdRanges.theSideRanges)) return true;
    }
    
    if (subdetId==static_cast<int>(SiStripDetId::TOB)) {
      if (!theTOBDetIdRanges.theLayerRanges.empty() && 
	  !this->insideRanges<int>(tTopo->tobLayer(detId), theTOBDetIdRanges.theLayerRanges)) return true;
      if (!theTOBDetIdRanges.theModuleRanges.empty() && 
	  !this->insideRanges<int>(tTopo->tobModule(detId), theTOBDetIdRanges.theModuleRanges)) return true;
      if (!theTOBDetIdRanges.theRodRanges.empty() && 
	  !this->insideRanges<int>(tTopo->tobRod(detId), theTOBDetIdRanges.theRodRanges)) return true;
      if (!theTOBDetIdRanges.theSideRanges.empty() && 
	  !this->insideRanges<int>(tTopo->tobSide(detId), theTOBDetIdRanges.theSideRanges)) return true;
    }

    if (subdetId==static_cast<int>(SiStripDetId::TEC)) {
      if (!theTECDetIdRanges.theWheelRanges.empty() && 
	  !this->insideRanges<int>(tTopo->tecWheel(detId), theTECDetIdRanges.theWheelRanges)) return true;
      if (!theTECDetIdRanges.thePetalRanges.empty() && 
	  !this->insideRanges<int>(tTopo->tecPetalNumber(detId), theTECDetIdRanges.thePetalRanges)) return true;
      if (!theTECDetIdRanges.theModuleRanges.empty() && 
	  !this->insideRanges<int>(tTopo->tecModule(detId), theTECDetIdRanges.theModuleRanges)) return true;
      if (!theTECDetIdRanges.theRingRanges.empty() && 
	  !this->insideRanges<int>(tTopo->tecRing(detId), theTECDetIdRanges.theRingRanges)) return true;
      if (!theTECDetIdRanges.theSideRanges.empty() && 
	  !this->insideRanges<int>(tTopo->tecSide(detId), theTECDetIdRanges.theSideRanges)) return true;
    }
    
  }
  
  return false;
}

//_________________________________________________________________________
template<typename T> bool AlignmentParameterSelector::insideRanges(T value,
								   const std::vector<T> &ranges,
								   bool isPhi) const
{
  // might become templated on <double> ?

  if (ranges.size()%2 != 0) {
    cms::Exception("BadConfig") << "@SUB=AlignmentParameterSelector::insideRanges" 
                                << " need even number of entries in ranges instead of "
                                << ranges.size();
    return false;
  }

  for (unsigned int i = 0; i < ranges.size(); i += 2) {
    if (isPhi) { // mapping into (-pi,+pi] and checking for range including sign flip area
      Geom::Phi<double> rangePhi1(ranges[i]);
      Geom::Phi<double> rangePhi2(ranges[i+1]);
      Geom::Phi<double> valuePhi(value);
      if (rangePhi1 <= valuePhi && valuePhi < rangePhi2) { // 'normal'
        return true;
      }
      if (rangePhi2  < rangePhi1 && (rangePhi1 <= valuePhi || valuePhi < rangePhi2)) {// 'sign flip'
        return true;
      }
    } else if (ranges[i] <= value && value < ranges[i+1]) {
      return true;
    }
  }
  
  return false;
}

//_________________________________________________________________________
template<> bool AlignmentParameterSelector::insideRanges<int>(int value,
							      const std::vector<int> &ranges,
							      bool /*isPhi*/) const
{
  if (ranges.size()%2 != 0) {
    cms::Exception("BadConfig") << "@SUB=AlignmentParameterSelector::insideRanges" 
                                << " need even number of entries in ranges instead of "
                                << ranges.size();
    return false;
  }

  for (unsigned int i = 0; i < ranges.size(); i += 2) {
    if (ranges[i] <= value && value <= ranges[i+1]) return true;
  }
  
  return false;
}

bool AlignmentParameterSelector::isMemberOfVector(int value, const std::vector<int> &values) const
{
  if (std::find(values.begin(), values.end(), value)!=values.end()) return true;
  return false;
}

//__________________________________________________________________________________________________
std::vector<std::string> 
AlignmentParameterSelector::decompose(const std::string &s, std::string::value_type delimiter) const
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
std::vector<char> AlignmentParameterSelector::convertParamSel(const std::string &selString) const
{

  // Convert selString into vector<char> of same length.
  // Note: Old implementation in AlignmentParameterBuilder was rigid in length,
  // expecting RigidBodyAlignmentParameters::N_PARAM.
  // But I prefer to be more general and allow other Alignables. It will throw anyway if
  // RigidBodyAlignmentParameters are build with wrong selection length.
  std::vector<char> result(selString.size());

  for (std::string::size_type pos = 0; pos < selString.size(); ++pos) {
    result[pos] = selString[pos];
  }

  return result;
}


//________________________________________________________________________________
std::string AlignmentParameterSelector::setSpecials(const std::string &name)
{
  // Use new string only, although direct erasing of found indicator causes problems for 'DSS',
  // but 'DSS' makes absolutely no sense!
  std::string newName(name); 

  const std::string::size_type ss = newName.rfind("SS");
  if (ss != std::string::npos) {
    newName.erase(ss, 2); // 2: length of 'SS'
    theOnlySS = true;
  } else {
    theOnlySS = false;
  }

  const std::string::size_type ds = newName.rfind("DS");
  if (ds != std::string::npos) {
    newName.erase(ds, 2); // 2: length of 'DS'
    theOnlyDS = true;
  } else {
    theOnlyDS = false;
  }
  
  const std::string::size_type size = newName.size();
  const std::string::size_type layers = newName.rfind("Layers");
  if (layers != std::string::npos && size - layers - 2 == 6 // 2 digits, 6: length of 'Layers'
      && isdigit(newName[size-1]) && isdigit(newName[size-2])) {
    theSelLayers = true;
    theMinLayer = newName[size-2] - '0';
    theMaxLayer = newName[size-1] - '0';
    newName.erase(layers);
  } else {
    theSelLayers = false;
    theMinLayer = -1;
    theMaxLayer = 99999;
  }

  theRphiOrStereoDetUnit = Both;
  if (newName.rfind("Unit") != std::string::npos) {
    const std::string::size_type uRph = newName.rfind("UnitRphi");
    if (uRph != std::string::npos) {
      newName.erase(uRph + 4, 4); // keep 'Unit' (4) and erase 'Rphi' (4)
      theRphiOrStereoDetUnit = Rphi;
    }
    const std::string::size_type uSte = newName.rfind("UnitStereo");
    if (uSte != std::string::npos) {
      newName.erase(uSte + 4, 6); // keep 'Unit' (4) and erase 'Stereo' (6)
      theRphiOrStereoDetUnit = Stereo;
    }
  }

  if (newName != name) {
    LogDebug("Alignment") << "@SUB=AlignmentParameterSelector::setSpecials"
                          << name << " becomes " << newName << ", makes theOnlySS " << theOnlySS
                          << ", theOnlyDS " << theOnlyDS << ", theSelLayers " << theSelLayers
                          << ", theMinLayer " << theMinLayer << ", theMaxLayer " << theMaxLayer
			  << ", theRphiOrStereoDetUnit " << theRphiOrStereoDetUnit;
  }

  return newName;
}

//________________________________________________________________________________
unsigned int AlignmentParameterSelector::addAllDets(const std::vector<char> &paramSel)
{
  unsigned int numAli = 0;

  numAli += this->add(theTracker->barrelGeomDets(), paramSel);          // TIB+TOB
  numAli += this->add(theTracker->endcapGeomDets(), paramSel);          // TEC
  numAli += this->add(theTracker->TIDGeomDets(), paramSel);             // TID
  numAli += this->add(theTracker->pixelHalfBarrelGeomDets(), paramSel); // PixelBarrel
  numAli += this->add(theTracker->pixelEndcapGeomDets(), paramSel);     // PixelEndcap

  return numAli;
}

//________________________________________________________________________________
unsigned int AlignmentParameterSelector::addAllRods(const std::vector<char> &paramSel)
{
  unsigned int numAli = 0;

  numAli += this->add(theTracker->barrelRods(), paramSel);             // TIB+TOB    
  numAli += this->add(theTracker->pixelHalfBarrelLadders(), paramSel); // PixelBarrel
  numAli += this->add(theTracker->endcapPetals(), paramSel);           // TEC        
  numAli += this->add(theTracker->TIDRings(), paramSel);               // TID        
  numAli += this->add(theTracker->pixelEndcapPetals(), paramSel);      // PixelEndcap

  return numAli;
}

//________________________________________________________________________________
unsigned int AlignmentParameterSelector::addAllLayers(const std::vector<char> &paramSel)
{
  unsigned int numAli = 0;

  numAli += this->add(theTracker->barrelLayers(), paramSel);          // TIB+TOB    
  numAli += this->add(theTracker->pixelHalfBarrelLayers(), paramSel); // PixelBarrel
  numAli += this->add(theTracker->endcapLayers(), paramSel);          // TEC
  numAli += this->add(theTracker->TIDLayers(), paramSel);             // TID
  numAli += this->add(theTracker->pixelEndcapLayers(), paramSel);     // PixelEndcap

  return numAli;
}

//________________________________________________________________________________
unsigned int AlignmentParameterSelector::addAllAlignables(const std::vector<char> &paramSel)
{
  unsigned int numAli = 0;

  numAli += this->addAllDets(paramSel);
  numAli += this->addAllRods(paramSel);
  numAli += this->addAllLayers(paramSel);
  numAli += this->add(theTracker->components(), paramSel);

  return numAli;
}
