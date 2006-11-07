/** \file AlignmentParameterSelector.cc
 *  \author Gero Flucke, Nov. 2006
 *
 *  $Date: 2006/11/03 16:46:58 $
 *  $Revision: 1.1 $
 *  (last update by $Author: flucke $)
 */

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterSelector.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h" // for enums TID/TIB/etc.

#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/Phi.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//________________________________________________________________________________
AlignmentParameterSelector::AlignmentParameterSelector(AlignableTracker *aliTracker) :
  theTracker(aliTracker), theSelectedAlignables(), 
  theRangesEta(), theRangesPhi(), theRangesR(), theRangesZ(),
  theOnlyDS(false), theOnlySS(false), theSelLayers(false), theMinLayer(-1), theMaxLayer(999)
{
}

//________________________________________________________________________________
AlignmentParameterSelector::~AlignmentParameterSelector()
{
}

//________________________________________________________________________________
const std::vector<Alignable*>& AlignmentParameterSelector::selectedAlignables() const
{
  return theSelectedAlignables;
}

//________________________________________________________________________________
const std::vector<std::vector<bool> >& AlignmentParameterSelector::selectedParameters() const
{
  return theSelectedParameters;
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
  theRangesZ.clear();
}

//__________________________________________________________________________________________________
unsigned int AlignmentParameterSelector::addSelections(const edm::ParameterSet &pSet)
{

  const std::vector<std::string> selections
    = pSet.getParameter<std::vector<std::string> >("alignableParamSelector");
  
  unsigned int addedSets = 0;

  // loop via index instead of iterator due to possible enlargement inside loop
  for (unsigned int iSel = 0; iSel < selections.size(); ++iSel) {

    std::vector<std::string> decompSel(this->decompose(selections[iSel], ','));
    if (decompSel.empty()) continue; // edm::LogError or even cms::Exception??

//      // special scenarios have to be given in configuration
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
      throw cms::Exception("BadConfig") << "@SUB=AlignmentParameterSelector::addSelections"
                                        << selections[iSel]<<" from alignableParamSelector: "
                                        << " should have at least 2 ','-separated parts";
    } else if (decompSel.size() > 2) {
      const edm::ParameterSet geoSel(pSet.getParameter<edm::ParameterSet>(decompSel[2].c_str()));
      this->addSelection(decompSel[0], this->decodeParamSel(decompSel[1]), geoSel);
    } else {
      this->clearGeometryCuts();
      this->addSelection(decompSel[0], this->decodeParamSel(decompSel[1]));
    }
    
    ++addedSets;
  }

  return addedSets;
}

//________________________________________________________________________________
void AlignmentParameterSelector::setGeometryCuts(const edm::ParameterSet &pSet)
{

  theRangesEta = pSet.getParameter<std::vector<double> >("etaRanges");
  theRangesPhi = pSet.getParameter<std::vector<double> >("phiRanges");
  theRangesR   = pSet.getParameter<std::vector<double> >("rRanges"  );
  theRangesZ   = pSet.getParameter<std::vector<double> >("zRanges"  );
}

//________________________________________________________________________________
unsigned int AlignmentParameterSelector::addSelection(const std::string &name,
                                                      const std::vector<bool> &paramSel,
                                                      const edm::ParameterSet &pSet)
{
  this->setGeometryCuts(pSet);
  return this->addSelection(name, paramSel);
}

//________________________________________________________________________________
unsigned int AlignmentParameterSelector::addSelection(const std::string &name, 
                                                      const std::vector<bool> &paramSel)
{

  unsigned int numAli = 0;

  if      (name == "AllDets")       numAli += this->addAllDets(paramSel);
  else if (name == "AllRods")       numAli += this->addAllRods(paramSel);
  else if (name == "AllLayers")     numAli += this->addAllLayers(paramSel);
  else if (name == "AllComponents") numAli += this->add(theTracker->components(), paramSel);
  else if (name == "AllAlignables") numAli += this->addAllAlignables(paramSel);
  //
  // TIB+TOB
  //
  else if (name == "BarrelRods")    numAli += this->add(theTracker->barrelRods(), paramSel);
  else if (name == "BarrelDets")    numAli += this->add(theTracker->barrelGeomDets(), paramSel);
  else if (name == "BarrelLayers")  numAli += this->add(theTracker->barrelLayers(), paramSel);
  else if (name == "BarrelDSRods") {
    theOnlyDS = true;
    numAli += this->add(theTracker->barrelRods(), paramSel);
    theOnlyDS = false;
  } else if (name == "BarrelSSRods") {
    theOnlySS = true;
    numAli += this->add(theTracker->barrelRods(), paramSel);
    theOnlySS = false;
  } else if (name == "BarrelDSLayers") { // new
    theOnlyDS = true;
    numAli += this->add(theTracker->barrelLayers(), paramSel);
    theOnlyDS = false;
  } else if (name == "BarrelSSLayers") { // new
    theOnlySS = true;
    numAli += this->add(theTracker->barrelLayers(), paramSel);
    theOnlySS = false;
  } else if (name == "TOBDSRods") { // new for CSA06Selection
    theOnlyDS = true; 
    numAli += this->add(theTracker->outerBarrelRods(), paramSel);
    theOnlyDS = false;
  } else if (name == "TOBSSRodsLayers15") { // new for CSA06Selection
    // FIXME: make Layers15 flexible
    theSelLayers = theOnlySS = true; 
    theMinLayer = 1;
    theMaxLayer = 5; //  TOB outermost layer (6) kept fixed
    numAli += this->add(theTracker->outerBarrelRods(), paramSel);
    theSelLayers = theOnlySS = false;
  } else if (name == "TIBDSDets") { // new for CSA06Selection
    theOnlyDS = true; 
    numAli += this->add(theTracker->innerBarrelGeomDets(), paramSel);
    theOnlyDS = false;
  } else if (name == "TIBSSDets") { // new for CSA06Selection
    theOnlySS = true; 
    numAli += this->add(theTracker->innerBarrelGeomDets(), paramSel);
    theOnlySS = false;
  }
  //
  // PXBarrel
  //
  else if (name == "PixelHalfBarrelDets") {
    numAli += this->add(theTracker->pixelHalfBarrelGeomDets(), paramSel);
  } else if (name == "PixelHalfBarrelLadders") {
    numAli += this->add(theTracker->pixelHalfBarrelLadders(), paramSel);
  } else if (name == "PixelHalfBarrelLayers") {
    numAli += this->add(theTracker->pixelHalfBarrelLayers(), paramSel);
  } else if (name == "PixelHalfBarrelLaddersLayers12") {
    // FIXME: make Layers12 flexible
    theSelLayers = true; 
    theMinLayer = 1;
    theMaxLayer = 2;
    numAli += this->add(theTracker->pixelHalfBarrelLadders(), paramSel);
    theSelLayers = false;
  }
  //
  // PXEndcap
  //
  else if (name == "PXECDets") numAli += this->add(theTracker->pixelEndcapGeomDets(), paramSel);
  else if (name == "PXECPetals") numAli += this->add(theTracker->pixelEndcapPetals(), paramSel);
  else if (name == "PXECLayers") numAli += this->add(theTracker->pixelEndcapLayers(), paramSel);
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
  else if (name == "TIDLayers")     numAli += this->add(theTracker->TIDLayers(), paramSel);
  else if (name == "TIDRings")      numAli += this->add(theTracker->TIDRings(), paramSel);
  else if (name == "TIDDets")       numAli += this->add(theTracker->TIDGeomDets(), paramSel);
  //
  // TEC
  //
  else if (name == "TECDets")       numAli += this->add(theTracker->endcapGeomDets(), paramSel);
  else if (name == "TECPetals")     numAli += this->add(theTracker->endcapPetals(), paramSel);
  else if (name == "TECLayers")     numAli += this->add(theTracker->endcapLayers(), paramSel);
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
  // not found!
  else { // @SUB-syntax is not supported by exception, but anyway useful information... 
    throw cms::Exception("BadConfig") <<"@SUB=TrackerAlignmentSelector::addSelection"
				      << ": Selection '" << name << "' invalid!";
  }
  
  return numAli;
}

//________________________________________________________________________________
unsigned int AlignmentParameterSelector::add(const std::vector<Alignable*> &alignables,
                                             const std::vector<bool> &paramSel)
{
  unsigned int numAli = 0;

  // loop on Alignable objects
  for (std::vector<Alignable*>::const_iterator iAli = alignables.begin();
       iAli != alignables.end(); ++iAli) {
    bool keep = true;
    
    if (theOnlySS || theOnlyDS || theSelLayers) {
      TrackerAlignableId idProvider;
      std::pair<int,int> typeLayer = idProvider.typeAndLayerFromAlignable(*iAli);
      int type  = typeLayer.first;
      int layer = typeLayer.second;

      // select on single/double sided barrel layers
      if (theOnlySS // only single sided
	  && (abs(type) == StripSubdetector::TIB || abs(type) == StripSubdetector::TOB)
	  && layer <= 2) {
	  keep = false;
      }
      if (theOnlyDS // only double sided
	  && (abs(type) == StripSubdetector::TIB || abs(type) == StripSubdetector::TOB)
	  && layer > 2) {
	  keep = false;
      }
      // reject layers
      if (theSelLayers && (layer < theMinLayer || layer > theMaxLayer)) {
	keep = false;
      }
    }
    // check ranges
    if (keep && this->outsideRanges(*iAli)) keep = false;

    if (keep) {
      theSelectedAlignables.push_back(*iAli);
      theSelectedParameters.push_back(paramSel);
      ++numAli;
    }
  }

  return numAli;
}

//_________________________________________________________________________
bool AlignmentParameterSelector::outsideRanges(const Alignable *alignable) const
{

  const GlobalPoint position(alignable->globalPosition());

  if (!theRangesEta.empty() && !this->insideRanges((position.eta()), theRangesEta)) return true;
  if (!theRangesPhi.empty() && !this->insideRanges((position.phi()), theRangesPhi,true))return true;
  if (!theRangesR.empty()   && !this->insideRanges((position.perp()),theRangesR)) return true;
  if (!theRangesZ.empty()   && !this->insideRanges((position.z()),   theRangesZ)) return true;

  return false;
}

//_________________________________________________________________________
bool AlignmentParameterSelector::insideRanges(double value, const std::vector<double> &ranges,
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
std::vector<bool> AlignmentParameterSelector::decodeParamSel(const std::string &selString) const
{

  // Note: old implementation in AlignmentParameterBuilder tolerated other chars than 0,
  // but was rigid in length, expecting RigidBodyAlignmentParameters::N_PARAM.
  // But I prefer to be more general and allow other Alignables. It will throw anyway if
  // RigidBodyAlignmentParameters are build with wrong selection length.
  // I do not tolerate other chars to detect if another kind of string was mixed up.
  std::vector<bool> result(selString.size());

  for (std::string::size_type pos = 0; pos < selString.size(); ++pos) {
    switch (selString[pos]) {
    case '0':
      result[pos] = false;
      break;
    case '1':
      result[pos] = true;
      break;
    default:
      throw cms::Exception("BadConfig") <<"@SUB=AlignmentParameterSelector::decodeSelections"
                                        << selString << " must contain only '0' and '1'";
    }
  }

  return result;
}

//________________________________________________________________________________
unsigned int AlignmentParameterSelector::addAllDets(const std::vector<bool> &paramSel)
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
unsigned int AlignmentParameterSelector::addAllRods(const std::vector<bool> &paramSel)
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
unsigned int AlignmentParameterSelector::addAllLayers(const std::vector<bool> &paramSel)
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
unsigned int AlignmentParameterSelector::addAllAlignables(const std::vector<bool> &paramSel)
{
  unsigned int numAli = 0;

  numAli += this->addAllDets(paramSel);
  numAli += this->addAllRods(paramSel);
  numAli += this->addAllLayers(paramSel);
  numAli += this->add(theTracker->components(), paramSel);

  return numAli;
}
