/** \file AlignableSelector.cc
 *  \author Gero Flucke, Nov. 2006
 *
 *  $Date: 2006/10/20 13:24:55 $
 *  $Revision: 1.5 $
 *  (last update by $Author$)
 */

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignableSelector.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h" // for enums TID/TIB/etc.

#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/Phi.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//________________________________________________________________________________
AlignableSelector::AlignableSelector(AlignableTracker *aliTracker) :
  theTracker(aliTracker), theSelectedAlignables(), 
  theRangesEta(), theRangesPhi(), theRangesR(), theRangesZ(),
  theOnlyDS(false), theOnlySS(false), theSelLayers(false), theMinLayer(-1), theMaxLayer(999)
{
}

//________________________________________________________________________________
AlignableSelector::~AlignableSelector()
{
}

//________________________________________________________________________________
const std::vector<Alignable*>& AlignableSelector::selectedAlignables() const
{
  return theSelectedAlignables;
}

//________________________________________________________________________________
void AlignableSelector::clear()
{
  theSelectedAlignables.clear();

  theRangesEta.clear();
  theRangesPhi.clear();
  theRangesR.clear();
  theRangesZ.clear();
}

//________________________________________________________________________________
void AlignableSelector::setGeometryCuts(const edm::ParameterSet &pSet)
{

  theRangesEta = pSet.getParameter<std::vector<double> >("etaRanges");
  theRangesPhi = pSet.getParameter<std::vector<double> >("phiRanges");
  theRangesR   = pSet.getParameter<std::vector<double> >("rRanges"  );
  theRangesZ   = pSet.getParameter<std::vector<double> >("zRanges"  );
}

//________________________________________________________________________________
unsigned int AlignableSelector::addSelection(const std::string &name, const edm::ParameterSet &pSet)
{
  this->setGeometryCuts(pSet);
  return this->addSelection(name);
}

//________________________________________________________________________________
unsigned int AlignableSelector::addSelection(const std::string &name)
{

  unsigned int numAli = 0;

  if      (name == "AllDets")       numAli += this->addAllDets();
  else if (name == "AllRods")       numAli += this->addAllRods();
  else if (name == "AllLayers")     numAli += this->addAllLayers();
  else if (name == "AllComponents") numAli += this->add(theTracker->components());
  else if (name == "AllAlignables") numAli += this->addAllAlignables();
  //
  // TIB+TOB
  //
  else if (name == "BarrelRods")    numAli += this->add(theTracker->barrelRods());
  else if (name == "BarrelDets")    numAli += this->add(theTracker->barrelGeomDets());
  else if (name == "BarrelLayers")  numAli += this->add(theTracker->barrelLayers());
  else if (name == "BarrelDSRods") {
    theOnlyDS = true;
    numAli += this->add(theTracker->barrelRods());
    theOnlyDS = false;
  } else if (name == "BarrelSSRods") {
    theOnlySS = true;
    numAli += this->add(theTracker->barrelRods());
    theOnlySS = false;
  } else if (name == "BarrelDSLayers") { // new
    theOnlyDS = true;
    numAli += this->add(theTracker->barrelLayers());
    theOnlyDS = false;
  } else if (name == "BarrelSSLayers") { // new
    theOnlySS = true;
    numAli += this->add(theTracker->barrelLayers());
    theOnlySS = false;
  } else if (name == "TOBDSRods") { // new for CSA06Selection
    theOnlyDS = true; 
    numAli += this->add(theTracker->outerBarrelRods());
    theOnlyDS = false;
  } else if (name == "TOBSSRodsLayers15") { // new for CSA06Selection
    // FIXME: make Layers15 flexible
    theSelLayers = theOnlySS = true; 
    theMinLayer = 1;
    theMaxLayer = 5; //  TOB outermost layer (6) kept fixed
    numAli += this->add(theTracker->outerBarrelRods());
    theSelLayers = theOnlySS = false;
  } else if (name == "TIBDSDets") { // new for CSA06Selection
    theOnlyDS = true; 
    numAli += this->add(theTracker->innerBarrelGeomDets());
    theOnlyDS = false;
  } else if (name == "TIBSSDets") { // new for CSA06Selection
    theOnlySS = true; 
    numAli += this->add(theTracker->innerBarrelGeomDets());
    theOnlySS = false;
  }
  //
  // PXBarrel
  //
  else if (name == "PixelHalfBarrelDets") {
    numAli += this->add(theTracker->pixelHalfBarrelGeomDets());
  } else if (name == "PixelHalfBarrelLadders") {
    numAli += this->add(theTracker->pixelHalfBarrelLadders());
  } else if (name == "PixelHalfBarrelLayers") {
    numAli += this->add(theTracker->pixelHalfBarrelLayers());
  } else if (name == "PixelHalfBarrelLaddersLayers12") {
    // FIXME: make Layers12 flexible
    theSelLayers = true; 
    theMinLayer = 1;
    theMaxLayer = 2;
    numAli += this->add(theTracker->pixelHalfBarrelLadders());
    theSelLayers = false;
  }
  //
  // PXEndcap
  //
  else if (name == "PXECDets")      numAli += this->add(theTracker->pixelEndcapGeomDets());
  else if (name == "PXECPetals")    numAli += this->add(theTracker->pixelEndcapPetals());
  else if (name == "PXECLayers")    numAli += this->add(theTracker->pixelEndcapLayers());
  //
  // Pixel Barrel+endcap
  //
  else if (name == "PixelDets") {
    numAli += this->add(theTracker->pixelHalfBarrelGeomDets());
    numAli += this->add(theTracker->pixelEndcapGeomDets());
  } else if (name == "PixelRods") {
    numAli += this->add(theTracker->pixelHalfBarrelLadders());
    numAli += this->add(theTracker->pixelEndcapPetals());
  } else if (name == "PixelLayers") {
    numAli += this->add(theTracker->pixelHalfBarrelLayers());
    numAli += this->add(theTracker->pixelEndcapLayers());
  }
  //
  // TID
  //
  else if (name == "TIDLayers")     numAli += this->add(theTracker->TIDLayers());
  else if (name == "TIDRings")      numAli += this->add(theTracker->TIDRings());
  else if (name == "TIDDets")       numAli += this->add(theTracker->TIDGeomDets());
  //
  // TEC
  //
  else if (name == "TECDets")       numAli += this->add(theTracker->endcapGeomDets()); 
  else if (name == "TECPetals")     numAli += this->add(theTracker->endcapPetals());
  else if (name == "TECLayers")     numAli += this->add(theTracker->endcapLayers());
  //
  // StripEndcap (TID+TEC)
  //
  else if (name == "EndcapDets") {
    numAli += this->add(theTracker->TIDGeomDets());
    numAli += this->add(theTracker->endcapGeomDets()); 
  } else if (name == "EndcapPetals") {
    numAli += this->add(theTracker->TIDRings());
    numAli += this->add(theTracker->endcapPetals());
  } else if (name == "EndcapLayers") {
    numAli += this->add(theTracker->TIDLayers());
    numAli += this->add(theTracker->endcapLayers());
  }
  //
  // Strip Barrel+endcap
  //
  else if (name == "StripDets") {
    numAli += this->add(theTracker->barrelGeomDets());
    numAli += this->add(theTracker->TIDGeomDets());
    numAli += this->add(theTracker->endcapGeomDets()); 
  } else if (name == "StripRods") {
    numAli += this->add(theTracker->barrelRods());
    numAli += this->add(theTracker->TIDRings());
    numAli += this->add(theTracker->endcapPetals());
  } else if (name == "StripLayers") {
    numAli += this->add(theTracker->barrelLayers());
    numAli += this->add(theTracker->TIDLayers());
    numAli += this->add(theTracker->endcapLayers());
  }
  /*
  //
  // Custom scenarios from AlignmentParameterBuilder: replaced by subsequent selections in config
  //
  else if (name == "ScenarioA") {
    std::vector<bool> mysel(6,false);
    // pixel barrel dets x,y,z
    mysel[RigidBodyAlignmentParameters::dx]=true;
    mysel[RigidBodyAlignmentParameters::dy]=true;
    mysel[RigidBodyAlignmentParameters::dz]=true;
    numAli += this->add(theTracker->pixelHalfBarrelGeomDets(),mysel);
    // strip barrel double sided
    theOnlyDS = true;
    numAli += this->add(theTracker->barrelRods(),mysel);
    theOnlyDS = false;
    // strip barrel single sided
    mysel[RigidBodyAlignmentParameters::dy]=false;
    theOnlySS = true;
    numAli += this->add(theTracker->barrelRods(),mysel);
    theOnlySS = false;
  } else if (name == "ScenarioB") {
    std::vector<bool> mysel(6,false);
    // pixel barrel ladders x,y,z
    mysel[RigidBodyAlignmentParameters::dx]=true;
    mysel[RigidBodyAlignmentParameters::dy]=true;
    mysel[RigidBodyAlignmentParameters::dz]=true;
    numAli += this->add(theTracker->pixelHalfBarrelLadders(),mysel);
    // strip barrel layers double sided
    theOnlyDS = true;
    numAli += this->add(theTracker->barrelLayers(),mysel);
    theOnlyDS = false;
    // strip barrel layers single sided
    mysel[RigidBodyAlignmentParameters::dy]=false;
    theOnlySS = true;
    numAli += this->add(theTracker->barrelLayers(),mysel);
    theOnlySS = false;
  } else if (name == "CustomStripLayers") {
    std::vector<bool> mysel(6,false);
    mysel[RigidBodyAlignmentParameters::dx]=true;
    mysel[RigidBodyAlignmentParameters::dy]=true;
    mysel[RigidBodyAlignmentParameters::dz]=true;
    // strip barrel layers double sided
    theOnlyDS = true;
    numAli += this->add(theTracker->barrelLayers(),mysel);
    theOnlyDS = false;
    // strip barrel layers single sided
    mysel[RigidBodyAlignmentParameters::dz]=false;
    theOnlySS = true;
    numAli += this->add(theTracker->barrelLayers(),mysel);
    theOnlySS = false;
    // TID
    mysel[RigidBodyAlignmentParameters::dz]=true;
    numAli += this->add(theTracker->TIDLayers(),mysel);
    // TEC
    mysel[RigidBodyAlignmentParameters::dz]=false;
    numAli += this->add(theTracker->endcapLayers(),mysel);
  } else if (name == "CustomStripRods") {
    std::vector<bool> mysel(6,false);
    mysel[RigidBodyAlignmentParameters::dx]=true;
    mysel[RigidBodyAlignmentParameters::dy]=true;
    mysel[RigidBodyAlignmentParameters::dz]=true;
    // strip barrel layers double sided
    theOnlyDS = true;
    numAli += this->add(theTracker->barrelRods(),mysel);
    theOnlyDS = false;
    // strip barrel layers single sided
    mysel[RigidBodyAlignmentParameters::dy]=false;
    theOnlySS = true;
    numAli += this->add(theTracker->barrelRods(),mysel);
    theOnlySS = false;
    // TID
    mysel[RigidBodyAlignmentParameters::dy]=true;
    numAli += this->add(theTracker->TIDRings(),mysel);
    // TEC
    mysel[RigidBodyAlignmentParameters::dz]=false;
    numAli += this->add(theTracker->endcapPetals(),mysel);
  } else if (name == "CSA06Selection") {
    std::vector<bool> mysel(6,false);
    mysel[RigidBodyAlignmentParameters::dx]=true;
    mysel[RigidBodyAlignmentParameters::dy]=true;
    mysel[RigidBodyAlignmentParameters::dz]=true;
    mysel[RigidBodyAlignmentParameters::dalpha]=true;
    mysel[RigidBodyAlignmentParameters::dbeta]=true;
    mysel[RigidBodyAlignmentParameters::dgamma]=true;
//  TOB outermost layer (5) kept fixed
    theSelLayers=true; theMinLayer=1; theMaxLayer=5;
//  TOB rods double sided   
    theOnlyDS=true;
    add(theAlignableTracker->outerBarrelRods(),mysel);
    theOnlyDS=false;
// TOB rods single sided   
    mysel[RigidBodyAlignmentParameters::dy]=false;
    mysel[RigidBodyAlignmentParameters::dz]=false;
    theOnlySS=true;
    add(theAlignableTracker->outerBarrelRods(),mysel);
    theOnlySS=false;
    mysel[RigidBodyAlignmentParameters::dy]=true;
    mysel[RigidBodyAlignmentParameters::dz]=true;
//
    theSelLayers=false; 
 // TIB dets double sided   
    theOnlyDS=true;
    add(theAlignableTracker->innerBarrelGeomDets(),mysel);
    theOnlyDS=false;
 // TIB dets single sided   
    mysel[RigidBodyAlignmentParameters::dy]=false;
    mysel[RigidBodyAlignmentParameters::dz]=false;
    theOnlySS=true;
    add(theAlignableTracker->innerBarrelGeomDets(),mysel);
    theOnlySS=false;
  }
  */
  // not found!
  else { // @SUB-syntax is not supported by exception, but anyway useful information... 
    throw cms::Exception("BadConfig") <<"@SUB=TrackerAlignmentSelector::addSelection"
				      << ": Selection '" << name << "' invalid!";
  }
  
  return numAli;
}

//________________________________________________________________________________
unsigned int AlignableSelector::add(const std::vector<Alignable*> &alignables)
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
      ++numAli;
    }
  }

  return numAli;
}

//_________________________________________________________________________
bool AlignableSelector::outsideRanges(const Alignable *alignable) const
{

  const GlobalPoint position(alignable->globalPosition());

  if (!theRangesEta.empty() && !this->insideRanges((position.eta()), theRangesEta)) return true;
  if (!theRangesPhi.empty() && !this->insideRanges((position.phi()), theRangesPhi,true))return true;
  if (!theRangesR.empty()   && !this->insideRanges((position.perp()),theRangesR)) return true;
  if (!theRangesZ.empty()   && !this->insideRanges((position.z()),   theRangesZ)) return true;

  return false;
}

//_________________________________________________________________________
bool AlignableSelector::insideRanges(double value, const std::vector<double> &ranges,
                                            bool isPhi) const
{
  // might become templated on <double> ?

  if (ranges.size()%2 != 0) {
    cms::Exception("BadConfig") << "@SUB=AlignableSelector::insideRanges" 
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

//________________________________________________________________________________
unsigned int AlignableSelector::addAllDets()
{
  unsigned int numAli = 0;

  numAli += this->add(theTracker->barrelGeomDets());          // TIB+TOB
  numAli += this->add(theTracker->endcapGeomDets());          // TEC
  numAli += this->add(theTracker->TIDGeomDets());             // TID
  numAli += this->add(theTracker->pixelHalfBarrelGeomDets()); // PixelBarrel
  numAli += this->add(theTracker->pixelEndcapGeomDets());     // PixelEndcap

  return numAli;
}

//________________________________________________________________________________
unsigned int AlignableSelector::addAllRods()
{
  unsigned int numAli = 0;

  numAli += this->add(theTracker->barrelRods());             // TIB+TOB    
  numAli += this->add(theTracker->pixelHalfBarrelLadders()); // PixelBarrel
  numAli += this->add(theTracker->endcapPetals());           // TEC        
  numAli += this->add(theTracker->TIDRings());               // TID        
  numAli += this->add(theTracker->pixelEndcapPetals());      // PixelEndcap

  return numAli;
}

//________________________________________________________________________________
unsigned int AlignableSelector::addAllLayers()
{
  unsigned int numAli = 0;

  numAli += this->add(theTracker->barrelLayers());          // TIB+TOB    
  numAli += this->add(theTracker->pixelHalfBarrelLayers()); // PixelBarrel
  numAli += this->add(theTracker->endcapLayers());          // TEC
  numAli += this->add(theTracker->TIDLayers());             // TID
  numAli += this->add(theTracker->pixelEndcapLayers());     // PixelEndcap

  return numAli;
}

//________________________________________________________________________________
unsigned int AlignableSelector::addAllAlignables()
{
  unsigned int numAli = 0;

  numAli += this->addAllDets();
  numAli += this->addAllRods();
  numAli += this->addAllLayers();
  numAli += this->add(theTracker->components());

  return numAli;
}
