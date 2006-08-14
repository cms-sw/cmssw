#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"

#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"
#include "Alignment/CommonAlignmentParametrization/interface/CompositeRigidBodyAlignmentParameters.h"

// This class's header

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterBuilder.h"


//__________________________________________________________________________________________________
AlignmentParameterBuilder::AlignmentParameterBuilder( AlignableTracker* alignableTracker )
{

  theAlignableTracker = alignableTracker;

  theTrackerAlignableId = new TrackerAlignableId();
  theOnlyDS=false;
  theOnlySS=false;
  theSelLayers=false;
  theMinLayer=-1;
  theMaxLayer=999;

}

//__________________________________________________________________________________________________
void AlignmentParameterBuilder::addSelection( std::string name, std::vector<bool> sel )
{

  edm::LogWarning("Alignment") << "[AlignmentParameterBuilder] Called for selection >" << name<<"<";

  if      (name == "AllDets")       addAllDets(sel);
  else if (name == "AllRods")       addAllRods(sel);
  else if (name == "AllLayers")     addAllLayers(sel);
  else if (name == "AllComponents") addAllComponents(sel);
  else if (name == "AllAlignables") addAllAlignables(sel);

  // TIB+TOB
  else if (name == "BarrelRods")    add(theAlignableTracker->barrelRods(),sel);
  else if (name == "BarrelDets")    add(theAlignableTracker->barrelGeomDets(),sel);
  else if (name == "BarrelLayers")  add(theAlignableTracker->barrelLayers(),sel);

  else if (name == "BarrelDSRods")  
	{
	  theOnlyDS = true;
	  add( theAlignableTracker->barrelRods(), sel );
	  theOnlyDS = false;
	}

  // PXBarrel
  else if (name == "PixelHalfBarrelDets")
	add(theAlignableTracker->pixelHalfBarrelGeomDets(),sel);
  else if (name == "PixelHalfBarrelLadders") 
	add(theAlignableTracker->pixelHalfBarrelLadders(),sel);
  else if (name == "PixelHalfBarrelLayers")  
	add(theAlignableTracker->pixelHalfBarrelLayers(),sel);

  else if (name == "PixelHalfBarrelLaddersLayers12") {
    theSelLayers=true; theMinLayer=1; theMaxLayer=2;
    add(theAlignableTracker->pixelHalfBarrelLadders(),sel);
  }


  // PXEndcap
  else if (name == "PXECDets")      add(theAlignableTracker->pixelEndcapGeomDets(),sel);
  else if (name == "PXECPetals")    add(theAlignableTracker->pixelEndcapPetals(),sel);
  else if (name == "PXECLayers")    add(theAlignableTracker->pixelEndcapLayers(),sel);

  // Pixel Barrel+endcap
  else if (name == "PixelDets") {
    add(theAlignableTracker->pixelHalfBarrelGeomDets(),sel);
    add(theAlignableTracker->pixelEndcapGeomDets(),sel);
  }
  else if (name == "PixelRods") {
    add(theAlignableTracker->pixelHalfBarrelLadders(),sel);
    add(theAlignableTracker->pixelEndcapPetals(),sel);
  }
  else if (name == "PixelLayers") {
    add(theAlignableTracker->pixelHalfBarrelLayers(),sel);
    add(theAlignableTracker->pixelEndcapLayers(),sel);
  }

  // TID
  else if (name == "TIDLayers")     add(theAlignableTracker->TIDLayers(),sel);
  else if (name == "TIDRings")      add(theAlignableTracker->TIDRings(),sel);
  else if (name == "TIDDets")       add(theAlignableTracker->TIDGeomDets(),sel);

  // TEC
  else if (name == "TECDets")       add(theAlignableTracker->endcapGeomDets(),sel); 
  else if (name == "TECPetals")     add(theAlignableTracker->endcapPetals(),sel);
  else if (name == "TECLayers")     add(theAlignableTracker->endcapLayers(),sel);

  // StripEndcap (TID+TEC)
  else if (name == "EndcapDets") {
    add(theAlignableTracker->TIDGeomDets(),sel);
    add(theAlignableTracker->endcapGeomDets(),sel); 
  }
  else if (name == "EndcapPetals") {
    add(theAlignableTracker->TIDRings(),sel);
    add(theAlignableTracker->endcapPetals(),sel);
  }
  else if (name == "EndcapLayers") {
    add(theAlignableTracker->TIDLayers(),sel);
    add(theAlignableTracker->endcapLayers(),sel);
  }

  // Strip Barrel+endcap
  else if (name == "StripDets") {
    add(theAlignableTracker->barrelGeomDets(),sel);
    add(theAlignableTracker->TIDGeomDets(),sel);
    add(theAlignableTracker->endcapGeomDets(),sel); 
  }
  else if (name == "StripRods") {
    add(theAlignableTracker->barrelRods(),sel);
    add(theAlignableTracker->TIDRings(),sel);
    add(theAlignableTracker->endcapPetals(),sel);
  }
  else if (name == "StripLayers") {
    add(theAlignableTracker->barrelLayers(),sel);
    add(theAlignableTracker->TIDLayers(),sel);
    add(theAlignableTracker->endcapLayers(),sel);
  }


  // Custom scenarios

  else if (name == "ScenarioA") {
	std::vector<bool> mysel(6,false);
    // pixel barrel dets x,y,z
    mysel[RigidBodyAlignmentParameters::dx]=true;
    mysel[RigidBodyAlignmentParameters::dy]=true;
    mysel[RigidBodyAlignmentParameters::dz]=true;
    add(theAlignableTracker->pixelHalfBarrelGeomDets(),mysel);
    // strip barrel double sided
    theOnlyDS = true;
    add(theAlignableTracker->barrelRods(),mysel);
    theOnlyDS = false;
    // strip barrel single sided
    mysel[RigidBodyAlignmentParameters::dy]=false;
    theOnlySS = true;
    add(theAlignableTracker->barrelRods(),mysel);
    theOnlySS = false;
  }

  else if (name == "ScenarioB") {
	std::vector<bool> mysel(6,false);
    // pixel barrel ladders x,y,z
    mysel[RigidBodyAlignmentParameters::dx]=true;
    mysel[RigidBodyAlignmentParameters::dy]=true;
    mysel[RigidBodyAlignmentParameters::dz]=true;
    add(theAlignableTracker->pixelHalfBarrelLadders(),mysel);
    // strip barrel layers double sided
    theOnlyDS = true;
    add(theAlignableTracker->barrelLayers(),mysel);
    theOnlyDS = false;
    // strip barrel layers single sided
    mysel[RigidBodyAlignmentParameters::dy]=false;
    theOnlySS = true;
    add(theAlignableTracker->barrelLayers(),mysel);
    theOnlySS = false;
  }


  else if (name == "CustomStripLayers") {
	std::vector<bool> mysel(6,false);
    mysel[RigidBodyAlignmentParameters::dx]=true;
    mysel[RigidBodyAlignmentParameters::dy]=true;
    mysel[RigidBodyAlignmentParameters::dz]=true;
    // strip barrel layers double sided
    theOnlyDS = true;
    add(theAlignableTracker->barrelLayers(),mysel);
    theOnlyDS = false;
    // strip barrel layers single sided
    mysel[RigidBodyAlignmentParameters::dz]=false;
    theOnlySS = true;
    add(theAlignableTracker->barrelLayers(),mysel);
    theOnlySS = false;
    // TID
    mysel[RigidBodyAlignmentParameters::dz]=true;
    add(theAlignableTracker->TIDLayers(),mysel);
    // TEC
    mysel[RigidBodyAlignmentParameters::dz]=false;
    add(theAlignableTracker->endcapLayers(),mysel);
  }

  else if (name == "CustomStripRods") {
	std::vector<bool> mysel(6,false);
    mysel[RigidBodyAlignmentParameters::dx]=true;
    mysel[RigidBodyAlignmentParameters::dy]=true;
    mysel[RigidBodyAlignmentParameters::dz]=true;
    // strip barrel layers double sided
    theOnlyDS = true;
    add(theAlignableTracker->barrelRods(),mysel);
    theOnlyDS = false;
    // strip barrel layers single sided
    mysel[RigidBodyAlignmentParameters::dy]=false;
    theOnlySS = true;
    add(theAlignableTracker->barrelRods(),mysel);
    theOnlySS = false;
    // TID
    mysel[RigidBodyAlignmentParameters::dy]=true;
    add(theAlignableTracker->TIDRings(),mysel);
    // TEC
    mysel[RigidBodyAlignmentParameters::dz]=false;
    add(theAlignableTracker->endcapPetals(),mysel);
  }

  else 
    edm::LogError("BadConfig")<<"[AlignmentParameterBuilder] Selection invalid!";

  edm::LogInfo("Warning") << "[AlignmentParameterBuilder] Added " 
    << theAlignables.size()<< " alignables in total";

}


//__________________________________________________________________________________________________
void AlignmentParameterBuilder::addAllDets( std::vector<bool>sel )
{

  add(theAlignableTracker->barrelGeomDets(),sel);          // TIB+TOB
  add(theAlignableTracker->endcapGeomDets(),sel);          // TEC
  add(theAlignableTracker->TIDGeomDets(),sel);             // TID
  add(theAlignableTracker->pixelHalfBarrelGeomDets(),sel); // PixelBarrel
  add(theAlignableTracker->pixelEndcapGeomDets(),sel);     // PixelEndcap

  edm::LogInfo("Alignment") << "Initialized for "
											<< theAlignables.size() << " dets";
}


//__________________________________________________________________________________________________
void AlignmentParameterBuilder::addAllRods(std::vector<bool>sel)
{
  add(theAlignableTracker->barrelRods(),sel);
  add(theAlignableTracker->pixelHalfBarrelLadders(),sel);
  add(theAlignableTracker->endcapPetals(),sel);
  add(theAlignableTracker->TIDRings(),sel);
  add(theAlignableTracker->pixelEndcapPetals(),sel);

  edm::LogInfo("Alignment") << "Initialized for "
											<< theAlignables.size() << " rods";
}


//__________________________________________________________________________________________________
void AlignmentParameterBuilder::addAllLayers(std::vector<bool>sel)
{
  add(theAlignableTracker->barrelLayers(),sel);
  add(theAlignableTracker->pixelHalfBarrelLayers(),sel);
  add(theAlignableTracker->endcapLayers(),sel);
  add(theAlignableTracker->TIDLayers(),sel);
  add(theAlignableTracker->pixelEndcapLayers(),sel);

  edm::LogInfo("Alignment") << "Initialized for "
											<< theAlignables.size() << " layers";

}


//__________________________________________________________________________________________________
void AlignmentParameterBuilder::addAllComponents(std::vector<bool>sel)
{
  add(theAlignableTracker->components(),sel);
  edm::LogInfo("Alignment") << "Initialized for "
											<< theAlignables.size() 
											<< " Components (HalfBarrel/Endcap)";
}


//__________________________________________________________________________________________________
void AlignmentParameterBuilder::addAllAlignables(std::vector<bool>sel)
{

  add(theAlignableTracker->barrelGeomDets(),sel);          
  add(theAlignableTracker->endcapGeomDets(),sel);          
  add(theAlignableTracker->TIDGeomDets(),sel);             
  add(theAlignableTracker->pixelHalfBarrelGeomDets(),sel); 
  add(theAlignableTracker->pixelEndcapGeomDets(),sel);     

  add(theAlignableTracker->barrelRods(),sel);
  add(theAlignableTracker->pixelHalfBarrelLadders(),sel);
  add(theAlignableTracker->endcapPetals(),sel);
  add(theAlignableTracker->TIDRings(),sel);
  add(theAlignableTracker->pixelEndcapPetals(),sel);

  add(theAlignableTracker->barrelLayers(),sel);
  add(theAlignableTracker->pixelHalfBarrelLayers(),sel);
  add(theAlignableTracker->endcapLayers(),sel);
  add(theAlignableTracker->TIDLayers(),sel);
  add(theAlignableTracker->pixelEndcapLayers(),sel);

  add(theAlignableTracker->components(),sel);


  edm::LogInfo("Alignment") << "Initialized for "
											<< theAlignables.size() 
											<< " Components (HalfBarrel/Endcap)";

}


//__________________________________________________________________________________________________
void AlignmentParameterBuilder::add( const std::vector<Alignable*>& alignables, 
									 std::vector<bool> sel )
{

  int num_adu = 0;
  int num_det = 0;
  int num_hlo = 0;

  // loop on Alignable objects
  for ( std::vector<Alignable*>::const_iterator ia=alignables.begin();
        ia!=alignables.end();  ia++ ) {
    Alignable* ali=(*ia);

    // select on single/double sided barrel layers
	std::pair<int,int> tl=theTrackerAlignableId->typeAndLayerFromAlignable( ali );
    int type = tl.first;
    int layer = tl.second;

    bool keep=true;
    if (theOnlySS) // only single sided
      if ( (abs(type)==3 || abs(type)==5) && layer<2 ) 
		keep=false;

    if (theOnlyDS) // only double sided
      if ( (abs(type)==3 || abs(type)==5) && layer>1 )
		keep=false;

    // reject layers
    if ( theSelLayers && (layer<theMinLayer || layer>theMaxLayer) )  
	  keep=false;


    if (keep) {

	  AlgebraicVector par(6,0);
	  AlgebraicSymMatrix cov(6,0);

	  AlignableDet* alidet = dynamic_cast<AlignableDet*>(ali);
	  if (alidet !=0) { // alignable Det
		RigidBodyAlignmentParameters* dap = 
		  new RigidBodyAlignmentParameters(ali,par,cov,sel);
        ali->setAlignmentParameters(dap);
		num_det++;
	  }
	  else { // higher level object
		CompositeRigidBodyAlignmentParameters* dap = 
		  new CompositeRigidBodyAlignmentParameters(ali,par,cov,sel);
		ali->setAlignmentParameters(dap);
		num_hlo++;
	  }

	  theAlignables.push_back(ali);
	  num_adu++;

    }
  }

  edm::LogWarning("Alignment") << "Added " << num_adu 
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
    << " alignables, so that " << theAlignables.size() << " alignables left";
  
}

