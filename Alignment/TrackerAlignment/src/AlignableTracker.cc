#include <sys/time.h>

// Framework
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


// Geometry interface
#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerStringToEnum.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"

// Tracker components
#include "Alignment/TrackerAlignment/interface/AlignableTrackerHalfBarrel.h"
#include "Alignment/TrackerAlignment/interface/AlignableTrackerBarrelLayer.h"
#include "Alignment/TrackerAlignment/interface/AlignablePixelHalfBarrel.h"
#include "Alignment/TrackerAlignment/interface/AlignablePixelHalfBarrelLayer.h"
#include "Alignment/TrackerAlignment/interface/AlignableTrackerEndcap.h"
#include "Alignment/TrackerAlignment/interface/AlignableTID.h"
#include "Alignment/TrackerAlignment/interface/AlignableTrackerCompositeBuilder.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/Alignment/interface/AlignmentSorter.h"

#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"


//--------------------------------------------------------------------------------------------------
AlignableTracker::AlignableTracker( const GeometricDet* geometricDet, 
									const TrackerGeometry* trackerGeometry )
{

  // The XML geometry is accessed through the GeometricDet class.
  // Since this geometry does not contain exactly the same structures we
  // need in alignment, we implement the alignable hierarchy here.
  //

  edm::LogInfo("AlignableTracker") << "Constructing alignable objects"; 
  
  theTrackerGeometry = trackerGeometry;

  // Check that we really have the full tracker
  if ( geometricDet->type() != GeometricDet::Tracker )
	throw cms::Exception("LogicError")
	  << "Wrong type for argument 1 of AlignableTracker constructor"
	  << std::endl << "Type is " << geometricDet->type() << ", should be "
	  << GeometricDet::Tracker << " (geometric tracker)";
  
  // Get container of tracker components
  _DetContainer m_components = geometricDet->components();

  buildTPE( theTrackerGeometry->detsPXF() ); // build forward pixel

  // Loop over main components and call sub-builders
  for ( _DetContainer::iterator iDet = m_components.begin();
		iDet != m_components.end(); iDet++ )
    {
	  
      // List of known sub-structures of the tracker
      // See GeometricDet::GDEnumType
      switch ( (*iDet)->type() )
	{
	case (GeometricDet::PixelBarrel) : buildTPB( *iDet ); break;
	case (GeometricDet::PixelEndCap) : /* buildTPE( *iDet ); */ break; // do nothing, already built above
	case (GeometricDet::TIB) : buildTIB( *iDet ); break;
	case (GeometricDet::TID) : buildTID( *iDet ); break;
	case (GeometricDet::TOB) : buildTOB( *iDet ); break;
	case (GeometricDet::TEC) : buildTEC( *iDet ); break;
	default: ;
	  throw cms::Exception("LogicError") 
	    << "Unknown detector type: " << (*iDet)->type();
	}
      
    }

  // Set links to mothers recursively
  recursiveSetMothers( this );
  
  edm::LogInfo("AlignableTracker") << "Constructing alignable objects DONE"; 


}


//--------------------------------------------------------------------------------------------------
void AlignableTracker::buildTOB( const GeometricDet* navigator )
{

  // The assumption here is that geometric rods are ordered by layer,
  // and layers are ordered by layer number, alternating positive and negative z.
  // Thus, when the Z coordinate of the rods goes from >0 to <0 (and vice-versa),
  // a new layer is started (see implementation below). Same in TIB and TOB.
  // P.S. To make it more stable, a check on the detId is also performed...
  
  LogDebug("Position") << "Constructing TOB"; 
  
  // First retrieve and build alignable rods
  _DetContainer m_geometricRods = this->getAllComponents( navigator, GeometricDet::rod );
  AlignableTrackerCompositeBuilder<AlignableTrackerRod> builder;

  // Loop on geometric rods and create AlignableTrackerRods and AlignableTrackerLayers
  std::vector<AlignableTrackerRod*> m_tmpRods;
  std::vector<AlignableTrackerBarrelLayer*> m_forwardLayers;
  std::vector<AlignableTrackerBarrelLayer*> m_backwardLayers;
  float curZ = 0;
  TOBDetId curDetId;
  for ( _DetContainer::iterator iRod = m_geometricRods.begin(); 
		iRod != m_geometricRods.end(); iRod++ )
	{
	  AlignableTrackerRod* tmpRod = builder.buildAlignable( (*iRod)->components(), 
															theTrackerGeometry );
	  
	  // When Z changes sign, we start a new layer...
	  TOBDetId tmpDetId( (*iRod)->geographicalID() );
	  if ( m_tmpRods.size() > 0 && 
		   ( tmpRod->globalPosition().z()*curZ<0 || tmpDetId.layer() != curDetId.layer() ) )
		{
		  AlignableTrackerBarrelLayer* tmpLayer = new AlignableTrackerBarrelLayer( m_tmpRods );
		  if ( tmpLayer->globalPosition().z() > 0 ) m_forwardLayers.push_back( tmpLayer );
		  else m_backwardLayers.push_back( tmpLayer );
		  
		  m_tmpRods.clear();
		}

	  curZ = tmpRod->globalPosition().z();
	  curDetId = TOBDetId( (*iRod)->geographicalID() );
	  m_tmpRods.push_back( tmpRod );
	}
  // Special treatment for last layer (outside loop)
  if ( m_tmpRods.size() != 0 )
	{
	  AlignableTrackerBarrelLayer* tmpLayer = new AlignableTrackerBarrelLayer( m_tmpRods );
	  if ( tmpLayer->globalPosition().z() > 0 ) m_forwardLayers.push_back( tmpLayer );
	  else m_backwardLayers.push_back( tmpLayer );
	}

  
  // And now, finally create the half barrels
  AlignableTrackerHalfBarrel* forwardHalf  = new AlignableTrackerHalfBarrel( m_forwardLayers ); 
  AlignableTrackerHalfBarrel* backwardHalf = new AlignableTrackerHalfBarrel( m_backwardLayers );

  // Store these products
  theOuterHalfBarrels.push_back( forwardHalf );
  theOuterHalfBarrels.push_back( backwardHalf );

  addComponent( forwardHalf );
  addComponent( backwardHalf );
}


//--------------------------------------------------------------------------------------------------
void AlignableTracker::buildTIB( const GeometricDet* navigator )
{

  // The assumption here is that geometric rods are ordered by layer,
  // and layers are ordered by layer number, alternating positive and negative z.
  // Thus, when the Z coordinate of the rods goes from >0 to <0 (and vice-versa),
  // a new layer is started (see implementation below). Same in TIB and TOB.
  // P.S. To make it more stable, a check on the detId is also performed...

  LogDebug("AlignableTracker") << "Constructing TIB"; 

  // First retrieve and build alignable rods (TIB rods are Strings...)
  _DetContainer m_geometricRods = this->getAllComponents( navigator, GeometricDet::strng );
  AlignableTrackerCompositeBuilder<AlignableTrackerRod> builder;

  // Loop on geometric rods and create AlignableTrackerRods and AlignableTrackerLayers
  std::vector<AlignableTrackerRod*> m_tmpRods;
  std::vector<AlignableTrackerBarrelLayer*> m_forwardLayers;
  std::vector<AlignableTrackerBarrelLayer*> m_backwardLayers;
  float curZ = 0;
  TIBDetId curDetId;
  for ( _DetContainer::iterator iRod = m_geometricRods.begin(); 
		iRod != m_geometricRods.end(); iRod++ )
	{
	  AlignableTrackerRod* tmpRod = builder.buildAlignable( (*iRod)->components(), 
															theTrackerGeometry );

	  // When Z changes, we start a new layer...
	  TIBDetId tmpDetId( (*iRod)->geographicalID() );
	  if ( m_tmpRods.size() > 0 && 
		   ( tmpRod->globalPosition().z()*curZ<0 || tmpDetId.layer() != curDetId.layer() ) )
		{
		  AlignableTrackerBarrelLayer* tmpLayer = new AlignableTrackerBarrelLayer( m_tmpRods );
		  if ( tmpLayer->globalPosition().z() > 0 ) m_forwardLayers.push_back( tmpLayer );
		  else m_backwardLayers.push_back( tmpLayer );
		  
		  m_tmpRods.clear();
		}

	  curZ = tmpRod->globalPosition().z();
	  curDetId = TIBDetId( (*iRod)->geographicalID() );
	  m_tmpRods.push_back( tmpRod );
	}
  // Special treatment for last layer (outside loop)
  if ( m_tmpRods.size() != 0 )
	{
	  AlignableTrackerBarrelLayer* tmpLayer = new AlignableTrackerBarrelLayer( m_tmpRods );
	  if ( tmpLayer->globalPosition().z() > 0 ) m_forwardLayers.push_back( tmpLayer );
	  else m_backwardLayers.push_back( tmpLayer );
	}
  
  // And now, finally create the half barrels
  AlignableTrackerHalfBarrel* forwardHalf  = new AlignableTrackerHalfBarrel( m_forwardLayers ); 
  AlignableTrackerHalfBarrel* backwardHalf = new AlignableTrackerHalfBarrel( m_backwardLayers );

  // Store these products
  theInnerHalfBarrels.push_back( forwardHalf );
  theInnerHalfBarrels.push_back( backwardHalf );

  addComponent( forwardHalf );
  addComponent( backwardHalf );

}


//--------------------------------------------------------------------------------------------------
void AlignableTracker::buildTID( const GeometricDet* navigator )
{

  // The assumption here is that each TID layer is made of 3 consecutive TID rings.

  LogDebug("AlignableTracker") << "Constructing TID"; 
  
  // First retrieve and build alignable rings
  _DetContainer m_geometricRings = this->getAllComponents( navigator, GeometricDet::ring );
  AlignableTrackerCompositeBuilder<AlignableTIDRing> builder;

  // Loop on geometric rings and create AlignableRings and AlignableLayers
  std::vector<AlignableTIDRing*> m_tmpRings;
  std::vector<AlignableTIDLayer*> m_forwardLayers;
  std::vector<AlignableTIDLayer*> m_backwardLayers;
  int nRings = 0;
  for ( _DetContainer::iterator iRing = m_geometricRings.begin(); 
		iRing != m_geometricRings.end(); iRing++ )
	{
	  nRings++;
	  AlignableTIDRing* tmpRing = builder.buildAlignable( (*iRing)->components(), 
														  theTrackerGeometry );
	  m_tmpRings.push_back( tmpRing );

	  if ( nRings == 3 )
		{
		  AlignableTIDLayer* tmpLayer = new AlignableTIDLayer( m_tmpRings );
		  if ( tmpLayer->globalPosition().z() > 0 ) m_forwardLayers.push_back( tmpLayer );
		  else m_backwardLayers.push_back( tmpLayer );
		  
		  m_tmpRings.clear();
		  nRings = 0;
		}
	}


  // And now, finally create the TID (routine is called once for each endcap)
  AlignableTID* m_TID;
  if ( m_forwardLayers.size() ) m_TID = new AlignableTID( m_forwardLayers ); 
  else m_TID = new AlignableTID( m_backwardLayers );

  // Store these products (forward first)
  if ( m_forwardLayers.size() ) theTIDs.insert( theTIDs.begin(), m_TID );
  else theTIDs.push_back( m_TID );

  addComponent( m_TID );

}



//--------------------------------------------------------------------------------------------------
void AlignableTracker::buildTEC( const GeometricDet* navigator )
{

  // The TEC rings contain the DetUnits, but they are not
  // implemented in the alignable hierarchy. So we collect the petals,
  // and build them into layers. 
  // Note: This routine is called twice (once for forward, once for backward EC)
  
  LogDebug("AlignableTracker") << "Constructing TEC"; 


  _DetContainer m_geometricWheels = this->getAllComponents( navigator, GeometricDet::wheel );
  AlignableTrackerCompositeBuilder<AlignableTrackerPetal> builder;

  // Loop on geometric wheels and create AlignableTrackerPetals and AlignableTrackerEndcapLayers
  std::vector<AlignableTrackerPetal*> m_tmpPetals;
  std::vector<AlignableTrackerEndcapLayer*> m_forwardLayers;
  std::vector<AlignableTrackerEndcapLayer*> m_backwardLayers;
  for ( _DetContainer::iterator iWheel = m_geometricWheels.begin(); 
		iWheel != m_geometricWheels.end(); iWheel++ ) 
	{
	  
	  _DetContainer m_geometricPetals = (*iWheel)->components();
	  std::vector<AlignableTrackerPetal*> m_tmpPetals;

	  for ( _DetContainer::iterator iPetal = m_geometricPetals.begin(); 
			iPetal != m_geometricPetals.end(); iPetal++ )
		{
		  AlignableTrackerPetal* tmpPetal = builder.buildAlignable( (*iPetal)->components(), 
																	theTrackerGeometry );		  
		  m_tmpPetals.push_back( tmpPetal );
		  
		}

	  AlignableTrackerEndcapLayer* tmpLayer = new AlignableTrackerEndcapLayer( m_tmpPetals );
	  if ( tmpLayer->globalPosition().z() > 0 ) m_forwardLayers.push_back( tmpLayer );
	  else m_backwardLayers.push_back( tmpLayer );

	}
  
  // And now, finally create the endcap (routine is called once for each endcap)
  AlignableTrackerEndcap* m_Endcap;
  if ( m_forwardLayers.size() ) m_Endcap = new AlignableTrackerEndcap( m_forwardLayers ); 
  else m_Endcap = new AlignableTrackerEndcap( m_backwardLayers );

  // Store these products (forward first)
  if ( m_forwardLayers.size() ) theEndcaps.insert( theEndcaps.begin(), m_Endcap );
  else theEndcaps.push_back( m_Endcap );

  addComponent( m_Endcap );
  
}


//--------------------------------------------------------------------------------------------------
void AlignableTracker::buildTPB( const GeometricDet* navigator )
{
  // Ladders in layers are not correctly ordered for our purpose.
  // This makes it a little more difficult: use two levels of GeometricDets
  // (rods and layers) and assign rods to layers according to x position.
  // Note: There are no AlignableLadders; AlignableTrackerRod is used instead.

  LogDebug("AlignableTracker") << "Constructing TPB"; 

  // First retrieve and build alignable rods (Pixel rods are ladders...)
  _DetContainer m_geometricLayers = this->getAllComponents( navigator, GeometricDet::layer );
  AlignableTrackerCompositeBuilder<AlignableTrackerRod> builder;

  // Loop on geometric layers
  std::vector<AlignablePixelHalfBarrelLayer*> m_rightLayers;
  std::vector<AlignablePixelHalfBarrelLayer*> m_leftLayers;

  for ( _DetContainer::iterator iLayer = m_geometricLayers.begin(); 
		iLayer != m_geometricLayers.end(); iLayer++ )
	{

	  _DetContainer m_geometricRods = (*iLayer)->components();
	  std::vector<AlignableTrackerRod*> m_tmpPositiveRods;
	  std::vector<AlignableTrackerRod*> m_tmpNegativeRods;

	  // Loop on geometric rods to create alignable rods and layers
	  for ( _DetContainer::iterator iRod = m_geometricRods.begin();
			  iRod != m_geometricRods.end(); iRod++ )
		{
		  
		  AlignableTrackerRod* tmpRod = builder.buildAlignable( (*iRod)->components(), 
																theTrackerGeometry );		  
		  // Assign rod according to X
		  if ( tmpRod->globalPosition().x() > 0 ) m_tmpPositiveRods.push_back( tmpRod );
		  else m_tmpNegativeRods.push_back( tmpRod );
		}

	  // Now create layers and start with new geometric layer
	  m_rightLayers.push_back( new AlignablePixelHalfBarrelLayer( m_tmpPositiveRods ) );
	  m_leftLayers.push_back( new AlignablePixelHalfBarrelLayer( m_tmpNegativeRods ) );

	  m_tmpPositiveRods.clear();
	  m_tmpNegativeRods.clear();

	}
  
  // And now, finally create the half barrels
  AlignablePixelHalfBarrel* rightHalf  = new AlignablePixelHalfBarrel( m_rightLayers ); 
  AlignablePixelHalfBarrel* leftHalf = new AlignablePixelHalfBarrel( m_leftLayers );

  // Store these products
  thePixelHalfBarrels.push_back( rightHalf );
  thePixelHalfBarrels.push_back( leftHalf );

  addComponent( rightHalf );
  addComponent( leftHalf );

}

void AlignableTracker::buildTPE( const TrackerGeometry::DetContainer& dets )
{

  static const unsigned int maxPanel    = 2;  // max no. of panel per blade
  static const unsigned int maxBlade    = 24; // max no. of blade per disk
  static const unsigned int maxDisk     = 3;  // max no. of disk per cylinder
  static const unsigned int maxCylinder = 2;  // 2 half cylinders per endcap
  static const unsigned int maxEndcap   = 2;  // 2 endcaps in forward pixel

  LogDebug("AlignableTracker") << "Constructing TPE"; 

  timeval t0;
  gettimeofday(&t0, 0);
  LogDebug("AlignableTracker") << "t0 = " << t0.tv_usec << " us ";

// In order not to depend on the order of the input DetContainer,
// we define flags to indicate the existence of a structure;
// 0 if it hasn't been created.
// We use arrays instead of maps for speed.

  Alignable* newPanel[maxEndcap * maxDisk * maxBlade * maxPanel] = {0};
  Alignable* newBlade[maxEndcap * maxDisk * maxBlade] = {0};
  Alignable* newHalfDisk[maxEndcap * maxCylinder * maxDisk] = {0};
  Alignable* newHalfCylinder[maxEndcap * maxCylinder] = {0};
  Alignable* newEndcap[maxEndcap] = {0};

// Structures in the forward pixel hierarchy

  Alignables& sensors       = theTracker["pixelEndcapSensors"];
  Alignables& panels        = theTracker["pixelEndcapPanels"];
  Alignables& blades        = theTracker["pixelEndcapBlades"];
  Alignables& halfDisks     = theTracker["pixelEndcapHalfDisks"];
  Alignables& halfCylinders = theTracker["pixelEndcapHalfCylinders"];
  Alignables& endcaps       = theTracker["pixelEndcaps"];

// Build sensors

  unsigned int nDet = dets.size();

  sensors.resize(nDet, 0);

  for (unsigned int i = 0; i < nDet; ++i)
  {
    sensors[i] = new AlignableDet(dets[i]);
  }

// Build panels

  unsigned int nSensor = sensors.size();

  for (unsigned int i = 0; i < nSensor; ++i)
  {
    Alignable* sensor = sensors[i];

    PXFDetId id = sensor->id();

    unsigned int e = id.side() - 1;
    unsigned int d = e * maxDisk  + id.disk()  - 1;
    unsigned int b = d * maxBlade + id.blade() - 1;
    unsigned int p = b * maxPanel + id.panel() - 1;

    Alignable*& panel = newPanel[p];

    if (0 == panel)
    { // create new panel with id and rot of 1st sensor
      panel = new AlignableComposite( id.rawId(), AlignableObjectId::Panel,
				      sensor->globalRotation() );
      panels.push_back(panel);
    }

    panel->addComponent(sensor);
  }

// Build blades

  unsigned int nPanel = panels.size();

  for (unsigned int i = 0; i < nPanel; ++i)
  {
    Alignable* panel = panels[i];

    PXFDetId id = panel->id();

    unsigned int e = id.side() - 1;
    unsigned int d = e * maxDisk  + id.disk()  - 1;
    unsigned int b = d * maxBlade + id.blade() - 1;

    Alignable*& blade = newBlade[b];

    if (0 == blade)
    { // create new blade with id and rot of 1st panel
      blade = new AlignableComposite( id.rawId(), AlignableObjectId::Blade,
				      panel->globalRotation() );
      blades.push_back(blade);
    }

    blade->addComponent(panel);
  }

// // Build half disks (split along y-axis)

  unsigned int nBlade = blades.size();

  for (unsigned int i = 0; i < nBlade; ++i)
  {
    Alignable* blade = blades[i];

    PXFDetId id = blade->id();

    unsigned int b = id.blade(); // 1 to 24 in increasing phi
    unsigned int e = id.side() - 1;
    unsigned int c = e * maxCylinder + (b > 6 && b < 19 ? 0 : 1);
    unsigned int d = c * maxDisk + id.disk() - 1;

    Alignable*& halfDisk = newHalfDisk[d];

    if (0 == halfDisk)
    {  // create new half disk with id of 1st blade and identity rot
      halfDisk = new AlignableComposite(id.rawId(), AlignableObjectId::HalfDisk);
      halfDisks.push_back(halfDisk);
    }

    halfDisk->addComponent(blade);
  }

// Build half cylinders

  unsigned int nHalfDisk = halfDisks.size();

  for (unsigned int i = 0; i < nHalfDisk; ++i)
  {
    Alignable* halfDisk = halfDisks[i];

    PXFDetId id = halfDisk->id();

    unsigned int b = id.blade(); // 1 to 24 in increasing phi
    unsigned int e = id.side() - 1;
    unsigned int c = e * maxCylinder + (b > 6 && b < 19 ? 0 : 1);

    Alignable*& halfCylinder = newHalfCylinder[c];

    if (0 == halfCylinder)
    {  // create new half cylinder with id of 1st half disk and identity rot
      halfCylinder = new AlignableComposite(id.rawId(), AlignableObjectId::HalfCylinder);
      halfCylinders.push_back(halfCylinder);
    }

    halfCylinder->addComponent(halfDisk);
  }

// Build endcaps

  unsigned int nHalfCylinder = halfCylinders.size();

  for (unsigned int i = 0; i < nHalfCylinder; ++i)
  {
    Alignable* halfCylinder = halfCylinders[i];

    PXFDetId id = halfCylinder->id();

    unsigned int e = id.side() - 1;

    Alignable*& endcap = newEndcap[e];

    if (0 == endcap)
    {  // create new endcap with id of 1st half cylinder and identity rot
      endcap = new AlignableComposite(id.rawId(), AlignableObjectId::PixelEndcap);
      endcaps.push_back(endcap);
    }

    endcap->addComponent(halfCylinder);
  }

// Add pixel endcaps to alignable tracker

  for (unsigned int i = 0; i < endcaps.size(); ++i) addComponent(endcaps[i]);

  timeval t1;
  gettimeofday(&t1, 0);
  LogDebug("AlignableTracker") << "Time taken: " <<  (t1.tv_usec - t0.tv_usec) << " us\n";
}


//--------------------------------------------------------------------------------------------------
Alignable::Alignables AlignableTracker::outerHalfBarrels()
{

  Alignables result;
  copy( theOuterHalfBarrels.begin(), theOuterHalfBarrels.end(), back_inserter(result) );
  return result;

}


//--------------------------------------------------------------------------------------------------
Alignable::Alignables AlignableTracker::innerHalfBarrels()
{

  Alignables result;
  copy( theInnerHalfBarrels.begin(), theInnerHalfBarrels.end(), back_inserter(result) );
  return result;

}

//--------------------------------------------------------------------------------------------------
Alignable::Alignables AlignableTracker::pixelHalfBarrels()
{

  Alignables result;
  copy( thePixelHalfBarrels.begin(), thePixelHalfBarrels.end(), back_inserter(result) );
  return result;

}

//--------------------------------------------------------------------------------------------------
Alignable::Alignables AlignableTracker::endCaps()
{

  Alignables result;
  copy( theEndcaps.begin(), theEndcaps.end(), back_inserter(result) );
  return result;

}

//--------------------------------------------------------------------------------------------------
Alignable::Alignables AlignableTracker::TIDs()
{

  Alignables result;
  copy( theTIDs.begin(), theTIDs.end(), back_inserter(result) );
  return result;

}

//--------------------------------------------------------------------------------------------------
AlignableTrackerHalfBarrel  &AlignableTracker::outerHalfBarrel(unsigned int i)
{
  
  if (i >= theOuterHalfBarrels.size() )
	throw cms::Exception("LogicError") 
	  << "Barrel index (" << i << ") out of range";
  else 
	return *(theOuterHalfBarrels[i]);

}


//--------------------------------------------------------------------------------------------------
AlignablePixelHalfBarrel  &AlignableTracker::pixelHalfBarrel(unsigned int i)
{ 
  
  if (i >= thePixelHalfBarrels.size() )
	throw cms::Exception("LogicError") 
	  << "Pixel Half Barrel index (" << i << ") out of range"; 
  else
	return *(thePixelHalfBarrels[i]);                  

}


//--------------------------------------------------------------------------------------------------
AlignableTrackerHalfBarrel  &AlignableTracker::innerHalfBarrel(unsigned int i)
{

  if (i >= theInnerHalfBarrels.size() )
	throw cms::Exception("LogicError") 
	  << "Barrel index (" << i << ") out of range";
  else
	return *(theInnerHalfBarrels[i]);

}


//--------------------------------------------------------------------------------------------------
AlignableTID  &AlignableTracker::TID(unsigned int i)
{

  if (i >= theTIDs.size() )
	throw cms::Exception("LogicError") 
	  << "TID index (" << i << ") out of range";
  else
	return *(theTIDs[i]);

}


//--------------------------------------------------------------------------------------------------
AlignableTrackerEndcap  &AlignableTracker::endCap(unsigned int i)
{

  if (i >= theEndcaps.size() )
	throw cms::Exception("LogicError") 
	  << "Endcap index (" << i << ") out of range";
  else
	return *(theEndcaps[i]);

}


//--------------------------------------------------------------------------------------------------
// AlignableTrackerEndcap  &AlignableTracker::pixelEndCap(unsigned int i)
// {

//   if (i >= thePixelEndcaps.size() )
// 	throw cms::Exception("LogicError") 
// 	  << "Pixel endcap index (" << i << ") out of range";
//   else
// 	return *(thePixelEndcaps[i]);

// }


//--------------------------------------------------------------------------------------------------
Alignable::Alignables AlignableTracker::barrelGeomDets()
{

  Alignables ib = innerBarrelGeomDets();
  Alignables ob = outerBarrelGeomDets();
  Alignables result( ib.size() + ob.size() );
  merge( ib.begin(), ib.end(), ob.begin(), ob.end(), result.begin() );

  return result;

}

//--------------------------------------------------------------------------------------------------
Alignable::Alignables AlignableTracker::TIBTIDGeomDets()
{

  Alignables ib = innerBarrelGeomDets();
  Alignables tid = TIDGeomDets();
  Alignables result( ib.size() + tid.size() );
  merge( ib.begin(), ib.end(), tid.begin(), tid.end(), result.begin() );

  return result;

}
//--------------------------------------------------------------------------------------------------
Alignable::Alignables AlignableTracker::outerBarrelGeomDets()
{

  Alignables result;
  for ( unsigned int i=0; i<theOuterHalfBarrels.size();i++ )
	for ( int j=0; j<outerHalfBarrel(i).size(); j++ )
	  for ( int k=0; k<outerHalfBarrel(i).layer(j).size();k++ )
		for ( int l=0; l<outerHalfBarrel(i).layer(j).rod(k).size(); l++ )
		  result.push_back(&outerHalfBarrel(i).layer(j).rod(k).det(l));

  return result;

}


//--------------------------------------------------------------------------------------------------
Alignable::Alignables AlignableTracker::pixelHalfBarrelGeomDets()
{  

  Alignables result;  
  for ( unsigned int i=0; i<thePixelHalfBarrels.size();i++ )   
	for ( int j=0; j<pixelHalfBarrel(i).size(); j++ )  
	  for ( int k=0; k<pixelHalfBarrel(i).layer(j).size(); k++ )   
		for ( int l=0; l<pixelHalfBarrel(i).layer(j).ladder(k).size(); l++) 
		  result.push_back(&pixelHalfBarrel(i).layer(j).ladder(k).det(l)); 
  
  return result;		    		    

} 


//--------------------------------------------------------------------------------------------------
Alignable::Alignables AlignableTracker::innerBarrelGeomDets()
{

  Alignables result;
  for ( unsigned int i=0; i<theInnerHalfBarrels.size(); i++ )
    for ( int j=0; j<innerHalfBarrel(i).size(); j++ )
      for ( int k=0; k<innerHalfBarrel(i).layer(j).size(); k++ ) 
		for ( int l=0; l<innerHalfBarrel(i).layer(j).rod(k).size(); l++ ) 
		  result.push_back(&innerHalfBarrel(i).layer(j).rod(k).det(l));

  return result;		    		    

}


//--------------------------------------------------------------------------------------------------
Alignable::Alignables AlignableTracker::endcapGeomDets()
{

  Alignables result;
  for (unsigned int i=0; i<theEndcaps.size();i++)
	for (int j=0; j<endCap(i).size(); j++)
	  for (int k=0; k<endCap(i).layer(j).size(); k++) 
		for (int l=0; l<endCap(i).layer(j).petal(k).size(); l++) 
		  result.push_back(&endCap(i).layer(j).petal(k).det(l));
  
  return result;		    		    

}


//--------------------------------------------------------------------------------------------------
Alignable::Alignables AlignableTracker::TIDGeomDets()
{

  Alignables result;
  for ( unsigned int i=0; i<theTIDs.size(); i++ )
    for ( int j=0; j<TID(i).size(); j++ )
      for ( int k=0; k<TID(i).layer(j).size(); k++ ) 
		for ( int l=0; l<TID(i).layer(j).ring(k).size(); l++ ) 
		  result.push_back(&TID(i).layer(j).ring(k).det(l));
  
  return result;		    		    

}


//--------------------------------------------------------------------------------------------------
Alignable::Alignables AlignableTracker::barrelRods()
{

  Alignables ib = innerBarrelRods();
  Alignables ob = outerBarrelRods();
  Alignables result( ib.size() + ob.size() );
  merge(ib.begin(),ib.end(),ob.begin(),ob.end(),result.begin());

  return result;

}


//--------------------------------------------------------------------------------------------------
Alignable::Alignables AlignableTracker::outerBarrelRods()
{

  Alignables result;
  for ( unsigned int i=0; i<theOuterHalfBarrels.size();i++ )
    for ( int j=0; j<outerHalfBarrel(i).size(); j++ )
      for ( int k=0; k<outerHalfBarrel(i).layer(j).size(); k++ )
		result.push_back(&outerHalfBarrel(i).layer(j).rod(k));

  return result;

}


//--------------------------------------------------------------------------------------------------
Alignable::Alignables AlignableTracker::innerBarrelRods()
{

  Alignables result;
  for ( unsigned int i=0; i<theInnerHalfBarrels.size();i++ )
    for ( int j=0; j<innerHalfBarrel(i).size(); j++ )
      for ( int k=0; k<innerHalfBarrel(i).layer(j).size(); k++ ) 
		result.push_back(&innerHalfBarrel(i).layer(j).rod(k));

  return result;		    		    

}


//--------------------------------------------------------------------------------------------------
Alignable::Alignables AlignableTracker::pixelHalfBarrelLadders()
{  

  Alignables result;
  for ( unsigned int i=0; i<thePixelHalfBarrels.size(); i++ )
    for ( int j=0; j<pixelHalfBarrel(i).size(); j++ )
      for ( int k=0; k<pixelHalfBarrel(i).layer(j).size(); k++ )
		result.push_back(&pixelHalfBarrel(i).layer(j).ladder(k));

  return result;		    		    

}


//--------------------------------------------------------------------------------------------------
Alignable::Alignables AlignableTracker::endcapPetals()
{
  
  Alignables result;
  for ( unsigned int i=0; i<theEndcaps.size(); i++ )
    for ( int j=0; j<endCap(i).size(); j++ )
      for ( int k=0; k<endCap(i).layer(j).size(); k++ ) 
		result.push_back(&endCap(i).layer(j).petal(k));

  return result;

}


//--------------------------------------------------------------------------------------------------
Alignable::Alignables AlignableTracker::TIDRings()
{

  Alignables result;
  for ( unsigned int i=0; i<theTIDs.size();i++ )
    for ( int j=0; j<TID(i).size(); j++ )
      for ( int k=0; k<TID(i).layer(j).size(); k++ ) 
		result.push_back(&TID(i).layer(j).ring(k));

  return result;

}


//--------------------------------------------------------------------------------------------------
Alignable::Alignables AlignableTracker::barrelLayers()
{

  Alignables ib = innerBarrelLayers();
  Alignables ob = outerBarrelLayers();
  Alignables result( ib.size() + ob.size() );
  merge(ib.begin(),ib.end(),ob.begin(),ob.end(),result.begin());
  return result;

}


//--------------------------------------------------------------------------------------------------
Alignable::Alignables AlignableTracker::outerBarrelLayers()
{

  Alignables result;
  for ( unsigned int i=0; i<theOuterHalfBarrels.size(); i++ )
    for ( int j=0; j<outerHalfBarrel(i).size(); j++ ) 
	  result.push_back(&outerHalfBarrel(i).layer(j));

  return result;		    		    

}


//--------------------------------------------------------------------------------------------------
Alignable::Alignables AlignableTracker::pixelHalfBarrelLayers()
{  

  Alignables result;           
  for ( unsigned int i=0; i<thePixelHalfBarrels.size();i++)               
    for ( int j=0; j<pixelHalfBarrel(i).size(); j++ ) 
	  result.push_back(&pixelHalfBarrel(i).layer(j)); 

  return result;		    		    

}


//--------------------------------------------------------------------------------------------------
Alignable::Alignables AlignableTracker::innerBarrelLayers()
{

  Alignables result;
  for ( unsigned int i=0; i<theInnerHalfBarrels.size(); i++ )
    for ( int j=0; j<innerHalfBarrel(i).size(); j++ ) 
	  result.push_back(&innerHalfBarrel(i).layer(j));

  return result;    		    

}


//--------------------------------------------------------------------------------------------------
Alignable::Alignables AlignableTracker::endcapLayers()
{

  Alignables result;
  for ( unsigned int i=0; i<theEndcaps.size(); i++ )
    for ( int j=0; j<endCap(i).size(); j++ ) 
	  result.push_back(&endCap(i).layer(j));

  return result;		    		    

}


//--------------------------------------------------------------------------------------------------
Alignable::Alignables AlignableTracker::TIDLayers()
{

  Alignables result;
  for ( unsigned int i=0; i<theTIDs.size(); i++ )
    for (int j=0; j<TID(i).size(); j++ ) 
      result.push_back(&TID(i).layer(j));

  return result;

}


//--------------------------------------------------------------------------------------------------
void AlignableTracker::dump( void ) const
{

  std::cout << "--------------" << std::endl 
			<< " Tracker components" << std::endl
			<< "--------------" << std::endl;

  for ( std::vector<AlignableTrackerHalfBarrel*>::const_iterator iDet = theOuterHalfBarrels.begin();
		iDet != theOuterHalfBarrels.end(); iDet++ )
	(*iDet)->dump();
  for ( std::vector<AlignableTrackerHalfBarrel*>::const_iterator iDet = theInnerHalfBarrels.begin();
		iDet != theInnerHalfBarrels.end(); iDet++ )
	(*iDet)->dump();
  for ( std::vector<AlignablePixelHalfBarrel*>::const_iterator iDet = thePixelHalfBarrels.begin();
		iDet != thePixelHalfBarrels.end(); iDet++ )
	(*iDet)->dump();

  for ( std::vector<AlignableTrackerEndcap*>::const_iterator iDet = theEndcaps.begin();
		iDet != theEndcaps.end(); iDet++ )
	(*iDet)->dump();

  const Alignables& pxe = getAlignables("pixelEndcaps");

  for (unsigned int i = 0; i < pxe.size(); ++i) pxe[i]->dump();

  for ( std::vector<AlignableTID*>::const_iterator iDet = theTIDs.begin();
		iDet != theTIDs.end(); iDet++ )
	(*iDet)->dump();

  std::cout << "--------------" << std::endl;

}


//__________________________________________________________________________________________________
Alignments* AlignableTracker::alignments( void ) const
{

  Alignables comp = this->components();
  Alignments* m_alignments = new Alignments();
  // Add components recursively
  for ( Alignables::iterator i=comp.begin(); i!=comp.end(); i++ )
    {
      Alignments* tmpAlignments = (*i)->alignments();
      std::copy( tmpAlignments->m_align.begin(), tmpAlignments->m_align.end(), 
				 std::back_inserter(m_alignments->m_align) );
	  delete tmpAlignments;
    }

  std::sort( m_alignments->m_align.begin(), m_alignments->m_align.end(), 
			 lessAlignmentDetId<AlignTransform>() );

  return m_alignments;

}


//__________________________________________________________________________________________________
AlignmentErrors* AlignableTracker::alignmentErrors( void ) const
{

  Alignables comp = this->components();
  AlignmentErrors* m_alignmentErrors = new AlignmentErrors();

  // Add components recursively
  for ( Alignables::iterator i=comp.begin(); i!=comp.end(); i++ )
    {
	  AlignmentErrors* tmpAlignmentErrors = (*i)->alignmentErrors();
      std::copy( tmpAlignmentErrors->m_alignError.begin(), tmpAlignmentErrors->m_alignError.end(), 
				 std::back_inserter(m_alignmentErrors->m_alignError) );
	  delete tmpAlignmentErrors;
    }

  std::sort( m_alignmentErrors->m_alignError.begin(), m_alignmentErrors->m_alignError.end(), 
			 lessAlignmentDetId<AlignTransformError>() );

  return m_alignmentErrors;

}


//--------------------------------------------------------------------------------------------------
std::vector<const GeometricDet*>
AlignableTracker::getAllComponents( 
								   const GeometricDet* Det,
								   const GeometricDet::GDEnumType type 
								   ) const
{

  // Return components of given GeometricDet with given type

  _DetContainer _temp;
  _DetContainer m_components = Det->components();

  if ( Det->type() == type ) _temp.push_back( Det );
  else
	{
	  for ( _DetContainer::iterator it = m_components.begin();
			it != m_components.end(); it++ )
		{
		  _DetContainer _temp2 = this->getAllComponents( *it, type );
		  copy( _temp2.begin(), _temp2.end(), back_inserter(_temp) );
		}
	}

  return _temp;
  

}


//__________________________________________________________________________________________________
void AlignableTracker::recursiveSetMothers( Alignable* alignable )
{

  Alignables components = alignable->components();
  for ( Alignables::iterator iter = components.begin();
		iter != components.end(); iter++ )
	{
	  (*iter)->setMother( alignable );
	  recursiveSetMothers( *iter );
	}

}

const Alignable::Alignables& AlignableTracker::getAlignables( const std::string& structure ) const
{
  std::map<std::string, Alignables>::const_iterator e = theTracker.find(structure);

  if (theTracker.end() == e)
  {
    throw cms::Exception("LogicError")
      << "Structure " << structure << " does not exist in the tracker";
  }

  return e->second;
}
