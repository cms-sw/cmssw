// Geometry
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

// Alignment
#include "Alignment/CommonAlignment/interface/AlignableBuilder.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/Alignment/interface/AlignmentSorter.h"

#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"


//__________________________________________________________________________________________________
AlignableTracker::AlignableTracker( const TrackerGeometry* tkGeom ):
  AlignableComposite( 0, align::Tracker, RotationType() )
{

  // Get levels from geometry
  detsToAlignables(tkGeom->detsPXB(), "TPBModule");
  detsToAlignables(tkGeom->detsPXF(), "TPEModule");
  detsToAlignables(tkGeom->detsTIB(), "TIBModule");
  detsToAlignables(tkGeom->detsTID(), "TIDModule");
  detsToAlignables(tkGeom->detsTOB(), "TOBModule");
  detsToAlignables(tkGeom->detsTEC(), "TECModule");

  buildTPB();
  buildTPE();
  buildTIB();
  buildTID();
  buildTOB();
  buildTEC();
  buildTRK();

  theId = this->components()[0]->id(); // as all composites: id of first component
}


//__________________________________________________________________________________________________
void AlignableTracker::detsToAlignables( const TrackingGeometry::DetContainer& dets,
                                         const std::string& moduleName )
{
  unsigned int nDet = dets.size();

  align::Alignables& alis = alignableLists_.get(moduleName);
  alis.reserve(nDet);

  for (unsigned int i = 0; i < nDet; ++i)
  {
    // skip components of glued det
    SiStripDetId detId = dets[i]->geographicalId();
    if ( detId.subdetId() < SiStripDetId::TIB || !detId.glued() )
      alis.push_back( new AlignableDet(dets[i]) );
  }

}


//__________________________________________________________________________________________________
void AlignableTracker::buildBarrel(const std::string& subDet)
{
  AlignableObjectId objId;

  align::Alignables& halfBarrels = alignableLists_.find(subDet + "HalfBarrel");

  std::string barrelName = subDet + "Barrel";

  Alignable*& barrel = alignables_.get(barrelName);

  barrel = new AlignableComposite( halfBarrels[0]->id(),
                                   objId.nameToType(barrelName),
                                   RotationType() );

  barrel->addComponent(halfBarrels[0]);
  barrel->addComponent(halfBarrels[1]);
}

//__________________________________________________________________________________________________
void AlignableTracker::buildTPB()
{
  AlignableBuilder builder(align::TPBModule, tkCounters_ );

  builder.addLevelInfo(align::TPBLadder    , true, 22); // max 22 ladders per layer
  builder.addLevelInfo(align::TPBLayer     , false, 3); // 3 layers per half barrel
  builder.addLevelInfo(align::TPBHalfBarrel, false, 2); // 2 half barrels in TPB
  builder.buildAll( alignableLists_ );

  buildBarrel("TPB");
}

//__________________________________________________________________________________________________
void AlignableTracker::buildTPE()
{
  AlignableBuilder builder(align::TPEModule, tkCounters_);

  builder.addLevelInfo(align::TPEPanel       , true,  2); // 2 panels per blade
  builder.addLevelInfo(align::TPEBlade       , true, 12); // 12 blades per half disk
  builder.addLevelInfo(align::TPEHalfDisk    , false, 3); // max 3 disks per cylinder
  builder.addLevelInfo(align::TPEHalfCylinder, false, 2); // 2 HC per endcap
  builder.addLevelInfo(align::TPEEndcap      , false, 2); // 2 endcaps in TPE
  builder.buildAll( alignableLists_ );
}

//__________________________________________________________________________________________________
void AlignableTracker::buildTIB()
{
  AlignableBuilder builder(align::TIBModule, tkCounters_);

  builder.addLevelInfo(align::TIBString    , true, 28); // max 22 strings per surface
  builder.addLevelInfo(align::TIBSurface   , false, 2); // 2 surfaces per half shell
  builder.addLevelInfo(align::TIBHalfShell , false, 2); // 2 half shells per layer
  builder.addLevelInfo(align::TIBLayer     , false, 4); // 4 layers per half barrel
  builder.addLevelInfo(align::TIBHalfBarrel, false, 2); // 2 half barrels in TIB
  builder.buildAll( alignableLists_ );

  buildBarrel("TIB");
}

//__________________________________________________________________________________________________
void AlignableTracker::buildTID()
{
  AlignableBuilder builder(align::TIDModule, tkCounters_);

  builder.addLevelInfo(align::TIDSide  , false, 2); // 2 sides per ring
  builder.addLevelInfo(align::TIDRing  , false, 3); // 3 rings per disk
  builder.addLevelInfo(align::TIDDisk  , false, 3); // 3 disks per endcap
  builder.addLevelInfo(align::TIDEndcap, false, 2); // 2 endcaps in TID
  builder.buildAll( alignableLists_ );
}

//__________________________________________________________________________________________________
void AlignableTracker::buildTOB()
{
  AlignableBuilder builder(align::TOBModule, tkCounters_);

  builder.addLevelInfo(align::TOBRod       , true, 74); // max 74 rods per layer
  builder.addLevelInfo(align::TOBLayer     , false, 6); // 6 layers per half barrel
  builder.addLevelInfo(align::TOBHalfBarrel, false, 2); // 2 half barrels in TOB
  builder.buildAll( alignableLists_ );

  buildBarrel("TOB");
}

//__________________________________________________________________________________________________
void AlignableTracker::buildTEC()
{
  AlignableBuilder builder(align::TECModule, tkCounters_);

  builder.addLevelInfo(align::TECRing  , true,  7); // max 7 rings per petal
  builder.addLevelInfo(align::TECPetal , true,  8); // 8 petals per side
  builder.addLevelInfo(align::TECSide  , false, 2); // 2 sides per disk
  builder.addLevelInfo(align::TECDisk  , false, 9); // 9 disks per endcap
  builder.addLevelInfo(align::TECEndcap, false, 2); // 2 endcaps in TEC
  builder.buildAll( alignableLists_ );
}

//__________________________________________________________________________________________________
void AlignableTracker::buildTRK()
{
  Alignable*& pixel = alignables_.get("Pixel");
  Alignable*& strip = alignables_.get("Strip");

  Alignable* const& innerBarrel = alignables_.find("TIBBarrel");
  Alignable* const& outerBarrel = alignables_.find("TOBBarrel");
  Alignable* const& pixelBarrel = alignables_.find("TPBBarrel");
  const align::Alignables& innerEndcap = alignableLists_.find("TIDEndcap");
  const align::Alignables& outerEndcap = alignableLists_.find("TECEndcap");
  const align::Alignables& pixelEndcap = alignableLists_.find("TPEEndcap");

  pixel = new AlignableComposite( pixelBarrel->id(), align::Pixel, RotationType() );
  strip = new AlignableComposite( innerBarrel->id(), align::Strip, RotationType() );

  pixel->addComponent(pixelBarrel);
  pixel->addComponent(pixelEndcap[0]);
  pixel->addComponent(pixelEndcap[1]);

  strip->addComponent(innerBarrel);
  strip->addComponent(innerEndcap[0]);
  strip->addComponent(innerEndcap[1]);

  strip->addComponent(outerBarrel);
  strip->addComponent(outerEndcap[0]);
  strip->addComponent(outerEndcap[1]);

  addComponent(pixel); // add to tracker
  addComponent(strip); // add to tracker

  alignables_.get("Tracker") = this;
}


//__________________________________________________________________________________________________
align::Alignables AlignableTracker::merge( const Alignables& list1,
                                           const Alignables& list2 ) const
{
  Alignables all = list1;

  all.insert( all.end(), list2.begin(), list2.end() );

  return all;
}


//__________________________________________________________________________________________________
Alignments* AlignableTracker::alignments( void ) const
{

  align::Alignables comp = this->components();
  Alignments* m_alignments = new Alignments();
  // Add components recursively
  for ( align::Alignables::iterator i=comp.begin(); i!=comp.end(); i++ )
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

  align::Alignables comp = this->components();
  AlignmentErrors* m_alignmentErrors = new AlignmentErrors();

  // Add components recursively
  for ( align::Alignables::iterator i=comp.begin(); i!=comp.end(); i++ )
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
