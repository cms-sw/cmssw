#include "Alignment/CommonAlignment/interface/AlignableBuilder.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/Alignment/interface/AlignmentSorter.h"

#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

using namespace align;

AlignableTracker::AlignableTracker(const GeometricDet*,
				   const TrackerGeometry*):
  AlignableComposite( 0, Tracker, RotationType() )
{
  buildTPB();
  buildTPE();
  buildTIB();
  buildTID();
  buildTOB();
  buildTEC();
  buildTRK();
}

void AlignableTracker::buildBarrel(const std::string& subDet) const
{
  static AlignableObjectId objId;

  const Alignables& halfBarrels = AlignSetup<Alignables>::find(subDet + "HalfBarrel");

  std::string barrelName = subDet + "Barrel";

  Alignable*& barrel = AlignSetup<Alignable*>::get(barrelName);

  barrel = new AlignableComposite( halfBarrels[0]->id(),
				   objId.nameToType(barrelName),
				   RotationType() );

  barrel->addComponent(halfBarrels[0]);
  barrel->addComponent(halfBarrels[1]);
}

void AlignableTracker::buildTPB() const
{
  AlignableBuilder builder(TPBModule);

  builder.addLevelInfo(TPBLadder    , true, 22); // max 22 ladders per layer
  builder.addLevelInfo(TPBLayer     , false, 3); // 3 layers per half barrel
  builder.addLevelInfo(TPBHalfBarrel, false, 2); // 2 half barrels in TPB
  builder.buildAll();

  buildBarrel("TPB");
}

void AlignableTracker::buildTPE() const
{
  AlignableBuilder builder(TPEModule);

  builder.addLevelInfo(TPEPanel       , true,  2); // 2 panels per blade
  builder.addLevelInfo(TPEBlade       , true, 12); // 12 blades per half disk
  builder.addLevelInfo(TPEHalfDisk    , false, 3); // max 3 disks per cylinder
  builder.addLevelInfo(TPEHalfCylinder, false, 2); // 2 HC per endcap
  builder.addLevelInfo(TPEEndcap      , false, 2); // 2 endcaps in TPE
  builder.buildAll();
}

void AlignableTracker::buildTIB() const
{
  AlignableBuilder builder(TIBModule);

  builder.addLevelInfo(TIBString    , true, 28); // max 22 strings per surface
  builder.addLevelInfo(TIBSurface   , false, 2); // 2 surfaces per half shell
  builder.addLevelInfo(TIBHalfShell , false, 2); // 2 half shells per layer
  builder.addLevelInfo(TIBLayer     , false, 4); // 4 layers per half barrel
  builder.addLevelInfo(TIBHalfBarrel, false, 2); // 2 half barrels in TIB
  builder.buildAll();

  buildBarrel("TIB");
}

void AlignableTracker::buildTID() const
{
  AlignableBuilder builder(TIDModule);

  builder.addLevelInfo(TIDSide  , false, 2); // 2 sides per ring
  builder.addLevelInfo(TIDRing  , false, 3); // 3 rings per disk
  builder.addLevelInfo(TIDDisk  , false, 3); // 3 disks per endcap
  builder.addLevelInfo(TIDEndcap, false, 2); // 2 endcaps in TID
  builder.buildAll();
}

void AlignableTracker::buildTOB() const
{
  AlignableBuilder builder(TOBModule);

  builder.addLevelInfo(TOBRod       , true, 74); // max 74 rods per layer
  builder.addLevelInfo(TOBLayer     , false, 6); // 6 layers per half barrel
  builder.addLevelInfo(TOBHalfBarrel, false, 2); // 2 half barrels in TOB
  builder.buildAll();

  buildBarrel("TOB");
}

void AlignableTracker::buildTEC() const
{
  AlignableBuilder builder(TECModule);

  builder.addLevelInfo(TECRing  , true,  7); // max 7 rings per petal
  builder.addLevelInfo(TECPetal , true,  8); // 8 petals per side
  builder.addLevelInfo(TECSide  , false, 2); // 2 sides per disk
  builder.addLevelInfo(TECDisk  , false, 9); // 9 disks per endcap
  builder.addLevelInfo(TECEndcap, false, 2); // 2 endcaps in TEC
  builder.buildAll();
}

void AlignableTracker::buildTRK()
{
  Alignable*& pixel = AlignSetup<Alignable*>::get("Pixel");
  Alignable*& strip = AlignSetup<Alignable*>::get("Strip");

  Alignable* const& innerBarrel = AlignSetup<Alignable*>::find("TIBBarrel");
  Alignable* const& outerBarrel = AlignSetup<Alignable*>::find("TOBBarrel");
  Alignable* const& pixelBarrel = AlignSetup<Alignable*>::find("TPBBarrel");
  const Alignables& innerEndcap = AlignSetup<Alignables>::find("TIDEndcap");
  const Alignables& outerEndcap = AlignSetup<Alignables>::find("TECEndcap");
  const Alignables& pixelEndcap = AlignSetup<Alignables>::find("TPEEndcap");

  pixel = new AlignableComposite( pixelBarrel->id(), Pixel, RotationType() );
  strip = new AlignableComposite( innerBarrel->id(), Strip, RotationType() );

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

  AlignSetup<Alignable*>::get("Tracker") = this;
}

Alignables AlignableTracker::merge(const Alignables& list1,
				   const Alignables& list2) const
{
  Alignables all = list1;

  all.insert( all.end(), list2.begin(), list2.end() );

  return all;
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
