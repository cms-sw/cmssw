#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

#include "FWCore/Utilities/interface/Exception.h"
 
// Geometry
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

// Alignment
#include "Alignment/CommonAlignment/interface/AlignableBuilder.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/TrackerAlignment/interface/AlignableSiStripDet.h"
#include "Alignment/CommonAlignment/interface/AlignableDetUnit.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/Alignment/interface/AlignmentSorter.h"

//__________________________________________________________________________________________________
AlignableTracker::AlignableTracker( const TrackerGeometry* tkGeom, const TrackerTopology* tTopo ):
  AlignableComposite( 0, align::Tracker, RotationType() ), tTopo_(tTopo) // id not yet known
{

  // Get levels from geometry
  // for strip we create also <TIB/TID/TOB/TEC>ModuleUnit list for 1D components of 2D layers
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
  align::Alignables *aliUnits = 0;// If we need also units, they will be at moduleName + "Unit".

  for (unsigned int i = 0; i < nDet; ++i) {
   
    const unsigned int subdetId = dets[i]->geographicalId().subdetId();//don't check det()==Tracker
    if (subdetId == PixelSubdetector::PixelBarrel || subdetId == PixelSubdetector::PixelEndcap) {
      // Treat all pixel dets in same way with one AlignableDetUnit.
      const GeomDetUnit *detUnit = dynamic_cast<const GeomDetUnit*>(dets[i]);
      if (!detUnit) {
        throw cms::Exception("BadHierarchy") 
          << "[AlignableTracker] Pixel GeomDet (subdetector " << subdetId << ") not GeomDetUnit.\n";
      }
      alis.push_back(new AlignableDetUnit(detUnit));

      // Add pixel modules to list of units since they are in fact units
      if (!aliUnits) {
	aliUnits = &alignableLists_.get(moduleName + "Unit");
	aliUnits->reserve(576); // ugly hardcode to save some memory due to vector doubling
      }
      aliUnits->push_back(alis.back());
    
    } else if (subdetId == SiStripDetId::TIB || subdetId == SiStripDetId::TID
	       || subdetId == SiStripDetId::TOB || subdetId == SiStripDetId::TEC) {
      // In strip we have:
      // 1) 'Pure' 1D-modules like TOB layers 3-6 (not glued): AlignableDetUnit
      // 2) Composite 2D-modules like TOB layers 1&2 (not glued): AlignableDet
      // 3) The two 1D-components of case 2 (glued): AlignableDetUnit that is constructed
      //      inside AlignableDet-constructor of 'mother', only need to add to alignableLists_
      const SiStripDetId detId(dets[i]->geographicalId());
      if (!detId.glued()) { // 2D- or 'pure' 1D-module
        if (dets[i]->components().size()) { // 2D-module
	  const GluedGeomDet *gluedDet = dynamic_cast<GluedGeomDet*>(dets[i]);
	  if (!gluedDet) {
	    throw cms::Exception("LogicError") 
	      << "[AlignableTracker]" << "dynamic_cast<GluedGeomDet*> failed.\n";
	  }
          alis.push_back(new AlignableSiStripDet(gluedDet)); // components constructed within
          const align::Alignables detUnits(alis.back()->components());
	  // Ensure pointer existence and make list available via moduleName appended with "Unit"
	  if (!aliUnits) {
	    aliUnits = &alignableLists_.get(moduleName + "Unit");
	    aliUnits->reserve(576); // ugly hardcode to save some memory due to vector doubling
	  }
	  aliUnits->insert(aliUnits->end(), detUnits.begin(), detUnits.end()); // only 2...
	} else { // no components: pure 1D-module
          const GeomDetUnit *detUnit = dynamic_cast<const GeomDetUnit*>(dets[i]);
          if (!detUnit) {
            throw cms::Exception("BadHierarchy") 
              << "[AlignableTracker] pure 1D GeomDet (subdetector " << subdetId << ") not GeomDetUnit.\n";
          }
          alis.push_back(new AlignableDetUnit(detUnit));

	  // Add pure 1D-modules to list of units since they are in fact units
 	  if (!aliUnits) {
	    aliUnits = &alignableLists_.get(moduleName + "Unit");
	    aliUnits->reserve(576); // ugly hardcode to save some memory due to vector doubling
	  }
	  aliUnits->push_back(alis.back());
        }
      } // no else: glued components of AlignableDet constructed within AlignableDet, see above
    } else {
      throw cms::Exception("LogicError") 
	<< "[AlignableTracker] GeomDet of unknown subdetector.";
    }
  }

  LogDebug("Alignment") << "@SUB=AlignableTracker"
			<< alis.size() << " AlignableDet(Unit)s for " << moduleName;
  if (aliUnits) {
    LogDebug("Alignment") << "@SUB=AlignableTracker"
			  << aliUnits->size() << " AlignableDetUnits for " 
			  << moduleName + "Unit (capacity = " << aliUnits->capacity() << ").";
  }
  
}


//__________________________________________________________________________________________________
void AlignableTracker::buildBarrel(const std::string& subDet)
{
  align::Alignables& halfBarrels = alignableLists_.find(subDet + "HalfBarrel");

  const std::string barrelName(subDet + "Barrel");
  align::Alignables &barrel = alignableLists_.get(barrelName);
  barrel.reserve(1);
  barrel.push_back(new AlignableComposite(halfBarrels[0]->id(),AlignableObjectId::stringToId(barrelName),
					  RotationType()));
  barrel[0]->addComponent(halfBarrels[0]);
  barrel[0]->addComponent(halfBarrels[1]);
}

//__________________________________________________________________________________________________
void AlignableTracker::buildTPB()
{
  AlignableBuilder builder(align::TPBModule, tkCounters_, tTopo_);

  builder.addLevelInfo(align::TPBLadder    , true, 22); // max 22 ladders per layer
  builder.addLevelInfo(align::TPBLayer     , false, 3); // 3 layers per half barrel
  builder.addLevelInfo(align::TPBHalfBarrel, false, 2); // 2 half barrels in TPB
  builder.buildAll( alignableLists_ );

  buildBarrel("TPB");
}

//__________________________________________________________________________________________________
void AlignableTracker::buildTPE()
{
  AlignableBuilder builder(align::TPEModule, tkCounters_, tTopo_);

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
  AlignableBuilder builder(align::TIBModule, tkCounters_, tTopo_);

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
  AlignableBuilder builder(align::TIDModule, tkCounters_, tTopo_);

  builder.addLevelInfo(align::TIDSide  , false, 2); // 2 sides per ring
  builder.addLevelInfo(align::TIDRing  , false, 3); // 3 rings per disk
  builder.addLevelInfo(align::TIDDisk  , false, 3); // 3 disks per endcap
  builder.addLevelInfo(align::TIDEndcap, false, 2); // 2 endcaps in TID
  builder.buildAll( alignableLists_ );
}

//__________________________________________________________________________________________________
void AlignableTracker::buildTOB()
{
  AlignableBuilder builder(align::TOBModule, tkCounters_, tTopo_);

  builder.addLevelInfo(align::TOBRod       , true, 74); // max 74 rods per layer
  builder.addLevelInfo(align::TOBLayer     , false, 6); // 6 layers per half barrel
  builder.addLevelInfo(align::TOBHalfBarrel, false, 2); // 2 half barrels in TOB
  builder.buildAll( alignableLists_ );

  buildBarrel("TOB");
}

//__________________________________________________________________________________________________
void AlignableTracker::buildTEC()
{
  AlignableBuilder builder(align::TECModule, tkCounters_, tTopo_);

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
  // Build pixel, strip and full tracker 'by hand':
  
  // First create pixel:
  const align::Alignables &pixelBarrel = alignableLists_.find("TPBBarrel");
  const align::Alignables &pixelEndcap = alignableLists_.find("TPEEndcap");
  align::Alignables &pixel = alignableLists_.get("Pixel");
  pixel.reserve(1);
  pixel.push_back(new AlignableComposite(pixelBarrel[0]->id(), align::Pixel, RotationType()));
  pixel[0]->addComponent(pixelBarrel[0]);
  pixel[0]->addComponent(pixelEndcap[0]);
  pixel[0]->addComponent(pixelEndcap[1]);

  // Now create strip:
  const align::Alignables &innerBarrel = alignableLists_.find("TIBBarrel");
  const align::Alignables &outerBarrel = alignableLists_.find("TOBBarrel");
  const align::Alignables &innerEndcap = alignableLists_.find("TIDEndcap");
  const align::Alignables &outerEndcap = alignableLists_.find("TECEndcap");
  align::Alignables &strip = alignableLists_.get("Strip");
  strip.reserve(1);
  strip.push_back(new AlignableComposite(innerBarrel[0]->id(), align::Strip, RotationType()));
  strip[0]->addComponent(innerBarrel[0]);
  strip[0]->addComponent(innerEndcap[0]);
  strip[0]->addComponent(innerEndcap[1]);
  strip[0]->addComponent(outerBarrel[0]);
  strip[0]->addComponent(outerEndcap[0]);
  strip[0]->addComponent(outerEndcap[1]);

  // Finally add strip and pixel to tracker - that of course already exists:
  align::Alignables &tracker = alignableLists_.get("Tracker");
  tracker.reserve(1);
  tracker.push_back(this);
  this->addComponent(pixel[0]); // add to tracker
  this->addComponent(strip[0]); // add to tracker

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
