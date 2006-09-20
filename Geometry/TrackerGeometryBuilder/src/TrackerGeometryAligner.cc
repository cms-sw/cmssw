#include <map>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Vector/Rotation.h"
#include "CLHEP/Vector/ThreeVector.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometryAligner.h"

void TrackerGeometryAligner::applyAlignments( TrackerGeometry* tracker, 
					      const Alignments* alignments )
{

  edm::LogInfo("Starting") << "Starting to apply alignments";

  std::map<DetId,GeomDet*> m_Map = tracker->theMap;

  // Preliminary check (or we can't loop!)
  if ( alignments->m_align.size() != m_Map.size() )
	throw cms::Exception("GeometryMismatch") 
	  << "Size mismatch between geometry (size=" << m_Map.size() 
	  << ") and alignments (size=" << alignments->m_align.size() << ")";

  // Parallel loop on alignments and geomdets
  std::vector<AlignTransform>::const_iterator iAlign = alignments->m_align.begin();
  for ( std::map<DetId,GeomDet*>::iterator iPair = m_Map.begin(); 
		iPair != m_Map.end(); iPair++, iAlign++ )
	{
	  // Check DetIds
	  if ( (*iPair).first.rawId() != (*iAlign).rawId() )
	    throw cms::Exception("GeometryMismatch") 
	      << "DetId mismatch between geometry (rawId=" << (*iPair).first.rawId()
	      << ") and alignments (rawId=" << (*iAlign).rawId();

	  GeomDet* iGeomDet = (*iPair).second;
	  Hep3Vector alignPosition( (*iAlign).translation() );
	  HepRotation alignRotation( (*iAlign).rotation() );

          //// Dump before
          //edm::LogInfo("DumpBefore") << (*iPair).first.rawId() << " " << iGeomDet->position();

	  // Apply alignments
	  Surface::PositionType position( alignPosition.x(), alignPosition.y(), alignPosition.z() );
	  Surface::RotationType rotation( alignRotation.xx(), alignRotation.xy(), alignRotation.xz(),
					  alignRotation.yx(), alignRotation.yy(), alignRotation.yz(),
					  alignRotation.zx(), alignRotation.zy(), alignRotation.zz() );
	  DetPositioner::setGeomDetPosition( *iGeomDet, position, rotation );

          //// Dump after
          //edm::LogInfo("DumpAfter") << (*iPair).first.rawId() << " " << iGeomDet->position();
	}

  edm::LogInfo("Done") << "Finished to apply alignments";
}
