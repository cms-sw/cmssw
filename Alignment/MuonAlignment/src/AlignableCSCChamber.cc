/** \file
 *
 *  $Date: 2008/03/26 21:59:18 $
 *  $Revision: 1.10 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */
 
#include "Alignment/MuonAlignment/interface/AlignableCSCChamber.h"
#include "Alignment/CommonAlignment/interface/AlignableDetUnit.h"

AlignableCSCChamber::AlignableCSCChamber(const GeomDet *geomDet): AlignableDet(geomDet, false)  // addComponents loop is performed below
{
   theStructureType = align::AlignableCSCChamber;

   // set the APE of this chamber and all its layers
   // then re-set the APEs of the layers in the loop that follows
   if (geomDet->alignmentPositionError() != NULL) {
      setAlignmentPositionError(*geomDet->alignmentPositionError());
   }

   const std::vector<const GeomDet*>& geomDets = geomDet->components();
   for (std::vector<const GeomDet*>::const_iterator idet = geomDets.begin();  idet != geomDets.end();  ++idet) {
      AlignableDetUnit *layer = new AlignableDetUnit((*idet)->geographicalId().rawId(), (*idet)->surface());
      if ((*idet)->alignmentPositionError() != NULL) {
	 layer->setAlignmentPositionError(*((*idet)->alignmentPositionError()));
      }
      addComponent(layer);
   }

   // DO NOT let the chamber position become an average of the layers
   this->theSurface = geomDet->surface();
}

/// Printout the DetUnits in the CSC chamber
std::ostream& operator<< (std::ostream &os, const AlignableCSCChamber & r) {
   std::vector<Alignable*> theDets = r.components();

   os << "    This CSCChamber contains " << theDets.size() << " units" << std::endl ;
   os << "    position = " << r.globalPosition() << std::endl;
   os << "    (phi, r, z)= (" << r.globalPosition().phi() << "," << r.globalPosition().perp() << "," << r.globalPosition().z();
   os << "), orientation:" << std::endl<< r.globalRotation() << std::endl;
   
   os << "    total displacement and rotation: " << r.displacement() << std::endl;
   os << r.rotation() << std::endl;
 
   for (std::vector<Alignable*>::const_iterator idet = theDets.begin();  idet != theDets.end();  ++idet) {
      const align::Alignables& comp = (*idet)->components();

      for (unsigned int i = 0; i < comp.size(); ++i) {
	 os << "     Det position, phi, r: " 
	    << comp[i]->globalPosition() << " , "
	    << comp[i]->globalPosition().phi() << " , "
	    << comp[i]->globalPosition().perp() << std::endl; 
	 os << "     local  position, phi, r: " 
	    << r.surface().toLocal(comp[i]->globalPosition())        << " , "
	    << r.surface().toLocal(comp[i]->globalPosition()).phi()  << " , "
	    << r.surface().toLocal(comp[i]->globalPosition()).perp() << std::endl; 
      }
   }

   return os;
}
