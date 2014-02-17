/** \file
 *
 *  $Date: 2009/03/01 21:21:22 $
 *  $Revision: 1.12 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */
 
#include "Alignment/MuonAlignment/interface/AlignableCSCChamber.h"

AlignableCSCChamber::AlignableCSCChamber(const GeomDet *geomDet): AlignableDet(geomDet)
{
   theStructureType = align::AlignableCSCChamber;
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
