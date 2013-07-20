// #include "Utilities/Configuration/interface/Architecture.h"

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/03/09 14:38:23 $
 *  $Revision: 1.5 $
 *  \author N. Amapane - INFN Torino
 */

#include "MagneticField/GeomBuilder/src/bSector.h"
#include "Utilities/BinningTools/interface/ClusterizingHistogram.h"
#include "MagneticField/Layers/interface/MagBSector.h"
#include "MagneticField/Layers/interface/MagVerbosity.h"

#include "Utilities/General/interface/precomputed_value_sort.h"

#include <algorithm>

using namespace SurfaceOrientation;
using namespace std;


// Default ctor needed to have arrays.
MagGeoBuilderFromDDD::bSector::bSector(){}

MagGeoBuilderFromDDD::bSector::~bSector(){}

// The ctor is in charge of finding rods inside the sector.
MagGeoBuilderFromDDD::bSector::bSector(handles::const_iterator begin,
					handles::const_iterator end) :
  volumes(begin,end),
  msector(0)
{
  if (MagGeoBuilderFromDDD::debug) cout << "   Sector at Phi  " <<  volumes.front()->center().phi() << " " 
		  << volumes.back()->center().phi() <<  endl;

  if (volumes.size() == 1) {
    if (MagGeoBuilderFromDDD::debug) { 
      cout << "   Rod at: 0 elements: " << end-begin
	   << " unique volumes: ";
      volumeHandle::printUniqueNames(begin,end);
    }
    rods.push_back(bRod(begin,end));
  } else {
    // Clusterize in phi. Use bin edge so that complete clusters can be 
    // easily found (not trivial using bin centers!)
    // Unfortunately this makes the result more sensitive to the 
    // "resolution" parameter...
    // To avoid +-pi boundary problems, take phi distance from min phi.
    // Caveat of implicit conversions of Geom::Phi!!!

    // Sort volumes in DELTA phi - i.e. phi(j)-phi(i) > 0 if j>1.
    precomputed_value_sort(volumes.begin(), volumes.end(),
			   ExtractPhiMax(), LessDPhi());

    const Geom::Phi<float> resolution(0.01); // rad
    Geom::Phi<float> phi0 = volumes.front()->maxPhi();
    float phiMin = -(float) resolution;
    float phiMax = volumes.back()->maxPhi() - phi0 + resolution; ///FIXME: (float) resolution; ??

    ClusterizingHistogram hisPhi( int((phiMax-phiMin)/resolution) + 1,
				  phiMin, phiMax);
    
    handles::const_iterator first = volumes.begin();
    handles::const_iterator last = volumes.end();  

    for (handles::const_iterator i=first; i!=last; ++i){
      hisPhi.fill((*i)->maxPhi()-phi0);
    }
    vector<float> phiClust = hisPhi.clusterize(resolution);

    if (MagGeoBuilderFromDDD::debug) cout << "     Found " << phiClust.size() << " clusters in Phi, "
		    << " rods: " << endl;

    handles::const_iterator rodStart = first;
    handles::const_iterator separ = first;
    
    float DZ = (*max_element(first,last,LessZ()))->maxZ() -
      (*min_element(first,last,LessZ()))->minZ();    

    float DZ1 = 0.;
    for (unsigned int i=0; i<phiClust.size(); ++i) {
      float phiSepar;
      if (i<phiClust.size()-1) {
	phiSepar = (phiClust[i] + phiClust[i+1])/2.f;
      } else {
	phiSepar = phiMax;
      }
      if (MagGeoBuilderFromDDD::debug) cout << "       cluster " << i
		      << " phisepar " << phiSepar <<endl;
      while (separ < last && (*separ)->maxPhi()-phi0 < phiSepar ) {
	DZ1 += ((*separ)->maxZ() - (*separ)->minZ());
 	if (MagGeoBuilderFromDDD::debug) cout << "         " << (*separ)->name << " "
			<< (*separ)->maxPhi()-phi0  << " "
			<< (*separ)->maxZ() << " " << (*separ)->minZ() << " "
			<< DZ1 << endl;
	++separ;
      }

      // FIXME: print warning for small discrepancies. Tolerance (below)
      // had to be increased since discrepancies sum to up to ~ 2 mm.
      if (fabs(DZ-DZ1) > 0.001 && fabs(DZ-DZ1) < 0.5) {
	if (MagGeoBuilderFromDDD::debug) cout << "*** WARNING: Z lenght mismatch by " << DZ-DZ1
			<< " " << DZ << " " << DZ1 << endl;

      }
      if (fabs(DZ-DZ1) > 0.25 ) { // FIXME hardcoded tolerance
	if (MagGeoBuilderFromDDD::debug) cout << "       Incomplete, use also next cluster: " 
			<< DZ << " " << DZ1 << " " << DZ-DZ1 << endl;
	DZ1 = 0.;
	continue;
      } else if (DZ1>DZ+0.05) { // Wrong: went past max lenght // FIXME hardcoded tolerance
	cout << " *** ERROR: bSector finding messed up." << endl;
	volumeHandle::printUniqueNames(rodStart, separ);
	DZ1 = 0.;
      } else {
	if (MagGeoBuilderFromDDD::debug) {
	  cout << "       Rod at: " << phiClust[i] <<" elements: "
	       << separ-rodStart << " unique volumes: ";
	  volumeHandle::printUniqueNames(rodStart, separ);
	}
	
	rods.push_back(bRod(rodStart, separ));
	rodStart = separ;
	DZ1 = 0.;
      }
    }

    if (MagGeoBuilderFromDDD::debug) cout << "-----------------------" << endl;

  }
}


MagBSector* MagGeoBuilderFromDDD::bSector::buildMagBSector() const{
  if (msector==0) {
    vector<MagBRod*> mRods;
    for (vector<bRod>::const_iterator rod = rods.begin();
	 rod!=rods.end(); ++rod) {
      mRods.push_back((*rod).buildMagBRod());
    }
    msector = new MagBSector(mRods, volumes.front()->minPhi()); //FIXME
  }
  return msector;
}
