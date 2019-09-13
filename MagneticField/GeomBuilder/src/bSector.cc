/*
 *  See header file for a description of this class.
 *
 *  \author N. Amapane - INFN Torino
 */

#include "bSector.h"
#include "printUniqueNames.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "Utilities/BinningTools/interface/ClusterizingHistogram.h"
#include "MagneticField/Layers/interface/MagBSector.h"
#include "MagneticField/Layers/interface/MagVerbosity.h"
#include "Utilities/General/interface/precomputed_value_sort.h"

#include <algorithm>
#include <iostream>

using namespace SurfaceOrientation;
using namespace std;
using namespace magneticfield;

// Default ctor needed to have arrays.
bSector::bSector() : debug(false) {}

// The ctor is in charge of finding rods inside the sector.
bSector::bSector(handles::const_iterator begin, handles::const_iterator end, bool debugVal)
    : volumes(begin, end), msector(nullptr), debug(debugVal) {
  if (debug)
    cout << "   Sector at Phi  " << volumes.front()->center().phi() << " " << volumes.back()->center().phi() << endl;

  if (volumes.size() == 1) {
    if (debug) {
      cout << "   Rod at: 0 elements: " << end - begin << " unique volumes: ";
      printUniqueNames(begin, end);
    }
    rods.push_back(bRod(begin, end, debug));
  } else {
    // Clusterize in phi. Use bin edge so that complete clusters can be
    // easily found (not trivial using bin centers!)
    // Unfortunately this makes the result more sensitive to the
    // "resolution" parameter...
    // To avoid +-pi boundary problems, take phi distance from min phi.
    // Caveat of implicit conversions of Geom::Phi!!!

    // Sort volumes in DELTA phi - i.e. phi(j)-phi(i) > 0 if j>1.
    precomputed_value_sort(volumes.begin(), volumes.end(), ExtractPhiMax(), LessDPhi());

    const float resolution(0.01);  // rad
    Geom::Phi<float> phi0 = volumes.front()->maxPhi();
    float phiMin = -resolution;  ///FIXME: (float) resolution; ??

    // Careful -- Phi overloads arithmetic operators and will wrap around each step of a calculation,
    // so use casts to prevent intermediate value wrap-arounds.
    float phiTmp = static_cast<float>(volumes.back()->maxPhi()) - static_cast<float>(phi0) + resolution;
    const float phiMax = angle0to2pi::make0To2pi(phiTmp);  // Ensure 0-2pi

    if (debug) {
      cout << "volumes size = " << volumes.size();
      cout << ", phi0 = " << phi0 << ", volumes.back()->maxPhi() = " << volumes.back()->maxPhi();
      cout << ", phiMin = " << phiMin << endl << "phiMax = " << phiMax;
      cout << ", int((phiMax - phiMin) / resolution) + 1 = " << int((phiMax - phiMin) / resolution) + 1 << endl;
    }
    ClusterizingHistogram hisPhi(int((phiMax - phiMin) / resolution) + 1, phiMin, phiMax);

    handles::const_iterator first = volumes.begin();
    handles::const_iterator last = volumes.end();

    for (handles::const_iterator i = first; i != last; ++i) {
      hisPhi.fill((*i)->maxPhi() - phi0);
    }
    vector<float> phiClust = hisPhi.clusterize(resolution);

    if (debug)
      cout << "     Found " << phiClust.size() << " clusters in Phi, "
           << " rods: " << endl;

    handles::const_iterator rodStart = first;
    handles::const_iterator separ = first;

    float DZ = (*max_element(first, last, LessZ()))->maxZ() - (*min_element(first, last, LessZ()))->minZ();

    float DZ1 = 0.;
    for (unsigned int i = 0; i < phiClust.size(); ++i) {
      float phiSepar;
      if (i < phiClust.size() - 1) {
        phiSepar = (phiClust[i] + phiClust[i + 1]) / 2.f;
      } else {
        phiSepar = phiMax;
      }
      if (debug)
        cout << "       cluster " << i << " phisepar " << phiSepar << endl;
      while (separ < last && (*separ)->maxPhi() - phi0 < phiSepar) {
        DZ1 += ((*separ)->maxZ() - (*separ)->minZ());
        if (debug)
          cout << "         " << (*separ)->name << " " << (*separ)->maxPhi() - phi0 << " " << (*separ)->maxZ() << " "
               << (*separ)->minZ() << " " << DZ1 << endl;
        ++separ;
      }

      // FIXME: print warning for small discrepancies. Tolerance (below)
      // had to be increased since discrepancies sum to up to ~ 2 mm.
      if (fabs(DZ - DZ1) > 0.001 && fabs(DZ - DZ1) < 0.5) {
        if (debug)
          cout << "*** WARNING: Z lenght mismatch by " << DZ - DZ1 << " " << DZ << " " << DZ1 << endl;
      }
      if (fabs(DZ - DZ1) > 0.25) {  // FIXME hardcoded tolerance
        if (debug)
          cout << "       Incomplete, use also next cluster: " << DZ << " " << DZ1 << " " << DZ - DZ1 << endl;
        DZ1 = 0.;
        continue;
      } else if (DZ1 > DZ + 0.05) {  // Wrong: went past max lenght // FIXME hardcoded tolerance
        cout << " *** ERROR: bSector finding messed up." << endl;
        printUniqueNames(rodStart, separ);
        DZ1 = 0.;
      } else {
        if (debug) {
          cout << "       Rod at: " << phiClust[i] << " elements: " << separ - rodStart << " unique volumes: ";
          printUniqueNames(rodStart, separ);
        }

        rods.push_back(bRod(rodStart, separ, debug));
        rodStart = separ;
        DZ1 = 0.;
      }
    }

    if (rods.empty())
      cout << " *** ERROR: bSector has no rods " << DZ << " " << DZ1 << endl;
    if (debug)
      cout << "-----------------------" << endl;
  }
}

MagBSector* bSector::buildMagBSector() const {
  if (msector == nullptr) {
    vector<MagBRod*> mRods;
    for (vector<bRod>::const_iterator rod = rods.begin(); rod != rods.end(); ++rod) {
      mRods.push_back((*rod).buildMagBRod());
    }
    msector = new MagBSector(mRods, volumes.front()->minPhi());  //FIXME
    // Never deleted. When is it safe to delete it?
  }
  return msector;
}
