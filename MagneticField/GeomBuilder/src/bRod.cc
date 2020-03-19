/*
 *  See header file for a description of this class.
 *
 *  \author N. Amapane - INFN Torino
 */

#include "bRod.h"
#include "printUniqueNames.h"
#include "Utilities/BinningTools/interface/ClusterizingHistogram.h"
#include "MagneticField/Layers/interface/MagBRod.h"
#include "MagneticField/Layers/interface/MagVerbosity.h"
#include "Utilities/General/interface/precomputed_value_sort.h"

#include <iostream>

using namespace SurfaceOrientation;
using namespace magneticfield;

//The ctor is in charge of finding slabs inside the rod.
bRod::bRod(handles::const_iterator begin, handles::const_iterator end, bool debugVal)
    : volumes(begin, end), mrod(nullptr), debug(debugVal) {
  precomputed_value_sort(volumes.begin(), volumes.end(), ExtractZ());

  // Clusterize in Z
  const float resolution = 5.;  // cm
  float zmin = volumes.front()->center().z() - resolution;
  float zmax = volumes.back()->center().z() + resolution;
  ClusterizingHistogram hisZ(int((zmax - zmin) / resolution) + 1, zmin, zmax);

  if (debug)
    std::cout << "     Z slabs: " << zmin << " " << zmax << std::endl;

  handles::const_iterator first = volumes.begin();
  handles::const_iterator last = volumes.end();

  for (handles::const_iterator i = first; i != last; ++i) {
    hisZ.fill((*i)->center().z());
  }
  std::vector<float> zClust = hisZ.clusterize(resolution);

  if (debug)
    std::cout << "     Found " << zClust.size() << " clusters in Z, "
              << " slabs: " << std::endl;

  handles::const_iterator slabStart = first;
  handles::const_iterator separ = first;

  for (unsigned int i = 0; i < zClust.size() - 1; ++i) {
    float zSepar = (zClust[i] + zClust[i + 1]) / 2.f;
    while ((*separ)->center().z() < zSepar)
      ++separ;
    if (debug) {
      std::cout << "     Slab at: " << zClust[i] << " elements: " << separ - slabStart << " unique volumes: ";
      printUniqueNames(slabStart, separ);
    }

    slabs.push_back(bSlab(slabStart, separ, debug));
    slabStart = separ;
  }
  {
    if (debug) {
      std::cout << "     Slab at: " << zClust.back() << " elements: " << last - separ << " unique volumes: ";
      printUniqueNames(separ, last);
    }
    slabs.push_back(bSlab(separ, last, debug));
  }

  // Check that all slabs have the same dphi.
  std::vector<bSlab>::const_iterator i = slabs.begin();
  Geom::Phi<float> phimax = (*i).maxPhi();
  Geom::Phi<float> phimin = (*i).minPhi();
  for (++i; i != slabs.end(); ++i) {
    if (fabs(phimax - (*i).maxPhi()) > 0.001 || fabs(phimin - (*i).minPhi()) > 0.001) {
      if (debug) {
        std::cout << "*** WARNING: slabs in this rod have different dphi! minphi " << phimin;
        std::cout << " != " << (*i).minPhi() << " or maxphi " << phimax << " != " << (*i).maxPhi() << std::endl;
      }
    }
  }
}

MagBRod* bRod::buildMagBRod() const {
  if (mrod == nullptr) {
    std::vector<MagBSlab*> mSlabs;
    for (std::vector<bSlab>::const_iterator slab = slabs.begin(); slab != slabs.end(); ++slab) {
      mSlabs.push_back((*slab).buildMagBSlab());
    }
    mrod = new MagBRod(mSlabs, slabs.front().minPhi());  //FIXME
  }
  return mrod;
}
