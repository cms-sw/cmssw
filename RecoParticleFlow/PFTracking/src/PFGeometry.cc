#include "RecoParticleFlow/PFTracking/interface/PFGeometry.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"

PFGeometry::PFGeometry()
{
  if (!innerRadius_.size()) {
    // All distances are in cm
    // BeamPipe
    innerRadius_.push_back(2.5);
    outerRadius_.push_back(2.5);
    innerZ_.push_back(0.);
    outerZ_.push_back(500.);
    // PS1
    innerRadius_.push_back(45.0);
    outerRadius_.push_back(125.0);
    innerZ_.push_back(303.16);
    outerZ_.push_back(303.16);
    // PS2
    innerRadius_.push_back(45.0);
    outerRadius_.push_back(125.0);
    innerZ_.push_back(307.13);
    outerZ_.push_back(307.13);
    // ECALBarrel
    innerRadius_.push_back(129.0);
    outerRadius_.push_back(175.0);
    innerZ_.push_back(0.);
    outerZ_.push_back(304.5);
    // ECALEndcap
    innerRadius_.push_back(31.6);
    outerRadius_.push_back(171.1);
    innerZ_.push_back(317.0);
    outerZ_.push_back(388.0);
    // HCALBarrel
    innerRadius_.push_back(183.0);
    outerRadius_.push_back(285.0);
    innerZ_.push_back(0.);
    outerZ_.push_back(433.2);
    // HCALEndcap
    innerRadius_.push_back(31.6); // !!! Do not use : Probably wrong !!!    
    outerRadius_.push_back(285.0); // !!! Do not use : Probably wrong !!! 
    innerZ_.push_back(388.0);
    outerZ_.push_back(560.0);

    // HO Barrel
    innerRadius_.push_back(387.6);
    outerRadius_.push_back(410.2);
    innerZ_.push_back(0.);
    outerZ_.push_back(700.25);

    // Define reference surfaces
    tanTh_.push_back(innerRadius_[BeamPipe]/outerZ_[BeamPipe]);
    tanTh_.push_back(outerRadius_[PS1]/outerZ_[PS1]);
    tanTh_.push_back(outerRadius_[PS2]/outerZ_[PS2]);
    tanTh_.push_back(innerRadius_[ECALBarrel]/innerZ_[ECALEndcap]);
    tanTh_.push_back(innerRadius_[HCALBarrel]/innerZ_[HCALEndcap]);
    tanTh_.push_back(outerRadius_[HCALBarrel]/outerZ_[HCALEndcap]);
  }
}
