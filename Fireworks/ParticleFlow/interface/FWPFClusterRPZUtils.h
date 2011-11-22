#ifndef _FWPFCLUSTERRPZUTILS_H_
#define _FWPFCLUSTERRPZUTILS_H_

// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWPFClusterRPZUtils
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Simon Harris
//       Created:    17/02/2011
//

// System include files
#include "TEveScalableStraightLineSet.h"

// User include files
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "Fireworks/Core/interface/FWViewContext.h"

struct ScalableLines
{
   ScalableLines( TEveScalableStraightLineSet *ls, float et, float e, const FWViewContext *vc ) :
   m_ls(ls), m_et(et), m_energy(e), m_vc(vc){}

   TEveScalableStraightLineSet *m_ls;
   float m_et, m_energy;
   const FWViewContext *m_vc;
};

//-----------------------------------------------------------------------------
// FWPFClusterRPZUtils
//-----------------------------------------------------------------------------
class FWPFClusterRPZUtils
{
   public:
   // ---------------- Constructor(s)/Destructor ----------------------
      FWPFClusterRPZUtils(){}
      virtual ~FWPFClusterRPZUtils(){}

   // --------------------- Member Functions --------------------------
      float calculateEt( const reco::PFCluster&, float e );
      TEveScalableStraightLineSet *buildRhoPhiClusterLineSet( const reco::PFCluster&, const FWViewContext*, float r );
      TEveScalableStraightLineSet *buildRhoPhiClusterLineSet( const reco::PFCluster&, const FWViewContext*, 
                                                              float e, float et, float r );
      TEveScalableStraightLineSet *buildRhoZClusterLineSet( const reco::PFCluster&, const FWViewContext*, 
                                                            float caloTransAngle, float r, float z );
      TEveScalableStraightLineSet *buildRhoZClusterLineSet( const reco::PFCluster&, const FWViewContext*,
                                                            float caloTransAngle, float e, float et, float r, float z );

   private:
      FWPFClusterRPZUtils( const FWPFClusterRPZUtils& );                   // Disable default copy constructor
      const FWPFClusterRPZUtils& operator=( const FWPFClusterRPZUtils& );  // Disable default assignment operator
};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
