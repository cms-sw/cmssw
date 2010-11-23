#ifndef _FWPFLEGORECHIT_H_
#define _FWPFLEGORECHIT_H_

//
// Package:             Particle Flow
// Class:               FWPFLegoRecHit
// Original Author:     Simon Harris
//

#include "TEveBox.h"
#include "TEveCompound.h"
#include "TEveStraightLineSet.h"

#include "TEveCaloData.h"
#include "TEveChunkManager.h"

// User include files
#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "Fireworks/Core/interface/Context.h"

class FWPFEcalRecHitLegoProxyBuilder;

//-----------------------------------------------------------------------------
// LegoRechHit
//-----------------------------------------------------------------------------

class FWPFLegoRecHit
{
private:
   FWPFLegoRecHit( const FWPFLegoRecHit& );                    // Disable default
   const FWPFLegoRecHit& operator=( const FWPFLegoRecHit& );   // Disable default

   // ------------------------- Member Functions -------------------------------
   void setupEveBox( const std::vector<TEveVector> &corners );
   void convertToTower( std::vector<TEveVector> &corners, float scale );
   void buildTower( const std::vector<TEveVector> &corners, const FWViewContext *vc );
   void buildLineSet( const std::vector<TEveVector> &corners, const FWViewContext *vc );

   // --------------------------- Data Members ---------------------------------
   FWPFEcalRecHitLegoProxyBuilder* m_builder;

   TEveBox              *m_tower;
   TEveStraightLineSet  *m_ls;
   float                 m_energy;
   float                 m_et;


public:
   // -------------------- Constructor(s)/Destructors --------------------------
   FWPFLegoRecHit( const std::vector<TEveVector> &corners, TEveElement *comp, FWPFEcalRecHitLegoProxyBuilder*pb,
                   const FWViewContext *vc, float e, float et );
   virtual ~FWPFLegoRecHit(){}

   void updateScale( const FWViewContext *vc);
   void setSquareColor( Color_t c ) { m_ls->SetMarkerColor(c); m_ls->SetLineColor(kBlack); }

   TEveBox *getTower() { return m_tower; }
   void  setLine(int idx, float x1, float y1, float z1, float x2, float y2, float z2);
};
#endif
