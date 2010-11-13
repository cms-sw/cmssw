#ifndef _LEGORECHIT_H_
#define _LEGORECHIT_H_

//
// Package:             Particle Flow
// Class:               LegoRecHit
// Original Author:     Simon Harris
// $Id: LegoRecHit.h,v 1.1 2010/09/14 14:36:58 sharris Exp $
//

#include <math.h>

#include "TEveBox.h"
#include "TEveLine.h"
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"


//-----------------------------------------------------------------------------
// LegoRechHit
//-----------------------------------------------------------------------------

class LegoRecHit
{
private:
   LegoRecHit( const LegoRecHit& );          // Disable default copy constructor
   LegoRecHit& operator=( const LegoRecHit& ); // Disable default assignment operator

 /*************************************************************\
(                       MEMBER FUNCTIONS                        )
 \*************************************************************/

   void setupEveBox( TEveBox *eveBox, size_t numCorners, const std::vector<TEveVector> &corners );
   std::vector<TEveVector> convertToTower( const std::vector<TEveVector> &corners, float e_t, float scale );

 /*************************************************************\
(                        DATA MEMBERS                           )
 \*************************************************************/

   TEveBox *tower;
   TEveLine *lineSet;

 /***************************************************************\
(                   CONSTRUCTOR(S)/DESTRUCTOR                     )
 \***************************************************************/

public:
   LegoRecHit( size_t numCorners, const std::vector<TEveVector> &corners, TEveElement *comp, FWProxyBuilderBase *pb, float e_t, float scale );
   LegoRecHit(){}
   virtual ~LegoRecHit(){}

   void setSquareColor( Color_t c ) { lineSet->SetMainColor( c ); }

};
#endif
