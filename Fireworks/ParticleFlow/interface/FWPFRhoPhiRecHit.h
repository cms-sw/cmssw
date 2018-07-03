#ifndef _FWPFRHOPHIRECHIT_H_
#define _FWPFRHOPHIRECHIT_H_

// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWPFRhoPhiRecHit
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Simon Harris
//

// System include files
#include <cmath>
#include "TEveScalableStraightLineSet.h"
#include "TEveCompound.h"

// User include files
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"

//-----------------------------------------------------------------------------
// FWPFRhoPhiRecHit
//-----------------------------------------------------------------------------
class FWPFRhoPhiRecHit
{
   public:
   // ---------------- Constructor(s)/Destructor ----------------------
      FWPFRhoPhiRecHit( FWProxyBuilderBase *pb, TEveElement *iH, const FWViewContext *vc,
                        float E, float et, double lPhi, double rPhi, std::vector<TEveVector> &bCorners );
      virtual ~FWPFRhoPhiRecHit();

   // --------------------- Member Functions --------------------------
      void     updateScale( TEveScalableStraightLineSet *ls, Double_t scale, unsigned int i );
      void     updateScale( const FWViewContext *vc );
      void     addChild( FWProxyBuilderBase *pb, TEveElement *itemHolder, const FWViewContext *vc, float E, float et );
      void     buildRecHit( FWProxyBuilderBase *pb, TEveElement *itemHolder, const FWViewContext *vc, std::vector<TEveVector> &bCorners );
      void     clean();

   // ----------------------Accessor Methods --------------------------
      Double_t                       getlPhi()                              { return m_lPhi;        }
      TEveScalableStraightLineSet   *getLineSet()                           { return m_ls;          }
      void                           setHasChild( bool b )                  { m_hasChild = b;       }
      void                           setCorners( int i, TEveVector vec )    { m_corners[i] = vec;   }

   private:
      FWPFRhoPhiRecHit( const FWPFRhoPhiRecHit& );             // Stop default copy constructor
      FWPFRhoPhiRecHit& operator=( const FWPFRhoPhiRecHit& );  // Stop default assignment operator

   // ----------------------- Data Members ----------------------------
      bool                          m_hasChild;
      float                         m_energy;
      float                         m_et;
      Double_t                      m_lPhi;
      Double_t                      m_rPhi;
      TEveScalableStraightLineSet   *m_ls;
      FWPFRhoPhiRecHit              *m_child; 
      std::vector<TEveVector>       m_corners;
};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
