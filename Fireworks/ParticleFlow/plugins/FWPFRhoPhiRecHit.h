#ifndef _FWPFRHOPHIRECHIT_H_
#define _FWPFRHOPHIRECHIT_H_

#include <math.h>

#include "TEveScalableStraightLineSet.h"
#include "TEveCompound.h"

// User include files
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"

//-----------------------------------------------------------------------------
// RhoPhiRecHit
//-----------------------------------------------------------------------------

class FWPFRhoPhiRecHit
{
   private:
      FWPFRhoPhiRecHit( const FWPFRhoPhiRecHit& );             // Stop default copy constructor
      FWPFRhoPhiRecHit& operator=( const FWPFRhoPhiRecHit& );  // Stop default assignment operator

   // ----------------------- Data Members ---------------------------
      bool                          m_hasChild;       // Determines whether the tower has a child or not
      float                         m_energy;
      float                         m_et;
      Double_t                      m_lPhi;
      Double_t                      m_rPhi;
      Double_t                      m_currentScale;
      TEveScalableStraightLineSet   *m_ls;
      FWPFRhoPhiRecHit              *m_child;         // Pointer to child (the next stacked tower)
      std::vector<TEveVector>       m_corners;

   // --------------------- Member Functions -------------------------
      void modScale( const FWViewContext *vc );

   public:
   // ---------------- Constructor(s)/Destructor ----------------------
      FWPFRhoPhiRecHit( FWProxyBuilderBase *pb, TEveCompound *iH, const FWViewContext *vc,
                        float E, float et, double lPhi, double rPhi, std::vector<TEveVector> &bCorners );
      virtual ~FWPFRhoPhiRecHit();

      void     updateScale( TEveScalableStraightLineSet *ls, Double_t scale, unsigned int i );
      void     updateScale( const FWViewContext *vc );
      void     addChild( FWProxyBuilderBase *pb, TEveCompound *itemHolder, const FWViewContext *vc, float E, float et );
      void     buildRecHit( FWProxyBuilderBase *pb, TEveCompound *itemHolder, const FWViewContext *vc, std::vector<TEveVector> &bCorners );
      void     clean();

      Double_t       getlPhi()                              { return m_lPhi;        }
      void           setHasChild( bool b )                  { m_hasChild = b;       }
      void           setCorners( int i, TEveVector vec )    { m_corners[i] = vec;   }
};
#endif
