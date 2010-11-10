#ifndef _FWPFRHOPHIRECHIT_H_
#define _FWPFRHOPHIRECHIT_H_

#include <math.h>

#include "TEveScalableStraightLineSet.h"
#include "TEvePointSet.h"
#include "TEveCompound.h"

// User includes
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"

//-----------------------------------------------------------------------------
// RhoPhiRecHit
//-----------------------------------------------------------------------------

class FWPFRhoPhiRecHit
{
	private:
		FWPFRhoPhiRecHit( const FWPFRhoPhiRecHit& );			// Stop default copy constructor
		FWPFRhoPhiRecHit& operator=( const FWPFRhoPhiRecHit& );	// Disable default assignment operator

	// ----------------------- Data Members ---------------------------
		Double_t				m_currentScale;
		double					m_lPhi;
		double					m_rPhi;
		float 					m_energy;
		float 					m_et;
		const FWViewContext		*m_vc;
		TEveVector				m_centre;
		std::vector<TEveVector> m_corners;
		std::vector<TEveVector> m_creationPoint;
		TEveScalableStraightLineSet *m_ls;
		std::vector<FWPFRhoPhiRecHit*> m_children;
		TEveCompound			*m_itemHolder;

	// --------------------- Member Functions -------------------------
		void 					CalculateEt();
		void					PushCreationPoint( TEveVector );
		void					ModScale();

	public:
	// ---------------- Constructor(s)/Destructor ----------------------
		FWPFRhoPhiRecHit( FWProxyBuilderBase *pb, TEveCompound *iH, const FWViewContext *vc, const TEveVector &centre, 
						  float E, double lPhi, double rPhi, bool build = false );
		virtual ~FWPFRhoPhiRecHit();

		void 					BuildRecHit( FWProxyBuilderBase *pb, TEveCompound *itemHolder );
		void 					Add( FWProxyBuilderBase *pb, TEveCompound *iCompound, const FWViewContext *vc, float E );
		void 					updateScale( TEveScalableStraightLineSet *ls, Double_t scale, unsigned int i );

		void					SetEnergy( float E ) 				{ m_energy = E; 			}
		void					SetEt( float et )					{ m_et = et; 				}
		void					SetCorners( int i, TEveVector vec )	{ m_corners[i] = vec;		}

		float					GetlPhi()							{ return m_lPhi;			}
		float					GetrPhi()							{ return m_rPhi;			}
		float					GetEt()								{ return m_et;				}
		float					GetEnergy()							{ return m_energy;			}
		TEveVector				GetCentre()							{ return m_centre;			}
		TEveVector				GetCorners( int i )					{ return m_corners[i];		}
		TEveVector				GetCreationPoint( int i )			{ return m_creationPoint[i];}

		TEveScalableStraightLineSet *GetLineSet()					{ return m_ls;				}
};
#endif
