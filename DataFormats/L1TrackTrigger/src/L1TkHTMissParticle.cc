#include "DataFormats/L1TrackTrigger/interface/L1TkHTMissParticle.h"

using namespace l1t ;

L1TkHTMissParticle::L1TkHTMissParticle()
{
}

L1TkHTMissParticle::L1TkHTMissParticle(
        const LorentzVector& p4,
        const double& etTotal,
	const edm::RefProd< L1TkJetParticleCollection >& jetCollRef,
        const edm::Ref< L1TkPrimaryVertexCollection >& avtxRef,
        int bx )
   : L1Candidate(p4 ),
     EtTot_( etTotal ),
     jetCollectionRef_( jetCollRef ),
     vtxRef_( avtxRef ),
     bx_( bx )
{

   if ( vtxRef_.isNonnull() ) {
	float z = getVtxRef() -> getZvertex() ;
	setVtx( z );
   }

}


