#include "DataFormats/L1TrackTrigger/interface/L1TkEtMissParticle.h"

using namespace l1extra ;

L1TkEtMissParticle::L1TkEtMissParticle()
{
}

L1TkEtMissParticle::L1TkEtMissParticle(
        const LorentzVector& p4,
        EtMissType type,
        const double& etTotal,
        const double& etMissPU,
        const double& etTotalPU,
        const edm::Ref< L1TkPrimaryVertexCollection >& avtxRef,
        int bx )
   : LeafCandidate( ( char ) 0, p4 ),
     type_( type ),
     etTot_( etTotal ),
     etMissPU_ ( etMissPU ),
     etTotalPU_ ( etTotalPU ),
     vtxRef_( avtxRef ),
     bx_( bx )
{
}


