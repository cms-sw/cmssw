#include "RecoParticleFlow/PFProducer/interface/PFCandConnector.h"

 std::auto_ptr<reco::PFCandidateCollection>
      PFCandConnector::connect(std::auto_ptr<reco::PFCandidateCollection>& pfCand) {
      std::vector<bool> connected( pfCand->size(), false );
      // loop on primary
      for( unsigned int prim=0; prim!= pfCand->size(); ++prim) {
        if( isSecondary(pfCand->at(prim)) ) continue;
        // loop on secondary
        for( unsigned int sec=0; sec!= pfCand->size(); ++sec) {
            if( prim==sec || connected[sec] || !isSecondary(pfCand->at(sec))) continue;
            if( shouldBeLinked( pfCand->at(prim), pfCand->at(sec) ) ) {
                link( pfCand->at(prim), pfCand->at(sec) );
                connected[sec]=true;
            }
         }
         pfC_->push_back(pfCand->at(prim));
       }
       return pfC_;
 }

   bool PFCandConnector::isSecondary( const reco::PFCandidate& pf ) const {
     
     return false;
     /*
       // nuclear
       if( pf.flag( reco::PFCandidate::T_FROM_NUCLINT ) ) return true;
      // conversion
       // || pf.flag( reco::PFCandidate::T_FROM_GAMMACONV ) ) return true;
       else return false;
     */
   }

   void PFCandConnector::link( reco::PFCandidate& primPFC, reco::PFCandidate& secPFC ) const {
         // TODO: Add a flag in PFCandidate :GAMMA_TO_GAMMACONV
         // nuclear
     // if( primPFC.flag( reco::PFCandidate::T_TO_NUCLINT ) )
     //             secPFC.setFlag( reco::PFCandidate::T_FROM_NUCLINT, true );
         // conversion
         //else if( primPFC.flag( reco::PFCandidate::GAMMA_TO_GAMMACONV) )
         //               secPFC.setFlag( T_FROM_GAMMACONV, true );

         primPFC.addDaughter(secPFC);
   }

    bool PFCandConnector::shouldBeLinked( const reco::PFCandidate& primPFC,
                                          const reco::PFCandidate& secPFC ) const {
    // TODO : methods PFCandidate::conversionRef()
        // nuclear
      //       if(primPFC.nuclearRef().isNonnull() && secPFC.nuclearRef().isNonnull() ) {
      //   if( primPFC.nuclearRef() == secPFC.nuclearRef()) return true;
      //   }
        // conversion
        //else if( primPFC.conversionRef().isValid() && secPFC.conversionRef().isValid() ){
        // if( primPFC.conversionRef() == secPFC.conversionRef()) return true;
        //}
       return false;
    }

