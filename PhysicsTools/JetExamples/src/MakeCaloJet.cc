#include "PhysicsTools/JetExamples/interface/MakeCaloJet.h"
#include "PhysicsTools/JetExamples/interface/ProtoJet.h"
#include "PhysicsTools/Candidate/interface/CompositeRefCandidate.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
using namespace reco;

void MakeCaloJet(const CandidateCollection &ctc, const std::vector<ProtoJet>& protoJets, CandidateCollection& caloJets){
  AddFourMomenta addp4;
   //Loop over the transient protoJets 
   for( std::vector<ProtoJet>::const_iterator i = protoJets.begin(); i != protoJets.end(); ++i ){
     const ProtoJet & p = *i;
     CompositeRefCandidate * jet = new CompositeRefCandidate;
     CandidateRefs towers = p.getTowerList();
     for( CandidateRefs::iterator i = towers.begin(); i != towers.end(); ++ i ) {
       jet->addDaughter( * i );
     }
     jet->set( addp4 );
     caloJets.push_back( jet );
   }
};
