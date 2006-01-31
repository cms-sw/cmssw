#include "PhysicsTools/JetExamples/interface/MakeCaloJet.h"
#include "PhysicsTools/JetExamples/interface/ProtoJet.h"
#include "PhysicsTools/Candidate/interface/CompositeCandidate.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
using namespace aod;

void MakeCaloJet(const CandidateCollection &ctc, const std::vector<ProtoJet>& protoJets, CandidateCollection& caloJets){
  AddFourMomenta addp4;
   //Loop over the transient protoJets 
   for( std::vector<ProtoJet>::const_iterator i = protoJets.begin(); i != protoJets.end(); ++i ){
     const ProtoJet & p = *i;
     CompositeCandidate * jet = new CompositeCandidate;
     std::vector<const Candidate*> towers = p.getTowerList();
     for( std::vector<const Candidate*>::const_iterator i = towers.begin(); i != towers.end(); ++ i ) {
       jet->addDaughter( * *i );
     }
     jet->set( addp4 );
     caloJets.push_back( jet );
   }
};
