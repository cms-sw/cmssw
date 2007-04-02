// system include files
#include <memory>
#include <string>
#include <iostream>
#include <vector>

// user include files
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/GenJet.h"

#include "PhysicsTools/JetMCUtils/interface/JetMCTag.h"
#include "PhysicsTools/JetMCUtils/interface/CandMCTag.h"

#include "PhysicsTools/JetMCAlgos/interface/printGenEvent.h"

using namespace std;
using namespace reco;
using namespace edm;
using namespace JetMCTagUtils;
using namespace CandMCTagUtils;

printGenEvent::printGenEvent(const edm::ParameterSet& iConfig)
{
  source_ = iConfig.getParameter<InputTag> ("src");
}

void printGenEvent::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  cout << "[printGenJet] analysing event " << iEvent.id() << endl;
  
  try {
    iEvent.getByLabel (source_,genJets);
    iEvent.getByLabel ("genParticleCandidates", particles );
    iSetup.getData( pdt_ );
  } catch(std::exception& ce) {
    cerr << "[printGenJet] caught std::exception " << ce.what() << endl;
    return;
  }

  //
  // Printout for GenEvent
  //
  cout << endl;
  cout << "**********************" << endl;
  cout << "* GenEvent           *" << endl;
  cout << "**********************" << endl;

  printf(" idx  |    ID -       Name |Stat|  Mo1  Mo2  Da1  Da2 |nMo nDa|    pt       eta     phi   |isB isC|\n");
  int idx  = -1;
  int iMo1 = -1;
  int iMo2 = -1;
  int iDa1 = -1;
  int iDa2 = -1;
  std::vector<const reco::Candidate *> cands_;
  cands_.clear();
  vector<const Candidate *>::const_iterator found = cands_.begin();
  for( CandidateCollection::const_iterator p = particles->begin();
       p != particles->end(); ++ p ) {
    cands_.push_back( & * p );
  }

  for( CandidateCollection::const_iterator p  = particles->begin();
                                           p != particles->end(); 
                                           p ++) {
      // Particle Name
      int id = p->pdgId();
      const ParticleData * pd = pdt_->particle( id );
      const char* particleName = (char*)( pd->name().c_str() );  

      // Particle Index
      idx =  p - particles->begin();

      // Particles Mothers and Daighters
      iMo1 = -1;
      iMo2 = -1;
      iDa1 = -1;
      iDa2 = -1;
      int nMo = p->numberOfMothers();
      int nDa = p->numberOfDaughters();

      found = find( cands_.begin(), cands_.end(), p->mother(0) );
      if ( found != cands_.end() ) iMo1 = found - cands_.begin() ;

      found = find( cands_.begin(), cands_.end(), p->mother(nMo-1) );
      if ( found != cands_.end() ) iMo2 = found - cands_.begin() ;
     
      found = find( cands_.begin(), cands_.end(), p->daughter(0) );
      if ( found != cands_.end() ) iDa1 = found - cands_.begin() ;

      found = find( cands_.begin(), cands_.end(), p->daughter(nDa-1) );
      if ( found != cands_.end() ) iDa2 = found - cands_.begin() ;

      // particle decay flavour
      bool isB = decayFromBHadron(*p);
      bool isC = decayFromCHadron(*p);

      printf(" %4d | %5d - %10s | %2d | %4d %4d %4d %4d | %2d %2d | %7.3f %10.3f %6.3f |  %1d  %1d |\n",
             idx,
             p->pdgId(),
             particleName,
             p->status(),
             iMo1,iMo2,iDa1,iDa2,nMo,nDa,
             p->pt(),
             p->eta(),
             p->phi(),
             isB,isC  );
  }
  
  //
  // Printout for GenJet Collection
  //
  cout << endl;
  cout << "**********************" << endl;
  cout << "* GenJetCollection   *" << endl;
  cout << "**********************" << endl;
  for( CandidateCollection::const_iterator f  = genJets->begin();
                                           f != genJets->end();
                                           f++) {

    double bRatio = EnergyRatioFromBHadrons(*f);
    double cRatio = EnergyRatioFromCHadrons(*f);

    printf("[GenJetTest] (pt,eta,phi | bRatio cRatio) = %6.2f %5.2f %5.2f | %5.3f %5.3f |\n",
	     f->et(),
	     f->eta(),
	     f->phi(),
             bRatio,
             cRatio  );

    for( Candidate::const_iterator c  = f->begin();
                                   c != f->end();
                                   c ++) {
      const Candidate* theMasterClone;
      bool isB = false;
      bool isC = false;
      if (c->hasMasterClone ()) {
        theMasterClone = c->masterClone().get();
        isB = decayFromBHadron(*theMasterClone);
        isC = decayFromCHadron(*theMasterClone);
      }
      printf("        [Constituents] (pt,eta,phi | isB,isC) = %6.2f %5.2f %5.2f | %1d %1d |\n",
                c->et(),
                c->eta(),
                c->phi(),
                isB,isC  );
    }       
  }
}
