// 
// PYLIST(1) equivalent to be used with GenParticleCandidate
// Caveats: 
// Status 3 particles can have daughter both with status 2 and 3
// In Pythia this is not the same
// mother-daughter relations are corrects but special care
// have to be taken when looking at mother-daughter relation which
// involve status 2 and 3 particles
//
// Author: Attilio  
// Date: 13.06.2007
//
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/Ref.h"

// system include files
#include <memory>
#include <string>
#include <iostream>
//#include <vector>

using namespace std;
using namespace reco;
using namespace edm;

class ParticleListDrawer : public edm::EDAnalyzer {
  public:
    explicit ParticleListDrawer(const edm::ParameterSet & ) {};
    ~ParticleListDrawer() {};
    void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  private:

    edm::InputTag source_;
    edm::Handle<reco::CandidateCollection> particles;
    edm::ESHandle<ParticleDataTable> pdt_;

};

void ParticleListDrawer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  cout << "[ParticleListDrawer] analysing event " << iEvent.id() << endl;
  
  try {
    iEvent.getByLabel ("genParticleCandidates", particles );
    iSetup.getData( pdt_ );
  } catch(std::exception& ce) {
    cerr << "[ParticleListDrawer] caught std::exception " << ce.what() << endl;
    return;
  }

  //
  // Printout for GenEvent
  //
  cout << endl;
  cout << "**********************" << endl;
  cout << "* GenEvent           *" << endl;
  cout << "**********************" << endl;

  printf(" idx  |    ID -       Name |Stat|  Mo1  Mo2  Da1  Da2 |nMo nDa|    pt       eta     phi   |     px         py         pz        m     |\n");
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

      printf(" %4d | %5d - %10s | %2d | %4d %4d %4d %4d | %2d %2d | %7.3f %10.3f %6.3f | %10.3f %10.3f %10.3f %8.3f |\n",
             idx,
             p->pdgId(),
             particleName,
             p->status(),
             iMo1,iMo2,iDa1,iDa2,nMo,nDa,
             p->pt(),
             p->eta(),
             p->phi(),
             p->px(),
             p->py(),
             p->pz(),
             p->mass()
            );
  }
  
}

DEFINE_FWK_MODULE( ParticleListDrawer );

