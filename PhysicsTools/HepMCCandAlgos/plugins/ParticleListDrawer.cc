#include <memory>
#include <string>
#include <iostream>
#include <sstream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/Ref.h"

/**
   \class   ParticleListDrawer ParticleListDrawer.h "PhysicsTools/HepMCCandAlgos/plugins/ParticleListDrawer.h"

   \brief   Module to analyze the particle listing as provided by common event generators 

   Module to analyze the particle listing as provided by common event generators equivalent 
   to PYLIST(1) (from pythia). It is expected to run on vectors of  reo::GenParticles. For 
   an example of use have a look to: 

   PhysicsTools/HepMCCandAlgos/test/testParticleTreeDrawer.py

   Caveats: 
   Status 3 particles can have daughters both with status 2 and 3. In pythia this is not 
   the same mother-daughter. The relations are correct but special care has to be taken 
   when looking at mother-daughter relation which involve status 2 and 3 particles.
*/


using namespace std;
using namespace reco;
using namespace edm;

class ParticleListDrawer : public edm::EDAnalyzer {
  public:
    explicit ParticleListDrawer(const edm::ParameterSet & );
    ~ParticleListDrawer() {};
    void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  private:
    std::string getParticleName( int id ) const;

    edm::InputTag src_;
    edm::ESHandle<ParticleDataTable> pdt_;
    int maxEventsToPrint_; // Must be signed, because -1 is used for no limit
    unsigned int nEventAnalyzed_;
    bool printOnlyHardInteraction_;
    bool printVertex_;
    bool useMessageLogger_;
};

ParticleListDrawer::ParticleListDrawer(const edm::ParameterSet & pset) :
  src_(pset.getParameter<InputTag>("src")),
  maxEventsToPrint_ (pset.getUntrackedParameter<int>("maxEventsToPrint",1)),
  nEventAnalyzed_(0),
  printOnlyHardInteraction_(pset.getUntrackedParameter<bool>("printOnlyHardInteraction", false)),
  printVertex_(pset.getUntrackedParameter<bool>("printVertex", false)),
  useMessageLogger_(pset.getUntrackedParameter<bool>("useMessageLogger", false)) {
}

std::string ParticleListDrawer::getParticleName(int id) const
{
  const ParticleData * pd = pdt_->particle( id );
  if (!pd) {
    std::ostringstream ss;
    ss << "P" << id;
    return ss.str();
  } else
    return pd->name();
}

void ParticleListDrawer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {  
  Handle<reco::CandidateView> particles;
  iEvent.getByLabel (src_, particles );
  iSetup.getData( pdt_ );

  if(maxEventsToPrint_ < 0 || nEventAnalyzed_ < static_cast<unsigned int>(maxEventsToPrint_)) {
    ostringstream out;
    char buf[256];

    out << endl
	<< "[ParticleListDrawer] analysing particle collection " << src_.label() << endl;

    snprintf(buf, 256, " idx  |    ID -       Name |Stat|  Mo1  Mo2  Da1  Da2 |nMo nDa|    pt       eta     phi   |     px         py         pz        m     |"); 
    out << buf;
    if (printVertex_) {
      snprintf(buf, 256, "        vx       vy        vz     |");
      out << buf;
    }
    out << endl; 

    int idx  = -1;
    int iMo1 = -1;
    int iMo2 = -1;
    int iDa1 = -1;
    int iDa2 = -1;
    vector<const reco::Candidate *> cands;
    vector<const Candidate *>::const_iterator found = cands.begin();
    for(CandidateView::const_iterator p = particles->begin();
	p != particles->end(); ++ p) {
      cands.push_back(&*p);
    }

    for(CandidateView::const_iterator p  = particles->begin();
	p != particles->end(); 
	p ++) {
      if (printOnlyHardInteraction_ && p->status() != 3) continue;

      // Particle Name
      int id = p->pdgId();
      string particleName = getParticleName(id);
      
      // Particle Index
      idx =  p - particles->begin();

      // Particles Mothers and Daighters
      iMo1 = -1;
      iMo2 = -1;
      iDa1 = -1;
      iDa2 = -1;
      int nMo = p->numberOfMothers();
      int nDa = p->numberOfDaughters();

      found = find(cands.begin(), cands.end(), p->mother(0));
      if(found != cands.end()) iMo1 = found - cands.begin() ;

      found = find(cands.begin(), cands.end(), p->mother(nMo-1));
      if(found != cands.end()) iMo2 = found - cands.begin() ;
     
      found = find(cands.begin(), cands.end(), p->daughter(0));
      if(found != cands.end()) iDa1 = found - cands.begin() ;

      found = find(cands.begin(), cands.end(), p->daughter(nDa-1));
      if(found != cands.end()) iDa2 = found - cands.begin() ;

      char buf[256];
      snprintf(buf, 256,
	     " %4d | %5d - %10s | %2d | %4d %4d %4d %4d | %2d %2d | %7.3f %10.3f %6.3f | %10.3f %10.3f %10.3f %8.3f |",
             idx,
             p->pdgId(),
             particleName.c_str(),
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
      out << buf;

      if (printVertex_) {
        snprintf(buf, 256, " %10.3f %10.3f %10.3f |",
                 p->vertex().x(),
                 p->vertex().y(),
                 p->vertex().z());
        out << buf;
      }

      out << endl;
    }
    nEventAnalyzed_++;

    if (useMessageLogger_)
      LogVerbatim("ParticleListDrawer") << out.str();
    else
      cout << out.str();
  }
}

DEFINE_FWK_MODULE(ParticleListDrawer);

