/**
  \class    pat::PATGenCandsFromSimTracksProducer PATGenCandsFromSimTracksProducer.h "PhysicsTools/PatAlgos/interface/PATGenCandsFromSimTracksProducer.h"
  \brief    Produces reco::GenParticle from SimTracks

   The PATGenCandsFromSimTracksProducer produces GenParticles from SimTracks, so they can be used for MC matching.
   

  \author   Jordan Tucker (original module), Giovanni Petrucciani (PAT integration)
  \version  $Id: PATGenCandsFromSimTracksProducer.h,v 1.6 2008/05/06 20:13:50 gpetrucc Exp $
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "PhysicsTools/Utilities/interface/StringCutObjectSelector.h"
#include "SimGeneral/HepPDTRecord/interface/PdtEntry.h"


namespace pat {
class PATGenCandsFromSimTracksProducer : public edm::EDProducer {
public:
  explicit PATGenCandsFromSimTracksProducer(const edm::ParameterSet&);
  ~PATGenCandsFromSimTracksProducer() {}

private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() {}

  edm::InputTag src_;
  int setStatus_;
  std::set<int>         pdgIds_; // these are the ones we really use
  std::vector<PdtEntry> pdts_;   // these are needed before we get the EventSetup

  typedef StringCutObjectSelector<reco::GenParticle> StrFilter;
  std::auto_ptr<StrFilter> filter_;
  
};
}

using namespace std;
using namespace edm;
using namespace reco;
using pat::PATGenCandsFromSimTracksProducer;

PATGenCandsFromSimTracksProducer::PATGenCandsFromSimTracksProducer(const ParameterSet& cfg) :
  src_(cfg.getParameter<InputTag>("src")),            // source sim tracks & vertices
  setStatus_(cfg.getParameter<int32_t>("setStatus"))  // set status of GenParticle to this code
{
    // Possibly allow a list of particle types
    if (cfg.exists("particleTypes")) {
        pdts_ = cfg.getParameter<vector<PdtEntry> >("particleTypes");
    }

    // Possibly allow a string cut
    if (cfg.existsAs<string>("filter")) {
        string filter = cfg.getParameter<string>("filter");
        if (!filter.empty()) {
            filter_ = auto_ptr<StrFilter>(new StrFilter(filter));
        }
    }

    produces<GenParticleCollection>();
}

void PATGenCandsFromSimTracksProducer::beginJob(const EventSetup &iSetup) 
{
    if (!pdts_.empty()) {
        pdgIds_.clear();
        for (vector<PdtEntry>::iterator itp = pdts_.begin(), edp = pdts_.end(); itp != edp; ++itp) {
            itp->setup(iSetup); // decode string->pdgId and vice-versa
            pdgIds_.insert(abs(itp->pdgId()));
        }
        pdts_.clear();
    }
}

void PATGenCandsFromSimTracksProducer::produce(Event& event,
					    const EventSetup& eSetup) {
  // Simulated tracks (i.e. GEANT particles).
  Handle<SimTrackContainer> simtracks;
  event.getByLabel(src_, simtracks);

  // and the associated vertices
  Handle<SimVertexContainer> simvertices;
  event.getByLabel(src_, simvertices);

  // make the output collection
  auto_ptr<GenParticleCollection> cands(new GenParticleCollection);

  for (SimTrackContainer::const_iterator isimtrk = simtracks->begin();
          isimtrk != simtracks->end(); ++isimtrk) {

      // Skip PYTHIA tracks.
      if (isimtrk->genpartIndex() != -1) continue; 

      // Maybe apply the PdgId filter
      if (!pdgIds_.empty()) { // if we have a filter on pdg ids
           if (pdgIds_.find(abs(isimtrk->type())) == pdgIds_.end()) continue;
      }

      // Make up a GenParticleCandidate from the GEANT track info.
      int charge = static_cast<int>(isimtrk->charge());
      Particle::LorentzVector p4 = isimtrk->momentum();
      Particle::Point vtx; // = (0,0,0) by default
      if (!isimtrk->noVertex())
          vtx = (*simvertices)[isimtrk->vertIndex()].position();
      int status = 1;

      GenParticle genp(charge, p4, vtx, isimtrk->type(), status, true);

      // Maybe apply filter on the particle
      if (filter_.get() != 0) {
        if (!(*filter_)(genp)) continue;
      }
      cands->push_back(genp);
  }

  event.put(cands);
}

DEFINE_FWK_MODULE(PATGenCandsFromSimTracksProducer);
