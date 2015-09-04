/**
  \class    pat::PATGenCandsFromSimTracksProducer PATGenCandsFromSimTracksProducer.h "PhysicsTools/PatAlgos/interface/PATGenCandsFromSimTracksProducer.h"
  \brief    Produces reco::GenParticle from SimTracks

   The PATGenCandsFromSimTracksProducer produces GenParticles from SimTracks, so they can be used for MC matching.


  \author   Jordan Tucker (original module), Giovanni Petrucciani (PAT integration)
  \version  $Id: PATGenCandsFromSimTracksProducer.cc,v 1.8 2010/10/20 23:09:25 wmtan Exp $
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

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "SimGeneral/HepPDTRecord/interface/PdtEntry.h"

#include <ext/algorithm>

namespace pat {
  class PATGenCandsFromSimTracksProducer : public edm::stream::EDProducer<> {
  public:
    explicit PATGenCandsFromSimTracksProducer(const edm::ParameterSet&);
    ~PATGenCandsFromSimTracksProducer() {}
    
  private:
    virtual void produce(edm::Event&, const edm::EventSetup&) override;
    
    bool firstEvent_;
    edm::EDGetTokenT<edm::SimTrackContainer> simTracksToken_;
    edm::EDGetTokenT<edm::SimVertexContainer> simVertexToken_;
    int setStatus_;
    std::set<int>         pdgIds_; // these are the ones we really use
    std::vector<PdtEntry> pdts_;   // these are needed before we get the EventSetup
    std::set<int>         motherPdgIds_; // these are the ones we really use
    std::vector<PdtEntry> motherPdts_;   // these are needed before we get the EventSetup
    
    typedef StringCutObjectSelector<reco::GenParticle> StrFilter;
    const StrFilter filter_;
    
    /// If true, I'll try to make a link from the GEANT particle to a GenParticle
    bool makeMotherLink_;
    /// If true, I'll save GenParticles corresponding to the ancestors of this GEANT particle. Common ancestors are only written once.
    bool writeAncestors_;
    
    /// Collection of GenParticles I need to make refs to. It must also have its associated vector<int> of barcodes, aligned with them.
    edm::EDGetTokenT<reco::GenParticleCollection> genParticlesToken_;
    edm::EDGetTokenT<std::vector<int> > genBarcodesToken_;
    
  /// Global context for all recursive methods
    struct GlobalContext {
      GlobalContext(const edm::SimTrackContainer &simtks1,
                    const edm::SimVertexContainer &simvtxs1,
                    const edm::Handle<reco::GenParticleCollection> &gens1,
                    const edm::Handle<std::vector<int> >           &genBarcodes1,
                    bool                                            barcodesAreSorted1,
                    reco::GenParticleCollection                     & output1,
                    const edm::RefProd<reco::GenParticleCollection> & refprod1) :
        simtks(simtks1), simvtxs(simvtxs1),
        gens(gens1), genBarcodes(genBarcodes1), barcodesAreSorted(barcodesAreSorted1),
        output(output1), refprod(refprod1), simTksProcessed() {}
      // GEANT info
      const edm::SimTrackContainer &simtks;
      const edm::SimVertexContainer &simvtxs;
      // PYTHIA info
      const edm::Handle<reco::GenParticleCollection> &gens;
      const edm::Handle<std::vector<int> >           &genBarcodes;
      bool                                            barcodesAreSorted;
      // MY OUTPUT info
      reco::GenParticleCollection                     & output;
      const edm::RefProd<reco::GenParticleCollection> & refprod;
      // BOOK-KEEPING
      std::map<unsigned int,int> simTksProcessed; // key = sim track id;
      // val = 0: not processed;
      //       i>0:  (index+1) in my output
      //       i<0: -(index+1) in pythia [NOT USED]
    };
    
    /// Find the mother of a given GEANT track (or NULL if it can't be found).
    const SimTrack * findGeantMother(const SimTrack &tk, const GlobalContext &g) const ;
    /// Find the GenParticle reference for a given GEANT or PYTHIA track.
    ///  - if the track corresponds to a PYTHIA particle, return a ref to that particle
    ///  - otherwise, if this simtrack has no mother simtrack, return a null ref
    ///  - otherwise, if writeAncestors is true,  make a GenParticle for it and return a ref to it
    ///  - otherwise, if writeAncestors is false, return the ref to the GEANT mother of this track
    edm::Ref<reco::GenParticleCollection> findRef(const SimTrack &tk, GlobalContext &g) const ;
    
    /// Used by findRef if the track is a PYTHIA particle
    edm::Ref<reco::GenParticleCollection> generatorRef_(const SimTrack &tk, const GlobalContext &g) const ;
    /// Make a GenParticle for this SimTrack, with a given mother
    reco::GenParticle makeGenParticle_(const SimTrack &tk, const edm::Ref<reco::GenParticleCollection> & mother, const GlobalContext &g) const ;
    
    
    
    struct LessById {
      bool operator()(const SimTrack &tk1, const SimTrack &tk2) const { return tk1.trackId() < tk2.trackId(); }
      bool operator()(const SimTrack &tk1, unsigned int    id ) const { return tk1.trackId() < id;            }
      bool operator()(unsigned int     id, const SimTrack &tk2) const { return id            < tk2.trackId(); }
    };    
  };
}

using namespace std;
using namespace edm;
using namespace reco;
using pat::PATGenCandsFromSimTracksProducer;

PATGenCandsFromSimTracksProducer::PATGenCandsFromSimTracksProducer(const ParameterSet& cfg) :
  firstEvent_(true),
  simTracksToken_(consumes<SimTrackContainer>(cfg.getParameter<InputTag>("src"))),            // source sim tracks
  simVertexToken_(consumes<SimVertexContainer>(cfg.getParameter<InputTag>("src"))),            // source sim  vertices
  setStatus_(cfg.getParameter<int32_t>("setStatus")), // set status of GenParticle to this code
  filter_( cfg.existsAs<string>("filter") ? cfg.getParameter<string>("filter") : std::string("1 == 1") ),
  makeMotherLink_(cfg.existsAs<bool>("makeMotherLink") ? cfg.getParameter<bool>("makeMotherLink") : false),
  writeAncestors_(cfg.existsAs<bool>("writeAncestors") ? cfg.getParameter<bool>("writeAncestors") : false),
  genParticlesToken_(mayConsume<GenParticleCollection>(cfg.getParameter<InputTag>("genParticles"))),
  genBarcodesToken_(mayConsume<std::vector<int> >(cfg.getParameter<InputTag>("genParticles")))
{
    // Possibly allow a list of particle types
    if (cfg.exists("particleTypes")) {
        pdts_ = cfg.getParameter<vector<PdtEntry> >("particleTypes");
    }
    if (cfg.exists("motherTypes")) {
        motherPdts_ = cfg.getParameter<vector<PdtEntry> >("motherTypes");
    }
    
    if (writeAncestors_ && !makeMotherLink_) {
        edm::LogWarning("Configuration") << "PATGenCandsFromSimTracksProducer: " <<
            "you have set 'writeAncestors' to 'true' and 'makeMotherLink' to false;" <<
            "GEANT particles with generator level (e.g.PYHIA) mothers won't have mother links.\n";
    }
    produces<GenParticleCollection>();
}

const SimTrack *
PATGenCandsFromSimTracksProducer::findGeantMother(const SimTrack &tk, const GlobalContext &g) const {
   assert(tk.genpartIndex() == -1); // MUST NOT be called with a PYTHIA track
   if (!tk.noVertex()) {
       const SimVertex &vtx = g.simvtxs[tk.vertIndex()];
       if (!vtx.noParent()) {
           unsigned int idx = vtx.parentIndex();
           SimTrackContainer::const_iterator it = std::lower_bound(g.simtks.begin(), g.simtks.end(), idx, LessById());
           if ((it != g.simtks.end()) && (it->trackId() == idx)) {
                return &*it;
           }
       }
   }
   return 0;
}

edm::Ref<reco::GenParticleCollection>
PATGenCandsFromSimTracksProducer::findRef(const SimTrack &tk, GlobalContext &g) const {
    if (tk.genpartIndex() != -1) return makeMotherLink_ ? generatorRef_(tk, g) : edm::Ref<reco::GenParticleCollection>();
    const SimTrack * simMother = findGeantMother(tk, g);

    edm::Ref<reco::GenParticleCollection> motherRef;
    if (simMother != 0) motherRef = findRef(*simMother,g);

    if (writeAncestors_) {
        // If writing ancestors, I need to serialize myself, and then to return a ref to me
        // But first check if I've already been serialized
        std::map<unsigned int,int>::const_iterator it = g.simTksProcessed.find(tk.trackId());
        if (it != g.simTksProcessed.end()) {
            // just return a ref to it
            assert(it->second > 0);
            return edm::Ref<reco::GenParticleCollection>(g.refprod, (it->second) - 1);
        } else {
            // make genParticle, save, update the map, and return ref to myself
            reco::GenParticle p = makeGenParticle_(tk, motherRef, g);
            g.output.push_back(p);
            g.simTksProcessed[tk.trackId()] = g.output.size();
            return edm::Ref<reco::GenParticleCollection>(g.refprod, g.output.size()-1 );
        }
    } else {
        // Otherwise, I just return a ref to my mum
        return motherRef;
    }
}

edm::Ref<reco::GenParticleCollection>
PATGenCandsFromSimTracksProducer::generatorRef_(const SimTrack &st, const GlobalContext &g) const {
    assert(st.genpartIndex() != -1);
    // Note that st.genpartIndex() is the barcode, not the index within GenParticleCollection, so I have to search the particle
    std::vector<int>::const_iterator it;
    if (g.barcodesAreSorted) {
        it = std::lower_bound(g.genBarcodes->begin(), g.genBarcodes->end(), st.genpartIndex());
    } else {
        it = std::find(       g.genBarcodes->begin(), g.genBarcodes->end(), st.genpartIndex());
    }

    // Check that I found something
    // I need to check '*it == st.genpartIndex()' because lower_bound just finds the right spot for an item in a sorted list, not the item
    if ((it != g.genBarcodes->end()) && (*it == st.genpartIndex())) {
        return reco::GenParticleRef(g.gens, it - g.genBarcodes->begin());
    } else {
        return reco::GenParticleRef();
    }
}

reco::GenParticle
PATGenCandsFromSimTracksProducer::makeGenParticle_(const SimTrack &tk, const edm::Ref<reco::GenParticleCollection> & mother, const GlobalContext &g) const {
    // Make up a GenParticleCandidate from the GEANT track info.
    int charge = static_cast<int>(tk.charge());
    Particle::LorentzVector p4 = tk.momentum();
    Particle::Point vtx; // = (0,0,0) by default
    if (!tk.noVertex()) vtx = g.simvtxs[tk.vertIndex()].position();
    GenParticle gp(charge, p4, vtx, tk.type(), setStatus_, true);
    if (mother.isNonnull()) gp.addMother(mother);
    return gp;
}


void PATGenCandsFromSimTracksProducer::produce(Event& event,
                                               const EventSetup& iSetup) {

  if (firstEvent_){
    if (!pdts_.empty()) {
      pdgIds_.clear();
      for (vector<PdtEntry>::iterator itp = pdts_.begin(), edp = pdts_.end(); itp != edp; ++itp) {
	itp->setup(iSetup); // decode string->pdgId and vice-versa
	pdgIds_.insert(std::abs(itp->pdgId()));
      }
      pdts_.clear();
    }
    if (!motherPdts_.empty()) {
      motherPdgIds_.clear();
      for (vector<PdtEntry>::iterator itp = motherPdts_.begin(), edp = motherPdts_.end(); itp != edp; ++itp) {
	itp->setup(iSetup); // decode string->pdgId and vice-versa
	motherPdgIds_.insert(std::abs(itp->pdgId()));
      }
      motherPdts_.clear();
    }
    firstEvent_ = false;
  }

  // Simulated tracks (i.e. GEANT particles).
  Handle<SimTrackContainer> simtracks;
  event.getByToken(simTracksToken_, simtracks);

  // Need to check that SimTrackContainer is sorted; otherwise, copy and sort :-(
  std::auto_ptr<SimTrackContainer> simtracksTmp;
  const SimTrackContainer * simtracksSorted = &* simtracks;
  if (makeMotherLink_ || writeAncestors_) {
      if (!__gnu_cxx::is_sorted(simtracks->begin(), simtracks->end(), LessById())) {
          simtracksTmp.reset(new SimTrackContainer(*simtracks));
          std::sort(simtracksTmp->begin(), simtracksTmp->end(), LessById());
          simtracksSorted = &* simtracksTmp;
      }
  }

  // Get the associated vertices
  Handle<SimVertexContainer> simvertices;
  event.getByToken(simVertexToken_, simvertices);

  // Get the GenParticles and barcodes, if needed to set mother links
  Handle<GenParticleCollection> gens;
  Handle<std::vector<int> > genBarcodes;
  bool barcodesAreSorted = true;
  if (makeMotherLink_) {
      event.getByToken(genParticlesToken_, gens);
      event.getByToken(genBarcodesToken_, genBarcodes);
      if (gens->size() != genBarcodes->size()) throw cms::Exception("Corrupt data") << "Barcodes not of the same size as GenParticles!\n";
      barcodesAreSorted = __gnu_cxx::is_sorted(genBarcodes->begin(), genBarcodes->end());
  }


  // make the output collection
  auto_ptr<GenParticleCollection> cands(new GenParticleCollection);
  edm::RefProd<GenParticleCollection> refprod = event.getRefBeforePut<GenParticleCollection>();

  GlobalContext globals(*simtracksSorted, *simvertices, gens, genBarcodes, barcodesAreSorted, *cands, refprod);

  for (SimTrackContainer::const_iterator isimtrk = simtracks->begin();
          isimtrk != simtracks->end(); ++isimtrk) {

      // Skip PYTHIA tracks.
      if (isimtrk->genpartIndex() != -1) continue;

      // Maybe apply the PdgId filter
      if (!pdgIds_.empty()) { // if we have a filter on pdg ids
           if (pdgIds_.find(std::abs(isimtrk->type())) == pdgIds_.end()) continue;
      }

      GenParticle genp = makeGenParticle_(*isimtrk, Ref<GenParticleCollection>(), globals);

      // Maybe apply filter on the particle
      if (!(filter_(genp))) continue;


      if (!motherPdgIds_.empty()) {
           const SimTrack *motherSimTk = findGeantMother(*isimtrk, globals);
           if (motherSimTk == 0) continue;
           if (motherPdgIds_.find(std::abs(motherSimTk->type())) == motherPdgIds_.end()) continue;
      }

      if (makeMotherLink_ || writeAncestors_) {
          Ref<GenParticleCollection> motherRef;
          const SimTrack * mother = findGeantMother(*isimtrk, globals);
          if (mother != 0) motherRef = findRef(*mother, globals);
          if (motherRef.isNonnull()) genp.addMother(motherRef);
      }

      cands->push_back(genp);
  }

  // Write to the Event, and get back a handle (which can be useful for debugging)
  edm::OrphanHandle<reco::GenParticleCollection> orphans = event.put(cands);

#ifdef DEBUG_PATGenCandsFromSimTracksProducer
  std::cout << "Produced a list of " << orphans->size() << " genParticles." << std::endl;
  for (GenParticleCollection::const_iterator it = orphans->begin(), ed = orphans->end(); it != ed; ++it) {
      std::cout << "    ";
      std::cout << "GenParticle #" << (it - orphans->begin()) << ": pdgId " << it->pdgId()
                << ", pt = " << it->pt() << ", eta = " << it->eta() << ", phi = " << it->phi()
                << ", rho = " << it->vertex().Rho() << ", z = " << it->vertex().Z() << std::endl;
      edm::Ref<GenParticleCollection> mom = it->motherRef();
      size_t depth = 2;
      while (mom.isNonnull()) {
          if (mom.id() == orphans.id()) {
              // I need to re-make the ref because they are not working until this module returns.
              mom = edm::Ref<GenParticleCollection>(orphans, mom.key());
          }
          for (size_t i = 0; i < depth; ++i) std::cout << "    ";
          std::cout << "GenParticleRef [" << mom.id() << "/" << mom.key() << "]: pdgId " << mom->pdgId() << ", status = " << mom->status()
                    << ", pt = " << mom->pt() << ", eta = " << mom->eta() << ", phi = " << mom->phi()
                    << ", rho = " << mom->vertex().Rho() << ", z = " << mom->vertex().Z() << std::endl;
          if (mom.id() != orphans.id()) break;
          if ((mom->motherRef().id() == mom.id()) && (mom->motherRef().key() == mom.key())) {
              throw cms::Exception("Corrupt Data") << "A particle is it's own mother.\n";
          }
          mom = mom->motherRef();
          depth++;
      }
  }
  std::cout << std::endl;
#endif

}

DEFINE_FWK_MODULE(PATGenCandsFromSimTracksProducer);
