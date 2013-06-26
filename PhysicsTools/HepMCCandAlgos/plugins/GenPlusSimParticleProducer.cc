/**
  \class    pat::GenPlusSimParticleProducer GenPlusSimParticleProducer.h "PhysicsTools/PatAlgos/interface/GenPlusSimParticleProducer.h"
  \brief    Produces reco::GenParticle from SimTracks

The GenPlusSimParticleProducer produces GenParticles from SimTracks, 
so they can be used for MC matching. A copy of the original genParticle list
is made and the genParticles created from the GEANT/FAMOS particles are added
to the list including all ancestors and correct mother/daughter references

Sample useage in cfg.py file:
process.genParticlePlusGEANT = cms.EDProducer("GenPlusSimParticleProducer",
        src           = cms.InputTag("g4SimHits"), # use "famosSimHits" for FAMOS
        setStatus     = cms.int32(8),             # set status = 8 for GEANT GPs
        particleTypes = cms.vstring("pi+"),       # also picks pi- (optional)
        filter        = cms.vstring("pt > 0.0"),  # just for testing
        genParticles   = cms.InputTag("genParticles") # original genParticle list
)   

  \author   Jordan Tucker (original module), Keith Ulmer (generalization)
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
class GenPlusSimParticleProducer : public edm::EDProducer {
public:
  explicit GenPlusSimParticleProducer(const edm::ParameterSet&);
  ~GenPlusSimParticleProducer() {}

private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() {}

  bool firstEvent_;
  edm::InputTag src_;
  int setStatus_;
  std::set<int>         pdgIds_; // these are the ones we really use
  std::vector<PdtEntry> pdts_;   // these are needed before we get the EventSetup

  typedef StringCutObjectSelector<reco::GenParticle> StrFilter;
  std::auto_ptr<StrFilter> filter_;

  /// Collection of GenParticles I need to make refs to. It must also have its associated vector<int> of barcodes, aligned with them.
  edm::InputTag genParticles_;

  /// Try to link the GEANT particle to the generator particle it came from
  /** Arguments:
   * -- Specific --
   *    gp: GenParticle made from the GEANT particle
   *    st: The GEANT simTrack for which we create a genParticle
   *
   * -- GEANT related --
   *    simtks:  A list of GEANT tracks, sorted by track id
   *    simvtxs: The list of GEANT vertices, in their natural order (skimtks have pointers into this vector!)
   *
   * -- GenParticle related --
   *    gens             : Handle to the GenParticles, to make the ref to
   *    genBarcodes      : Barcodes for each GenParticle, in a vector aligned to the GenParticleCollection.
   *    barcodesAreSorted: true if the barcodes are sorted (which means I can use binary_search on them) */
  void addGenParticle(const SimTrack &stMom,
		      const SimTrack &stDau,
		      unsigned int momidx,
		      const edm::SimTrackContainer &simtks,
		      const edm::SimVertexContainer &simvtxs,
		      reco::GenParticleCollection &mergedGens,
		      const reco::GenParticleRefProd ref,
		      std::vector<int> &genBarcodes,
		      bool &barcodesAreSorted ) const ;
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
using pat::GenPlusSimParticleProducer;

GenPlusSimParticleProducer::GenPlusSimParticleProducer(const ParameterSet& cfg) :
  firstEvent_(true),
  src_(cfg.getParameter<InputTag>("src")),            // source sim tracks & vertices
  setStatus_(cfg.getParameter<int32_t>("setStatus")), // set status of GenParticle to this code
  genParticles_(cfg.getParameter<InputTag>("genParticles")) // get the genParticles to add GEANT particles to
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
  produces<vector<int> >();
}

void GenPlusSimParticleProducer::addGenParticle( const SimTrack &stMom,
						    const SimTrack &stDau,
						    unsigned int momidx,
						    const SimTrackContainer &simtracksSorted, 
						    const SimVertexContainer &simvertices, 
						    reco::GenParticleCollection &mergedGens,
						    const GenParticleRefProd ref,
						    std::vector<int> &genBarcodes,
						    bool &barcodesAreSorted) const 
{
  // Make the genParticle for stDau and add it to the new collection and update the parent-child relationship
  // Make up a GenParticleCandidate from the GEANT track info.
  int charge = static_cast<int>(stDau.charge());
  Particle::LorentzVector p4 = stDau.momentum();
  Particle::Point vtx; // = (0,0,0) by default
  if (!stDau.noVertex()) vtx = simvertices[stDau.vertIndex()].position();
  GenParticle genp(charge, p4, vtx, stDau.type(), setStatus_, true);
  
  // Maybe apply filter on the particle
  if (filter_.get() != 0) {
    if (!(*filter_)(genp)) return;
  }
  
  reco::GenParticleRef parentRef(ref, momidx);
  genp.addMother(parentRef);
  mergedGens.push_back(genp);
  // get the index for the daughter just added
  unsigned int dauidx = mergedGens.size()-1;

  // update add daughter relationship
  reco::GenParticle & cand = mergedGens[ momidx ];
  cand.addDaughter( GenParticleRef( ref, dauidx) );

  //look for simtrack daughters of stDau to see if we need to recur further down the chain
  
  for (SimTrackContainer::const_iterator isimtrk = simtracksSorted.begin();
       isimtrk != simtracksSorted.end(); ++isimtrk) {
    if (!isimtrk->noVertex()) {
      // Pick the vertex (isimtrk.vertIndex() is really an index)
      const SimVertex &vtx = simvertices[isimtrk->vertIndex()];
      
      // Check if the vertex has a parent track (otherwise, we're lost)
      if (!vtx.noParent()) {
	
	// Now note that vtx.parentIndex() is NOT an index, it's a track id, so I have to search for it 
	unsigned int idx = vtx.parentIndex();
	SimTrackContainer::const_iterator it = std::lower_bound(simtracksSorted.begin(), simtracksSorted.end(), idx, LessById());
	if ((it != simtracksSorted.end()) && (it->trackId() == idx)) {
	  if (it->trackId()==stDau.trackId()) {
	    //need the genparticle index of stDau which is dauidx
	    addGenParticle(stDau, *isimtrk, dauidx, simtracksSorted, simvertices, mergedGens, ref, genBarcodes, barcodesAreSorted);
	  }
	}
      }
    }
  }
}

void GenPlusSimParticleProducer::produce(Event& event,
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
    firstEvent_ = false;
  }

  // Simulated tracks (i.e. GEANT particles).
  Handle<SimTrackContainer> simtracks;
  event.getByLabel(src_, simtracks);
  
  // Need to check that SimTrackContainer is sorted; otherwise, copy and sort :-(
  std::auto_ptr<SimTrackContainer> simtracksTmp;
  const SimTrackContainer * simtracksSorted = &* simtracks;
  if (!__gnu_cxx::is_sorted(simtracks->begin(), simtracks->end(), LessById())) {
    simtracksTmp.reset(new SimTrackContainer(*simtracks));
    std::sort(simtracksTmp->begin(), simtracksTmp->end(), LessById());
    simtracksSorted = &* simtracksTmp;
  }
  
  // Get the associated vertices
  Handle<SimVertexContainer> simvertices;
  event.getByLabel(src_, simvertices);
  
  // Get the GenParticles and barcodes, if needed to set mother and daughter links
  Handle<GenParticleCollection> gens;
  Handle<std::vector<int> > genBarcodes;
  bool barcodesAreSorted = true;
  event.getByLabel(genParticles_, gens);
  event.getByLabel(genParticles_, genBarcodes);
  if (gens->size() != genBarcodes->size()) throw cms::Exception("Corrupt data") << "Barcodes not of the same size as GenParticles!\n";
  
  // make the output collection
  auto_ptr<GenParticleCollection> candsPtr(new GenParticleCollection);
  GenParticleCollection & cands = * candsPtr;
  
  const GenParticleRefProd ref = event.getRefBeforePut<GenParticleCollection>();
  
  // add the original genParticles to the merged output list
  for( size_t i = 0; i < gens->size(); ++ i ) {
    reco::GenParticle cand((*gens)[i]);
    cands.push_back(cand);
  }
  
  // make new barcodes vector and fill it with the original list
  auto_ptr<vector<int> > newGenBarcodes( new vector<int>() );
  for (unsigned int i = 0; i < genBarcodes->size(); ++i) {
    newGenBarcodes->push_back((*genBarcodes)[i]);
  }
  barcodesAreSorted = __gnu_cxx::is_sorted(newGenBarcodes->begin(), newGenBarcodes->end());
  
  for( size_t i = 0; i < cands.size(); ++ i ) {
    reco::GenParticle & cand = cands[ i ];
    size_t nDaus = cand.numberOfDaughters();
    GenParticleRefVector daus = cand.daughterRefVector();
    cand.resetDaughters( ref.id() );
    for ( size_t d = 0; d < nDaus; ++d) {
      cand.addDaughter( GenParticleRef( ref, daus[d].key() ) );
    }
    
    size_t nMoms = cand.numberOfMothers();
    GenParticleRefVector moms = cand.motherRefVector();
    cand.resetMothers( ref.id() );
    for ( size_t m = 0; m < nMoms; ++m) {
      cand.addMother( GenParticleRef( ref, moms[m].key() ) );
    }
  }
  
  for (SimTrackContainer::const_iterator isimtrk = simtracks->begin();
       isimtrk != simtracks->end(); ++isimtrk) {
    
    // Skip PYTHIA tracks.
    if (isimtrk->genpartIndex() != -1) continue; 
    
    // Maybe apply the PdgId filter
    if (!pdgIds_.empty()) { // if we have a filter on pdg ids
      if (pdgIds_.find(std::abs(isimtrk->type())) == pdgIds_.end()) continue;
    }
    
    // find simtrack that has a genParticle match to its parent
    // Look at the production vertex. If there is no vertex, I can do nothing...
    if (!isimtrk->noVertex()) {
      
      // Pick the vertex (isimtrk.vertIndex() is really an index)
      const SimVertex &vtx = (*simvertices)[isimtrk->vertIndex()];
      
      // Check if the vertex has a parent track (otherwise, we're lost)
      if (!vtx.noParent()) {
	
	// Now note that vtx.parentIndex() is NOT an index, it's a track id, so I have to search for it 
	unsigned int idx = vtx.parentIndex();
	SimTrackContainer::const_iterator it = std::lower_bound(simtracksSorted->begin(), simtracksSorted->end(), idx, LessById());
	if ((it != simtracksSorted->end()) && (it->trackId() == idx)) { //it is the parent sim track
	  if (it->genpartIndex() != -1) {
	    std::vector<int>::const_iterator itIndex;
	    if (barcodesAreSorted) {
	      itIndex = std::lower_bound(genBarcodes->begin(), genBarcodes->end(), it->genpartIndex());
	    } else {
	      itIndex = std::find(       genBarcodes->begin(), genBarcodes->end(), it->genpartIndex());
	    }
	    
	    // Check that I found something
	    // I need to check '*itIndex == it->genpartIndex()' because lower_bound just finds the right spot for an item in a sorted list, not the item
	    if ((itIndex != genBarcodes->end()) && (*itIndex == it->genpartIndex())) {
	      // Ok, I'll make the genParticle for st and add it to the new collection updating the map and parent-child relationship
	      // pass the mother and daughter sim tracks and the mother genParticle to method to create the daughter genParticle and recur
	      unsigned int momidx = itIndex - genBarcodes->begin();
	      addGenParticle(*it, *isimtrk, momidx, *simtracksSorted, *simvertices, *candsPtr, ref, *newGenBarcodes, barcodesAreSorted);
	    }
	  }
	}
      }
    }   
  }
  
  event.put(candsPtr);
  event.put(newGenBarcodes);
}

DEFINE_FWK_MODULE(GenPlusSimParticleProducer);
