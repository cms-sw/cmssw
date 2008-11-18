/**
  \class    pat::PATGenCandsFromSimTracksProducer PATGenCandsFromSimTracksProducer.h "PhysicsTools/PatAlgos/interface/PATGenCandsFromSimTracksProducer.h"
  \brief    Produces reco::GenParticle from SimTracks

   The PATGenCandsFromSimTracksProducer produces GenParticles from SimTracks, so they can be used for MC matching.
   

  \author   Jordan Tucker (original module), Giovanni Petrucciani (PAT integration)
  \version  $Id: PATGenCandsFromSimTracksProducer.cc,v 1.2.4.1 2008/11/18 11:10:19 gpetrucc Exp $
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

#include <ext/algorithm>

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

  /// If true, I'll try to make a link from the GEANT particle to a GenParticle  
  bool makeMotherLink_;
  /// Collection of GenParticles I need to make refs to. It must also have its associated vector<int> of barcodes, aligned with them.
  edm::InputTag genParticles_;

  /// Try to link the GEANT particle to the generator particle it came from
  /** Arguments:
   * -- Specific --
   *    gp: GenParticle made from the GEANT particle, to which I must add the mother link
   *    st: The GEANT simTrack of which I'm searching the mother
   *
   * -- GEANT related --
   *    simtks:  A list of GEANT tracks, sorted by track id
   *    simvtxs: The list of GEANT vertices, in their natural order (skimtks have pointers into this vector!)
   *
   * -- GenParticle related --
   *    gens             : Handle to the GenParticles, to make the ref to
   *    genBarcodes      : Barcodes for each GenParticle, in a vector aligned to the GenParticleCollection.
   *    barcodesAreSorted: true if the barcodes are sorted (which means I can use binary_search on them) */
  void trySetParent(reco::GenParticle &gp,
                    const SimTrack &st,
                    const edm::SimTrackContainer &simtks,
                    const edm::SimVertexContainer &simvtxs,
                    const edm::Handle<reco::GenParticleCollection> &gens,
                    const std::vector<int>                         &genBarcodes,
                    bool                                            barcodesAreSorted ) const ;
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
  src_(cfg.getParameter<InputTag>("src")),            // source sim tracks & vertices
  setStatus_(cfg.getParameter<int32_t>("setStatus")), // set status of GenParticle to this code
  makeMotherLink_(cfg.existsAs<bool>("makeMotherLink") ? cfg.getParameter<bool>("makeMotherLink") : false),
  genParticles_(makeMotherLink_ ? cfg.getParameter<InputTag>("genParticles") : edm::InputTag())
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

void PATGenCandsFromSimTracksProducer::trySetParent(GenParticle &gp, 
                                                    const SimTrack &st,
                                                    const SimTrackContainer &simtks, 
                                                    const SimVertexContainer &simvtxs, 
                                                    const edm::Handle<GenParticleCollection> &gens,
                                                    const std::vector<int>                   &genBarcodes,
                                                    bool                                      barcodesAreSorted) const 
{
    // First check if this SimTrack corresponds to a generator particle
    if (st.genpartIndex() != -1) {

        // Note that st.genpartIndex() is the barcode, not the index within GenParticleCollection, so I have to search the particle
        std::vector<int>::const_iterator it;
        if (barcodesAreSorted) {
            it = std::lower_bound(genBarcodes.begin(), genBarcodes.end(), st.genpartIndex());
        } else {
            it = std::find(       genBarcodes.begin(), genBarcodes.end(), st.genpartIndex());
        }

        // Check that I found something
        // I need to check '*it == st.genpartIndex()' because lower_bound just finds the right spot for an item in a sorted list, not the item
        if ((it != genBarcodes.end()) && (*it == st.genpartIndex())) {
            // Ok, I'll make the ref
            reco::GenParticleRef rf(gens, it - genBarcodes.begin());
            gp.addMother(rf);
        } 

    } else {
        // Particle was produced by GEANT, I need to track it back

        // Look at the production vertex. If there is no vertex, I can do nothing...
        if (!st.noVertex()) {

            // Pick the vertex (st.vertIndex() is really an index)
            const SimVertex &vtx = simvtxs[st.vertIndex()];

            // Check if the vertex has a parent track (otherwise, we're lost)
            if (!vtx.noParent()) {
    
                // Now note that vtx.parentIndex() is NOT an index, it's a track id, so I have to search for it 
                unsigned int idx = vtx.parentIndex();
                SimTrackContainer::const_iterator it = std::lower_bound(simtks.begin(), simtks.end(), idx, LessById());
                if ((it != simtks.end()) && (it->trackId() == idx)) {
                    // Found the track, start recurring...
                    trySetParent(gp, *it, simtks, simvtxs, gens, genBarcodes, barcodesAreSorted);
                }
            }
        }
    }
}


void PATGenCandsFromSimTracksProducer::produce(Event& event,
					    const EventSetup& eSetup) {
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

  // Get the GenParticles and barcodes, if needed to set mother links
  Handle<GenParticleCollection> gens;
  Handle<std::vector<int> > genBarcodes;
  bool barcodesAreSorted = true;
  if (makeMotherLink_) {
      event.getByLabel(genParticles_, gens);
      event.getByLabel(genParticles_, genBarcodes);
      if (gens->size() != genBarcodes->size()) throw cms::Exception("Corrupt data") << "Barcodes not of the same size as GenParticles!\n";
      barcodesAreSorted = __gnu_cxx::is_sorted(genBarcodes->begin(), genBarcodes->end());
  }    



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

      GenParticle genp(charge, p4, vtx, isimtrk->type(), setStatus_, true);

      // Maybe apply filter on the particle
      if (filter_.get() != 0) {
        if (!(*filter_)(genp)) continue;
      }

      if (makeMotherLink_) trySetParent(genp, *isimtrk, *simtracksSorted, *simvertices, gens, *genBarcodes, barcodesAreSorted);

      cands->push_back(genp);
  }

  event.put(cands);
}

DEFINE_FWK_MODULE(PATGenCandsFromSimTracksProducer);
