#ifndef GEMValidation_SimHitMatcher_h
#define GEMValidation_SimHitMatcher_h

/**\class SimHitMatcher

 Description: Matching of SimHit for SimTrack in CSC & GEM

 Original Author:  "Vadim Khotilovich"
*/

#include "GEMCode/GEMValidation/src/BaseMatcher.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <vector>
#include <map>
#include <set>

class CSCGeometry;
class GEMGeometry;

class SimHitMatcher : public BaseMatcher
{
public:
  
  SimHitMatcher(const SimTrack& t, const SimVertex& v,
      const edm::ParameterSet& ps, const edm::Event& ev, const edm::EventSetup& es);
  
  ~SimHitMatcher();

  /// access to all the GEM SimHits
  const edm::PSimHitContainer& simHitsGEM() const {return gem_hits_;}
  /// access to all the CSC SimHits
  const edm::PSimHitContainer& simHitsCSC() const {return csc_hits_;}
  /// access to all the ME0 SimHits
  const edm::PSimHitContainer& simHitsME0() const {return me0_hits_;}
  /// access to all the RPC SimHits
  const edm::PSimHitContainer& simHitsRPC() const {return rpc_hits_;}

  /// GEM partitions' detIds with SimHits
  std::set<unsigned int> detIdsGEM() const;
  /// ME0 partitions' detIds with SimHits
  std::set<unsigned int> detIdsME0() const;
  /// RPC partitions' detIds with SimHits
  std::set<unsigned int> detIdsRPC() const;
  /// CSC layers' detIds with SimHits
  /// by default, only returns those from ME1b
  std::set<unsigned int> detIdsCSC(int csc_type = CSC_ME1b) const;

  /// GEM detid's with hits in 2 layers of coincidence pads
  /// those are layer==1 only detid's
  std::set<unsigned int> detIdsGEMCoincidences() const;
  /// RPC detid's with hits in 2 layers of coincidence pads
  /// those are layer==1 only detid's
  std::set<unsigned int> detIdsME0Coincidences(int min_n_layers = 2) const;

  /// GEM chamber detIds with SimHits
  std::set<unsigned int> chamberIdsGEM() const;
  /// ME0 chamber detIds with SimHits
  std::set<unsigned int> chamberIdsME0() const;
  /// RPC chamber detIds with SimHits
  std::set<unsigned int> chamberIdsRPC() const;
  /// CSC chamber detIds with SimHits
  std::set<unsigned int> chamberIdsCSC(int csc_type = CSC_ME1b) const;

  /// GEM superchamber detIds with SimHits
  std::set<unsigned int> superChamberIdsGEM() const;
  /// GEM superchamber detIds with SimHits 2 layers of coincidence pads
  std::set<unsigned int> superChamberIdsGEMCoincidences() const;

  /// ME0 superchamber detIds with SimHits
  std::set<unsigned int> superChamberIdsME0() const;
  /// ME0 superchamber detIds with SimHits >=2 layers of coincidence pads
  std::set<unsigned int> superChamberIdsME0Coincidences(int min_n_layers = 2) const;

  /// simhits from a particular partition (GEM)/layer (CSC), chamber or superchamber
  const edm::PSimHitContainer& hitsInDetId(unsigned int) const;
  const edm::PSimHitContainer& hitsInChamber(unsigned int) const;
  const edm::PSimHitContainer& hitsInSuperChamber(unsigned int) const;

  /// #layers with hits
  /// for CSC: "super-chamber" means chamber
  int nLayersWithHitsInSuperChamber(unsigned int) const;

  /// How many pads with simhits in GEM did this simtrack get?
  int nPadsWithHits() const;
  /// How many coincidence pads with simhits in GEM did this simtrack get?
  int nCoincidencePadsWithHits() const;

  /// How many ME0 chambers with minimum number of layer with simhits did this simtrack get?
  int nCoincidenceME0Chambers(int min_n_layers = 2) const;

  /// How many CSC chambers with minimum number of layer with simhits did this simtrack get?
  int nCoincidenceCSCChambers(int min_n_layers = 4) const;

  /// calculate Global average position for a provided collection of simhits
  GlobalPoint simHitsMeanPosition(const edm::PSimHitContainer& sim_hits) const;

  /// calculate average strip (strip for GEM/ME0, half-strip for CSC) number for a provided collection of simhits
  float simHitsMeanStrip(const edm::PSimHitContainer& sim_hits) const;

  std::set<int> hitStripsInDetId(unsigned int, int margin_n_strips = 0) const;  // GEM/ME0 or CSC
  std::set<int> hitWiregroupsInDetId(unsigned int, int margin_n_wg = 0) const; // CSC
  std::set<int> hitPadsInDetId(unsigned int) const; // GEM
  std::set<int> hitCoPadsInDetId(unsigned int) const; // GEM coincidence pads with hits

  // what unique partitions numbers were hit by this simtrack?
  std::set<int> hitPartitions() const; // GEM

private:

  void init();

  std::vector<unsigned int> getIdsOfSimTrackShower(unsigned  trk_id,
      const edm::SimTrackContainer& simTracks, const edm::SimVertexContainer& simVertices);

  void matchSimHitsToSimTrack(std::vector<unsigned int> track_ids,
      const edm::PSimHitContainer& csc_hits, const edm::PSimHitContainer& gem_hits);

  bool simMuOnlyCSC_;
  bool simMuOnlyGEM_;
  bool simMuOnlyRPC_;
  bool simMuOnlyME0_;

  bool discardEleHitsCSC_;
  bool discardEleHitsGEM_;
  bool discardEleHitsRPC_;
  bool discardEleHitsME0_;

  std::string simInputLabel_;

  std::map<unsigned int, unsigned int> trkid_to_index_;

  edm::PSimHitContainer no_hits_;

  edm::PSimHitContainer csc_hits_;
  std::map<unsigned int, edm::PSimHitContainer > csc_detid_to_hits_;
  std::map<unsigned int, edm::PSimHitContainer > csc_chamber_to_hits_;

  edm::PSimHitContainer gem_hits_;
  std::map<unsigned int, edm::PSimHitContainer > gem_detid_to_hits_;
  std::map<unsigned int, edm::PSimHitContainer > gem_chamber_to_hits_;
  std::map<unsigned int, edm::PSimHitContainer > gem_superchamber_to_hits_;

  edm::PSimHitContainer me0_hits_;
  std::map<unsigned int, edm::PSimHitContainer > me0_detid_to_hits_;
  std::map<unsigned int, edm::PSimHitContainer > me0_chamber_to_hits_;
  std::map<unsigned int, edm::PSimHitContainer > me0_superchamber_to_hits_;

  edm::PSimHitContainer rpc_hits_;
  std::map<unsigned int, edm::PSimHitContainer > rpc_detid_to_hits_;
  std::map<unsigned int, edm::PSimHitContainer > rpc_chamber_to_hits_;

  // detids with hits in pads
  std::map<unsigned int, std::set<int> > gem_detids_to_pads_;
  // detids with hits in 2-layer pad coincidences
  std::map<unsigned int, std::set<int> > gem_detids_to_copads_;

  bool verboseGEM_;
  bool verboseCSC_;
  bool verboseRPC_;
  bool verboseME0_;

  edm::InputTag gemSimHitInput_;
  edm::InputTag cscSimHitInput_;
  edm::InputTag rpcSimHitInput_;
  edm::InputTag me0SimHitInput_;
};

#endif
