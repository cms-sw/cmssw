#ifndef _SimHitMatcher_h_
#define _SimHitMatcher_h_

/**\class SimHitMatcher

 Description: Matching of SimHit for SimTrack in CSC & GEM

 Original Author:  "Vadim Khotilovich"
 $Id$
*/

#include "BaseMatcher.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
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
  
  SimHitMatcher(const SimTrack* t, const SimVertex* v,
      const edm::ParameterSet* ps, const edm::Event* ev, const edm::EventSetup* es);
  
  ~SimHitMatcher();

  edm::PSimHitContainer& simHitsGEM() {return gem_hits_;}
  edm::PSimHitContainer& simHitsCSC() {return csc_hits_;}

  // partition (GEM)/layer (CSC) detIds with SimHits
  std::set<unsigned int> detIdsGEM();
  std::set<unsigned int> detIdsCSC();

  // detid's with hits in 2 layers of coincidence pads
  // those are layer==1 only detid's
  std::set<unsigned int> detIdsGEMCoincidences();

  // chamber detIds with SimHits
  std::set<unsigned int> chamberIdsGEM();
  std::set<unsigned int> chamberIdsCSC();

  // superchamber detIds with SimHits
  std::set<unsigned int> superChamberIdsGEM();
  std::set<unsigned int> superChamberIdsGEMCoincidences();

  // simhits from a particular partition (GEM)/layer (CSC), chamber or superchamber
  edm::PSimHitContainer& hitsInDetId(unsigned int);
  edm::PSimHitContainer& hitsInChamber(unsigned int);
  edm::PSimHitContainer& hitsInSuperChamber(unsigned int);

  /// #layers with hits
  /// for CSC: "super-chamber" means chamber
  int nLayersWithHitsInSuperChamber(unsigned int);

  /// How many pads with simhits in GEM did this simtrack get?
  int nPadsWithHits();
  /// How many coincidence pads with simhits in GEM did this simtrack get?
  int nCoincidencePadsWithHits();

  /// How many CSC chambers with minimum number of layer with simhits did this simtrack get?
  int nCoincidenceCSCChambers(int min_n_layers = 4);

  /// calculate Global average position for a provided collection of simhits
  GlobalPoint simHitsMeanPosition(const edm::PSimHitContainer& sim_hits);

  std::set<int> hitStripsInDetId(unsigned int);  // GEM or CSC
  std::set<int> hitWiregroupsInDetId(unsigned int); // CSC
  std::set<int> hitPadsInDetId(unsigned int); // GEM
  std::set<int> hitCoPadsInDetId(unsigned int); // GEM coincidence pads with hits

  // what unique partitions numbers were hit by this simtrack?
  std::set<int> hitPartitions(); // GEM

private:

  void init();

  std::vector<unsigned int> getIdsOfSimTrackShower(unsigned  trk_id,
      const edm::SimTrackContainer& simTracks, const edm::SimVertexContainer& simVertices);

  void matchSimHitsToSimTrack(std::vector<unsigned int> track_ids,
      const edm::PSimHitContainer& me11_hits, const edm::PSimHitContainer& gem_hits);

  bool simMuOnlyCSC_;
  bool simMuOnlyGEM_;
  bool discardEleHitsCSC_;
  bool discardEleHitsGEM_;
  std::string simInputLabel_;

  const CSCGeometry* csc_geo_;
  const GEMGeometry* gem_geo_;

  std::map<unsigned int, unsigned int> trkid_to_index_;

  edm::PSimHitContainer no_hits_;

  edm::PSimHitContainer csc_hits_;
  std::map<unsigned int, edm::PSimHitContainer > csc_detid_to_hits_;
  std::map<unsigned int, edm::PSimHitContainer > csc_chamber_to_hits_;

  edm::PSimHitContainer gem_hits_;
  std::map<unsigned int, edm::PSimHitContainer > gem_detid_to_hits_;
  std::map<unsigned int, edm::PSimHitContainer > gem_chamber_to_hits_;
  std::map<unsigned int, edm::PSimHitContainer > gem_superchamber_to_hits_;

  // detids with hits in pads
  std::map<unsigned int, std::set<int> > gem_detids_to_pads_;
  // detids with hits in 2-layer pad coincidences
  std::map<unsigned int, std::set<int> > gem_detids_to_copads_;
};

#endif
