#include "L1Trigger/L1TMuonEndCap/interface/SectorProcessor.hh"


SectorProcessor::SectorProcessor() {

}

SectorProcessor::~SectorProcessor() {

}

void SectorProcessor::configure(
    const L1TMuon::GeometryTranslator* tp_geom,
    const SectorProcessorLUT* lut,
    const PtAssignmentEngine* pt_assign_engine,
    int verbose, int endcap, int sector,
    int minBX, int maxBX, int bxWindow, int bxShiftCSC, int bxShiftRPC,
    const std::vector<int>& zoneBoundaries, int zoneOverlap, int zoneOverlapRPC,
    bool includeNeighbor, bool duplicateTheta, bool fixZonePhi, bool useNewZones, bool fixME11Edges,
    const std::vector<std::string>& pattDefinitions, const std::vector<std::string>& symPattDefinitions, bool useSymPatterns,
    int thetaWindow, int thetaWindowRPC, bool bugME11Dupes,
    int maxRoadsPerZone, int maxTracks, bool useSecondEarliest, bool bugSameSectorPt0,
    bool readPtLUTFile, bool fixMode15HighPt, bool bug9BitDPhi, bool bugMode7CLCT, bool bugNegPt, bool bugGMTPhi
) {
  assert(MIN_ENDCAP <= endcap && endcap <= MAX_ENDCAP);
  assert(MIN_TRIGSECTOR <= sector && sector <= MAX_TRIGSECTOR);

  assert(tp_geom != nullptr);
  assert(lut != nullptr);
  assert(pt_assign_engine != nullptr);

  tp_geom_          = tp_geom;
  lut_              = lut;
  pt_assign_engine_ = pt_assign_engine;

  verbose_  = verbose;
  endcap_   = endcap;
  sector_   = sector;

  minBX_       = minBX;
  maxBX_       = maxBX;
  bxWindow_    = bxWindow;
  bxShiftCSC_  = bxShiftCSC;
  bxShiftRPC_  = bxShiftRPC;

  zoneBoundaries_     = zoneBoundaries;
  zoneOverlap_        = zoneOverlap;
  zoneOverlapRPC_     = zoneOverlapRPC;
  includeNeighbor_    = includeNeighbor;
  duplicateTheta_     = duplicateTheta;
  fixZonePhi_         = fixZonePhi;
  useNewZones_        = useNewZones;
  fixME11Edges_       = fixME11Edges;

  pattDefinitions_    = pattDefinitions;
  symPattDefinitions_ = symPattDefinitions;
  useSymPatterns_     = useSymPatterns;

  thetaWindow_        = thetaWindow;
  thetaWindowRPC_     = thetaWindowRPC;
  bugME11Dupes_       = bugME11Dupes;

  maxRoadsPerZone_    = maxRoadsPerZone;
  maxTracks_          = maxTracks;
  useSecondEarliest_  = useSecondEarliest;
  bugSameSectorPt0_   = bugSameSectorPt0;

  readPtLUTFile_      = readPtLUTFile;
  fixMode15HighPt_    = fixMode15HighPt;
  bug9BitDPhi_        = bug9BitDPhi;
  bugMode7CLCT_       = bugMode7CLCT;
  bugNegPt_           = bugNegPt;
  bugGMTPhi_          = bugGMTPhi;
}

void SectorProcessor::process(
    EventNumber_t ievent,
    const TriggerPrimitiveCollection& muon_primitives,
    EMTFHitCollection& out_hits,
    EMTFTrackCollection& out_tracks
) const {

  // List of converted hits, extended from previous BXs
  // deque (double-ended queue) is similar to a vector, but allows insertion or deletion of elements at both beginning and end
  std::deque<EMTFHitCollection> extended_conv_hits;

  // List of best track candidates, extended from previous BXs
  std::deque<EMTFTrackCollection> extended_best_track_cands;

  // Map of pattern detector --> lifetime, tracked across BXs
  std::map<pattern_ref_t, int> patt_lifetime_map;

  int delayBX = bxWindow_ - 1;

  for (int bx = minBX_; bx <= maxBX_ + delayBX; ++bx) {
    if (verbose_ > 0) {  // debug
      std::cout << "Endcap: " << endcap_ << " Sector: " << sector_ << " Event: " << ievent << " BX: " << bx << std::endl;
    }

    process_single_bx(
        bx,
        muon_primitives,
        out_hits,
        out_tracks,
        extended_conv_hits,
        extended_best_track_cands,
        patt_lifetime_map
    );

    // Drop earliest BX outside of BX window
    if (bx >= minBX_ + delayBX) {
      extended_conv_hits.pop_front();

      int n = zone_array<int>().size();
      extended_best_track_cands.erase(extended_best_track_cands.end()-n, extended_best_track_cands.end());  // pop_back
    }
  }  // end loop over bx

  return;
}

void SectorProcessor::process_single_bx(
    int bx,
    const TriggerPrimitiveCollection& muon_primitives,
    EMTFHitCollection& out_hits,
    EMTFTrackCollection& out_tracks,
    std::deque<EMTFHitCollection>& extended_conv_hits,
    std::deque<EMTFTrackCollection>& extended_best_track_cands,
    std::map<pattern_ref_t, int>& patt_lifetime_map
) const {

  // ___________________________________________________________________________
  // Configure

  PrimitiveSelection prim_sel;
  prim_sel.configure(
      verbose_, endcap_, sector_, bx,
      bxShiftCSC_, bxShiftRPC_,
      includeNeighbor_, duplicateTheta_,
      bugME11Dupes_
  );

  PrimitiveConversion prim_conv;
  prim_conv.configure(
      tp_geom_, lut_,
      verbose_, endcap_, sector_, bx,
      bxShiftCSC_, bxShiftRPC_,
      zoneBoundaries_, zoneOverlap_, zoneOverlapRPC_,
      duplicateTheta_, fixZonePhi_, useNewZones_, fixME11Edges_,
      bugME11Dupes_
  );

  PatternRecognition patt_recog;
  patt_recog.configure(
      verbose_, endcap_, sector_, bx,
      bxWindow_,
      pattDefinitions_, symPattDefinitions_, useSymPatterns_,
      maxRoadsPerZone_, useSecondEarliest_
  );

  PrimitiveMatching prim_match;
  prim_match.configure(
      verbose_, endcap_, sector_, bx,
      fixZonePhi_, useNewZones_,
      bugME11Dupes_
  );

  AngleCalculation angle_calc;
  angle_calc.configure(
      verbose_, endcap_, sector_, bx,
      bxWindow_,
      thetaWindow_, thetaWindowRPC_,
      bugME11Dupes_
  );

  BestTrackSelection btrack_sel;
  btrack_sel.configure(
      verbose_, endcap_, sector_, bx,
      bxWindow_,
      maxRoadsPerZone_, maxTracks_, useSecondEarliest_,
      bugSameSectorPt0_
  );

  SingleHitTrack single_hit;
  single_hit.configure(
      verbose_, endcap_, sector_, bx,
      maxTracks_,
      false 
  );

  PtAssignment pt_assign;
  pt_assign.configure(
      pt_assign_engine_,
      verbose_, endcap_, sector_, bx,
      readPtLUTFile_, fixMode15HighPt_,
      bug9BitDPhi_, bugMode7CLCT_, bugNegPt_,
      bugGMTPhi_
  );

  std::map<int, TriggerPrimitiveCollection> selected_csc_map;
  std::map<int, TriggerPrimitiveCollection> selected_rpc_map;

  EMTFHitCollection conv_hits;

  zone_array<EMTFRoadCollection> zone_roads;  // each zone has its road collection

  zone_array<EMTFTrackCollection> zone_tracks;  // each zone has its track collection

  EMTFTrackCollection best_tracks;

  // ___________________________________________________________________________
  // Process

  // Select muon primitives that belong to this sector and this BX.
  // Put them into maps with an index that roughly corresponds to
  // each input link. From src/PrimitiveSelection.cc.
  prim_sel.process(CSCTag(), muon_primitives, selected_csc_map);
  prim_sel.process(RPCTag(), muon_primitives, selected_rpc_map);

  // Convert trigger primitives into "converted hits"
  // A converted hit consists of integer representations of phi, theta, and zones
  // From src/PrimitiveConversion.cc
  prim_conv.process(CSCTag(), selected_csc_map, conv_hits);
  prim_conv.process(RPCTag(), selected_rpc_map, conv_hits);
  extended_conv_hits.push_back(conv_hits);

  // Detect patterns in all zones, find 3 best roads in each zone
  // From src/PatternRecognition.cc
  patt_recog.process(extended_conv_hits, patt_lifetime_map, zone_roads);

  // Match the trigger primitives to the roads, create tracks
  // From src/PrimitiveMatching.cc
  prim_match.process(extended_conv_hits, zone_roads, zone_tracks);

  // Calculate deflection angles for each track and fill track variables
  // From src/AngleCalculation.cc
  angle_calc.process(zone_tracks);
  extended_best_track_cands.insert(extended_best_track_cands.begin(), zone_tracks.begin(), zone_tracks.end());  // push_front

  // Identify 3 best tracks
  // From src/BestTrackSelection.cc
  btrack_sel.process(extended_best_track_cands, best_tracks);

  // Insert single LCTs from station 1 as tracks
  // From src/SingleHitTracks.cc
  single_hit.process(conv_hits, best_tracks);

  // Construct pT address, assign pT
  // From src/PtAssignment.cc
  pt_assign.process(best_tracks);

  // ___________________________________________________________________________
  // Output

  out_hits.insert(out_hits.end(), conv_hits.begin(), conv_hits.end());
  out_tracks.insert(out_tracks.end(), best_tracks.begin(), best_tracks.end());

  return;
}
