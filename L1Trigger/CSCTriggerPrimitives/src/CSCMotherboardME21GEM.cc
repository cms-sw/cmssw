#include <L1Trigger/CSCTriggerPrimitives/src/CSCMotherboardME21GEM.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h>
#include <Geometry/GEMGeometry/interface/GEMGeometry.h>
#include <Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h>
#include <DataFormats/Math/interface/deltaPhi.h>
#include <iomanip> 
#include "boost/container/flat_set.hpp"

const double CSCMotherboardME21GEM::lut_wg_eta_odd[112][2] = {
{ 0,2.441},{ 1,2.435},{ 2,2.425},{ 3,2.414},{ 4,2.404},{ 5,2.394},{ 6,2.384},{ 7,2.374},
{ 8,2.365},{ 9,2.355},{10,2.346},{11,2.336},{12,2.327},{13,2.317},{14,2.308},{15,2.299},
{16,2.290},{17,2.281},{18,2.273},{19,2.264},{20,2.255},{21,2.247},{22,2.238},{23,2.230},
{24,2.221},{25,2.213},{26,2.205},{27,2.197},{28,2.189},{29,2.181},{30,2.173},{31,2.165},
{32,2.157},{33,2.149},{34,2.142},{35,2.134},{36,2.127},{37,2.119},{38,2.112},{39,2.104},
{40,2.097},{41,2.090},{42,2.083},{43,2.075},{44,2.070},{45,2.059},{46,2.054},{47,2.047},
{48,2.041},{49,2.034},{50,2.027},{51,2.020},{52,2.014},{53,2.007},{54,2.000},{55,1.994},
{56,1.988},{57,1.981},{58,1.975},{59,1.968},{60,1.962},{61,1.956},{62,1.950},{63,1.944},
{64,1.937},{65,1.931},{66,1.924},{67,1.916},{68,1.909},{69,1.902},{70,1.895},{71,1.888},
{72,1.881},{73,1.875},{74,1.868},{75,1.861},{76,1.854},{77,1.848},{78,1.841},{79,1.835},
{80,1.830},{81,1.820},{82,1.815},{83,1.809},{84,1.803},{85,1.796},{86,1.790},{87,1.784},
{88,1.778},{89,1.772},{90,1.766},{91,1.760},{92,1.754},{93,1.748},{94,1.742},{95,1.736},
{96,1.731},{97,1.725},{98,1.719},{99,1.714},{100,1.708},{101,1.702},{102,1.697},{103,1.691},
{104,1.686},{105,1.680},{106,1.675},{107,1.670},{108,1.664},{109,1.659},{110,1.654},{111,1.648},
};

const double CSCMotherboardME21GEM::lut_wg_eta_even[112][2] = {
{ 0,2.412},{ 1,2.405},{ 2,2.395},{ 3,2.385},{ 4,2.375},{ 5,2.365},{ 6,2.355},{ 7,2.345},
{ 8,2.335},{ 9,2.325},{10,2.316},{11,2.306},{12,2.297},{13,2.288},{14,2.279},{15,2.270},
{16,2.261},{17,2.252},{18,2.243},{19,2.234},{20,2.226},{21,2.217},{22,2.209},{23,2.200},
{24,2.192},{25,2.184},{26,2.175},{27,2.167},{28,2.159},{29,2.151},{30,2.143},{31,2.135},
{32,2.128},{33,2.120},{34,2.112},{35,2.105},{36,2.097},{37,2.090},{38,2.082},{39,2.075},
{40,2.068},{41,2.060},{42,2.053},{43,2.046},{44,2.041},{45,2.030},{46,2.025},{47,2.018},
{48,2.011},{49,2.005},{50,1.998},{51,1.991},{52,1.985},{53,1.978},{54,1.971},{55,1.965},
{56,1.958},{57,1.952},{58,1.946},{59,1.939},{60,1.933},{61,1.927},{62,1.921},{63,1.915},
{64,1.909},{65,1.902},{66,1.895},{67,1.887},{68,1.880},{69,1.873},{70,1.866},{71,1.859},
{72,1.853},{73,1.846},{74,1.839},{75,1.832},{76,1.826},{77,1.819},{78,1.812},{79,1.806},
{80,1.801},{81,1.792},{82,1.787},{83,1.780},{84,1.774},{85,1.768},{86,1.762},{87,1.756},
{88,1.750},{89,1.744},{90,1.738},{91,1.732},{92,1.726},{93,1.720},{94,1.714},{95,1.708},
{96,1.702},{97,1.697},{98,1.691},{99,1.685},{100,1.680},{101,1.674},{102,1.669},{103,1.663},
{104,1.658},{105,1.652},{106,1.647},{107,1.642},{108,1.636},{109,1.631},{110,1.626},{111,1.621},
};

// LUT with bending angles of the GEM-CSC high efficiency patterns (98%)
// 1st index: pt value = {5,10,15,20,30,40}
// 2nd index: bending angle for odd numbered chambers
// 3rd index: bending angle for even numbered chambers
const double CSCMotherboardME21GEM::lut_pt_vs_dphi_gemcsc[8][3] = {
  {3, 0.01832829, 0.01003643 },
  {5, 0.01095490, 0.00631625 },
  {7, 0.00786026, 0.00501017 },
  {10, 0.00596349, 0.00414560 },
  {15, 0.00462411, 0.00365550 },
  {20, 0.00435298, 0.00361550 },
  {30, 0.00465160, 0.00335700 },
  {40, 0.00372145, 0.00366262 }
};

CSCMotherboardME21GEM::CSCMotherboardME21GEM(unsigned endcap, unsigned station,
                               unsigned sector, unsigned subsector,
                               unsigned chamber,
                               const edm::ParameterSet& conf) :
  CSCMotherboard(endcap, station, sector, subsector, chamber, conf)
{
  const edm::ParameterSet commonParams(conf.getParameter<edm::ParameterSet>("commonParam"));
  runME21ILT_ = commonParams.getParameter<bool>("runME21ILT");
  
  if (!isSLHC) edm::LogError("L1CSCTPEmulatorConfigError")
    << "+++ Upgrade CSCMotherboardME21GEM constructed while isSLHC is not set! +++\n";
  
  const edm::ParameterSet me21tmbParams(conf.getParameter<edm::ParameterSet>("me21tmbSLHCGEM"));

  // whether to not reuse CLCTs that were used by previous matching ALCTs
  // in ALCT-to-CLCT algorithm
  drop_used_clcts = me21tmbParams.getParameter<bool>("tmbDropUsedClcts");

  match_earliest_clct_me21_only = me21tmbParams.getParameter<bool>("matchEarliestClctME21Only");

  tmb_cross_bx_algo = me21tmbParams.getParameter<unsigned int>("tmbCrossBxAlgorithm");

  // maximum lcts per BX in ME2
  max_me21_lcts = me21tmbParams.getParameter<unsigned int>("maxME21LCTs");

  pref[0] = match_trig_window_size/2;
  for (unsigned int m=2; m<match_trig_window_size; m+=2)
  {
    pref[m-1] = pref[0] - m/2;
    pref[m]   = pref[0] + m/2;
  }
  
  //----------------------------------------------------------------------------------------//

  //       G E M  -  C S C   I N T E G R A T E D   L O C A L   A L G O R I T H M

  //----------------------------------------------------------------------------------------//

  /// Do GEM matching?
  do_gem_matching = me21tmbParams.getParameter<bool>("doGemMatching");

  /// GEM matching dphi and deta
  gem_match_delta_phi_odd = me21tmbParams.getParameter<double>("gemMatchDeltaPhiOdd");
  gem_match_delta_phi_even = me21tmbParams.getParameter<double>("gemMatchDeltaPhiEven");
  gem_match_delta_eta = me21tmbParams.getParameter<double>("gemMatchDeltaEta");

  /// delta BX for GEM pads matching
  gem_match_delta_bx = me21tmbParams.getParameter<int>("gemMatchDeltaBX");

  /// min eta of LCT for which we require GEM match (we don't throw out LCTs below this min eta)
  gem_match_min_eta = me21tmbParams.getParameter<double>("gemMatchMinEta");
  gem_match_max_eta = me21tmbParams.getParameter<double>("gemMatchMaxEta");

  /// whether to throw out GEM-fiducial LCTs that have no gem match
  gem_clear_nomatch_lcts = me21tmbParams.getParameter<bool>("gemClearNomatchLCTs");

  // debug gem matching
  debug_gem_matching = me21tmbParams.getParameter<bool>("debugMatching");
  debug_luts = me21tmbParams.getParameter<bool>("debugLUTs");
  debug_gem_dphi = me21tmbParams.getParameter<bool>("debugGEMDphi");

  //  deltas used to construct GEM coincidence pads
  maxDeltaBXInCoPad_ = me21tmbParams.getParameter<int>("maxDeltaBXInCoPad");
  maxDeltaPadInCoPad_ = me21tmbParams.getParameter<int>("maxDeltaPadInCoPad");

  //  deltas used to match to GEM pads
  maxDeltaBXPad_ = me21tmbParams.getParameter<int>("maxDeltaBXPad");
  maxDeltaPadPadOdd_ = me21tmbParams.getParameter<int>("maxDeltaPadPadOdd");
  maxDeltaPadPadEven_ = me21tmbParams.getParameter<int>("maxDeltaPadPadEven");
  maxDeltaWg_ = me21tmbParams.getParameter<int>("maxDeltaWg");

  //  deltas used to match to GEM coincidence pads
  maxDeltaBXCoPad_ = me21tmbParams.getParameter<int>("maxDeltaBXCoPad");
  maxDeltaPadCoPad_ = me21tmbParams.getParameter<int>("maxDeltaPadCoPad");

  // drop low quality stubs if they don't have GEMs
  dropLowQualityCLCTsNoGEMs_ = me21tmbParams.getParameter<bool>("dropLowQualityCLCTsNoGEMs");
  dropLowQualityALCTsNoGEMs_ = me21tmbParams.getParameter<bool>("dropLowQualityALCTsNoGEMs");

  // correct LCT timing with GEMs
  correctLCTtimingWithGEM_ = me21tmbParams.getParameter<bool>("correctLCTtimingWithGEM");

  // build LCT from ALCT and GEM
  buildLCTfromALCTandGEM_ = me21tmbParams.getParameter<bool>("buildLCTfromALCTandGEM");
  buildLCTfromCLCTandGEM_ = me21tmbParams.getParameter<bool>("buildLCTfromCLCTandGEM");

  // LCT ghostbusting
  doLCTGhostBustingWithGEMs_ = me21tmbParams.getParameter<bool>("doLCTGhostBustingWithGEMs");

  // use "old" or "new" dataformat for integrated LCTs?
  useOldLCTDataFormat_ = me21tmbParams.getParameter<bool>("useOldLCTDataFormat");

  // promote ALCT-GEM pattern
  promoteALCTGEMpattern_ = me21tmbParams.getParameter<bool>("promoteALCTGEMpattern");

  // promote ALCT-GEM quality
  promoteALCTGEMquality_ = me21tmbParams.getParameter<bool>("promoteALCTGEMquality");
  promoteCLCTGEMquality_ = me21tmbParams.getParameter<bool>("promoteCLCTGEMquality");
}

CSCMotherboardME21GEM::~CSCMotherboardME21GEM() 
{
}

void CSCMotherboardME21GEM::clear()
{
  CSCMotherboard::clear();
  
  for (int bx = 0; bx < MAX_LCT_BINS; bx++)
    for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
      for (int i=0;i<2;i++)
        allLCTs[bx][mbx][i].clear();

  gemRollToEtaLimits_.clear();
  cscWgToGemRoll_.clear();
  gemPadToCscHs_.clear();
  cscHsToGemPad_.clear();
  pads_.clear();
  coPads_.clear();
}

void
CSCMotherboardME21GEM::run(const CSCWireDigiCollection* wiredc,
                    const CSCComparatorDigiCollection* compdc,
                    const GEMPadDigiCollection* gemPads) 
{
  clear();

  if (!( alct and clct and runME21ILT_))
  {
    if (infoV >= 0) edm::LogError("L1CSCTPEmulatorSetupError")
      << "+++ run() called for non-existing ALCT/CLCT processor! +++ \n";
    return;
  }

  alct->run(wiredc); // run anodeLCT
  clct->run(compdc); // run cathodeLCT

  bool gemGeometryAvailable(false);
  if (gem_g != nullptr) {
    if (infoV >= 0) edm::LogInfo("L1CSCTPEmulatorSetupInfo")
      << "+++ run() called for GEM-CSC integrated trigger! +++ \n";
    gemGeometryAvailable = true;
  }

  // retrieve CSCChamber geometry                                                                                                                                       
  CSCTriggerGeomManager* geo_manager(CSCTriggerGeometry::get());
  const CSCChamber* cscChamber(geo_manager->chamber(theEndcap, theStation, theSector, theSubsector, theTrigChamber));
  const CSCDetId csc_id(cscChamber->id());

  if (runME21ILT_){
    
    // check for GE2/1 geometry
    if ((not gemGeometryAvailable) or (gemGeometryAvailable and (gem_g->stations()).size()==2)) {
      if (infoV >= 0) edm::LogError("L1CSCTPEmulatorSetupError")
        << "+++ run() called for GEM-CSC integrated trigger without valid GE21 geometry! +++ \n";
      return;
    }

    // trigger geometry
    const CSCLayer* keyLayer(cscChamber->layer(3));
    const CSCLayerGeometry* keyLayerGeometry(keyLayer->geometry());

    //    const bool isEven(csc_id%2==0);
    const int region((theEndcap == 1) ? 1: -1);
    const bool isEven(csc_id.chamber()%2==0);
    const GEMDetId gem_id_long(region, 1, 3, 1, csc_id.chamber(), 0);
    const GEMChamber* gemChamberLong(gem_g->chamber(gem_id_long));
    
    // LUT<roll,<etaMin,etaMax> >    
    gemRollToEtaLimits_ = createGEMRollEtaLUT();

    if (debug_luts){
      std::cout<<"csc id "<< csc_id <<" "<< csc_id.rawId() << (isEven ? " even" : " odd") << " chamber" << csc_id.chamber()<<std::endl;      
      if (gemRollToEtaLimits_.size())
        for(auto p : gemRollToEtaLimits_) {
          std::cout << "pad "<< p.first << " min eta " << (p.second).first << " max eta " << (p.second).second << std::endl;
        }
    }

    // loop on all wiregroups to create a LUT <WG,rollMin,rollMax>
    const int numberOfWG(keyLayerGeometry->numberOfWireGroups());
    for (int i = 0; i< numberOfWG; ++i){
      auto eta(isEven ? lut_wg_eta_even[i][1] : lut_wg_eta_odd[i][1]); 
      cscWgToGemRoll_[i] = assignGEMRoll(eta);
    }
    if (debug_luts){
      for(auto p : cscWgToGemRoll_) {
        std::cout << "WG "<< p.first << " GEM roll " << p.second << std::endl;
      }
    }

    auto randRoll(gemChamberLong->etaPartition(2));
    auto nStrips(keyLayerGeometry->numberOfStrips());
    for (float i = 0; i< nStrips; i = i+0.5){
      const LocalPoint lpCSC(keyLayerGeometry->topology()->localPosition(i));
      const GlobalPoint gp(keyLayer->toGlobal(lpCSC));
      const LocalPoint lpGEM(randRoll->toLocal(gp));
      const int HS(i/0.5);
      const bool edge(HS < 5 or HS > 155);
      const float pad(edge ? -99 : randRoll->pad(lpGEM));
      // HS are wrapped-around
      cscHsToGemPad_[HS] = std::make_pair(std::floor(pad),std::ceil(pad));
    }
    if (debug_luts){
      std::cout << "detId " << csc_id << std::endl;
      for(auto p : cscHsToGemPad_) {
        std::cout << "CSC HS "<< p.first << " GEM Pad low " << (p.second).first << " GEM Pad high " << (p.second).second << std::endl;
      }
    }

    // pick any roll 
    const int nGEMPads(randRoll->npads());
    for (int i = 0; i< nGEMPads; ++i){
      const LocalPoint lpGEM(randRoll->centreOfPad(i));
      const GlobalPoint gp(randRoll->toGlobal(lpGEM));
      const LocalPoint lpCSC(keyLayer->toLocal(gp));
      const float strip(keyLayerGeometry->strip(lpCSC));
      // HS are wrapped-around
      gemPadToCscHs_[i] = (int) (strip - 0.25)/0.5;
    }
    if (debug_luts){
      std::cout << "detId " << csc_id << std::endl;
      for(auto p : gemPadToCscHs_) {
        std::cout << "GEM Pad "<< p.first << " CSC HS : " << p.second << std::endl;
      }
    }
    
    //select correct scenario, even or odd
    maxDeltaPadPad_ = (isEven ? maxDeltaPadPadEven_ : maxDeltaPadPadOdd_);
    // build coincidence pads
    std::auto_ptr<GEMCoPadDigiCollection> pCoPads(new GEMCoPadDigiCollection());
    buildCoincidencePads(gemPads, *pCoPads, csc_id);
    
    // retrieve pads and copads in a certain BX window for this CSC 
    pads_.clear();
    coPads_.clear();
    retrieveGEMPads(gemPads, gem_id_long);
    retrieveGEMCoPads(pCoPads.get(), gem_id_long); 
  }

  int used_clct_mask[20];
  for (int c=0;c<20;++c) used_clct_mask[c]=0;

  const bool hasPads(pads_.size()!=0);
  const bool hasCoPads(hasPads and coPads_.size()!=0);

  // ALCT centric matching
  for (int bx_alct = 0; bx_alct < CSCAnodeLCTProcessor::MAX_ALCT_BINS; bx_alct++)
  {
    if (alct->bestALCT[bx_alct].isValid())
    {
      const int bx_clct_start(bx_alct - match_trig_window_size/2);
      const int bx_clct_stop(bx_alct + match_trig_window_size/2);
      const int bx_copad_start(bx_alct - maxDeltaBXCoPad_);
      const int bx_copad_stop(bx_alct + maxDeltaBXCoPad_);

      if (debug_gem_matching){ 
        std::cout << "========================================================================" << std::endl;
        std::cout << "ALCT-CLCT matching in ME2/1 chamber: " << cscChamber->id() << std::endl;
        std::cout << "------------------------------------------------------------------------" << std::endl;
        std::cout << "+++ Best ALCT Details: ";
        alct->bestALCT[bx_alct].print();
        std::cout << "+++ Second ALCT Details: ";
        alct->secondALCT[bx_alct].print();
        
        printGEMTriggerPads(bx_clct_start, bx_clct_stop);      
        
        std::cout << "------------------------------------------------------------------------" << std::endl;
        std::cout << "Attempt ALCT-CLCT matching in ME2/1 in bx range: [" << bx_clct_start << "," << bx_clct_stop << "]" << std::endl;
      }

      // ALCT-to-CLCT
      int nSuccesFulMatches = 0;
      for (int bx_clct = bx_clct_start; bx_clct <= bx_clct_stop; bx_clct++)
      {
        if (bx_clct < 0 or bx_clct >= CSCCathodeLCTProcessor::MAX_CLCT_BINS) continue;
        if (drop_used_clcts and used_clct_mask[bx_clct]) continue;
        if (clct->bestCLCT[bx_clct].isValid())
        {
          // clct quality
          const int quality(clct->bestCLCT[bx_clct].getQuality());
          // low quality ALCT
          const bool lowQualityALCT(alct->bestALCT[bx_alct].getQuality() == 0);
          // low quality ALCT or CLCT
          const bool lowQuality(quality<4 or lowQualityALCT);
          if (debug_gem_matching) std::cout << "++Valid ME21 CLCT: " << clct->bestCLCT[bx_clct] << std::endl;

            // pick the pad that corresponds 
	  auto matchingPads(matchingGEMPads(clct->bestCLCT[bx_clct], alct->bestALCT[bx_alct], pads_[bx_clct], false));
	  auto matchingCoPads(matchingGEMPads(clct->bestCLCT[bx_clct], alct->bestALCT[bx_alct], coPads_[bx_clct], true));
          if (runME21ILT_ and dropLowQualityCLCTsNoGEMs_ and lowQuality and hasPads){
            int nFound(matchingPads.size());
            const bool clctInEdge(clct->bestCLCT[bx_clct].getKeyStrip() < 5 or clct->bestCLCT[bx_clct].getKeyStrip() > 155);
            if (clctInEdge){
              if (debug_gem_matching) std::cout << "\tInfo: low quality CLCT in CSC chamber edge, don't care about GEM pads" << std::endl;
            }
            else {
              if (nFound != 0){
                if (debug_gem_matching) std::cout << "\tInfo: low quality CLCT with " << nFound << " matching GEM trigger pads" << std::endl;
              }
              else {
                if (debug_gem_matching) std::cout << "\tWarning: low quality CLCT without matching GEM trigger pad" << std::endl;
                continue;
              }
            }
          }

          // check timing
          if (runME21ILT_ and correctLCTtimingWithGEM_){
            int nFound(matchingCoPads.size());
            if (nFound != 0 and bx_alct == 6 and bx_clct != 6){
              if (debug_gem_matching) std::cout << "\tInfo: CLCT with incorrect timing" << std::endl;
              continue;
            }
          }
          
          ++nSuccesFulMatches;
      
          int mbx = bx_clct-bx_clct_start;

          correlateLCTsGEM(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
			   clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct],
			   allLCTs[bx_alct][mbx][0], allLCTs[bx_alct][mbx][1], matchingPads, matchingCoPads);
          if (debug_gem_matching) {
          //	    if (infoV > 1) LogTrace("CSCMotherboard")
            std::cout << "Successful ALCT-CLCT match in ME21: bx_alct = " << bx_alct
                      << "; match window: [" << bx_clct_start << "; " << bx_clct_stop
                      << "]; bx_clct = " << bx_clct << std::endl;
            std::cout << "+++ Best CLCT Details: ";
            clct->bestCLCT[bx_clct].print();
            std::cout << "+++ Second CLCT Details: ";
            clct->secondCLCT[bx_clct].print();
          }
          if (allLCTs[bx_alct][mbx][0].isValid()) {
            used_clct_mask[bx_clct] += 1;
            if (match_earliest_clct_me21_only) break;
          }
        }
      }

      // ALCT-to-GEM matching
      int nSuccesFulGEMMatches = 0;
      if (runME21ILT_ and nSuccesFulMatches==0 and buildLCTfromALCTandGEM_){
        if (debug_gem_matching) std::cout << "++No valid ALCT-CLCT matches in ME21" << std::endl;
        for (int bx_gem = bx_copad_start; bx_gem <= bx_copad_stop; bx_gem++) {
          if (not hasCoPads) {
            continue;
          }
          
          // find the best matching copad - first one 
          auto copads(matchingGEMPads(alct->bestALCT[bx_alct], coPads_[bx_gem], true));             
          if (debug_gem_matching) std::cout << "\t++Number of matching GEM CoPads in BX " << bx_alct << " : "<< copads.size() << std::endl;
          if (copads.size()==0) {
            continue;
          }
          
          correlateLCTsGEM(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
                           *(copads.at(0)).second, allLCTs[bx_alct][0][0], allLCTs[bx_alct][0][1]);
          if (allLCTs[bx_alct][0][0].isValid()) {
            ++nSuccesFulGEMMatches;            
            if (match_earliest_clct_me21_only) break;
          }
          if (debug_gem_matching) {
            std::cout << "Successful ALCT-GEM CoPad match in ME21: bx_alct = " << bx_alct << std::endl << std::endl;
            std::cout << "------------------------------------------------------------------------" << std::endl << std::endl;
          }
        }
      }

      if (debug_gem_matching) {
        std::cout << "========================================================================" << std::endl;
        std::cout << "Summary: " << std::endl;
        if (nSuccesFulMatches>1)
          std::cout << "Too many successful ALCT-CLCT matches in ME21: " << nSuccesFulMatches
                    << ", CSCDetId " << cscChamber->id()
                    << ", bx_alct = " << bx_alct
                    << "; match window: [" << bx_clct_start << "; " << bx_clct_stop << "]" << std::endl;
        else if (nSuccesFulMatches==1)
          std::cout << "1 successful ALCT-CLCT match in ME21: " 
                    << " CSCDetId " << cscChamber->id()
                    << ", bx_alct = " << bx_alct
                    << "; match window: [" << bx_clct_start << "; " << bx_clct_stop << "]" << std::endl;
        else if (nSuccesFulGEMMatches==1)
          std::cout << "1 successful ALCT-GEM match in ME21: " 
                    << " CSCDetId " << cscChamber->id()
                    << ", bx_alct = " << bx_alct
                    << "; match window: [" << bx_clct_start << "; " << bx_clct_stop << "]" << std::endl;
        else 
          std::cout << "Unsuccessful ALCT-CLCT match in ME21: " 
                    << "CSCDetId " << cscChamber->id()
                    << ", bx_alct = " << bx_alct
                    << "; match window: [" << bx_clct_start << "; " << bx_clct_stop << "]" << std::endl;
      }
    }
    // at this point we have invalid ALCTs --> try GEM pad matching
    else{
      auto coPads(coPads_[bx_alct]);
      if (runME21ILT_ and coPads.size() and buildLCTfromCLCTandGEM_) {
        //const int bx_clct_start(bx_alct - match_trig_window_size/2);
        //const int bx_clct_stop(bx_alct + match_trig_window_size/2);
        
        if (debug_gem_matching){ 
          std::cout << "========================================================================" << std::endl;
         // std::cout << "GEM-CLCT matching in ME2/1 chamber: " << cscChamber->id() << " in bx range: [" << bx_clct_start << "," << bx_clct_stop << "]" << std::endl;
	  std::cout <<"GEM-CLCT matching in ME2/1 chamber: "<< cscChamber->id()<< "in bx:"<<bx_alct<<std::endl;
	  std::cout << "------------------------------------------------------------------------" << std::endl;
        }
        // GEM-to-CLCT
        int nSuccesFulMatches = 0;
        //for (int bx_clct = bx_clct_start; bx_clct <= bx_clct_stop; bx_clct++)
       // {
         // if (bx_clct < 0 or bx_clct >= CSCCathodeLCTProcessor::MAX_CLCT_BINS) continue;
          if (drop_used_clcts and used_clct_mask[bx_alct]) continue;
          if (clct->bestCLCT[bx_alct].isValid())
          {          
            const int quality(clct->bestCLCT[bx_alct].getQuality());
            // only use high-Q stubs for the time being
            if (quality < 4) continue;

            ++nSuccesFulMatches;
            
            int mbx = std::abs(clct->bestCLCT[bx_alct].getBX()-bx_alct);
            int bx_gem = (coPads[0].second)->bx()+lct_central_bx;	    
            correlateLCTsGEM(clct->bestCLCT[bx_alct], clct->secondCLCT[bx_alct], *(coPads[0].second), GEMDetId(coPads[0].first).roll(),
                             allLCTs[bx_gem][mbx][0], allLCTs[bx_gem][mbx][1]);
            if (debug_gem_matching) {
              //	    if (infoV > 1) LogTrace("CSCMotherboard")
              std::cout << "Successful GEM-CLCT match in ME21: bx_alct = " << bx_alct <<std::endl;
                        //<< "; match window: [" << bx_clct_start << "; " << bx_clct_stop
                        //<< "]; bx_clct = " << bx_clct << std::endl;
              std::cout << "+++ Best CLCT Details: ";
              clct->bestCLCT[bx_alct].print();
              std::cout << "+++ Second CLCT Details: ";
              clct->secondCLCT[bx_alct].print();
            }
            if (allLCTs[bx_gem][mbx][0].isValid()) {
              used_clct_mask[bx_alct] += 1;
              if (match_earliest_clct_me21_only) break;
            }
        }
      }
    }
  }
  
  // Possibly use some discrimination from GEMs
  if (gemGeometryAvailable and runME21ILT_ and do_gem_matching) matchGEMPads();

  // reduction of nLCTs per each BX
  for (int bx = 0; bx < MAX_LCT_BINS; bx++)
  {
    // counting
    unsigned int n=0;
    for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
      for (int i=0;i<2;i++)
      {
        int cbx = bx + mbx - match_trig_window_size/2;
        if (allLCTs[bx][mbx][i].isValid())
        {
          ++n;
	  if (infoV > 0) LogDebug("CSCMotherboard") 
	    << "LCT"<<i+1<<" "<<bx<<"/"<<cbx<<": "<<allLCTs[bx][mbx][i]<<std::endl;
        }
      }
    
    // some simple cross-bx sorting algorithms
    if (tmb_cross_bx_algo == 1 and (n>2))
    {
      n=0;
      for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
        for (int i=0;i<2;i++)
        {
          if (allLCTs[bx][pref[mbx]][i].isValid())
          {
            n++;
            if (n>2) allLCTs[bx][pref[mbx]][i].clear();
          }
        }

      n=0;
      for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
        for (int i=0;i<2;i++)
        {
          int cbx = bx + mbx - match_trig_window_size/2;
          if (allLCTs[bx][mbx][i].isValid())
          {
            n++;
            if (infoV > 0) LogDebug("CSCMotherboard") 
              << "LCT"<<i+1<<" "<<bx<<"/"<<cbx<<": "<<allLCTs[bx][mbx][i]<<std::cout;
          }
        }
      if (infoV > 0 and n>0) LogDebug("CSCMotherboard") 
        <<"bx "<<bx<<" nnLCT:"<<n<<" "<<n<<std::endl;
    } // x-bx sorting
  }
  
  bool first = true;
  unsigned int n=0;
  for (auto p : readoutLCTs()) {
    if (debug_gem_matching and first){
      std::cout << "========================================================================" << std::endl;
      std::cout << "Counting the final LCTs" << std::endl;
      std::cout << "========================================================================" << std::endl;
      first = false;
      std::cout << "tmb_cross_bx_algo: " << tmb_cross_bx_algo << std::endl;        
    }
    n++;
    if (debug_gem_matching)
      std::cout << "LCT "<<n<<"  " << p <<std::endl;
  }
}


//readout LCTs 
std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME21GEM::readoutLCTs()
{
  return getLCTs();
}

//getLCTs when we use different sort algorithm
std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME21GEM::getLCTs()
{
  std::vector<CSCCorrelatedLCTDigi> result;
  for (int bx = 0; bx < MAX_LCT_BINS; bx++) {
    std::vector<CSCCorrelatedLCTDigi> tmpV;
    if (tmb_cross_bx_algo == 2) {
      tmpV = sortLCTsByQuality(bx);
      result.insert(result.end(), tmpV.begin(), tmpV.end());
    }
    else if (tmb_cross_bx_algo == 3) {
      tmpV = sortLCTsByGEMDPhi(bx);
      result.insert(result.end(), tmpV.begin(), tmpV.end());
    }
    else {
      for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++) {
        for (int i=0;i<2;i++) {
          if (allLCTs[bx][mbx][i].isValid()) {
            result.push_back(allLCTs[bx][mbx][i]);
          }
        }
      }
    }
  }
  return result;
}

//sort LCTs by Quality in each BX
std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME21GEM::sortLCTsByQuality(int bx)
{
  std::vector<CSCCorrelatedLCTDigi> LCTs;
  LCTs.clear();
  for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++) 
    for (int i=0;i<2;i++)
      if (allLCTs[bx][mbx][i].isValid())  
        LCTs.push_back(allLCTs[bx][mbx][i]);

  //std::cout<<"LCT before sorting in Bx:"<<bx<<std::endl;
  //for (auto p : LCTs)
    //  std::cout<< p <<std::endl;
  // return sorted vector with 2 highest quality LCTs
  std::sort(LCTs.begin(), LCTs.end(), CSCMotherboard::sortByQuality);
  if (LCTs.size()> max_me21_lcts) LCTs.erase(LCTs.begin()+max_me21_lcts, LCTs.end());
  //std::cout<<"LCT after sorting by quality in BX:"<<bx<<std::endl;
  //for (auto p : LCTs)
   //   std::cout<< p <<std::endl;
  return  LCTs;
}

//sort LCTs by dphi in each BX
std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME21GEM::sortLCTsByGEMDPhi(int bx)
{
  std::vector<CSCCorrelatedLCTDigi> LCTs;
  LCTs.clear();
  for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++) 
    for (int i=0;i<2;i++)
      if (allLCTs[bx][mbx][i].isValid())  
        LCTs.push_back(allLCTs[bx][mbx][i]);

  // return sorted vector with 2 highest quality LCTs
  std::sort(LCTs.begin(), LCTs.end(), CSCMotherboard::sortByGEMDphi);
  if (LCTs.size()> max_me21_lcts) LCTs.erase(LCTs.begin()+max_me21_lcts, LCTs.end());
  return  LCTs;
}


void CSCMotherboardME21GEM::correlateLCTsGEM(CSCALCTDigi bestALCT,
					  CSCALCTDigi secondALCT,
					  CSCCLCTDigi bestCLCT,
					  CSCCLCTDigi secondCLCT,
					  CSCCorrelatedLCTDigi& lct1,
					  CSCCorrelatedLCTDigi& lct2,
					  const GEMPadsBX& pads, 
					  const GEMPadsBX& copads)
{
  // check for pads 
  const int nPads(pads.size());
  const int nCoPads(copads.size());
  const bool hasPads(nPads!=0);
  const bool hasCoPads(nCoPads!=0);

  bool anodeBestValid     = bestALCT.isValid();
  bool anodeSecondValid   = secondALCT.isValid();
  bool cathodeBestValid   = bestCLCT.isValid();
  bool cathodeSecondValid = secondCLCT.isValid();

  if (anodeBestValid and !anodeSecondValid)     secondALCT = bestALCT;
  if (!anodeBestValid and anodeSecondValid)     bestALCT   = secondALCT;
  if (cathodeBestValid and !cathodeSecondValid) secondCLCT = bestCLCT;
  if (!cathodeBestValid and cathodeSecondValid) bestCLCT   = secondCLCT;

  // ALCT-CLCT matching conditions are defined by "trig_enable" configuration
  // parameters.
  if ((alct_trig_enable  and bestALCT.isValid()) or
      (clct_trig_enable  and bestCLCT.isValid()) or
      (match_trig_enable and bestALCT.isValid() and bestCLCT.isValid()))
  {
    lct1 = constructLCTsGEM(bestALCT, bestCLCT, hasPads, hasCoPads);
    lct1.setTrknmb(1);
  }

  if (((secondALCT != bestALCT) or (secondCLCT != bestCLCT)) and
      ((alct_trig_enable  and secondALCT.isValid()) or
       (clct_trig_enable  and secondCLCT.isValid()) or
       (match_trig_enable and secondALCT.isValid() and secondCLCT.isValid())))
  {
    lct2 = constructLCTsGEM(secondALCT, secondCLCT, hasPads, hasCoPads);
    lct2.setTrknmb(2);
  }
}


void CSCMotherboardME21GEM::correlateLCTsGEM(CSCALCTDigi bestALCT,
					  CSCALCTDigi secondALCT,
					  GEMPadDigi gemPad,
					  CSCCorrelatedLCTDigi& lct1,
					  CSCCorrelatedLCTDigi& lct2)
{
  bool anodeBestValid     = bestALCT.isValid();
  bool anodeSecondValid   = secondALCT.isValid();

  if (anodeBestValid and !anodeSecondValid)     secondALCT = bestALCT;
  if (!anodeBestValid and anodeSecondValid)     bestALCT   = secondALCT;

  if ((alct_trig_enable  and bestALCT.isValid()) or
      (match_trig_enable and bestALCT.isValid()))
  {
    lct1 = constructLCTsGEM(bestALCT, gemPad, useOldLCTDataFormat_);
    lct1.setTrknmb(1);
    // lct1.setGEMDPhi(0.0);
  }

  if ((alct_trig_enable  and secondALCT.isValid()) or
      (match_trig_enable and secondALCT.isValid() and secondALCT != bestALCT))
  {
    lct2 = constructLCTsGEM(secondALCT, gemPad, useOldLCTDataFormat_);
    lct2.setTrknmb(2);
    // lct2.setGEMDPhi(0.0);
  }
}


void CSCMotherboardME21GEM::correlateLCTsGEM(CSCCLCTDigi bestCLCT,
                                             CSCCLCTDigi secondCLCT,
                                             GEMPadDigi gemPad, int roll,
                                             CSCCorrelatedLCTDigi& lct1,
                                             CSCCorrelatedLCTDigi& lct2)
{
  bool cathodeBestValid     = bestCLCT.isValid();
  bool cathodeSecondValid   = secondCLCT.isValid();

  if (cathodeBestValid and !cathodeSecondValid)     secondCLCT = bestCLCT;
  if (!cathodeBestValid and cathodeSecondValid)     bestCLCT   = secondCLCT;

  if ((clct_trig_enable  and bestCLCT.isValid()) or
      (match_trig_enable and bestCLCT.isValid()))
  {
    lct1 = constructLCTsGEM(bestCLCT, gemPad, roll, useOldLCTDataFormat_);
    lct1.setTrknmb(1);
  }

  if ((clct_trig_enable  and secondCLCT.isValid()) or
       (match_trig_enable and secondCLCT.isValid() and secondCLCT != bestCLCT))
    {
    lct2 = constructLCTsGEM(secondCLCT, gemPad, roll, useOldLCTDataFormat_);
    lct2.setTrknmb(2);
  }
}


CSCCorrelatedLCTDigi CSCMotherboardME21GEM::constructLCTsGEM(const CSCALCTDigi& alct,
                                                          const GEMPadDigi& gem, 
                                                          bool oldDataFormat) 
{    
  if (oldDataFormat){
    // CLCT pattern number - set it to a highest value
    // hack to get LCTs in the CSCTF
    unsigned int pattern = promoteALCTGEMpattern_ ? 10 : 0;
    
    // LCT quality number - set it to a very high value 
    // hack to get LCTs in the CSCTF
    unsigned int quality = promoteALCTGEMquality_ ? 14 : 11;

    // Bunch crossing
    int bx = alct.getBX();
    
    // get keyStrip from LUT
    int keyStrip = gemPadToCscHs_[gem.pad()];
    
    // get wiregroup from ALCT
    int wg = alct.getKeyWG();

     // construct correlated LCT; temporarily assign track number of 0.
    return CSCCorrelatedLCTDigi(0, 1, quality, wg, keyStrip, pattern, 0, bx, 0, 0, 0, theTrigChamber);
  } 
  else {    
    // CLCT pattern number - no pattern
    unsigned int pattern = 0;

    // LCT quality number
    unsigned int quality = 1;
    
    // Bunch crossing
    int bx = gem.bx() + lct_central_bx;
    
    // get keyStrip from LUT
    int keyStrip = gemPadToCscHs_[gem.pad()];
    // get wiregroup from ALCT
    int wg = alct.getKeyWG();
    
    // construct correlated LCT; temporarily assign track number of 0.
    return CSCCorrelatedLCTDigi(0, 1, quality, wg, keyStrip, pattern, 0, bx, 0, 0, 0, theTrigChamber);
  }
}

CSCCorrelatedLCTDigi CSCMotherboardME21GEM::constructLCTsGEM(const CSCCLCTDigi& clct,
                                                          const GEMPadDigi& gem, int roll, 
                                                          bool oldDataFormat) 
{
  if (oldDataFormat){
    // CLCT pattern number - for the time being, do not include GEMs in the pattern
    unsigned int pattern = encodePattern(clct.getPattern(), clct.getStripType());
    
    // LCT quality number -  dummy quality
    unsigned int quality = promoteCLCTGEMquality_ ? 14 : 11;
    
    // Bunch crossing: pick GEM bx
    int bx = gem.bx() + lct_central_bx;
    
    // pick a random WG in the roll range    
    int wg(20);
    
    // construct correlated LCT; temporarily assign track number of 0.
    return CSCCorrelatedLCTDigi(0, 1, quality, wg, clct.getKeyStrip(), pattern, clct.getBend(), bx, 0, 0, 0, theTrigChamber);
  }
  else {
    // CLCT pattern number - no pattern
    unsigned int pattern = 0;//encodePatternGEM(clct.getPattern(), clct.getStripType());
    
    // LCT quality number -  dummy quality
    unsigned int quality = 5;//findQualityGEM(alct, gem);
    
    // Bunch crossing: get it from cathode LCT if anode LCT is not there.
    int bx = gem.bx() + lct_central_bx;;
    
    // ALCT WG
    int wg(0);
    
    // construct correlated LCT; temporarily assign track number of 0.
    return CSCCorrelatedLCTDigi(0, 1, quality, wg, 0, pattern, 0, bx, 0, 0, 0, theTrigChamber);
  }
}


CSCCorrelatedLCTDigi CSCMotherboardME21GEM::constructLCTsGEM(const CSCALCTDigi& aLCT, const CSCCLCTDigi& cLCT, 
							  bool hasPad, bool hasCoPad)
{
  // CLCT pattern number
  unsigned int pattern = encodePattern(cLCT.getPattern(), cLCT.getStripType());

  // LCT quality number
  unsigned int quality = findQualityGEM(aLCT, cLCT, hasPad, hasCoPad);

  // Bunch crossing: get it from cathode LCT if anode LCT is not there.
  int bx = aLCT.isValid() ? aLCT.getBX() : cLCT.getBX();

  // construct correlated LCT; temporarily assign track number of 0.
  int trknmb = 0;
  CSCCorrelatedLCTDigi thisLCT(trknmb, 1, quality, aLCT.getKeyWG(),
                               cLCT.getKeyStrip(), pattern, cLCT.getBend(),
                               bx, 0, 0, 0, theTrigChamber);
  return thisLCT;
}


unsigned int CSCMotherboardME21GEM::findQualityGEM(const CSCALCTDigi& aLCT, const CSCCLCTDigi& cLCT,
						bool hasPad, bool hasCoPad)
{

  /*
    Same LCT quality definition as standard LCTs
    c4 takes GEMs into account!!!
  */

  unsigned int quality = 0;

  if (!isTMB07) {
    bool isDistrip = (cLCT.getStripType() == 0);

    if (aLCT.isValid() && !(cLCT.isValid())) {    // no CLCT
      if (aLCT.getAccelerator()) {quality =  1;}
      else                       {quality =  3;}
    }
    else if (!(aLCT.isValid()) && cLCT.isValid()) { // no ALCT
      if (isDistrip)             {quality =  4;}
      else                       {quality =  5;}
    }
    else if (aLCT.isValid() && cLCT.isValid()) { // both ALCT and CLCT
      if (aLCT.getAccelerator()) {quality =  2;} // accelerator muon
      else {                                     // collision muon
        // CLCT quality is, in fact, the number of layers hit, so subtract 3
        // to get quality analogous to ALCT one.
        int sumQual = aLCT.getQuality() + (cLCT.getQuality()-3);
        if (sumQual < 1 || sumQual > 6) {
          if (infoV >= 0) edm::LogWarning("L1CSCTPEmulatorWrongValues")
            << "+++ findQuality: sumQual = " << sumQual << "+++ \n";
        }
        if (isDistrip) { // distrip pattern
          if (sumQual == 2)      {quality =  6;}
          else if (sumQual == 3) {quality =  7;}
          else if (sumQual == 4) {quality =  8;}
          else if (sumQual == 5) {quality =  9;}
          else if (sumQual == 6) {quality = 10;}
        }
        else {            // halfstrip pattern
          if (sumQual == 2)      {quality = 11;}
          else if (sumQual == 3) {quality = 12;}
          else if (sumQual == 4) {quality = 13;}
          else if (sumQual == 5) {quality = 14;}
          else if (sumQual == 6) {quality = 15;}
        }
      }
    }
  }
#ifdef OLD
  else {
    // Temporary definition, used until July 2008.
    // First if statement is fictitious, just to help the CSC TF emulator
    // handle such cases (one needs to make sure they will be accounted for
    // in the new quality definition.
    if (!(aLCT.isValid()) || !(cLCT.isValid())) {
      if (aLCT.isValid() && !(cLCT.isValid()))      quality = 1; // no CLCT
      else if (!(aLCT.isValid()) && cLCT.isValid()) quality = 2; // no ALCT
      else quality = 0; // both absent; should never happen.
    }
    else {
      // Sum of ALCT and CLCT quality bits.  CLCT quality is, in fact, the
      // number of layers hit, so subtract 3 to put it to the same footing as
      // the ALCT quality.
      int sumQual = aLCT.getQuality() + (cLCT.getQuality()-3);
      if (sumQual < 1 || sumQual > 6) {
        if (infoV >= 0) edm::LogWarning("L1CSCTPEmulatorWrongValues")
          << "+++ findQuality: Unexpected sumQual = " << sumQual << "+++\n";
      }

      // LCT quality is basically the sum of ALCT and CLCT qualities, but split
      // in two groups depending on the CLCT pattern id (higher quality for
      // straighter patterns).
      int offset = 0;
      if (cLCT.getPattern() <= 7) offset = 4;
      else                        offset = 9;
      quality = offset + sumQual;
    }
  }
#endif
  else {
    // 2008 definition.
    if (!(aLCT.isValid()) || !(cLCT.isValid())) {
      if (aLCT.isValid() && !(cLCT.isValid()))      quality = 1; // no CLCT
      else if (!(aLCT.isValid()) && cLCT.isValid()) quality = 2; // no ALCT
      else quality = 0; // both absent; should never happen.
    }
    else {
      const int pattern(cLCT.getPattern());
      if (pattern == 1) quality = 3; // layer-trigger in CLCT
      else {
        // ALCT quality is the number of layers hit minus 3.
        // CLCT quality is the number of layers hit.
	int n_gem = 0;  
	if (hasPad) n_gem = 1;
	if (hasCoPad) n_gem = 2;
	const bool a4((aLCT.getQuality() >= 1) or (aLCT.getQuality() >= 0 and n_gem >=1));
	const bool c4((cLCT.getQuality() >= 4) or (cLCT.getQuality() >= 3 and n_gem>=1));
        //              quality = 4; "reserved for low-quality muons in future"
        if      (!a4 && !c4) quality = 5; // marginal anode and cathode
        else if ( a4 && !c4) quality = 6; // HQ anode, but marginal cathode
        else if (!a4 &&  c4) quality = 7; // HQ cathode, but marginal anode
        else if ( a4 &&  c4) {
          if (aLCT.getAccelerator()) quality = 8; // HQ muon, but accel ALCT
          else {
            // quality =  9; "reserved for HQ muons with future patterns
            // quality = 10; "reserved for HQ muons with future patterns
            if (pattern == 2 || pattern == 3)      quality = 11;
            else if (pattern == 4 || pattern == 5) quality = 12;
            else if (pattern == 6 || pattern == 7) quality = 13;
            else if (pattern == 8 || pattern == 9) quality = 14;
            else if (pattern == 10)                quality = 15;
            else {
              if (infoV >= 0) edm::LogWarning("L1CSCTPEmulatorWrongValues")
                << "+++ findQuality: Unexpected CLCT pattern id = "
                << pattern << "+++\n";
            }
          }
        }
      }
    }
  }
  return quality;
}


void CSCMotherboardME21GEM::buildCoincidencePads(const GEMPadDigiCollection* out_pads, 
	                                         GEMCoPadDigiCollection& out_co_pads,
						 CSCDetId csc_id)
{
  gemCoPadV.clear();

  // build coincidences
  for (auto det_range = out_pads->begin(); det_range != out_pads->end(); ++det_range) {
    const GEMDetId& id = (*det_range).first;

    // same chamber
    if (id.region() != csc_id.zendcap() or 
	id.ring() != csc_id.ring() or id.chamber() != csc_id.chamber()) continue;

    // build coincidences only for long superchamber pads
    if (id.station() != 3) continue;
    
    // all coincidences detIDs will have layer=1
    if (id.layer() != 1) continue;
    
    // find the corresponding id with layer=2
    GEMDetId co_id(id.region(), id.ring(), id.station(), 2, id.chamber(), id.roll());
    
    auto co_pads_range = out_pads->get(co_id);
    // empty range = no possible coincidence pads
    if (co_pads_range.first == co_pads_range.second) continue;
      
    // now let's correlate the pads in two layers of this partition
    const auto& pads_range = (*det_range).second;
    for (auto p = pads_range.first; p != pads_range.second; ++p) {
      for (auto co_p = co_pads_range.first; co_p != co_pads_range.second; ++co_p) {
	  // check the match in pad
        if (std::abs(p->pad() - co_p->pad()) > maxDeltaPadInCoPad_) continue;
        // check the match in BX
        if (std::abs(p->bx() - co_p->bx()) > maxDeltaBXInCoPad_ ) continue;
        
        // make a new coincidence pad digi
        gemCoPadV.push_back(GEMCoPadDigi(id.roll(),*p,*co_p));
        out_co_pads.insertDigi(id, GEMCoPadDigi(id.roll(),*p,*co_p));
      }
    }
  }
}


std::map<int,std::pair<double,double> >
CSCMotherboardME21GEM::createGEMRollEtaLUT()
{
  std::map<int,std::pair<double,double> > result;

  auto chamber(gem_g->chamber(GEMDetId(1,1,3,1,1,0)));
  if (chamber==nullptr) return result;

  for(int i = 1; i<= chamber->nEtaPartitions(); ++i){
    auto roll(chamber->etaPartition(i));
    if (roll==nullptr) continue;
    const float half_striplength(roll->specs()->specificTopology().stripLength()/2.);
    const LocalPoint lp_top(0., half_striplength, 0.);
    const LocalPoint lp_bottom(0., -half_striplength, 0.);
    const GlobalPoint gp_top(roll->toGlobal(lp_top));
    const GlobalPoint gp_bottom(roll->toGlobal(lp_bottom));
    //result[i] = std::make_pair(floorf(gp_top.eta() * 100) / 100, ceilf(gp_bottom.eta() * 100) / 100);
    result[i] = std::make_pair(gp_top.eta(), gp_bottom.eta());
  }
  return result;
}

void CSCMotherboardME21GEM::retrieveGEMPads(const GEMPadDigiCollection* gemPads, unsigned id)
{
  auto superChamber(gem_g->superChamber(id));
  for (auto ch : superChamber->chambers()) {
    for (auto roll : ch->etaPartitions()) {
      GEMDetId roll_id(roll->id());
      auto pads_in_det = gemPads->get(roll_id);
      for (auto pad = pads_in_det.first; pad != pads_in_det.second; ++pad) {
        auto id_pad = std::make_pair(roll_id, &(*pad));
        const int bx_shifted(lct_central_bx + pad->bx());
        for (int bx = bx_shifted - maxDeltaBXPad_;bx <= bx_shifted + maxDeltaBXPad_; ++bx) {
	  pads_[bx].push_back(id_pad);  
        }
      }
    }
  }
}

void CSCMotherboardME21GEM::retrieveGEMCoPads(const GEMCoPadDigiCollection* gemPads, unsigned id)
{
  auto superChamber(gem_g->superChamber(id));
  for (auto ch : superChamber->chambers()) {
    for (auto roll : ch->etaPartitions()) {
      GEMDetId roll_id(roll->id());
      auto pads_in_det = gemPads->get(roll_id);
      for (auto pad = pads_in_det.first; pad != pads_in_det.second; ++pad) {
	GEMPadDigi gfirst((*pad).first());
	auto id_pad = std::make_pair(roll_id, &gfirst);
	const int bx_shifted(lct_central_bx + (pad->first()).bx());
	for (int bx = bx_shifted - maxDeltaBXPad_;bx <= bx_shifted + maxDeltaBXPad_; ++bx) {
	  if(bx != lct_central_bx) continue;
	  coPads_[bx].push_back(id_pad);  
	}
      }
    }
  }
}

void CSCMotherboardME21GEM::printGEMTriggerPads(int bx_start, int bx_stop, bool iscopad)
{
  // pads or copads?
  auto thePads(!iscopad ? pads_ : coPads_ );
  const bool hasPads(thePads.size()!=0);
  
  std::cout << "------------------------------------------------------------------------" << std::endl;
  bool first = true;
  for (int bx = bx_start; bx <= bx_stop; bx++) {
    // print only the pads for the central BX
    //if (bx!=lct_central_bx and iscopad) continue;
    std::vector<std::pair<unsigned int, const GEMPadDigi*> > in_pads = thePads[bx];
    if (first) {
      if (!iscopad) std::cout << "* GEM trigger pads: " << std::endl;
      else          std::cout << "* GEM trigger coincidence pads: " << std::endl;
    }
    first = false;
    if (!iscopad) std::cout << "N(pads) BX " << bx << " : " << in_pads.size() << std::endl;
    else          std::cout << "N(copads) BX " << bx << " : " << in_pads.size() << std::endl;
    if (hasPads){
      for (auto pad : in_pads){
        auto roll_id(GEMDetId(pad.first));
        std::cout << "\tdetId " << pad.first << " " << roll_id << ", pad = " << pad.second->pad() << ", BX = " << pad.second->bx() + 6 << std::endl;
      }
    }
    else
      break;
  }
}


CSCMotherboardME21GEM::GEMPadsBX  
CSCMotherboardME21GEM::matchingGEMPads(const CSCCLCTDigi& clct, const GEMPadsBX& pads, bool isCoPad, bool first)
{
  CSCMotherboardME21GEM::GEMPadsBX result;

  // fetch the low and high pad edges for the long superchambers
  int deltaPad(isCoPad ? maxDeltaPadCoPad_ : maxDeltaPadPad_);
  int deltaBX(isCoPad ? maxDeltaBXCoPad_ : maxDeltaBXPad_);
  int clct_bx = clct.getBX();
  const int lowPad(cscHsToGemPad_[clct.getKeyStrip()].first);
  const int highPad(cscHsToGemPad_[clct.getKeyStrip()].second);
  const bool debug(false);
  if (debug) std::cout << "lowpad " << lowPad << " highpad " << highPad << " delta pad " << deltaPad <<std::endl;
  for (auto p: pads){
    if (debug) std::cout<<"DetId"<<GEMDetId(p.first)<<"   "<< *(p.second)<<std::endl;
    auto padRoll((p.second)->pad());
    int pad_bx = (p.second)->bx()+lct_central_bx;
    if (debug) std::cout << "padRoll " << padRoll << std::endl;
    if (std::abs(clct_bx-pad_bx)>deltaBX) continue;
    if (std::abs(lowPad - padRoll) <= deltaPad or std::abs(padRoll - highPad) <= deltaPad){
    if (debug) std::cout << "++Matches! " << std::endl;
      result.push_back(p);
      if (first) return result;      
    }
  }
  return result;
}


CSCMotherboardME21GEM::GEMPadsBX 
CSCMotherboardME21GEM::matchingGEMPads(const CSCALCTDigi& alct, const GEMPadsBX& pads, bool isCoPad, bool first)
{
  CSCMotherboardME21GEM::GEMPadsBX result;
  int deltaBX(isCoPad ? maxDeltaBXCoPad_ : maxDeltaBXPad_);
  int alct_bx = alct.getBX();
  int Wg = alct.getKeyWG();
  std::vector<int> Rolls;
  Rolls.push_back(cscWgToGemRoll_[Wg]);
  if (Wg>=maxDeltaWg_ && cscWgToGemRoll_[Wg] != cscWgToGemRoll_[Wg-maxDeltaWg_]) 
      Rolls.push_back(cscWgToGemRoll_[Wg-maxDeltaWg_]); 
  if ((unsigned int)(Wg+maxDeltaWg_)<cscWgToGemRoll_.size() && cscWgToGemRoll_[Wg] != cscWgToGemRoll_[Wg+maxDeltaWg_])
      Rolls.push_back(cscWgToGemRoll_[Wg+maxDeltaWg_]);

  const bool debug(false);
  if (debug) std::cout << "ALCT keyWG " << alct.getKeyWG() << std::endl;
  for (auto alctRoll : Rolls)
  {
  if (debug) std::cout <<"roll " << alctRoll << std::endl;
  for (auto p: pads){
    auto padRoll(GEMDetId(p.first).roll());
    int pad_bx = (p.second)->bx()+lct_central_bx;
    if (debug) std::cout<<"Detid "<< GEMDetId(p.first) <<"  "<< *(p.second)<<std::endl;
    if (debug) std::cout << "Candidate ALCT: " << padRoll << std::endl;
    if (std::abs(alct_bx-pad_bx)>deltaBX) continue;
    if (alctRoll !=  padRoll) continue;
    if (debug) std::cout << "++Matches! " << std::endl;
    result.push_back(p);
    if (first) return result;
  }
  }
  return result;
}


CSCMotherboardME21GEM::GEMPadsBX 
CSCMotherboardME21GEM::matchingGEMPads(const CSCCLCTDigi& clct, const CSCALCTDigi& alct, const GEMPadsBX& pads, 
                                    bool isCoPad, bool first)
{
  CSCMotherboardME21GEM::GEMPadsBX result;

  // Fetch all (!) pads matching to ALCTs and CLCTs
  auto padsClct(matchingGEMPads(clct, pads, isCoPad, false));
  auto padsAlct(matchingGEMPads(alct, pads, isCoPad, false));

  const bool debug(false);
  if (debug) std::cout << "-----------------------------------------------------------------------"<<std::endl;
  // Check if the pads overlap
  for (auto p : padsAlct){
    if (debug) std::cout<< "Candidate ALCT: " << p.first << " " << *(p.second) << std::endl;
    for (auto q: padsClct){
      if (debug) std::cout<< "++Candidate CLCT: " << q.first << " " << *(q.second) << std::endl;
      // look for exactly the same pads
      if ((p.first != q.first) or GEMPadDigi(*(p.second)) != *(q.second)) continue;
      if (debug) std::cout << "++Matches! " << std::endl;
      result.push_back(p);
      if (first) return result;
    }
  }
  if (debug) std::cout << "-----------------------------------------------------------------------"<<std::endl;
  return result;
}

void CSCMotherboardME21GEM::matchGEMPads()
{
  /*
  using namespace std;

  // check if we have any LCTs at all
  int nlct = 0;
  for (int bx = 0; bx < MAX_LCT_BINS; bx++)
    for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
      for (int i=0;i<2;i++)
      {
        CSCCorrelatedLCTDigi& lct = allLCTs[bx][mbx][i];
        if (lct.isValid()) nlct++;
      }
  if (nlct == 0) return;

  // retrieve CSCChamber geometry
  CSCTriggerGeomManager* geo_manager = CSCTriggerGeometry::get();
  CSCChamber* cscChamber = geo_manager->chamber(theEndcap, theStation, theSector, theSubsector, theTrigChamber);
  const CSCDetId csc_id(cscChamber->id());
  const int chamber(csc_id.chamber());
  const bool is_odd(chamber%2==1);
  

  if (debug_gem_dphi) std::cout<<"++++++++  matchGEMPads "<< csc_id <<" +++++++++ "<<std::endl;

  // check if there are any pads 
  if (pads_.empty()) {
    if (debug_gem_dphi) std::cout<<"igotnopads"<<std::endl;
    return;
  }

  // walk over BXs
  for (int bx = 0; bx < MAX_LCT_BINS; ++bx)
  {
    auto in_pads = pads_.find(bx);

    // walk over potential LCTs in this BX
    for (unsigned int mbx = 0; mbx < match_trig_window_size; ++mbx)
      for (int i=0; i<2; ++i)
      {
        CSCCorrelatedLCTDigi& lct = allLCTs[bx][mbx][i];
        if (!lct.isValid() or fabs(lct.getGEMDPhi()) < 0.000001) continue;
        if (debug_gem_dphi) std::cout<<"LCTbefore "<<bx<<" "<<mbx<<" "<<i<<" "<<lct;

        // use -99 as default value whe we don't know if there could have been a gem match
        lct.setGEMDPhi(-99.);

        // "strip" here is actually a half-strip in geometry's terms
        // note that LCT::getStrip() starts from 0
        float fractional_strip = 0.5 * (lct.getStrip() + 1) - 0.25;
        auto layer_geo = cscChamber->layer(CSCConstants::KEY_CLCT_LAYER)->geometry();
        // LCT::getKeyWG() also starts from 0
        float wire = layer_geo->middleWireOfGroup(lct.getKeyWG() + 1);

        LocalPoint csc_intersect = layer_geo->intersectionOfStripAndWire(fractional_strip, wire);
        GlobalPoint csc_gp = csc_g->idToDet(csc_id)->surface().toGlobal(csc_intersect);

        // is LCT located in the high efficiency GEM eta range?
        bool gem_fid = ( std::abs(csc_gp.eta()) >= gem_match_min_eta );

        if (debug_gem_dphi) std::cout<<" lct eta "<<csc_gp.eta()<<" phi "<<csc_gp.phi()<<std::endl;

        if (!gem_fid)
        {
          if (debug_gem_dphi) std::cout<<"    -- lct pass no gem req"<<std::endl;
          continue;
        }
        // use 100 ad default value when within gem fiducial region
        lct.setGEMDPhi(100.);
	
        if (in_pads == pads_.end()) // has no potential GEM hits with similar BX -> zap it
        {
          if (gem_clear_nomatch_lcts) lct.clear();
          if (debug_gem_dphi) std::cout<<"    -- no gem"<<std::endl;
          continue;
        }
        if (debug_gem_dphi) std::cout<<"    -- gem possible"<<std::endl;
        // use 99 ad default value when we expect there to be a gem match
        lct.setGEMDPhi(99.);

        // to consider a GEM pad as "matched" it has to be 
        // within specified delta_eta and delta_phi ranges
        // and if there are multiple ones, only the min|delta_phi| is considered as matched
        bool gem_matched = false;
        //int gem_bx = 99;
        float min_dphi = 99.;
        for (auto& id_pad: in_pads->second)
        {
          GEMDetId gem_id(id_pad.first);
          LocalPoint gem_lp = gem_g->etaPartition(gem_id)->centreOfPad(id_pad.second->pad());
          GlobalPoint gem_gp = gem_g->idToDet(gem_id)->surface().toGlobal(gem_lp);
          float dphi = deltaPhi(csc_gp.phi(), gem_gp.phi());
          float deta = csc_gp.eta() - gem_gp.eta();
          if (debug_gem_dphi) std::cout<<"    gem with dphi "<< std::abs(dphi) <<" deta "<< std::abs(deta) <<std::endl;

          if( (              std::abs(deta) <= gem_match_delta_eta        ) and // within delta_eta
              ( (  is_odd and std::abs(dphi) <= gem_match_delta_phi_odd ) or
                ( !is_odd and std::abs(dphi) <= gem_match_delta_phi_even ) ) and // within delta_phi
              ( std::abs(dphi) < std::abs(min_dphi) )                          // minimal delta phi
            )
          {
            gem_matched = true;
            min_dphi = dphi;
            //gem_bx = id_pad.second->bx();
          }
        }
        if (gem_matched)
        {
          if (debug_gem_dphi) std::cout<<" GOT MATCHED GEM!"<<std::endl;
          lct.setGEMDPhi(min_dphi);
	  // assing the bit value
	  int oddEven = int(not is_odd) + 1;
	  int numberOfBendAngles(sizeof lut_pt_vs_dphi_gemcsc / sizeof *lut_pt_vs_dphi_gemcsc);
	  int iFound = 0;
	  if (abs(min_dphi) < lut_pt_vs_dphi_gemcsc[numberOfBendAngles-1][oddEven]) iFound = numberOfBendAngles;
	  else {
	    for (int i=0; i< numberOfBendAngles-1; ++i) {
	      if (debug_gem_dphi) std::cout<<"is_odd "<<is_odd <<" min_dphi "<<abs(min_dphi)<<" bend angle lib "<<i<<" "<<lut_pt_vs_dphi_gemcsc[i][oddEven]<< std::endl;
	      if (abs(min_dphi) < lut_pt_vs_dphi_gemcsc[i][oddEven] and abs(min_dphi) > lut_pt_vs_dphi_gemcsc[i+1][oddEven]) 
		iFound = i+1;
	    }
	  }
	  lct.setGEMDPhiBits(iFound);
	  if (debug_gem_dphi) std::cout<<"found bend angle "<<abs(min_dphi)<<" "<<lct.getGEMDPhiBits()<<" "<<lut_pt_vs_dphi_gemcsc[iFound][oddEven]<<" "<<iFound << std::endl;
        }
        else
        {
          if (debug_gem_dphi) std::cout<<" no gem match";
          if (gem_clear_nomatch_lcts)
          {
            lct.clear();
            if (debug_gem_dphi) std::cout<<" - cleared lct";
          }
          if (debug_gem_dphi) std::cout<<std::endl;
        }
        if (debug_gem_dphi) std::cout<<"LCTafter "<<bx<<" "<<mbx<<" "<<i<<" "<<lct;
      }
  }

  // final count
  int nlct_after = 0;
  for (int bx = 0; bx < MAX_LCT_BINS; bx++)
    for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
      for (int i=0;i<2;i++)
      {
        if (allLCTs[bx][mbx][i].isValid()) nlct_after++;
      }
  if (debug_gem_dphi) std::cout<<"before "<<nlct<<"  after "<<nlct_after<<std::endl;
  */
}


int CSCMotherboardME21GEM::assignGEMRoll(double eta)
{
  int result = -99;
  for(auto p : gemRollToEtaLimits_) {  
    const float minEta((p.second).first);
    const float maxEta((p.second).second);
    if (minEta <= eta and eta <= maxEta) {
      result = p.first;
      break;
    }
  }
  return result;
}


std::vector<GEMCoPadDigi> CSCMotherboardME21GEM::readoutCoPads()
{
  return gemCoPadV;
}
