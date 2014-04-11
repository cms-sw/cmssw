#include <L1Trigger/CSCTriggerPrimitives/src/CSCMotherboardME21.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h>
#include <Geometry/GEMGeometry/interface/GEMGeometry.h>
#include <Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h>
#include <iomanip> 

const double CSCMotherboardME21::lut_wg_eta_odd[112][2] = {
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

const double CSCMotherboardME21::lut_wg_eta_even[112][2] = {
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

CSCMotherboardME21::CSCMotherboardME21(unsigned endcap, unsigned station,
                               unsigned sector, unsigned subsector,
                               unsigned chamber,
                               const edm::ParameterSet& conf) :
  CSCMotherboard(endcap, station, sector, subsector, chamber, conf)
{
  edm::ParameterSet commonParams = conf.getParameter<edm::ParameterSet>("commonParam");
  
  if (!isSLHC) edm::LogError("L1CSCTPEmulatorConfigError")
    << "+++ Upgrade CSCMotherboardME21 constructed while isSLHC is not set! +++\n";
  
  edm::ParameterSet alctParams = conf.getParameter<edm::ParameterSet>("alctSLHC");
  edm::ParameterSet clctParams = conf.getParameter<edm::ParameterSet>("clctSLHC");
  edm::ParameterSet tmbParams = conf.getParameter<edm::ParameterSet>("tmbSLHC");
  edm::ParameterSet me21tmbParams = tmbParams.getParameter<edm::ParameterSet>("me21ILT");

  // central bx for LCT is 6 for simulation
  lct_central_bx = tmbParams.getUntrackedParameter<int>("lctCentralBX", 6);

  // whether to not reuse CLCTs that were used by previous matching ALCTs
  // in ALCT-to-CLCT algorithm
  drop_used_clcts = tmbParams.getUntrackedParameter<bool>("tmbDropUsedClcts",true);

  //----------------------------------------------------------------------------------------//

  //       G E M  -  C S C   I N T E G R A T E D   L O C A L   A L G O R I T H M

  //----------------------------------------------------------------------------------------//

  // masterswitch
  runME21ILT_ = me21tmbParams.getUntrackedParameter<bool>("runME21ILT",false);

  // debug gem matching
  debug_gem_matching = me21tmbParams.getUntrackedParameter<bool>("debugGemMatching", false);

  // print available pads
  print_available_pads = me21tmbParams.getUntrackedParameter<bool>("printAvailablePads", false);

  //  deltas used to construct GEM coincidence pads
  maxDeltaBXInCoPad_ = me21tmbParams.getUntrackedParameter<int>("maxDeltaBXInCoPad",1);
  maxDeltaRollInCoPad_ = me21tmbParams.getUntrackedParameter<int>("maxDeltaRollInCoPad",0);
  maxDeltaPadInCoPad_ = me21tmbParams.getUntrackedParameter<int>("maxDeltaPadInCoPad",0);

  //  deltas used to match to GEM pads
  maxDeltaBXPad_ = me21tmbParams.getUntrackedParameter<int>("maxDeltaBXPad",1);
  maxDeltaRollPad_ = me21tmbParams.getUntrackedParameter<int>("maxDeltaRollPad",0);
  maxDeltaPadPad_ = me21tmbParams.getUntrackedParameter<int>("maxDeltaPadPad",0);

  //  deltas used to match to GEM coincidence pads
  maxDeltaBXCoPad_ = me21tmbParams.getUntrackedParameter<int>("maxDeltaBXCoPad",1);
  maxDeltaRollCoPad_ = me21tmbParams.getUntrackedParameter<int>("maxDeltaRollCoPad",0);
  maxDeltaPadCoPad_ = me21tmbParams.getUntrackedParameter<int>("maxDeltaPadCoPad",0);

  // drop low quality stubs if they don't have GEMs
  dropLowQualityCLCTsNoGEMs_ = me21tmbParams.getUntrackedParameter<bool>("dropLowQualityCLCTsNoGEMs",false);
  dropLowQualityALCTsNoGEMs_ = me21tmbParams.getUntrackedParameter<bool>("dropLowQualityALCTsNoGEMs",false);

  // use only the central BX for GEM matching
  centralBXonlyGEM_ = me21tmbParams.getUntrackedParameter<bool>("centralBXonlyGEM",false);
  
  // build LCT from ALCT and GEM
  buildLCTfromALCTandGEM_ = me21tmbParams.getUntrackedParameter<bool>("buildLCTfromALCTandGEM",false);
  buildLCTfromCLCTandGEM_ = me21tmbParams.getUntrackedParameter<bool>("buildLCTfromCLCTandGEM",false);
}

CSCMotherboardME21::~CSCMotherboardME21() 
{
}

void CSCMotherboardME21::clear()
{
  CSCMotherboard::clear();

  cscWgToGemRollShort_.clear();
  cscWgToGemRollLong_.clear();
  gemPadToCscHs_.clear();
  cscHsToGemPad_.clear();
  padsShort_.clear();
  padsLong_.clear();
  coPadsShort_.clear();
  coPadsLong_.clear();
}

void
CSCMotherboardME21::run(const CSCWireDigiCollection* wiredc,
                    const CSCComparatorDigiCollection* compdc,
                    const GEMCSCPadDigiCollection* gemPads) 
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
    const GEMDetId gem_id_short(region, 1, 2, 1, csc_id.chamber(), 0);
    const GEMDetId gem_id_long(region, 1, 3, 1, csc_id.chamber(), 0);
    //    const GEMChamber* gemChamberShort(gem_g->chamber(gem_id_short));
    const GEMChamber* gemChamberLong(gem_g->chamber(gem_id_long));
    
    // LUT<roll,<etaMin,etaMax> >    
    gemPadToEtaLimitsShort_ = createGEMPadLUT(false);
    gemPadToEtaLimitsLong_ = createGEMPadLUT(true);

    bool debug(false);
    if (debug){
      if (gemPadToEtaLimitsShort_.size())
        for(auto p : gemPadToEtaLimitsShort_) {
          std::cout << "pad "<< p.first << " min eta " << (p.second).first << " max eta " << (p.second).second << std::endl;
        }
      if (gemPadToEtaLimitsLong_.size())
        for(auto p : gemPadToEtaLimitsLong_) {
          std::cout << "pad "<< p.first << " min eta " << (p.second).first << " max eta " << (p.second).second << std::endl;
        }
    }
    
    auto randRoll(gemChamberLong->etaPartition(2));
    auto nStrips(keyLayerGeometry->numberOfStrips());
    for (float i = 0; i< nStrips; i = i+0.5){
      const LocalPoint lpCSC(keyLayerGeometry->topology()->localPosition(i));
      const GlobalPoint gp(keyLayer->toGlobal(lpCSC));
      const LocalPoint lpGEM(randRoll->toLocal(gp));
      const int HS(i/0.5);
      const float pad(randRoll->pad(lpGEM));
      // HS are wrapped-around
      cscHsToGemPad_[nStrips*2-HS] = std::make_pair(std::floor(pad),std::ceil(pad));
    }
    debug = false;
    if (debug){
      std::cout << "detId " << csc_id << std::endl;
      for(auto p : cscHsToGemPad_) {
        std::cout << "CSC HS "<< p.first << " GEM Pad low " << (p.second).first << " GEM Pad high " << (p.second).second << std::endl;
      }
    }

    // pick any roll (from short or long superchamber)
    const int nGEMPads(randRoll->npads());
    for (int i = 0; i< nGEMPads; ++i){
      const LocalPoint lpGEM(randRoll->centreOfPad(i));
      const GlobalPoint gp(randRoll->toGlobal(lpGEM));
      const LocalPoint lpCSC(keyLayer->toLocal(gp));
      const float strip(keyLayerGeometry->strip(lpCSC));
      // HS are wrapped-around
      gemPadToCscHs_[i] = nStrips*2-(int) (strip - 0.25)/0.5;
    }
    debug = false;
    if (debug){
      std::cout << "detId " << csc_id << std::endl;
      for(auto p : gemPadToCscHs_) {
        std::cout << "GEM Pad "<< p.first << " CSC HS : " << p.second << std::endl;
      }
    }
    
    // build coincidence pads
    std::auto_ptr<GEMCSCPadDigiCollection> pCoPads(new GEMCSCPadDigiCollection());
    buildCoincidencePads(gemPads, *pCoPads);
    
    // retrieve pads and copads in a certain BX window for this CSC 
    padsShort_ = retrieveGEMPads(gemPads, gem_id_short);
    padsLong_ = retrieveGEMPads(gemPads, gem_id_long);
    coPadsShort_ = retrieveGEMPads(pCoPads.get(), gem_id_short, true);
    coPadsLong_ = retrieveGEMPads(pCoPads.get(), gem_id_long, true); 
  }

  return;

  int used_clct_mask[20];
  for (int c=0;c<20;++c) used_clct_mask[c]=0;

  // ALCT centric matching
  for (int bx_alct = 0; bx_alct < CSCAnodeLCTProcessor::MAX_ALCT_BINS; bx_alct++)
  {
    if (alct->bestALCT[bx_alct].isValid())
    {
      const int bx_clct_start(bx_alct - match_trig_window_size/2);
      const int bx_clct_stop(bx_alct + match_trig_window_size/2);
//       const int bx_gem_start(bx_alct - maxDeltaBXCoPad_);
//       const int bx_gem_stop(bx_alct + maxDeltaBXCoPad_);

      if (print_available_pads){ 
        std::cout << "========================================================================" << std::endl;
        std::cout << "ALCT-CLCT matching in ME2/1 chamber: " << cscChamber->id() << std::endl;
        std::cout << "------------------------------------------------------------------------" << std::endl;
        std::cout << "+++ Best ALCT Details: ";
        alct->bestALCT[bx_alct].print();
        std::cout << "+++ Second ALCT Details: ";
        alct->secondALCT[bx_alct].print();
        
        printGEMTriggerPads(bx_clct_start, bx_clct_stop, true);      
        printGEMTriggerPads(bx_clct_start, bx_clct_stop, true, true);      
        printGEMTriggerPads(bx_clct_start, bx_clct_stop, false);      
        printGEMTriggerPads(bx_clct_start, bx_clct_stop, false, true);      
        
        std::cout << "------------------------------------------------------------------------" << std::endl;
        std::cout << "Attempt ALCT-CLCT matching in ME2/1 in bx range: [" << bx_clct_start << "," << bx_clct_stop << "]" << std::endl;
      }
      
      // ALCT-to-CLCT matching
      int nSuccesFulMatches = 0;
      for (int bx_clct = bx_clct_start; bx_clct <= bx_clct_stop; bx_clct++)
      {
        if (bx_clct < 0 or bx_clct >= CSCCathodeLCTProcessor::MAX_CLCT_BINS) continue;
        if (drop_used_clcts and used_clct_mask[bx_clct]) continue;
        if (clct->bestCLCT[bx_clct].isValid())
        {
	  //	  const int quality(clct->bestCLCT[bx_clct].getQuality());
          if (print_available_pads) std::cout << "++Valid ME21 CLCT: " << clct->bestCLCT[bx_clct] << std::endl;

	  //	  if (runME21ILT_ and dropLowQualityCLCTsNoGEMs_ and quality < 4 and hasPads){
	  //             // pick the pad that corresponds 
	  //             auto matchingPads(matchingGEMPads(clct->bestCLCT[bx_clct], pads_[bx_clct], ME1B, false));
	  //             int nFound(matchingPads.size());
	  //             const bool clctInEdge(clct->bestCLCT[bx_clct].getKeyStrip() < 5 or clct->bestCLCT[bx_clct].getKeyStrip() > 124);
	  //             if (clctInEdge){
	  //               if (print_available_pads) std::cout << "\tInfo: low quality CLCT in CSC chamber edge, don't care about GEM pads" << std::endl;
	  // 	    }
	  //             else {
	  //               if (nFound != 0){
	  //                 if (print_available_pads) std::cout << "\tInfo: low quality CLCT with " << nFound << " matching GEM trigger pads" << std::endl;
	  //               }
	  //               else {
	  //                 if (print_available_pads) std::cout << "\tWarning: low quality CLCT without matching GEM trigger pad" << std::endl;
	  //                 continue;
	  //               }
	  //             }
	  //           }
          
          ++nSuccesFulMatches;
          //	    if (infoV > 1) LogTrace("CSCMotherboard")
	  //          int mbx = bx_clct-bx_clct_start;
          correlateLCTs(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
                        clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct]);
	  if (print_available_pads) {
	    std::cout << "Successful ALCT-CLCT match in ME21: bx_alct = " << bx_alct
		      << "; match window: [" << bx_clct_start << "; " << bx_clct_stop
		      << "]; bx_clct = " << bx_clct << std::endl;
	    std::cout << "+++ Best CLCT Details: ";
	    clct->bestCLCT[bx_clct].print();
	    std::cout << "+++ Second CLCT Details: ";
	    clct->secondCLCT[bx_clct].print();
	  }
          
//           if (allLCTs1b[bx_alct][mbx][0].isValid()) {
//             used_clct_mask[bx_clct] += 1;
//             if (match_earliest_clct_me11_only) break;
//           }
        }
      }
    }
  }
}
      /*
      // ALCT-to-GEM matching
      int nSuccesFulGEMMatches = 0;
      if (runME21ILT_ and nSuccesFulMatches==0 and buildLCTfromALCTandGEM_ME21_){
        if (print_available_pads) std::cout << "++No valid ALCT-CLCT matches in ME21" << std::endl;
        for (int bx_gem = bx_gem_start; bx_gem <= bx_gem_stop; bx_gem++) {
          if (not hasCoPads) continue;
          // find the best matching copad - first one 
          try {
            auto copads(matchingGEMPads(alct->bestALCT[bx_alct], coPads_[bx_gem],ME1B,true));             
            if (print_available_pads) std::cout << "\t++Number of matching GEM CoPads in BX " << bx_alct << " : "<< copads.size() << std::endl;
            ++nSuccesFulGEMMatches;            
            correlateLCTsGEM(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
                             *(copads.at(0)).second, allLCTs1b[bx_alct][0][0], allLCTs1b[bx_alct][0][1]);
            if (allLCTs1b[bx_alct][0][0].isValid()) {
              if (match_earliest_clct_me11_only) break;
            }
            if (print_available_pads) 
              std::cout << "Successful ALCT-GEM CoPad match in ME21: bx_alct = " << bx_alct << std::endl << std::endl;
          }
          catch (const std::out_of_range& oor) {
            if (print_available_pads) std::cout << "\t++No valid GEM CoPads in BX: " << bx_alct << std::endl;
            continue;
          }
          if (print_available_pads) 
            std::cout << "------------------------------------------------------------------------" << std::endl << std::endl;
        }
      }

      if (print_available_pads) {
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

        std::cout << "------------------------------------------------------------------------" << std::endl;
        std::cout << "Attempt ALCT-CLCT matching in ME2/1 in bx range: [" << bx_clct_start << "," << bx_clct_stop << "]" << std::endl;
      }
      */

      /*
	int bx_alct_matched = 0;
  for (int bx_clct = 0; bx_clct < CSCCathodeLCTProcessor::MAX_CLCT_BINS; bx_clct++) {
    if (clct->bestCLCT[bx_clct].isValid()) {
      bool is_matched = false;
      int bx_alct_start = bx_clct - match_trig_window_size/2;
      int bx_alct_stop  = bx_clct + match_trig_window_size/2;
      
      for (int bx_alct = bx_alct_start; bx_alct <= bx_alct_stop; bx_alct++) {
        if (bx_alct < 0 || bx_alct >= CSCAnodeLCTProcessor::MAX_ALCT_BINS) continue;
        // default: do not reuse ALCTs that were used with previous CLCTs
        if (drop_used_alcts && used_alct_mask[bx_alct]) continue;
        if (alct->bestALCT[bx_alct].isValid()) {
          if (infoV > 1) LogTrace("CSCMotherboardME21")
            << "Successful ALCT-CLCT match: bx_clct = " << bx_clct
            << "; match window: [" << bx_alct_start << "; " << bx_alct_stop
            << "]; bx_alct = " << bx_alct;
          correlateLCTs(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
                        clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct]);
          used_alct_mask[bx_alct] += 1;
          is_matched = true;
          bx_alct_matched = bx_alct;
          break;
        }
      }
      // No ALCT within the match time interval found: report CLCT-only LCT
      // (use dummy ALCTs).
      if (!is_matched) {
        if (infoV > 1) LogTrace("CSCMotherboardME21")
          << "Unsuccessful ALCT-CLCT match (CLCT only): bx_clct = "
          << bx_clct << "; match window: [" << bx_alct_start
          << "; " << bx_alct_stop << "]";
        correlateLCTs(alct->bestALCT[bx_clct], alct->secondALCT[bx_clct],
                      clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct]);
      }
    }
    // No valid CLCTs; attempt to make ALCT-only LCT.  Use only ALCTs
    // which have zeroth chance to be matched at later cathode times.
    // (I am not entirely sure this perfectly matches the firmware logic.)
    // Use dummy CLCTs.
    else {
      int bx_alct = bx_clct - match_trig_window_size/2;
      if (bx_alct >= 0 && bx_alct > bx_alct_matched) {
        if (alct->bestALCT[bx_alct].isValid()) {
          if (infoV > 1) LogTrace("CSCMotherboardME21")
            << "Unsuccessful ALCT-CLCT match (ALCT only): bx_alct = "
            << bx_alct;
          correlateLCTs(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
                        clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct]);
          }
      }
    }
  }

  if (infoV > 0) {
    for (int bx = 0; bx < MAX_LCT_BINS; bx++) {
      if (firstLCT[bx].isValid())
        LogDebug("CSCMotherboardME21") << firstLCT[bx];
      if (secondLCT[bx].isValid())
        LogDebug("CSCMotherboardME21") << secondLCT[bx];
    }
      */
      
/*
  
void CSCMotherboardME21::correlateLCTs(CSCALCTDigi bestALCT,
CSCALCTDigi secondALCT,
                                   CSCCLCTDigi bestCLCT,
                                   CSCCLCTDigi secondCLCT) {

  bool anodeBestValid     = bestALCT.isValid();
  bool anodeSecondValid   = secondALCT.isValid();
  bool cathodeBestValid   = bestCLCT.isValid();
  bool cathodeSecondValid = secondCLCT.isValid();

  if (anodeBestValid && !anodeSecondValid)     secondALCT = bestALCT;
  if (!anodeBestValid && anodeSecondValid)     bestALCT   = secondALCT;
  if (cathodeBestValid && !cathodeSecondValid) secondCLCT = bestCLCT;
  if (!cathodeBestValid && cathodeSecondValid) bestCLCT   = secondCLCT;

  // ALCT-CLCT matching conditions are defined by "trig_enable" configuration
  // parameters.
  if ((alct_trig_enable  && bestALCT.isValid()) ||
      (clct_trig_enable  && bestCLCT.isValid()) ||
      (match_trig_enable && bestALCT.isValid() && bestCLCT.isValid())) {
    CSCCorrelatedLCTDigi lct = constructLCTs(bestALCT, bestCLCT);
    int bx = lct.getBX();
    if (bx >= 0 && bx < MAX_LCT_BINS) {
      firstLCT[bx] = lct;
      firstLCT[bx].setTrknmb(1);
    }
    else {
      if (infoV > 0) edm::LogWarning("L1CSCTPEmulatorOutOfTimeLCT")
        << "+++ Bx of first LCT candidate, " << bx
        << ", is not within the allowed range, [0-" << MAX_LCT_BINS-1
        << "); skipping it... +++\n";
    }
  }

  if (((secondALCT != bestALCT) || (secondCLCT != bestCLCT)) &&
      ((alct_trig_enable  && secondALCT.isValid()) ||
       (clct_trig_enable  && secondCLCT.isValid()) ||
       (match_trig_enable && secondALCT.isValid() && secondCLCT.isValid()))) {
    CSCCorrelatedLCTDigi lct = constructLCTs(secondALCT, secondCLCT);
    int bx = lct.getBX();
    if (bx >= 0 && bx < MAX_LCT_BINS) {
      secondLCT[bx] = lct;
      secondLCT[bx].setTrknmb(2);
    }
    else {
      if (infoV > 0) edm::LogWarning("L1CSCTPEmulatorOutOfTimeLCT")
        << "+++ Bx of second LCT candidate, " << bx
        << ", is not within the allowed range, [0-" << MAX_LCT_BINS-1
        << "); skipping it... +++\n";
    }
  }
}

// This method calculates all the TMB words and then passes them to the
// constructor of correlated LCTs.
CSCCorrelatedLCTDigi CSCMotherboardME21::constructLCTs(const CSCALCTDigi& aLCT,
                                                   const CSCCLCTDigi& cLCT) {
  // CLCT pattern number
  unsigned int pattern = encodePattern(cLCT.getPattern(), cLCT.getStripType());

  // LCT quality number
  unsigned int quality = findQuality(aLCT, cLCT);

  // Bunch crossing: get it from cathode LCT if anode LCT is not there.
  int bx = aLCT.isValid() ? aLCT.getBX() : cLCT.getBX();

  // construct correlated LCT; temporarily assign track number of 0.
  int trknmb = 0;
  CSCCorrelatedLCTDigi thisLCT(trknmb, 1, quality, aLCT.getKeyWG(),
                               cLCT.getKeyStrip(), pattern, cLCT.getBend(),
                               bx, 0, 0, 0, theTrigChamber);
  return thisLCT;
}

// CLCT pattern number: encodes the pattern number itself and
// whether the pattern consists of half-strips or di-strips.
unsigned int CSCMotherboardME21::encodePattern(const int ptn,
                                           const int stripType) {
  const int kPatternBitWidth = 4;
  unsigned int pattern;

  if (!isTMB07) {
    // Cathode pattern number is a kPatternBitWidth-1 bit word.
    pattern = (abs(ptn) & ((1<<(kPatternBitWidth-1))-1));

    // The pattern has the MSB (4th bit in the default version) set if it
    // consists of half-strips.
    if (stripType) {
      pattern = pattern | (1<<(kPatternBitWidth-1));
    }
  }
  else {
    // In the TMB07 firmware, LCT pattern is just a 4-bit CLCT pattern.
    pattern = (abs(ptn) & ((1<<kPatternBitWidth)-1));
  }

  return pattern;
}
*/


void CSCMotherboardME21::buildCoincidencePads(const GEMCSCPadDigiCollection* out_pads, GEMCSCPadDigiCollection& out_co_pads)
{
  // build coincidences
  for (auto det_range = out_pads->begin(); det_range != out_pads->end(); ++det_range) {
    const GEMDetId& id = (*det_range).first;
    if (id.station() != 1) continue;
    
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
        
        // always use layer1 pad's BX as a copad's BX
        GEMCSCPadDigi co_pad_digi(p->pad(), p->bx());
        out_co_pads.insertDigi(id, co_pad_digi);
      }
    }
  }
}


std::map<int,std::pair<double,double> >
CSCMotherboardME21::createGEMPadLUT(bool isLong)
{
  std::map<int,std::pair<double,double> > result;

  const int st(isLong ? 3 : 2);
  // no distinction between even and odd in GE2/1
  auto chamber(gem_g->chamber(GEMDetId(1,1,st,1,1,0)));
  if (chamber==nullptr) return result;

  for(int i = 1; i<= chamber->nEtaPartitions(); ++i){
    auto roll(chamber->etaPartition(i));
    if (roll==nullptr) continue;
    const float half_striplength(roll->specs()->specificTopology().stripLength()/2.);
    const LocalPoint lp_top(0., half_striplength, 0.);
    const LocalPoint lp_bottom(0., -half_striplength, 0.);
    const GlobalPoint gp_top(roll->toGlobal(lp_top));
    const GlobalPoint gp_bottom(roll->toGlobal(lp_bottom));
    result[i] = std::make_pair(gp_top.eta(), gp_bottom.eta());
  }
  return result;
}

std::map<int, std::vector<std::pair<unsigned int, const GEMCSCPadDigi*> > >
CSCMotherboardME21::retrieveGEMPads(const GEMCSCPadDigiCollection* gemPads, unsigned id, bool iscopad)
{
  int deltaBX(iscopad ? maxDeltaBXCoPad_ : maxDeltaBXPad_);
  GEMPads result;

  auto superChamber(gem_g->superChamber(id));
  for (auto ch : superChamber->chambers()) {
    for (auto roll : ch->etaPartitions()) {
      GEMDetId roll_id(roll->id());
      auto pads_in_det = gemPads->get(roll_id);
      for (auto pad = pads_in_det.first; pad != pads_in_det.second; ++pad) {
        auto id_pad = std::make_pair(roll_id(), &(*pad));
        const int bx_shifted(lct_central_bx + pad->bx());
        for (int bx = bx_shifted - deltaBX;bx <= bx_shifted + deltaBX; ++bx) {
          if (iscopad){
            if(bx != lct_central_bx) continue;
            result[bx].push_back(id_pad);  
          }else{
            result[bx].push_back(id_pad);  
          }
        }
      }
    }
  }
  return result;
}


void CSCMotherboardME21::printGEMTriggerPads(int bx_start, int bx_stop, bool isShort, bool iscopad)
{
  // pads or copads?
  auto thePads(!iscopad ? (isShort ? padsShort_ : padsLong_) : (isShort ? coPadsShort_ : coPadsLong_));
  const bool hasPads(thePads.size()!=0);
  
  std::cout << "------------------------------------------------------------------------" << std::endl;
  bool first = true;
  for (int bx = bx_start; bx <= bx_stop; bx++) {
    // print only the pads for the central BX
    if (centralBXonlyGEM_ and bx!=lct_central_bx) continue;
    if (bx!=lct_central_bx and iscopad) continue;
    std::vector<std::pair<unsigned int, const GEMCSCPadDigi*> > in_pads = thePads[bx];
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
        std::cout << "\tdetId " << pad.first << " " << roll_id << ", pad = " << pad.second->pad() << ", BX = " << pad.second->bx() + 6;
//         if (isPadInOverlap(roll_id.roll())) std::cout << " (in overlap)" << std::endl;
//         else std::cout << std::endl;
      }
    }
    else
      break;
  }
}
