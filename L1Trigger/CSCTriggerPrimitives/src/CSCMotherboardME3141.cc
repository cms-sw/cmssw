#include <L1Trigger/CSCTriggerPrimitives/src/CSCMotherboardME3141.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/RPCGeometry/interface/RPCRollSpecs.h>

const double CSCMotherboardME3141::lut_wg_me31_eta_odd[96][2] = {
{ 0,2.421},{ 1,2.415},{ 2,2.406},{ 3,2.397},{ 4,2.388},{ 5,2.379},{ 6,2.371},{ 7,2.362},
{ 8,2.353},{ 9,2.345},{10,2.336},{11,2.328},{12,2.319},{13,2.311},{14,2.303},{15,2.295},
{16,2.287},{17,2.279},{18,2.271},{19,2.263},{20,2.255},{21,2.248},{22,2.240},{23,2.232},
{24,2.225},{25,2.217},{26,2.210},{27,2.203},{28,2.195},{29,2.188},{30,2.181},{31,2.174},
{32,2.169},{33,2.157},{34,2.151},{35,2.142},{36,2.134},{37,2.126},{38,2.118},{39,2.110},
{40,2.102},{41,2.094},{42,2.087},{43,2.079},{44,2.071},{45,2.064},{46,2.056},{47,2.049},
{48,2.041},{49,2.034},{50,2.027},{51,2.019},{52,2.012},{53,2.005},{54,1.998},{55,1.991},
{56,1.984},{57,1.977},{58,1.970},{59,1.964},{60,1.957},{61,1.950},{62,1.944},{63,1.937},
{64,1.932},{65,1.922},{66,1.917},{67,1.911},{68,1.905},{69,1.898},{70,1.892},{71,1.886},
{72,1.880},{73,1.874},{74,1.868},{75,1.861},{76,1.855},{77,1.850},{78,1.844},{79,1.838},
{80,1.832},{81,1.826},{82,1.820},{83,1.815},{84,1.809},{85,1.803},{86,1.798},{87,1.792},
{88,1.787},{89,1.781},{90,1.776},{91,1.770},{92,1.765},{93,1.759},{94,1.754},{95,1.749},
};

const double CSCMotherboardME3141::lut_wg_me31_eta_even[96][2] = {
{ 0,2.447},{ 1,2.441},{ 2,2.432},{ 3,2.423},{ 4,2.414},{ 5,2.405},{ 6,2.396},{ 7,2.388},
{ 8,2.379},{ 9,2.371},{10,2.362},{11,2.354},{12,2.345},{13,2.337},{14,2.329},{15,2.321},
{16,2.313},{17,2.305},{18,2.297},{19,2.289},{20,2.281},{21,2.273},{22,2.266},{23,2.258},
{24,2.251},{25,2.243},{26,2.236},{27,2.228},{28,2.221},{29,2.214},{30,2.207},{31,2.200},
{32,2.195},{33,2.183},{34,2.176},{35,2.168},{36,2.160},{37,2.152},{38,2.144},{39,2.136},
{40,2.128},{41,2.120},{42,2.112},{43,2.104},{44,2.097},{45,2.089},{46,2.082},{47,2.074},
{48,2.067},{49,2.059},{50,2.052},{51,2.045},{52,2.038},{53,2.031},{54,2.023},{55,2.016},
{56,2.009},{57,2.003},{58,1.996},{59,1.989},{60,1.982},{61,1.975},{62,1.969},{63,1.962},
{64,1.957},{65,1.948},{66,1.943},{67,1.936},{68,1.930},{69,1.924},{70,1.917},{71,1.911},
{72,1.905},{73,1.899},{74,1.893},{75,1.887},{76,1.881},{77,1.875},{78,1.869},{79,1.863},
{80,1.857},{81,1.851},{82,1.845},{83,1.840},{84,1.834},{85,1.828},{86,1.823},{87,1.817},
{88,1.811},{89,1.806},{90,1.800},{91,1.795},{92,1.790},{93,1.784},{94,1.779},{95,1.774},
};

const double CSCMotherboardME3141::lut_wg_me41_eta_odd[96][2] = {
{ 0,2.399},{ 1,2.394},{ 2,2.386},{ 3,2.378},{ 4,2.370},{ 5,2.362},{ 6,2.354},{ 7,2.346},
{ 8,2.339},{ 9,2.331},{10,2.323},{11,2.316},{12,2.308},{13,2.301},{14,2.293},{15,2.286},
{16,2.279},{17,2.272},{18,2.264},{19,2.257},{20,2.250},{21,2.243},{22,2.236},{23,2.229},
{24,2.223},{25,2.216},{26,2.209},{27,2.202},{28,2.196},{29,2.189},{30,2.183},{31,2.176},
{32,2.172},{33,2.161},{34,2.157},{35,2.150},{36,2.144},{37,2.138},{38,2.132},{39,2.126},
{40,2.119},{41,2.113},{42,2.107},{43,2.101},{44,2.095},{45,2.089},{46,2.083},{47,2.078},
{48,2.072},{49,2.066},{50,2.060},{51,2.055},{52,2.049},{53,2.043},{54,2.038},{55,2.032},
{56,2.027},{57,2.021},{58,2.016},{59,2.010},{60,2.005},{61,1.999},{62,1.994},{63,1.989},
{64,1.985},{65,1.977},{66,1.973},{67,1.968},{68,1.963},{69,1.958},{70,1.953},{71,1.947},
{72,1.942},{73,1.937},{74,1.932},{75,1.928},{76,1.923},{77,1.918},{78,1.913},{79,1.908},
{80,1.903},{81,1.898},{82,1.894},{83,1.889},{84,1.884},{85,1.879},{86,1.875},{87,1.870},
{88,1.866},{89,1.861},{90,1.856},{91,1.852},{92,1.847},{93,1.843},{94,1.838},{95,1.834},
};

const double CSCMotherboardME3141::lut_wg_me41_eta_even[96][2] = {
{ 0,2.423},{ 1,2.418},{ 2,2.410},{ 3,2.402},{ 4,2.394},{ 5,2.386},{ 6,2.378},{ 7,2.370},
{ 8,2.362},{ 9,2.355},{10,2.347},{11,2.339},{12,2.332},{13,2.324},{14,2.317},{15,2.310},
{16,2.302},{17,2.295},{18,2.288},{19,2.281},{20,2.274},{21,2.267},{22,2.260},{23,2.253},
{24,2.246},{25,2.239},{26,2.233},{27,2.226},{28,2.219},{29,2.213},{30,2.206},{31,2.199},
{32,2.195},{33,2.185},{34,2.180},{35,2.174},{36,2.168},{37,2.161},{38,2.155},{39,2.149},
{40,2.143},{41,2.137},{42,2.131},{43,2.125},{44,2.119},{45,2.113},{46,2.107},{47,2.101},
{48,2.095},{49,2.089},{50,2.084},{51,2.078},{52,2.072},{53,2.067},{54,2.061},{55,2.055},
{56,2.050},{57,2.044},{58,2.039},{59,2.033},{60,2.028},{61,2.023},{62,2.017},{63,2.012},
{64,2.008},{65,2.000},{66,1.996},{67,1.991},{68,1.986},{69,1.981},{70,1.976},{71,1.971},
{72,1.966},{73,1.961},{74,1.956},{75,1.951},{76,1.946},{77,1.941},{78,1.936},{79,1.931},
{80,1.926},{81,1.921},{82,1.917},{83,1.912},{84,1.907},{85,1.902},{86,1.898},{87,1.893},
{88,1.889},{89,1.884},{90,1.879},{91,1.875},{92,1.870},{93,1.866},{94,1.861},{95,1.857},
};

CSCMotherboardME3141::CSCMotherboardME3141(unsigned endcap, unsigned station,
                               unsigned sector, unsigned subsector,
                               unsigned chamber,
                               const edm::ParameterSet& conf) :
  CSCMotherboard(endcap, station, sector, subsector, chamber, conf)
{
  edm::ParameterSet commonParams = conf.getParameter<edm::ParameterSet>("commonParam");
  
  if (!isSLHC) edm::LogError("L1CSCTPEmulatorConfigError")
    << "+++ Upgrade CSCMotherboardME3141 constructed while isSLHC is not set! +++\n";
  
  edm::ParameterSet alctParams = conf.getParameter<edm::ParameterSet>("alctSLHC");
  edm::ParameterSet clctParams = conf.getParameter<edm::ParameterSet>("clctSLHC");
  edm::ParameterSet tmbParams = conf.getParameter<edm::ParameterSet>("tmbSLHC");
  edm::ParameterSet me3141tmbParams = tmbParams.getParameter<edm::ParameterSet>("me3141ILT");

  // central bx for LCT is 6 for simulation
  lct_central_bx = tmbParams.getUntrackedParameter<int>("lctCentralBX", 6);

  // whether to not reuse CLCTs that were used by previous matching ALCTs
  // in ALCT-to-CLCT algorithm
  drop_used_clcts = me3141tmbParams.getUntrackedParameter<bool>("tmbDropUsedClcts",true);

  //----------------------------------------------------------------------------------------//

  //       R P C  -  C S C   I N T E G R A T E D   L O C A L   A L G O R I T H M

  //----------------------------------------------------------------------------------------//

  // masterswitch
  runME3141ILT_ = me3141tmbParams.getUntrackedParameter<bool>("runME3141ILT",false);

  // debug rpc matching
  debugRPCMatching_ = tmbParams.getUntrackedParameter<bool>("debugRPCMatching", false);

  // deltas used to match to RPC pads
  maxDeltaBXRPC_ = tmbParams.getUntrackedParameter<int>("maxDeltaBXRPC",0);
  maxDeltaRollRPC_ = tmbParams.getUntrackedParameter<int>("maxDeltaRollRPC",0);
  maxDeltaStripRPC_ = tmbParams.getUntrackedParameter<int>("maxDeltaStripRPC",1);

  // drop low quality stubs if they don't have RPCs
  dropLowQualityCLCTsNoRPC_ = tmbParams.getUntrackedParameter<bool>("dropLowQualityCLCTsNoRPCs",false);
  dropLowQualityALCTsNoRPCs_ = tmbParams.getUntrackedParameter<bool>("dropLowQualityALCTsNoRPCs",false);
}

CSCMotherboardME3141::~CSCMotherboardME3141() 
{
}


void
CSCMotherboardME3141::run(const CSCWireDigiCollection* wiredc,
			  const CSCComparatorDigiCollection* compdc,
			  const RPCDigiCollection* rpcDigis) 
{
  clear();

  if (!( alct and clct and runME3141ILT_))
  {
    if (infoV >= 0) edm::LogError("L1CSCTPEmulatorSetupError")
      << "+++ run() called for non-existing ALCT/CLCT processor! +++ \n";
    return;
  }

  alct->run(wiredc); // run anodeLCT
  clct->run(compdc); // run cathodeLCT

  bool rpcGeometryAvailable(false);
  if (rpc_g != nullptr) {
    if (infoV >= 0) edm::LogInfo("L1CSCTPEmulatorSetupInfo")
      << "+++ run() called for RPC-CSC integrated trigger! +++ \n";
    rpcGeometryAvailable = true;
  }

  // retrieve CSCChamber geometry                                                                                                                                       
  CSCTriggerGeomManager* geo_manager(CSCTriggerGeometry::get());
  const CSCChamber* cscChamber(geo_manager->chamber(theEndcap, theStation, theSector, theSubsector, theTrigChamber));
  const CSCDetId csc_id(cscChamber->id());

  if (runME3141ILT_){
    
    // check for RE3/1-RE4/1 geometry
    if ((not rpcGeometryAvailable) or (rpcGeometryAvailable and not hasRE31andRE41())) {
      if (infoV >= 0) edm::LogInfo("L1CSCTPEmulatorSetupError")
        << "+++ run() called for RPC-CSC integrated trigger without valid RPC geometry! +++ \n";
      return;
    }
    // trigger geometry
    const CSCLayer* keyLayer(cscChamber->layer(3));
    const CSCLayerGeometry* keyLayerGeometry(keyLayer->geometry());
    const int region((theEndcap == 1) ? 1: -1);
    const bool isEven(csc_id%2==0);
    const RPCDetId rpc_id(region,1,theStation,theSector,1,theTrigChamber,0);
    const RPCChamber* rpcChamber(rpc_g->chamber(rpc_id));
    
    // LUT<roll,<etaMin,etaMax> >    
    rpcRollToEtaLimits_ = createRPCRollLUT(rpc_id);
    
    bool debug(false);
    if (debug){
      if (rpcRollToEtaLimits_.size()) {
        for(auto p : rpcRollToEtaLimits_) {
          std::cout << "roll "<< p.first << " min eta " << (p.second).first << " max eta " << (p.second).second << std::endl;
        }
      }
    }

    // loop on all wiregroups to create a LUT <WG,roll>
    const int numberOfWG(keyLayerGeometry->numberOfWireGroups());
    for (int i = 0; i< numberOfWG; ++i){
      auto eta(theStation==3 ? 
               (isEven ? lut_wg_me31_eta_even[i][1] : lut_wg_me31_eta_odd[i][1]) : 
               (isEven ? lut_wg_me41_eta_even[i][1] : lut_wg_me41_eta_odd[i][1]));
      cscWgToRpcRoll_[i] = assignRPCRoll(eta);
    }
    debug = false;
    if (debug){
      for(auto p : cscWgToRpcRoll_) {
        auto eta(theStation==3 ? 
                 (isEven ? lut_wg_me31_eta_even[p.first][1] : lut_wg_me31_eta_odd[p.first][1]) : 
                 (isEven ? lut_wg_me41_eta_even[p.first][1] : lut_wg_me41_eta_odd[p.first][1]));
        
        std::cout << "WG "<< p.first << " RPC roll " << p.second << " " 
                  << rpcRollToEtaLimits_[p.second].first << " " 
                  << rpcRollToEtaLimits_[p.second].second << " " << eta << std::endl;
      }
    }

    // pick any roll
    auto randRoll(rpcChamber->roll(2));

    auto nStrips(keyLayerGeometry->numberOfStrips());
    for (float i = 0; i< nStrips; i = i+0.5){
      const LocalPoint lpCSC(keyLayerGeometry->topology()->localPosition(i));
      const GlobalPoint gp(keyLayer->toGlobal(lpCSC));
      const LocalPoint lpRPC(randRoll->toLocal(gp));
      const int HS(i/0.5);
      const bool edge(HS < 4 or HS > 93);
      const float strip(edge ? -99 : randRoll->strip(lpRPC));
      // HS are wrapped-around
      cscHsToRpcStrip_[nStrips*2-HS] = std::make_pair(std::floor(strip),std::ceil(strip));
    }
    debug = true;
    if (debug){
      std::cout << "detId " << csc_id << std::endl;
      std::cout << "CSCHSToRPCStrip LUT in" << std::endl;
      for(auto p : cscHsToRpcStrip_) {
        std::cout << "CSC HS "<< p.first << " RPC Strip low " << (p.second).first << " RPC Strip high " << (p.second).second << std::endl;
      }
    }

    const int nRPCStrips(randRoll->nstrips());
    for (int i = 0; i< nRPCStrips; ++i){
      const LocalPoint lpRPC(randRoll->centreOfStrip(i));
      const GlobalPoint gp(randRoll->toGlobal(lpRPC));
      const LocalPoint lpCSC(keyLayer->toLocal(gp));
      const float strip(keyLayerGeometry->strip(lpCSC));
      // HS are wrapped-around
      rpcStripToCscHs_[i] = nStrips*2-(int) (strip - 0.25)/0.5;
    }
    debug = false;
    if (debug){
      std::cout << "detId " << csc_id << std::endl;
      std::cout << "RPCStripToCSCHs LUT" << std::endl;
      for(auto p : rpcStripToCscHs_) {
        std::cout << "RPC Strip "<< p.first << " CSC HS: " << p.second << std::endl;
      }
    }

    rpcDigis_.clear();
    retrieveRPCDigis(rpcDigis, rpc_id);
  }
  
  //  const bool hasRPCDigis(rpcDigis_.size()!=0);
  
  int used_alct_mask[20];
  for (int a=0;a<20;++a) used_alct_mask[a]=0;
  
  int bx_alct_matched = 0; // bx of last matched ALCT
  for (int bx_clct = 0; bx_clct < CSCCathodeLCTProcessor::MAX_CLCT_BINS; bx_clct++) {
    // There should be at least one valid ALCT or CLCT for a
    // correlated LCT to be formed.  Decision on whether to reject
    // non-complete LCTs (and if yes of which type) is made further
    // upstream.
    if (clct->bestCLCT[bx_clct].isValid()) {
      // Look for ALCTs within the match-time window.  The window is
      // centered at the CLCT bx; therefore, we make an assumption
      // that anode and cathode hits are perfectly synchronized.  This
      // is always true for MC, but only an approximation when the
      // data is analyzed (which works fairly good as long as wide
      // windows are used).  To get rid of this assumption, one would
      // need to access "full BX" words, which are not readily
      // available.
      bool is_matched = false;
      int bx_alct_start = bx_clct - match_trig_window_size/2;
      int bx_alct_stop  = bx_clct + match_trig_window_size/2;
      // Empirical correction to match 2009 collision data (firmware change?)
      // (but don't do it for SLHC case, assume it would not be there)
      if (!isSLHC) bx_alct_stop += match_trig_window_size%2;
      
      for (int bx_alct = bx_alct_start; bx_alct <= bx_alct_stop; bx_alct++) {
	if (bx_alct < 0 || bx_alct >= CSCAnodeLCTProcessor::MAX_ALCT_BINS)
	  continue;
	// default: do not reuse ALCTs that were used with previous CLCTs
	if (drop_used_alcts && used_alct_mask[bx_alct]) continue;
	if (alct->bestALCT[bx_alct].isValid()) {
	  if (infoV > 1) LogTrace("CSCMotherboard")
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
	if (infoV > 1) LogTrace("CSCMotherboard")
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
	  if (infoV > 1) LogTrace("CSCMotherboard")
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
	LogDebug("CSCMotherboard") << firstLCT[bx];
      if (secondLCT[bx].isValid())
	LogDebug("CSCMotherboard") << secondLCT[bx];
    }
  }
}

// check that the RE31 and RE41 chambers are really there
bool CSCMotherboardME3141::hasRE31andRE41()
{
  auto aRE31(rpc_g->chamber(RPCDetId(1,1,3,2,1,1,0)));
  auto aRE41(rpc_g->chamber(RPCDetId(-1,1,4,3,1,2,0)));  
  return aRE31 and aRE41;
}


std::map<int,std::pair<double,double> > CSCMotherboardME3141::createRPCRollLUT(RPCDetId id)
{
  std::map<int,std::pair<double,double> > result;

  auto chamber(rpc_g->chamber(id));
  if (chamber==nullptr) return result;

  for(int i = 1; i<= chamber->nrolls(); ++i){
    auto roll(chamber->roll(i));
    if (roll==nullptr) continue;
    
    const float half_striplength(roll->specs()->specificTopology().stripLength()/2.);
    const LocalPoint lp_top(0., half_striplength, 0.);
    const LocalPoint lp_bottom(0., -half_striplength, 0.);
    const GlobalPoint gp_top(roll->toGlobal(lp_top));
    const GlobalPoint gp_bottom(roll->toGlobal(lp_bottom));
    result[i] = std::make_pair(std::abs(gp_top.eta()), std::abs(gp_bottom.eta()));
  }
  return result;
}


int CSCMotherboardME3141::assignRPCRoll(double eta)
{
  int result = -99;
  for(auto p : rpcRollToEtaLimits_) {
    const float minEta((p.second).first);
    const float maxEta((p.second).second);
    if (minEta <= eta and eta <= maxEta) {
      result = p.first;
      break;
    }
  }
  return result;
}


void CSCMotherboardME3141::retrieveRPCDigis(const RPCDigiCollection* rpcDigis, unsigned id)
{
  auto chamber(rpc_g->chamber(RPCDetId(id)));
  for (auto roll : chamber->rolls()) {
    RPCDetId roll_id(roll->id());
    auto digis_in_det = rpcDigis->get(roll_id);
    for (auto digi = digis_in_det.first; digi != digis_in_det.second; ++digi) {
      auto id_digi = std::make_pair(roll_id(), &(*digi));
      const int bx_shifted(lct_central_bx + digi->bx());
      for (int bx = bx_shifted - maxDeltaBXRPC_;bx <= bx_shifted + maxDeltaBXRPC_; ++bx) {
        rpcDigis_[bx].push_back(id_digi);  
      }
    }
  }
}
