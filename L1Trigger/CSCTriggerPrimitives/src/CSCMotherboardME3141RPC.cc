#include "L1Trigger/CSCTriggerPrimitives/src/CSCMotherboardME3141RPC.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "boost/container/flat_set.hpp"

const double CSCMotherboardME3141RPC::lut_wg_me31_eta_odd[96][2] = {
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

const double CSCMotherboardME3141RPC::lut_wg_me31_eta_even[96][2] = {
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

const double CSCMotherboardME3141RPC::lut_wg_me41_eta_odd[96][2] = {
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

const double CSCMotherboardME3141RPC::lut_wg_me41_eta_even[96][2] = {
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

// LUT with bending angles of the RPC-CSC high efficiency patterns (98%)
// 1st index: pt value = {5,10,15,20,30,40}
// 2nd index: bending angle for odd numbered chambers
// 3rd index: bending angle for even numbered chambers
const double CSCMotherboardME3141RPC::lut_pt_vs_dphi_rpccsc_me31[8][3] = {
  {3.,  0.02203511, 0.00930056},
  {5.,  0.02203511, 0.00930056},
  {7 ,  0.0182579 , 0.00790009},
  {10., 0.01066000, 0.00483286},
  {15., 0.00722795, 0.00363230},
  {20., 0.00562598, 0.00304878},
  {30., 0.00416544, 0.00253782},
  {40., 0.00342827, 0.00230833} };

const double CSCMotherboardME3141RPC::lut_pt_vs_dphi_rpccsc_me41[8][3] = {
  {3.,  0.02203511, 0.00930056},
  {5.,  0.02203511, 0.00930056},
  {7 ,  0.0182579 , 0.00790009},
  {10., 0.01066000, 0.00483286},
  {15., 0.00722795, 0.00363230},
  {20., 0.00562598, 0.00304878},
  {30., 0.00416544, 0.00253782},
  {40., 0.00342827, 0.00230833} };

CSCMotherboardME3141RPC::CSCMotherboardME3141RPC(unsigned endcap, unsigned station,
                               unsigned sector, unsigned subsector,
                               unsigned chamber,
                               const edm::ParameterSet& conf) :
  CSCMotherboard(endcap, station, sector, subsector, chamber, conf)
{
  const edm::ParameterSet commonParams(conf.getParameter<edm::ParameterSet>("commonParam"));
  runME3141ILT_ = commonParams.getParameter<bool>("runME3141ILT");

  if (!isSLHC) edm::LogError("L1CSCTPEmulatorConfigError")
    << "+++ Upgrade CSCMotherboardME3141RPC constructed while isSLHC is not set! +++\n";

  const edm::ParameterSet me3141tmbParams(conf.getParameter<edm::ParameterSet>("me3141tmbSLHCRPC"));

  // whether to not reuse CLCTs that were used by previous matching ALCTs
  // in ALCT-to-CLCT algorithm
  drop_used_clcts = me3141tmbParams.getParameter<bool>("tmbDropUsedClcts");

  match_earliest_clct_me3141_only = me3141tmbParams.getParameter<bool>("matchEarliestClctME3141Only");

  tmb_cross_bx_algo = me3141tmbParams.getParameter<unsigned int>("tmbCrossBxAlgorithm");

  // maximum lcts per BX in ME2
  max_me3141_lcts = me3141tmbParams.getParameter<unsigned int>("maxME3141LCTs");

  pref[0] = match_trig_window_size/2;
  for (unsigned int m=2; m<match_trig_window_size; m+=2)
  {
    pref[m-1] = pref[0] - m/2;
    pref[m]   = pref[0] + m/2;
  }

  //----------------------------------------------------------------------------------------//

  //       R P C  -  C S C   I N T E G R A T E D   L O C A L   A L G O R I T H M

  //----------------------------------------------------------------------------------------//

  // debug
  debug_luts_ = me3141tmbParams.getParameter<bool>("debugLUTs");
  debug_rpc_matching_ = me3141tmbParams.getParameter<bool>("debugMatching");

  // deltas used to match to RPC digis
  maxDeltaBXRPC_ = me3141tmbParams.getParameter<int>("maxDeltaBXRPC");
  maxDeltaStripRPCOdd_ = me3141tmbParams.getParameter<int>("maxDeltaStripRPCOdd");
  maxDeltaStripRPCEven_ = me3141tmbParams.getParameter<int>("maxDeltaStripRPCEven");
  maxDeltaWg_ = me3141tmbParams.getParameter<int>("maxDeltaWg");

  // use "old" or "new" dataformat for integrated LCTs?
  useOldLCTDataFormat_ = me3141tmbParams.getParameter<bool>("useOldLCTDataFormat");

  // drop low quality stubs if they don't have RPCs
  dropLowQualityCLCTsNoRPCs_ = me3141tmbParams.getParameter<bool>("dropLowQualityCLCTsNoRPCs");

  // build LCT from CLCT and RPC
  buildLCTfromALCTandRPC_ = me3141tmbParams.getParameter<bool>("buildLCTfromALCTandRPC");
  buildLCTfromCLCTandRPC_ = me3141tmbParams.getParameter<bool>("buildLCTfromCLCTandRPC");
  buildLCTfromLowQstubandRPC_ = me3141tmbParams.getParameter<bool>("buildLCTfromLowQstubandRPC");

  // promote ALCT-RPC pattern
  promoteALCTRPCpattern_ = me3141tmbParams.getParameter<bool>("promoteALCTRPCpattern");

  // promote ALCT-CLCT-RPC quality
  promoteALCTRPCquality_ = me3141tmbParams.getParameter<bool>("promoteALCTRPCquality");
  promoteCLCTRPCquality_ = me3141tmbParams.getParameter<bool>("promoteCLCTRPCquality");
}

CSCMotherboardME3141RPC::~CSCMotherboardME3141RPC()
{
}

void CSCMotherboardME3141RPC::clear()
{
  CSCMotherboard::clear();

  for (int bx = 0; bx < MAX_LCT_BINS; bx++)
    for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
      for (int i=0;i<2;i++)
        allLCTs[bx][mbx][i].clear();

  rpcRollToEtaLimits_.clear();
  cscWgToRpcRoll_.clear();
  rpcStripToCscHs_.clear();
  cscHsToRpcStrip_.clear();
  rpcDigis_.clear();
}

void
CSCMotherboardME3141RPC::run(const CSCWireDigiCollection* wiredc,
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

  alctV = alct->run(wiredc); // run anodeLCT
  clctV = clct->run(compdc); // run cathodeLCT

  bool rpcGeometryAvailable(false);
  if (rpc_g != nullptr) {
    if (infoV >= 0) edm::LogInfo("L1CSCTPEmulatorSetupInfo")
      << "+++ run() called for RPC-CSC integrated trigger! +++ \n";
    rpcGeometryAvailable = true;
  }
  const bool hasCorrectRPCGeometry((not rpcGeometryAvailable) or (rpcGeometryAvailable and not hasRE31andRE41()));

  // retrieve CSCChamber geometry
  CSCTriggerGeomManager* geo_manager(CSCTriggerGeometry::get());
  const CSCChamber* cscChamber(geo_manager->chamber(theEndcap, theStation, theSector, theSubsector, theTrigChamber));
  const CSCDetId csc_id(cscChamber->id());

  // trigger geometry
  const CSCLayer* keyLayer(cscChamber->layer(3));
  const CSCLayerGeometry* keyLayerGeometry(keyLayer->geometry());
  const int region((theEndcap == 1) ? 1: -1);
  const bool isEven(csc_id.chamber()%2==0);
  const int csc_trig_sect(CSCTriggerNumbering::triggerSectorFromLabels(csc_id));
  const int csc_trig_id( CSCTriggerNumbering::triggerCscIdFromLabels(csc_id));
  const int csc_trig_chid((3*(csc_trig_sect-1)+csc_trig_id)%18 +1);
  const int rpc_trig_sect((csc_trig_chid-1)/3+1);
  const int rpc_trig_subsect((csc_trig_chid-1)%3+1);
  const RPCDetId rpc_id(region,1,theStation,rpc_trig_sect,1,rpc_trig_subsect,0);
  const RPCChamber* rpcChamber(rpc_g->chamber(rpc_id));

  if (runME3141ILT_){

    // check for RE3/1-RE4/1 geometry
    if (hasCorrectRPCGeometry) {
      if (infoV >= 0) edm::LogInfo("L1CSCTPEmulatorSetupError")
        << "+++ run() called for RPC-CSC integrated trigger without valid RPC geometry! +++ \n";
      return;
    }

    // LUT<roll,<etaMin,etaMax> >
    rpcRollToEtaLimits_ = createRPCRollLUT(rpc_id);

    if (debug_luts_){
      std::cout << "RPC det " <<rpc_id<<"  CSC det "<< csc_id << std::endl;
      for (const auto& p : rpcRollToEtaLimits_) {
        std::cout << "roll "<< p.first << " min eta " << (p.second).first << " max eta " << (p.second).second << std::endl;
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
    if (debug_luts_){
      for (const auto& p : cscWgToRpcRoll_) {
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
      const bool edge(HS < 5 or HS > 155);
      const float strip(edge ? -99 : randRoll->strip(lpRPC));
      // HS are wrapped-around
      cscHsToRpcStrip_[HS] = std::make_pair(std::floor(strip),std::ceil(strip));
    }
    if (debug_luts_){
      std::cout << "detId " << csc_id << std::endl;
      std::cout << "CSCHSToRPCStrip LUT in" << std::endl;
      for (const auto& p : cscHsToRpcStrip_) {
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
      rpcStripToCscHs_[i] = (int) (strip - 0.25)/0.5;
    }
    if (debug_luts_){
      std::cout << "detId " << csc_id << std::endl;
      std::cout << "RPCStripToCSCHs LUT" << std::endl;
      for (const auto& p : rpcStripToCscHs_) {
        std::cout << "RPC Strip "<< p.first << " CSC HS: " << p.second << std::endl;
      }
    }
    //select correct scenarios, even or odd
    maxDeltaStripRPC_ = (isEven ?  maxDeltaStripRPCEven_ :  maxDeltaStripRPCOdd_);

    rpcDigis_.clear();
    retrieveRPCDigis(rpcDigis, rpc_id.rawId());
  }

  const bool hasRPCDigis(!rpcDigis_.empty());

  int used_clct_mask[20];
  for (int c=0;c<20;++c) used_clct_mask[c]=0;

  // ALCT centric matching
  for (int bx_alct = 0; bx_alct < CSCAnodeLCTProcessor::MAX_ALCT_BINS; bx_alct++)
  {
    if (alct->bestALCT[bx_alct].isValid())
    {
      const int bx_clct_start(bx_alct - match_trig_window_size/2);
      const int bx_clct_stop(bx_alct + match_trig_window_size/2);
      if (debug_rpc_matching_){
        std::cout << "========================================================================" << std::endl;
        std::cout << "ALCT-CLCT matching in ME" << theStation << "/1 chamber: " << csc_id << std::endl;
        std::cout << "------------------------------------------------------------------------" << std::endl;
        std::cout << "+++ Best ALCT Details: " << alct->bestALCT[bx_alct] << std::endl;
        std::cout << "+++ Second ALCT Details: " << alct->secondALCT[bx_alct] << std::endl;
        std::cout << "------------------------------------------------------------------------" << std::endl;
        if (hasRPCDigis) std::cout << "RPC Chamber " << rpc_id << std::endl;
        if (hasRPCDigis) printRPCTriggerDigis(bx_clct_start, bx_clct_stop);

        std::cout << "------------------------------------------------------------------------" << std::endl;
        std::cout << "Attempt ALCT-CLCT matching in ME" << theStation << "/1 in bx range: [" << bx_clct_start << "," << bx_clct_stop << "]" << std::endl;
      }

      // low quality ALCT
      const bool lowQualityALCT(alct->bestALCT[bx_alct].getQuality() == 0);

      // ALCT-to-CLCT
      int nSuccesFulMatches = 0;
      for (int bx_clct = bx_clct_start; bx_clct <= bx_clct_stop; bx_clct++) {
        if (bx_clct < 0 or bx_clct >= CSCCathodeLCTProcessor::MAX_CLCT_BINS) continue;
        if (drop_used_clcts and used_clct_mask[bx_clct]) continue;
        if (clct->bestCLCT[bx_clct].isValid()) {

          // pick the digi that corresponds
          auto matchingDigis(matchingRPCDigis(clct->bestCLCT[bx_clct], alct->bestALCT[bx_alct], rpcDigis_[bx_alct], false));

          // clct quality
          const int quality(clct->bestCLCT[bx_clct].getQuality());
          // low quality ALCT or CLCT
          const bool lowQuality(quality<4 or lowQualityALCT);

          if (runME3141ILT_ and dropLowQualityCLCTsNoRPCs_ and lowQuality and hasRPCDigis){
            int nFound(!matchingDigis.empty());
            const bool clctInEdge(clct->bestCLCT[bx_clct].getKeyStrip() < 5 or clct->bestCLCT[bx_clct].getKeyStrip() > 155);
            if (clctInEdge){
              if (debug_rpc_matching_) std::cout << "\tInfo: low quality ALCT or CLCT in CSC chamber edge, don't care about RPC digis" << std::endl;
            }
            else {
              if (nFound != 0){
                if (debug_rpc_matching_) std::cout << "\tInfo: low quality ALCT or CLCT with " << nFound << " matching RPC trigger digis" << std::endl;
              }
              else {
                if (debug_rpc_matching_) std::cout << "\tWarning: low quality ALCT or CLCT without matching RPC trigger digi" << std::endl;
                continue;
              }
            }
          }

          int mbx = bx_clct-bx_clct_start;
          correlateLCTsRPC(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
                           clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct],
                           allLCTs[bx_alct][mbx][0], allLCTs[bx_alct][mbx][1], matchingDigis);
          ++nSuccesFulMatches;
          if (debug_rpc_matching_) {
            //            if (infoV > 1) LogTrace("CSCMotherboard")
            std::cout
              << "Successful ALCT-CLCT match: bx_clct = " << bx_clct
              << "; match window: [" << bx_clct_start << "; " << bx_clct_stop
	              << "]; bx_alct = " << bx_alct << std::endl;
            std::cout << "+++ Best CLCT Details: " << clct->secondCLCT[bx_clct]<< std::endl;
            std::cout << "+++ Second CLCT Details: " << clct->secondCLCT[bx_clct]<< std::endl;
          }
          if (allLCTs[bx_alct][mbx][0].isValid()) {
            used_clct_mask[bx_clct] += 1;
            if (match_earliest_clct_me3141_only) break;
          }
        }
      }
      // ALCT-RPC digi matching
      int nSuccesFulRPCMatches = 0;
      if (runME3141ILT_ and nSuccesFulMatches==0 and buildLCTfromALCTandRPC_ and hasRPCDigis){
        if (debug_rpc_matching_) std::cout << "++No valid ALCT-CLCT matches in ME"<<theStation<<"1" << std::endl;
        for (int bx_rpc = bx_clct_start; bx_rpc <= bx_clct_stop; bx_rpc++) {
          if (lowQualityALCT and !buildLCTfromLowQstubandRPC_) continue; // build lct from low-Q ALCTs and rpc if para is set true
          if (not hasRPCDigis) continue;

          // find the best matching copad - first one
          auto digis(matchingRPCDigis(alct->bestALCT[bx_alct], rpcDigis_[bx_rpc], true));
          if (debug_rpc_matching_) std::cout << "\t++Number of matching RPC Digis in BX " << bx_alct << " : "<< digis.size() << std::endl;
          if (digis.empty()) continue;

          correlateLCTsRPC(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
                           digis.at(0).second, allLCTs[bx_alct][0][0], allLCTs[bx_alct][0][1]);
          if (allLCTs[bx_alct][0][0].isValid()) {
            ++nSuccesFulRPCMatches;
            if (match_earliest_clct_me3141_only) break;
          }
          if (debug_rpc_matching_) {
            std::cout << "Successful ALCT-RPC digi match in ME"<<theStation<<"1: bx_alct = " << bx_alct << std::endl << std::endl;
            std::cout << "------------------------------------------------------------------------" << std::endl << std::endl;
          }
        }
      }
    }
    else{
      auto digis(rpcDigis_[bx_alct]);
      if (runME3141ILT_ and !digis.empty() and buildLCTfromCLCTandRPC_) {
        //const int bx_clct_start(bx_alct - match_trig_window_size/2);
        //const int bx_clct_stop(bx_alct + match_trig_window_size/2);
        // RPC-to-CLCT
        int nSuccesFulMatches = 0;
       // for (int bx_clct = bx_clct_start; bx_clct <= bx_clct_stop; bx_clct++)
       // {
        //  if (bx_clct < 0 or bx_clct >= CSCCathodeLCTProcessor::MAX_CLCT_BINS) continue;
          if (drop_used_clcts and used_clct_mask[bx_alct]) continue;
          if (clct->bestCLCT[bx_alct].isValid())
          {
            if (debug_rpc_matching_){
              std::cout << "========================================================================" << std::endl;
              std::cout << "RPC-CLCT matching in ME" << theStation << "/1 chamber: " << cscChamber->id() << " in bx: "<<bx_alct<< std::endl;
              std::cout << "------------------------------------------------------------------------" << std::endl;
            }
            const int quality(clct->bestCLCT[bx_alct].getQuality());
            // we also use low-Q stubs for the time being
            if (quality < 4 and !buildLCTfromLowQstubandRPC_) continue;

            ++nSuccesFulMatches;

            int mbx = std::abs(clct->bestCLCT[bx_alct].getBX()-bx_alct);
            int bx_rpc = lct_central_bx;
            correlateLCTsRPC(clct->bestCLCT[bx_alct], clct->secondCLCT[bx_alct], digis[0].second, RPCDetId(digis[0].first).roll(),
                             allLCTs[bx_rpc][mbx][0], allLCTs[bx_rpc][mbx][1]);
            if (debug_rpc_matching_) {
              //	    if (infoV > 1) LogTrace("CSCMotherboard")
              std::cout << "Successful RPC-CLCT match in ME"<<theStation<<"/1: bx_alct = " << bx_alct
                        << std::endl;
              std::cout << "+++ Best CLCT Details: "<< clct->bestCLCT[bx_alct]<< std::endl;
              std::cout << "+++ Second CLCT Details: " << clct->secondCLCT[bx_alct]<< std::endl;
            }
            if (allLCTs[bx_rpc][mbx][0].isValid()) {
              used_clct_mask[bx_alct] += 1;
              if (match_earliest_clct_me3141_only) break;
            }
          }
      }
    }
  }

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
                             << "LCT"<<i+1<<" "<<bx<<"/"<<cbx<<": "<<allLCTs[bx][mbx][i]<<std::endl;
          }
        }
      if (infoV > 0 and n>0) LogDebug("CSCMotherboard")
        <<"bx "<<bx<<" nnLCT:"<<n<<" "<<n<<std::endl;
    } // x-bx sorting
  }

  bool first = true;
  unsigned int n=0;
  for (const auto& p : readoutLCTs()) {
    if (debug_rpc_matching_ and first){
      std::cout << "========================================================================" << std::endl;
      std::cout << "Counting the final LCTs" << std::endl;
      std::cout << "========================================================================" << std::endl;
      first = false;
      std::cout << "tmb_cross_bx_algo: " << tmb_cross_bx_algo << std::endl;
    }
    n++;
    if (debug_rpc_matching_)
      std::cout << "LCT "<<n<<"  " << p <<std::endl;
  }
}

// check that the RE31 and RE41 chambers are really there
bool CSCMotherboardME3141RPC::hasRE31andRE41()
{
  // just pick two random chambers
  auto aRE31(rpc_g->chamber(RPCDetId(1,1,3,2,1,1,0)));
  auto aRE41(rpc_g->chamber(RPCDetId(-1,1,4,3,1,2,0)));
  return aRE31 and aRE41;
}


std::map<int,std::pair<double,double> > CSCMotherboardME3141RPC::createRPCRollLUT(RPCDetId id)
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
    //result[i] = std::make_pair(floorf(gp_top.eta() * 100) / 100, ceilf(gp_bottom.eta() * 100) / 100);
    result[i] = std::make_pair(std::abs(gp_top.eta()), std::abs(gp_bottom.eta()));
  }
  return result;
}


int CSCMotherboardME3141RPC::assignRPCRoll(double eta)
{
  int result = -99;
  for (const auto& p : rpcRollToEtaLimits_) {
    const float minEta((p.second).first);
    const float maxEta((p.second).second);
    if (minEta <= eta and eta <= maxEta) {
      result = p.first;
      break;
    }
  }
  return result;
}


void CSCMotherboardME3141RPC::retrieveRPCDigis(const RPCDigiCollection* rpcDigis, unsigned id)
{
  if (rpcDigis == nullptr) return;

  auto chamber(rpc_g->chamber(RPCDetId(id)));
  for (auto roll : chamber->rolls()) {
    RPCDetId roll_id(roll->id());
    auto digis_in_det = rpcDigis->get(roll_id);
    for (auto digi = digis_in_det.first; digi != digis_in_det.second; ++digi) {
      const int bx_shifted(lct_central_bx + digi->bx());
      for (int bx = bx_shifted - maxDeltaBXRPC_;bx <= bx_shifted + maxDeltaBXRPC_; ++bx) {
        rpcDigis_[bx].push_back(std::make_pair(roll_id(), *digi));
      }
    }
  }
}


void CSCMotherboardME3141RPC::printRPCTriggerDigis(int bx_start, int bx_stop)
{
  std::cout << "------------------------------------------------------------------------" << std::endl;
  bool first = true;
  for (int bx = bx_start; bx <= bx_stop; bx++) {
    std::vector<std::pair<unsigned int, const RPCDigi> > in_strips = rpcDigis_[bx];
    if (first) {
      std::cout << "* RPC trigger digis: " << std::endl;
    }
    first = false;
    std::cout << "N(digis) BX " << bx << " : " << in_strips.size() << std::endl;
    for (const auto& digi : in_strips){
      const auto roll_id(RPCDetId(digi.first));
      std::cout << "\tdetId " << digi.first << " " << roll_id << ", digi = " << digi.second.strip() << ", BX = " << digi.second.bx() + 6 << std::endl;
    }
  }
}


CSCMotherboardME3141RPC::RPCDigisBX
CSCMotherboardME3141RPC::matchingRPCDigis(const CSCCLCTDigi& clct, const RPCDigisBX& digis, bool first)
{
  CSCMotherboardME3141RPC::RPCDigisBX result;

  const int lowStrip(cscHsToRpcStrip_[clct.getKeyStrip()].first);
  const int highStrip(cscHsToRpcStrip_[clct.getKeyStrip()].second);
  const bool debug(false);
  if (debug) std::cout << "lowStrip " << lowStrip << " highStrip " << highStrip << " delta strip " << maxDeltaStripRPC_ <<std::endl;
  for (const auto& p: digis){
    auto strip((p.second).strip());
    if (debug) std::cout << "strip " << strip << std::endl;
    if (std::abs(lowStrip - strip) <= maxDeltaStripRPC_ or std::abs(strip - highStrip) <= maxDeltaStripRPC_){
    if (debug) std::cout << "++Matches! " << std::endl;
      result.push_back(p);
      if (first) return result;
    }
  }
  return result;
}


CSCMotherboardME3141RPC::RPCDigisBX
CSCMotherboardME3141RPC::matchingRPCDigis(const CSCALCTDigi& alct, const RPCDigisBX& digis, bool first)
{
  CSCMotherboardME3141RPC::RPCDigisBX result;

  int Wg = alct.getKeyWG();
  std::vector<int> Rolls;
  Rolls.push_back(cscWgToRpcRoll_[Wg]);
  if (Wg>=maxDeltaWg_ && cscWgToRpcRoll_[Wg] != cscWgToRpcRoll_[Wg-maxDeltaWg_])
      Rolls.push_back(cscWgToRpcRoll_[Wg-maxDeltaWg_]);
  if ((unsigned int)(Wg+maxDeltaWg_)<cscWgToRpcRoll_.size() && cscWgToRpcRoll_[Wg] != cscWgToRpcRoll_[Wg+maxDeltaWg_])
      Rolls.push_back(cscWgToRpcRoll_[Wg+maxDeltaWg_]);

  const bool debug(false);
  if (debug) std::cout << "ALCT keyWG " << alct.getKeyWG() << std::endl;
  for (auto alctRoll : Rolls)
  {
  if (debug) std::cout << " roll " << alctRoll << std::endl;
  for (const auto& p: digis){
    auto digiRoll(RPCDetId(p.first).roll());
    if (debug) std::cout << "Candidate ALCT: " << digiRoll << std::endl;
    if (alctRoll !=  digiRoll) continue;
    if (debug) std::cout << "++Matches! " << std::endl;
    result.push_back(p);
    if (first) return result;
  }
  }
  return result;
}


CSCMotherboardME3141RPC::RPCDigisBX
CSCMotherboardME3141RPC::matchingRPCDigis(const CSCCLCTDigi& clct, const CSCALCTDigi& alct, const RPCDigisBX& digis, bool first)
{
  CSCMotherboardME3141RPC::RPCDigisBX result;

  // Fetch all (!) digis matching to ALCTs and CLCTs
  auto digisClct(matchingRPCDigis(clct, digis, false));
  auto digisAlct(matchingRPCDigis(alct, digis, false));

  const bool debug(false);
  if (debug) std::cout << "-----------------------------------------------------------------------"<<std::endl;
  // Check if the digis overlap
  for (const auto& p : digisAlct){
    if (debug) std::cout<< "Candidate RPC digis for ALCT: " << p.first << " " << p.second << std::endl;
    for (auto q: digisClct){
      if (debug) std::cout<< "++Candidate RPC digis for CLCT: " << q.first << " " << q.second << std::endl;
      // look for exactly the same digis
      if (p.first != q.first) continue;
      //      if (not RPCDigi(*(p.second))==*(q.second)) continue;
      if (debug) std::cout << "++Matches! " << std::endl;
      result.push_back(p);
      if (first) return result;
    }
  }
  if (debug) std::cout << "-----------------------------------------------------------------------"<<std::endl;
  return result;
}

unsigned int CSCMotherboardME3141RPC::findQualityRPC(const CSCALCTDigi& aLCT, const CSCCLCTDigi& cLCT, bool hasRPC)
{

  /*
    Same LCT quality definition as standard LCTs
    a4 and c4 takes RPCs into account!!!
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
        int n_rpc = 0;
        if (hasRPC) n_rpc = 1;
        const bool a4((aLCT.getQuality() >= 1 and aLCT.getQuality() != 4) or
                      (aLCT.getQuality() == 4 and n_rpc >=1));
        const bool c4((cLCT.getQuality() >= 4) or (cLCT.getQuality() >= 3 and n_rpc>=1));
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


void CSCMotherboardME3141RPC::correlateLCTsRPC(CSCCLCTDigi bestCLCT,
                                             CSCCLCTDigi secondCLCT,
                                             RPCDigi rpcDigi, int roll,
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
    lct1 = constructLCTsRPC(bestCLCT, rpcDigi, roll, useOldLCTDataFormat_);
    lct1.setTrknmb(1);
  }

  if ((clct_trig_enable  and secondCLCT.isValid()) or
       (match_trig_enable and secondCLCT.isValid() and secondCLCT != bestCLCT))
    {
    lct2 = constructLCTsRPC(secondCLCT, rpcDigi, roll, useOldLCTDataFormat_);
    lct2.setTrknmb(2);
  }
}


void CSCMotherboardME3141RPC::correlateLCTsRPC(CSCALCTDigi bestALCT,
					  CSCALCTDigi secondALCT,
					  RPCDigi rpcDigi,
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
    lct1 = constructLCTsRPC(bestALCT, rpcDigi, useOldLCTDataFormat_);
    lct1.setTrknmb(1);
  }

  if ((alct_trig_enable  and secondALCT.isValid()) or
      (match_trig_enable and secondALCT.isValid() and secondALCT != bestALCT))
  {
    lct2 = constructLCTsRPC(secondALCT, rpcDigi, useOldLCTDataFormat_);
    lct2.setTrknmb(2);
  }
}


void CSCMotherboardME3141RPC::correlateLCTsRPC(CSCALCTDigi bestALCT, CSCALCTDigi secondALCT,
					    CSCCLCTDigi bestCLCT, CSCCLCTDigi secondCLCT,
					    CSCCorrelatedLCTDigi& lct1, CSCCorrelatedLCTDigi& lct2,
					    const RPCDigisBX& digis)
{
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
      (match_trig_enable and bestALCT.isValid() and bestCLCT.isValid())){
    lct1 = constructLCTsRPC(bestALCT, bestCLCT, digis);
    lct1.setTrknmb(1);
  }

  if (((secondALCT != bestALCT) or (secondCLCT != bestCLCT)) and
      ((alct_trig_enable  and secondALCT.isValid()) or
       (clct_trig_enable  and secondCLCT.isValid()) or
       (match_trig_enable and secondALCT.isValid() and secondCLCT.isValid()))){
    lct2 = constructLCTsRPC(secondALCT, secondCLCT, digis);
    lct2.setTrknmb(2);
  }
}

CSCCorrelatedLCTDigi CSCMotherboardME3141RPC::constructLCTsRPC(const CSCALCTDigi& aLCT, const CSCCLCTDigi& cLCT, const RPCDigisBX& digis)
{
  // CLCT pattern number
  unsigned int pattern = encodePattern(cLCT.getPattern(), cLCT.getStripType());

  // LCT quality number
  unsigned int quality = findQualityRPC(aLCT, cLCT, !digis.empty());

  // Bunch crossing: get it from cathode LCT if anode LCT is not there.
  int bx = aLCT.isValid() ? aLCT.getBX() : cLCT.getBX();
  if (!digis.empty()) bx = lct_central_bx + digis[0].second.bx(); // fix this!!!

  // construct correlated LCT; temporarily assign track number of 0.
  int trknmb = 0;
  CSCCorrelatedLCTDigi thisLCT(trknmb, 1, quality, aLCT.getKeyWG(),
                               cLCT.getKeyStrip(), pattern, cLCT.getBend(),
                               bx, 0, 0, 0, theTrigChamber);
  return thisLCT;
}


CSCCorrelatedLCTDigi CSCMotherboardME3141RPC::constructLCTsRPC(const CSCCLCTDigi& clct,
                                                          const RPCDigi& rpc, int roll,
                                                          bool oldDataFormat)
{
  if (oldDataFormat){
    // CLCT pattern number - for the time being, do not include RPCs in the pattern
    unsigned int pattern = encodePattern(clct.getPattern(), clct.getStripType());

    // LCT quality number -  dummy quality
    unsigned int quality = promoteCLCTRPCquality_ ? 14 : 11;

    // Bunch crossing: pick RPC bx
    int bx = rpc.bx() + lct_central_bx;

    // pick a random WG in the roll range
    int wg(4);

    // construct correlated LCT; temporarily assign track number of 0.
    return CSCCorrelatedLCTDigi(0, 1, quality, wg, clct.getKeyStrip(), pattern, clct.getBend(), bx, 0, 0, 0, theTrigChamber);
  }
  else {
    // CLCT pattern number - no pattern
    unsigned int pattern = 0;//encodePatternRPC(clct.getPattern(), clct.getStripType());

    // LCT quality number -  dummy quality
    unsigned int quality = 5;//findQualityRPC(alct, rpc);

    // Bunch crossing: get it from cathode LCT if anode LCT is not there.
    int bx = rpc.bx() + lct_central_bx;;

    // ALCT WG
    int wg(0);

    // construct correlated LCT; temporarily assign track number of 0.
    return CSCCorrelatedLCTDigi(0, 1, quality, wg, 0, pattern, 0, bx, 0, 0, 0, theTrigChamber);
  }
}


CSCCorrelatedLCTDigi CSCMotherboardME3141RPC::constructLCTsRPC(const CSCALCTDigi& alct,
                                                               const RPCDigi& rpc,
                                                               bool oldDataFormat)
{
  if (oldDataFormat){
    // CLCT pattern number - set it to a highest value
    // hack to get LCTs in the CSCTF
    unsigned int pattern = promoteALCTRPCpattern_ ? 10 : 0;

    // LCT quality number - set it to a very high value
    // hack to get LCTs in the CSCTF
    unsigned int quality = promoteALCTRPCquality_ ? 14 : 11;

    // Bunch crossing
    int bx = rpc.bx() + lct_central_bx;

    // get keyStrip from LUT
    int keyStrip = rpcStripToCscHs_[rpc.strip()];

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
    int bx = rpc.bx() + lct_central_bx;

    // get keyStrip from LUT
    int keyStrip = rpcStripToCscHs_[rpc.strip()];
    // get wiregroup from ALCT
    int wg = alct.getKeyWG();

    // construct correlated LCT; temporarily assign track number of 0.
    return CSCCorrelatedLCTDigi(0, 1, quality, wg, keyStrip, pattern, 0, bx, 0, 0, 0, theTrigChamber);
  }
}


//readout LCTs
std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME3141RPC::readoutLCTs()
{
  return getLCTs();
}

//getLCTs when we use different sort algorithm
std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME3141RPC::getLCTs()
{
  std::vector<CSCCorrelatedLCTDigi> result;
  for (int bx = 0; bx < MAX_LCT_BINS; bx++) {
    std::vector<CSCCorrelatedLCTDigi> tmpV;
    if (tmb_cross_bx_algo == 3) {
      tmpV = sortLCTsByQuality(bx);
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
std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME3141RPC::sortLCTsByQuality(int bx)
{
  std::vector<CSCCorrelatedLCTDigi> LCTs;
  LCTs.clear();
  for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
    for (int i=0;i<2;i++)
      if (allLCTs[bx][mbx][i].isValid())
        LCTs.push_back(allLCTs[bx][mbx][i]);

  // return sorted vector with 2 highest quality LCTs
  std::sort(LCTs.begin(), LCTs.end(), CSCMotherboard::sortByQuality);
  if (LCTs.size()> max_me3141_lcts) LCTs.erase(LCTs.begin()+max_me3141_lcts, LCTs.end());
  return  LCTs;
}
