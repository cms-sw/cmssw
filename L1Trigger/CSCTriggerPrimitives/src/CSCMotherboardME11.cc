//-----------------------------------------------------------------------------
//
//   Class: CSCMotherboardME11
//
//   Description:
//    Extended CSCMotherboard for ME11 to handle ME1a and ME1b separately
//
//   Author List: Vadim Khotilovich 12 May 2009
//
//   $Date: 2012/10/18 05:01:39 $
//   $Revision: 1.1.4.3 $
//
//-----------------------------------------------------------------------------

#include <L1Trigger/CSCTriggerPrimitives/src/CSCMotherboardME11.h>
//#include <Utilities/Timing/interface/TimingReport.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>



// LUT for which ME1/1 wire group can cross which ME1/a halfstrip
// 1st index: WG number
// 2nd index: inclusive HS range
const int CSCMotherboardME11::lut_wg_vs_hs_me1a[48][2] = {
{0, 95},{0, 95},{0, 95},{0, 95},{0, 95},
{0, 95},{0, 95},{0, 95},{0, 95},{0, 95},
{0, 95},{0, 95},{0, 77},{0, 61},{0, 39},
{0, 22},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{-1,-1},{-1,-1},{-1,-1} };
// a modified LUT for ganged ME1a
const int CSCMotherboardME11::lut_wg_vs_hs_me1ag[48][2] = {
{0, 31},{0, 31},{0, 31},{0, 31},{0, 31},
{0, 31},{0, 31},{0, 31},{0, 31},{0, 31},
{0, 31},{0, 31},{0, 31},{0, 31},{0, 31},
{0, 22},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{-1,-1},{-1,-1},{-1,-1} };

// LUT for which ME1/1 wire group can cross which ME1/b halfstrip
// 1st index: WG number
// 2nd index: inclusive HS range
const int CSCMotherboardME11::lut_wg_vs_hs_me1b[48][2] = {
{-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{100, 127},{73, 127},{47, 127},{22, 127},{0, 127},
{0, 127},{0, 127},{0, 127},{0, 127},{0, 127},
{0, 127},{0, 127},{0, 127},{0, 127},{0, 127},
{0, 127},{0, 127},{0, 127},{0, 127},{0, 127},
{0, 127},{0, 127},{0, 127},{0, 127},{0, 127},
{0, 127},{0, 127},{0, 127},{0, 127},{0, 127},
{0, 127},{0, 127},{0, 127},{0, 127},{0, 105},
{0, 93},{0, 78},{0, 63} };


CSCMotherboardME11::CSCMotherboardME11(unsigned endcap, unsigned station,
			       unsigned sector, unsigned subsector,
			       unsigned chamber,
			       const edm::ParameterSet& conf) :
		CSCMotherboard(endcap, station, sector, subsector, chamber, conf)
{
  edm::ParameterSet commonParams = conf.getParameter<edm::ParameterSet>("commonParam");

  // special configuration parameters for ME11 treatment
  smartME1aME1b = commonParams.getUntrackedParameter<bool>("smartME1aME1b", true);
  disableME1a = commonParams.getUntrackedParameter<bool>("disableME1a", false);
  gangedME1a = commonParams.getUntrackedParameter<bool>("gangedME1a", false);

  if (!isSLHC) edm::LogError("L1CSCTPEmulatorConfigError")
    << "+++ Upgrade CSCMotherboardME11 constructed while isSLHC is not set! +++\n";
  if (!smartME1aME1b) edm::LogError("L1CSCTPEmulatorConfigError")
    << "+++ Upgrade CSCMotherboardME11 constructed while smartME1aME1b is not set! +++\n";

  edm::ParameterSet alctParams = conf.getParameter<edm::ParameterSet>("alctSLHC");
  edm::ParameterSet clctParams = conf.getParameter<edm::ParameterSet>("clctSLHC");
  edm::ParameterSet tmbParams = conf.getParameter<edm::ParameterSet>("tmbSLHC");

  clct1a = new CSCCathodeLCTProcessor(endcap, station, sector, subsector, chamber, clctParams, commonParams, tmbParams);
  clct1a->setRing(4);

  match_earliest_alct_me11_only = tmbParams.getUntrackedParameter<bool>("matchEarliestAlctME11Only",true);
  match_earliest_clct_me11_only = tmbParams.getUntrackedParameter<bool>("matchEarliestClctME11Only",true);

  // if true: use regular CLCT-to-ALCT matching in TMB
  // if false: do ALCT-to-CLCT matching
  clct_to_alct = tmbParams.getUntrackedParameter<bool>("clctToAlct",true);

  // whether to not reuse CLCTs that were used by previous matching ALCTs
  // in ALCT-to-CLCT algorithm
  drop_used_clcts = tmbParams.getUntrackedParameter<bool>("tmbDropUsedClcts",true);

  tmb_cross_bx_algo = tmbParams.getUntrackedParameter<unsigned int>("tmbCrossBxAlgorithm");

  // maximum lcts per BX in ME11: 2, 3, 4 or 999
  max_me11_lcts = tmbParams.getUntrackedParameter<unsigned int>("maxME11LCTs",4);

  pref[0] = match_trig_window_size/2;
  for (unsigned int m=2; m<match_trig_window_size; m+=2)
  {
    pref[m-1] = pref[0] - m/2;
    pref[m]   = pref[0] + m/2;
  }
}


CSCMotherboardME11::CSCMotherboardME11() : CSCMotherboard()
{
  // Constructor used only for testing.

  clct1a = new CSCCathodeLCTProcessor();
  clct1a->setRing(4);

  pref[0] = match_trig_window_size/2;
  for (unsigned int m=2; m<match_trig_window_size; m+=2)
  {
    pref[m-1] = pref[0] - m/2;
    pref[m]   = pref[0] + m/2;
  }
}


CSCMotherboardME11::~CSCMotherboardME11()
{
  if (clct1a) delete clct1a;
}


void CSCMotherboardME11::clear()
{
  CSCMotherboard::clear();
  if (clct1a) clct1a->clear();
  for (int bx = 0; bx < MAX_LCT_BINS; bx++)
  {
    //firstLCT1a[bx].clear();
    //secondLCT1a[bx].clear();
    for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
      for (int i=0;i<2;i++)
      {
        allLCTs1b[bx][mbx][i].clear();
        allLCTs1a[bx][mbx][i].clear();
      }
  }
}

// Set configuration parameters obtained via EventSetup mechanism.
void CSCMotherboardME11::setConfigParameters(const CSCDBL1TPParameters* conf)
{
  alct->setConfigParameters(conf);
  clct->setConfigParameters(conf);
  clct1a->setConfigParameters(conf);
  // No config. parameters in DB for the TMB itself yet.
}


void CSCMotherboardME11::run(const CSCWireDigiCollection* wiredc,
                             const CSCComparatorDigiCollection* compdc)
{
  clear();
  
  if (!( alct && clct &&  clct1a && smartME1aME1b))
  {
    if (infoV >= 0) edm::LogError("L1CSCTPEmulatorSetupError")
      << "+++ run() called for non-existing ALCT/CLCT processor! +++ \n";
    return;
  }

  alctV = alct->run(wiredc); // run anodeLCT
  clctV1b = clct->run(compdc); // run cathodeLCT in ME1/b
  clctV1a = clct1a->run(compdc); // run cathodeLCT in ME1/a

  //int n_clct_a=0, n_clct_b=0;
  //if (clct1a->bestCLCT[6].isValid() && clct1a->bestCLCT[6].getBX()==6) n_clct_a++;
  //if (clct1a->secondCLCT[6].isValid() && clct1a->secondCLCT[6].getBX()==6) n_clct_a++;

  int used_alct_mask[20], used_alct_mask_1a[20];
  int used_clct_mask[20], used_clct_mask_1a[20];
  for (int b=0;b<20;b++)
    used_alct_mask[b] = used_alct_mask_1a[b] = used_clct_mask[b] = used_clct_mask_1a[b] = 0;

  // CLCT-centric CLCT-to-ALCT matching
  if (clct_to_alct) for (int bx_clct = 0; bx_clct < CSCCathodeLCTProcessor::MAX_CLCT_BINS; bx_clct++)
  {
    // matching in ME1b
    if (clct->bestCLCT[bx_clct].isValid())
    {
      int bx_alct_start = bx_clct - match_trig_window_size/2;
      int bx_alct_stop  = bx_clct + match_trig_window_size/2;
      for (int bx_alct = bx_alct_start; bx_alct <= bx_alct_stop; bx_alct++)
      {
        if (bx_alct < 0 || bx_alct >= CSCAnodeLCTProcessor::MAX_ALCT_BINS) continue;
        if (drop_used_alcts && used_alct_mask[bx_alct]) continue;
        if (alct->bestALCT[bx_alct].isValid())
        {
          if (infoV > 1) LogTrace("CSCMotherboard")
            << "Successful CLCT-ALCT match in ME1b: bx_clct = " << bx_clct
            << "; match window: [" << bx_alct_start << "; " << bx_alct_stop
            << "]; bx_alct = " << bx_alct;
          int mbx = bx_alct_stop - bx_alct;
          correlateLCTs(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
                        clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct],
                        allLCTs1b[bx_alct][mbx][0], allLCTs1b[bx_alct][mbx][1], ME1B);
          if (allLCTs1b[bx_alct][mbx][0].isValid())
          {
            used_alct_mask[bx_alct] += 1;
            if (match_earliest_alct_me11_only) break;
          }
        }
      }
      // Do not report CLCT-only LCT for ME1b
    }
    // matching in ME1a
    if (clct1a->bestCLCT[bx_clct].isValid())
    {
      int bx_alct_start = bx_clct - match_trig_window_size/2;
      int bx_alct_stop  = bx_clct + match_trig_window_size/2;
      for (int bx_alct = bx_alct_start; bx_alct <= bx_alct_stop; bx_alct++)
      {
        if (bx_alct < 0 || bx_alct >= CSCAnodeLCTProcessor::MAX_ALCT_BINS) continue;
        if (drop_used_alcts && used_alct_mask_1a[bx_alct]) continue;
        if (alct->bestALCT[bx_alct].isValid())
        {
          if (infoV > 1) LogTrace("CSCMotherboard")
            << "Successful CLCT-ALCT match in ME1a: bx_clct = " << bx_clct
            << "; match window: [" << bx_alct_start << "; " << bx_alct_stop
            << "]; bx_alct = " << bx_alct;
          int mbx = bx_alct_stop - bx_alct;
          correlateLCTs(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
                        clct1a->bestCLCT[bx_clct], clct1a->secondCLCT[bx_clct],
                        allLCTs1a[bx_alct][mbx][0], allLCTs1a[bx_alct][mbx][1], ME1A);
          if (allLCTs1a[bx_alct][mbx][0].isValid())
          {
            used_alct_mask_1a[bx_alct] += 1;
            if (match_earliest_alct_me11_only) break;
          }
        }
      }
      // Do not report CLCT-only LCT for ME1b
    }
    // Do not attempt to make ALCT-only LCT for ME1b
  } // end of CLCT-centric matching

  // ALCT-centric ALCT-to-CLCT matching
  else for (int bx_alct = 0; bx_alct < CSCAnodeLCTProcessor::MAX_ALCT_BINS; bx_alct++)
  {
    if (alct->bestALCT[bx_alct].isValid())
    {
      int bx_clct_start = bx_alct - match_trig_window_size/2;
      int bx_clct_stop  = bx_alct + match_trig_window_size/2;
      
      // matching in ME1b
      for (int bx_clct = bx_clct_start; bx_clct <= bx_clct_stop; bx_clct++)
      {
        if (bx_clct < 0 || bx_clct >= CSCCathodeLCTProcessor::MAX_CLCT_BINS) continue;
        if (drop_used_clcts && used_clct_mask[bx_clct]) continue;
        if (clct->bestCLCT[bx_clct].isValid())
        {
          if (infoV > 1) LogTrace("CSCMotherboard")
            << "Successful ALCT-CLCT match in ME1b: bx_alct = " << bx_alct
            << "; match window: [" << bx_clct_start << "; " << bx_clct_stop
            << "]; bx_clct = " << bx_clct;
          int mbx = bx_clct-bx_clct_start;
          correlateLCTs(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
                        clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct],
                        allLCTs1b[bx_alct][mbx][0], allLCTs1b[bx_alct][mbx][1], ME1B);
          if (allLCTs1b[bx_alct][mbx][0].isValid())
          {
            used_clct_mask[bx_clct] += 1;
            if (match_earliest_clct_me11_only) break;
          }
        }
      }

      // matching in ME1a
      for (int bx_clct = bx_clct_start; bx_clct <= bx_clct_stop; bx_clct++)
      {
        if (bx_clct < 0 || bx_clct >= CSCCathodeLCTProcessor::MAX_CLCT_BINS) continue;
        if (drop_used_clcts && used_clct_mask_1a[bx_clct]) continue;
        if (clct1a->bestCLCT[bx_clct].isValid())
        {
          if (infoV > 1) LogTrace("CSCMotherboard")
            << "Successful ALCT-CLCT match in ME1a: bx_alct = " << bx_alct
            << "; match window: [" << bx_clct_start << "; " << bx_clct_stop
            << "]; bx_clct = " << bx_clct;
          int mbx = bx_clct-bx_clct_start;
          correlateLCTs(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
                        clct1a->bestCLCT[bx_clct], clct1a->secondCLCT[bx_clct],
                        allLCTs1a[bx_alct][mbx][0], allLCTs1a[bx_alct][mbx][1], ME1A);
          if (allLCTs1a[bx_alct][mbx][0].isValid())
          {
            used_clct_mask_1a[bx_clct] += 1;
            if (match_earliest_clct_me11_only) break;
          }
        }
      }
    }
  } // end of ALCT-centric matching

  // reduction of nLCTs per each BX
  for (int bx = 0; bx < MAX_LCT_BINS; bx++)
  {
    // counting
    unsigned int n1a=0, n1b=0;
    for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
      for (int i=0;i<2;i++)
      {
        int cbx = bx + mbx - match_trig_window_size/2;
        if (allLCTs1b[bx][mbx][i].isValid())
        {
          n1b++;
          if (infoV > 0) LogDebug("CSCMotherboard") << "1b LCT"<<i+1<<" "<<bx<<"/"<<cbx<<": "<<allLCTs1b[bx][mbx][i];
        }
        if (allLCTs1a[bx][mbx][i].isValid())
        {
          n1a++;
          if (infoV > 0) LogDebug("CSCMotherboard") << "1a LCT"<<i+1<<" "<<bx<<"/"<<cbx<<": "<<allLCTs1a[bx][mbx][i];
        }
      }
    if (infoV > 0 && n1a+n1b>0) LogDebug("CSCMotherboard") <<"bx "<<bx<<" nLCT:"<<n1a<<" "<<n1b<<" "<<n1a+n1b;

    // some simple cross-bx sorting algorithms
    if (tmb_cross_bx_algo == 1 && (n1a>2 || n1b>2) )
    {
      n1a=0, n1b=0;
      for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
        for (int i=0;i<2;i++)
        {
          if (allLCTs1b[bx][pref[mbx]][i].isValid())
          {
            n1b++;
            if (n1b>2) allLCTs1b[bx][pref[mbx]][i].clear();
          }
          if (allLCTs1a[bx][pref[mbx]][i].isValid())
          {
            n1a++;
            if (n1a>2) allLCTs1a[bx][pref[mbx]][i].clear();
          }
        }

      if (infoV > 0) LogDebug("CSCMotherboard") <<"After x-bx sorting:";
      n1a=0, n1b=0;
      for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
        for (int i=0;i<2;i++)
        {
          int cbx = bx + mbx - match_trig_window_size/2;
          if (allLCTs1b[bx][mbx][i].isValid())
          {
            n1b++;
            if (infoV > 0) LogDebug("CSCMotherboard") << "1b LCT"<<i+1<<" "<<bx<<"/"<<cbx<<": "<<allLCTs1b[bx][mbx][i];
          }
          if (allLCTs1a[bx][mbx][i].isValid())
          {
            n1a++;
            if (infoV > 0) LogDebug("CSCMotherboard") << "1a LCT"<<i+1<<" "<<bx<<"/"<<cbx<<": "<<allLCTs1a[bx][mbx][i];
          }
        }
      if (infoV > 0 && n1a+n1b>0) LogDebug("CSCMotherboard") <<"bx "<<bx<<" nnLCT:"<<n1a<<" "<<n1b<<" "<<n1a+n1b;
    } // x-bx sorting

    // Maximum 2 or 3 per whole ME11 per BX case:
    // (supposedly, now we should have max 2 per bx in each 1a and 1b)
    if ( n1a+n1b > max_me11_lcts )
    {
      // do it simple so far: take all low eta 1/b stubs
      unsigned int nLCT=n1b;
      n1a=0;
      // right now nLCT<=2; cut 1a if necessary
      for (unsigned int mbx=0; mbx<match_trig_window_size; mbx++)
        for (int i=0;i<2;i++)
          if (allLCTs1a[bx][mbx][i].isValid()) {
            nLCT++;
            if (nLCT>max_me11_lcts) allLCTs1a[bx][mbx][i].clear();
            else n1a++;
          }
      if (infoV > 0 && nLCT>0) LogDebug("CSCMotherboard") <<"bx "<<bx<<" nnnLCT:"<<n1a<<" "<<n1b<<" "<<n1a+n1b;
    }
  }// reduction per bx

  //if (infoV > 1) LogTrace("CSCMotherboardME11")<<"clct_count E:"<<theEndcap<<"S:"<<theStation<<"R:"<<1<<"C:"
  //  <<CSCTriggerNumbering::chamberFromTriggerLabels(theSector,theSubsector, theStation, theTrigChamber)
  //  <<"  a "<<n_clct_a<<"  b "<<n_clct_b<<"  ab "<<n_clct_a+n_clct_b;
}


std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME11::readoutLCTs1a()
{
  return readoutLCTs(ME1A);
}


std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME11::readoutLCTs1b()
{
  return readoutLCTs(ME1B);
}


// Returns vector of read-out correlated LCTs, if any.  Starts with
// the vector of all found LCTs and selects the ones in the read-out
// time window.
std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME11::readoutLCTs(int me1ab)
{
  std::vector<CSCCorrelatedLCTDigi> tmpV;

  // The start time of the L1A*LCT coincidence window should be related
  // to the fifo_pretrig parameter, but I am not completely sure how.
  // Just choose it such that the window is centered at bx=7.  This may
  // need further tweaking if the value of tmb_l1a_window_size changes.
  //static int early_tbins = 4;
  // The number of LCT bins in the read-out is given by the
  // tmb_l1a_window_size parameter, forced to be odd
  static int lct_bins   = 
    (tmb_l1a_window_size % 2 == 0) ? tmb_l1a_window_size + 1 : tmb_l1a_window_size;
  static int late_tbins = early_tbins + lct_bins;


  // Start from the vector of all found correlated LCTs and select
  // those within the LCT*L1A coincidence window.
  int bx_readout = -1;
  std::vector<CSCCorrelatedLCTDigi> all_lcts;
  if (me1ab == ME1A) all_lcts = getLCTs1a();
  if (me1ab == ME1B) all_lcts = getLCTs1b();
  std::vector <CSCCorrelatedLCTDigi>::const_iterator plct = all_lcts.begin();
  for (; plct != all_lcts.end(); plct++)
  {
    if (!plct->isValid()) continue;

    int bx = (*plct).getBX();
    // Skip LCTs found too early relative to L1Accept.
    if (bx <= early_tbins) continue;

    // Skip LCTs found too late relative to L1Accept.
    if (bx > late_tbins) continue;

    // If (readout_earliest_2) take only LCTs in the earliest bx in the read-out window:
    // in digi->raw step, LCTs have to be packed into the TMB header, and
    // currently there is room just for two.
    if (readout_earliest_2 && (bx_readout == -1 || bx == bx_readout) )
    {
      tmpV.push_back(*plct);
      if (bx_readout == -1) bx_readout = bx;
    }
    else tmpV.push_back(*plct);
  }
  return tmpV;
}


// Returns vector of found correlated LCTs, if any.
std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME11::getLCTs1b()
{
  std::vector<CSCCorrelatedLCTDigi> tmpV;

  for (int bx = 0; bx < MAX_LCT_BINS; bx++) 
    for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
      for (int i=0;i<2;i++)
        if (allLCTs1b[bx][mbx][i].isValid()) tmpV.push_back(allLCTs1b[bx][mbx][i]);
  return tmpV;
}

// Returns vector of found correlated LCTs, if any.
std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME11::getLCTs1a()
{
  std::vector<CSCCorrelatedLCTDigi> tmpV;
  
  // disabled ME1a
  if (mpc_block_me1a || disableME1a) return tmpV;

  // Report all LCTs found.
  for (int bx = 0; bx < MAX_LCT_BINS; bx++)
    for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++) 
      for (int i=0;i<2;i++)
        if (allLCTs1a[bx][mbx][i].isValid())  tmpV.push_back(allLCTs1a[bx][mbx][i]);
  return tmpV;
}


bool CSCMotherboardME11::doesALCTCrossCLCT(CSCALCTDigi &a, CSCCLCTDigi &c, int me)
{
  if ( !c.isValid() || !a.isValid() ) return false;
  int key_hs = c.getKeyStrip();
  int key_wg = a.getKeyWG();
  if ( me == ME1A )
  {
    if ( !gangedME1a )
    {
      // wrap around ME11 HS number for -z endcap
      if (theEndcap==2) key_hs = 95 - key_hs;
      if ( key_hs >= lut_wg_vs_hs_me1a[key_wg][0] && 
           key_hs <= lut_wg_vs_hs_me1a[key_wg][1]    ) return true;
      return false;
    }
    else
    {
      if (theEndcap==2) key_hs = 31 - key_hs;
      if ( key_hs >= lut_wg_vs_hs_me1ag[key_wg][0] &&
           key_hs <= lut_wg_vs_hs_me1ag[key_wg][1]    ) return true;
      return false;
    }
  }
  if ( me == ME1B)
  {
    if (theEndcap==2) key_hs = 127 - key_hs;
    if ( key_hs >= lut_wg_vs_hs_me1b[key_wg][0] && 
         key_hs <= lut_wg_vs_hs_me1b[key_wg][1]      ) return true;
  }
  return false;
}


void CSCMotherboardME11::correlateLCTs(CSCALCTDigi bestALCT,
				   CSCALCTDigi secondALCT,
				   CSCCLCTDigi bestCLCT,
				   CSCCLCTDigi secondCLCT,
				   CSCCorrelatedLCTDigi& lct1,
				   CSCCorrelatedLCTDigi& lct2)
{
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
      (match_trig_enable && bestALCT.isValid() && bestCLCT.isValid()))
  {
    lct1 = constructLCTs(bestALCT, bestCLCT);
    lct1.setTrknmb(1);
  }

  if (((secondALCT != bestALCT) || (secondCLCT != bestCLCT)) &&
      ((alct_trig_enable  && secondALCT.isValid()) ||
       (clct_trig_enable  && secondCLCT.isValid()) ||
       (match_trig_enable && secondALCT.isValid() && secondCLCT.isValid())))
  {
    lct2 = constructLCTs(secondALCT, secondCLCT);
    lct2.setTrknmb(2);
  }
}


void CSCMotherboardME11::correlateLCTs(CSCALCTDigi bestALCT,
				   CSCALCTDigi secondALCT,
				   CSCCLCTDigi bestCLCT,
				   CSCCLCTDigi secondCLCT,
				   CSCCorrelatedLCTDigi& lct1,
				   CSCCorrelatedLCTDigi& lct2,
                                   int me)
{
  // assume that always anodeBestValid && cathodeBestValid
  
  if (secondALCT == bestALCT) secondALCT.clear();
  if (secondCLCT == bestCLCT) secondCLCT.clear();

  int ok11 = doesALCTCrossCLCT( bestALCT, bestCLCT, me);
  int ok12 = doesALCTCrossCLCT( bestALCT, secondCLCT, me);
  int ok21 = doesALCTCrossCLCT( secondALCT, bestCLCT, me);
  int ok22 = doesALCTCrossCLCT( secondALCT, secondCLCT, me);
  int code = (ok11<<3) | (ok12<<2) | (ok21<<1) | (ok22);

  int dbg=0;
  int ring = me;
  int chamb= CSCTriggerNumbering::chamberFromTriggerLabels(theSector,theSubsector, theStation, theTrigChamber);
  CSCDetId did(theEndcap, theStation, ring, chamb, 0);
  if (dbg) LogTrace("CSCMotherboardME11")<<"debug correlateLCTs in "<<did<<std::endl
	   <<"ALCT1: "<<bestALCT<<std::endl
	   <<"ALCT2: "<<secondALCT<<std::endl
	   <<"CLCT1: "<<bestCLCT<<std::endl
	   <<"CLCT2: "<<secondCLCT<<std::endl
	   <<"ok 11 12 21 22 code = "<<ok11<<" "<<ok12<<" "<<ok21<<" "<<ok22<<" "<<code<<std::endl;

  if ( code==0 ) return;

  // LUT defines correspondence between possible ok## combinations
  // and resulting lct1 and lct2
  int lut[16][2] = {
          //ok: 11 12 21 22
    {0 ,0 }, // 0  0  0  0
    {22,0 }, // 0  0  0  1
    {21,0 }, // 0  0  1  0
    {21,22}, // 0  0  1  1
    {12,0 }, // 0  1  0  0
    {12,22}, // 0  1  0  1
    {12,21}, // 0  1  1  0
    {12,21}, // 0  1  1  1
    {11,0 }, // 1  0  0  0
    {11,22}, // 1  0  0  1
    {11,21}, // 1  0  1  0
    {11,22}, // 1  0  1  1
    {11,12}, // 1  1  0  0
    {11,22}, // 1  1  0  1
    {11,12}, // 1  1  1  0
    {11,22}, // 1  1  1  1
  };

  if (dbg) LogTrace("CSCMotherboardME11")<<"lut 0 1 = "<<lut[code][0]<<" "<<lut[code][1]<<std::endl;

  switch (lut[code][0]) {
    case 11:
      lct1 = constructLCTs(bestALCT, bestCLCT);
      break;
    case 12:
      lct1 = constructLCTs(bestALCT, secondCLCT);
      break;
    case 21:
      lct1 = constructLCTs(secondALCT, bestCLCT);
      break;
    case 22:
      lct1 = constructLCTs(secondALCT, secondCLCT);
      break;
    default: return;  
  }
  lct1.setTrknmb(1);

  if (dbg) LogTrace("CSCMotherboardME11")<<"lct1: "<<lct1<<std::endl;
  
  switch (lut[code][1])
  {
    case 12:
      lct2 = constructLCTs(bestALCT, secondCLCT);
      lct2.setTrknmb(2);
      if (dbg) LogTrace("CSCMotherboardME11")<<"lct2: "<<lct2<<std::endl;
      return;
    case 21:
      lct2 = constructLCTs(secondALCT, bestCLCT);
      lct2.setTrknmb(2);
      if (dbg) LogTrace("CSCMotherboardME11")<<"lct2: "<<lct2<<std::endl;
      return;
    case 22:
      lct2 = constructLCTs(secondALCT, secondCLCT);
      lct2.setTrknmb(2);
      if (dbg) LogTrace("CSCMotherboardME11")<<"lct2: "<<lct2<<std::endl;
      return;
    default: return;
  }
  if (dbg) LogTrace("CSCMotherboardME11")<<"out of correlateLCTs"<<std::endl;

  return;
}

