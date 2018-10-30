#include "L1Trigger/CSCTriggerPrimitives/src/CSCMotherboardME3141.h"

CSCMotherboardME3141::CSCMotherboardME3141(unsigned endcap, unsigned station,
                                     unsigned sector, unsigned subsector,
                                     unsigned chamber,
                                     const edm::ParameterSet& conf) :
  CSCUpgradeMotherboard(endcap, station, sector, subsector, chamber, conf)
{
  if (!isSLHC_ or !runME3141ILT_) edm::LogError("CSCMotherboardME3141|ConfigError")
    << "+++ Upgrade CSCMotherboardME3141 constructed while isSLHC is not set! +++\n";
}

CSCMotherboardME3141::CSCMotherboardME3141()
  : CSCUpgradeMotherboard()
{
  if (!isSLHC_ or !runME3141ILT_) edm::LogError("CSCMotherboardME3141|ConfigError")
    << "+++ Upgrade CSCMotherboardME3141 constructed while isSLHC is not set! +++\n";
}

CSCMotherboardME3141::~CSCMotherboardME3141()
{
}

void
CSCMotherboardME3141::run(const CSCWireDigiCollection* wiredc,
                          const CSCComparatorDigiCollection* compdc)
{
  clear();

  if (!( alctProc and clctProc))
  {
    if (infoV >= 0) edm::LogError("CSCMotherboardME3141|SetupError")
      << "+++ run() called for non-existing ALCT/CLCT processor! +++ \n";
    return;
  }

  alctProc->setCSCGeometry(csc_g);
  clctProc->setCSCGeometry(csc_g);

  alctV = alctProc->run(wiredc); // run anodeLCT
  clctV = clctProc->run(compdc); // run cathodeLCT

  // if there are no ALCTs and no CLCTs, it does not make sense to run this TMB
  if (alctV.empty() and clctV.empty()) return;

  int used_clct_mask[20];
  for (int c=0;c<20;++c) used_clct_mask[c]=0;

  // ALCT centric matching
  for (int bx_alct = 0; bx_alct < CSCConstants::MAX_ALCT_TBINS; bx_alct++)
  {
    if (alctProc->bestALCT[bx_alct].isValid())
    {
      const int bx_clct_start(bx_alct - match_trig_window_size/2 - alctClctOffset);
      const int bx_clct_stop(bx_alct + match_trig_window_size/2 - alctClctOffset);

      if (debug_matching){
        LogTrace("CSCMotherboardME3141") << "========================================================================" << std::endl;
        LogTrace("CSCMotherboardME3141") << "ALCT-CLCT matching in ME34/1 chamber: " << cscChamber->id() << std::endl;
        LogTrace("CSCMotherboardME3141") << "------------------------------------------------------------------------" << std::endl;
        LogTrace("CSCMotherboardME3141") << "+++ Best ALCT Details: ";
        alctProc->bestALCT[bx_alct].print();
        LogTrace("CSCMotherboardME3141") << "+++ Second ALCT Details: ";
        alctProc->secondALCT[bx_alct].print();

        LogTrace("CSCMotherboardME3141") << "------------------------------------------------------------------------" << std::endl;
        LogTrace("CSCMotherboardME3141") << "Attempt ALCT-CLCT matching in ME34/13 in bx range: [" << bx_clct_start << "," << bx_clct_stop << "]" << std::endl;
      }

      // ALCT-to-CLCT
      for (int bx_clct = bx_clct_start; bx_clct <= bx_clct_stop; bx_clct++)
      {
        if (bx_clct < 0 or bx_clct >= CSCConstants::MAX_CLCT_TBINS) continue;
        if (drop_used_clcts and used_clct_mask[bx_clct]) continue;
        if (clctProc->bestCLCT[bx_clct].isValid())
        {
          if (debug_matching) LogTrace("CSCMotherboardME3141") << "++Valid ME21 CLCT: " << clctProc->bestCLCT[bx_clct] << std::endl;

          int mbx = bx_clct-bx_clct_start;
          CSCMotherboardME3141::correlateLCTs(alctProc->bestALCT[bx_alct], alctProc->secondALCT[bx_alct],
                                              clctProc->bestCLCT[bx_clct], clctProc->secondCLCT[bx_clct],
                                              allLCTs(bx_alct,mbx,0), allLCTs(bx_alct,mbx,1));
          if (infoV > 1)
            LogTrace("CSCMotherboardME3141") << "Successful ALCT-CLCT match in ME21: bx_alct = " << bx_alct
                                             << "; match window: [" << bx_clct_start << "; " << bx_clct_stop
                                             << "]; bx_clct = " << bx_clct << std::endl;
          LogTrace("CSCMotherboardME3141") << "+++ Best CLCT Details: ";
          clctProc->bestCLCT[bx_clct].print();
          LogTrace("CSCMotherboardME3141") << "+++ Second CLCT Details: ";
          clctProc->secondCLCT[bx_clct].print();
          if (allLCTs(bx_alct,mbx,0).isValid()) {
            used_clct_mask[bx_clct] += 1;
            if (match_earliest_clct_only) break;
          }
        }
      }
    }
  }

  // reduction of nLCTs per each BX
  for (int bx = 0; bx < CSCConstants::MAX_LCT_TBINS; bx++)
  {
    // counting
    unsigned int n=0;
    for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
      for (int i=0;i<CSCConstants::MAX_LCTS_PER_CSC;i++)
      {
        int cbx = bx + mbx - match_trig_window_size/2;
        if (allLCTs(bx,mbx,i).isValid())
        {
          ++n;
          if (infoV > 0) LogDebug("CSCMotherboardME3141")
                           << "LCT"<<i+1<<" "<<bx<<"/"<<cbx<<": "<<allLCTs(bx,mbx,i)<<std::endl;
        }
      }

    // some simple cross-bx sorting algorithms
    if (tmb_cross_bx_algo == 1 and (n>2))
    {
      n=0;
      for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
        for (int i=0;i<CSCConstants::MAX_LCTS_PER_CSC;i++)
        {
          if (allLCTs(bx,pref[mbx],i).isValid())
          {
            n++;
            if (n>2) allLCTs(bx,pref[mbx],i).clear();
          }
	      }

      n=0;
      for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
        for (int i=0;i<CSCConstants::MAX_LCTS_PER_CSC;i++)
        {
          int cbx = bx + mbx - match_trig_window_size/2;
          if (allLCTs(bx,mbx,i).isValid())
          {
            n++;
            if (infoV > 0) LogDebug("CSCMotherboardME3141")
                             << "LCT"<<i+1<<" "<<bx<<"/"<<cbx<<": "<<allLCTs(bx,mbx,i)<< std::endl;
          }
	      }
      if (infoV > 0 and n>0) LogDebug("CSCMotherboardME3141")
                               <<"bx "<<bx<<" nnLCT:"<<n<<" "<<n<<std::endl;
    } // x-bx sorting
  }

  bool first = true;
  unsigned int n=0;
  for (const auto& p : readoutLCTs()) {
    if (debug_matching and first){
      LogTrace("CSCMotherboardME3141") << "========================================================================" << std::endl;
      LogTrace("CSCMotherboardME3141") << "Counting the final LCTs" << std::endl;
      LogTrace("CSCMotherboardME3141") << "========================================================================" << std::endl;
      first = false;
      LogTrace("CSCMotherboardME3141") << "tmb_cross_bx_algo: " << tmb_cross_bx_algo << std::endl;
    }
    n++;
    if (debug_matching)
      LogTrace("CSCMotherboardME3141") << "LCT "<<n<<"  " << p <<std::endl;
  }
}

void CSCMotherboardME3141::correlateLCTs(const CSCALCTDigi& bALCT, const CSCALCTDigi& sALCT,
                                         const CSCCLCTDigi& bCLCT, const CSCCLCTDigi& sCLCT,
                                         CSCCorrelatedLCTDigi& lct1, CSCCorrelatedLCTDigi& lct2) const
{
  CSCALCTDigi bestALCT = bALCT;
  CSCALCTDigi secondALCT = sALCT;
  CSCCLCTDigi bestCLCT = bCLCT;
  CSCCLCTDigi secondCLCT = sCLCT;

  const bool anodeBestValid     = bestALCT.isValid();
  const bool anodeSecondValid   = secondALCT.isValid();
  const bool cathodeBestValid   = bestCLCT.isValid();
  const bool cathodeSecondValid = secondCLCT.isValid();

  if (anodeBestValid and !anodeSecondValid)     secondALCT = bestALCT;
  if (!anodeBestValid and anodeSecondValid)     bestALCT   = secondALCT;
  if (cathodeBestValid and !cathodeSecondValid) secondCLCT = bestCLCT;
  if (!cathodeBestValid and cathodeSecondValid) bestCLCT   = secondCLCT;

  // ALCT-CLCT matching conditions are defined by "trig_enable" configuration
  // parameters.
  if ((alct_trig_enable  and bestALCT.isValid()) or
      (clct_trig_enable  and bestCLCT.isValid()) or
      (match_trig_enable and bestALCT.isValid() and bestCLCT.isValid())){
    lct1 = constructLCTs(bestALCT, bestCLCT, CSCCorrelatedLCTDigi::ALCTCLCT, 1);
  }

  if (((secondALCT != bestALCT) or (secondCLCT != bestCLCT)) and
      ((alct_trig_enable  and secondALCT.isValid()) or
       (clct_trig_enable  and secondCLCT.isValid()) or
       (match_trig_enable and secondALCT.isValid() and secondCLCT.isValid()))){
    lct2 = constructLCTs(secondALCT, secondCLCT, CSCCorrelatedLCTDigi::ALCTCLCT, 2);
  }
}

//readout LCTs
std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME3141::readoutLCTs() const
{
  std::vector<CSCCorrelatedLCTDigi> result;
  allLCTs.getMatched(result);
  if (tmb_cross_bx_algo == 2) CSCUpgradeMotherboard::sortLCTs(result, CSCUpgradeMotherboard::sortLCTsByQuality);
  return result;
}
