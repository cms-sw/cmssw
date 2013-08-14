/*
 * \file L1TGCT.cc
 *
 * $Date: 2012/04/04 09:56:36 $
 * $Revision: 1.57 $
 * \author J. Berryhill
 *
 * $Log: L1TGCT.cc,v $
 * Revision 1.57  2012/04/04 09:56:36  ghete
 * Clean up L1TDEMON, add TriggerType hist to RCT, GCT, enable correlation condition tests in GT, clean up HCAL files.
 *
 * Revision 1.56  2012/03/29 21:16:48  rovere
 * Removed all instances of hltTriggerTypeFilter from L1T DQM Code.
 *
 * Revision 1.55  2010/06/28 09:29:30  tapper
 * Reduced number of bins.
 *
 * Revision 1.54  2010/06/28 06:40:46  tapper
 * Reduced numbers of bins in correlation plots (MET vs MHT and SumET vs HT).
 *
 * Revision 1.53  2010/06/14 20:38:45  tapper
 * Fixed stupid bug in MET vs MHT phi correlation.
 *
 * Revision 1.52  2010/06/09 14:39:27  tapper
 * Fixed labels and binning again.
 *
 * Revision 1.51  2010/06/09 14:03:04  tapper
 * Fixed histogram titles and binning in projections.
 *
 * Revision 1.50  2010/05/30 10:01:59  tapper
 * Added one histogram, correlation of sum ET and HT and changed a few labels for the better.
 *
 * Revision 1.49  2010/04/30 12:50:22  tapper
 * Fixed number of bins and range for MHT phi.
 *
 * Revision 1.48  2010/04/05 11:34:58  tapper
 * Changed scales on 2D HF correlation plots. No idea why they had eta phi scales when they only have 3 bits....
 *
 * Revision 1.47  2010/04/02 16:32:42  tapper
 * 1. Changed GCT unpacker settings to unpack 5 BXs.
 * 2. Changed L1TGCT to plot only central BX distributions but all 5 BXs for timing plots.
 * 3. Made labels more descriptive in GCT emulator expert DQM.
 *
 * Revision 1.46  2009/11/19 14:39:15  puigh
 * modify beginJob
 *
 * Revision 1.45  2009/11/02 22:30:27  tapper
 * Err that'll teach me to test it properly.... fixed a bug in the HF ring histograms.
 *
 * Revision 1.44  2009/11/02 17:00:05  tapper
 * Changes to L1TdeGCT (to include energy sums), to L1TDEMON (should not make any difference now) and L1TGCT to add multiple BXs.
 *
 * Revision 1.43  2009/07/22 19:40:24  puigh
 * Update binning to reflect instrumentation
 *
 * Revision 1.42  2009/06/23 09:48:55  tapper
 * Added missing occupancy plot for central and forward jets.
 *
 * Revision 1.41  2009/06/22 15:58:20  tapper
 * Added MET vs MHT correlation plots (both for magnitude and phi). Still untested!
 *
 * Revision 1.39  2009/05/27 21:49:26  jad
 * updated Total and Missing Energy histograms and added Overlow plots
 *
 * Revision 1.38  2009/02/24 13:01:42  jad
 * Updated MET_PHI histogram to obey the correct limits
 *
 * Revision 1.37  2008/11/11 13:20:32  tapper
 * A whole list of house keeping:
 * 1. New shifter histogram with central and forward jets together.
 * 2. Relabelled Ring 0 and Ring 1 to Ring 1 and Ring 2 for HF rings.
 * 3. Tidied up some histograms names to make all consistent.
 * 4. Switched eta and phi in 2D plots to match RCT.
 * 5. Removed 1D eta and phi plots. Will not be needed for Qtests in future.
 *
 * Revision 1.36  2008/10/28 14:16:16  tapper
 * Tidied up and removed some unnecessary code.
 *
 * Revision 1.35  2008/10/24 08:38:54  jbrooke
 * fix empty jet plots
 *
 * Revision 1.34  2008/10/10 12:41:24  jbrooke
 * put back checks on energy sum vector size, change [] to .at()
 *
 * Revision 1.33  2008/09/21 14:37:51  jad
 * updated HF Sums & Counts and added individual Jet Candidates and differences
 *
 * Revision 1.30  2008/06/09 11:07:52  tapper
 * Removed electron sub-folders with histograms per eta and phi bin.
 *
 * Revision 1.29  2008/06/06 15:18:22  tapper
 * Removed errorSummary folder stuff.
 *
 * Revision 1.28  2008/06/02 11:08:58  tapper
 * Added HF ring histograms....
 *
 * Revision 1.27  2008/05/12 12:52:46  tapper
 * Fixed problem when no GCT data in the event.
 *
 * Revision 1.26  2008/05/09 16:42:27  ameyer
 * *** empty log message ***
 *
 * Revision 1.25  2008/04/29 15:24:49  tapper
 * Changed path to summary histograms.
 *
 * Revision 1.24  2008/04/28 09:23:07  tapper
 * Added 1D eta and phi histograms for electrons and jets as input to Q tests.
 *
 * Revision 1.23  2008/04/25 15:40:21  tapper
 * Added histograms to EventInfo//errorSummarySegments.
 *
 * Revision 1.22  2008/03/20 19:38:25  berryhil
 *
 *
 * organized message logger
 *
 * Revision 1.21  2008/03/14 20:35:46  berryhil
 *
 *
 * stripped out obsolete parameter settings
 *
 * rpc tpg restored with correct dn access and dbe handling
 *
 * Revision 1.20  2008/03/12 17:24:24  berryhil
 *
 *
 * eliminated log files, truncated HCALTPGXana histo output
 *
 * Revision 1.19  2008/03/01 00:40:00  lat
 * DQM core migration.
 *
 * Revision 1.18  2008/02/20 18:59:29  tapper
 * Ported GCTMonitor histograms into L1TGCT
 *
 * Revision 1.17  2008/01/22 18:56:02  muzaffar
 * include cleanup. Only for cc/cpp files
 *
 * Revision 1.16  2007/12/21 17:41:20  berryhil
 *
 *
 * try/catch removal
 *
 * Revision 1.15  2007/11/19 15:08:22  lorenzo
 * changed top folder name
 *
 * Revision 1.14  2007/09/27 23:01:28  ratnik
 * QA campaign: fixes to compensate includes cleanup in  DataFormats/L1Trigger
 *
 * Revision 1.13  2007/09/27 16:56:26  wittich
 * verbosity fixes
 *
 * Revision 1.12  2007/09/26 15:26:23  berryhil
 *
 *
 * restored L1TGCT.cc
 *
 * Revision 1.10  2007/09/05 22:31:36  wittich
 * - Factorize getByLabels to approximate my understanding of what the
 *   HW can do.
 * - tested (loosely speaking) on GREJ' data.
 *
 * Revision 1.9  2007/09/04 02:54:19  wittich
 * - fix dupe ME in RCT
 * - put in rank>0 req in GCT
 * - various small other fixes
 *
 * Revision 1.8  2007/08/31 18:14:21  wittich
 * update GCT packages to reflect GctRawToDigi, and move to raw plots
 *
 * Revision 1.7  2007/08/31 11:02:56  wittich
 * cerr -> LogInfo
 *
 * Revision 1.6  2007/02/22 19:43:53  berryhil
 *
 *
 *
 * InputTag parameters added for all modules
 *
 *
 *
 */

#include "DQM/L1TMonitor/interface/L1TGCT.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"

// Trigger Headers

// GCT and RCT data formats
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DQMServices/Core/interface/DQMStore.h"

using namespace edm;

// Define statics for bins etc.
const unsigned int JETETABINS = 22;
const float JETETAMIN = -0.5;
const float JETETAMAX = 21.5;

const unsigned int EMETABINS = 22;
const float EMETAMIN = -0.5;
const float EMETAMAX = 21.5;

const unsigned int METPHIBINS = 72;
const float METPHIMIN = -0.5;
const float METPHIMAX = 71.5;

const unsigned int MHTPHIBINS = 18;
const float MHTPHIMIN = -0.5;
const float MHTPHIMAX = 17.5;

const unsigned int PHIBINS = 18;
const float PHIMIN = -0.5;
const float PHIMAX = 17.5;

const unsigned int OFBINS = 2;
const float OFMIN = -0.5;
const float OFMAX = 1.5;

const unsigned int BXBINS = 5;
const float BXMIN = -2.5;
const float BXMAX = 2.5;

// Bins for 3, 5, 6, 7, 10 and 12 bits
const unsigned int R3BINS = 8;
const float R3MIN = -0.5;
const float R3MAX = 7.5;
const unsigned int R5BINS = 32;
const float R5MIN = -0.5;
const float R5MAX = 31.5;
const unsigned int R6BINS = 64;
const float R6MIN = -0.5;
const float R6MAX = 63.5;
const unsigned int R7BINS = 128;
const float R7MIN = -0.5;
const float R7MAX = 127.5;
const unsigned int R10BINS = 1024;
const float R10MIN = -0.5;
const float R10MAX = 1023.5;
const unsigned int R12BINS = 4096;
const float R12MIN = -0.5;
const float R12MAX = 4095.5;

L1TGCT::L1TGCT(const edm::ParameterSet & ps) :
  gctCenJetsSource_(ps.getParameter<edm::InputTag>("gctCentralJetsSource")),
  gctForJetsSource_(ps.getParameter<edm::InputTag>("gctForwardJetsSource")),
  gctTauJetsSource_(ps.getParameter<edm::InputTag>("gctTauJetsSource")),
  gctEnergySumsSource_(ps.getParameter<edm::InputTag>("gctEnergySumsSource")),
  gctIsoEmSource_(ps.getParameter<edm::InputTag>("gctIsoEmSource")),
  gctNonIsoEmSource_(ps.getParameter<edm::InputTag>("gctNonIsoEmSource")),
  filterTriggerType_ (ps.getParameter< int >("filterTriggerType"))
{

  // verbosity switch
  verbose_ = ps.getUntrackedParameter < bool > ("verbose", false);

  if (verbose_)
    edm::LogInfo("L1TGCT") << "L1TGCT: constructor...." << std::endl;


  dbe = NULL;
  if (ps.getUntrackedParameter < bool > ("DQMStore", false)) {
    dbe = edm::Service < DQMStore > ().operator->();
    dbe->setVerbose(0);
  }

  outputFile_ = ps.getUntrackedParameter < std::string > ("outputFile", "");
  if (outputFile_.size() != 0) {
    edm::LogInfo("L1TGCT") << "L1T Monitoring histograms will be saved to "
                           << outputFile_ << std::endl;
  }

  bool disable = ps.getUntrackedParameter<bool>("disableROOToutput", false);
  if(disable){
    outputFile_="";
  }


  if (dbe != NULL) {
    dbe->setCurrentFolder("L1T/L1TGCT");
  }

}

L1TGCT::~L1TGCT()
{
}

void L1TGCT::beginJob(void)
{

  nev_ = 0;

  // get hold of back-end interface
  DQMStore *dbe = 0;
  dbe = edm::Service < DQMStore > ().operator->();

  if (dbe) {
    dbe->setCurrentFolder("L1T/L1TGCT");
    dbe->rmdir("L1T/L1TGCT");
  }


  if (dbe) {

    dbe->setCurrentFolder("L1T/L1TGCT");

    triggerType_ =
      dbe->book1D("TriggerType", "TriggerType", 17, -0.5, 16.5);

    l1GctAllJetsEtEtaPhi_ = dbe->book2D("AllJetsEtEtaPhi", "CENTRAL AND FORWARD JET E_{T}",
					JETETABINS, JETETAMIN, JETETAMAX,
                                        PHIBINS, PHIMIN, PHIMAX);
    l1GctCenJetsEtEtaPhi_ = dbe->book2D("CenJetsEtEtaPhi", "CENTRAL JET E_{T}",
					JETETABINS, JETETAMIN, JETETAMAX,
                                        PHIBINS, PHIMIN, PHIMAX); 
    l1GctForJetsEtEtaPhi_ = dbe->book2D("ForJetsEtEtaPhi", "FORWARD JET E_{T}",
					JETETABINS, JETETAMIN, JETETAMAX,
					PHIBINS, PHIMIN, PHIMAX); 
    l1GctTauJetsEtEtaPhi_ = dbe->book2D("TauJetsEtEtaPhi", "TAU JET E_{T}", 
					EMETABINS, EMETAMIN, EMETAMAX,
					PHIBINS, PHIMIN, PHIMAX); 
    l1GctIsoEmRankEtaPhi_ = dbe->book2D("IsoEmRankEtaPhi", "ISO EM E_{T}", 
					EMETABINS, EMETAMIN, EMETAMAX,
                                        PHIBINS, PHIMIN, PHIMAX); 		    
    l1GctNonIsoEmRankEtaPhi_ = dbe->book2D("NonIsoEmRankEtaPhi", "NON-ISO EM E_{T}",
                                           EMETABINS, EMETAMIN, EMETAMAX,
                                           PHIBINS, PHIMIN, PHIMAX); 
    l1GctAllJetsOccEtaPhi_ = dbe->book2D("AllJetsOccEtaPhi", "CENTRAL AND FORWARD JET OCCUPANCY",
					JETETABINS, JETETAMIN, JETETAMAX,
                                        PHIBINS, PHIMIN, PHIMAX);
    l1GctCenJetsOccEtaPhi_ = dbe->book2D("CenJetsOccEtaPhi", "CENTRAL JET OCCUPANCY",
					 JETETABINS, JETETAMIN, JETETAMAX,
                                         PHIBINS, PHIMIN, PHIMAX); 
    l1GctForJetsOccEtaPhi_ = dbe->book2D("ForJetsOccEtaPhi", "FORWARD JET OCCUPANCY",
					 JETETABINS, JETETAMIN, JETETAMAX,
					 PHIBINS, PHIMIN, PHIMAX);
    l1GctTauJetsOccEtaPhi_ = dbe->book2D("TauJetsOccEtaPhi", "TAU JET OCCUPANCY",
                                         EMETABINS, EMETAMIN, EMETAMAX,
					 PHIBINS, PHIMIN, PHIMAX); 
    l1GctIsoEmOccEtaPhi_ = dbe->book2D("IsoEmOccEtaPhi", "ISO EM OCCUPANCY",
                                       EMETABINS, EMETAMIN, EMETAMAX,
				       PHIBINS, PHIMIN, PHIMAX); 
    l1GctNonIsoEmOccEtaPhi_ = dbe->book2D("NonIsoEmOccEtaPhi", "NON-ISO EM OCCUPANCY",
                                          EMETABINS, EMETAMIN, EMETAMAX,
					  PHIBINS, PHIMIN, PHIMAX); 
  
    l1GctHFRing1PosEtaNegEta_ = dbe->book2D("HFRing1Corr", "HF RING1 E_{T} CORRELATION +/-  #eta",
                                            R3BINS, R3MIN, R3MAX, R3BINS, R3MIN, R3MAX); 
    l1GctHFRing2PosEtaNegEta_ = dbe->book2D("HFRing2Corr", "HF RING2 E_{T} CORRELATION +/-  #eta",
                                            R3BINS, R3MIN, R3MAX, R3BINS, R3MIN, R3MAX);
    l1GctHFRing1TowerCountPosEtaNegEta_ = dbe->book2D("HFRing1TowerCountCorr", "HF RING1 TOWER COUNT CORRELATION +/-  #eta",
                                                      R3BINS, R3MIN, R3MAX, R3BINS, R3MIN, R3MAX);
    l1GctHFRing2TowerCountPosEtaNegEta_ = dbe->book2D("HFRing2TowerCountCorr", "HF RING2 TOWER COUNT CORRELATION +/-  #eta",
                                                      R3BINS, R3MIN, R3MAX, R3BINS, R3MIN, R3MAX);

    //HF Ring stuff
    l1GctHFRing1TowerCountPosEta_ = dbe->book1D("HFRing1TowerCountPosEta", "HF RING1 TOWER COUNT  #eta  +", R3BINS, R3MIN, R3MAX);
    l1GctHFRing1TowerCountNegEta_ = dbe->book1D("HFRing1TowerCountNegEta", "HF RING1 TOWER COUNT  #eta  -", R3BINS, R3MIN, R3MAX);
    l1GctHFRing2TowerCountPosEta_ = dbe->book1D("HFRing2TowerCountPosEta", "HF RING2 TOWER COUNT  #eta  +", R3BINS, R3MIN, R3MAX);
    l1GctHFRing2TowerCountNegEta_ = dbe->book1D("HFRing2TowerCountNegEta", "HF RING2 TOWER COUNT  #eta  -", R3BINS, R3MIN, R3MAX);

    l1GctHFRing1ETSumPosEta_ = dbe->book1D("HFRing1ETSumPosEta", "HF RING1 E_{T}  #eta  +", R3BINS, R3MIN, R3MAX);
    l1GctHFRing1ETSumNegEta_ = dbe->book1D("HFRing1ETSumNegEta", "HF RING1 E_{T}  #eta  -", R3BINS, R3MIN, R3MAX);
    l1GctHFRing2ETSumPosEta_ = dbe->book1D("HFRing2ETSumPosEta", "HF RING2 E_{T}  #eta  +", R3BINS, R3MIN, R3MAX);
    l1GctHFRing2ETSumNegEta_ = dbe->book1D("HFRing2ETSumNegEta", "HF RING2 E_{T}  #eta  -", R3BINS, R3MIN, R3MAX);
    l1GctHFRingRatioPosEta_  = dbe->book1D("HFRingRatioPosEta", "HF RING E_{T} RATIO  #eta  +", R5BINS, R5MIN, R5MAX);
    l1GctHFRingRatioNegEta_  = dbe->book1D("HFRingRatioNegEta", "HF RING E_{T} RATIO  #eta  -", R5BINS, R5MIN, R5MAX);

    l1GctHFRingTowerCountOccBx_ = dbe->book2D("HFRingTowerCountOccBx", "HF RING TOWER COUNT PER BX",BXBINS, BXMIN, BXMAX, R3BINS, R3MIN, R3MAX);
    l1GctHFRingETSumOccBx_ = dbe->book2D("HFRingETSumOccBx", "HF RING E_{T} PER BX",BXBINS, BXMIN, BXMAX, R3BINS, R3MIN, R3MAX);
    
    // Rank histograms
    l1GctCenJetsRank_  = dbe->book1D("CenJetsRank", "CENTRAL JET E_{T}", R6BINS, R6MIN, R6MAX);
    l1GctForJetsRank_  = dbe->book1D("ForJetsRank", "FORWARD JET E_{T}", R6BINS, R6MIN, R6MAX);
    l1GctTauJetsRank_  = dbe->book1D("TauJetsRank", "TAU JET E_{T}", R6BINS, R6MIN, R6MAX);
    l1GctIsoEmRank_    = dbe->book1D("IsoEmRank", "ISO EM E_{T}", R6BINS, R6MIN, R6MAX);
    l1GctNonIsoEmRank_ = dbe->book1D("NonIsoEmRank", "NON-ISO EM E_{T}", R6BINS, R6MIN, R6MAX);

    l1GctAllJetsOccRankBx_ = dbe->book2D("AllJetsOccRankBx","ALL JETS E_{T} PER BX",BXBINS,BXMIN,BXMAX,R6BINS,R6MIN,R6MAX);
    l1GctAllEmOccRankBx_   = dbe->book2D("AllEmOccRankBx","ALL EM E_{T} PER BX",BXBINS,BXMIN,BXMAX,R6BINS,R6MIN,R6MAX);

    // Energy sums
    l1GctEtMiss_    = dbe->book1D("EtMiss", "MET", R12BINS, R12MIN, R12MAX);
    l1GctEtMissPhi_ = dbe->book1D("EtMissPhi", "MET  #phi", METPHIBINS, METPHIMIN, METPHIMAX);
    l1GctEtMissOf_  = dbe->book1D("EtMissOf", "MET OVERFLOW", OFBINS, OFMIN, OFMAX);
    l1GctEtMissOccBx_ = dbe->book2D("EtMissOccBx","MET PER BX",BXBINS,BXMIN,BXMAX,R12BINS,R12MIN,R12MAX);
    l1GctHtMiss_    = dbe->book1D("HtMiss", "MHT", R7BINS, R7MIN, R7MAX);
    l1GctHtMissPhi_ = dbe->book1D("HtMissPhi", "MHT  #phi", MHTPHIBINS, MHTPHIMIN, MHTPHIMAX);
    l1GctHtMissOf_  = dbe->book1D("HtMissOf", "MHT OVERFLOW", OFBINS, OFMIN, OFMAX);
    l1GctHtMissOccBx_ = dbe->book2D("HtMissOccBx","MHT PER BX",BXBINS,BXMIN,BXMAX,R7BINS,R7MIN,R7MAX);
    l1GctEtMissHtMissCorr_ = dbe->book2D("EtMissHtMissCorr", "MET MHT CORRELATION",
                                         R6BINS, R12MIN, R12MAX,
                                         R6BINS, R7MIN, R7MAX); 
    l1GctEtMissHtMissCorrPhi_ = dbe->book2D("EtMissHtMissPhiCorr", "MET MHT  #phi  CORRELATION",
                                            METPHIBINS, METPHIMIN, METPHIMAX,
                                            MHTPHIBINS, MHTPHIMIN, MHTPHIMAX);
    l1GctEtTotal_   = dbe->book1D("EtTotal", "SUM E_{T}", R12BINS, R12MIN, R12MAX);
    l1GctEtTotalOf_ = dbe->book1D("EtTotalOf", "SUM E_{T} OVERFLOW", OFBINS, OFMIN, OFMAX);
    l1GctEtTotalOccBx_ = dbe->book2D("EtTotalOccBx","SUM E_{T} PER BX",BXBINS,BXMIN,BXMAX,R12BINS,R12MIN,R12MAX);
    l1GctEtHad_     = dbe->book1D("EtHad", "H_{T}", R12BINS, R12MIN, R12MAX);
    l1GctEtHadOf_   = dbe->book1D("EtHadOf", "H_{T} OVERFLOW", OFBINS, OFMIN, OFMAX);
    l1GctEtHadOccBx_ = dbe->book2D("EtHadOccBx","H_{T} PER BX",BXBINS,BXMIN,BXMAX,R12BINS,R12MIN,R12MAX);
    l1GctEtTotalEtHadCorr_ = dbe->book2D("EtTotalEtHadCorr", "Sum E_{T} H_{T} CORRELATION",
                                         R6BINS, R12MIN, R12MAX,
                                         R6BINS, R12MIN, R12MAX); 
  }

}


void L1TGCT::endJob(void)
{
  if (verbose_)
    edm::LogInfo("L1TGCT") << "L1TGCT: end job...." << std::endl;
  edm::LogInfo("EndJob") << "analyzed " << nev_ << " events";

  if (outputFile_.size() != 0 && dbe) {
    dbe->save(outputFile_);
  }

  return;
}

void L1TGCT::analyze(const edm::Event & e, const edm::EventSetup & c)
{
  nev_++;
  if (verbose_) {
    edm::LogInfo("L1TGCT") << "L1TGCT: analyze...." << std::endl;
  }

  
  // filter according trigger type
  //  enum ExperimentType {
  //        Undefined          =  0,
  //        PhysicsTrigger     =  1,
  //        CalibrationTrigger =  2,
  //        RandomTrigger      =  3,
  //        Reserved           =  4,
  //        TracedEvent        =  5,
  //        TestTrigger        =  6,
  //        ErrorTrigger       = 15

  // fill a histogram with the trigger type, for normalization fill also last bin
  // ErrorTrigger + 1
  double triggerType = static_cast<double> (e.experimentType()) + 0.001;
  double triggerTypeLast = static_cast<double> (edm::EventAuxiliary::ExperimentType::ErrorTrigger)
                          + 0.001;
  triggerType_->Fill(triggerType);
  triggerType_->Fill(triggerTypeLast + 1);

  // filter only if trigger type is greater than 0, negative values disable filtering
  if (filterTriggerType_ >= 0) {

      // now filter, for real data only
      if (e.isRealData()) {
          if (!(e.experimentType() == filterTriggerType_)) {

              edm::LogInfo("L1TGCT") << "\n Event of TriggerType "
                      << e.experimentType() << " rejected" << std::endl;
              return;

          }
      }

  }

  // Get all the collections
  edm::Handle < L1GctEmCandCollection > l1IsoEm;
  edm::Handle < L1GctEmCandCollection > l1NonIsoEm;
  edm::Handle < L1GctJetCandCollection > l1CenJets;
  edm::Handle < L1GctJetCandCollection > l1ForJets;
  edm::Handle < L1GctJetCandCollection > l1TauJets;
  edm::Handle < L1GctHFRingEtSumsCollection > l1HFSums; 
  edm::Handle < L1GctHFBitCountsCollection > l1HFCounts;
  edm::Handle < L1GctEtMissCollection >  l1EtMiss;
  edm::Handle < L1GctHtMissCollection >  l1HtMiss;
  edm::Handle < L1GctEtHadCollection >   l1EtHad;
  edm::Handle < L1GctEtTotalCollection > l1EtTotal;

  e.getByLabel(gctIsoEmSource_, l1IsoEm);
  e.getByLabel(gctNonIsoEmSource_, l1NonIsoEm);
  e.getByLabel(gctCenJetsSource_, l1CenJets);
  e.getByLabel(gctForJetsSource_, l1ForJets);
  e.getByLabel(gctTauJetsSource_, l1TauJets);
  e.getByLabel(gctEnergySumsSource_, l1HFSums);
  e.getByLabel(gctEnergySumsSource_, l1HFCounts);  
  e.getByLabel(gctEnergySumsSource_, l1EtMiss);
  e.getByLabel(gctEnergySumsSource_, l1HtMiss);
  e.getByLabel(gctEnergySumsSource_, l1EtHad);
  e.getByLabel(gctEnergySumsSource_, l1EtTotal);


  // Fill histograms

  // Central jets
  if (l1CenJets.isValid()) {
    for (L1GctJetCandCollection::const_iterator cj = l1CenJets->begin();cj != l1CenJets->end(); cj++) {
      // only plot central BX
      if (cj->bx()==0) {
        l1GctCenJetsRank_->Fill(cj->rank());
        // only plot eta and phi maps for non-zero candidates
        if (cj->rank()) {
          l1GctAllJetsEtEtaPhi_->Fill(cj->regionId().ieta(),cj->regionId().iphi(),cj->rank());
          l1GctAllJetsOccEtaPhi_->Fill(cj->regionId().ieta(),cj->regionId().iphi());
          l1GctCenJetsEtEtaPhi_->Fill(cj->regionId().ieta(),cj->regionId().iphi(),cj->rank());
          l1GctCenJetsOccEtaPhi_->Fill(cj->regionId().ieta(),cj->regionId().iphi());
        }
      }
      if (cj->rank()) l1GctAllJetsOccRankBx_->Fill(cj->bx(),cj->rank()); // for all BX
    }
  } else {    
    edm::LogWarning("DataNotFound") << " Could not find l1CenJets label was " << gctCenJetsSource_ ;
  }

  // Forward jets
  if (l1ForJets.isValid()) {
    for (L1GctJetCandCollection::const_iterator fj = l1ForJets->begin(); fj != l1ForJets->end(); fj++) {
      // only plot central BX
      if (fj->bx()==0) {
        l1GctForJetsRank_->Fill(fj->rank());
        // only plot eta and phi maps for non-zero candidates
        if (fj->rank()) {
          l1GctAllJetsEtEtaPhi_->Fill(fj->regionId().ieta(),fj->regionId().iphi(),fj->rank());
          l1GctAllJetsOccEtaPhi_->Fill(fj->regionId().ieta(),fj->regionId().iphi());
          l1GctForJetsEtEtaPhi_->Fill(fj->regionId().ieta(),fj->regionId().iphi(),fj->rank());
          l1GctForJetsOccEtaPhi_->Fill(fj->regionId().ieta(),fj->regionId().iphi());    
        }
      }
      if (fj->rank()) l1GctAllJetsOccRankBx_->Fill(fj->bx(),fj->rank()); // for all BX
    }
  } else {    
    edm::LogWarning("DataNotFound") << " Could not find l1ForJets label was " << gctForJetsSource_ ;
  }

  // Tau jets
  if (l1TauJets.isValid()) {
    for (L1GctJetCandCollection::const_iterator tj = l1TauJets->begin(); tj != l1TauJets->end(); tj++) {
      // only plot central BX
      if (tj->bx()==0) {
        l1GctTauJetsRank_->Fill(tj->rank());
        // only plot eta and phi maps for non-zero candidates
        if (tj->rank()) {
          l1GctTauJetsEtEtaPhi_->Fill(tj->regionId().ieta(),tj->regionId().iphi(),tj->rank());
          l1GctTauJetsOccEtaPhi_->Fill(tj->regionId().ieta(),tj->regionId().iphi());
        }
      }
      if (tj->rank()) l1GctAllJetsOccRankBx_->Fill(tj->bx(),tj->rank()); // for all BX
    }
  } else {    
    edm::LogWarning("DataNotFound") << " Could not find l1TauJets label was " << gctTauJetsSource_ ;
  }

  // Missing ET
  if (l1EtMiss.isValid()) { 
    for (L1GctEtMissCollection::const_iterator met = l1EtMiss->begin(); met != l1EtMiss->end(); met++) {
      // only plot central BX
      if (met->bx()==0) {
        if (met->overFlow() == 0 && met->et() > 0) {
          //Avoid problems with met=0 candidates affecting MET_PHI plots
          l1GctEtMiss_->Fill(met->et());
          l1GctEtMissPhi_->Fill(met->phi());
        }
        l1GctEtMissOf_->Fill(met->overFlow());
      }
      if (met->overFlow() == 0 && met->et() > 0) l1GctEtMissOccBx_->Fill(met->bx(),met->et()); // for all BX
    }
  } else {
    edm::LogWarning("DataNotFound") << " Could not find l1EtMiss label was " << gctEnergySumsSource_ ;    
  }

  // Missing HT
  if (l1HtMiss.isValid()) { 
    for (L1GctHtMissCollection::const_iterator mht = l1HtMiss->begin(); mht != l1HtMiss->end(); mht++) {
      // only plot central BX
      if (mht->bx()==0) {
        if (mht->overFlow() == 0 && mht->et() > 0) {
          //Avoid problems with mht=0 candidates affecting MHT_PHI plots
          l1GctHtMiss_->Fill(mht->et());
          l1GctHtMissPhi_->Fill(mht->phi());
        }
        l1GctHtMissOf_->Fill(mht->overFlow());
      }
      if (mht->overFlow() == 0 && mht->et() > 0) l1GctHtMissOccBx_->Fill(mht->bx(),mht->et()); // for all BX
    }
  } else {
    edm::LogWarning("DataNotFound") << " Could not find l1HtMiss label was " << gctEnergySumsSource_ ;    
  }

  // Missing ET HT correlations
  if (l1HtMiss.isValid() && l1EtMiss.isValid()) { 
    if (l1HtMiss->size() == l1EtMiss->size()) {
      for (unsigned i=0; i<l1HtMiss->size(); i++) {
        if (l1HtMiss->at(i).overFlow() == 0 && l1EtMiss->at(i).overFlow() == 0 && 
            l1HtMiss->at(i).bx() == 0 && l1EtMiss->at(i).bx() == 0) {
          // Avoid problems overflows and only plot central BX
          l1GctEtMissHtMissCorr_->Fill(l1EtMiss->at(i).et(),l1HtMiss->at(i).et());
          if (l1HtMiss->at(i).et() && l1EtMiss->at(i).et()){ // Don't plot phi if one or other is zero
            l1GctEtMissHtMissCorrPhi_->Fill(l1EtMiss->at(i).phi(),l1HtMiss->at(i).phi());
          }
        }
      }
    }
  } else {
    edm::LogWarning("DataNotFound") << " Could not find l1EtMiss or l1HtMiss label was " << gctEnergySumsSource_ ;    
  }

  // HT 
  if (l1EtHad.isValid()) {
    for (L1GctEtHadCollection::const_iterator ht = l1EtHad->begin(); ht != l1EtHad->end(); ht++) {
      // only plot central BX
      if (ht->bx()==0) {
        l1GctEtHad_->Fill(ht->et());
        l1GctEtHadOf_->Fill(ht->overFlow());
      }
      l1GctEtHadOccBx_->Fill(ht->bx(),ht->et()); // for all BX
    }
  } else {
    edm::LogWarning("DataNotFound") << " Could not find l1EtHad label was " << gctEnergySumsSource_ ;    
  }

  // Total ET
  if (l1EtTotal.isValid()) {
    for (L1GctEtTotalCollection::const_iterator et = l1EtTotal->begin(); et != l1EtTotal->end(); et++) {
      // only plot central BX
      if (et->bx()==0) {
        l1GctEtTotal_->Fill(et->et());
        l1GctEtTotalOf_->Fill(et->overFlow());
      }
      l1GctEtTotalOccBx_->Fill(et->bx(),et->et()); // for all BX
    }
  } else {
    edm::LogWarning("DataNotFound") << " Could not find l1EtTotal label was " << gctEnergySumsSource_ ;    
  }

  // Total ET HT correlations
  if (l1EtTotal.isValid() && l1EtHad.isValid()) { 
    if (l1EtTotal->size() == l1EtHad->size()) {
      for (unsigned i=0; i<l1EtHad->size(); i++) {
        if (l1EtHad->at(i).overFlow() == 0 && l1EtTotal->at(i).overFlow() == 0 && 
            l1EtHad->at(i).bx() == 0 && l1EtTotal->at(i).bx() == 0) {
          // Avoid problems overflows and only plot central BX
          l1GctEtTotalEtHadCorr_->Fill(l1EtTotal->at(i).et(),l1EtHad->at(i).et());
        }
      }
    }
  } else {
    edm::LogWarning("DataNotFound") << " Could not find l1EtTotal or l1EtHad label was " << gctEnergySumsSource_ ;    
  }

  //HF Ring Et Sums
  if (l1HFSums.isValid()) {
    for (L1GctHFRingEtSumsCollection::const_iterator hfs=l1HFSums->begin(); hfs!=l1HFSums->end(); hfs++){ 
      // only plot central BX
      if (hfs->bx()==0) {
        // Individual ring Et sums
        l1GctHFRing1ETSumPosEta_->Fill(hfs->etSum(0));
        l1GctHFRing1ETSumNegEta_->Fill(hfs->etSum(1));
        l1GctHFRing2ETSumPosEta_->Fill(hfs->etSum(2));
        l1GctHFRing2ETSumNegEta_->Fill(hfs->etSum(3));
        // Ratio of ring Et sums
        if (hfs->etSum(2)!=0) l1GctHFRingRatioPosEta_->Fill((hfs->etSum(0))/(hfs->etSum(2)));
        if (hfs->etSum(3)!=0) l1GctHFRingRatioNegEta_->Fill((hfs->etSum(1))/(hfs->etSum(3)));
        // Correlate positive and neagative eta
        l1GctHFRing1PosEtaNegEta_->Fill(hfs->etSum(0),hfs->etSum(1));
        l1GctHFRing2PosEtaNegEta_->Fill(hfs->etSum(2),hfs->etSum(3));
      }
      // Occupancy vs BX
      for (unsigned i=0; i<4; i++){
        l1GctHFRingETSumOccBx_->Fill(hfs->bx(),hfs->etSum(i));
      }
    }
  } else {    
    edm::LogWarning("DataNotFound") << " Could not find l1HFSums label was " << gctEnergySumsSource_ ;
  }

  // HF Ring Counts
  if (l1HFCounts.isValid()) {
    for (L1GctHFBitCountsCollection::const_iterator hfc=l1HFCounts->begin(); hfc!=l1HFCounts->end(); hfc++){ 
      // only plot central BX
      if (hfc->bx()==0) {
        // Individual ring counts
        l1GctHFRing1TowerCountPosEta_->Fill(hfc->bitCount(0));
        l1GctHFRing1TowerCountNegEta_->Fill(hfc->bitCount(1));
        l1GctHFRing2TowerCountPosEta_->Fill(hfc->bitCount(2));
        l1GctHFRing2TowerCountNegEta_->Fill(hfc->bitCount(3));
        // Correlate positive and negative eta
        l1GctHFRing1TowerCountPosEtaNegEta_->Fill(hfc->bitCount(0),hfc->bitCount(1));
        l1GctHFRing2TowerCountPosEtaNegEta_->Fill(hfc->bitCount(2),hfc->bitCount(3));
      }
      // Occupancy vs BX
      for (unsigned i=0; i<4; i++){
        l1GctHFRingTowerCountOccBx_->Fill(hfc->bx(),hfc->bitCount(i));
      }
    }
  } else {    
    edm::LogWarning("DataNotFound") << " Could not find l1HFCounts label was " << gctEnergySumsSource_ ;
  }
  
  // Isolated EM
  if (l1IsoEm.isValid()) {
    for (L1GctEmCandCollection::const_iterator ie=l1IsoEm->begin(); ie!=l1IsoEm->end(); ie++) {
      // only plot central BX
      if (ie->bx()==0) {
        l1GctIsoEmRank_->Fill(ie->rank());
        // only plot eta and phi maps for non-zero candidates
        if (ie->rank()){ 
          l1GctIsoEmRankEtaPhi_->Fill(ie->regionId().ieta(),ie->regionId().iphi(),ie->rank());
          l1GctIsoEmOccEtaPhi_->Fill(ie->regionId().ieta(),ie->regionId().iphi());
        }
      }
      if (ie->rank()) l1GctAllEmOccRankBx_->Fill(ie->bx(),ie->rank()); // for all BX
    }
  } else {
    edm::LogWarning("DataNotFound") << " Could not find l1IsoEm label was " << gctIsoEmSource_ ;
  } 

  // Non-isolated EM
  if (l1NonIsoEm.isValid()) { 
    for (L1GctEmCandCollection::const_iterator ne=l1NonIsoEm->begin(); ne!=l1NonIsoEm->end(); ne++) {
      // only plot central BX
      if (ne->bx()==0) {
        l1GctNonIsoEmRank_->Fill(ne->rank());
        // only plot eta and phi maps for non-zero candidates
        if (ne->rank()){ 
          l1GctNonIsoEmRankEtaPhi_->Fill(ne->regionId().ieta(),ne->regionId().iphi(),ne->rank());
          l1GctNonIsoEmOccEtaPhi_->Fill(ne->regionId().ieta(),ne->regionId().iphi());
        }
      }
      if (ne->rank()) l1GctAllEmOccRankBx_->Fill(ne->bx(),ne->rank()); // for all BX
    }
  } else {
    edm::LogWarning("DataNotFound") << " Could not find l1NonIsoEm label was " << gctNonIsoEmSource_ ;
  }     
}

  
