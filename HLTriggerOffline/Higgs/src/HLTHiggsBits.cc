/** \class HLTHiggsBits
 *
 * See header file for documentation
 *
 *  $Date: 2008/03/05 14:06:35 $
 *  $Revision: 1.0 $
 *
 *  \author Mika Huhtinen
 *
 */

#include "HLTriggerOffline/Higgs/interface/HLTHiggsBits.h"
#include "HLTriggerOffline/Higgs/interface/HLTHiggsTruth.h"


#include "FWCore/Framework/interface/TriggerNames.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerSetup.h"
//#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"

#include <iomanip>
#include <string>

//
// constructors and destructor
//
HLTHiggsBits::HLTHiggsBits(const edm::ParameterSet& iConfig) :
  hlTriggerResults_ (iConfig.getParameter<edm::InputTag> ("HLTriggerResults")),
  l1ParticleMapTag_ (iConfig.getParameter<edm::InputTag> ("L1ExtraParticleMap")),
  l1GTReadoutRecTag_(iConfig.getParameter<edm::InputTag> ("L1GTReadoutRecord")),
  l1GTObjectMapTag_(iConfig.getParameter<edm::InputTag> ("L1GTObjectMapTag")),
  mctruth_ (iConfig.getParameter<edm::InputTag> ("MCTruth")),
  n_channel_ (iConfig.getParameter<int>("Nchannel")),
  triggerNames_(),
  n_true_(0),
  n_fake_(0),
  n_miss_(0),
  n_inmc_(0),
  n_L1acc_(0),
  n_L1acc_mc_(0),
  n_hlt_of_L1_(0),
  nEvents_(0),
  hlNames_(0),
  init_(false),
  histName(iConfig.getParameter<string>("histName")),
  hlt_bitnames(iConfig.getParameter<std::vector<string> >("hltBitNames"))
{

  n_hlt_bits=hlt_bitnames.size();
  cout << "Number of bit names : " << n_hlt_bits << endl;
  if (n_hlt_bits>20) {
    cout << "TOO MANY BITS REQUESTED - TREATING ONLY FIRST 20" << endl;
    n_hlt_bits=20;
  }

// 1:H->ZZ->4l, 2:H->WW->2l, 3: H->gg, 4:qqh->2tau, 5:H+->taunu, 6:qqh->inv
// The proper channel number has to be set in the cff-file
  cout << "Analyzing Higgs channel number " << n_channel_ << endl;

  // open the histogram file
  m_file=0; // set to null
  m_file=new TFile((histName+".root").c_str(),"RECREATE");
  m_file->cd();
  outfile.open((histName+".output").c_str());

  // Initialize the tree
  HltTree = 0;
  HltTree = new TTree("HltTree","");

  for (int i=0;i<n_hlt_bits;i++) {
    for (int j=0;j<n_hlt_bits+1;j++) {
      hlt_whichbit[0][i][j]=0;
      hlt_whichbit[1][i][j]=0;
      hlt_whichbit[2][i][j]=0;
    }
  }


  mct_analysis_.setup(iConfig, HltTree);

  const int kMaxEvents = 50000;
  hlt_nbits = new int[kMaxEvents];
  HltTree->Branch("NEventCount",&neventcount,"NEventCount/I");
  HltTree->Branch("HLT_nBits",hlt_nbits,"HLT_nBits[NEventCount]/I");

  // book histrograms fro the L1 & HLT efficiency wrt GEN and wrt GEN+MC-preselection
  trg_eff_gen = new TH1D("TRG_eff_gen","Trigger-eff-wrt-gen",3,0.5,3.5);
  trg_eff_gen ->GetXaxis() -> SetBinLabel(1,"Generated");
  trg_eff_gen ->GetXaxis() -> SetBinLabel(2,"L1-selected");
  trg_eff_gen ->GetXaxis() -> SetBinLabel(3,"HLT-selected");
  trg_eff_gen_mc = new TH1D("TRG_eff_gen_mc","Trigger-eff-wrt-MC-preselect",4,0.5,4.5);
  trg_eff_gen_mc ->GetXaxis() -> SetBinLabel(1,"Generated");
  trg_eff_gen_mc ->GetXaxis() -> SetBinLabel(2,"MC preselection");
  trg_eff_gen_mc ->GetXaxis() -> SetBinLabel(3,"L1-selected");
  trg_eff_gen_mc ->GetXaxis() -> SetBinLabel(4,"HLT-selected");
  trg_eff_gen -> SetStats(false);
  trg_eff_gen_mc -> SetStats(false);

  // book histograms for bit multiplicity, i.e. by how many paths an event is taken
  hlt_mult_hist = new TH1D("HLT-multiplicity","HLT-bit-multiplicity-of-GEN",n_hlt_bits+1,-0.5,n_hlt_bits+0.5);
  hlt_mult_hist_mc = new TH1D("HLT-multiplicity-mc","HLT-bit-multiplicity-of-MCacc",n_hlt_bits+1,-0.5,n_hlt_bits+0.5);
  hlt_mult_hist_l1 = new TH1D("HLT-multiplicity-l1","HLT-bit-multiplicity-of-L1acc",n_hlt_bits+1,-0.5,n_hlt_bits+0.5);

  //book histograms for individual bits, i.e. how often a given path is fired
  hlt_bit_hist = new TH1D("HLT-fired-bits","HLT-bits-wrt-GEN",n_hlt_bits,0.5,n_hlt_bits+0.5);
  hlt_bit_hist_mc = new TH1D("HLT-fired-bits-mc","HLT-bits-wrt-MCacc",n_hlt_bits,0.5,n_hlt_bits+0.5);
  hlt_bit_hist_l1 = new TH1D("HLT-fired-bits-l1","HLT-bits-wrt-L1acc",n_hlt_bits,0.5,n_hlt_bits+0.5);

  // book a cumulative distribution of paths - result (except last bin) will 
  // depend on order of path-names, as given in the cff !
  hlt_bit_cumul = new TH1D("HLT-cumulative","HLT-cumulative-efficiency-wrt-GEN",n_hlt_bits,0.5,n_hlt_bits+0.5);
  hlt_bit_cumul_mc = new TH1D("HLT-cumulative-mc","HLT-cumulative-efficiency-wrt-MCacc",n_hlt_bits,0.5,n_hlt_bits+0.5);
  hlt_bit_cumul_l1 = new TH1D("HLT-cumulative-l1","HLT-cumulative-efficiency-wrt-L1acc",n_hlt_bits,0.5,n_hlt_bits+0.5);
  for (int k=0;k<n_hlt_bits; k++) {
    hlt_bit_hist -> GetXaxis() -> SetBinLabel(k+1,hlt_bitnames[k].c_str());
    hlt_bit_hist_mc -> GetXaxis() -> SetBinLabel(k+1,hlt_bitnames[k].c_str());
    hlt_bit_hist_l1 -> GetXaxis() -> SetBinLabel(k+1,hlt_bitnames[k].c_str());
    hlt_bit_cumul -> GetXaxis() -> SetBinLabel(k+1,hlt_bitnames[k].c_str());
    hlt_bit_cumul_mc -> GetXaxis() -> SetBinLabel(k+1,hlt_bitnames[k].c_str());
    hlt_bit_cumul_l1 -> GetXaxis() -> SetBinLabel(k+1,hlt_bitnames[k].c_str());
  }
  for (int j=0;j<n_hlt_bits;j++) {
    string histname = "HLT-redundancy-for-"+hlt_bitnames[j];
    hlt_redundancy[j] = new TH1D(histname.c_str(),(hlt_bitnames[j]+"-redundancy").c_str(),n_hlt_bits,0.5,n_hlt_bits+0.5);
    hlt_redundancy_mc[j] = new TH1D((histname+"-of-MCacc").c_str(),(hlt_bitnames[j]+"-redundancy-MCsel").c_str(),n_hlt_bits,0.5,n_hlt_bits+0.5);
    hlt_redundancy_l1[j] = new TH1D((histname+"-of-L1acc").c_str(),(hlt_bitnames[j]+"-redundancy-L1acc").c_str(),n_hlt_bits,0.5,n_hlt_bits+0.5);
    hlt_redundancy[j] -> SetStats(false);
    hlt_redundancy_mc[j] -> SetStats(false);
    hlt_redundancy_l1[j] -> SetStats(false);
  }
  // the stats make no sense for these histograms
  hlt_mult_hist -> SetStats(false);
  hlt_mult_hist_mc -> SetStats(false);
  hlt_mult_hist_l1 -> SetStats(false);
  hlt_bit_hist -> SetStats(false);
  hlt_bit_hist_mc -> SetStats(false);
  hlt_bit_hist_l1 -> SetStats(false);
  hlt_bit_cumul -> SetStats(false);
  hlt_bit_cumul_mc -> SetStats(false);
  hlt_bit_cumul_l1 -> SetStats(false);
  cout << "booking OK " << endl;

}

HLTHiggsBits::~HLTHiggsBits()
{ }

//
// member functions
//


// ------------ method called to produce the data  ------------
void
HLTHiggsBits::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // accumulation of statistics for HLT bits used by Higgs analysis 

  using namespace std;
  using namespace edm;

  nEvents_++;


// MC truth part

  string errMsg("");
  edm::Handle<CandidateCollection> mctruth;
  try {iEvent.getByLabel(mctruth_,mctruth);} catch (...) { errMsg=errMsg + "  -- No Gen Particles";}

  // do the MC-preselection. This depends on the channel under study. with
  // wrong n_channel the result would be nonsense
  if (n_channel_== 1) {
    mct_analysis_.analyzeHZZ4l(*mctruth, HltTree);
  } else if (n_channel_ == 2) {
    mct_analysis_.analyzeHWW2l(*mctruth, HltTree);
  } else if (n_channel_ == 3) {
    mct_analysis_.analyzeHgg(*mctruth, HltTree);
  } else if (n_channel_ == 4) {
    mct_analysis_.analyzeH2tau(*mctruth, HltTree);
  } else if (n_channel_ == 5) {
    mct_analysis_.analyzeHtaunu(*mctruth, HltTree);
  } else if (n_channel_ == 6) {
    mct_analysis_.analyzeHinv(*mctruth, HltTree);
  }



// L1 part



//  const unsigned int nl1(l1extra::L1ParticleMap::kNumOfL1TriggerTypes);

  getL1Names(iEvent, iSetup);

  // get hold of L1GlobalReadoutRecord
  Handle<L1GlobalTriggerReadoutRecord> L1GTRR;
  try {iEvent.getByLabel(l1GTReadoutRecTag_,L1GTRR);} catch (...) {
    cout << "L1 Information not found" << endl;}
  if (L1GTRR.isValid()) {
    l1_decision = L1GTRR->decision();
//  the following code extracts the L1-trigger name, but for the time
//  being we only look at the overall decision.
//  code below is tested to work in 1_7_5
/*  code to get the L1 bits one by one - not used at the moment
    DecisionWord gtDecisionWord = L1GTRR->decisionWord();
    LogDebug("") << "L1GlobalTriggerReadoutRecord decision: " << l1_decision;
    cout << "L1GlobalTriggerReadoutRecord decision: " << l1_decision << endl;
    if (l1_decision) ++nL1Accepts_;
    const unsigned int numberL1Bits = L1GlobalTriggerReadoutSetup::NumberPhysTriggers;
    for (int k=0; k<numberL1Bits; k++) {
      cout << "decision word = " << gtDecisionWord[k] << " " << algoBitToName[k] << endl;
    }
  } else {
    LogDebug("") << "L1GlobalTriggerReadoutRecord with label ["+l1GTReadoutRecTag_.encode()+"] not found!";
    cout << "L1GlobalTriggerReadoutRecord not found " << endl;
    nErrors_++;
    return;
*/
  }

  if (l1_decision) {
    n_L1acc_++;
    if (mct_analysis_.decision()) n_L1acc_mc_++;
  }

// HLT part

  // get hold of HL TriggerResults
  try {iEvent.getByLabel(hlTriggerResults_,HLTR);} catch (...) {;}
  if (!HLTR.isValid()) {
    LogDebug("") << "HL TriggerResults with label ["+hlTriggerResults_.encode()+"] not found!";
    return;
  }

  // initialisation
  if (!init_) {
    init_=true;
    triggerNames_.init(*HLTR);
    hlNames_=triggerNames_.triggerNames();
  }


  if (mct_analysis_.decision()) n_inmc_++;
  // decision for each HL algorithm
  const unsigned int n(hlNames_.size());
  // HLT_fired counts the number of paths that have fired
  int HLT_fired=0;
  // wtrig if set to 1 for paths that have fired
  int wtrig[100]={0};
  for (unsigned int i=0; i!=n; ++i) {
    if (HLTR->accept(i)) {
      for (int j=0;j<n_hlt_bits;j++) {
        if (hlNames_[i] == hlt_bitnames[j]) {
          HLT_fired++;
	  wtrig[j]=1;
        }
      }
    }
  }
  if (HLT_fired > 0) {
    if (l1_decision) n_hlt_of_L1_++;
    // HLT takes this - see if it has passed the MC-preselection
    if (mct_analysis_.decision()) {
      n_true_++;
    } else {
      n_fake_++;
    }
    std::cout << "Event " << nEvents_ << " taken by " << HLT_fired << " triggers" << std::endl;
  } else if (mct_analysis_.decision()) {
    // we get here if HLT did not take an event that has passed MC-preselection
    n_miss_++;
  }
  hlt_nbits[nEvents_-1]=HLT_fired;
  hlt_mult_hist->Fill(HLT_fired);
  if (mct_analysis_.decision()) hlt_mult_hist_mc->Fill(HLT_fired);
  if (l1_decision) hlt_mult_hist_l1->Fill(HLT_fired);

  // the istaken*** flags are used to fill the cumulative histograms, i.e.
  // istaken=true as soon as any path has taken the event and stays true
  // thereafter
  bool istaken=false;
  bool istaken_mc=false;
  bool istaken_l1=false;
  for (int j=0;j<n_hlt_bits;j++) {
    if (wtrig[j]==1) {
      hlt_redundancy[j]->Fill(HLT_fired);
      hlt_bit_hist->Fill(j+1);
      istaken=true;
    } 
    if (istaken) hlt_bit_cumul->Fill(j+1);
    hlt_whichbit[0][j][HLT_fired]=hlt_whichbit[0][j][HLT_fired]+wtrig[j];
    hlt_whichbit[0][j][0]=hlt_whichbit[0][j][0]+wtrig[j];
    if (mct_analysis_.decision()) {
      if (wtrig[j]==1) {
        hlt_redundancy_mc[j]->Fill(HLT_fired);
        hlt_bit_hist_mc->Fill(j+1);
        istaken_mc=true;
      } 
      if (istaken_mc) hlt_bit_cumul_mc->Fill(j+1);
      hlt_whichbit[1][j][HLT_fired]=hlt_whichbit[1][j][HLT_fired]+wtrig[j];
      hlt_whichbit[1][j][0]=hlt_whichbit[1][j][0]+wtrig[j];
    }
    if (l1_decision) {
      if (wtrig[j]==1) {
        hlt_redundancy_l1[j]->Fill(HLT_fired);
        hlt_bit_hist_l1->Fill(j+1);
        istaken_l1=true;
      } 
      if (istaken_l1) hlt_bit_cumul_l1->Fill(j+1);
      hlt_whichbit[2][j][HLT_fired]=hlt_whichbit[2][j][HLT_fired]+wtrig[j];
      hlt_whichbit[2][j][0]=hlt_whichbit[2][j][0]+wtrig[j];
    }
  }
  neventcount=nEvents_;

  return;

}


void
HLTHiggsBits::getL1Names(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord;
  iEvent.getByLabel(l1GTObjectMapTag_.label(), gtObjectMapRecord);

  const std::vector<L1GlobalTriggerObjectMap>& objMapVec =
       gtObjectMapRecord->gtObjectMap();

  for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itMap = objMapVec.begin();
       itMap != objMapVec.end(); ++itMap) {
    int algoBit = (*itMap).algoBitNumber();
    std::string algoNameStr = (*itMap).algoName();
    algoBitToName[algoBit] = algoNameStr;
  }
}


void
HLTHiggsBits::endJob()
{
  // final printout of accumulated statistics

  cout << "Job ending " << endl;

  using namespace std;

  cout << "Number of events handled:                      " << nEvents_ << endl;
  cout << "Number of events seen in MC:                   " << n_inmc_ << ", (" << 100.0*n_inmc_/nEvents_ <<"%)" << endl;
  cout << "Number of events taken by L1:                  " << n_L1acc_ << ", (" << 100.0*n_L1acc_/nEvents_ << "%)" << endl;
  cout << "Number of events taken by L1, seen in MC:      " << n_L1acc_mc_ << ", (" << 100.0*n_L1acc_mc_/n_inmc_ << "%)" << endl;
  cout << "Number of events taken by HLT :                " << n_fake_+n_true_ << ", (" << 100.0*(n_true_+n_fake_)/nEvents_ <<"%)" << endl;
  cout << "Number of correct (matched to MC) HL triggers: " << n_true_ << ", (" << 100.0*n_true_/n_inmc_ <<"%)" << endl;
  cout << "HLT acceptance wrt L1                          " << n_hlt_of_L1_ << ", (" << 100.0*n_hlt_of_L1_/n_L1acc_ << "%)" << endl;
  cout << "Number of missed (seen in MC) HL triggers:     " << n_miss_ << endl;
  cout << "Number of wrong (no MC match) HL triggers:     " << n_fake_ << endl;

  outfile << "Number of events handled:                      " << nEvents_ << endl;
  outfile << "Number of events seen in MC:                   " << n_inmc_ << ", (" << 100.0*n_inmc_/nEvents_ <<"%)" << endl;
  outfile << "Number of events taken by L1:                  " << n_L1acc_ << ", (" << 100.0*n_L1acc_/nEvents_ << "%)" << endl;
  outfile << "Number of events taken by L1, seen in MC:      " << n_L1acc_mc_ << ", (" << 100.0*n_L1acc_mc_/n_inmc_ << "%)" << endl;
  outfile << "Number of events taken by HLT :                " << n_fake_+n_true_ << ", (" << 100.0*(n_true_+n_fake_)/nEvents_ <<"%)" << endl;
  outfile << "Number of correct (matched to MC) HL triggers: " << n_true_ << ", (" << 100.0*n_true_/n_inmc_ <<"%)" << endl;
  outfile << "HLT acceptance wrt L1                          " << n_hlt_of_L1_ << ", (" << 100.0*n_hlt_of_L1_/n_L1acc_ << "%)" << endl;
  outfile << "Number of missed (seen in MC) HL triggers:     " << n_miss_ << endl;
  outfile << "Number of wrong (no MC match) HL triggers:     " << n_fake_ << endl;

  trg_eff_gen -> Fill(1,nEvents_);
  trg_eff_gen -> Fill(2,n_L1acc_);
  trg_eff_gen -> Fill(3,n_hlt_of_L1_);

  trg_eff_gen_mc -> Fill(1,nEvents_);
  trg_eff_gen_mc -> Fill(2,n_inmc_);
  trg_eff_gen_mc -> Fill(3,n_L1acc_mc_);
  trg_eff_gen_mc -> Fill(4,n_true_);

  cout << "===== Events accepted by HLT (of all generated) =======" << endl;
  for (int i=0;i<n_hlt_bits;i++) {
    for (int j=1;j<n_hlt_bits+1;j++) {
      cout << hlt_whichbit[0][i][j] << ", ";
    }
    cout << hlt_whichbit[0][i][0] << ", " << hlt_bitnames[i] << endl;
  }
  cout << "===== Events accepted by HLT (of passed MC cuts) ======" << endl;
  for (int i=0;i<n_hlt_bits;i++) {
    for (int j=1;j<n_hlt_bits+1;j++) {
      cout << hlt_whichbit[1][i][j] << ", ";
    }
    cout << hlt_whichbit[1][i][0] << ", " << hlt_bitnames[i] << endl;
  }
  cout << "===== Events accepted by HLT (of passed L1) ===========" << endl;
  for (int i=0;i<n_hlt_bits;i++) {
    for (int j=1;j<n_hlt_bits+1;j++) {
      cout << hlt_whichbit[2][i][j] << ", ";
    }
    cout << hlt_whichbit[2][i][0] << ", " << hlt_bitnames[i] << endl;
  }

//  return;

  HltTree->Fill();
  m_file->cd(); 
  HltTree->Write();
  delete HltTree;

  double scale=1.0/nEvents_;
  double scale_mc=1.0/n_inmc_;
  double scale_l1=1.0/n_L1acc_;

  hlt_mult_hist->Scale(scale);
  hlt_mult_hist->Write();
  delete hlt_mult_hist;
  hlt_mult_hist_mc->Scale(scale_mc);
  hlt_mult_hist_mc->Write();
  delete hlt_mult_hist_mc;
  hlt_mult_hist_l1->Scale(scale_l1);
  hlt_mult_hist_l1->Write();
  delete hlt_mult_hist_l1;

  hlt_bit_hist->Scale(scale);
  hlt_bit_hist->Write();
  delete hlt_bit_hist;
  hlt_bit_hist_mc->Scale(scale_mc);
  hlt_bit_hist_mc->Write();
  delete hlt_bit_hist_mc;
  hlt_bit_hist_l1->Scale(scale_l1);
  hlt_bit_hist_l1->Write();
  delete hlt_bit_hist_l1;

  hlt_bit_cumul->Scale(scale);
  hlt_bit_cumul->Write();
  delete hlt_bit_cumul;
  hlt_bit_cumul_mc->Scale(scale_mc);
  hlt_bit_cumul_mc->Write();
  delete hlt_bit_cumul_mc;
  hlt_bit_cumul_l1->Scale(scale_l1);
  hlt_bit_cumul_l1->Write();
  delete hlt_bit_cumul_l1;

  for (int j=0;j<n_hlt_bits;j++) {
    hlt_redundancy[j]->Scale(scale);
    hlt_redundancy[j]->Write();
    delete hlt_redundancy[j];
    hlt_redundancy_mc[j]->Scale(scale_mc);
    hlt_redundancy_mc[j]->Write();
    delete hlt_redundancy_mc[j];
    hlt_redundancy_l1[j]->Scale(scale_l1);
    hlt_redundancy_l1[j]->Write();
    delete hlt_redundancy_l1[j];
  }
  HltTree = 0;

  if (m_file!=0) { // if there was a tree file...
    m_file->Write(); // write out the branches
    delete m_file; // close and delete the file
    m_file=0; // set to zero to clean up
  }


  return;
}

