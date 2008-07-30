/*
 * \file L1TGT.cc
 *
 * $Date: 2008/03/20 19:38:25 $
 * $Revision: 1.17 $
 * \author J. Berryhill, I. Mikulec
 *
 */

#include "DQM/L1TMonitor/interface/L1TGT.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

using namespace std;
using namespace edm;

L1TGT::L1TGT(const ParameterSet& ps)
  : gtSource_( ps.getParameter< InputTag >("gtSource") ),
    gtEvmSource_( ps.getParameter< InputTag >("gtEvmSource") )
{

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if(verbose_) cout << "L1TGT: constructor...." << endl;

  dbe = NULL;
  if ( ps.getUntrackedParameter<bool>("DQMStore", false) ) 
  {
    dbe = Service<DQMStore>().operator->();
    dbe->setVerbose(0);
  }

  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "");
  if ( outputFile_.size() != 0 ) {
    cout << "L1T Monitoring histograms will be saved to " << outputFile_.c_str() << endl;
  }

  bool disable = ps.getUntrackedParameter<bool>("disableROOToutput", false);
  if(disable){
    outputFile_="";
  }


  if ( dbe !=NULL ) {
    dbe->setCurrentFolder("L1T/L1TGT");
  }


}

L1TGT::~L1TGT()
{
}

void L1TGT::beginJob(const EventSetup& c)
{

  nev_ = 0;

  // get hold of back-end interface
  DQMStore* dbe = 0;
  dbe = Service<DQMStore>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("L1T/L1TGT");
    dbe->rmdir("L1T/L1TGT");
  }


  if ( dbe ) {
    dbe->setCurrentFolder("L1T/L1TGT");
    
    algo_bits = dbe->book1D("algo_bits", "GT algo bits", 128, -0.5, 127.5 );
    algo_bits->setAxisTitle("algorithm bits",1);
    algo_bits_corr = dbe->book2D("algo_bits_corr","GT algo bit correlation", 
        128, -0.5, 127.5, 128, -0.5, 127.5 );
    algo_bits_corr->setAxisTitle("algorithm bits",1);
    algo_bits_corr->setAxisTitle("algorithm bits",2);
    
    tt_bits = dbe->book1D("tt_bits", "GT technical trigger bits", 64, -0.5, 63.5 );
    tt_bits->setAxisTitle("technical trigger bits",1);
    tt_bits_corr = dbe->book2D("tt_bits_corr","GT tech. trig. bit correlation", 
        64, -0.5, 63.5, 64, -0.5, 63.5 );
    tt_bits_corr->setAxisTitle("technical trigger bits",1);
    tt_bits_corr->setAxisTitle("technical trigger bits",2);

    algo_tt_bits_corr = dbe->book2D("algo_tt_bits_corr","GT algo tech. trig. bit correlation", 
        128, -0.5, 127.5, 64, -0.5, 63.5 );
    algo_tt_bits_corr->setAxisTitle("algorithm bits",1);
    algo_tt_bits_corr->setAxisTitle("technical trigger bits",2);
    
    algo_bits_lumi = dbe->book2D("algo_bits_lumi", "GT algo bit rate per lumi segment", 
        250, 0., 250., 128, -0.5, 127.5);
    algo_bits_lumi->setAxisTitle("luminosity segment",1);
    algo_bits_lumi->setAxisTitle("algorithm bits",2);
    tt_bits_lumi = dbe->book2D("tt_bits_lumi", "GT tech. trig. bit rate per lumi segment", 
        250, 0., 250., 64, -0.5, 63.5);
    tt_bits_lumi->setAxisTitle("luminosity segment",1);
    tt_bits_lumi->setAxisTitle("technical trigger bits",2);
    
    event_type = dbe->book1D("event_type","GT event type", 10, -0.5, 9.5);
    event_type->setAxisTitle("event type",1);
    event_type->setBinLabel(2,"Physics",1);
    event_type->setBinLabel(3,"Calibration",1);
    event_type->setBinLabel(4,"Random",1);
    event_type->setBinLabel(6,"Traced",1);
    event_type->setBinLabel(7,"Test",1);
    event_type->setBinLabel(8,"Error",1);

    event_number = dbe->book1D("event_number", "GT Event number (from last resync)", 100, 0., 50000.);
    event_number->setAxisTitle("event number",1);
    event_lumi = dbe->bookProfile("event_lumi","GT Event number (from last resync) vs lumi section",
        250, 0., 250., 100, -0.1, 1.e15, "s");
    event_lumi->setAxisTitle("luminosity segment",1);
    event_lumi->setAxisTitle("event number",2);
    trigger_number = dbe->book1D("trigger_number", "GT Trigger number (from start run)", 100, 0., 50000.);
    trigger_number->setAxisTitle("trigger number",1);
    trigger_lumi = dbe->bookProfile("trigger_lumi","GT Trigger number (from start run) vs lumi section",
        250, 0., 250., 100, -0.1, 1.e15, "s");
    trigger_lumi->setAxisTitle("luminosity segment",1);
    trigger_lumi->setAxisTitle("trigger number",2);
    
    evnum_trignum_lumi = dbe->bookProfile("evnum_trignum_lumi","GT Event/Trigger number ratio vs lumi section",
        250, 0., 250., 100, -0.1, 2., "s");
    evnum_trignum_lumi->setAxisTitle("luminosity segment",1);
    evnum_trignum_lumi->setAxisTitle("event/trigger number ratio",2);
    orbit_lumi = dbe->bookProfile("orbit_lumi","GT orbit number vs lumi section",
        250, 0., 250., 100, -0.1, 1.e15, "s");
    orbit_lumi->setAxisTitle("luminosity segment",1);
    orbit_lumi->setAxisTitle("orbit number",2);
    setupversion_lumi = dbe->bookProfile("setupversion_lumi","GT setup version vs lumi section",
        250, 0., 250., 100, -0.1, 1.e10, "i");
    setupversion_lumi->setAxisTitle("luminosity segment",1);
    setupversion_lumi->setAxisTitle("prescale stup version",2);
    
    gtfe_bx = dbe->book1D("gtfe_bx","GTFE Bx number",3600, 0., 3600.);
    gtfe_bx->setAxisTitle("GTFE BX number",1);
    dbx_module = dbe->bookProfile("dbx_module", "delta Bx of GT modules wrt. GTFE",
        20,0.,20.,100,-4000.,4000.,"i");
    dbx_module->setAxisTitle("GT crate module",1);
    dbx_module->setAxisTitle("Module Bx - GTFE Bx",2);
    dbx_module->setBinLabel(1,"GTFEevm",1);
    dbx_module->setBinLabel(2,"TCS",1);
    dbx_module->setBinLabel(3,"FDL",1);
    dbx_module->setBinLabel(4,"FDLloc",1);
    dbx_module->setBinLabel(5,"PSB9",1);
    dbx_module->setBinLabel(6,"PSB9loc",1);
    dbx_module->setBinLabel(7,"PSB13",1);
    dbx_module->setBinLabel(8,"PSB13loc",1);
    dbx_module->setBinLabel(9,"PSB14",1);
    dbx_module->setBinLabel(10,"PSB14loc",1);
    dbx_module->setBinLabel(11,"PSB15",1);
    dbx_module->setBinLabel(12,"PSB15loc",1);
    dbx_module->setBinLabel(13,"PSB19",1);
    dbx_module->setBinLabel(14,"PSB19loc",1);
    dbx_module->setBinLabel(15,"PSB20",1);
    dbx_module->setBinLabel(16,"PSB20loc",1);
    dbx_module->setBinLabel(17,"PSB21",1);
    dbx_module->setBinLabel(18,"PSB21loc",1);
    dbx_module->setBinLabel(19,"GMT",1);
  }  
}


void L1TGT::endJob(void)
{
  if(verbose_) cout << "L1TGT: end job...." << endl;
  LogInfo("EndJob") << "analyzed " << nev_ << " events"; 

  if ( outputFile_.size() != 0  && dbe ) dbe->save(outputFile_);

  return;
}

void L1TGT::analyze(const Event& e, const EventSetup& c)
{
  nev_++; 
  if(verbose_) cout << "L1TGT: analyze...." << endl;

  // open main GT (DAQ) readout record - exit if failed
  Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecord;
  e.getByLabel(gtSource_,gtReadoutRecord);
     
  if (!gtReadoutRecord.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1GlobalTriggerReadoutRecord with label "
			       << gtSource_.label() ;
    return;
  }
  
  // initialize bx's to invalid value
  int gtfeBx = -1;
  int tcsBx = -1;
  int gtfeEvmBx = -1;
  int fdlBx[2] = { -1, -1};
  int psbBx[2][7] = { 
      {-1, -1, -1, -1, -1, -1, -1},
      {-1, -1, -1, -1, -1, -1, -1}};
  int gmtBx = -1;
  
  // get info from GTFE DAQ record
  L1GtfeWord gtfeWord = gtReadoutRecord->gtfeWord();
  gtfeBx = gtfeWord.bxNr();
  gtfe_bx->Fill(gtfeBx);
  setupversion_lumi->Fill(e.luminosityBlock(),gtfeWord.setupVersion());
  int gtfeActiveBoards = gtfeWord.activeBoards();
  
  
  // open EVM readout record if available
  Handle<L1GlobalTriggerEvmReadoutRecord> gtEvmReadoutRecord;
  e.getByLabel(gtEvmSource_,gtEvmReadoutRecord);
     
  if (!gtEvmReadoutRecord.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1GlobalTriggerEvmReadoutRecord with label "
                   << gtSource_.label() ;
  } else {
    // get all info from the EVM record if available
    
    L1GtfeWord gtfeEvmWord = gtEvmReadoutRecord->gtfeWord();
    gtfeEvmBx = gtfeEvmWord.bxNr();
    int gtfeEvmActiveBoards = gtfeEvmWord.activeBoards();
    
    if( isActive(gtfeEvmActiveBoards,TCS) ) { // if TCS present in the record
      
      L1TcsWord tcsWord = gtEvmReadoutRecord->tcsWord();
      tcsBx = tcsWord.bxNr();

      event_type->Fill(tcsWord.triggerType());
      orbit_lumi->Fill(e.luminosityBlock(),tcsWord.orbitNr());

      trigger_number->Fill(tcsWord.partTrigNr());
      event_number->Fill(tcsWord.eventNr());
      
      trigger_lumi->Fill(e.luminosityBlock(),tcsWord.partTrigNr());
      event_lumi->Fill(e.luminosityBlock(),tcsWord.eventNr());
      evnum_trignum_lumi->Fill(e.luminosityBlock(),double(tcsWord.eventNr())/double(tcsWord.partTrigNr()));
    }
  }
  
  // look for GMT readout collection from the same source if GMT active
  if( isActive(gtfeActiveBoards,GMT) ) {
    edm::Handle<L1MuGMTReadoutCollection> gmtReadoutCollection;
    e.getByLabel(gtSource_,gmtReadoutCollection);

    if (gmtReadoutCollection.isValid()) {
      gmtBx = gmtReadoutCollection->getRecord().getBxNr();
    }
  }

  // get info from FDL if active (including decision word)
  if( isActive(gtfeActiveBoards,FDL) ) {
    L1GtFdlWord fdlWord = gtReadoutRecord->gtFdlWord();
    fdlBx[0] = fdlWord.bxNr();
    fdlBx[1] = fdlWord.localBxNr();
    
    /// get Global Trigger algo and technical triger bit statistics
    DecisionWord gtDecisionWord = gtReadoutRecord->decisionWord();
    TechnicalTriggerWord gtTTWord = gtReadoutRecord->technicalTriggerWord();

    int dbitNumber = 0;
    DecisionWord::const_iterator GTdbitItr;
    algo_bits->Fill(-1.); // fill underflow to normalize
    for(GTdbitItr = gtDecisionWord.begin(); GTdbitItr != gtDecisionWord.end(); GTdbitItr++) {
      if (*GTdbitItr) {
        algo_bits->Fill(dbitNumber);
        algo_bits_lumi->Fill(e.luminosityBlock(),dbitNumber);
        int dbitNumber1 = 0;
        DecisionWord::const_iterator GTdbitItr1;
        for(GTdbitItr1 = gtDecisionWord.begin(); GTdbitItr1 != gtDecisionWord.end(); GTdbitItr1++) {
          if (*GTdbitItr1) algo_bits_corr->Fill(dbitNumber,dbitNumber1);
          dbitNumber1++; 
        }
        int tbitNumber1 = 0;
        TechnicalTriggerWord::const_iterator GTtbitItr1;
        for(GTtbitItr1 = gtTTWord.begin(); GTtbitItr1 != gtTTWord.end(); GTtbitItr1++) {
          if (*GTtbitItr1) tt_bits_corr->Fill(dbitNumber,tbitNumber1);
          tbitNumber1++; 
        }
      }
      dbitNumber++; 
    }

    int tbitNumber = 0;
    TechnicalTriggerWord::const_iterator GTtbitItr;
    tt_bits->Fill(-1.); // fill underflow to normalize
    for(GTtbitItr = gtTTWord.begin(); GTtbitItr != gtTTWord.end(); GTtbitItr++) {
      if (*GTtbitItr) {
        tt_bits->Fill(tbitNumber);
        tt_bits_lumi->Fill(e.luminosityBlock(),tbitNumber);
        int tbitNumber1 = 0;
        TechnicalTriggerWord::const_iterator GTtbitItr1;
        for(GTtbitItr1 = gtTTWord.begin(); GTtbitItr1 != gtTTWord.end(); GTtbitItr1++) {
          if (*GTtbitItr1) tt_bits_corr->Fill(tbitNumber,tbitNumber1);
          tbitNumber1++; 
        }
      }
      tbitNumber++; 
    }
  }
  
  // get info from active PSB's
  int ibit = PSB9; // first psb
  // for now hardcode psb id's - TODO - get them from Vasile's board maps...
  int psbID[7] = { 0xbb09, 0xbb0d, 0xbb0e,  0xbb0f,  0xbb13,  0xbb14,  0xbb15 };
  for(int i=0; i<7; i++) {
    if( isActive(gtfeActiveBoards,ibit) ) {
      L1GtPsbWord psbWord = gtReadoutRecord->gtPsbWord(psbID[i]);
      psbBx[0][i] = psbWord.bxNr();
      psbBx[1][i] = psbWord.localBxNr();
    }  
    ibit++;
  }
  
  //fill the dbx histo
  if(gtfeEvmBx>-1) dbx_module->Fill(0.,gtfeEvmBx-gtfeBx);
  if(tcsBx>-1) dbx_module->Fill(1., tcsBx-gtfeBx);
  for(int i=0; i<2; i++) {
    if(fdlBx[i]>-1) dbx_module->Fill(2.+i, fdlBx[i]-gtfeBx);
  }
  for(int j=0; j<7; j++) {
    for(int i=0; i<2; i++) {
      if(psbBx[i][j]>-1) dbx_module->Fill(4.+i+2*j, psbBx[i][j]-gtfeBx);
    }
  }
  if(gmtBx>-1) dbx_module->Fill(18., gmtBx-gtfeBx);
}

////////////////////////////////////////////////////////////////////////////////////////
bool L1TGT::isActive(int word, int bit) {
  if( word & (1<<bit) ) return true;
  return false;
}
