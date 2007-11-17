/*
 * \file L1TCSCTF.cc
 *
 * $Date: 2007/07/19 16:48:23 $
 * $Revision: 1.5 $
 * \author J. Berryhill
 *
 */

#include "DQM/L1TMonitor/interface/L1TCSCTF.h"
///KK
#include <sstream>  // for std::ostringstream
#include <string.h> // for memcpy and bzero
#include <DataFormats/L1CSCTrackFinder/interface/L1CSCStatusDigiCollection.h> // for CSC TF status
///
using namespace std;
using namespace edm;

L1TCSCTF::L1TCSCTF(const ParameterSet& ps)
  : csctfSource_( ps.getParameter< InputTag >("csctfSource") )
{

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if(verbose_) cout << "L1TCSCTF: constructor...." << endl;

  logFile_.open("L1TCSCTF.log");

  dbe = NULL;
  if ( ps.getUntrackedParameter<bool>("DaqMonitorBEInterface", false) )
  {
    dbe = Service<DaqMonitorBEInterface>().operator->();
    dbe->setVerbose(0);
  }

  monitorDaemon_ = false;
  if ( ps.getUntrackedParameter<bool>("MonitorDaemon", false) ) {
    Service<MonitorDaemon> daemon;
    daemon.operator->();
    monitorDaemon_ = true;
  }

  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "");
  if ( outputFile_.size() != 0 ) {
    cout << "L1T Monitoring histograms will be saved to " << outputFile_.c_str() << endl;
  }
  else{
    outputFile_ = "L1TDQM.root";
  }

  bool disable = ps.getUntrackedParameter<bool>("disableROOToutput", false);
  if(disable){
    outputFile_="";
  }

///KK
  emulation = ps.getUntrackedParameter<bool>("emulation", false);
  LogDebug("L1TCSCTF")<<"Emulation is set to "<<emulation;
  // Initialize slot<->sector assignment
  slot2sector = ps.getUntrackedParameter< std::vector<int> >("slot2sector",std::vector<int>(0));
  if( slot2sector.size() != 22 ){
    if( slot2sector.size() ) edm::LogError("L1TCSCTF")<<"Wrong 'untracked vint32 slot2sector' size."
       <<" SectorProcessor boards reside in some of 22 slots and assigned to 12 sectors. Using defaults";
    // Use default assignment
    LogInfo("L1TCSCTF|ctor")<<"Creating default slot<->sector assignment";
    slot2sector.resize(22);
    slot2sector[0] = 0; slot2sector[1] = 0; slot2sector[2] = 0;
    slot2sector[3] = 0; slot2sector[4] = 0; slot2sector[5] = 0;
    slot2sector[6] = 1; slot2sector[7] = 2; slot2sector[8] = 3;
    slot2sector[9] = 4; slot2sector[10]= 5; slot2sector[11]= 6;
    slot2sector[12]= 0; slot2sector[13]= 0;
    slot2sector[14]= 0; slot2sector[15]= 0;
    slot2sector[16]= 7; slot2sector[17]= 8; slot2sector[18]= 9;
    slot2sector[19]=10; slot2sector[20]=11; slot2sector[21]=12;
  } else {
    LogInfo("L1TCSCTF|ctor")<<"Reassigning slot<->sector map according to 'untracked vint32 slot2sector'";
    for(int slot=0; slot<22; slot++)
      if( slot2sector[slot]<0 || slot2sector[slot]>12 )
        throw cms::Exception("Invalid configuration")<<"L1TCSCTF: sector index is set out of range (slot2sector["<<slot<<"]="<<slot2sector[slot]<<", should be [0-12])";
  }

  // Clean up pointers
  bzero(cscsp_fmm_status,sizeof(cscsp_fmm_status));
  bzero(cscsp_errors,    sizeof(cscsp_errors));
  csctf_errors = 0;
///

  if ( dbe !=NULL ) {
    dbe->setCurrentFolder("L1TMonitor/L1TCSCTF");
  }

}

L1TCSCTF::~L1TCSCTF()
{
}

void L1TCSCTF::beginJob(const EventSetup& c)
{

  nev_ = 0;

  // get hold of back-end interface
  DaqMonitorBEInterface* dbe = 0;
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("L1TMonitor/L1TCSCTF");
    dbe->rmdir("L1TMonitor/L1TCSCTF");
  }


  if ( dbe )
  {
    dbe->setCurrentFolder("L1TMonitor/L1TCSCTF");

    csctfetavalue = dbe->book1D("CSC TF eta value",
       "CSC TF eta value", 100, -2.5, 2.5 ) ;
    csctfphivalue = dbe->book1D("CSC TF phi value",
       "CSC TF phi value", 100, 0.0, 6.2832 ) ;
    csctfptvalue = dbe->book1D("CSC TF pt value",
       "CSC TF pt value", 160, -0.5, 159.5 ) ;
    csctfptpacked = dbe->book1D("CSC TF pt_packed",
       "CSC TF pt_packed", 160, -0.5, 159.5 ) ;
    csctfquality = dbe->book1D("CSC TF quality",
       "CSC TF quality", 20, -0.5, 19.5 ) ;
    csctfchargevalue = dbe->book1D("CSC TF charge value",
       "CSC TF charge value", 2, -1.5, 1.5 ) ;
    csctfntrack = dbe->book1D("CSC TF ntrack",
       "CSC TF ntrack", 20, -0.5, 19.5 ) ;
///KK
    for(int sp=1; sp<=12; sp++){
       std::ostringstream name1;
       name1<<"CSC TF SP"<<(sp<10?"0":"")<<sp<<" FMM Status"<<std::ends;
       cscsp_fmm_status[sp] = dbe->book1D(name1.str().c_str(),name1.str().c_str(),7,0,7);
       std::ostringstream name2;
       name2<<"CSC TF SP"<<(sp<10?"0":"")<<sp<<" Errors"<<std::ends;
       cscsp_errors[sp] = dbe->book1D(name2.str().c_str(),name2.str().c_str(),17,0,17);
    }
    cscsp_fmm_status[0] = dbe->book1D("CSC TF 'Unknown SP' FMM Status","CSC TF 'Unknown SP' FMM Status",7,0,7);
    cscsp_errors    [0] = dbe->book1D("CSC TF 'Unknown SP' Errors",    "CSC TF 'Unknown SP' Errors",   17,0,17);
    csctf_errors = dbe->book1D("CSC TF Errors","CSC TF Errors",14,0,14);
///
  }
}


void L1TCSCTF::endJob(void)
{
  if(verbose_) cout << "L1TCSCTF: end job...." << endl;
  LogInfo("L1TCSCTF") << "analyzed " << nev_ << " events";

 if ( outputFile_.size() != 0  && dbe ) dbe->save(outputFile_);

 return;
}

void L1TCSCTF::analyze(const Event& e, const EventSetup& c)
{
  nev_++;
  if(verbose_) cout << "L1TCSCTF: analyze...." << endl;

///KK
// Following code works only with CSC TF emulation data.
// Unpaking of real data doesn't produce any of monitorable elements
//   which you can see in "emulation" scope (even they can be constructed from real data).

if(emulation){
///

  edm::Handle<std::vector<L1MuRegionalCand> > pCSCTFtracks;
  e.getByLabel(csctfSource_,pCSCTFtracks);
  int ncsctftrack = 0;
   for( vector<L1MuRegionalCand>::const_iterator
        CSCTFtrackItr =  pCSCTFtracks->begin() ;
        CSCTFtrackItr != pCSCTFtracks->end() ;
        ++CSCTFtrackItr )
   {

      ncsctftrack++;

     csctfetavalue->Fill(CSCTFtrackItr->etaValue());
     if (verbose_)
       {
     std::cout << "CSC TF etavalue " << CSCTFtrackItr->etaValue()
   	    << std::endl;
       }

     csctfphivalue->Fill(CSCTFtrackItr->phiValue());
     if (verbose_)
       {
     std::cout << "CSC TF phivalue " << CSCTFtrackItr->phiValue()
   	    << std::endl;
       }

     csctfptvalue->Fill(CSCTFtrackItr->ptValue());
     if (verbose_)
       {
     std::cout << "CSC TF ptvalue " << CSCTFtrackItr->ptValue()
   	    << std::endl;
       }

     csctfptpacked->Fill(CSCTFtrackItr->pt_packed());
     if (verbose_)
       {
     std::cout << "CSC TF pt_packed " << CSCTFtrackItr->pt_packed()
   	    << std::endl;
       }

     csctfquality->Fill(CSCTFtrackItr->quality());
     if (verbose_)
       {
     std::cout << "CSC TF quality " << CSCTFtrackItr->quality()
   	    << std::endl;
       }

     csctfchargevalue->Fill(CSCTFtrackItr->chargeValue());
     if (verbose_)
       {
     std::cout << "CSC TF charge value " << CSCTFtrackItr->chargeValue()
   	    << std::endl;
       }

    }

     csctfntrack->Fill(ncsctftrack);
     if (verbose_)
       {
     std::cout << "CSC TF ntrack " << ncsctftrack
   	    << std::endl;
       }

///KK
} else {

// Below you see very simple DQM analysis, based on real hardware status of the CSC TF crate
// The rest is to be written

    // Get CSC TF status
    edm::Handle<L1CSCStatusDigiCollection> pCSCTFstatus;
    e.getByLabel(csctfSource_,pCSCTFstatus);

    // Iterate over all SPs
    for(std::vector<L1CSCSPStatusDigi>::const_iterator stat=pCSCTFstatus->second.begin();
        stat!=pCSCTFstatus->second.end(); stat++){

        // Cumulative error word
        unsigned long error_word = 0;
        //
        int sector = 0;

       // Following SP slots are allowed in TF crate: 6-11 and 16-21
       if( stat->slot()<22 && slot2sector[stat->slot()]<13 ){
          sector = slot2sector[stat->slot()];

          // Run over 7 FMM states
          for(int state=0; state<7; state++)
             if( stat->FMM()&(1<<state) )
                cscsp_fmm_status[sector]->Fill(state);

          error_word = stat->SEs() | stat->SMs() | stat->BXs() | stat->AFs();

          // Run over 17 bits of cumulative error word
          for(int bit=0; bit<17; bit++)
             if( error_word&(1<<bit) )
                cscsp_errors[sector]->Fill(bit);

       } else
          LogError("L1TCSCTF")<<"Unacceptable slot<->sector assignment. "<<
            "Check that 'untracked vint32 slot2sector' is [0-12] range.";

       if( error_word ) csctf_errors->Fill(sector);
    }
    // Unpacking related problems
    if( pCSCTFstatus->first ) csctf_errors->Fill(13);
    }
///
}
