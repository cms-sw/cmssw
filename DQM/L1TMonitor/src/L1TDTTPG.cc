/*
 * \file L1TDTTPG.cc
 *
 * $Date: 2007/06/27 12:55:02 $
 * $Revision: 1.7 $
 * \author J. Berryhill
 *
 */

#include "DQM/L1TMonitor/interface/L1TDTTPG.h"

using namespace std;
using namespace edm;

L1TDTTPG::L1TDTTPG(const ParameterSet& ps)
  : dttpgSource_( ps.getParameter< InputTag >("dttpgSource") )
{

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if(verbose_) cout << "L1TDTTPG: constructor...." << endl;

  logFile_.open("L1TDTTPG.log");

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


  if ( dbe !=NULL ) {
    dbe->setCurrentFolder("L1TMonitor/L1TDTTPG");
  }


}

L1TDTTPG::~L1TDTTPG()
{
}

void L1TDTTPG::beginJob(const EventSetup& c)
{

  nev_ = 0;

  // get hold of back-end interface
  DaqMonitorBEInterface* dbe = 0;
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("L1TMonitor/L1TDTTPG");
    dbe->rmdir("L1TMonitor/L1TDTTPG");
  }


  if ( dbe ) 
  {
    dbe->setCurrentFolder("L1TMonitor/L1TDTTPG");

    dttpgphbx = dbe->book1D("DT_TPG_phi_bx", 
       "DT TPG phi bx", 50, -24.5, 24.5 ) ;  
    dttpgphwheel = dbe->book1D("DT_TPG_phi_wheel_number", 
       "DT TPG phi wheel number", 5, -2.5, 2.5 ) ;  
    dttpgphsector = dbe->book1D("DT_TPG_phi_sector_number", 
       "DT TPG phi sector number", 12, -0.5, 11.5 ) ;  
    dttpgphstation = dbe->book1D("DT_TPG_phi_station_number", 
       "DT TPG phi station number", 5, 0.5, 4.5 ) ;  
    dttpgphphi = dbe->book1D("DT_TPG_phi", 
       "DT TPG phi", 100, -2100., 2100. ) ;  
    dttpgphphiB = dbe->book1D("DT_TPG_phiB", 
       "DT TPG phiB", 100, -550., 550. ) ;  
    dttpgphquality = dbe->book1D("DT_TPG_phi_quality", 
       "DT TPG phi quality", 8, -0.5, 7.5 ) ;  
    dttpgphts2tag = dbe->book1D("DT_TPG_phi_Ts2Tag", 
       "DT TPG phi Ts2Tag", 2, -0.5, 1.5 ) ;  
    dttpgphbxcnt = dbe->book1D("DT_TPG_phi_BxCnt", 
       "DT TPG phi BxCnt", 10, -0.5, 9.5 ) ;  
    dttpgphntrack = dbe->book1D("DT_TPG_phi_ntrack", 
       "DT TPG phi ntrack", 20, -0.5, 19.5 ) ;  

    dttpgthbx = dbe->book1D("DT_TPG_theta_bx", 
       "DT TPG theta bx", 50, -24.5, 24.5 ) ;  
    dttpgthwheel = dbe->book1D("DT_TPG_theta_wheel_number", 
       "DT TPG theta wheel number", 5, -2.5, 2.5 ) ;  
    dttpgthsector = dbe->book1D("DT_TPG_theta_sector_number", 
       "DT TPG theta sector number", 12, -0.5, 11.5 ) ;  
    dttpgthstation = dbe->book1D("DT_TPG_theta_station_number", 
       "DT TPG theta station number", 5, -0.5, 4.5 ) ;  
    dttpgththeta = dbe->book1D("DT_TPG_theta", 
       "DT TPG theta", 20, -0.5, 19.5 ) ;  
    dttpgthquality = dbe->book1D("DT_TPG_theta_quality", 
       "DT TPG theta quality", 8, -0.5, 7.5 ) ;  
    dttpgthntrack = dbe->book1D("DT_TPG_theta_ntrack", 
       "DT TPG theta ntrack", 20, -0.5, 19.5 ) ;  
  }  
}


void L1TDTTPG::endJob(void)
{
  if(verbose_) cout << "L1TDTTPG: end job...." << endl;
  LogInfo("L1TDTTPG") << "analyzed " << nev_ << " events"; 

 if ( outputFile_.size() != 0  && dbe ) dbe->save(outputFile_);

 return;
}

void L1TDTTPG::analyze(const Event& e, const EventSetup& c)
{
  nev_++; 
  if(verbose_) cout << "L1TDTTPG: analyze...." << endl;

  edm::Handle<L1MuDTChambPhContainer > myL1MuDTChambPhContainer;  

  try {
  e.getByLabel(dttpgSource_,myL1MuDTChambPhContainer);
  }
  catch (...) {
    edm::LogInfo("L1TDTTPG") << "can't find L1MuDTChambPhContainer with label "
			       << dttpgSource_.label() ;
    return;
  }
  vector<L1MuDTChambPhDigi>* myPhContainer =  myL1MuDTChambPhContainer->getContainer();

  edm::Handle<L1MuDTChambThContainer > myL1MuDTChambThContainer;  
  try {
  e.getByLabel(dttpgSource_,myL1MuDTChambThContainer);
  }
  catch (...) {
    edm::LogInfo("L1TDTTPG") << "can't find L1MuDTChambThContainer with label "
			       << dttpgSource_.label() ;
    return;
  }
  vector<L1MuDTChambThDigi>* myThContainer =  myL1MuDTChambThContainer->getContainer();

  int ndttpgphtrack = 0;
  int ndttpgthtrack = 0; 

  
  for( vector<L1MuDTChambPhDigi>::const_iterator 
       DTPhDigiItr =  myPhContainer->begin() ;
       DTPhDigiItr != myPhContainer->end() ;
       ++DTPhDigiItr ) 
    {           
     ndttpgphtrack++;
     dttpgphwheel->Fill(DTPhDigiItr->whNum());
     if (verbose_)
       {
	cout << "DTTPG phi wheel number " << DTPhDigiItr->whNum() << endl;
			}
     dttpgphstation->Fill(DTPhDigiItr->stNum());
     if (verbose_)
       {   
        cout << "DTTPG phi station number " << DTPhDigiItr->stNum() << endl;
			}
     dttpgphsector->Fill(DTPhDigiItr->scNum());
     if (verbose_)
       {
        cout << "DTTPG phi sector number " << DTPhDigiItr->scNum() << endl;
			}
     dttpgphbx->Fill(DTPhDigiItr->bxNum());
     if (verbose_)
       {
        cout << "DTTPG phi bx number " << DTPhDigiItr->bxNum() << endl;
			}
     dttpgphphi->Fill(DTPhDigiItr->phi());
     if (verbose_)
       {
        cout << "DTTPG phi phi " << DTPhDigiItr->phi() << endl;
			}
     dttpgphphiB->Fill(DTPhDigiItr->phiB());
     if (verbose_)
       {
        cout << "DTTPG phi phiB " << DTPhDigiItr->phiB() << endl;
			}
     dttpgphquality->Fill(DTPhDigiItr->code());
     if (verbose_)
       {
        cout << "DTTPG phi quality " << DTPhDigiItr->code() << endl;
			}
     dttpgphts2tag->Fill(DTPhDigiItr->Ts2Tag());
     if (verbose_)
       {
        cout << "DTTPG phi ts2tag " << DTPhDigiItr->Ts2Tag() << endl;
			}
     dttpgphbxcnt->Fill(DTPhDigiItr->BxCnt());
     if (verbose_)
       {
        cout << "DTTPG phi bxcnt " << DTPhDigiItr->BxCnt() << endl;
			}
    }

  for( vector<L1MuDTChambThDigi>::const_iterator 
       DTThDigiItr =  myThContainer->begin() ;
       DTThDigiItr != myThContainer->end() ;
       ++DTThDigiItr ) 
    {           		
     ndttpgthtrack++;
     dttpgthwheel->Fill(DTThDigiItr->whNum());
     if (verbose_)
       {
      cout << "DTTPG theta wheel number " << DTThDigiItr->whNum() << endl;
			}
     dttpgthstation->Fill(DTThDigiItr->stNum());
     if (verbose_)
       {   
      cout << "DTTPG theta station number " << DTThDigiItr->stNum() << endl;
			}
     dttpgthsector->Fill(DTThDigiItr->scNum());
     if (verbose_)
       {
        cout << "DTTPG theta sector number " << DTThDigiItr->scNum() << endl;
			}
     dttpgthbx->Fill(DTThDigiItr->bxNum());
     if (verbose_)
       {
        cout << "DTTPG theta bx number " << DTThDigiItr->bxNum() << endl;
			}
     for (int j = 0; j < 7; j++)
       {
        dttpgththeta->Fill(DTThDigiItr->position(j));
	if (verbose_)
	  {
           cout << "DTTPG theta position " << DTThDigiItr->position(j) << endl;
			}
        dttpgthquality->Fill(DTThDigiItr->code(j));
	if (verbose_)
	  {
           cout << "DTTPG theta quality " << DTThDigiItr->code(j) << endl;
			}
       }

    }
    dttpgphntrack->Fill(ndttpgphtrack);
    if (verbose_)
      {
       cout << "DTTPG phi ntrack " << ndttpgphtrack << endl;
			}
    dttpgthntrack->Fill(ndttpgthtrack);
    if (verbose_)
      {
       cout << "DTTPG theta ntrack " << ndttpgthtrack << endl;
			}

}


