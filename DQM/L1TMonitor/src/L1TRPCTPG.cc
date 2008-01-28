/*
 * \file L1TRPCTPG.cc
 *
 * $Date: 2007/11/19 15:08:22 $
 * $Revision: 1.3 $
 * \author J. Berryhill
 *
 */

#include "DQM/L1TMonitor/interface/L1TRPCTPG.h"

using namespace std;
using namespace edm;

L1TRPCTPG::L1TRPCTPG(const ParameterSet& ps)
  : rpctpgSource_( ps.getParameter< InputTag >("rpctpgSource") )
{

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if(verbose_) cout << "L1TRPCTPG: constructor...." << endl;

  logFile_.open("L1TRPCTPG.log");

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
    dbe->setCurrentFolder("L1T/L1TRPCTPG");
  }


}

L1TRPCTPG::~L1TRPCTPG()
{
}

void L1TRPCTPG::beginJob(const EventSetup& c)
{

  nev_ = 0;

  // get hold of back-end interface
  DaqMonitorBEInterface* dbe = 0;
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("L1T/L1TRPCTPG");
    dbe->rmdir("L1T/L1TRPCTPG");
  }


  if ( dbe ) 
  {
    dbe->setCurrentFolder("L1T/L1TRPCTPG");
    rpctpgbx = dbe->book1D("RPCTPG_bx", 
       "RPCTPG bx 0", 3, -1.5, 1.5 ) ;
    rpctpgndigi[1] = dbe->book1D("RPCTPG_ndigi", 
       "RPCTPG nDigi bx 0", 100, -0.5, 99.5 ) ;
    rpctpgndigi[2] = dbe->book1D("RPCTPG_ndigi_+1", 
       "RPCTPG nDigi bx +1", 100, -0.5, 99.5 ) ;
    rpctpgndigi[0] = dbe->book1D("RPCTPG_ndigi_-1", 
       "RPCTPG nDigi bx -1", 100, -0.5, 99.5 ) ;
   }  
}


void L1TRPCTPG::endJob(void)
{
  if(verbose_) cout << "L1TRPCTPG: end job...." << endl;
  LogInfo("L1TRPCTPG") << "analyzed " << nev_ << " events"; 

 if ( outputFile_.size() != 0  && dbe ) dbe->save(outputFile_);

 return;
}

void L1TRPCTPG::analyze(const Event& e, const EventSetup& c)
{
  nev_++; 
  if(verbose_) cout << "L1TRPCTPG: analyze...." << endl;

  
  /// RPC Geometry
  edm::ESHandle<RPCGeometry> rpcGeo;
  c.get<MuonGeometryRecord>().get(rpcGeo);
  if (!rpcGeo.isValid()) {
    edm::LogInfo("L1TRPCTPG") << "can't find RPCGeometry" << endl;
    return;
  }

  /// DIGI     
  edm::Handle<RPCDigiCollection> rpcdigis;
  e.getByLabel(rpctpgSource_,rpcdigis);
    
  if (!rpcdigis.isValid()) {
    edm::LogInfo("L1TRPCTPG") << "can't find RPCDigiCollection with label "<< rpctpgSource_ << endl;
    return;
  }

  /// RecHits, perhaps to add later
  if (0){
  edm::Handle<RPCRecHitCollection> rpcHits;
  e.getByType(rpcHits);
     
  if (!rpcHits.isValid()) {
    edm::LogInfo("L1TRPCTPG") << "can't find RPCRecHitCollection of any type" << endl;
    return;
  }  
  }

    int numberofDigi[3] = {0,0,0};
    

  RPCDigiCollection::DigiRangeIterator collectionItr;
  for(collectionItr=rpcdigis->begin(); collectionItr!=rpcdigis->end(); ++collectionItr){

        RPCDetId detId=(*collectionItr ).first; 
	//        cout << "detId "<< detId << endl;

      std::vector<int> strips;
     std::vector<int> bxs;
     strips.clear(); 
     bxs.clear();
     RPCDigiCollection::const_iterator digiItr; 
     for (digiItr = ((*collectionItr ).second).first;
	  digiItr!=((*collectionItr).second).second; ++digiItr){
       
       int strip= (*digiItr).strip();
       strips.push_back(strip);
       int bx=(*digiItr).bx();
       rpctpgbx->Fill(bx);
       if (bx == -1) 
       {
        numberofDigi[0]++;
       }
       if (bx == 0) 
       { 
        numberofDigi[1]++;
       }
       if (bx == 2) 
       {
        numberofDigi[2]++;
       }
       bool bxExists = false;
       //std::cout <<"==> strip = "<<strip<<" bx = "<<bx<<std::endl;
       for(std::vector<int>::iterator existingBX= bxs.begin();
	   existingBX != bxs.end(); ++existingBX){
	 if (bx==*existingBX) {
	   bxExists=true;
	   break;
	 }
       }
      if(!bxExists)bxs.push_back(bx);
      
     }
  }

      rpctpgndigi[0]->Fill(numberofDigi[0]);
      rpctpgndigi[1]->Fill(numberofDigi[1]);
      rpctpgndigi[2]->Fill(numberofDigi[2]);

}

