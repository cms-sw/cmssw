/*
 * \file L1TRPCTPG.cc
 *
 * $Date: 2012/09/25 21:30:20 $
 * $Revision: 1.13 $
 * \author J. Berryhill
 *
 */

#include "DQM/L1TMonitor/interface/L1TRPCTPG.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
using namespace std;
using namespace edm;

L1TRPCTPG::L1TRPCTPG(const ParameterSet& ps)
  : rpctpgSource_( ps.getParameter< InputTag >("rpctpgSource") ),
    rpctfSource_( ps.getParameter< InputTag >("rpctfSource") )
{

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if(verbose_) cout << "L1TRPCTPG: constructor...." << endl;


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
    dbe->setCurrentFolder("L1T/L1TRPCTPG");
  }


}

L1TRPCTPG::~L1TRPCTPG()
{
}

void L1TRPCTPG::beginJob(void)
{

  nev_ = 0;

  // get hold of back-end interface
  DQMStore* dbe = 0;
  dbe = Service<DQMStore>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("L1T/L1TRPCTPG");
    dbe->rmdir("L1T/L1TRPCTPG");
  }


  if ( dbe ) 
  {
    dbe->setCurrentFolder("L1T/L1TRPCTPG");
    rpctpgbx = dbe->book1D("RPCTPG_bx", 
       "RPC digis bx - all events", 9, -4.5, 4.5 ) ;
    
    rpctpgndigi[1] = dbe->book1D("RPCTPG_ndigi", 
       "RPCTPG nDigi bx 0", 100, -0.5, 99.5 ) ;
    rpctpgndigi[2] = dbe->book1D("RPCTPG_ndigi_+1", 
       "RPCTPG nDigi bx +1", 100, -0.5, 99.5 ) ;
    rpctpgndigi[0] = dbe->book1D("RPCTPG_ndigi_-1", 
       "RPCTPG nDigi bx -1", 100, -0.5, 99.5 ) ;



    m_digiBxRPCBar = dbe->book1D("RPCDigiRPCBmu_noDTmu_bx",
       "RPC digis bx - RPC, !DT", 9, -4.5, 4.5 ) ;

    m_digiBxRPCEnd = dbe->book1D("RPCDigiRPCEmu_noCSCmu_bx",
         "RPC digis bx - RPC, !CSC", 9, -4.5, 4.5 ) ;

    m_digiBxDT = dbe->book1D("RPCDigiDTmu_noRPCBmu_bx",
         "RPC digis bx - !RPC, DT", 9, -4.5, 4.5 ) ;

    m_digiBxCSC = dbe->book1D("RPCDigiCSCmu_noRPCEmu_bx",
         "RPC digis bx - !RPC, CSC", 9, -4.5, 4.5 ) ;
   }  
}


void L1TRPCTPG::endJob(void)
{
  if(verbose_) cout << "L1TRPCTPG: end job...." << endl;
  LogInfo("EndJob") << "analyzed " << nev_ << " events"; 

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
    edm::LogInfo("DataNotFound") << "can't find RPCGeometry" << endl;
    return;
  }
//   char layerLabel[328];
//   char meId [328];
 

  /// DIGI     
  edm::Handle<RPCDigiCollection> rpcdigis;
  e.getByLabel(rpctpgSource_,rpcdigis);
    
  if (!rpcdigis.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find RPCDigiCollection with label "<< rpctpgSource_ << endl;
    return;
  }

  // Calculate the number of DT and CSC cands present
  edm::Handle<L1MuGMTReadoutCollection> pCollection;
  e.getByLabel(rpctfSource_,pCollection);
  
  if (!pCollection.isValid()) {
     edm::LogInfo("DataNotFound") << "can't find L1MuGMTReadoutCollection with label "
           << rpctfSource_.label() ;
     return;
  }
  
  L1MuGMTReadoutCollection const* gmtrc = pCollection.product();
  vector<L1MuGMTReadoutRecord> gmt_records = gmtrc->getRecords();
  vector<L1MuGMTReadoutRecord>::const_iterator RRItr;
  
  static int nRPCTrackBarrel, nRPCTrackEndcap , nDTTrack, nCSCTrack;
  nRPCTrackBarrel = 0;
  nRPCTrackEndcap = 0;
  nDTTrack = 0;
  nCSCTrack = 0;

  for( RRItr = gmt_records.begin() ;
       RRItr != gmt_records.end() ;
       RRItr++ )
  {
     // DTs
     vector<L1MuRegionalCand> DTCands = RRItr->getDTBXCands();
     for( vector<L1MuRegionalCand>::const_iterator
          ECItr = DTCands.begin() ;
          ECItr != DTCands.end() ;
          ++ECItr )
     {
        if (!ECItr->empty()) { ++nDTTrack; }
     }
      // CSCs
     vector<L1MuRegionalCand> CSCCands = RRItr->getCSCCands();
     for( vector<L1MuRegionalCand>::const_iterator
          ECItr = CSCCands.begin() ;
          ECItr != CSCCands.end() ;
          ++ECItr )
     {
        if (!ECItr->empty()) { ++nCSCTrack; }
     }

     //RPC barrel
     vector<L1MuRegionalCand> RPCBCands = RRItr->getBrlRPCCands();
     for( vector<L1MuRegionalCand>::const_iterator
          ECItr = RPCBCands.begin() ;
          ECItr != RPCBCands.end() ;
          ++ECItr )
     {
        if (!ECItr->empty()) { ++nRPCTrackBarrel; }
     }

     //RPC endcap
     vector<L1MuRegionalCand> RPCECands = RRItr->getFwdRPCCands();
     for( vector<L1MuRegionalCand>::const_iterator
          ECItr = RPCECands.begin() ;
          ECItr != RPCECands.end() ;
          ++ECItr )
     {
        if (!ECItr->empty()) { ++nRPCTrackEndcap; }
     }
  }

    int numberofDigi[3] = {0,0,0};
    

  RPCDigiCollection::DigiRangeIterator collectionItr;
  for(collectionItr=rpcdigis->begin(); collectionItr!=rpcdigis->end(); ++collectionItr){

    /*RPCDetId detId=(*collectionItr ).first; 

    
    uint32_t id=detId();
    char detUnitLabel[328];
    RPCGeomServ RPCname(detId);
    std::string nameRoll = RPCname.name();
    sprintf(detUnitLabel ,"%s",nameRoll.c_str());
    sprintf(layerLabel ,"%s",nameRoll.c_str());
    std::map<uint32_t, std::map<std::string,MonitorElement*> >::iterator meItr = rpctpgmeCollection.find(id);
    if (meItr == rpctpgmeCollection.end() || (rpctpgmeCollection.size()==0)) {
      rpctpgmeCollection[id]=L1TRPCBookME(detId);
    }
    std::map<std::string, MonitorElement*> meMap=rpctpgmeCollection[id];*/
    

//      std::vector<int> strips;
//      std::vector<int> bxs;
//      strips.clear(); 
//      bxs.clear();
     RPCDigiCollection::const_iterator digiItr; 
     for (digiItr = ((*collectionItr ).second).first;
	  digiItr!=((*collectionItr).second).second; ++digiItr){
       
       // strips is a list of hit strips (regardless of bx) for this roll
//        int strip= (*digiItr).strip();
//        strips.push_back(strip);
       int bx=(*digiItr).bx();
       rpctpgbx->Fill(bx);
       //

       if ( nRPCTrackBarrel == 0 &&  nDTTrack != 0) {
          m_digiBxDT->Fill(bx);
       } else if ( nRPCTrackBarrel != 0 &&  nDTTrack == 0) {
          m_digiBxRPCBar->Fill(bx);
       }

       if ( nRPCTrackEndcap == 0 &&  nCSCTrack != 0) {
          m_digiBxCSC->Fill(bx);
       } else if ( nRPCTrackEndcap != 0 &&  nCSCTrack == 0) {
          m_digiBxRPCEnd->Fill(bx);
       }




       
       if (bx == -1) 
       {
        numberofDigi[0]++;
       }
       if (bx == 0) 
       { 
//         sprintf(meId,"Occupancy_%s",detUnitLabel);
// 	meMap[meId]->Fill(strip);
        numberofDigi[1]++;
       }
       if (bx == 2) 
       {
        numberofDigi[2]++;
       }
       
//        sprintf(meId,"BXN_%s",detUnitLabel);
//        meMap[meId]->Fill(bx);
//        sprintf(meId,"BXN_vs_strip_%s",detUnitLabel);
//        meMap[meId]->Fill(strip,bx);
      
     }
  }

      rpctpgndigi[0]->Fill(numberofDigi[0]);
      rpctpgndigi[1]->Fill(numberofDigi[1]);
      rpctpgndigi[2]->Fill(numberofDigi[2]);

}

