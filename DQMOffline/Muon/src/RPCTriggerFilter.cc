/**\class RPCTriggerFilter.cc
 Description: Provides a trigger filter for L1 muon Candidates
*/
// Original Author:  Anna Cimmino
//         Created:  Mon May  5 17:01:23 CEST 2008

// user include files
#include "DQMOffline/Muon/src/RPCTriggerFilter.h"
//Data Formats
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
 
using namespace edm;
using namespace std;
RPCTriggerFilter::RPCTriggerFilter(const ParameterSet& ps)
{

  LogVerbatim ("rpctriggerfilter") << "[RPCTriggerFilter]: Constructor";

  enableFilter_= ps.getUntrackedParameter<bool>("EnableTriggerFilter", true);

  inputTag_ = ps.getUntrackedParameter<edm::InputTag>("GMTInputTag", edm::InputTag("gmt"));
 
  rpcBarOnly_=ps.getUntrackedParameter<bool>("RPCBarrelTrigger", false);
  rpcFwdOnly_=ps.getUntrackedParameter<bool>("RPCEndcapTrigger", false);
  rpcOnly_ = ps.getUntrackedParameter<bool>("RPCTrigger", true);
  dtOnly_ = ps.getUntrackedParameter<bool>("DTTrigger", false);
  cscOnly_ = ps.getUntrackedParameter<bool>("CSCTrigger", false);
  rpcAndDt_ = ps.getUntrackedParameter<bool>("RPCAndDT", false);
  rpcAndCsc_ = ps.getUntrackedParameter<bool>("RPCAndCSC", false);
  dtAndCsc_ = ps.getUntrackedParameter<bool>("DTAndCSC", false);
  rpcAndDtAndCsc_=ps.getUntrackedParameter<bool>("RPCAndDTAndCSC", true);
}

RPCTriggerFilter::~RPCTriggerFilter(){}

bool RPCTriggerFilter::filter(Event& iEvent, const EventSetup& iSetup)
{
  event_++;
  LogVerbatim ("rpctriggerfilter") << "[RPCTriggerFilter]: Filtering event n°"<<event_;
  if(!enableFilter_){goodEvent_++; return true;}

  Handle<L1MuGMTReadoutCollection> pCollection;
  try {
    iEvent.getByLabel(inputTag_.label(),pCollection);
  }
  catch (...) {
    return false;
  } 
  bool rpcBarFlag =false;
  bool  rpcFwdFlag=false;
  bool dtFlag=false;
  bool cscFlag=false;
  
  // get GMT readout collection
  L1MuGMTReadoutCollection const* gmtrc = pCollection.product();
  // get record vector
  vector<L1MuGMTReadoutRecord> gmt_records = gmtrc->getRecords();
  // loop over records of individual bx's
  vector<L1MuGMTReadoutRecord>::const_iterator RRItr;
  int muons_ =0;
  for( RRItr = gmt_records.begin(); RRItr != gmt_records.end(); RRItr++ ) 
  {   
    vector<L1MuGMTExtendedCand> GMTCands   = RRItr->getGMTCands();
   
    int BxInEvent = RRItr->getBxInEvent();
    if(BxInEvent!=0) continue;    
 
    //loop over GMT candidates in each record 
    vector<L1MuGMTExtendedCand>::const_iterator GMTItr;
    for( GMTItr = GMTCands.begin(); GMTItr != GMTCands.end(); ++GMTItr ) {
      
      muons_++;
      if(GMTItr->empty()) continue;
      
      if (GMTItr->isRPC()&& !GMTItr->isFwd()) rpcBarFlag= true;
      else if (GMTItr->isRPC()) rpcFwdFlag=true;
      else if (!GMTItr->isFwd())dtFlag=true;
      else cscFlag=true;
    }//loop over GMT candidates in each record END
  }//loop over records of individual bxs END
  LogVerbatim ("rpctriggerfilter") << "[RPCTriggerFilter]: Muon #"<<muons_;

  if( (rpcOnly_ && (rpcBarFlag || rpcFwdFlag)&& !dtFlag && !cscFlag) || // rpc Event
      (rpcBarOnly_ && !rpcFwdFlag && rpcBarFlag && !dtFlag && !cscFlag)|| //rpc Event in barrel
      (rpcFwdOnly_ && !rpcBarFlag && rpcFwdFlag && !dtFlag && !cscFlag)|| //rpc event in endcaps
      (dtOnly_ && !rpcFwdFlag && !rpcBarFlag && dtFlag && !cscFlag)|| //dt event
      (cscOnly_ && !rpcFwdFlag && !rpcBarFlag && !dtFlag && cscFlag)|| //csc event
      (rpcAndDt_ && !rpcFwdFlag && rpcBarFlag && dtFlag && !cscFlag)|| //rpc & dt event
      (rpcAndCsc_ && rpcFwdFlag && !rpcBarFlag && !dtFlag && cscFlag) ||// rpc & csc event
      (dtAndCsc_ && !rpcFwdFlag && !rpcBarFlag && dtFlag && cscFlag)||// dt & csc event
      (rpcAndDtAndCsc_ && (rpcFwdFlag ||rpcBarFlag) && dtFlag && cscFlag) ) { // rpc & dt & csc event
    goodEvent_++; 
    return true;
  }else {
    return false;
  }  
}

void RPCTriggerFilter::beginJob(const edm::EventSetup&)
{  //reset counters
  event_=0;
  goodEvent_=0;
}

void RPCTriggerFilter::endJob() {
  LogVerbatim ("rpctriggerfilter") << "[RPCTriggerFilter]: "<<event_<<" events checked. "<<goodEvent_<<" accepted!";
}

