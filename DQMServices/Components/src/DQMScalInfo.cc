/*
 * \file DQMDcsInfo.cc
 * \author A.Meyer - DESY
 * Last Update:
 *
 */

#include "DQMScalInfo.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

using namespace std;


// Framework


DQMScalInfo::DQMScalInfo(const edm::ParameterSet& ps)
{
  parameters_ = ps;

  scalfolder_          = parameters_.getUntrackedParameter<std::string>("dqmScalFolder", "Scal") ;
  gtCollection_        = consumes<L1GlobalTriggerReadoutRecord>(parameters_.getUntrackedParameter<std::string>("gtCollection","gtDigis"));
  dcsStatusCollection_ = consumes<DcsStatusCollection>(parameters_.getUntrackedParameter<std::string>("dcsStatusCollection","scalersRawToDigi"));
  l1tscollectionToken_ = consumes<Level1TriggerScalersCollection>(parameters_.getUntrackedParameter<std::string>("l1TSCollection", "scalersRawToDigi"));

}

DQMScalInfo::~DQMScalInfo(){
}

void DQMScalInfo::bookHistograms(DQMStore::IBooker & ibooker,
                                edm::Run const & /* iRun */,
                                edm::EventSetup const & /* iSetup */) {

  // Fetch GlobalTag information and fill the string/ME.
  ibooker.cd();
  ibooker.setCurrentFolder(scalfolder_ +"/L1TriggerScalers/");
  const int fracLS = 16;
  const int maxLS  = 250;
  hlresync_    = ibooker.book1D("lresync","Orbit of last resync",fracLS*maxLS,0,maxLS*262144);
  hlOC0_       = ibooker.book1D("lOC0","Orbit of last OC0",fracLS*maxLS,0,maxLS*262144);
  hlTE_        = ibooker.book1D("lTE","Orbit of last TestEnable",fracLS*maxLS,0,maxLS*262144);
  hlstart_     = ibooker.book1D("lstart","Orbit of last Start",fracLS*maxLS,0,maxLS*262144);
  hlEC0_       = ibooker.book1D("lEC0","Orbit of last EC0",fracLS*maxLS,0,maxLS*262144);
  hlHR_        = ibooker.book1D("lHR","Orbit of last HardReset",fracLS*maxLS,0,maxLS*262144);
}

void DQMScalInfo::analyze(const edm::Event& e, const edm::EventSetup& c){
  makeL1Scalars(e);
  return;
}

void
DQMScalInfo::makeL1Scalars(const edm::Event& e)
{
  edm::Handle<Level1TriggerScalersCollection> l1ts;
  e.getByToken(l1tscollectionToken_,l1ts);
  if(l1ts->size()==0) return;
  hlresync_->Fill((*l1ts)[0].lastResync());
  hlOC0_->Fill((*l1ts)[0].lastOrbitCounter0());
  hlTE_->Fill((*l1ts)[0].lastTestEnable());
  hlstart_->Fill((*l1ts)[0].lastStart());
  hlEC0_->Fill((*l1ts)[0].lastEventCounter0());
  hlHR_->Fill((*l1ts)[0].lastHardReset());  

  return ;
}
