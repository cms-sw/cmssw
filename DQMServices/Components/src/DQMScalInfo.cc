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

#include "DataFormats/Scalers/interface/Level1TriggerScalers.h"
#include "DataFormats/Scalers/interface/Level1TriggerRates.h"
#include "DataFormats/Scalers/interface/ScalersRaw.h"
#include "DataFormats/Scalers/interface/TimeSpec.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"

using namespace std;
using namespace edm;


// Framework


DQMScalInfo::DQMScalInfo(const edm::ParameterSet& ps)
{
  parameters_ = ps;

  scalfolder_          = parameters_.getUntrackedParameter<std::string>("dqmScalFolder", "Scal") ;
  gtCollection_        = consumes<L1GlobalTriggerReadoutRecord>(parameters_.getUntrackedParameter<edm::InputTag>("gtCollection", edm::InputTag("gtDigis")));
  dcsStatusCollection_ = consumes<DcsStatusCollection>(parameters_.getUntrackedParameter<edm::InputTag>("dcsStatusCollection", edm::InputTag("scalersRawToDigi")));
  l1tscollectionToken_ = consumes<Level1TriggerScalersCollection>(parameters_.getUntrackedParameter<edm::InputTag>("l1TSCollection", edm::InputTag("scalersRawToDigi")));
  lumicollectionToken_ = consumes<LumiScalersCollection>(parameters_.getUntrackedParameter<edm::InputTag>("lumiCollection", edm::InputTag("scalersRawToDigi")));
}

DQMScalInfo::~DQMScalInfo(){
}

void DQMScalInfo::bookHistograms(DQMStore::IBooker & ibooker,
                                edm::Run const & /* iRun */,
                                edm::EventSetup const & /* iSetup */) {

  const int maxNbins = 2001;

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

  hphysTrig_   = ibooker.book1D("Physics_Triggers", "Physics Triggers", maxNbins, -0.5, double(maxNbins)-0.5);
  hphysTrig_->setAxisTitle("Lumi Section", 1);

  ibooker.cd();
  ibooker.setCurrentFolder(scalfolder_ +"/LumiScalers/");
  hinstLumi_   = ibooker.book1D("Instant_Lumi", "Instant Lumi", maxNbins, -0.5, double(maxNbins)-0.5);
}

void DQMScalInfo::analyze(const edm::Event& e, const edm::EventSetup& c){
  makeL1Scalars(e);
  makeLumiScalars(e);
  return;
}

void
DQMScalInfo::makeL1Scalars(const edm::Event& e)
{
  edm::Handle<Level1TriggerScalersCollection> l1ts;
  e.getByToken(l1tscollectionToken_,l1ts);
  edm::Handle<LumiScalersCollection> lumiScalers;
  e.getByToken(lumicollectionToken_,lumiScalers);

  Level1TriggerScalersCollection::const_iterator it = l1ts->begin();

  if(l1ts->size()==0) return;
  hlresync_->Fill((*l1ts)[0].lastResync());
  hlOC0_->Fill((*l1ts)[0].lastOrbitCounter0());
  hlTE_->Fill((*l1ts)[0].lastTestEnable());
  hlstart_->Fill((*l1ts)[0].lastStart());
  hlEC0_->Fill((*l1ts)[0].lastEventCounter0());
  hlHR_->Fill((*l1ts)[0].lastHardReset());  

  unsigned int lumisection = it->lumiSegmentNr();
  if(lumisection){
    hphysTrig_->setBinContent(lumisection + 1, it->l1AsPhysics());
  }

  return ;
}

void
DQMScalInfo::makeLumiScalars(const edm::Event& e)
{
  edm::Handle<LumiScalersCollection> lumiScalers;
  e.getByToken(lumicollectionToken_,lumiScalers);

  LumiScalersCollection::const_iterator it = lumiScalers->begin();

  if(lumiScalers->size()){
    unsigned int lumisection = it->sectionNumber();
    if(lumisection){
      hinstLumi_->setBinContent(lumisection + 1, it->instantLumi());
    }
  }

  return;
}
