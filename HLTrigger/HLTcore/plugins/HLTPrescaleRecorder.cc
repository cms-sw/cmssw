/** \class HLTPrescaleRecorder
 *
 * See header file for documentation
 *
 *  $Date: 2010/02/16 10:24:52 $
 *  $Revision: 1.37 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTPrescaleRecorder.h"
#include "DataFormats/HLTReco/interface/HLTPrescaleTable.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<string>

using namespace std;
using namespace edm;
using namespace trigger;

//
// constructors and destructor
//
HLTPrescaleRecorder::HLTPrescaleRecorder(const edm::ParameterSet& ps) : 
  src_(ps.getParameter<int>("src")),
  run_(ps.getParameter<bool>("run")),
  lumi_(ps.getParameter<bool>("lumi")),
  event_(ps.getParameter<bool>("event")),
  ps_(0),
  hltInputTag_(ps.getParameter<InputTag>("hltInputTag")),
  hltHandle_(),
  hlt_()
{
  if(edm::Service<edm::service::PrescaleService>().isAvailable()) {
    ps_ = edm::Service<edm::service::PrescaleService>().operator->();
  } else if (src_==0) {
    LogError("HLTPrescaleRecorder")<<"PrescaleService requested as source but unavailable!";
  }

  if (run_)   produces<HLTPrescaleTable,edm::InRun>("Run");
  if (lumi_)  produces<HLTPrescaleTable,edm::InLumi>("Lumi");
  if (event_) produces<HLTPrescaleTable,edm::InEvent>("Event");

  LogInfo("HLTPrescaleRecorder")
    << "configured with src/run/lumi/event/tag: " << src_ << " "
    << run_ << " " << lumi_ << " " << event_ << " "
    << hltInputTag_.encode();

}

HLTPrescaleRecorder::~HLTPrescaleRecorder()
{
}

//
// member functions
//

void HLTPrescaleRecorder::beginRun(edm::Run& iRun, const edm::EventSetup& iSetup) {
  if (src_==1) {
    if (iRun.getByLabel(hltInputTag_,hltHandle_)) {
      hlt_=*hltHandle_;
    } else {
      hlt_=HLTPrescaleTable();
      LogError("HLTPrescaleRecorder")<<"HLTPrescaleTable not found in Run!";
    }
  }
  return;
}

void HLTPrescaleRecorder::beginLuminosityBlock(edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup) {
  if (src_==2) {
    if (iLumi.getByLabel(hltInputTag_,hltHandle_)) {
      hlt_=*hltHandle_;
    } else {
      hlt_=HLTPrescaleTable();
      LogError("HLTPrescaleRecorder")<<"HLTPrescaleTable not found in LumiBlock!";
    }
  }
  /// prescale service set index updated at lumi block boundaries
  if (src_==0) {
    if (ps_!=0) {
      hlt_=HLTPrescaleTable(ps_->getLvl1IndexDefault(), ps_->getLvl1Labels(), ps_->getPrescaleTable());
    } else {
      hlt_=HLTPrescaleTable();
      LogError("HLTPrescaleRecorder")<<"PrescaleService not found!";
    }
  }
  return;
}

void HLTPrescaleRecorder::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (src_==3) {
    if (iEvent.getByLabel(hltInputTag_,hltHandle_)) {
      hlt_=*hltHandle_;
    } else {
      hlt_=HLTPrescaleTable();
      LogError("HLTPrescaleRecorder")<<"HLTPrescaleTable not found in Event!";
    }
  }
  ///
  if (event_) {
    auto_ptr<HLTPrescaleTable> product (new HLTPrescaleTable(hlt_));
    iEvent.put(product,"Event");
  }
  return;
}

void HLTPrescaleRecorder::endLuminosityBlock(edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup) {
  if (lumi_) {
    auto_ptr<HLTPrescaleTable> product (new HLTPrescaleTable(hlt_));
    iLumi.put(product,"Lumi");
  }
  return;
}

void HLTPrescaleRecorder::endRun(edm::Run& iRun, const edm::EventSetup& iSetup) {
  if (run_) {
    auto_ptr<HLTPrescaleTable> product (new HLTPrescaleTable(hlt_));
    iRun.put(product,"Run");
  }

  const unsigned int n(hlt_.size());
  const vector<string>& labels(hlt_.labels());
  string out("");
  for (unsigned int i=0; i!=n; ++i) {out += " " + labels[i];}
  LogInfo("HLTPrescaleRecorder") << n << ": " << out;

  return;
}
