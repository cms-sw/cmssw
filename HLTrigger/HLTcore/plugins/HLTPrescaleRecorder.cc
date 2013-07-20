/** \class HLTPrescaleRecorder
 *
 * See header file for documentation
 *
 *  $Date: 2013/05/17 20:33:55 $
 *  $Revision: 1.8 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTPrescaleRecorder.h"

#include "CondFormats/HLTObjects/interface/HLTPrescaleTableCond.h"
#include "CondFormats/DataRecord/interface/HLTPrescaleTableRcd.h"

#include "DataFormats/Provenance/interface/ProcessHistory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"

#include <sys/time.h>
#include "DataFormats/Provenance/interface/Timestamp.h"

#include<string>
#include<ostream>

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
  condDB_(ps.getParameter<bool>("condDB")),
  psetName_(ps.getParameter<string>("psetName")),
  hltInputTag_(ps.getParameter<InputTag>("hltInputTag")),
  hltDBTag_(ps.getParameter<string>("hltDBTag")),
  ps_(0),
  db_(0),
  hltHandle_(),
  hltESHandle_(),
  hlt_()
{

  LogInfo("HLTPrescaleRecorder")
    << "src:run-lumi-event-condDB+psetName+tags: "
    << src_ << ":" << run_ << "-" << lumi_ << "-" << event_ << "-"
    << condDB_ << "+" << psetName_ << "+"
    << hltInputTag_.encode() << "+" << hltDBTag_;

  if(edm::Service<edm::service::PrescaleService>().isAvailable()) {
    ps_ = edm::Service<edm::service::PrescaleService>().operator->();
  } else if (src_==0) {
    LogError("HLTPrescaleRecorder")<<"PrescaleService requested as source but unavailable!";
  }

  if (edm::Service<cond::service::PoolDBOutputService>().isAvailable()) {
    db_ = edm::Service<cond::service::PoolDBOutputService>().operator->();
  } else if (condDB_) {
    LogError("HLTPrescaleRecorder")<<"PoolDBOutputService requested as destination but unavailable!";
  }

  if (run_)   produces<HLTPrescaleTable,edm::InRun>("Run");
  if (lumi_)  produces<HLTPrescaleTable,edm::InLumi>("Lumi");
  if (event_) produces<HLTPrescaleTable,edm::InEvent>("Event");

}

HLTPrescaleRecorder::~HLTPrescaleRecorder()
{
}

//
// member functions
//

void HLTPrescaleRecorder::beginRun(edm::Run const& iRun, const edm::EventSetup& iSetup) {

  hlt_=HLTPrescaleTable();

  if (src_==-1) {
    /// From PrescaleTable tracked PSet
    ParameterSet pPSet(getProcessParameterSet());
    ParameterSet iPS(pPSet.getParameter<ParameterSet>(psetName_));

    string defaultLabel(iPS.getParameter<std::string>("lvl1DefaultLabel"));
    vector<string> labels(iPS.getParameter<std::vector<std::string> >("lvl1Labels"));
    vector<ParameterSet> vpTable(iPS.getParameter<std::vector<ParameterSet> >("prescaleTable"));

    unsigned int set(0);
    const unsigned int n(labels.size());
    for (unsigned int i=0; i!=n; ++i) {
      if (labels[i]==defaultLabel) set=i;
    }

    map<string,vector<unsigned int> > table;
    const unsigned int m (vpTable.size());
    for (unsigned int i=0; i!=m; ++i) {
      table[vpTable[i].getParameter<std::string>("pathName")] = 
	vpTable[i].getParameter<std::vector<unsigned int> >("prescales");
    }
    hlt_=HLTPrescaleTable(set,labels,table);

  } else  if (src_==0) {
    /// From PrescaleService
    /// default index updated at lumi block boundaries
    if (ps_!=0) {
      hlt_=HLTPrescaleTable(ps_->getLvl1IndexDefault(), ps_->getLvl1Labels(), ps_->getPrescaleTable());
    } else {
      hlt_=HLTPrescaleTable();
      LogError("HLTPrescaleRecorder")<<"PrescaleService not found!";
    }
  } else if (src_==1) {
    /// From Run Block
    if (iRun.getByLabel(hltInputTag_,hltHandle_)) {
      hlt_=*hltHandle_;
    } else {
      LogError("HLTPrescaleRecorder")<<"HLTPrescaleTable not found in Run!";
    }
  } else if (src_==4) {
    /// From CondDB (needs ESProducer module as well)
    const HLTPrescaleTableRcd& hltRecord(iSetup.get<HLTPrescaleTableRcd>());
    hltRecord.get(hltDBTag_,hltESHandle_);
    hlt_=hltESHandle_->hltPrescaleTable();
  }

  return;
}

void HLTPrescaleRecorder::beginLuminosityBlock(edm::LuminosityBlock const& iLumi, const edm::EventSetup& iSetup) {

  if (src_==0) {
    /// From PrescaleService
    /// default index updated at lumi block boundaries
    if (ps_!=0) {
      hlt_=HLTPrescaleTable(ps_->getLvl1IndexDefault(), ps_->getLvl1Labels(), ps_->getPrescaleTable());
    } else {
      hlt_=HLTPrescaleTable();
      LogError("HLTPrescaleRecorder")<<"PrescaleService not found!";
    }
  } else if (src_==2) {
    /// From Lumi Block
    if (iLumi.getByLabel(hltInputTag_,hltHandle_)) {
      hlt_=*hltHandle_;
    } else {
      hlt_=HLTPrescaleTable();
      LogError("HLTPrescaleRecorder")<<"HLTPrescaleTable not found in LumiBlock!";
    }
  }

  return;
}

void HLTPrescaleRecorder::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  if (src_==3) {
    /// From Event Block
    if (iEvent.getByLabel(hltInputTag_,hltHandle_)) {
      hlt_=*hltHandle_;
    } else {
      hlt_=HLTPrescaleTable();
      LogError("HLTPrescaleRecorder")<<"HLTPrescaleTable not found in Event!";
    }
  }

  if (event_) {
    /// Writing to Event
    auto_ptr<HLTPrescaleTable> product (new HLTPrescaleTable(hlt_));
    iEvent.put(product,"Event");
  }

  return;
}
void HLTPrescaleRecorder::endLuminosityBlock(edm::LuminosityBlock const& iLumi, const edm::EventSetup& iSetup) {
}

void HLTPrescaleRecorder::endLuminosityBlockProduce(edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup) {

  if (lumi_) {
    /// Writing to Lumi Block
    auto_ptr<HLTPrescaleTable> product (new HLTPrescaleTable(hlt_));
    iLumi.put(product,"Lumi");
  }
  return;
}

void HLTPrescaleRecorder::endRun(edm::Run const& iRun, const edm::EventSetup& iSetup) {

  /// Dump to logfile
  ostringstream oss;
  const unsigned int n(hlt_.size());
  oss << "PrescaleTable: # of labels = " << n << endl;
  const vector<string>& labels(hlt_.labels());
  for (unsigned int i=0; i!=n; ++i) {
    oss << " " << i << "/'" << labels.at(i) << "'";
  }
  oss << endl;
  const map<string,vector<unsigned int> >& table(hlt_.table());
  oss << "PrescaleTable: # of paths = " << table.size() << endl;
  const map<string,vector<unsigned int> >::const_iterator tb(table.begin());
  const map<string,vector<unsigned int> >::const_iterator te(table.end());
  for (map<string,vector<unsigned int> >::const_iterator ti=tb; ti!=te; ++ti) {
    for (unsigned int i=0; i!=n; ++i) {
      oss << " " << ti->second.at(i);
    }
    oss << " " << ti->first << endl;
  }
  LogVerbatim("HLTPrescaleRecorder") << oss.str();

  if (condDB_) {
    /// Writing to CondDB (needs PoolDBOutputService)
    if (db_!=0) {
      HLTPrescaleTableCond* product (new HLTPrescaleTableCond(hlt_));
      const string rcdName("HLTPrescaleTableRcd");
      if ( db_->isNewTagRequest(rcdName) ) {
	db_->createNewIOV<HLTPrescaleTableCond>(product,
	      db_->beginOfTime(),db_->endOfTime(),rcdName);
      } else {
	::timeval tv;
	gettimeofday(&tv,0);
	edm::Timestamp tstamp((unsigned long long)tv.tv_sec);
	db_->appendSinceTime<HLTPrescaleTableCond>(product,
//            db_->currentTime()
              tstamp.value()
		,rcdName);
      }
    } else {
      LogError("HLTPrescaleRecorder") << "PoolDBOutputService not available!";
    }
  }

  return;
}

void HLTPrescaleRecorder::endRunProduce(edm::Run& iRun, const edm::EventSetup& iSetup) {
   if (run_) {
     /// Writing to Run Block
     auto_ptr<HLTPrescaleTable> product (new HLTPrescaleTable(hlt_));
     iRun.put(product,"Run");
   }
}
