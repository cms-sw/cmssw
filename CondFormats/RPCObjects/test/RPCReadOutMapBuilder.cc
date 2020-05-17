// system include files
#include <memory>
#include <iostream>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "CondFormats/RPCObjects/interface/DccSpec.h"
#include "CondFormats/RPCObjects/interface/TriggerBoardSpec.h"
#include "CondFormats/RPCObjects/interface/LinkConnSpec.h"
#include "CondFormats/RPCObjects/interface/LinkBoardSpec.h"
#include "CondFormats/RPCObjects/interface/ChamberLocationSpec.h"
#include "CondFormats/RPCObjects/interface/FebLocationSpec.h"
#include "CondFormats/RPCObjects/interface/FebConnectorSpec.h"
#include "CondFormats/RPCObjects/interface/ChamberStripSpec.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

using namespace std;
using namespace edm;

class RPCReadOutMapBuilder : public edm::EDAnalyzer {
public:
  explicit RPCReadOutMapBuilder(const edm::ParameterSet&);
  ~RPCReadOutMapBuilder() override;
  void beginJob() override;
  void endJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override {}

private:
  RPCReadOutMapping* cabling;
  string m_record;
};

RPCReadOutMapBuilder::RPCReadOutMapBuilder(const edm::ParameterSet& iConfig)
    : m_record(iConfig.getParameter<std::string>("record")) {
  cout << " HERE record: " << m_record << endl;
  ::putenv(const_cast<char*>(std::string("CORAL_AUTH_USER=me").c_str()));
  ::putenv(const_cast<char*>(std::string("CORAL_AUTH_PASSWORD=test").c_str()));
}

RPCReadOutMapBuilder::~RPCReadOutMapBuilder() { cout << "DTOR called" << endl; }

// ------------ method called to store map -------------------
void RPCReadOutMapBuilder::endJob() {
  cout << "Now writing to DB" << endl;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    cout << "db service unavailable" << endl;
    return;
  } else {
    cout << "DB service OK" << endl;
  }

  try {
    if (mydbservice->isNewTagRequest(m_record)) {
      mydbservice->createNewIOV<RPCReadOutMapping>(
          cabling, mydbservice->beginOfTime(), mydbservice->endOfTime(), m_record);
    } else {
      mydbservice->appendSinceTime<RPCReadOutMapping>(cabling, mydbservice->currentTime(), m_record);
    }
  } catch (std::exception& e) {
    cout << "std::exception:  " << e.what();
  } catch (...) {
    cout << "Unknown error caught " << endl;
  }
  cout << "... all done, end" << endl;
}

// ------------ method called to produce the data  ------------
void RPCReadOutMapBuilder::beginJob() {
  cout << "BeginJob method " << endl;
  cout << "Building RPC Cabling" << endl;
  cabling = new RPCReadOutMapping("My map V-TEST");
  {
    DccSpec dcc(790);
    for (int idtb = 1; idtb <= 68; idtb++) {
      TriggerBoardSpec tb(idtb);

      for (int idlc = 0; idlc <= 17; idlc++) {
        LinkConnSpec lc(idlc);
        for (int idlb = 0; idlb <= 2; idlb++) {
          bool master = (idlb == 0);
          LinkBoardSpec lb(master, idlb, 0);
          for (int ifeb = 0; ifeb <= 5; ifeb++) {
            FebLocationSpec febLocation = {3, 2, 1, 2};
            ChamberLocationSpec chamber = {1, 5, 3, 1, 1, 1, 1};
            FebConnectorSpec febConn(ifeb, chamber, febLocation);
            /*              for (int istrip=0; istrip <= 15; istrip++) {
              int chamberStrip = ifeb*16+istrip;
              int cmsStrip = chamberStrip;
              ChamberStripSpec strip = {istrip, chamberStrip, cmsStrip};
              febConn.add( strip);
            } */
            //            febConn.addStrips(16,1,1,1,1);
            lb.add(febConn);
          }
          lc.add(lb);
        }
        tb.add(lc);
      }
      dcc.add(tb);
    }
    cabling->add(dcc);
  }
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RPCReadOutMapBuilder);
