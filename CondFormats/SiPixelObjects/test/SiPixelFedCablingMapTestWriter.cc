// system include files
#include <memory>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDLink.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

class EventSetup;
class Event;

using namespace std;
using namespace edm;
using namespace sipixelobjects;

class SiPixelFedCablingMapTestWriter : public edm::EDAnalyzer {
public:
  explicit SiPixelFedCablingMapTestWriter(const edm::ParameterSet&);
  ~SiPixelFedCablingMapTestWriter() override;
  void beginJob() override;
  void endJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override {}

private:
  SiPixelFedCablingTree* cablingTree;
  string m_record;
};

SiPixelFedCablingMapTestWriter::SiPixelFedCablingMapTestWriter(const edm::ParameterSet& iConfig)
    : cablingTree(nullptr), m_record(iConfig.getParameter<std::string>("record")) {
  cout << " HERE record: " << m_record << endl;
  ::putenv((char*)"CORAL_AUTH_USER=konec");
  ::putenv((char*)"CORAL_AUTH_PASSWORD=test");
}

void SiPixelFedCablingMapTestWriter::endJob() {
  cout << "Convert Tree to Map";

  SiPixelFedCablingMap* cablingMap = new SiPixelFedCablingMap(cablingTree);
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
      mydbservice->createNewIOV<SiPixelFedCablingMap>(
          cablingMap, mydbservice->beginOfTime(), mydbservice->endOfTime(), m_record);
    } else {
      mydbservice->appendSinceTime<SiPixelFedCablingMap>(cablingMap, mydbservice->currentTime(), m_record);
    }
  } catch (std::exception& e) {
    cout << "std::exception:  " << e.what();
  } catch (...) {
    cout << "Unknown error caught " << endl;
  }
  cout << "... all done, end" << endl;
}

SiPixelFedCablingMapTestWriter::~SiPixelFedCablingMapTestWriter() { cout << "DTOR !" << endl; }

// ------------ method called to produce the data  ------------
void SiPixelFedCablingMapTestWriter::beginJob() {
  cout << "BeginJob method " << endl;
  cout << "Building FED Cabling" << endl;
  cablingTree = new SiPixelFedCablingTree("My map V-TEST");

  PixelROC r1;
  PixelROC r2;

  PixelFEDLink link(2);
  PixelFEDLink::ROCs rocs;
  rocs.push_back(r1);
  rocs.push_back(r2);
  link.add(rocs);

  PixelFEDCabling fed(0);
  fed.addLink(link);
  cablingTree->addFed(fed);
  cout << "PRINTING MAP:" << endl << cablingTree->print(3) << endl;
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiPixelFedCablingMapTestWriter);
