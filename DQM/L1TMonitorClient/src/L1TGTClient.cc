#include "DQM/L1TMonitorClient/interface/L1TGTClient.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "DQMServices/Core/interface/MonitorElementBaseT.h"
#include "DQMServices/ClientConfig/interface/SubscriptionHandle.h"
#include "DQMServices/ClientConfig/interface/QTestHandle.h"
#include <DQMServices/UI/interface/MonitorUIRoot.h>
#include "DQMServices/CoreROOT/interface/MonitorElementRootT.h"
#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TPaveStats.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TF1.h>
#include <TRandom.h>
using namespace edm;
using namespace std;

L1TGTClient::L1TGTClient(const edm::ParameterSet& iConfig): L1TBaseClient()
{


  saveOutput = iConfig.getUntrackedParameter<bool>("saveOutput", false);
  outputFile = iConfig.getUntrackedParameter<string>("outputFile", "L1TGTClient.root");
  stdalone = iConfig.getUntrackedParameter<bool>("Standalone",false);
  qualityCriterionName = iConfig.getUntrackedParameter<string>("qualityTestName","testYRange");
  
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  dbe->showDirStructure();
  dbe->setVerbose(1);

  if(stdalone){ 
  getMESubscriptionListFromFile = iConfig.getUntrackedParameter<bool>("getMESubscriptionListFromFile", true);
  getQualityTestsFromFile = iConfig.getUntrackedParameter<bool>("getQualityTestsFromFile", true);

  subscriber=new SubscriptionHandle;
  qtHandler=new QTestHandle;

  if (getMESubscriptionListFromFile)
  subscriber->getMEList("MESubscriptionList.xml"); 
  if (getQualityTestsFromFile)
  qtHandler->configureTests("QualityTests.xml",mui_);

  }
  
  LogInfo( "TriggerDQM");
}

// ---------------------------------------------------------

L1TGTClient::~L1TGTClient()
{
 
 LogInfo("TriggerDQM")<<"[TriggerDQM]: ending... ";

}


// ------------ method called to for each event  ------------

void
L1TGTClient::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

// access to Geometry if needed

/*    
edm::ESHandle<DTGeometry> pDT;
iSetup.get<MuonGeometryRecord>().get( pDT );     
for ( std::vector<GeomDet*>::const_iterator iGeomDet = pDT->dets().begin(); iGeomDet != pDT->dets().end(); iGeomDet++ ) this->fillTree( *iGeomDet );
*/


  nevents++;
  if (!stdalone || (nevents%10 == 0))    LogInfo("TriggerDQM")<<"[TriggerDQM]: event analyzed "<<nevents;

}

// ------------ method called once each job just before starting event loop  ------------

void L1TGTClient::beginJob(const edm::EventSetup&)
{

  LogInfo("TriggerDQM")<<"[TriggerDQM]: Begin Job";
  LogInfo("TriggerDQM")<<"[TriggerDQM]: Standalone = "<<stdalone;
  nevents = 0;
  dbe->setCurrentFolder("L1TMonitor/QTests");
  normGTFEBx = dbe->book1D("normGTFEBx", "normGTFEBx",  100, 0., 5000.);


  if(stdalone){
   subscriber->makeSubscriptions(mui_);
   qtHandler->attachTests(mui_);	
  }

}

// ------------ method called once each job just after ending the event loop  ------------
void L1TGTClient::endJob() {

   LogInfo("TriggerDQM")<<"[TriggerDQM]: endJob";

}

void L1TGTClient::endLuminosityBlock(const edm::LuminosityBlock & iLumiSection, const edm::EventSetup & iSetup) {

  LogInfo("TriggerDQM")<<"[TriggerDQM]: end Lumi Section.";

// if(stdalone) mui_->doMonitoring();


// ----------------- QT examples example



// ----------------- get bin content and create new ME, then perform QT

  TH1F * bxHisto = this->get1DHisto("L1TMonitor/L1TGT/GT FE Bx", dbe);
	 
    if(bxHisto) {
	
        int nEntries = bxHisto->GetEntries(); 
	 
        int lastBinX=(*bxHisto).GetNbinsX();
	 

           for(int xBin=1; xBin<=lastBinX; xBin++) {

                float xContent = bxHisto->GetBinContent(xBin);
                xContent = xContent/nEntries;
                normGTFEBx->setBinContent(xBin, xContent);
		
 
           }
     }


// ----------------- save results
   
     if(stdalone){
	     mui_->runQTests();
             qtHandler->checkGolbalQTStatus(mui_);
     }
  
  this->getReport("L1TMonitor/QTests/normGTFEBx", dbe, qualityCriterionName);
 
 	      dbe->save(outputFile);

}

