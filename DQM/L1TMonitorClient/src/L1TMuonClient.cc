#include "DQM/L1TMonitorClient/interface/L1TMuonClient.h"
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

L1TMuonClient::L1TMuonClient(const edm::ParameterSet& iConfig): L1TBaseClient()
{


  saveOutput = iConfig.getUntrackedParameter<bool>("saveOutput", false);
  outputFile = iConfig.getUntrackedParameter<string>("outputFile", "L1TMuonMonitor.root");
  ptCriterionName = iConfig.getUntrackedParameter<string>("ptTestName","dttfPTinRange");
  phiCriterionName = iConfig.getUntrackedParameter<string>("phiTestName","dttfPhiTest");
  qualityCriterionName = iConfig.getUntrackedParameter<string>("qualityTestName","dttfQualityInRange");
  stdalone = iConfig.getUntrackedParameter<bool>("Standalone",false);
  
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  dbe->showDirStructure();
  dbe->setVerbose(1);
  dbe->setCurrentFolder("L1TMonitor/QTests");

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

L1TMuonClient::~L1TMuonClient()
{
 
 LogInfo("TriggerDQM")<<"[TriggerDQM]: ending... ";

}


// ------------ method called to for each event  ------------

void
L1TMuonClient::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
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

  if (!stdalone || (nevents%50 == 0)){
  

  if(stdalone) mui_->doMonitoring();


// ----------------- QT examples example

  this->getReport("Collector/GlobalDQM/L1TMonitor/L1TDTTPG/dttf_p_pt", dbe, ptCriterionName);

  this->getReport("Collector/GlobalDQM/L1TMonitor/L1TDTTPG/dttf_p_phi", dbe, phiCriterionName);

  this->getReport("Collector/GlobalDQM/L1TMonitor/L1TDTTPG/dttf_p_qual", dbe, qualityCriterionName);

  this->getReport("Collector/GlobalDQM/L1TMonitor/L1TDTTPG/DT_TPG_phi_quality", dbe, qualityCriterionName);



// ----------------- get bin content and create new ME, then perform QT

  
  TH1F * bxHisto = this->get1DHisto("Collector/GlobalDQM/L1TMonitor/L1TGT/GT FE Bx", dbe);
	 
    if(bxHisto) {
	
//        LogInfo("TriggerDQM") << "Number entries in test_ = " << bxHisto->GetEntries(); 
	 
        int lastBinX=(*bxHisto).GetNbinsX();
	 

           for(int xBin=1; xBin<=lastBinX; xBin++) {

                float xContent = bxHisto->GetBinContent(xBin);

 
                 newME->setBinContent(xBin, xContent);

           }
     }

	  
  this->getReport("L1TMonitor/QTests/newME_", dbe, qualityCriterionName);

       
// ----------------- fit example

   for(unsigned i = 0; i != 100; ++i) gausExample->Fill(gRandom->Gaus(3., 2.));

// ME -> Histogram	  

   TH1F * gausExampleHisto = this->get1DHisto("L1TMonitor/QTests/Gaussian", dbe);

   TF1 *gaussfit=new TF1("gaussfit","gaus");

	  if (gausExampleHisto) {
	      
                float nentries = gausExampleHisto->GetEntries();
	      
 		float mean,rms;
		Double_t par;
		mean=gausExampleHisto->GetMean();
		rms=gausExampleHisto->GetRMS();
                LogInfo("TriggerDQM") << "MEAN = " << mean << "  RMS = " << rms; 
		gausExampleHisto->Fit("gaussfit","QL","",mean-3*rms,mean+3*rms);
		// get values from gaussian fit
		par = gaussfit->GetParameter(2);
		meanfit.push_back(par);
               
           }

// ----------------- save results
   
	    if(stdalone){
	     mui_->runQTests();
             qtHandler->checkGolbalQTStatus(mui_);
            }
  
 	    if ( saveOutput && nevents%5 == 0) {
 
// 	       for (int it=0;  it < meanfit.size(); it++) {

// 	    	MeanFitME->setBinContent(it,meanfit[it]);
 
// 	       }
 
 	      dbe->save(outputFile);
// 	      meanfit.clear();
 	    }

    }

}

// ------------ method called once each job just before starting event loop  ------------

void L1TMuonClient::beginJob(const edm::EventSetup&)
{

  LogInfo("TriggerDQM")<<"[TriggerDQM]: Begin Job";
  LogInfo("TriggerDQM")<<"[TriggerDQM]: Standalone = "<<stdalone;
  nevents = 0;

  PtTestBadChannels = dbe->book1D("ptTestBadChannels", "ptTestBadChannels", 32, -0.5, 31.5);

  PhiTestBadChannels = dbe->book1D("phiTestBadChannels", "phiTestBadChannels", 256, -0.5, 255.5);

  QualTestBadChannels = dbe->book1D("qualTestBadChannels", "qualTestBadChannels", 8, -0.5, 7.5);

  QualTestBadChannels_ = dbe->book1D("qualTestBadChannels_", "qualTestBadChannels_", 8, -0.5, 7.5);
  
  testBadChannels_ = dbe->book1D("testBadChannels_", "testBadChannels_", 8, -0.5, 7.5);
  
  MeanFitME = dbe->book1D("meanFitME", "meanFitME", 22, 0, 22);

  newME = dbe->book1D("newME_", "newME_",  100, 0., 5000.);

  gausExample = dbe->book1D("Gaussian", "Gaussian",  50, 0., 6.);
  

  if(stdalone){
  subscriber->makeSubscriptions(mui_);
  qtHandler->attachTests(mui_);	
  }

}

// ------------ method called once each job just after ending the event loop  ------------
void L1TMuonClient::endJob() {

   LogInfo("TriggerDQM")<<"[TriggerDQM]: endJob";

}

