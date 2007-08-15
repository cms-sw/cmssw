#include "DQM/L1TMonitorClient/interface/L1TCaloClient.h"
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
#include <TRandom.h>
using namespace edm;
using namespace std;

L1TCaloClient::L1TCaloClient(const edm::ParameterSet& iConfig): L1TBaseClient()
{


  saveOutput = iConfig.getUntrackedParameter<bool>("saveOutput", false);
  outputFile = iConfig.getUntrackedParameter<string>("outputFile", "L1TMonitor.root");
  occCriterionName = iConfig.getUntrackedParameter<string>("occTestName","occInRange");
  stdalone = iConfig.getUntrackedParameter<bool>("Standalone",false);
  
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  dbe->showDirStructure();
  dbe->setVerbose(1);
  dbe->setCurrentFolder("L1TMonitor/Qtests");
  
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

L1TCaloClient::~L1TCaloClient()
{
 
 LogInfo("TriggerDQM")<<"[TriggerDQM]: ending... ";


}


// ------------ method called to for each event  ------------

void
L1TCaloClient::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
// access to Geometry if needed

/*    
edm::ESHandle<CaloGeometry> gHandle;
iSetup.get<IdealGeometryRecord>().get(gHandle);     
const CaloSubdetectorGeometry barrelGeometry = gHandle->getSubdetectorGeometry(DetId::Ecal,EcalBarrel);
*/

  nevents++;
  if (!stdalone || (nevents%10 == 0))    LogInfo("TriggerDQM")<<"[TriggerDQM]: event analyzed "<<nevents;

  if (!stdalone || (nevents%50 == 0)){
  

  if(stdalone) mui_->doMonitoring();

// ME -> Histogram	  


// ----------------- QT examples example

  this->getReport("Collector/GlobalDQM/L1TMonitor/RCT/RctIsoEmOccEtaPhi", dbe, occCriterionName);

          MonitorElement * occupancy =	dbe->get("Collector/GlobalDQM/L1TMonitor/RCT/RctIsoEmOccEtaPhi");
	  
	  if(occupancy)  LogInfo("TriggerDQM") << "Found ME !!!"; 
	  
	  MonitorElementT<TNamed>* occup_temp = dynamic_cast<MonitorElementT<TNamed>*>(occupancy);
           
       if (occup_temp) {
	      
        TH2F * occupancyHisto = dynamic_cast<TH2F*> (occup_temp->operator->());
 
         if (occupancyHisto) {

           int lastBinX=(*occupancyHisto).GetNbinsX();

 
           int lastBinY=(*occupancyHisto).GetNbinsY();

           TH1D* prox=occupancyHisto->ProjectionX(); // X projection

           TH1D* proy=occupancyHisto->ProjectionY(); // Y projection
       
////////////////  X occupancy profile (eta)

	       for(int xBin=1; xBin<=lastBinX; xBin++) {
	
	         if(prox->GetBinContent(xBin)!=0){
	
	           float xOccupContent = prox->GetBinContent(xBin);
	
	           MEprox->setBinContent(xBin, xOccupContent);
	
	         }
	
	       }
 

////////////////  Y occupancy profile (phi?)

	       for(int yBin=1; yBin<=lastBinY; yBin++) {
	
	         if(proy->GetBinContent(yBin)!=0){
	
	         float yOccupContent = proy->GetBinContent(yBin);
	
	         MEproy->setBinContent(yBin, yOccupContent);
	
	         }
	 
	       }
       
             }

	   }
       
         }
    
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

// ------------ method called once each job just before starting event loop  ------------

void L1TCaloClient::beginJob(const edm::EventSetup&)
{

  LogInfo("TriggerDQM")<<"[TriggerDQM]: Begin Job";
  
  nevents = 0;

  MEprox = dbe->book1D("RctIsoEmOccEtaPhi_prox", "isoEmOccProX", 18, 0, 18);

  MEproy = dbe->book1D("RctIsoEmOccEtaPhi_proy", "isoEmOccProY", 22, 0, 22);
  

}

// ------------ method called once each job just after ending the event loop  ------------
void L1TCaloClient::endJob() {

   LogInfo("TriggerDQM")<<"[TriggerDQM]: endJob";

}




























































/*
    std::vector<std::string> contentVec;
   dbe->getContents(contentVec);
   
  for (std::vector<std::string>::iterator it = contentVec.begin();
       it != contentVec.end(); it++) {

       std::string::size_type dirCharNumber = it->find( ":", 0 );

       std::string dirName=it->substr(0 , dirCharNumber);

       dirName+= "/";

       std::string meCollectionName=it->substr(dirCharNumber+1);

       int CollectionNameSize = meCollectionName.length();

       std::string::size_type SourceCharNumber = it->rfind("/");

       std::string sourceName = it->substr(SourceCharNumber+1);

       int sourceNameSize = sourceName.length()-CollectionNameSize-1;
      
       sourceName = it->substr(SourceCharNumber+1, sourceNameSize);
       
//       if(source != sourceName) continue; // return only ME belonging to the source calling
                                          // this function
       std::string reminingNames=meCollectionName;
       bool anotherME=true;

       while(anotherME){
       if(reminingNames.find(",") == std::string::npos) anotherME =false;
       std::string::size_type singleMeNameCharNumber= reminingNames.find( ",", 0 );
       std::string singleMeName=reminingNames.substr(0 , singleMeNameCharNumber );

         
       if(singleMeName != "EcalTpOccEtaPhi") continue;
    	
	MonitorElement * occupancy = dbe->get(singleMeName);



       string MeanCriterionName = parameters.getUntrackedParameter<string>("meanTestName","NoiseMeanInRange");
       const QReport * theMeanQReport = occupancy->getQReport(MeanCriterionName);
           if(theMeanQReport) {
             vector<dqm::me_util::Channel> badChannels = theMeanQReport->getBadChannels();
             for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin();
        	  channel != badChannels.end(); channel++) {
               LogInfo("tTrigCalibration") << " Bad mean channels: " << (*channel).getBin() << "  Contents : "<< (*channel).getContents();
               LogInfo("tTrigCalibration") << theMeanQReport->getMessage() << " ------- " << theMeanQReport->getStatus();
             }
           }




       }



// this is the way to get histograms


dbe->save("MeanTest.root");
    }
*/
