#include "DQMOffline/Hcal/interface/HcalNoiseRatesClient.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

HcalNoiseRatesClient::HcalNoiseRatesClient(const edm::ParameterSet& iConfig):conf_(iConfig)
{

  outputFile_ = iConfig.getUntrackedParameter<std::string>("outputFile", "myfile.root");

  dbe_ = edm::Service<DQMStore>().operator->();
  if (!dbe_) {
    edm::LogError("HcalNoiseRatesClient") << "unable to get DQMStore service, upshot is no client histograms will be made";
  }
  if(iConfig.getUntrackedParameter<bool>("DQMStore", false)) {
    if(dbe_) dbe_->setVerbose(0);
  }
 
  debug_ = false;
  verbose_ = false;

  dirName_=iConfig.getParameter<std::string>("DQMDirName");
  if(dbe_) dbe_->setCurrentFolder(dirName_);
 
}


HcalNoiseRatesClient::~HcalNoiseRatesClient()
{ 
  
}

void HcalNoiseRatesClient::beginJob()
{
 

}

void HcalNoiseRatesClient::endJob() 
{
   if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}

void HcalNoiseRatesClient::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
 
}


void HcalNoiseRatesClient::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  runClient_();
}

//dummy analysis function
void HcalNoiseRatesClient::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{
  
}

void HcalNoiseRatesClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c)
{ 
//  runClient_();
}

void HcalNoiseRatesClient::runClient_()
{
  if(!dbe_) return; //we dont have the DQMStore so we cant do anything
  dbe_->setCurrentFolder(dirName_);

  if (verbose_) std::cout << "\nrunClient" << std::endl; 

  std::vector<MonitorElement*> hcalMEs;

  // Since out folders are fixed to three, we can just go over these three folders
  // i.e., CaloTowersD/CaloTowersTask, HcalRecHitsD/HcalRecHitTask, HcalNoiseRatesD/NoiseRatesTask.
  std::vector<std::string> fullPathHLTFolders = dbe_->getSubdirs();
  for(unsigned int i=0;i<fullPathHLTFolders.size();i++) {

    if (verbose_) std::cout <<"\nfullPath: "<< fullPathHLTFolders[i] << std::endl;
    dbe_->setCurrentFolder(fullPathHLTFolders[i]);

    std::vector<std::string> fullSubPathHLTFolders = dbe_->getSubdirs();
    for(unsigned int j=0;j<fullSubPathHLTFolders.size();j++) {

      if (verbose_) std::cout <<"fullSub: "<<fullSubPathHLTFolders[j] << std::endl;

      if( strcmp(fullSubPathHLTFolders[j].c_str(), "HcalNoiseRatesD/NoiseRatesTask") ==0  ){
         hcalMEs = dbe_->getContents(fullSubPathHLTFolders[j]);
         if (verbose_) std::cout <<"hltMES size : "<<hcalMEs.size()<<std::endl;
         if( !NoiseRatesEndjob(hcalMEs) ) std::cout<<"\nError in NoiseRatesEndjob!"<<std::endl<<std::endl;
      }

    }    

  }

}

// called after entering the HcalNoiseRatesD/NoiseRatesTask directory
// hcalMEs are within that directory
int HcalNoiseRatesClient::NoiseRatesEndjob(const std::vector<MonitorElement*> &hcalMEs){

   int useAllHistos = 0;
   MonitorElement* hLumiBlockCount =0;
   for(unsigned int ih=0; ih<hcalMEs.size(); ih++){
      if( strcmp(hcalMEs[ih]->getName().c_str(), "hLumiBlockCount") ==0  ){
         hLumiBlockCount = hcalMEs[ih];
         useAllHistos =1;
      } 
   } 
   if( useAllHistos !=0 && useAllHistos !=1 ) return 0;

// FIXME: dummy lumiCountMap.size since hLumiBlockCount is disabled
// in a general case.
   int lumiCountMapsize = -1; // dummy
   if (useAllHistos) hLumiBlockCount->Fill(0.0, lumiCountMapsize);

   return 1;

}

DEFINE_FWK_MODULE(HcalNoiseRatesClient);
