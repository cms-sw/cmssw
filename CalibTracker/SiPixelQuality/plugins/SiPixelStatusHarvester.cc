
/*
 *  See header file for a description of this class.
 *
 *  author
 */

// CMSSW FW
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// Condition Format
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityFromDbRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
// CondOutput
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// Dataformat of SiPixel status in ALCAPROMPT data
#include "CalibTracker/SiPixelQuality/interface/SiPixelDetectorStatus.h"
//#include "CondCore/Utilities/bin/cmscond_export_iov.cpp"
//#include "CondCore/Utilities/interface/Utilities.h"

// harvest helper class
#include "CalibTracker/SiPixelQuality/interface/SiPixelStatusManager.h"
// header file
#include "CalibTracker/SiPixelQuality/plugins/SiPixelStatusHarvester.h"

#include <iostream> 
#include <cstring>

using namespace edm;

//--------------------------------------------------------------------------------------------------
SiPixelStatusHarvester::SiPixelStatusHarvester(const edm::ParameterSet& iConfig) :
  outputBase_(iConfig.getParameter<ParameterSet>("SiPixelStatusManagerParameters").getUntrackedParameter<std::string>("outputBase")),
  aveDigiOcc_(iConfig.getParameter<ParameterSet>("SiPixelStatusManagerParameters").getUntrackedParameter<int>("aveDigiOcc", 20000)),
  nLumi_(iConfig.getParameter<edm::ParameterSet>("SiPixelStatusManagerParameters").getUntrackedParameter<int>("resetEveryNLumi")),
  moduleName_(iConfig.getParameter<ParameterSet>("SiPixelStatusManagerParameters").getUntrackedParameter<std::string>("moduleName")),
  label_(iConfig.getParameter<ParameterSet>("SiPixelStatusManagerParameters").getUntrackedParameter<std::string>("label")),
  siPixelStatusManager_(iConfig, consumesCollector()) {  

  recordName_ = iConfig.getUntrackedParameter<std::string>("recordName", "SiPixelQualityFromDbRcd");
  debug_ = iConfig.getUntrackedParameter<bool>("debug",false);
  dumpTxt_ = iConfig.getUntrackedParameter<bool>("dumpTxt",false);
  outTxtFileName_ = iConfig.getUntrackedParameter<std::string>("txtFileName");
  
}

//--------------------------------------------------------------------------------------------------
SiPixelStatusHarvester::~SiPixelStatusHarvester(){}

//--------------------------------------------------------------------------------------------------
void SiPixelStatusHarvester::beginJob() { }

//--------------------------------------------------------------------------------------------------
void SiPixelStatusHarvester::endJob() { }  

//--------------------------------------------------------------------------------------------------
void SiPixelStatusHarvester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
}

//--------------------------------------------------------------------------------------------------
void SiPixelStatusHarvester::beginRun(const edm::Run&, const edm::EventSetup& iSetup){
  siPixelStatusManager_.reset();
  endLumiBlock_ = 0;

  edm::ESHandle<SiPixelQuality> qualityInfo;
  iSetup.get<SiPixelQualityFromDbRcd>().get( qualityInfo );
  badPixelInfo_ = qualityInfo.product();

}

//--------------------------------------------------------------------------------------------------
void SiPixelStatusHarvester::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup){

  siPixelStatusManager_.createPayloads();

  std::map<edm::LuminosityBlockNumber_t,std::map<int,std::vector<int>> > stuckTBMsMap = siPixelStatusManager_.getStuckTBMsRocs();

  std::map<edm::LuminosityBlockNumber_t,SiPixelDetectorStatus> siPixelStatusMap = siPixelStatusManager_.getBadComponents();
  edm::Service<cond::service::PoolDBOutputService> poolDbService;

  if(poolDbService.isAvailable() ) {// if(poolDbService.isAvailable() )

    // start producing tag for permanent component removed
    SiPixelQuality *siPixelQualityPermBad = new SiPixelQuality();
    const std::vector<SiPixelQuality::disabledModuleType> badComponentList = badPixelInfo_->getBadComponentList();
    for(unsigned int i = 0; i<badComponentList.size();i++){
        siPixelQualityPermBad->addDisabledModule(badComponentList[i]);
    }

    // stuckTBM tag from FED error 25 with permanent component removed
    for(SiPixelStatusManager::stuckTBMsMap_iterator it=stuckTBMsMap.begin(); it!=stuckTBMsMap.end();it++){

          cond::Time_t thisIOV = 1;
          edm::LuminosityBlockID lu(iRun.id().run(),it->first);
          thisIOV = (cond::Time_t)(lu.value());

          SiPixelQuality *siPixelQuality = new SiPixelQuality();

          std::map<int, std::vector<int> > tmpStuckTBMs = it->second;
          for(std::map<int, std::vector<int> >::iterator ilist = tmpStuckTBMs.begin(); ilist!=tmpStuckTBMs.end();ilist++){

             int detid = ilist->first;

             SiPixelQuality::disabledModuleType BadModule;

             BadModule.DetID = uint32_t(detid);
             BadModule.errorType = 3;

             BadModule.BadRocs = 0;
             std::vector<uint32_t> BadRocList;
             std::vector<int> list = ilist->second;

             for(unsigned int i=0; i<list.size();i++){
                 // only include rocs that are not permanent known bad
                 int iroc =  list[i];
                 if(!badPixelInfo_->IsRocBad(detid, iroc)){
                   BadRocList.push_back(uint32_t(iroc));
                 }
             }
             // change module error type if all ROCs are bad
             if(BadRocList.size()==16) 
                BadModule.errorType = 0;

             short badrocs = 0;
             for(std::vector<uint32_t>::iterator iter = BadRocList.begin(); iter != BadRocList.end(); ++iter){
                   badrocs +=  1 << *iter; // 1 << *iter = 2^{*iter} using bitwise shift
             }
             // fill the badmodule only if there is(are) bad ROC(s) in it
             if(badrocs!=0){
               BadModule.BadRocs = badrocs;
               siPixelQuality->addDisabledModule(BadModule);
             }
          }

          if (poolDbService->isNewTagRequest(recordName_+"_stuckTBM") ) {
              edm::LogInfo("SiPixelStatusHarvester")
                 << "new tag requested for stuckTBM" << std::endl;
              poolDbService->writeOne<SiPixelQuality>(siPixelQuality, thisIOV, recordName_+"_stuckTBM");
          }
          else {
             edm::LogInfo("SiPixelStatusHarvester")
                << "no new tag requested, appending IOV for stuckTBM" << std::endl;
              poolDbService->writeOne<SiPixelQuality>(siPixelQuality, thisIOV, recordName_+"_stuckTBM");
          }

    }

    // Payload for PCL combines permanent bad/stuckTBM/other
    for(SiPixelStatusManager::siPixelStatusMap_iterator it=siPixelStatusMap.begin(); it!=siPixelStatusMap.end();it++){

          cond::Time_t thisIOV = 1;

          if (outputBase_ == "runbased") {
               thisIOV = (cond::Time_t) iRun.id().run();
          }
          else if (outputBase_ == "nLumibased" || outputBase_ == "dynamicLumibased" ) {
             edm::LuminosityBlockID lu(iRun.id().run(),it->first);
             thisIOV = (cond::Time_t)(lu.value());
          }

          SiPixelDetectorStatus tmpSiPixelStatus = it->second;
          double DetAverage = tmpSiPixelStatus.perRocDigiOcc();

          // For the IOV of which the statistics is too low, for e.g., a cosmic run
          // When using dynamicLumibased harvester or runbased harvester
          // this only happens when the full run is lack of statistics 
          if(DetAverage<aveDigiOcc_) {

            edm::LogInfo("SiPixelStatusHarvester")
                 << "Tag requested for prompt in low statistics IOV in the "<<outputBase_<<" harvester"<< std::endl;
            poolDbService->writeOne<SiPixelQuality>(siPixelQualityPermBad, thisIOV, recordName_+"_prompt");
            // add empty bad components to "other" tag
            SiPixelQuality* siPixelQualityDummy = new SiPixelQuality();
            edm::LogInfo("SiPixelStatusHarvester")
                 << "Tag requested for other in low statistics IOV in the "<<outputBase_<<" harvester"<< std::endl;
            poolDbService->writeOne<SiPixelQuality>(siPixelQualityDummy, thisIOV, recordName_+"_other");

            continue;

          } 

          ///////////////////////////////////////////////////////////////////////////////////////////////////

          // create the DB object
          // payload including all : permanent bad + other + stuckTBM
          SiPixelQuality *siPixelQualityPCL = new SiPixelQuality();
          // payload for prompt reco : permanent bad + other sources of bad components
          SiPixelQuality *siPixelQualityPrompt = new SiPixelQuality();
          // payload for : other sources of bad components
          SiPixelQuality *siPixelQualityOther = new SiPixelQuality();

          // get badROC list due to stuck TBM
          std::map<int, std::vector<int> > tmpStuckTBMs = tmpSiPixelStatus.getStuckTBMsRocs();

          std::map<int, SiPixelModuleStatus> detectorStatus = tmpSiPixelStatus.getDetectorStatus();
          std::map<int, SiPixelModuleStatus>::iterator itModEnd = detectorStatus.end();
          for (std::map<int, SiPixelModuleStatus>::iterator itMod = detectorStatus.begin(); itMod != itModEnd; ++itMod) {

               // create the bad module list
               SiPixelQuality::disabledModuleType BadModulePCL, BadModulePrompt, BadModuleOther;

               int detid = itMod->first;
               BadModulePCL.DetID = uint32_t(detid); 
               BadModulePrompt.DetID = uint32_t(detid); BadModuleOther.DetID = uint32_t(detid);

               BadModulePCL.errorType = 3;
               BadModulePrompt.errorType = 3; BadModuleOther.errorType = 3;

               BadModulePCL.BadRocs = 0; 
               BadModulePrompt.BadRocs = 0; BadModuleOther.BadRocs = 0;

               std::vector<uint32_t> BadRocListPCL, BadRocListPrompt, BadRocListOther;

               SiPixelModuleStatus modStatus = itMod->second;
               std::vector<int> listStuckTBM = tmpStuckTBMs[detid];

               for (int iroc = 0; iroc < modStatus.nrocs(); ++iroc) {

                   unsigned long int rocOccupancy = modStatus.digiOccROC(iroc);

                   // Bad ROC are from low DIGI Occ ROCs
                   if(rocOccupancy<1.e-4*DetAverage){

                     BadRocListPCL.push_back(uint32_t(iroc));
                     std::vector<int>::iterator it = std::find(listStuckTBM.begin(), listStuckTBM.end(),iroc);

                     // from prompt =  permanent bad + other
                     if(it==listStuckTBM.end() || badPixelInfo_->IsRocBad(detid, iroc)) 
                     // if permanent or not stuck TBM( in the stuckTBM list but not permanent)
                       BadRocListPrompt.push_back(uint32_t(iroc));

                     // other source of bad components
                     if(it==listStuckTBM.end() && !(badPixelInfo_->IsRocBad(detid, iroc))) 
                     // if not permanent and not stuck TBM
                       BadRocListOther.push_back(uint32_t(iroc));                     
                    
                   }
               }

               if(BadRocListPCL.size()==16) BadModulePCL.errorType = 0;
               if(BadRocListPrompt.size()==16) BadModulePrompt.errorType = 0;
               if(BadRocListOther.size()==16) BadModuleOther.errorType = 0;

               short badrocsPCL = 0;
               for(std::vector<uint32_t>::iterator iterPCL = BadRocListPCL.begin(); iterPCL != BadRocListPCL.end(); ++iterPCL){
                   badrocsPCL +=  1 << *iterPCL; // 1 << *iter = 2^{*iter} using bitwise shift 
               } 
               // fill the badmodule only if there is(are) bad ROC(s) in it
               if(badrocsPCL!=0){
                 BadModulePCL.BadRocs = badrocsPCL;
                 siPixelQualityPCL->addDisabledModule(BadModulePCL);
               }

               short badrocsPrompt = 0;
               for(std::vector<uint32_t>::iterator iterPrompt = BadRocListPrompt.begin(); iterPrompt != BadRocListPrompt.end(); ++iterPrompt){
                   badrocsPrompt +=  1 << *iterPrompt; // 1 << *iter = 2^{*iter} using bitwise shift
               }
               // fill the badmodule only if there is(are) bad ROC(s) in it
               if(badrocsPrompt!=0){
                 BadModulePrompt.BadRocs = badrocsPrompt;
                 siPixelQualityPrompt->addDisabledModule(BadModulePrompt);
               }

               short badrocsOther = 0;
               for(std::vector<uint32_t>::iterator iterOther = BadRocListOther.begin(); iterOther != BadRocListOther.end(); ++iterOther){
                   badrocsOther +=  1 << *iterOther; // 1 << *iter = 2^{*iter} using bitwise shift
               }
               // fill the badmodule only if there is(are) bad ROC(s) in it
               if(badrocsOther!=0){
                 BadModuleOther.BadRocs = badrocsOther;
                 siPixelQualityOther->addDisabledModule(BadModuleOther);
               }

         } // end module loop

         if(debug_) // only produce the tag for all sources of bad components for debugging reason
	     poolDbService->writeOne<SiPixelQuality>(siPixelQualityPCL, thisIOV, recordName_+"_PCL");

         if (poolDbService->isNewTagRequest(recordName_+"_prompt")) {
             edm::LogInfo("SiPixelStatusHarvester")
                 << "new tag requested for prompt" << std::endl;
             poolDbService->writeOne<SiPixelQuality>(siPixelQualityPrompt, thisIOV, recordName_+"_prompt");
         }
         else {
            edm::LogInfo("SiPixelStatusHarvester")
               << "no new tag requested, appending IOV for prompt" << std::endl;
             poolDbService->writeOne<SiPixelQuality>(siPixelQualityPrompt, thisIOV, recordName_+"_prompt");
         }

         if (poolDbService->isNewTagRequest(recordName_+"_other")) {
             edm::LogInfo("SiPixelStatusHarvester")
                 << "new tag requested for other" << std::endl;
             poolDbService->writeOne<SiPixelQuality>(siPixelQualityOther, thisIOV, recordName_+"_other");
         }
         else {
            edm::LogInfo("SiPixelStatusHarvester")
               << "no new tag requested, appending IOV for other" << std::endl;
             poolDbService->writeOne<SiPixelQuality>(siPixelQualityOther, thisIOV, recordName_+"_other");
         }

         if (dumpTxt_){ // text dump for the DIGI occuancy for all pixels in all ROCs for the pixle detector
            std::string outTxt = Form("%s_Run%d_Lumi%d_SiPixelStatus.txt", outTxtFileName_.c_str(), iRun.id().run(),it->first);
            tmpSiPixelStatus.dumpToFile(outTxt); 
         }

     }// loop over IOV-structured Map (payloads)

     // Add a dummy IOV starting from last lumisection+1 to close the tag for the run
     if(outputBase_ == "nLumibased" || outputBase_ == "dynamicLumibased"){

         edm::LuminosityBlockID lu(iRun.id().run(),endLumiBlock_+1);
         cond::Time_t thisIOV = (cond::Time_t)(lu.value());

         poolDbService->writeOne<SiPixelQuality>(siPixelQualityPermBad, thisIOV, recordName_+"_prompt");

         // add empty bad components to "other" tag
         SiPixelQuality* siPixelQualityDummy = new SiPixelQuality();
         poolDbService->writeOne<SiPixelQuality>(siPixelQualityDummy, thisIOV, recordName_+"_other");

     }

  } // end of if(poolDbService.isAvailable() )

}

//--------------------------------------------------------------------------------------------------
void SiPixelStatusHarvester::beginLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iEventSetup) { 
}

//--------------------------------------------------------------------------------------------------
void SiPixelStatusHarvester::endLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iEVentSetup) {

  siPixelStatusManager_.readLumi(iLumi);
  // update endLumiBlock_ by current lumi block
  if(endLumiBlock_<iLumi.luminosityBlock())
    endLumiBlock_ = iLumi.luminosityBlock();

}


DEFINE_FWK_MODULE(SiPixelStatusHarvester);
