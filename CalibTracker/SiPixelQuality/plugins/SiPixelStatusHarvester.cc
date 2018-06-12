
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
  aveDigiOcc_(iConfig.getParameter<ParameterSet>("SiPixelStatusManagerParameters").getUntrackedParameter<int>("aveDigiOcc")),
  nLumi_(iConfig.getParameter<edm::ParameterSet>("SiPixelStatusManagerParameters").getUntrackedParameter<int>("resetEveryNLumi")),
  moduleName_(iConfig.getParameter<ParameterSet>("SiPixelStatusManagerParameters").getUntrackedParameter<std::string>("moduleName")),
  label_(iConfig.getParameter<ParameterSet>("SiPixelStatusManagerParameters").getUntrackedParameter<std::string>("label")),
  siPixelStatusManager_(iConfig, consumesCollector()){  

  debug_ = iConfig.getUntrackedParameter<bool>("debug");
  recordName_ = iConfig.getUntrackedParameter<std::string>("recordName", "SiPixelQualityFromDbRcd");
  
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

  std::map<edm::LuminosityBlockNumber_t,std::map<int,std::vector<int>> > FEDerror25Map = siPixelStatusManager_.getFEDerror25Rocs();

  std::map<edm::LuminosityBlockNumber_t,SiPixelDetectorStatus> siPixelStatusMap = siPixelStatusManager_.getBadComponents();
  edm::Service<cond::service::PoolDBOutputService> poolDbService;

  if(poolDbService.isAvailable() ) {// if(poolDbService.isAvailable() )

    // start producing tag for permanent component removed
    SiPixelQuality *siPixelQualityPermBad = new SiPixelQuality();
    const std::vector<SiPixelQuality::disabledModuleType> badComponentList = badPixelInfo_->getBadComponentList();
    for(unsigned int i = 0; i<badComponentList.size();i++){
        siPixelQualityPermBad->addDisabledModule(badComponentList[i]);
    }
    if(debug_==true){ // only produced for debugging reason
        cond::Time_t thisIOV = (cond::Time_t) iRun.id().run();
        poolDbService->writeOne<SiPixelQuality>(siPixelQualityPermBad, thisIOV, recordName_+"_permanentBad");
    }

    // IOV for final payloads. FEDerror25 and pcl
    std::map<edm::LuminosityBlockNumber_t, edm::LuminosityBlockNumber_t> finalIOV;
    std::map<edm::LuminosityBlockNumber_t, edm::LuminosityBlockNumber_t> fedError25IOV;
    std::map<edm::LuminosityBlockNumber_t, edm::LuminosityBlockNumber_t> pclIOV;

    // stuckTBM tag from FED error 25 with permanent component removed
    for(SiPixelStatusManager::FEDerror25Map_iterator it=FEDerror25Map.begin(); it!=FEDerror25Map.end();it++){

          cond::Time_t thisIOV = 1;
          edm::LuminosityBlockID lu(iRun.id().run(),it->first);
          thisIOV = (cond::Time_t)(lu.value());

          SiPixelQuality *siPixelQuality = new SiPixelQuality();
          SiPixelQuality *siPixelQuality_FEDerror25 = new SiPixelQuality();

          std::map<int, std::vector<int> > tmpFEDerror25 = it->second;
          for(std::map<int, std::vector<int> >::iterator ilist = tmpFEDerror25.begin(); ilist!=tmpFEDerror25.end();ilist++){

             int detid = ilist->first;

             SiPixelQuality::disabledModuleType BadModule, BadModule_FEDerror25;

             BadModule.DetID = uint32_t(detid); BadModule_FEDerror25.DetID = uint32_t(detid);
             BadModule.errorType = 3;           BadModule_FEDerror25.errorType = 3;

             BadModule.BadRocs = 0;             BadModule_FEDerror25.BadRocs = 0;
             std::vector<uint32_t> BadRocList, BadRocList_FEDerror25;
             std::vector<int> list = ilist->second;

             for(unsigned int i=0; i<list.size();i++){
                 // only include rocs that are not permanent known bad
                 int iroc =  list[i];
                 BadRocList_FEDerror25.push_back(uint32_t(iroc));
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

             // change module error type if all ROCs are bad
             if(BadRocList_FEDerror25.size()==16)
                BadModule_FEDerror25.errorType = 0;

             short badrocs_FEDerror25 = 0;
             for(std::vector<uint32_t>::iterator iter = BadRocList_FEDerror25.begin(); iter != BadRocList_FEDerror25.end(); ++iter){
                   badrocs_FEDerror25 +=  1 << *iter; // 1 << *iter = 2^{*iter} using bitwise shift
             }
             // fill the badmodule only if there is(are) bad ROC(s) in it
             if(badrocs_FEDerror25!=0){
                BadModule_FEDerror25.BadRocs = badrocs_FEDerror25;
                siPixelQuality_FEDerror25->addDisabledModule(BadModule_FEDerror25);
             }

          } // loop over modules

          finalIOV[it->first] = it->first;
          fedError25IOV[it->first] = it->first;

          poolDbService->writeOne<SiPixelQuality>(siPixelQuality, thisIOV, recordName_+"_stuckTBM");
          if(debug_==true) // only produced for debugging reason
             poolDbService->writeOne<SiPixelQuality>(siPixelQuality_FEDerror25, thisIOV, recordName_+"_FEDerror25");

          delete siPixelQuality;
          delete siPixelQuality_FEDerror25;         

    }

    // IOV for PCL combines permanent bad/stuckTBM/other
    for(SiPixelStatusManager::siPixelStatusMap_iterator it=siPixelStatusMap.begin(); it!=siPixelStatusMap.end();it++){
        finalIOV[it->first] = it->first;
        pclIOV[it->first] = it->first;
    }

    // loop over final IOV
    std::map<edm::LuminosityBlockNumber_t, edm::LuminosityBlockNumber_t>::iterator itIOV;
    for(itIOV=finalIOV.begin();itIOV!=finalIOV.end();itIOV++){

          cond::Time_t thisIOV = 1;
          edm::LuminosityBlockID lu(iRun.id().run(),itIOV->first);
          thisIOV = (cond::Time_t)(lu.value());

          edm::LuminosityBlockNumber_t lumiStuckTBMs = SiPixelStatusHarvester::stepIOV(itIOV->first,fedError25IOV);
          edm::LuminosityBlockNumber_t lumiPCL = SiPixelStatusHarvester::stepIOV(itIOV->first,pclIOV);

          // get badROC list due to FEDerror25 = stuckTBM + permanent bad components
          std::map<int, std::vector<int> > tmpFEDerror25 = FEDerror25Map[lumiStuckTBMs];
          // get SiPixelDetectorStatus
          SiPixelDetectorStatus tmpSiPixelStatus = siPixelStatusMap[lumiPCL];
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

            delete siPixelQualityDummy;
            continue;

          } 

          ///////////////////////////////////////////////////////////////////////////////////////////////////

          // create the DB object
          // payload including all : PCL = permanent bad + other + stuckTBM
          SiPixelQuality *siPixelQualityPCL = new SiPixelQuality();
          SiPixelQuality *siPixelQualityPrompt = new SiPixelQuality();
          SiPixelQuality *siPixelQualityOther = new SiPixelQuality();

          std::map<int, SiPixelModuleStatus> detectorStatus = tmpSiPixelStatus.getDetectorStatus();
          std::map<int, SiPixelModuleStatus>::iterator itModEnd = detectorStatus.end();
          for (std::map<int, SiPixelModuleStatus>::iterator itMod = detectorStatus.begin(); itMod != itModEnd; ++itMod) {

               // create the bad module list for PCL, prompt and other
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
               std::vector<int> listFEDerror25 = tmpFEDerror25[detid];

               for (int iroc = 0; iroc < modStatus.nrocs(); ++iroc) {

                   unsigned int rocOccupancy = modStatus.digiOccROC(iroc);

                   // Bad ROC are from low DIGI Occ ROCs
                   if(rocOccupancy<1.e-4*DetAverage){

                     //PCL bad roc list
                     BadRocListPCL.push_back(uint32_t(iroc));
                     //FEDerror25 list
                     std::vector<int>::iterator it = std::find(listFEDerror25.begin(), listFEDerror25.end(),iroc);

                     // from prompt = PCL bad - stuckTBM =  PCL bad - FEDerror25 + permanent bad
                     if(it==listFEDerror25.end() || badPixelInfo_->IsRocBad(detid, iroc))
                        // if not FEDerror25 or permanent bad
                        BadRocListPrompt.push_back(uint32_t(iroc));
                     // other source of bad components = prompt - permanent bad = PCL bad - FEDerror25
                     // or to be safe, say not FEDerro25 and not permanent bad
                     if(it==listFEDerror25.end() && !(badPixelInfo_->IsRocBad(detid, iroc))){
                        // if not permanent and not stuck TBM
                        BadRocListOther.push_back(uint32_t(iroc)); 

                     }
                   }

               } // loop over ROCs

               if(BadRocListPCL.size()==16) BadModulePCL.errorType = 0;
               if(BadRocListPrompt.size()==16) BadModulePrompt.errorType = 0;
               if(BadRocListOther.size()==16) BadModuleOther.errorType = 0;

               // pcl
               short badrocsPCL = 0;
               for(std::vector<uint32_t>::iterator iterPCL = BadRocListPCL.begin(); iterPCL != BadRocListPCL.end(); ++iterPCL){
                   badrocsPCL +=  1 << *iterPCL; // 1 << *iter = 2^{*iter} using bitwise shift 
               } 
               if(badrocsPCL!=0){
                 BadModulePCL.BadRocs = badrocsPCL;
                 siPixelQualityPCL->addDisabledModule(BadModulePCL);
               }

               // prompt
               short badrocsPrompt = 0;
               for(std::vector<uint32_t>::iterator iterPrompt = BadRocListPrompt.begin(); iterPrompt != BadRocListPrompt.end(); ++iterPrompt){
                   badrocsPrompt +=  1 << *iterPrompt; // 1 << *iter = 2^{*iter} using bitwise shift
               }
               if(badrocsPrompt!=0){
                 BadModulePrompt.BadRocs = badrocsPrompt;
                 siPixelQualityPrompt->addDisabledModule(BadModulePrompt);
               }

               // other
               short badrocsOther= 0;
               for(std::vector<uint32_t>::iterator iterOther = BadRocListOther.begin(); iterOther != BadRocListOther.end(); ++iterOther){
                   badrocsOther +=  1 << *iterOther; // 1 << *iter = 2^{*iter} using bitwise shift
               }
               if(badrocsOther!=0){
                 BadModuleOther.BadRocs = badrocsOther;
                 siPixelQualityOther->addDisabledModule(BadModuleOther);
               }
     
         } // end module loop
 
         //PCL
         if(debug_==true) // only produced for debugging reason
             poolDbService->writeOne<SiPixelQuality>(siPixelQualityPCL, thisIOV, recordName_+"_PCL");

         // prompt
         poolDbService->writeOne<SiPixelQuality>(siPixelQualityPrompt, thisIOV, recordName_+"_prompt");

         // other
         poolDbService->writeOne<SiPixelQuality>(siPixelQualityOther, thisIOV, recordName_+"_other");

         delete siPixelQualityPCL;
         delete siPixelQualityPrompt;
         delete siPixelQualityOther;

     }// loop over IOV

     // Add a dummy IOV starting from last lumisection+1 to close the tag for the run
     if(outputBase_ == "nLumibased" || outputBase_ == "dynamicLumibased"){

         edm::LuminosityBlockID lu(iRun.id().run(),endLumiBlock_+1);
         cond::Time_t thisIOV = (cond::Time_t)(lu.value());

         poolDbService->writeOne<SiPixelQuality>(siPixelQualityPermBad, thisIOV, recordName_+"_prompt");

         // add empty bad components to "other" tag
         SiPixelQuality* siPixelQualityDummy = new SiPixelQuality();
         poolDbService->writeOne<SiPixelQuality>(siPixelQualityDummy, thisIOV, recordName_+"_other");
         delete siPixelQualityDummy;
     }

     delete siPixelQualityPermBad;

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

// step function for IOV
edm::LuminosityBlockNumber_t SiPixelStatusHarvester::stepIOV(edm::LuminosityBlockNumber_t pin, std::map<edm::LuminosityBlockNumber_t,edm::LuminosityBlockNumber_t> IOV){

   std::map<edm::LuminosityBlockNumber_t, edm::LuminosityBlockNumber_t>::iterator itIOV;
   for(itIOV=IOV.begin();itIOV!=IOV.end();itIOV++){
       std::map<edm::LuminosityBlockNumber_t, edm::LuminosityBlockNumber_t>::iterator nextItIOV;
       nextItIOV = itIOV; nextItIOV++;

       if(nextItIOV!=IOV.end()){ 
          if(pin>=itIOV->first && pin<nextItIOV->first){
             return itIOV->first;
          }
       }
       else{
          if(pin>=itIOV->first){
             return itIOV->first;
          }
       }

   }

   // return the firstIOV in case all above fail
   return (IOV.begin())->first;
   
}

DEFINE_FWK_MODULE(SiPixelStatusHarvester);
