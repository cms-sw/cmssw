
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

// Pixel geometry and cabling map
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCabling.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"

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
  SiPixelPhase1Base(iConfig),
  outputBase_(iConfig.getParameter<ParameterSet>("SiPixelStatusManagerParameters").getUntrackedParameter<std::string>("outputBase")),
  aveDigiOcc_(iConfig.getParameter<ParameterSet>("SiPixelStatusManagerParameters").getUntrackedParameter<int>("aveDigiOcc")),
  nLumi_(iConfig.getParameter<edm::ParameterSet>("SiPixelStatusManagerParameters").getUntrackedParameter<int>("resetEveryNLumi")),
  moduleName_(iConfig.getParameter<ParameterSet>("SiPixelStatusManagerParameters").getUntrackedParameter<std::string>("moduleName")),
  label_(iConfig.getParameter<ParameterSet>("SiPixelStatusManagerParameters").getUntrackedParameter<std::string>("label")){  

  SiPixelStatusManager* siPixelStatusManager = new SiPixelStatusManager(iConfig, consumesCollector());
  siPixelStatusManager_ = *siPixelStatusManager;
  debug_ = iConfig.getUntrackedParameter<bool>("debug");
  recordName_ = iConfig.getUntrackedParameter<std::string>("recordName", "SiPixelQualityFromDbRcd");
  
  sensorSize_.clear();
  pixelO2O_.clear();

  siPixelStatusManager_.reset();
  endLumiBlock_ = 0;
  countLumi_ = 0;

}

//--------------------------------------------------------------------------------------------------
SiPixelStatusHarvester::~SiPixelStatusHarvester(){}

//--------------------------------------------------------------------------------------------------
void SiPixelStatusHarvester::beginJob() { }

//--------------------------------------------------------------------------------------------------
void SiPixelStatusHarvester::endJob() { }  

//--------------------------------------------------------------------------------------------------
void SiPixelStatusHarvester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {}

//--------------------------------------------------------------------------------------------------
void SiPixelStatusHarvester::endRunProduce(edm::Run& iRun, const edm::EventSetup& iSetup){

  // tracker geometry and cabling map to convert offline row/column (module) to online row/column
  edm::ESHandle<TrackerGeometry> tmpTkGeometry;
  iSetup.get<TrackerDigiGeometryRecord>().get(tmpTkGeometry);
  trackerGeometry_ = tmpTkGeometry.product();

  edm::ESHandle<SiPixelFedCablingMap> pixelCabling;
  iSetup.get<SiPixelFedCablingMapRcd>().get(pixelCabling);
  cablingMap_ = pixelCabling.product();

  for (TrackerGeometry::DetContainer::const_iterator it = trackerGeometry_->dets().begin(); it != trackerGeometry_->dets().end(); it++){

       const PixelGeomDetUnit *pgdu = dynamic_cast<const PixelGeomDetUnit*>((*it));
       if (pgdu == nullptr) continue;
       DetId detId = (*it)->geographicalId();
       int detid = detId.rawId();

       const PixelTopology* topo = static_cast<const PixelTopology*>(&pgdu->specificTopology());
       // number of row/columns for a given module
       int rowsperroc = topo->rowsperroc();
       int colsperroc = topo->colsperroc();

       int nROCrows = pgdu->specificTopology().nrows()/rowsperroc;
       int nROCcolumns = pgdu->specificTopology().ncolumns()/colsperroc;
       unsigned int nrocs = nROCrows*nROCcolumns;
       sensorSize_[detid] = nrocs;

       std::map<int, std::pair<int,int> > rocToOfflinePixel;

       std::vector<sipixelobjects::CablingPathToDetUnit> path = (cablingMap_->det2PathMap()).find(detId.rawId())->second;
       typedef std::vector<sipixelobjects::CablingPathToDetUnit>::const_iterator IT;
       for(IT it = path.begin(); it != path.end(); ++it) {
           // Pixel ROC building from path in cabling map
           const sipixelobjects::PixelROC *roc = cablingMap_->findItem(*it);
           int idInDetUnit = (int) roc->idInDetUnit();

           // local to global conversion
           sipixelobjects::LocalPixel::RocRowCol local = {rowsperroc/2, colsperroc/2};
           sipixelobjects::GlobalPixel global = roc->toGlobal(sipixelobjects::LocalPixel(local));
 
           rocToOfflinePixel[idInDetUnit] = std::pair<int,int>(global.row, global.col);

       }

       pixelO2O_[detid] = rocToOfflinePixel;
  }

  // Permananent bad components
  edm::ESHandle<SiPixelQuality> qualityInfo;
  iSetup.get<SiPixelQualityFromDbRcd>().get( qualityInfo );
  badPixelInfo_ = qualityInfo.product();

  // read in SiPixel occupancy data in ALCARECO/ALCAPROMPT
  siPixelStatusManager_.createPayloads();
  std::map<edm::LuminosityBlockNumber_t,std::map<int,std::vector<int>> > FEDerror25Map = siPixelStatusManager_.getFEDerror25Rocs();
  std::map<edm::LuminosityBlockNumber_t,SiPixelDetectorStatus> siPixelStatusMap = siPixelStatusManager_.getBadComponents();

  // DB service
  edm::Service<cond::service::PoolDBOutputService> poolDbService;

  if(poolDbService.isAvailable() ) {// if(poolDbService.isAvailable() )

    std::vector<int> vDetId;
    // start producing tag for permanent component removed
    SiPixelQuality *siPixelQualityPermBad = new SiPixelQuality();
    const std::vector<SiPixelQuality::disabledModuleType> badComponentList = badPixelInfo_->getBadComponentList();
    for(unsigned int i = 0; i<badComponentList.size();i++){

        siPixelQualityPermBad->addDisabledModule(badComponentList[i]);

        uint32_t detId = badComponentList[i].DetID;
        int detid = int(detId);
        unsigned int nroc = sensorSize_[detid];

        for(int iroc = 0; iroc<int(nroc); iroc++){
            if(badPixelInfo_->IsRocBad(detId, short(iroc)) ){
                 std::map<int, std::pair<int,int> > rocToOfflinePixel = pixelO2O_[detid];
                 int row = rocToOfflinePixel[iroc].first;
                 int column = rocToOfflinePixel[iroc].second;
                 histo[PERMANENTBADROC].fill(detId, nullptr, column, row);
            }
        }

    }
    if(debug_==true){ // only produced for debugging reason
        cond::Time_t thisIOV = (cond::Time_t) iRun.id().run();
        poolDbService->writeOne<SiPixelQuality>(siPixelQualityPermBad, thisIOV, recordName_+"_permanentBad");
    }

    // IOV for final payloads. FEDerror25 and pcl
    std::map<edm::LuminosityBlockNumber_t, edm::LuminosityBlockNumber_t> finalIOV;
    std::map<edm::LuminosityBlockNumber_t, edm::LuminosityBlockNumber_t> fedError25IOV;
    std::map<edm::LuminosityBlockNumber_t, edm::LuminosityBlockNumber_t> pclIOV;

    //int nLumiBlock_ = countLumi_-FEDerror25Map.begin()->first+1;

    // container for SiPixelQuality for the whole run
    std::map<int, SiPixelQuality*> siPixelQualityStuckTBM_Tag;

    // stuckTBM tag from FED error 25 with permanent component removed
    for(SiPixelStatusManager::FEDerror25Map_iterator it=FEDerror25Map.begin(); it!=FEDerror25Map.end();it++){

          cond::Time_t thisIOV = 1;
          edm::LuminosityBlockID lu(iRun.id().run(),it->first);
          thisIOV = (cond::Time_t)(lu.value());

          int interval = 0;
          // interval is the number of lumi sections in the IOV
          SiPixelStatusManager::FEDerror25Map_iterator nextIt = std::next(it);
          if(nextIt!=FEDerror25Map.end()) interval = int(nextIt->first - it->first);
          else interval = int(endLumiBlock_ - it->first + 1);

          SiPixelQuality *siPixelQuality_stuckTBM = new SiPixelQuality();
          SiPixelQuality *siPixelQuality_FEDerror25 = new SiPixelQuality();

          std::map<int, std::vector<int> > tmpFEDerror25 = it->second;
          for(std::map<int, std::vector<int> >::iterator ilist = tmpFEDerror25.begin(); ilist!=tmpFEDerror25.end();ilist++){

             int detid = ilist->first;
             uint32_t detId = uint32_t(detid);

             SiPixelQuality::disabledModuleType BadModule_stuckTBM, BadModule_FEDerror25;

             BadModule_stuckTBM.DetID = uint32_t(detid); BadModule_FEDerror25.DetID = uint32_t(detid);
             BadModule_stuckTBM.errorType = 3;           BadModule_FEDerror25.errorType = 3;

             BadModule_stuckTBM.BadRocs = 0;             BadModule_FEDerror25.BadRocs = 0;
             std::vector<uint32_t> BadRocList_stuckTBM, BadRocList_FEDerror25;
             std::vector<int> list = ilist->second;

             for(unsigned int i=0; i<list.size();i++){

                 int iroc =  list[i];
                 std::map<int, std::pair<int,int> > rocToOfflinePixel = pixelO2O_[detid];
                 int row = rocToOfflinePixel[iroc].first;
                 int column = rocToOfflinePixel[iroc].second;

                 BadRocList_FEDerror25.push_back(uint32_t(iroc));
                 for (int iLumi = 0; iLumi<interval;iLumi++){
                      histo[FEDERRORROC].fill(detId, nullptr, column, row);// 1.0/nLumiBlock_);
                 }

                 // only include rocs that are not permanent known bad
                 if(!badPixelInfo_->IsRocBad(detId, short(iroc))){ // stuckTBM = FEDerror25 - permanent bad
                    BadRocList_stuckTBM.push_back(uint32_t(iroc));
                    for (int iLumi = 0; iLumi<interval;iLumi++){
                         histo[STUCKTBMROC].fill(detId, nullptr, column, row);//, 1.0/nLumiBlock_);
                    }
                 }

             }

             // change module error type if all ROCs are bad
             if(BadRocList_stuckTBM.size()==sensorSize_[detid]) 
                BadModule_stuckTBM.errorType = 0;

             short badrocs_stuckTBM = 0;
             for(std::vector<uint32_t>::iterator iter = BadRocList_stuckTBM.begin(); iter != BadRocList_stuckTBM.end(); ++iter){
                   badrocs_stuckTBM +=  1 << *iter; // 1 << *iter = 2^{*iter} using bitwise shift
             }
             // fill the badmodule only if there is(are) bad ROC(s) in it
             if(badrocs_stuckTBM!=0){
               BadModule_stuckTBM.BadRocs = badrocs_stuckTBM;
               siPixelQuality_stuckTBM->addDisabledModule(BadModule_stuckTBM);
             }

             // change module error type if all ROCs are bad
             if(BadRocList_FEDerror25.size()==sensorSize_[detid])
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

          siPixelQualityStuckTBM_Tag[it->first] = siPixelQuality_stuckTBM;

          finalIOV[it->first] = it->first;
          fedError25IOV[it->first] = it->first;

          if(debug_==true) // only produced for debugging reason
             poolDbService->writeOne<SiPixelQuality>(siPixelQuality_FEDerror25, thisIOV, recordName_+"_FEDerror25");

          delete siPixelQuality_FEDerror25;         

    }

    // IOV for PCL output tags that "combines" permanent bad/stuckTBM/other
    for(SiPixelStatusManager::siPixelStatusMap_iterator it=siPixelStatusMap.begin(); it!=siPixelStatusMap.end();it++){
        finalIOV[it->first] = it->first;
        pclIOV[it->first] = it->first;
    }

    // loop over final IOV
    std::map<edm::LuminosityBlockNumber_t, edm::LuminosityBlockNumber_t>::iterator itIOV;

    // container for SiPixelQuality for the whole run
    std::map<int, SiPixelQuality*> siPixelQualityPCL_Tag;
    std::map<int, SiPixelQuality*> siPixelQualityPrompt_Tag;
    std::map<int, SiPixelQuality*> siPixelQualityOther_Tag;

    for(itIOV=finalIOV.begin();itIOV!=finalIOV.end();itIOV++){

          int interval = 0;
          std::map<edm::LuminosityBlockNumber_t, edm::LuminosityBlockNumber_t>::iterator nextItIOV = std::next(itIOV);
          if(nextItIOV!=finalIOV.end()) interval = int(nextItIOV->first - itIOV->first);
          else interval = int(endLumiBlock_ - itIOV->first + 1);

          edm::LuminosityBlockNumber_t lumiFEDerror25 = SiPixelStatusHarvester::stepIOV(itIOV->first,fedError25IOV);
          edm::LuminosityBlockNumber_t lumiPCL = SiPixelStatusHarvester::stepIOV(itIOV->first,pclIOV);

          // get badROC list due to FEDerror25 = stuckTBM + permanent bad components
          std::map<int, std::vector<int> > tmpFEDerror25 = FEDerror25Map[lumiFEDerror25];
          // get SiPixelDetectorStatus
          SiPixelDetectorStatus tmpSiPixelStatus = siPixelStatusMap[lumiPCL];
          double DetAverage = tmpSiPixelStatus.perRocDigiOcc();

          // For the IOV of which the statistics is too low, for e.g., a cosmic run
          // When using dynamicLumibased harvester or runbased harvester
          // this only happens when the full run is lack of statistics 
          if(DetAverage<aveDigiOcc_) {

            edm::LogInfo("SiPixelStatusHarvester")
                 << "Tag requested for prompt in low statistics IOV in the "<<outputBase_<<" harvester"<< std::endl;
            siPixelQualityPCL_Tag[itIOV->first] = siPixelQualityPermBad;
            siPixelQualityPrompt_Tag[itIOV->first] = siPixelQualityPermBad;            

            // loop over modules to fill the PROMPT DQM plots with permanent bad components
            std::map<int, SiPixelModuleStatus> detectorStatus = tmpSiPixelStatus.getDetectorStatus();
            std::map<int, SiPixelModuleStatus>::iterator itModEnd = detectorStatus.end();
            for (std::map<int, SiPixelModuleStatus>::iterator itMod = detectorStatus.begin(); itMod != itModEnd; ++itMod) {
            
                 int detid = itMod->first;
                 uint32_t detId = uint32_t(detid);
                 SiPixelModuleStatus modStatus = itMod->second;

                 for (int iroc = 0; iroc < modStatus.nrocs(); ++iroc) {

                       if(badPixelInfo_->IsRocBad(detId, short(iroc))){
                         std::map<int, std::pair<int,int> > rocToOfflinePixel = pixelO2O_[detid];
                         int row = rocToOfflinePixel[iroc].first;
                         int column = rocToOfflinePixel[iroc].second;                                                                                                 for (int iLumi = 0; iLumi<interval;iLumi++){
                           histo[PROMPTBADROC].fill(detId, nullptr, column, row);//, 1.0/nLumiBlock_);
                         }

                       } // if permanent BAD

                 } // loop over ROCs

            } // loop over modules

            // add empty bad components to "other" tag
            edm::LogInfo("SiPixelStatusHarvester")
                 << "Tag requested for other in low statistics IOV in the "<<outputBase_<<" harvester"<< std::endl;
            siPixelQualityOther_Tag[itIOV->first] = new SiPixelQuality();

            continue;

          } 

          ///////////////////////////////////////////////////////////////////////////////////////////////////

          // create the DB object
          // payload including all : PCL = permanent bad (low DIGI ROC) + other + stuckTBM
          SiPixelQuality *siPixelQualityPCL = new SiPixelQuality();
          SiPixelQuality *siPixelQualityOther = new SiPixelQuality();
          // Prompt = permanent bad(low DIGI + low eff/damaged ROCs + other)     
          SiPixelQuality *siPixelQualityPrompt = new SiPixelQuality();

          // loop over modules
          std::map<int, SiPixelModuleStatus> detectorStatus = tmpSiPixelStatus.getDetectorStatus();
          std::map<int, SiPixelModuleStatus>::iterator itModEnd = detectorStatus.end();
          for (std::map<int, SiPixelModuleStatus>::iterator itMod = detectorStatus.begin(); itMod != itModEnd; ++itMod) {

               // create the bad module list for PCL and other
               SiPixelQuality::disabledModuleType BadModulePCL, BadModuleOther;

               int detid = itMod->first; 
               uint32_t detId = uint32_t(detid);

               BadModulePCL.DetID = uint32_t(detid); BadModuleOther.DetID = uint32_t(detid);
               BadModulePCL.errorType = 3; BadModuleOther.errorType = 3;
               BadModulePCL.BadRocs = 0; BadModuleOther.BadRocs = 0;

               std::vector<uint32_t> BadRocListPCL, BadRocListOther;

               // module status and FEDerror25 status for module with DetId detId
               SiPixelModuleStatus modStatus = itMod->second;
               std::vector<int> listFEDerror25 = tmpFEDerror25[detid];

               for (int iroc = 0; iroc < modStatus.nrocs(); ++iroc) {

                   unsigned int rocOccupancy = modStatus.digiOccROC(iroc);

                   // Bad ROC are from low DIGI Occ ROCs
                   if(rocOccupancy<1.e-4*DetAverage){ // if BAD

                     std::map<int, std::pair<int,int> > rocToOfflinePixel = pixelO2O_[detid];
                     int row = rocToOfflinePixel[iroc].first;
                     int column = rocToOfflinePixel[iroc].second;

                     //PCL bad roc list
                     BadRocListPCL.push_back(uint32_t(iroc));
                     for (int iLumi = 0; iLumi<interval;iLumi++){
                         histo[BADROC].fill(detId, nullptr, column, row);//, 1.0/nLumiBlock_);
                     }

                     //FEDerror25 list
                     std::vector<int>::iterator it = std::find(listFEDerror25.begin(), listFEDerror25.end(),iroc);

                     // other source of bad components = PCL bad - FEDerror25 - permanent bad
                     if(it==listFEDerror25.end() && !(badPixelInfo_->IsRocBad(detId, short(iroc)))){
                        // if neither permanent nor FEDerror25
                        BadRocListOther.push_back(uint32_t(iroc)); 
                        for (int iLumi = 0; iLumi<interval;iLumi++){
                            histo[OTHERBADROC].fill(detId, nullptr, column, row);//, 1.0/nLumiBlock_);
                        }
                     }

                   }// if BAD

               } // loop over ROCs

               // errorType 0 means the full module is bad
               if(BadRocListPCL.size()==sensorSize_[detid]) BadModulePCL.errorType = 0;
               if(BadRocListOther.size()==sensorSize_[detid]) BadModuleOther.errorType = 0;

               // PCL
               short badrocsPCL = 0;
               for(std::vector<uint32_t>::iterator iterPCL = BadRocListPCL.begin(); iterPCL != BadRocListPCL.end(); ++iterPCL){
                   badrocsPCL +=  1 << *iterPCL; // 1 << *iter = 2^{*iter} using bitwise shift 
               } 
               if(badrocsPCL!=0){
                 BadModulePCL.BadRocs = badrocsPCL;
                 siPixelQualityPCL->addDisabledModule(BadModulePCL);
               }

               // Other
               short badrocsOther = 0;
               for(std::vector<uint32_t>::iterator iterOther = BadRocListOther.begin(); iterOther != BadRocListOther.end(); ++iterOther){
                   badrocsOther +=  1 << *iterOther; // 1 << *iter = 2^{*iter} using bitwise shift
               }
               if(badrocsOther!=0){
                 BadModuleOther.BadRocs = badrocsOther;
                 siPixelQualityOther->addDisabledModule(BadModuleOther);
               }

               // start constructing bad components for prompt = "other" + permanent
               SiPixelQuality::disabledModuleType BadModulePrompt;
               BadModulePrompt.DetID = uint32_t(detid);
               BadModulePrompt.errorType = 3;
               BadModulePrompt.BadRocs = 0;

               std::vector<uint32_t> BadRocListPrompt;
               for (int iroc = 0; iroc < modStatus.nrocs(); ++iroc) {
                   // if in permannet bad tag or is in other tag 
                   if(badPixelInfo_->IsRocBad(detId, short(iroc))|| ((badrocsOther >> short(iroc))&0x1)){
                      BadRocListPrompt.push_back(uint32_t(iroc));

                      std::map<int, std::pair<int,int> > rocToOfflinePixel = pixelO2O_[detid];
                      int row = rocToOfflinePixel[iroc].first;
                      int column = rocToOfflinePixel[iroc].second;
                      for (int iLumi = 0; iLumi<interval;iLumi++){
                           histo[PROMPTBADROC].fill(detId, nullptr, column, row);//, 1.0/nLumiBlock_);
                      }
                   } // if bad
               } // loop over all ROCs

               // errorType 0 means the full module is bad
               if(BadRocListPrompt.size()==sensorSize_[detid]) BadModulePrompt.errorType = 0;

               short badrocsPrompt = 0;
               for(std::vector<uint32_t>::iterator iterPrompt = BadRocListPrompt.begin(); iterPrompt != BadRocListPrompt.end(); ++iterPrompt){
                   badrocsPrompt +=  1 << *iterPrompt; // 1 << *iter = 2^{*iter} using bitwise shift
               }
               if(badrocsPrompt!=0){
                 BadModulePrompt.BadRocs = badrocsPrompt;
                 siPixelQualityPrompt->addDisabledModule(BadModulePrompt);
               }

         } // end module loop
 
         // PCL
         siPixelQualityPCL_Tag[itIOV->first] = siPixelQualityPCL;
         // Prompt
         siPixelQualityPrompt_Tag[itIOV->first] = siPixelQualityPrompt;
         // Other
         siPixelQualityOther_Tag[itIOV->first] = siPixelQualityOther;

     }// loop over IOV
 
     // Now construct the tags made of payloads 
     // and only append newIOV if this payload differs wrt last

     //PCL
     if(debug_==true)// only produced for debugging reason
       SiPixelStatusHarvester::constructTag(siPixelQualityPCL_Tag, poolDbService, "PCL", iRun);
     // other
     SiPixelStatusHarvester::constructTag(siPixelQualityOther_Tag, poolDbService, "other", iRun);
     // prompt
     SiPixelStatusHarvester::constructTag(siPixelQualityPrompt_Tag, poolDbService, "prompt", iRun);
     // stuckTBM
     SiPixelStatusHarvester::constructTag(siPixelQualityStuckTBM_Tag, poolDbService, "stuckTBM", iRun);

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
     countLumi_++;
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

bool SiPixelStatusHarvester::equal(SiPixelQuality* a, SiPixelQuality* b){

  std::vector<SiPixelQuality::disabledModuleType> badRocListA;
  std::vector<SiPixelQuality::disabledModuleType> badRocListB;

  for(unsigned int ia = 0; ia < (a->getBadComponentList()).size();ia++){
        badRocListA.push_back((a->getBadComponentList())[ia]); 
  }
  for(unsigned int ib = 0; ib < (b->getBadComponentList()).size();ib++){
        badRocListB.push_back((b->getBadComponentList())[ib]);
  }

  if(badRocListA.size()!=badRocListB.size()) return false;

  // ordering ROCs by DetId
  std::sort(badRocListA.begin(),badRocListA.end(),SiPixelQuality::BadComponentStrictWeakOrdering());
  std::sort(badRocListB.begin(),badRocListB.end(),SiPixelQuality::BadComponentStrictWeakOrdering());

  for(unsigned int i = 0; i<badRocListA.size();i++){

        uint32_t detIdA = badRocListA[i].DetID;
        uint32_t detIdB = badRocListB[i].DetID;
        if(detIdA!=detIdB) return false;
        else{
           unsigned short BadRocsA = badRocListA[i].BadRocs;
           unsigned short BadRocsB = badRocListB[i].BadRocs;
           if(BadRocsA!=BadRocsB) return false;
        }

  }

  //if the module list is the same, and for each module, roc list is the same
  //the two SiPixelQualitys are equal
  return true;


}

void SiPixelStatusHarvester::constructTag(std::map<int,SiPixelQuality*>siPixelQualityTag, edm::Service<cond::service::PoolDBOutputService>& poolDbService, std::string tagName,edm::Run& iRun){

     for (std::map<int, SiPixelQuality*>::iterator qIt = siPixelQualityTag.begin(); qIt != siPixelQualityTag.end(); ++qIt) {

            edm::LuminosityBlockID lu(iRun.id().run(),qIt->first);
            cond::Time_t thisIOV = (cond::Time_t)(lu.value());

            SiPixelQuality* thisPayload =  qIt->second;
            if(qIt==siPixelQualityTag.begin())
                poolDbService->writeOne<SiPixelQuality>(thisPayload, thisIOV, recordName_+"_"+tagName);
            else{
                SiPixelQuality* prevPayload = (std::prev(qIt))->second;
                if(!SiPixelStatusHarvester::equal(thisPayload,prevPayload)) // only append newIOV if this payload differs wrt last
                  poolDbService->writeOne<SiPixelQuality>(thisPayload, thisIOV, recordName_+"_"+tagName);
            }
     }

}

DEFINE_FWK_MODULE(SiPixelStatusHarvester);
