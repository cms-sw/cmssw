#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"

#include "DQM/SiStripMonitorHardware/interface/CMHistograms.hh"



CMHistograms::CMHistograms()
{
  dqm_ = 0;
  histogramConfig_.clear();

  categories_[0] = "NoClusters";
  categories_[1] = "Clusters";
  categories_[2] = "Shots";
  categories_[3] = "APV1sup";
  categories_[4] = "APV0sup";

  for (unsigned int i(0); i<500; i++){
    doFed_[i] = false;
  }

}

CMHistograms::~CMHistograms()
{
}
  
void CMHistograms::initialise(const edm::ParameterSet& iConfig,
			      std::ostringstream* pDebugStream
			      )
{
  getConfigForHistogram("DataPresent",iConfig,pDebugStream);

  getConfigForHistogram("MedianAPV0",iConfig,pDebugStream);
  getConfigForHistogram("MedianAPV1",iConfig,pDebugStream);
  getConfigForHistogram("MedianAPV0vsTime",iConfig,pDebugStream);
  getConfigForHistogram("MedianAPV1vsTime",iConfig,pDebugStream);

  getConfigForHistogram("ShotMedianAPV0",iConfig,pDebugStream);
  getConfigForHistogram("ShotMedianAPV1",iConfig,pDebugStream);
  getConfigForHistogram("ShotChannels",iConfig,pDebugStream);
  getConfigForHistogram("MedianAPV1vsAPV0",iConfig,pDebugStream);
  getConfigForHistogram("MedianAPV1minusAPV0",iConfig,pDebugStream);
  getConfigForHistogram("MedianAPV1minusAPV0vsTime",iConfig,pDebugStream);
  getConfigForHistogram("MedianAPV1minusAPV0minusShotMedianAPV1",iConfig,pDebugStream);
  getConfigForHistogram("MedianAPV0minusAPV1minusShotMedianAPV1",iConfig,pDebugStream);

  getConfigForHistogram("MedianAPV1vsAPV0perFED",iConfig,pDebugStream);
  getConfigForHistogram("MedianAPV1minusAPV0perFED",iConfig,pDebugStream);

  getConfigForHistogram("MedianAPV0perChannel",iConfig,pDebugStream);
  getConfigForHistogram("MedianAPV1perChannel",iConfig,pDebugStream);
  getConfigForHistogram("MedianAPV0minusAPV1perChannel",iConfig,pDebugStream);

  
  tkMapConfigName_ = "TkHistoMap";
  getConfigForHistogram(tkMapConfigName_,iConfig,pDebugStream);

  if (iConfig.exists("FedIdVec")){
    std::vector<unsigned int> lIdVec = iConfig.getUntrackedParameter<std::vector<unsigned int> >("FedIdVec");
    for (unsigned int i(0); i<lIdVec.size(); i++){
      if (lIdVec.at(i) < 500) doFed_[lIdVec.at(i)] = true;
    }
  }
}

void CMHistograms::fillHistograms(std::vector<CMvalues> aVec, float aTime, unsigned int aFedId)
{

  if (doFed_[aFedId]){
    bookFEDHistograms(aFedId);
  }

  for (unsigned iEle(0); iEle<aVec.size(); iEle++){//loop on elements

    CMvalues lVal = aVec.at(iEle);

    if (lVal.Medians.first >= 500 || lVal.Medians.second >= 500) {
      std::cout << "----- WARNING ! New max found: " << lVal.Medians.first << " " << lVal.Medians.second << " " << __FILE__ << " " << __LINE__ << std::endl;
    }




    if (lVal.IsShot) {
      fillHistogram(shotChannels_,lVal.ChannelID,aFedId);
      fillHistogram(shotMedianAPV0_,lVal.ShotMedians.first);
      fillHistogram(shotMedianAPV1_,lVal.ShotMedians.second);
      fillHistogram(medianAPV1minusAPV0minusShotMedianAPV1_,lVal.Medians.second-lVal.Medians.first-lVal.ShotMedians.second);
      fillHistogram(medianAPV0minusAPV1minusShotMedianAPV1_,lVal.Medians.first-lVal.Medians.second-lVal.ShotMedians.second);
    }

    bool lCat[5] = {
      lVal.Length == 7, //no clusters found
      lVal.Length > 7, //clusters found
      lVal.IsShot, //isShot
      lVal.Medians.second-lVal.Medians.first > 4, //APV1>APV0, by 4 units at least
      lVal.Medians.first-lVal.Medians.second > 4 //APV1>APV0, by 4 units at least};
    };

    for (unsigned int i(0); i<5; i++){
      if (lCat[i]) {
	fillHistogram(medianAPV0_[i],lVal.Medians.first);
	fillHistogram(medianAPV1_[i],lVal.Medians.second);
	fillHistogram(medianAPV0vsTime_[i],aTime,lVal.Medians.first);
	fillHistogram(medianAPV1vsTime_[i],aTime,lVal.Medians.second);

	fillHistogram(medianAPV1vsAPV0_[i],lVal.Medians.first,lVal.Medians.second);
	fillHistogram(medianAPV1minusAPV0_[i],lVal.Medians.second-lVal.Medians.first);
	fillHistogram(medianAPV1minusAPV0vsTime_[i],aTime,lVal.Medians.second-lVal.Medians.first);

	if (doFed_[aFedId]) {
	  fillHistogram(medianAPV1vsAPV0perFED_[i][aFedId],lVal.Medians.first,lVal.Medians.second);
	  fillHistogram(medianAPV1minusAPV0perFED_[i][aFedId],lVal.Medians.second-lVal.Medians.first);
	}
      }
    }

    if (doFed_[aFedId]){
      fillHistogram(medianAPV0perChannel_[aFedId].at(lVal.ChannelID),lVal.Medians.first);
      fillHistogram(medianAPV1perChannel_[aFedId].at(lVal.ChannelID),lVal.Medians.second);
      fillHistogram(medianAPV0minusAPV1perChannel_[aFedId].at(lVal.ChannelID),lVal.Medians.first-lVal.Medians.second);
    }

  }//loop on elements

}


void CMHistograms::bookTopLevelHistograms(DQMStore* dqm)
{
  //get the pointer to the dqm object
  dqm_ = dqm;


  //book FED level histograms
  //get FED IDs
  //const unsigned int siStripFedIdMin = FEDNumbering::MINSiStripFEDID;
  const unsigned int siStripFedIdMax = FEDNumbering::MAXSiStripFEDID;
  histosBooked_.resize(siStripFedIdMax+1,false);

  std::string lDir = dqm_->pwd()+"/";

  for (unsigned int i(0); i<5; i++){

    dqm_->cd(lDir);

    //std::cout << "Folder: " << lDir+categories_[i] << std::endl;
    
    dqm_->setCurrentFolder(lDir+categories_[i]);

    medianAPV0_[i] = bookHistogram("MedianAPV0",
				   "MedianAPV0"+categories_[i],
				   "median APV0",
				   250,0,500,
				   "median APV0");

    medianAPV1_[i] = bookHistogram("MedianAPV1",
				   "MedianAPV1"+categories_[i],
				   "median APV1",
				   250,0,500,
				   "median APV1");

    medianAPV0vsTime_[i] = bookProfile("MedianAPV0vsTime",
				       "MedianAPV0vsTime"+categories_[i],
				       "median APV0 vs time",
				       0,1000);

    medianAPV1vsTime_[i] = bookProfile("MedianAPV1vsTime",
				       "MedianAPV1vsTime"+categories_[i],
				       "median APV1 vs time",
				       0,1000);

    medianAPV1vsAPV0_[i] = book2DHistogram("MedianAPV1vsAPV0",
					   "MedianAPV1vsAPV0"+categories_[i],
					   "median APV1 vs APV0",
					   250,0,500,250,0,500,
					   "median APV0","median APV1");

    medianAPV1minusAPV0_[i] = bookHistogram("MedianAPV1minusAPV0",
					    "MedianAPV1minusAPV0"+categories_[i],
					    "median APV1 - median APV0",
					    500,-500,500,
					    "median APV1 - median APV0");

    medianAPV1minusAPV0vsTime_[i] = bookProfile("MedianAPV1minusAPV0vsTime",
						"MedianAPV1minusAPV0vsTime"+categories_[i],
						"#Delta(medians) vs time",
						-1000,1000);

    if (i==2){//isShot
      shotMedianAPV0_ = bookHistogram("ShotMedianAPV0","ShotMedianAPV0",
				      "median shot APV0",
				      100,-50,50,
				      "median shot APV0");
      shotMedianAPV1_ = bookHistogram("ShotMedianAPV1","ShotMedianAPV1",
				      "median shot APV1",
				      100,-50,50,
				      "median shot APV1");
      shotChannels_ = book2DHistogram("ShotChannels","ShotChannels",
				      "Channels with shots",
				      96,0,96,440,50,490,
				      "Channel id","FED id");
      medianAPV1minusAPV0minusShotMedianAPV1_ = bookHistogram("MedianAPV1minusAPV0minusShotMedianAPV1","MedianAPV1minusAPV0minusShotMedianAPV1",
						     "(median APV1 - median APV0)-shot median APV1",
						     500,-50,50,
						     "(median APV1 - median APV0)-shot median APV1");
      medianAPV0minusAPV1minusShotMedianAPV1_ = bookHistogram("MedianAPV0minusAPV1minusShotMedianAPV1","MedianAPV0minusAPV1minusShotMedianAPV1",
						     "(median APV0 - median APV1)-shot median APV1",
						     500,-50,50,
						     "(median APV0 - median APV1)-shot median APV1");
    }

  }

  dqm_->cd(lDir);

  //book map after, as it creates a new folder...
  if (histogramConfig_[tkMapConfigName_].enabled){
    //const std::string dqmPath = dqm_->pwd();
    tkmapCM_[0] = new TkHistoMap("SiStrip/TkHisto","TkHMap_MeanCMAPV0",0.,500);
    tkmapCM_[1] = new TkHistoMap("SiStrip/TkHisto","TkHMap_RmsCMAPV0",0.,500);
    tkmapCM_[2] = new TkHistoMap("SiStrip/TkHisto","TkHMap_MeanCMAPV1",0.,500);
    tkmapCM_[3] = new TkHistoMap("SiStrip/TkHisto","TkHMap_RmsCMAPV1",0.,500);
    tkmapCM_[4] = new TkHistoMap("SiStrip/TkHisto","TkHMap_MeanCMAPV0minusAPV1",-500.,500);
    tkmapCM_[5] = new TkHistoMap("SiStrip/TkHisto","TkHMap_RmsCMAPV0minusAPV1",-500.,500);
  }
  else {
    tkmapCM_[0] = 0;
    tkmapCM_[1] = 0;
    tkmapCM_[2] = 0;
    tkmapCM_[3] = 0;
    tkmapCM_[4] = 0;
    tkmapCM_[5] = 0;
  }


}


void CMHistograms::bookFEDHistograms(unsigned int fedId)
{
  if (!histosBooked_[fedId]) {
    //will do that only once

    for (unsigned int i(0); i<5;i++){
      bookFEDHistograms(fedId,i);
    }

    bookChannelsHistograms(fedId);

    histosBooked_[fedId] = true;
  }
}

void CMHistograms::bookFEDHistograms(unsigned int fedId, unsigned int aCategory)
{
  SiStripFedKey fedKey(fedId,0,0,0);
  std::stringstream fedIdStream;
  fedIdStream << fedId;

  dqm_->setCurrentFolder(fedKey.path()+categories_[aCategory]);
    
  medianAPV1vsAPV0perFED_[aCategory][fedId] = book2DHistogram("MedianAPV1vsAPV0perFED",
							      "MedianAPV1vsAPV0forFED"+fedIdStream.str()+categories_[aCategory],
							      "median APV1 vs APV0 for FED "+fedIdStream.str(),
							      250,0,500,250,0,500,
							      "APV0","APV1");
    
  medianAPV1minusAPV0perFED_[aCategory][fedId] = bookHistogram("MedianAPV1minusAPV0perFED",
							       "MedianAPV1minusAPV0forFED"+fedIdStream.str()+categories_[aCategory],
							       "median APV1 - median APV0 for FED "+fedIdStream.str(),
							       500,-500,500,
							       "#Delta(medians)");

}


void CMHistograms::bookChannelsHistograms(unsigned int fedId)
{
  SiStripFedKey fedKey(fedId,0,0,0);
  std::stringstream fedIdStream;
  fedIdStream << fedId;

  dqm_->setCurrentFolder(fedKey.path());

  for (unsigned int iCh(0); iCh < 96; iCh++){

    std::ostringstream lName0,lTitle0,lName1,lTitle1,lName2,lTitle2;
    lName0 << "MedianAPV0ForFed" << fedId << "Channel" << iCh;
    lTitle0 << "Median APV0 for FED/Ch " << fedId << "/" << iCh ;
    lName1 << "MedianAPV1ForFed" << fedId << "Channel" << iCh;
    lTitle1 << "Median APV1 for FED/Ch " << fedId << "/" << iCh ;
    lName2 << "MedianAPV0minusAPV1ForFed" << fedId << "Channel" << iCh;
    lTitle2 << "Median APV0-APV1 for FED/Ch " << fedId << "/" << iCh ;

    medianAPV0perChannel_[fedId].push_back( bookHistogram("MedianAPV0perChannel",
							  lName0.str(),
							  lTitle0.str(),
							  250,0,500,
							  "median APV0")
					    );

    medianAPV1perChannel_[fedId].push_back( bookHistogram("MedianAPV1perChannel",
							  lName1.str(),
							  lTitle1.str(),
							  250,0,500,
							  "median APV1")
					    );

    medianAPV0minusAPV1perChannel_[fedId].push_back( bookHistogram("MedianAPV0minusAPV1perChannel",
								   lName2.str(),
								   lTitle2.str(),
								   250,-500,500,
								   "median APV0-APV1")
						     );



  }


}

void CMHistograms::bookAllFEDHistograms()
{
  //get FED IDs
  const unsigned int siStripFedIdMin = FEDNumbering::MINSiStripFEDID;
  const unsigned int siStripFedIdMax = FEDNumbering::MAXSiStripFEDID;
  //book them
  for (unsigned int iFed = siStripFedIdMin; iFed <= siStripFedIdMax; iFed++) {
    for (unsigned int i(0); i<5; i++){
      bookFEDHistograms(iFed,i);
    }
  }
}

std::string CMHistograms::tkHistoMapName(unsigned int aIndex){
  return tkMapConfigName_;
}

TkHistoMap * CMHistograms::tkHistoMapPointer(unsigned int aIndex){
  assert(aIndex < 6);
  return tkmapCM_[aIndex];
}
