#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"

#include "DQM/SiStripMonitorHardware/interface/FEDHistograms.hh"



FEDHistograms::FEDHistograms()
{
  dqm_ = 0;
}

FEDHistograms::~FEDHistograms()
{
}
  
void FEDHistograms::initialise(const edm::ParameterSet& iConfig,
			       std::ostringstream* pDebugStream
			       )
{
  getConfigForHistogram(fedEventSize_,"FedEventSize",iConfig,pDebugStream);
  getConfigForHistogram(fedMaxEventSizevsTime_,"FedMaxEventSizevsTime",iConfig,pDebugStream);

  getConfigForHistogram(dataPresent_,"DataPresent",iConfig,pDebugStream);
  getConfigForHistogram(anyFEDErrors_,"AnyFEDErrors",iConfig,pDebugStream);
  getConfigForHistogram(anyDAQProblems_,"AnyDAQProblems",iConfig,pDebugStream);
  getConfigForHistogram(anyFEProblems_,"AnyFEProblems",iConfig,pDebugStream);
  getConfigForHistogram(corruptBuffers_,"CorruptBuffers",iConfig,pDebugStream);
  getConfigForHistogram(badChannelStatusBits_,"BadChannelStatusBits",iConfig,pDebugStream);
  getConfigForHistogram(badActiveChannelStatusBits_,"BadActiveChannelStatusBits",iConfig,pDebugStream);
  
  getConfigForHistogram(feOverflows_,"FEOverflows",iConfig,pDebugStream);
  getConfigForHistogram(feMissing_,"FEMissing",iConfig,pDebugStream);
  getConfigForHistogram(badMajorityAddresses_,"BadMajorityAddresses",iConfig,pDebugStream);
  getConfigForHistogram(badMajorityInPartition_,"BadMajorityInPartition",iConfig,pDebugStream);
  getConfigForHistogram(feMajFracTIB_,"FeMajFracTIB",iConfig,pDebugStream);
  getConfigForHistogram(feMajFracTOB_,"FeMajFracTOB",iConfig,pDebugStream);
  getConfigForHistogram(feMajFracTECB_,"FeMajFracTECB",iConfig,pDebugStream);
  getConfigForHistogram(feMajFracTECF_,"FeMajFracTECF",iConfig,pDebugStream);

  getConfigForHistogram(dataMissing_,"DataMissing",iConfig,pDebugStream);
  getConfigForHistogram(badIDs_,"BadIDs",iConfig,pDebugStream);
  getConfigForHistogram(badDAQPacket_,"BadDAQPacket",iConfig,pDebugStream);
  getConfigForHistogram(invalidBuffers_,"InvalidBuffers",iConfig,pDebugStream);
  getConfigForHistogram(badDAQCRCs_,"BadDAQCRCs",iConfig,pDebugStream);
  getConfigForHistogram(badFEDCRCs_,"BadFEDCRCs",iConfig,pDebugStream);
  
  getConfigForHistogram(feOverflowDetailed_,"FEOverflowsDetailed",iConfig,pDebugStream);
  getConfigForHistogram(feMissingDetailed_,"FEMissingDetailed",iConfig,pDebugStream);
  getConfigForHistogram(badMajorityAddressDetailed_,"BadMajorityAddressesDetailed",iConfig,pDebugStream);
  getConfigForHistogram(badStatusBitsDetailed_,"BadAPVStatusBitsDetailed",iConfig,pDebugStream);
  getConfigForHistogram(apvErrorDetailed_,"APVErrorBitsDetailed",iConfig,pDebugStream);
  getConfigForHistogram(apvAddressErrorDetailed_,"APVAddressErrorBitsDetailed",iConfig,pDebugStream);
  getConfigForHistogram(unlockedDetailed_,"UnlockedBitsDetailed",iConfig,pDebugStream);
  getConfigForHistogram(outOfSyncDetailed_,"OOSBitsDetailed",iConfig,pDebugStream);
  
  getConfigForHistogram(nFEDErrors_,"nFEDErrors",iConfig,pDebugStream);
  getConfigForHistogram(nFEDDAQProblems_,"nFEDDAQProblems",iConfig,pDebugStream);
  getConfigForHistogram(nFEDsWithFEProblems_,"nFEDsWithFEProblems",iConfig,pDebugStream);
  getConfigForHistogram(nFEDCorruptBuffers_,"nFEDCorruptBuffers",iConfig,pDebugStream);
  getConfigForHistogram(nBadChannelStatusBits_,"nBadChannelStatusBits",iConfig,pDebugStream);
  getConfigForHistogram(nBadActiveChannelStatusBits_,"nBadActiveChannelStatusBits",iConfig,pDebugStream);
  getConfigForHistogram(nFEDsWithFEOverflows_,"nFEDsWithFEOverflows",iConfig,pDebugStream);
  getConfigForHistogram(nFEDsWithMissingFEs_,"nFEDsWithMissingFEs",iConfig,pDebugStream);
  getConfigForHistogram(nFEDsWithFEBadMajorityAddresses_,"nFEDsWithFEBadMajorityAddresses",iConfig,pDebugStream);

  getConfigForHistogram(nFEDErrorsvsTime_,"nFEDErrorsvsTime",iConfig,pDebugStream);
  getConfigForHistogram(nFEDCorruptBuffersvsTime_,"nFEDCorruptBuffersvsTime",iConfig,pDebugStream);
  getConfigForHistogram(nFEDsWithFEProblemsvsTime_,"nFEDsWithFEProblemsvsTime",iConfig,pDebugStream);

  getConfigForHistogram(nUnconnectedChannels_,"nUnconnectedChannels",iConfig,pDebugStream);

  getConfigForHistogram(nTotalBadChannels_,"nTotalBadChannels",iConfig,pDebugStream);
  getConfigForHistogram(nTotalBadActiveChannels_,"nTotalBadActiveChannels",iConfig,pDebugStream);

  getConfigForHistogram(nTotalBadChannelsvsTime_,"nTotalBadChannelsvsTime",iConfig,pDebugStream);
  getConfigForHistogram(nTotalBadActiveChannelsvsTime_,"nTotalBadActiveChannelsvsTime",iConfig,pDebugStream);

  getConfigForHistogram(nAPVStatusBit_,"nAPVStatusBit",iConfig,pDebugStream);
  getConfigForHistogram(nAPVError_,"nAPVError",iConfig,pDebugStream);
  getConfigForHistogram(nAPVAddressError_,"nAPVAddressError",iConfig,pDebugStream);
  getConfigForHistogram(nUnlocked_,"nUnlocked",iConfig,pDebugStream);
  getConfigForHistogram(nOutOfSync_,"nOutOfSync",iConfig,pDebugStream);

  getConfigForHistogram(nAPVStatusBitvsTime_,"nAPVStatusBitvsTime",iConfig,pDebugStream);
  getConfigForHistogram(nAPVErrorvsTime_,"nAPVErrorvsTime",iConfig,pDebugStream);
  getConfigForHistogram(nAPVAddressErrorvsTime_,"nAPVAddressErrorvsTime",iConfig,pDebugStream);
  getConfigForHistogram(nUnlockedvsTime_,"nUnlockedvsTime",iConfig,pDebugStream);
  getConfigForHistogram(nOutOfSyncvsTime_,"nOutOfSyncvsTime",iConfig,pDebugStream);

  getConfigForHistogram(tkMapConfig_,"TkHistoMap",iConfig,pDebugStream);

  getConfigForHistogram(feTimeDiffTIB_,"FETimeDiffTIB",iConfig,pDebugStream);
  getConfigForHistogram(feTimeDiffTOB_,"FETimeDiffTOB",iConfig,pDebugStream);
  getConfigForHistogram(feTimeDiffTECB_,"FETimeDiffTECB",iConfig,pDebugStream);
  getConfigForHistogram(feTimeDiffTECF_,"FETimeDiffTECF",iConfig,pDebugStream);

  getConfigForHistogram(apveAddress_,"ApveAddress",iConfig,pDebugStream);
  getConfigForHistogram(feMajAddress_,"FeMajAddress",iConfig,pDebugStream);

  getConfigForHistogram(medianAPV0_,"MedianAPV0",iConfig,pDebugStream);
  getConfigForHistogram(medianAPV1_,"MedianAPV1",iConfig,pDebugStream);

  getConfigForHistogram(lumiErrorFraction_,"ErrorFractionByLumiBlock",iConfig,pDebugStream);

}

void FEDHistograms::fillCountersHistograms(const FEDErrors::FEDCounters & fedLevelCounters, 
					   const FEDErrors::ChannelCounters & chLevelCounters, 
					   const unsigned int aMaxSize,
					   const double aTime )
{
  fillHistogram(fedMaxEventSizevsTime_,aTime,aMaxSize);

  fillHistogram(nFEDErrors_,fedLevelCounters.nFEDErrors);
  fillHistogram(nFEDDAQProblems_,fedLevelCounters.nDAQProblems);
  fillHistogram(nFEDsWithFEProblems_,fedLevelCounters.nFEDsWithFEProblems);
  fillHistogram(nFEDCorruptBuffers_,fedLevelCounters.nCorruptBuffers);
  fillHistogram(nFEDsWithFEOverflows_,fedLevelCounters.nFEDsWithFEOverflows);
  fillHistogram(nFEDsWithFEBadMajorityAddresses_,fedLevelCounters.nFEDsWithFEBadMajorityAddresses);
  fillHistogram(nFEDsWithMissingFEs_,fedLevelCounters.nFEDsWithMissingFEs);
  fillHistogram(nBadChannelStatusBits_,fedLevelCounters.nBadChannels);
  fillHistogram(nBadActiveChannelStatusBits_,fedLevelCounters.nBadActiveChannels);

  fillHistogram(nFEDErrorsvsTime_,aTime,fedLevelCounters.nFEDErrors);
  fillHistogram(nFEDCorruptBuffersvsTime_,aTime,fedLevelCounters.nCorruptBuffers);
  fillHistogram(nFEDsWithFEProblemsvsTime_,aTime,fedLevelCounters.nFEDsWithFEProblems);

  fillHistogram(nUnconnectedChannels_,chLevelCounters.nNotConnected);

  fillHistogram(nTotalBadChannels_,fedLevelCounters.nTotalBadChannels);
  fillHistogram(nTotalBadActiveChannels_,fedLevelCounters.nTotalBadActiveChannels);

  fillHistogram(nTotalBadChannelsvsTime_,aTime,fedLevelCounters.nTotalBadChannels);
  fillHistogram(nTotalBadActiveChannelsvsTime_,aTime,fedLevelCounters.nTotalBadActiveChannels);
  
  fillHistogram(nAPVStatusBit_,chLevelCounters.nAPVStatusBit);
  fillHistogram(nAPVError_,chLevelCounters.nAPVError);
  fillHistogram(nAPVAddressError_,chLevelCounters.nAPVAddressError);
  fillHistogram(nUnlocked_,chLevelCounters.nUnlocked);
  fillHistogram(nOutOfSync_,chLevelCounters.nOutOfSync);

  fillHistogram(nAPVStatusBitvsTime_,aTime,chLevelCounters.nAPVStatusBit);
  fillHistogram(nAPVErrorvsTime_,aTime,chLevelCounters.nAPVError);
  fillHistogram(nAPVAddressErrorvsTime_,aTime,chLevelCounters.nAPVAddressError);
  fillHistogram(nUnlockedvsTime_,aTime,chLevelCounters.nUnlocked);
  fillHistogram(nOutOfSyncvsTime_,aTime,chLevelCounters.nOutOfSync);

}

void FEDHistograms::fillFEDHistograms(FEDErrors & aFedErr, 
				      const unsigned int aEvtSize,
				      bool lFullDebug)
{
  const FEDErrors::FEDLevelErrors & lFedLevelErrors = aFedErr.getFEDLevelErrors();
  const unsigned int lFedId = aFedErr.fedID();

  fillHistogram(fedEventSize_,lFedId,aEvtSize);

  if (lFedLevelErrors.DataPresent) fillHistogram(dataPresent_,lFedId);

  if (lFedLevelErrors.HasCabledChannels && lFedLevelErrors.DataMissing) fillHistogram(dataMissing_,lFedId);
  
  if (lFedLevelErrors.InvalidBuffers) fillHistogram(invalidBuffers_,lFedId);
  else if (lFedLevelErrors.BadFEDCRCs) fillHistogram(badFEDCRCs_,lFedId);
  else if (lFedLevelErrors.BadDAQCRCs) fillHistogram(badDAQCRCs_,lFedId);
  else if (lFedLevelErrors.BadIDs) fillHistogram(badIDs_,lFedId);
  else if (lFedLevelErrors.BadDAQPacket) fillHistogram(badDAQPacket_,lFedId);
  else if (lFedLevelErrors.CorruptBuffer) fillHistogram(corruptBuffers_,lFedId);

  if (aFedErr.anyFEDErrors()) fillHistogram(anyFEDErrors_,lFedId);
  if (lFedLevelErrors.HasCabledChannels && aFedErr.anyDAQProblems()) fillHistogram(anyDAQProblems_,lFedId);
  if (aFedErr.anyFEProblems()) fillHistogram(anyFEProblems_,lFedId);

  if (lFedLevelErrors.FEsOverflow) fillHistogram(feOverflows_,lFedId);
  if (lFedLevelErrors.FEsMissing) fillHistogram(feMissing_,lFedId);
  if (lFedLevelErrors.FEsBadMajorityAddress) fillHistogram(badMajorityAddresses_,lFedId);

  if (lFedLevelErrors.BadChannelStatusBit) fillHistogram(badChannelStatusBits_,lFedId);
  if (lFedLevelErrors.BadActiveChannelStatusBit) fillHistogram(badActiveChannelStatusBits_,lFedId);

  std::vector<FEDErrors::FELevelErrors> & lFeVec = aFedErr.getFELevelErrors();
  
  for (unsigned int iFe(0); iFe<lFeVec.size(); iFe++){
    fillFEHistograms(lFedId,lFeVec[iFe]);
  }

  std::vector<FEDErrors::ChannelLevelErrors> & lChVec = aFedErr.getChannelLevelErrors();
  for (unsigned int iCh(0); iCh < lChVec.size(); iCh++){
    fillChannelsHistograms(lFedId,lChVec[iCh],lFullDebug);
  }

  std::vector<FEDErrors::APVLevelErrors> & lAPVVec = aFedErr.getAPVLevelErrors();
  for (unsigned int iApv(0); iApv < lAPVVec.size(); iApv++){
    fillAPVsHistograms(lFedId,lAPVVec[iApv],lFullDebug);
  }


}



//fill a histogram if the pointer is not NULL (ie if it has been booked)
void FEDHistograms::fillFEHistograms(const unsigned int aFedId, 
				     const FEDErrors::FELevelErrors & aFeLevelErrors)
{
  const unsigned short lFeId = aFeLevelErrors.FeID;
  if ( (feOverflowDetailed_.enabled && aFeLevelErrors.Overflow) ||
       (badMajorityAddressDetailed_.enabled && aFeLevelErrors.BadMajorityAddress) ||
       (feMissingDetailed_.enabled && aFeLevelErrors.Missing)
       ) bookFEDHistograms(aFedId);
  if (aFeLevelErrors.Overflow) fillHistogram(feOverflowDetailedMap_[aFedId],lFeId);
  else if (aFeLevelErrors.Missing) fillHistogram(feMissingDetailedMap_[aFedId],lFeId);
  else if (aFeLevelErrors.BadMajorityAddress) fillHistogram(badMajorityAddressDetailedMap_[aFedId],lFeId);
  

  if (aFeLevelErrors.TimeDifference != 0) {
    if (aFeLevelErrors.SubDetID == 2 || aFeLevelErrors.SubDetID == 3 || aFeLevelErrors.SubDetID == 4)
      fillHistogram(feTimeDiffTIB_,aFeLevelErrors.TimeDifference);
    else if (aFeLevelErrors.SubDetID == 5)
      fillHistogram(feTimeDiffTOB_,aFeLevelErrors.TimeDifference);
    else if (aFeLevelErrors.SubDetID == 0)
      fillHistogram(feTimeDiffTECB_,aFeLevelErrors.TimeDifference);
    else if (aFeLevelErrors.SubDetID == 1)
      fillHistogram(feTimeDiffTECF_,aFeLevelErrors.TimeDifference);
    fillHistogram(apveAddress_,aFeLevelErrors.Apve);
    fillHistogram(feMajAddress_,aFeLevelErrors.FeMaj);  
  }
}

//fill a histogram if the pointer is not NULL (ie if it has been booked)
void FEDHistograms::fillChannelsHistograms(const unsigned int aFedId, 
					   const FEDErrors::ChannelLevelErrors & aChErr, 
					   bool fullDebug)
{
  unsigned int lChId = aChErr.ChannelID;

  if ( (unlockedDetailed_.enabled && aChErr.Unlocked) ||
       (outOfSyncDetailed_.enabled && aChErr.OutOfSync)
       ) bookFEDHistograms(aFedId,fullDebug);

  if (aChErr.Unlocked) {
    fillHistogram(unlockedDetailedMap_[aFedId],lChId);
  }
  if (aChErr.OutOfSync) {
    fillHistogram(outOfSyncDetailedMap_[aFedId],lChId);    
  }
}


void FEDHistograms::fillAPVsHistograms(const unsigned int aFedId, 
				       const FEDErrors::APVLevelErrors & aAPVErr, 
				       bool fullDebug)
{
  unsigned int lChId = aAPVErr.APVID;

if ( (badStatusBitsDetailed_.enabled && aAPVErr.APVStatusBit) ||
     (apvErrorDetailed_.enabled && aAPVErr.APVError) ||
     (apvAddressErrorDetailed_.enabled && aAPVErr.APVAddressError)
     ) bookFEDHistograms(aFedId,fullDebug);

 if (aAPVErr.APVStatusBit) fillHistogram(badStatusBitsDetailedMap_[aFedId],lChId);
 if (aAPVErr.APVError) fillHistogram(apvErrorDetailedMap_[aFedId],lChId);
 if (aAPVErr.APVAddressError) fillHistogram(apvAddressErrorDetailedMap_[aFedId],lChId);
}

void FEDHistograms::fillMajorityHistograms(const unsigned int aPart,
					   const float aValue,
					   const std::vector<unsigned int> & aFedIdVec){
  if (aPart==0) fillHistogram(feMajFracTIB_,aValue);
  else if (aPart==1) fillHistogram(feMajFracTOB_,aValue);
  else if (aPart==2) fillHistogram(feMajFracTECB_,aValue);
  else if (aPart==3) fillHistogram(feMajFracTECF_,aValue);

  for (unsigned int iFed(0); iFed<aFedIdVec.size(); ++iFed){
    fillHistogram(badMajorityInPartition_,aFedIdVec[iFed]);
  }

}

bool FEDHistograms::feMajHistosEnabled(){
  return ( feMajFracTIB_.enabled ||
	   feMajFracTOB_.enabled ||
	   feMajFracTECB_.enabled ||
	   feMajFracTECF_.enabled ||
	   badMajorityInPartition_.enabled );
}

void FEDHistograms::fillLumiHistograms(const FEDErrors::LumiErrors & aLumErr){
  if (lumiErrorFraction_.enabled && lumiErrorFraction_.monitorEle) {
    lumiErrorFraction_.monitorEle->Reset();
    for (unsigned int iD(0); iD<aLumErr.nTotal.size(); iD++){
      if (aLumErr.nTotal[iD] > 0) fillHistogram(lumiErrorFraction_,iD+1,static_cast<float>(aLumErr.nErrors[iD])/aLumErr.nTotal[iD]);
    }
  }
}



bool FEDHistograms::cmHistosEnabled() {
  return (medianAPV0_.enabled || medianAPV1_.enabled);
}

MonitorElement * FEDHistograms::cmHistPointer(bool aApv1)
{
  if (!aApv1) return medianAPV0_.monitorEle;
  else return medianAPV1_.monitorEle;
}

void FEDHistograms::bookTopLevelHistograms(DQMStore* dqm)
{
  //get FED IDs
  const unsigned int siStripFedIdMin = FEDNumbering::MINSiStripFEDID;
  const unsigned int siStripFedIdMax = FEDNumbering::MAXSiStripFEDID;

  //get the pointer to the dqm object
  dqm_ = dqm;

  //book FED level histograms
  histosBooked_.resize(siStripFedIdMax+1,false);
  debugHistosBooked_.resize(siStripFedIdMax+1,false);

  //book histos
  bookProfile(fedEventSize_,
	      "FedEventSize",
	      "Average FED buffer Size (B) per Event",
	      siStripFedIdMax-siStripFedIdMin+1,
	      siStripFedIdMin-0.5,siStripFedIdMax+0.5,
	      0,
	      42241, //total number of channels
	      "FED-ID",
	      "<FED buffer Size> (B)"
	      );


  bookHistogram(dataPresent_,"DataPresent",
		"Number of events where the data from a FED is seen",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");

  bookHistogram(nTotalBadChannels_,
		"nTotalBadChannels",
		"Number of channels with any error",
		"Total # bad enabled channels");

  bookHistogram(nTotalBadActiveChannels_,
		"nTotalBadActiveChannels",
		"Number of active channels with any error",
		"Total # bad active channels");

  //dqm_->setCurrentFolder(fedKey.path());

  //std::ostringstream lDirName;
  //lDirName << fedKey.path();
  //lDirName << 

  const std::string lBaseDir = dqm_->pwd();

  dqm_->setCurrentFolder(lBaseDir+"/FED");

  bookHistogram(nFEDErrors_,
		"nFEDErrors",
		"Number of FEDs with errors (FED or FE Level) per event",
		"# FEDErrors");

  bookHistogram(nFEDDAQProblems_,
		"nFEDDAQProblems",
		"Number of FEDs with DAQ problems per event",
		"# FEDDAQProblems");

  bookHistogram(nFEDsWithFEProblems_,
		"nFEDsWithFEProblems",
		"Number of FEDs with FE problems per event",
		"# FEDs with FE problems");

  bookHistogram(nFEDCorruptBuffers_,
		"nFEDCorruptBuffers",
		"Number of FEDs with corrupt buffers per event",
		"# FEDs with corrupt buffer");


  dqm_->setCurrentFolder(lBaseDir+"/FED/VsId");

  bookHistogram(dataMissing_,"DataMissing",
		"Number of events where the data from a FED with cabled channels is missing",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");

  bookHistogram(anyFEDErrors_,"AnyFEDErrors",
		"Number of buffers with any FED error (excluding bad channel status bits, FE problems except overflows) per FED",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");

  bookHistogram(corruptBuffers_,"CorruptBuffers",
		"Number of corrupt FED buffers per FED",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");

  bookHistogram(invalidBuffers_,"InvalidBuffers",
		"Number of invalid FED buffers per FED",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");

  bookHistogram(anyDAQProblems_,"AnyDAQProblems",
		"Number of buffers with any problems flagged in DAQ header (including CRC)",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");

  bookHistogram(badIDs_,"BadIDs",
		"Number of buffers with non-SiStrip source IDs in DAQ header",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");

  bookHistogram(badDAQCRCs_,"BadDAQCRCs",
		"Number of buffers with bad CRCs from the DAQ",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");

  bookHistogram(badFEDCRCs_,"BadFEDCRCs",
		"Number of buffers with bad CRCs from the FED",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");

  bookHistogram(badDAQPacket_,"BadDAQPacket",
		"Number of buffers with (non-CRC) problems flagged in DAQ header/trailer",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");

  dqm_->setCurrentFolder(lBaseDir+"/FE");

  bookHistogram(nFEDsWithFEOverflows_,
		"nFEDsWithFEOverflows",
		"Number FEDs with FE units which overflowed per event",
		"# FEDs with FE overflows");

  bookHistogram(nFEDsWithFEBadMajorityAddresses_,
		"nFEDsWithFEBadMajorityAddresses",
		"Number of FEDs with FE units with a bad majority address per event",
		"# FEDs with bad address");

  bookHistogram(nFEDsWithMissingFEs_,
		"nFEDsWithMissingFEs",
		"Number of FEDs with missing FE unit payloads per event",
		"# FEDs with missing FEs");

  bookHistogram(feMajFracTIB_,"FeMajFracTIB",
		"Fraction of FEs matching majority address in TIB partition",
		101,0,1.01,"n(majAddrFE)/n(totFE)");

  bookHistogram(feMajFracTOB_,"FeMajFracTOB",
		"Fraction of FEs matching majority address in TOB partition",
		101,0,1.01,"n(majAddrFE)/n(totFE)");

  bookHistogram(feMajFracTECB_,"FeMajFracTECB",
		"Fraction of FEs matching majority address in TECB partition",
		101,0,1.01,"n(majAddrFE)/n(totFE)");

  bookHistogram(feMajFracTECF_,"FeMajFracTECF",
		"Fraction of FEs matching majority address in TECF partition",
		101,0,1.01,"n(majAddrFE)/n(totFE)");


  dqm_->setCurrentFolder(lBaseDir+"/FE/APVe");

  bookHistogram(feTimeDiffTIB_,"FETimeDiffTIB",
		"(TimeLoc FE - TimeLoc APVe) for TIB/TID, when different",
		401,
		-200,201,"#Delta_{TimeLoc}(FE-APVe)");

  bookHistogram(feTimeDiffTOB_,"FETimeDiffTOB",
		"(TimeLoc FE - TimeLoc APVe) for TOB, when different",
		401,
		-200,201,"#Delta_{TimeLoc}(FE-APVe)");

  bookHistogram(feTimeDiffTECB_,"FETimeDiffTECB",
		"(TimeLoc FE - TimeLoc APVe) for TECB, when different",
		401,
		-200,201,"#Delta_{TimeLoc}(FE-APVe)");

  bookHistogram(feTimeDiffTECF_,"FETimeDiffTECF",
		"(TimeLoc FE - TimeLoc APVe) for TECF, when different",
		401,
		-200,201,"#Delta_{TimeLoc}(FE-APVe)");


  bookHistogram(apveAddress_,"ApveAddress",
		"apve Address",
		256,0,256,
		"apveAddress");

  bookHistogram(feMajAddress_,"FeMajAddress",
		"FE Majority Address",
		256,0,256,
		"feMajAddress");


  dqm_->setCurrentFolder(lBaseDir+"/FE/VsId");

  bookHistogram(anyFEProblems_,"AnyFEProblems",
		"Number of buffers with any FE unit problems",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  
  bookHistogram(feOverflows_,"FEOverflows",
		"Number of buffers with one or more FE overflow",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");

  bookHistogram(badMajorityAddresses_,"BadMajorityAddresses",
		"Number of buffers with one or more FE with a bad majority APV address",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");

  bookHistogram(badMajorityInPartition_,"BadMajorityInPartition",
		"Number of buffers with >=1 FE with FEaddress != majority in partition",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");

  bookHistogram(feMissing_,"FEMissing",
		"Number of buffers with one or more FE unit payload missing",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  

  dqm_->setCurrentFolder(lBaseDir+"/Fiber");

  bookHistogram(nBadChannelStatusBits_,
		"nBadChannelStatusBits",
		"Number of channels with bad status bits per event",
		"# bad enabled channels");

  bookHistogram(nBadActiveChannelStatusBits_,
		"nBadActiveChannelStatusBits",
		"Number of active channels with bad status bits per event",
		"# bad active channels");

  bookHistogram(nUnlocked_,
		"nUnlocked",
		"Number of channels Unlocked per event",
		"# channels unlocked");

  bookHistogram(nOutOfSync_,
		"nOutOfSync",
		"Number of channels OutOfSync per event",
		"# channels out-of-sync");

  bookHistogram(nUnconnectedChannels_,
		"nUnconnectedChannels",
		"Number of channels not connected per event",
		"# unconnected channels");

  dqm_->setCurrentFolder(lBaseDir+"/Fiber/VsId");

  bookHistogram(badChannelStatusBits_,"BadChannelStatusBits",
		"Number of buffers with one or more enabled channel with bad status bits",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");

  bookHistogram(badActiveChannelStatusBits_,"BadActiveChannelStatusBits",
		"Number of buffers with one or more active channel with bad status bits",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");

  dqm_->setCurrentFolder(lBaseDir+"/APV");

  bookHistogram(medianAPV0_,"MedianAPV0",
		"Median APV0",
		"medianAPV0");
  
  bookHistogram(medianAPV1_,"MedianAPV1",
		"Median APV1",
		"MedianAPV1");
  
  bookHistogram(nAPVStatusBit_,
		"nAPVStatusBit",
		"Number of APVs with APVStatusBit error per event",
		"# APVs with APVStatusBit error");

  bookHistogram(nAPVError_,
		"nAPVError",
		"Number of APVs with APVError per event",
		"#APVs with APVError");

  bookHistogram(nAPVAddressError_,
		"nAPVAddressError",
		"Number of APVs with APVAddressError per event",
		"#APVs with APVAddressError");


  dqm_->setCurrentFolder(lBaseDir+"/Trends");

  bookProfile(fedMaxEventSizevsTime_,
	      "FedMaxEventSizevsTime",
	      "Max FED buffer Size (B) per Event vs time",
	      0,
	      42241, //total number of channels
	      "Time",
	      "Max FED buffer Size (B)"
	      );

  bookProfile(nTotalBadChannelsvsTime_,
	      "nTotalBadChannelsvsTime",
	      "Number of channels with any error vs time",
	      0,
	      42241, //total number of channels
	      "Time",
	      "Total # bad enabled channels"
	      );


  bookProfile(nTotalBadActiveChannelsvsTime_,
	      "nTotalBadActiveChannelsvsTime",
	      "Number of active channels with any error vs time",
	      0,
	      42241, //total number of channels
	      "Time",
	      "Total # bad active channels"
	      );

  dqm_->setCurrentFolder(lBaseDir+"/Trends/FED");

  bookProfile(nFEDErrorsvsTime_,
	      "nFEDErrorsvsTime",
	      "Number of FEDs with any error vs time",
	      0,
	      42241, //total number of channels
	      "Time",
	      "# FEDErrors"
	      );

  bookProfile(nFEDCorruptBuffersvsTime_,
	      "nFEDCorruptBuffersvsTime",
	      "Number of FEDs with corrupt buffer vs time",
	      0,
	      42241, //total number of channels
	      "Time",
	      "# FEDCorruptBuffer"
	      );

  dqm_->setCurrentFolder(lBaseDir+"/Trends/FE");

  bookProfile(nFEDsWithFEProblemsvsTime_,
	      "nFEDsWithFEProblemsvsTime",
	      "Number of FEDs with any FE error vs time",
	      0,
	      42241, //total number of channels
	      "Time",
	      "# FEDsWithFEProblems"
	      );

  dqm_->setCurrentFolder(lBaseDir+"/Trends/Fiber");

  bookProfile(nUnlockedvsTime_,
	      "nUnlockedvsTime",
	      "Number of channels Unlocked vs time",
	      0,
	      42241, //total number of channels
	      "Time",
	      "# channels unlocked "
	      );

  bookProfile(nOutOfSyncvsTime_,
	      "nOutOfSyncvsTime",
	      "Number of channels OutOfSync vs time",
	      0,
	      42241, //total number of channels
	      "Time",
	      "# channels out-of-sync"
	      );

  dqm_->setCurrentFolder(lBaseDir+"/Trends/APV");

  bookProfile(nAPVStatusBitvsTime_,
	      "nAPVStatusBitvsTime",
	      "Number of APVs with APVStatusBit error vs time",
	      0,
	      42241, //total number of channels
	      "Time",
	      "# APVs with APVStatusBit error"
	      );

  bookProfile(nAPVErrorvsTime_,
	      "nAPVErrorvsTime",
	      "Number of APVs with APVError vs time",
	      0,
	      42241, //total number of channels
	      "Time",
	      "# APVs with APVError"
	      );

  bookProfile(nAPVAddressErrorvsTime_,
	      "nAPVAddressErrorvsTime",
	      "Number of APVs with APVAddressError vs time",
	      0,
	      42241, //total number of channels
	      "Time",
	      "# APVs with APVAddressError"
	      );

  dqm_->setCurrentFolder(lBaseDir+"/PerLumiSection");

  bookHistogram(lumiErrorFraction_,
		"lumiErrorFraction",
		"Fraction of error per lumi section vs subdetector",
		6,0.5,6.5,
		"SubDetId");

  //Set special property for lumi ME
  if (lumiErrorFraction_.enabled && lumiErrorFraction_.monitorEle) {
    lumiErrorFraction_.monitorEle->setLumiFlag();
    lumiErrorFraction_.monitorEle->setBinLabel(1, "TECB");
    lumiErrorFraction_.monitorEle->setBinLabel(2, "TECF");
    lumiErrorFraction_.monitorEle->setBinLabel(3, "TIB");
    lumiErrorFraction_.monitorEle->setBinLabel(4, "TIDB");
    lumiErrorFraction_.monitorEle->setBinLabel(5, "TIDF");
    lumiErrorFraction_.monitorEle->setBinLabel(6, "TOB");
  }

  //book map after, as it creates a new folder...
  if (tkMapConfig_.enabled){
    //const std::string dqmPath = dqm_->pwd();
    tkmapFED_ = new TkHistoMap("SiStrip/TkHisto","TkHMap_FractionOfBadChannels",0.,1);
  }
  else tkmapFED_ = 0;

}

void FEDHistograms::bookFEDHistograms(unsigned int fedId,
				      bool fullDebugMode
				      )
{


  if (!histosBooked_[fedId]) {


    //will do that only once
    SiStripFedKey fedKey(fedId,0,0,0);
    std::stringstream fedIdStream;
    fedIdStream << fedId;
    dqm_->setCurrentFolder(fedKey.path());
    bookHistogram(feOverflowDetailed_,
		  feOverflowDetailedMap_[fedId],
		  "FEOverflowsForFED"+fedIdStream.str(),
		  "FE overflows per FE unit for FED ID "+fedIdStream.str(),
		  sistrip::FEUNITS_PER_FED,0,sistrip::FEUNITS_PER_FED,
		  "FE-Index");
    bookHistogram(badMajorityAddressDetailed_,
		  badMajorityAddressDetailedMap_[fedId],
		  "BadMajorityAddressesForFED"+fedIdStream.str(),
		  "Bad majority APV addresses per FE unit for FED ID "+fedIdStream.str(),
		  sistrip::FEUNITS_PER_FED,0,sistrip::FEUNITS_PER_FED,
		  "FE-Index");
    bookHistogram(feMissingDetailed_,
		  feMissingDetailedMap_[fedId],
		  "FEMissingForFED"+fedIdStream.str(),
		  "Buffers with FE Unit payload missing per FE unit for FED ID "+fedIdStream.str(),
		  sistrip::FEUNITS_PER_FED,0,sistrip::FEUNITS_PER_FED,
		  "FE-Index");
    bookHistogram(badStatusBitsDetailed_,
		  badStatusBitsDetailedMap_[fedId],
		  "BadAPVStatusBitsForFED"+fedIdStream.str(),
		  "Bad apv status bits for FED ID "+fedIdStream.str(),
		  sistrip::APVS_PER_FED,0,sistrip::APVS_PER_FED,
		  "APV-Index");
     histosBooked_[fedId] = true;
  }
  if (fullDebugMode && !debugHistosBooked_[fedId]) {
    //will do that only once
    SiStripFedKey fedKey(fedId,0,0,0);
    std::stringstream fedIdStream;
    fedIdStream << fedId;
    dqm_->setCurrentFolder(fedKey.path());

    bookHistogram(apvErrorDetailed_,
		  apvErrorDetailedMap_[fedId],
		  "APVErrorBitsForFED"+fedIdStream.str(),
		  "APV errors for FED ID "+fedIdStream.str(),
		  sistrip::APVS_PER_FED,0,sistrip::APVS_PER_FED,
		  "APV-Index");
    bookHistogram(apvAddressErrorDetailed_,
		  apvAddressErrorDetailedMap_[fedId],
		  "APVAddressErrorBitsForFED"+fedIdStream.str(),
		  "Wrong APV address errors for FED ID "+fedIdStream.str(),
		  sistrip::APVS_PER_FED,0,sistrip::APVS_PER_FED,
		  "APV-Index");
    bookHistogram(unlockedDetailed_,
		  unlockedDetailedMap_[fedId],
		  "UnlockedBitsForFED"+fedIdStream.str(),
		  "Unlocked channels for FED ID "+fedIdStream.str(),
		  sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED,
		  "Channel-Index");
    bookHistogram(outOfSyncDetailed_,
		  outOfSyncDetailedMap_[fedId],
		  "OOSBitsForFED"+fedIdStream.str(),
		  "Out of sync channels for FED ID "+fedIdStream.str(),
		  sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED,
		  "Channel-Index");
    debugHistosBooked_[fedId] = true;
  }
}

void FEDHistograms::bookAllFEDHistograms()
{
  //get FED IDs
  const unsigned int siStripFedIdMin = FEDNumbering::MINSiStripFEDID;
  const unsigned int siStripFedIdMax = FEDNumbering::MAXSiStripFEDID;
  //book them
  for (unsigned int iFed = siStripFedIdMin; iFed <= siStripFedIdMax; iFed++) {
    bookFEDHistograms(iFed,true);
  }
}

bool FEDHistograms::tkHistoMapEnabled(unsigned int aIndex){
  return tkMapConfig_.enabled;
}

TkHistoMap * FEDHistograms::tkHistoMapPointer(unsigned int aIndex){
  return tkmapFED_;
}
