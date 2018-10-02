#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "EventFilter/SiStripRawToDigi/interface/PipeAddrToTimeLookupTable.h"

#include "DQM/SiStripMonitorHardware/interface/HistogramBase.hh"
#include "DQM/SiStripMonitorHardware/interface/FEDErrors.hh"
#include "DQM/SiStripCommon/interface/TkHistoMap.h"


FEDErrors::FEDErrors()
{
  //initialiseLumiBlock();
  initialiseEvent();
}

FEDErrors::~FEDErrors()
{

}

void FEDErrors::initialiseLumiBlock() {
  lumiErr_.nTotal.clear();
  lumiErr_.nErrors.clear();
  //6 subdetectors:
  //TECB,TECF,TIB,TIDB,TIDF,TOB
  lumiErr_.nTotal.resize(6,0);
  lumiErr_.nErrors.resize(6,0);
}

void FEDErrors::initialiseEvent() {
  fedID_ = 0;
  failUnpackerFEDCheck_ = false;

  connected_.clear();
  detid_.clear();
  nChInModule_.clear();

  subDetId_.clear();

  lFedCounter_.nFEDErrors = 0;
  lFedCounter_.nDAQProblems = 0;
  lFedCounter_.nFEDsWithFEProblems = 0;
  lFedCounter_.nCorruptBuffers = 0;
  lFedCounter_.nBadChannels = 0;
  lFedCounter_.nBadActiveChannels = 0;
  lFedCounter_.nFEDsWithFEOverflows = 0;
  lFedCounter_.nFEDsWithFEBadMajorityAddresses = 0;
  lFedCounter_.nFEDsWithMissingFEs = 0;
  lFedCounter_.nTotalBadChannels = 0;
  lFedCounter_.nTotalBadActiveChannels = 0;

  lChCounter_.nNotConnected = 0;
  lChCounter_.nUnlocked = 0;
  lChCounter_.nOutOfSync = 0;
  lChCounter_.nAPVStatusBit = 0;
  lChCounter_.nAPVError = 0;
  lChCounter_.nAPVAddressError = 0;

  feCounter_.nFEOverflows = 0;
  feCounter_.nFEBadMajorityAddresses = 0;
  feCounter_.nFEMissing = 0;

  fedErrors_.HasCabledChannels = false;
  fedErrors_.DataPresent = false;
  fedErrors_.DataMissing = false;
  fedErrors_.InvalidBuffers = false;
  fedErrors_.BadFEDCRCs = false;
  fedErrors_.BadDAQCRCs = false;
  fedErrors_.BadIDs = false;
  fedErrors_.BadDAQPacket = false;
  fedErrors_.CorruptBuffer = false;
  fedErrors_.FEsOverflow = false;
  fedErrors_.FEsMissing = false;
  fedErrors_.FEsBadMajorityAddress = false;
  fedErrors_.BadChannelStatusBit = false;
  fedErrors_.BadActiveChannelStatusBit = false;

  feErrors_.clear();

  chErrorsDetailed_.clear();

  apvErrors_.clear();

  chErrors_.clear();

  eventProp_.deltaBX=0;
}



void FEDErrors::initialiseFED(const unsigned int aFedID,
			      const SiStripFedCabling* aCabling,
                              const TrackerTopology* tTopo,
			      const bool initVars)
{
  fedID_ = aFedID;
  failUnpackerFEDCheck_ = false;

  //initialise first.  if no channel connected in one FE, subdetid =
  //0.  in the loop on channels, if at least one channel is connected
  //and has a valid ID, the subdet value will be changed to the right
  //one.
  if (initVars) {

    subDetId_.resize(sistrip::FEUNITS_PER_FED,0);

    connected_.resize(sistrip::FEDCH_PER_FED,false);
    detid_.resize(sistrip::FEDCH_PER_FED,0);
    nChInModule_.resize(sistrip::FEDCH_PER_FED,0);

    for (unsigned int iCh = 0;
	 iCh < sistrip::FEDCH_PER_FED;
	 iCh++) {

      const FedChannelConnection & lConnection = aCabling->fedConnection(fedID_,iCh);
      connected_[iCh] = lConnection.isConnected();
      detid_[iCh] = lConnection.detId();
      nChInModule_[iCh] = lConnection.nApvPairs();

      unsigned short lFeNumber = static_cast<unsigned int>(iCh/sistrip::FEDCH_PER_FEUNIT);
      unsigned int lDetid = detid_[iCh];

      if (lDetid && lDetid != sistrip::invalid32_ && connected_[iCh]) {
	unsigned int lSubid = 6;
	// 3=TIB, 4=TID, 5=TOB, 6=TEC (TECB here)
	switch(DetId(lDetid).subdetId()) {
	case 3:
	  lSubid = 2; //TIB
	  break;

	case 4:
	  {

	    if (tTopo->tidSide(lDetid) == 2) lSubid = 4; //TIDF
	    else lSubid = 3; //TIDB
	    break;
	  }

	case 5:
	  lSubid = 5; //TOB
	  break;

	case 6:
	  {

	    if (tTopo->tecSide(lDetid) == 2) lSubid = 1; //TECF
	    else lSubid = 0; //TECB
	    break;
	  }

	default:
	  lSubid = 6;
	  break;

	}
	subDetId_[lFeNumber] = lSubid;
	//if (iCh%12==0) std::cout << fedID_ << " " << lFeNumber << " " << subDetId_[lFeNumber] << std::endl;

      }
    }


    feCounter_.nFEOverflows = 0;
    feCounter_.nFEBadMajorityAddresses = 0;
    feCounter_.nFEMissing = 0;

    fedErrors_.HasCabledChannels = false;
    fedErrors_.DataPresent = false;
    fedErrors_.DataMissing = false;
    fedErrors_.InvalidBuffers = false;
    fedErrors_.BadFEDCRCs = false;
    fedErrors_.BadDAQCRCs = false;
    fedErrors_.BadIDs = false;
    fedErrors_.BadDAQPacket = false;
    fedErrors_.CorruptBuffer = false;
    fedErrors_.FEsOverflow = false;
    fedErrors_.FEsMissing = false;
    fedErrors_.FEsBadMajorityAddress = false;
    fedErrors_.BadChannelStatusBit = false;
    fedErrors_.BadActiveChannelStatusBit = false;

    feErrors_.clear();

    chErrorsDetailed_.clear();

    apvErrors_.clear();

    chErrors_.clear();
  }
}

bool FEDErrors::checkDataPresent(const FEDRawData& aFedData)
{

  if (!aFedData.size() || !aFedData.data()) {
    for (unsigned int iCh = 0;
	 iCh < sistrip::FEDCH_PER_FED;
	 iCh++) {
      if (connected_[iCh]){
	fedErrors_.HasCabledChannels = true;
	fedErrors_.DataMissing = true;
	return false;
      }
    }
    fedErrors_.DataMissing = true;
    fedErrors_.HasCabledChannels = false;
    return false;
  } else {
    fedErrors_.DataPresent = true;
    for (unsigned int iCh = 0;
	 iCh < sistrip::FEDCH_PER_FED;
	 iCh++) {
      if (connected_[iCh]){
	fedErrors_.HasCabledChannels = true;
	break;
      }
    }
    return true;
  }

}

bool FEDErrors::failUnpackerFEDCheck()
{
  return failUnpackerFEDCheck_;
}


bool FEDErrors::fillFatalFEDErrors(const FEDRawData& aFedData,
				   const unsigned int aPrintDebug)
{

  std::unique_ptr<const sistrip::FEDBufferBase> bufferBase;
  try {
    bufferBase.reset(new sistrip::FEDBufferBase(aFedData.data(),aFedData.size()));
  } catch (const cms::Exception& e) {
    fedErrors_.InvalidBuffers = true;
    failUnpackerFEDCheck_ = true;
    //don't check anything else if the buffer is invalid
    return false;
  }

  //CRC checks
  //if CRC fails then don't continue as if the buffer has been corrupted in DAQ then anything else could be invalid
  if (!bufferBase->checkNoSlinkCRCError()) {
    fedErrors_.BadFEDCRCs = true;
    return false;
  } else if (!bufferBase->checkCRC()) {
    failUnpackerFEDCheck_ = true;
    fedErrors_.BadDAQCRCs = true;
    return false;
  }
  //next check that it is a SiStrip buffer
  //if not then stop checks
  if (!bufferBase->checkSourceIDs() || !bufferBase->checkNoUnexpectedSourceID()) {
    fedErrors_.BadIDs = true;
    return false;
  }
  //if so then do DAQ header/trailer checks
  //if these fail then buffer may be incomplete and checking contents doesn't make sense
  else if (!bufferBase->doDAQHeaderAndTrailerChecks()) {
    failUnpackerFEDCheck_ = true;
    fedErrors_.BadDAQPacket = true;
    return false;
  }

  //now do checks on header
  //check that tracker special header is consistent
  if ( !(bufferBase->checkBufferFormat() &&
	 bufferBase->checkHeaderType() &&
	 bufferBase->checkReadoutMode()) ) {
    failUnpackerFEDCheck_ = true;
    fedErrors_.InvalidBuffers = true;
    //do not return false if debug printout of the buffer done below...
    if (!printDebug() || aPrintDebug<3 ) return false;
  }

  //FE unit overflows
  if (!bufferBase->checkNoFEOverflows()) {
    failUnpackerFEDCheck_ = true;
    fedErrors_.FEsOverflow = true;
    //do not return false if debug printout of the buffer done below...
    if (!printDebug() || aPrintDebug<3 ) return false;
  }

  return true;

}

bool FEDErrors::fillCorruptBuffer(const sistrip::FEDBuffer* aBuffer)
{
  //corrupt buffer checks
  if (!(aBuffer->checkChannelLengthsMatchBufferLength() &&
	aBuffer->checkChannelPacketCodes() &&
	aBuffer->checkFEUnitLengths())) {
    fedErrors_.CorruptBuffer = true;

    return false;
  }

  return true;

}

float FEDErrors::fillNonFatalFEDErrors(const sistrip::FEDBuffer* aBuffer,
				       const SiStripFedCabling* aCabling)
{
  unsigned int lBadChans = 0;
  unsigned int lTotChans = 0;
  for (unsigned int iCh = 0; iCh < sistrip::FEDCH_PER_FED; iCh++) {//loop on channels
    bool lIsConnected = false;
    if (aCabling) {
      const FedChannelConnection & lConnection = aCabling->fedConnection(fedID_,iCh);
      lIsConnected = lConnection.isConnected();
    }
    else lIsConnected = connected_[iCh];

    if (!lIsConnected) continue;
    lTotChans++;
    if (!aBuffer->channelGood(iCh, true)) lBadChans++;
  }

  return static_cast<float>(lBadChans*1.0/lTotChans);
}


bool FEDErrors::fillFEDErrors(const FEDRawData& aFedData,
			      bool & aFullDebug,
			      const unsigned int aPrintDebug,
			      unsigned int & aCounterMonitoring,
			      unsigned int & aCounterUnpacker,
			      const bool aDoMeds,
			      MonitorElement *aMedianHist0,
			      MonitorElement *aMedianHist1,
			      const bool aDoFEMaj,
			      std::vector<std::vector<std::pair<unsigned int,unsigned int> > > & aFeMajFrac
			      )
{
  //try to construct the basic buffer object (do not check payload)
  //if this fails then count it as an invalid buffer and stop checks since we can't understand things like buffer ordering

  if (!fillFatalFEDErrors(aFedData,aPrintDebug)) return false;

  //need to construct full object to go any further
  std::unique_ptr<const sistrip::FEDBuffer> buffer;
  buffer.reset(new sistrip::FEDBuffer(aFedData.data(),aFedData.size(),true));

  //fill remaining unpackerFEDcheck
  if (!buffer->checkChannelLengths()) failUnpackerFEDCheck_= true;

  //payload checks, only if none of the above error occured
  if (!this->anyFEDErrors()) {

    bool lCorruptCheck = fillCorruptBuffer(buffer.get());
    if (aPrintDebug>1 && !lCorruptCheck) {
      edm::LogWarning("SiStripMonitorHardware")
	<< "CorruptBuffer check failed for FED " << fedID_
	<< std::endl
	<< " -- buffer->checkChannelLengthsMatchBufferLength() = " << buffer->checkChannelLengthsMatchBufferLength()
	<< std::endl
	<< " -- buffer->checkChannelPacketCodes() = " << buffer->checkChannelPacketCodes()
	<< std::endl
	<< " -- buffer->checkFEUnitLengths() = " << buffer->checkFEUnitLengths()
	<< std::endl;
    }

    //corruptBuffer concerns the payload: header info should still be reliable...
    //so analyze FE and channels to fill histograms.

    //fe check...
    fillFEErrors(buffer.get(),aDoFEMaj,aFeMajFrac);

    //channel checks
    fillChannelErrors(buffer.get(),
		      aFullDebug,
		      aPrintDebug,
		      aCounterMonitoring,
		      aCounterUnpacker,
		      aDoMeds,
		      aMedianHist0,
		      aMedianHist1
		      );

  }


  if (printDebug() && aPrintDebug>2) {
    const sistrip::FEDBufferBase* debugBuffer = nullptr;

    if (buffer.get()) debugBuffer = buffer.get();
    //else if (bufferBase.get()) debugBuffer = bufferBase.get();
    if (debugBuffer) {
      std::vector<FEDErrors::APVLevelErrors> & lChVec = getAPVLevelErrors();
      std::ostringstream debugStream;
      if (!lChVec.empty()) {
	std::sort(lChVec.begin(),lChVec.end());
        debugStream << "[FEDErrors] Cabled channels which had errors: ";

        for (unsigned int iBadCh(0); iBadCh < lChVec.size(); iBadCh++) {
          print(lChVec[iBadCh],debugStream);
        }
        debugStream << std::endl;
        debugStream << "[FEDErrors] Active (have been locked in at least one event) cabled channels which had errors: ";
	for (unsigned int iBadCh(0); iBadCh < lChVec.size(); iBadCh++) {
          if ((lChVec[iBadCh]).IsActive) print(lChVec[iBadCh],debugStream);
        }

      }
      debugStream << (*debugBuffer) << std::endl;
      debugBuffer->dump(debugStream);
      debugStream << std::endl;
      edm::LogInfo("SiStripMonitorHardware") << "[FEDErrors] Errors found in FED " << fedID_;
      edm::LogVerbatim("SiStripMonitorHardware") << debugStream.str();
    }
  }

  return !(anyFEDErrors());
}

bool FEDErrors::fillFEErrors(const sistrip::FEDBuffer* aBuffer,
			     const bool aDoFEMaj,
			     std::vector<std::vector<std::pair<unsigned int,unsigned int> > > & aFeMajFrac)
{
  bool foundOverflow = false;
  bool foundBadMajority = false;
  bool foundMissing = false;
  for (unsigned int iFE = 0; iFE < sistrip::FEUNITS_PER_FED; iFE++) {

    FEDErrors::FELevelErrors lFeErr;
    lFeErr.FeID = iFE;
    lFeErr.SubDetID = subDetId_[iFE];
    lFeErr.Overflow = false;
    lFeErr.Missing = false;
    lFeErr.BadMajorityAddress = false;
    lFeErr.TimeDifference = 0;
    lFeErr.Apve = 0;
    lFeErr.FeMaj = 0;
    //check for cabled channels
    bool hasCabledChannels = false;
    for (unsigned int feUnitCh = 0; feUnitCh < sistrip::FEDCH_PER_FEUNIT; feUnitCh++) {
      if (connected_[iFE*sistrip::FEDCH_PER_FEUNIT+feUnitCh]) {
        hasCabledChannels = true;
        break;
      }
    }

    if (!hasCabledChannels) continue;

    if (aBuffer->feOverflow(iFE)) {
      lFeErr.Overflow = true;
      foundOverflow = true;
      addBadFE(lFeErr);
      //if FE overflowed then address isn't valid
      continue;
    }
    if (!aBuffer->feEnabled(iFE)) continue;

    //check for missing data
    if (!aBuffer->fePresent(iFE)) {
      //if (hasCabledChannels) {
      lFeErr.Missing = true;
      foundMissing = true;
      addBadFE(lFeErr);
      //}
      continue;
    }
    //two independent checks for the majority address of a FE:
    //first is done inside the FED,
    //second is comparing explicitely the FE majAddress with the APVe address.
    //!aBuffer->checkFEUnitAPVAddresses(): for all FE's....
    //want to do it only for this FE... do it directly with the time difference.
    if (aBuffer->majorityAddressErrorForFEUnit(iFE)){
      lFeErr.BadMajorityAddress = true;
      foundBadMajority = true;
      //no continue to fill the timeDifference.
    }

    //need fullDebugHeader to fill histo with time difference between APVe and FEmajAddress
    const sistrip::FEDFEHeader* header = aBuffer->feHeader();
    const sistrip::FEDFullDebugHeader* debugHeader = dynamic_cast<const sistrip::FEDFullDebugHeader*>(header);
    // if (debugHeader) {
    //   unsigned int apveTime = static_cast<unsigned int>(sistrip::FEDAddressConversion::timeLocation(aBuffer->apveAddress()));
    //   unsigned int feTime = static_cast<unsigned int>(sistrip::FEDAddressConversion::timeLocation(debugHeader->feUnitMajorityAddress(iFE)));
    //   if ((apveTime == 200 && aBuffer->apveAddress()) || feTime == 200) {
    // 	std::cout << "FED " << fedID_ << ", iFE = " << iFE << std::endl
    // 		<< " -- aBuffer->apveAddress() = " << static_cast<unsigned int>(aBuffer->apveAddress())
    // // 		<< ", debugHeader = " << debugHeader
    // // 		<< ", header->feGood(iFE) = " << aBuffer->feGood(iFE)
    //  		<< ", debugHeader->feUnitMajorityAddress(iFE) " << static_cast<unsigned int>(debugHeader->feUnitMajorityAddress(iFE))
    //  		<< std::endl
    //  		<< " -- timeLoc(feUnitMajAddr) = "
    //  		<< static_cast<unsigned int>(sistrip::FEDAddressConversion::timeLocation(debugHeader->feUnitMajorityAddress(iFE)))
    //  		<< ", timeLoc(apveAddr) = "
    //  		<< static_cast<unsigned int>(sistrip::FEDAddressConversion::timeLocation(aBuffer->apveAddress()))
    // // 		<< ", aBuffer->checkFEUnitAPVAddresses() = "
    // // 		<< aBuffer->checkFEUnitAPVAddresses()
    //  		<< std::endl;
    // 	std::cout << "My checks = "
    // 		<< ", feOverflows = " << lFeErr.Overflow << " " << foundOverflow
    // 		<< ", feMissing = " << lFeErr.Missing << " " << foundMissing
    // 		<< ", feBadMajAddr = " << lFeErr.BadMajorityAddress  << " " << foundBadMajority
    // 		<< std::endl;

    // 	std::cout << "TimeDiff = " << feTime-apveTime << std::endl;

    // 	//   std::cout << "aBuffer->checkFEUnitAPVAddresses() = " << aBuffer->checkFEUnitAPVAddresses() << std::endl;
    //   }
    // }

    lFeErr.Apve = aBuffer->apveAddress();

    if (debugHeader){

      lFeErr.FeMaj = debugHeader->feUnitMajorityAddress(iFE);

      if (aDoFEMaj){
	if (lFeErr.SubDetID == 2 || lFeErr.SubDetID == 3 || lFeErr.SubDetID == 4)
	  aFeMajFrac[0].push_back(std::pair<unsigned int, unsigned int>(fedID_,lFeErr.FeMaj));
	else if (lFeErr.SubDetID == 5)
	  aFeMajFrac[1].push_back(std::pair<unsigned int, unsigned int>(fedID_,lFeErr.FeMaj));
	else if (lFeErr.SubDetID == 0)
	  aFeMajFrac[2].push_back(std::pair<unsigned int, unsigned int>(fedID_,lFeErr.FeMaj));
	else if (lFeErr.SubDetID == 1)
	  aFeMajFrac[3].push_back(std::pair<unsigned int, unsigned int>(fedID_,lFeErr.FeMaj));
      }


      if (aBuffer->apveAddress()){
	lFeErr.TimeDifference = //0;
	  static_cast<unsigned int>(sistrip::FEDAddressConversion::timeLocation(debugHeader->feUnitMajorityAddress(iFE)))-static_cast<unsigned int>(sistrip::FEDAddressConversion::timeLocation(aBuffer->apveAddress()));
	//aBuffer->apveAddress(), debugHeader->feUnitMajorityAddress(iFE)
	//FEDAddressConversion::timeLocation(const uint8_t aPipelineAddress)
      }
    }

    if (foundBadMajority || lFeErr.TimeDifference != 0){
      addBadFE(lFeErr);


 //      LogDebug("SiStripMonitorHardware") << " -- Problem found with FE maj address :" << std::endl
// 					 << " --- FED = " << fedID_
// 					 << ", iFE = " << iFE << std::endl
// 					 << " --- aBuffer->apveAddress() = " << static_cast<unsigned int>(aBuffer->apveAddress())
// 					 << std::endl
// 					 << " --- debugHeader->feUnitMajorityAddress(iFE) " << static_cast<unsigned int>(debugHeader->feUnitMajorityAddress(iFE))<< std::endl
// 					 << " --- timeLoc(feUnitMajAddr) = "
// 					 << static_cast<unsigned int>(sistrip::FEDAddressConversion::timeLocation(debugHeader->feUnitMajorityAddress(iFE)))<< std::endl
// 					 << " --- timeLoc(apveAddr) = "
// 					 << static_cast<unsigned int>(sistrip::FEDAddressConversion::timeLocation(aBuffer->apveAddress())) << std::endl
// 					 << " --- aBuffer->checkFEUnitAPVAddresses() = "
// 					 << aBuffer->checkFEUnitAPVAddresses()
// 					 << std::endl;


    }

  }

  return !(foundOverflow || foundMissing || foundBadMajority);
}

bool FEDErrors::fillChannelErrors(const sistrip::FEDBuffer* aBuffer,
				  bool & aFullDebug,
				  const unsigned int aPrintDebug,
				  unsigned int & aCounterMonitoring,
				  unsigned int & aCounterUnpacker,
				  const bool aDoMeds,
				  MonitorElement *aMedianHist0,
				  MonitorElement *aMedianHist1
				  )
{
  bool foundError = false;

  const sistrip::FEDFEHeader* header = aBuffer->feHeader();
  const sistrip::FEDFullDebugHeader* debugHeader = dynamic_cast<const sistrip::FEDFullDebugHeader*>(header);

  aFullDebug = debugHeader;

  bool lMedValid = aBuffer->readoutMode() == sistrip::READOUT_MODE_ZERO_SUPPRESSED;

  //this method is not called if there was anyFEDerrors(),
  //so only corruptBuffer+FE check are useful.
  bool lPassedMonitoringFEDcheck = !fedErrors_.CorruptBuffer;

  for (unsigned int iCh = 0; iCh < sistrip::FEDCH_PER_FED; iCh++) {//loop on channels

    bool lFailUnpackerChannelCheck = (!aBuffer->channelGood(iCh, true) && connected_[iCh]) || failUnpackerFEDCheck_;
    bool lFailMonitoringChannelCheck = !lPassedMonitoringFEDcheck && connected_[iCh];


    FEDErrors::ChannelLevelErrors lChErr;
    lChErr.ChannelID = iCh;
    lChErr.Connected = connected_[iCh];
    lChErr.IsActive = false;
    lChErr.Unlocked = false;
    lChErr.OutOfSync = false;

    if (!connected_[iCh]) {
      //to fill histo with unconnected channels
      addBadChannel(lChErr);
      foundError = true;
    }
    else {//if channel connected
      if (!aBuffer->feGood(static_cast<unsigned int>(iCh/sistrip::FEDCH_PER_FEUNIT))) {
	lFailMonitoringChannelCheck = true;
	foundError = true;
      }
      else {//if FE good

	bool apvBad[2] = {false,false};
	sistrip::FEDChannelStatus lStatus = sistrip::CHANNEL_STATUS_NO_PROBLEMS;
	//CHANNEL_STATUS_NO_PROBLEMS
	//CHANNEL_STATUS_LOCKED
	//CHANNEL_STATUS_IN_SYNC
	//CHANNEL_STATUS_APV1_ADDRESS_GOOD
	//CHANNEL_STATUS_APV1_NO_ERROR_BIT
	//CHANNEL_STATUS_APV0_ADDRESS_GOOD
	//CHANNEL_STATUS_APV0_NO_ERROR_BIT

	if (debugHeader) {
	  lStatus = debugHeader->getChannelStatus(iCh);
	  apvBad[0] =
	    !(lStatus & sistrip::CHANNEL_STATUS_LOCKED) ||
	    !(lStatus & sistrip::CHANNEL_STATUS_IN_SYNC) ||
	    !(lStatus & sistrip::CHANNEL_STATUS_APV0_ADDRESS_GOOD) ||
	    !(lStatus & sistrip::CHANNEL_STATUS_APV0_NO_ERROR_BIT);
	  apvBad[1] =
	    !(lStatus & sistrip::CHANNEL_STATUS_LOCKED) ||
	    !(lStatus & sistrip::CHANNEL_STATUS_IN_SYNC) ||
	    !(lStatus & sistrip::CHANNEL_STATUS_APV1_ADDRESS_GOOD) ||
	    !(lStatus & sistrip::CHANNEL_STATUS_APV1_NO_ERROR_BIT);
	  //if (!debugHeader->unlocked(iCh)) {
	  if (lStatus & sistrip::CHANNEL_STATUS_LOCKED) {
	    lChErr.IsActive = true;
	    if (lStatus == sistrip::CHANNEL_STATUS_NO_PROBLEMS) continue;
	    //if (debugHeader->outOfSyncFromBit(iCh)) {
	    if (!(lStatus & sistrip::CHANNEL_STATUS_IN_SYNC)) {
	      lChErr.OutOfSync = true;
	    }
	  }
	  else {
	    lChErr.Unlocked = true;
	  }
	} else {
	  //if (header->checkChannelStatusBits(iCh)) activeChannel = true;
	  apvBad[0] = !header->checkStatusBits(iCh,0);
	  apvBad[1] = !header->checkStatusBits(iCh,1);
	  if (!apvBad[0] && !apvBad[1]) {
	    lChErr.IsActive = true;
	    continue;
	  }
	}

	if (lChErr.Unlocked || lChErr.OutOfSync) addBadChannel(lChErr);

	//std::ostringstream lMode;
	//lMode << aBuffer->readoutMode();

	bool lFirst = true;

	for (unsigned int iAPV = 0; iAPV < 2; iAPV++) {//loop on APVs

	  FEDErrors::APVLevelErrors lAPVErr;
	  lAPVErr.APVID = 2*iCh+iAPV;
	  lAPVErr.ChannelID = iCh;
	  lAPVErr.Connected = connected_[iCh];
	  lAPVErr.IsActive = lChErr.IsActive;
	  lAPVErr.APVStatusBit = false;
	  lAPVErr.APVError = false;
	  lAPVErr.APVAddressError = false;

	  //if (!header->checkStatusBits(iCh,iAPV)){
	  if (apvBad[iAPV]) {
	    lFailMonitoringChannelCheck = true;
	    lAPVErr.APVStatusBit = true;
	    foundError = true;
	  }

	  if (debugHeader && !lChErr.Unlocked && !lChErr.OutOfSync) {
	    //if (debugHeader->apvErrorFromBit(iCh,iAPV)) {
	    if ( (iAPV==0 && !(lStatus & sistrip::CHANNEL_STATUS_APV0_NO_ERROR_BIT)) ||
		 (iAPV==1 && !(lStatus & sistrip::CHANNEL_STATUS_APV1_NO_ERROR_BIT))) {
	      lAPVErr.APVError = true;
	    }
	    //if (debugHeader->apvAddressErrorFromBit(iCh,iAPV)) {
	    if ((iAPV==0 && !(lStatus & sistrip::CHANNEL_STATUS_APV0_ADDRESS_GOOD)) ||
		(iAPV==1 && !(lStatus & sistrip::CHANNEL_STATUS_APV1_ADDRESS_GOOD))) {
	      lAPVErr.APVAddressError = true;
	    }
	  }

	  if ( lAPVErr.APVStatusBit ||
	       lAPVErr.APVError ||
	       lAPVErr.APVAddressError
	       ) addBadAPV(lAPVErr, lFirst);
	}//loop on APVs

      }//if FE good
    }//if connected


    if (lFailUnpackerChannelCheck != lFailMonitoringChannelCheck && connected_[iCh]){
      if (aPrintDebug>1) {
	std::ostringstream debugStream;
	debugStream << "[FEDErrors] ------ WARNING: FED " << fedID_ << ", channel " << iCh
		    << ", isConnected = " << connected_[iCh] << std::endl
		    << "[FEDErrors] --------- Monitoring Channel check " ;
	if (lFailMonitoringChannelCheck) debugStream << "failed." << std::endl;
	else debugStream << "passed." << std::endl ;
	debugStream << "[FEDErrors] --------- Unpacker Channel check " ;
	if (lFailUnpackerChannelCheck) debugStream << "failed." << std::endl;
	else debugStream << "passed." << std::endl;
	debugStream << "[FEDErrors] --------- fegood = "
		    << aBuffer->feGood(static_cast<unsigned int>(iCh/sistrip::FEDCH_PER_FEUNIT))
		    << std::endl
		    << "[FEDErrors] --------- unpacker FED check = " << failUnpackerFEDCheck_ << std::endl;
	edm::LogError("SiStripMonitorHardware") << debugStream.str();
      }

      if (lFailMonitoringChannelCheck) aCounterMonitoring++;
      if (lFailUnpackerChannelCheck) aCounterUnpacker++;
    }

    if (lMedValid && !foundError && lPassedMonitoringFEDcheck && aDoMeds) {
      //get CM values
      const sistrip::FEDChannel & lChannel = aBuffer->channel(iCh);

      HistogramBase::fillHistogram(aMedianHist0,
				   lChannel.cmMedian(0));
      HistogramBase::fillHistogram(aMedianHist1,
				   lChannel.cmMedian(1));

    }

  }//loop on channels

  return !foundError;
}

void FEDErrors::fillBadChannelList(const bool doTkHistoMap,
				   TkHistoMap *aTkMapPointer,
				   MonitorElement *aFedIdVsApvId,
				   unsigned int & aNBadChannels,
				   unsigned int & aNBadActiveChannels,
           unsigned int & aNBadChannels_perFEDID)
{

  uint32_t lPrevId = 0;
  uint16_t nBad = 0;
  uint16_t lPrevTot = 0;
  bool hasBeenProcessed = false;
  bool lFailFED = failMonitoringFEDCheck();

  for (unsigned int iCh = 0;
       iCh < sistrip::FEDCH_PER_FED;
       iCh++) {//loop on channels

    if (!connected_[iCh]) continue;
    if (!detid_[iCh] || detid_[iCh] == sistrip::invalid32_) continue;

    if (lPrevId==0) {
      lPrevId = detid_[iCh];
      lPrevTot = nChInModule_[iCh];
    }

    unsigned int feNumber = static_cast<unsigned int>(iCh/sistrip::FEDCH_PER_FEUNIT);

    bool isBadFE = false;
    bool isMissingFE = false;
    //feErrors vector of FE 0 - 7, each FE has channels 0 -11, 12 .. - ,... - 96
    for (unsigned int badfe(0); badfe<feErrors_.size(); badfe++) {
      if ((feErrors_[badfe]).FeID == feNumber) {
	isBadFE = true;
	if ((feErrors_[badfe]).Missing) isMissingFE = true;
	break;
      }
    }

    bool isBadChan = false;
    bool isActiveChan = false;
    //FED errors, apv
    //for (unsigned int badCh(0); badCh<chErrors_.size(); badCh++) {
    //if (chErrors_[badCh].first == iCh) {
    //if (chErrors_[badCh].second) isActiveChan = true;
    //isBadChan = true;
    //break;
    //}
    //}

    //apvErrors_
    bool isBadApv1 = false;
    bool isBadApv2 = false;
    for (unsigned int badApv(0); badApv<apvErrors_.size(); badApv++) {
      if ((apvErrors_[badApv]).ChannelID == iCh) {
	isBadChan = true;
	if (apvErrors_[badApv].IsActive) isActiveChan = true;
      }
      if (apvErrors_[badApv].APVID == 2 * iCh ) isBadApv1 = true;
      if (apvErrors_[badApv].APVID == 2 * iCh + 1 )
	{
	  isBadApv2 = true;
	  break;
	}
    }


    if (detid_[iCh] == lPrevId){
      if (hasBeenProcessed) hasBeenProcessed = false;
    }
    //fill vector for previous detid
    if (detid_[iCh] != lPrevId){
      processDet(lPrevId,
		 lPrevTot,
		 doTkHistoMap,
		 nBad,
		 aTkMapPointer
		 );
      lPrevId = detid_[iCh];
      lPrevTot = nChInModule_[iCh];
      hasBeenProcessed = true;
    }

    bool lHasErr = lFailFED || isBadFE || isBadChan;
    incrementLumiErrors(lHasErr,subDetId_[feNumber]);

    if ( lHasErr ) {
      nBad++;
      aNBadChannels++;
      aNBadChannels_perFEDID = aNBadChannels_perFEDID+1;
      //define as active channel if channel locked AND not from an unlocked FE.
      if ((isBadChan && isActiveChan) || lFailFED || (isBadFE && !isMissingFE)) aNBadActiveChannels++;
      if ( isBadApv1 || lFailFED || isBadFE ) HistogramBase::fillHistogram ( aFedIdVsApvId , 2 * iCh , fedID_ ) ;
      if ( isBadApv2 || lFailFED || isBadFE ) HistogramBase::fillHistogram ( aFedIdVsApvId , 2 * iCh + 1 , fedID_ ) ;
    }



  }//loop on channels

  if (!hasBeenProcessed){
    processDet(lPrevId,
	       lPrevTot,
	       doTkHistoMap,
	       nBad,
	       aTkMapPointer
	       );
  }


}

void FEDErrors::fillEventProperties(long long dbx) {
  eventProp_.deltaBX = dbx;
}

void FEDErrors::incrementLumiErrors(const bool hasError,
				    const unsigned int aSubDet){
  if (lumiErr_.nTotal.empty()) return;
  if (aSubDet >= lumiErr_.nTotal.size()) {
    edm::LogError("SiStripMonitorHardware") << " -- FED " << fedID_
					    << ", invalid subdetid : " << aSubDet
					    << ", size of lumiErr : "
					    << lumiErr_.nTotal.size()
					    << std::endl;
  }
  else {
    if (hasError) lumiErr_.nErrors[aSubDet]++;
    lumiErr_.nTotal[aSubDet]++;
  }
}

void FEDErrors::processDet(const uint32_t aPrevId,
			   const uint16_t aPrevTot,
			   const bool doTkHistoMap,
			   uint16_t & nBad,
			   TkHistoMap *aTkMapPointer
			   )
{
  if (aPrevTot < nBad){
    edm::LogError("SiStripMonitorHardware") << " -- Number of bad channels in det " << aPrevId
					    << " = " << nBad
					    << ", total number of pairs for this det = " << aPrevTot
					    << std::endl;
  }

  //tkHistoMap takes a uint & as argument
  uint32_t lDetid = aPrevId;
  if (aPrevTot != 0 && doTkHistoMap && aTkMapPointer)
    HistogramBase::fillTkHistoMap(aTkMapPointer,lDetid,static_cast<float>(nBad)/aPrevTot);

  nBad=0;
}


const bool FEDErrors::failMonitoringFEDCheck()
{
  return ( anyFEDErrors() ||
	   fedErrors_.CorruptBuffer
	   );
}

const bool FEDErrors::anyDAQProblems()
{
  return ( fedErrors_.DataMissing ||
	   fedErrors_.InvalidBuffers ||
	   fedErrors_.BadFEDCRCs ||
	   fedErrors_.BadDAQCRCs ||
	   fedErrors_.BadIDs ||
	   fedErrors_.BadDAQPacket
	   );
}

const bool FEDErrors::anyFEDErrors()
{
  return ( fedErrors_.InvalidBuffers ||
	   fedErrors_.BadFEDCRCs ||
	   fedErrors_.BadDAQCRCs ||
	   fedErrors_.BadIDs ||
	   fedErrors_.BadDAQPacket ||
	   fedErrors_.FEsOverflow
	   );
}

const bool FEDErrors::anyFEProblems()
{
  return ( fedErrors_.FEsOverflow ||
	   fedErrors_.FEsMissing ||
	   fedErrors_.FEsBadMajorityAddress
	   );
}

const bool FEDErrors::printDebug()
{
  return ( anyFEDErrors()  ||
	   anyFEProblems() ||
	   fedErrors_.CorruptBuffer ||
	   fedErrors_.BadChannelStatusBit
	   );

}

const unsigned int FEDErrors::fedID(){
  return fedID_;
}


FEDErrors::FEDCounters & FEDErrors::getFEDErrorsCounters()
{
  return lFedCounter_;
}

FEDErrors::ChannelCounters & FEDErrors::getChannelErrorsCounters()
{
  return lChCounter_;
}

FEDErrors::FECounters & FEDErrors::getFEErrorsCounters()
{
  return feCounter_;
}

FEDErrors::FEDLevelErrors & FEDErrors::getFEDLevelErrors()
{
  return fedErrors_;
}

FEDErrors::EventProperties & FEDErrors::getEventProperties()
{
  return eventProp_;
}

std::vector<FEDErrors::FELevelErrors> & FEDErrors::getFELevelErrors()
{
  return feErrors_;
}

std::vector<FEDErrors::ChannelLevelErrors> & FEDErrors::getChannelLevelErrors()
{
  return chErrorsDetailed_;
}

std::vector<FEDErrors::APVLevelErrors> & FEDErrors::getAPVLevelErrors()
{
  return apvErrors_;
}

std::vector<std::pair<unsigned int,bool> > & FEDErrors::getBadChannels()
{
  return chErrors_;
}

const FEDErrors::LumiErrors & FEDErrors::getLumiErrors(){
  return lumiErr_;
}

void FEDErrors::addBadFE(const FEDErrors::FELevelErrors & aFE)
{
  if (aFE.Overflow)  {
    fedErrors_.FEsOverflow = true;
    (feCounter_.nFEOverflows)++;
  }
  else if (aFE.Missing)  {
    fedErrors_.FEsMissing = true;
    (feCounter_.nFEMissing)++;
    feErrors_.push_back(aFE);
  }
  else if (aFE.BadMajorityAddress) {
    fedErrors_.FEsBadMajorityAddress = true;
    (feCounter_.nFEBadMajorityAddresses)++;
    feErrors_.push_back(aFE);
  }
  else if (aFE.TimeDifference != 0) {
    feErrors_.push_back(aFE);
  }
}

void FEDErrors::addBadChannel(const FEDErrors::ChannelLevelErrors & aChannel)
{
  if (aChannel.Connected) chErrorsDetailed_.push_back(aChannel);
  incrementChannelCounters(aChannel);
}

void FEDErrors::addBadAPV(const FEDErrors::APVLevelErrors & aAPV, bool & aFirst)
{
  apvErrors_.push_back(aAPV);
  incrementAPVCounters(aAPV);
  if (aAPV.APVStatusBit && aFirst) {
    fedErrors_.BadChannelStatusBit = true;
    lFedCounter_.nBadChannels++;
    chErrors_.push_back(std::pair<unsigned int, bool>(aAPV.ChannelID,aAPV.IsActive));
    if (aAPV.IsActive) {
      //print(aAPV);
      fedErrors_.BadActiveChannelStatusBit = true;
      lFedCounter_.nBadActiveChannels++;
      //std::cout << "------ nBadActiveChannels = " << FEDErrors::getFEDErrorsCounters().nBadActiveChannels << std::endl;
    }
    aFirst = false;
  }
}


void FEDErrors::incrementFEDCounters()
{
  if (fedErrors_.InvalidBuffers ||
      fedErrors_.BadFEDCRCs     ||
      fedErrors_.BadDAQCRCs     ||
      fedErrors_.BadIDs         ||
      fedErrors_.BadDAQPacket
      ) {
    lFedCounter_.nDAQProblems++;
    lFedCounter_.nFEDErrors++;
  }

  //FElevel errors
  if (fedErrors_.FEsOverflow){
    lFedCounter_.nFEDsWithFEOverflows++;
  }
  else if (fedErrors_.FEsMissing){
    lFedCounter_.nFEDsWithMissingFEs++;
  }
  else if (fedErrors_.FEsBadMajorityAddress){
    lFedCounter_.nFEDsWithFEBadMajorityAddresses++;
  }

  if (fedErrors_.FEsOverflow ||
      fedErrors_.FEsBadMajorityAddress ||
      fedErrors_.FEsMissing
      ){
    lFedCounter_.nFEDsWithFEProblems++;
    lFedCounter_.nFEDErrors++;
  }
  else if (fedErrors_.CorruptBuffer) {
    lFedCounter_.nCorruptBuffers++;
    lFedCounter_.nFEDErrors++;
  }


}


void FEDErrors::incrementChannelCounters(const FEDErrors::ChannelLevelErrors & aChannel)
{
  if (aChannel.Unlocked && aChannel.Connected) lChCounter_.nUnlocked++;
  if (aChannel.OutOfSync && aChannel.Connected) lChCounter_.nOutOfSync++;
  if (!aChannel.Connected) lChCounter_.nNotConnected++;
}

void FEDErrors::incrementAPVCounters(const FEDErrors::APVLevelErrors & aAPV)
{
  if (aAPV.Connected && aAPV.IsActive){
    if (aAPV.APVStatusBit) lChCounter_.nAPVStatusBit++;
    if (aAPV.APVAddressError) lChCounter_.nAPVAddressError++;
    if (aAPV.APVError) lChCounter_.nAPVError++;
  }
}


bool FEDErrors::ChannelLevelErrors::operator <(const FEDErrors::ChannelLevelErrors & aErr) const{
  if (this->ChannelID < aErr.ChannelID) return true;
  return false;
}



bool FEDErrors::APVLevelErrors::operator <(const FEDErrors::APVLevelErrors & aErr) const{
  if (this->ChannelID < aErr.ChannelID) return true;
  return false;
}


void FEDErrors::print(const FEDErrors::FEDCounters & aFEDCounter, std::ostream & aOs)
{

  aOs << std::endl;
  aOs << "[FEDErrors]============================================" << std::endl
      << "[FEDErrors]==== Printing FEDCounters information : ====" << std::endl
      << "[FEDErrors]============================================" << std::endl
      << "[FEDErrors]======== nFEDErrors = " << aFEDCounter.nFEDErrors << std::endl
      << "[FEDErrors]======== nDAQProblems = " << aFEDCounter.nDAQProblems << std::endl
      << "[FEDErrors]======== nFEDsWithFEProblems = " << aFEDCounter.nFEDsWithFEProblems << std::endl
      << "[FEDErrors]======== nCorruptBuffers = " << aFEDCounter.nCorruptBuffers << std::endl
      << "[FEDErrors]======== nBadChannels = " << aFEDCounter.nBadChannels << std::endl
      << "[FEDErrors]======== nBadActiveChannels = " << aFEDCounter.nBadActiveChannels << std::endl
      << "[FEDErrors]======== nFEDsWithFEOverflows = " << aFEDCounter.nFEDsWithFEOverflows << std::endl
      << "[FEDErrors]======== nFEDsWithFEBadMajorityAddresses = " << aFEDCounter.nFEDsWithFEBadMajorityAddresses << std::endl
      << "[FEDErrors]======== nFEDsWithMissingFEs = " << aFEDCounter.nFEDsWithMissingFEs << std::endl
      << "[FEDErrors]======== nTotalBadChannels = " << aFEDCounter.nTotalBadChannels << std::endl
      << "[FEDErrors]======== nTotalBadActiveChannels = " << aFEDCounter.nTotalBadActiveChannels << std::endl
      << "[FEDErrors]============================================" << std::endl;


}

void FEDErrors::print(const FEDErrors::FECounters & aFECounter, std::ostream & aOs)
{

  aOs << std::endl;
  aOs << "[FEDErrors]============================================" << std::endl
      << "[FEDErrors]==== Printing FECounters information :  ====" << std::endl
      << "[FEDErrors]============================================" << std::endl
      << "[FEDErrors]======== nFEOverflows = " << aFECounter.nFEOverflows << std::endl
      << "[FEDErrors]======== nFEBadMajorityAddresses = " << aFECounter.nFEBadMajorityAddresses << std::endl
      << "[FEDErrors]======== nFEMissing = " << aFECounter.nFEMissing << std::endl
      << "[FEDErrors]============================================" << std::endl;


}

void FEDErrors::print(const FEDErrors::FEDLevelErrors & aFEDErr, std::ostream & aOs)
{

  aOs << std::endl;
  aOs << "[FEDErrors]============================================" << std::endl
      << "[FEDErrors]==== Printing FED errors information :  ====" << std::endl
      << "[FEDErrors]============================================" << std::endl
      << "[FEDErrors]======== HasCabledChannels = " << aFEDErr.HasCabledChannels << std::endl
      << "[FEDErrors]======== DataPresent = " << aFEDErr.DataPresent << std::endl
      << "[FEDErrors]======== DataMissing = " << aFEDErr.DataMissing << std::endl
      << "[FEDErrors]======== InvalidBuffers = " << aFEDErr.InvalidBuffers << std::endl
      << "[FEDErrors]======== BadFEDCRCs = " << aFEDErr.BadFEDCRCs << std::endl
      << "[FEDErrors]======== BadDAQCRCs = " << aFEDErr.BadDAQCRCs << std::endl
      << "[FEDErrors]======== BadIDs = " << aFEDErr.BadIDs << std::endl
      << "[FEDErrors]======== BadDAQPacket = " << aFEDErr.BadDAQPacket << std::endl
      << "[FEDErrors]======== CorruptBuffer = " << aFEDErr.CorruptBuffer << std::endl
      << "[FEDErrors]======== FEOverflows = " << aFEDErr.FEsOverflow << std::endl
      << "[FEDErrors]======== FEMissing = " << aFEDErr.FEsMissing << std::endl
      << "[FEDErrors]======== BadMajorityAddresses = " << aFEDErr.FEsBadMajorityAddress << std::endl
       << "[FEDErrors]============================================" << std::endl;

}

void FEDErrors::print(const FEDErrors::FELevelErrors & aErr, std::ostream & aOs)
{

  aOs << std::endl;
  aOs << "[FEDErrors]============================================" << std::endl
      << "[FEDErrors]==== Printing FE errors information :   ====" << std::endl
      << "[FEDErrors]============================================" << std::endl
      << "[FEDErrors]======== FE #" << aErr.FeID << std::endl
      << "[FEDErrors]======== subdet " << aErr.SubDetID << std::endl
      << "[FEDErrors]======== FEOverflow = " << aErr.Overflow << std::endl
      << "[FEDErrors]======== FEMissing = " << aErr.Missing << std::endl
      << "[FEDErrors]======== BadMajorityAddresses = " << aErr.BadMajorityAddress << std::endl
      << "[FEDErrors]======== TimeDifference = " << aErr.TimeDifference << std::endl
      << "[FEDErrors]============================================" << std::endl;

}

void FEDErrors::print(const FEDErrors::ChannelLevelErrors & aErr, std::ostream & aOs)
{
  aOs << std::endl;
  aOs << "[FEDErrors]=================================================" << std::endl
      << "[FEDErrors]==== Printing channel errors information :   ====" << std::endl
      << "[FEDErrors]=================================================" << std::endl
      << "[FEDErrors]============ Channel #" << aErr.ChannelID  << std::endl
      << "[FEDErrors]============ connected  = " << aErr.Connected << std::endl
      << "[FEDErrors]============ isActive  = " << aErr.IsActive << std::endl
      << "[FEDErrors]============ Unlocked = " << aErr.Unlocked << std::endl
      << "[FEDErrors]============ OutOfSync = " << aErr.OutOfSync << std::endl
      << "[FEDErrors]=================================================" << std::endl;
}


void FEDErrors::print(const FEDErrors::APVLevelErrors & aErr, std::ostream & aOs)
{
  aOs << std::endl;
  aOs << "[FEDErrors]=================================================" << std::endl
      << "[FEDErrors]==== Printing APV errors information :       ====" << std::endl
      << "[FEDErrors]=================================================" << std::endl
      << "[FEDErrors]============ APV #" << aErr.APVID  << std::endl
      << "[FEDErrors]============ Channel #" << aErr.ChannelID  << std::endl
      << "[FEDErrors]============ connected  = " << aErr.Connected << std::endl
      << "[FEDErrors]============ isActive  = " << aErr.IsActive << std::endl
      << "[FEDErrors]============ APVStatusBit = " << aErr.APVStatusBit << std::endl
      << "[FEDErrors]============ APVError = " << aErr.APVError << std::endl
      << "[FEDErrors]============ APVAddressError = " << aErr.APVAddressError << std::endl
      << "[FEDErrors]=================================================" << std::endl;
}
