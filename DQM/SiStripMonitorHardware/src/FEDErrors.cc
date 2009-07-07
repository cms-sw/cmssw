#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "DQM/SiStripMonitorHardware/interface/FEDErrors.hh"


FEDErrors::FEDErrors()
{
  fedID_ = 0;

  for (unsigned int iCh = 0; 
       iCh < sistrip::FEDCH_PER_FED; 
       iCh++) {
    connected_[iCh] = false;
  }

  FEDCounters & lFedCounter = FEDErrors::getFEDErrorsCounters();
  lFedCounter.nFEDErrors = 0;
  lFedCounter.nDAQProblems = 0;
  lFedCounter.nFEDsWithFEProblems = 0;
  lFedCounter.nCorruptBuffers = 0;
  lFedCounter.nBadChannels = 0;
  lFedCounter.nBadActiveChannels = 0;
  lFedCounter.nFEDsWithFEOverflows = 0;
  lFedCounter.nFEDsWithFEBadMajorityAddresses = 0;
  lFedCounter.nFEDsWithMissingFEs = 0;
  lFedCounter.nTotalBadChannels = 0;
  lFedCounter.nTotalBadActiveChannels = 0;

  ChannelCounters & lChCounter = FEDErrors::getChannelErrorsCounters();
  lChCounter.nUnlocked = 0;
  lChCounter.nOutOfSync = 0;
  lChCounter.nAPVStatusBit = 0;
  lChCounter.nAPVError = 0;
  lChCounter.nAPVAddressError = 0;

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

FEDErrors::~FEDErrors()
{

}

void FEDErrors::initialise(const unsigned int aFedID,
			   const SiStripFedCabling* aCabling)
{
  fedID_ = aFedID;

  for (unsigned int iCh = 0; 
       iCh < sistrip::FEDCH_PER_FED; 
       iCh++) {
    connected_[iCh] = (aCabling->connection(fedID_,iCh)).isConnected();
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

bool FEDErrors::failUnpackerFEDCheck(const FEDRawData & fedData)
{
  // construct FEDBuffer
  bool lFail = false;
  std::auto_ptr<sistrip::FEDBuffer> buffer;
  try {
    buffer.reset(new sistrip::FEDBuffer(fedData.data(),fedData.size()));
    if (!buffer->doChecks()) lFail = true;
    //throw cms::Exception("FEDBuffer") << "FED Buffer check fails.";
  }
  catch (const cms::Exception& e) { 
    lFail = true;
  }
  failUnpackerFEDCheck_ = lFail;
  return lFail;
}

bool FEDErrors::fillFEDErrors(const FEDRawData& aFedData, 
			      bool & aFullDebug,
			      const unsigned int aPrintDebug
			      )
{
  //try to construct the basic buffer object (do not check payload)
  //if this fails then count it as an invalid buffer and stop checks since we can't understand things like buffer ordering

  std::auto_ptr<const sistrip::FEDBufferBase> bufferBase;
  try {
    bufferBase.reset(new sistrip::FEDBufferBase(aFedData.data(),aFedData.size()));
  } catch (const cms::Exception& e) {
    fedErrors_.InvalidBuffers = true;
    //don't check anything else if the buffer is invalid
    return false;
  }
  //CRC checks
  //if CRC fails then don't continue as if the buffer has been corrupted in DAQ then anything else could be invalid
  if (!bufferBase->checkNoSlinkCRCError()) {
    fedErrors_.BadFEDCRCs = true;
    return false;
  } else if (!bufferBase->checkCRC()) {
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
    fedErrors_.BadDAQPacket = true;
    return false;
  }

  //now do checks on header
  //check that tracker special header is consistent
  if ( !(bufferBase->checkBufferFormat() && bufferBase->checkHeaderType() && bufferBase->checkReadoutMode() && bufferBase->checkAPVEAddressValid()) ) {
    fedErrors_.InvalidBuffers = true;
  }
  //FE unit overflows
  if (!bufferBase->checkNoFEOverflows()) { 
    fedErrors_.FEsOverflow = true;
  }
  
  //need to construct full object to go any further
  std::auto_ptr<const sistrip::FEDBuffer> buffer;
  buffer.reset(new sistrip::FEDBuffer(aFedData.data(),aFedData.size(),true));

  //std::ostringstream lMode;
  //lMode << buffer->readoutMode();
  
  //payload checks
  if (!this->anyFEDErrors()) {
    //corrupt buffer checks
    if (!buffer->doCorruptBufferChecks()) {
      fedErrors_.CorruptBuffer = true;
    }

    //corruptBuffer concerns the payload: header info should still be reliable...
    //so analyze FE and channels to fill histograms.

    //fe check... 
    fillFEErrors(buffer.get());
    
    //channel checks
    fillChannelErrors(buffer.get(),
		      aFullDebug,
		      aPrintDebug
		      );

  }

   
  if (printDebug() && aPrintDebug>2) {
    const sistrip::FEDBufferBase* debugBuffer = NULL;

    if (buffer.get()) debugBuffer = buffer.get();
    else if (bufferBase.get()) debugBuffer = bufferBase.get();
    if (debugBuffer) {
      std::vector<FEDErrors::APVLevelErrors> & lChVec = getAPVLevelErrors();
      std::ostringstream debugStream;
      if (lChVec.size()) {
	std::sort(lChVec.begin(),lChVec.end());
        debugStream << "[FEDErrors] Cabled channels which had errors: ";
	
        for (unsigned int iBadCh(0); iBadCh < lChVec.size(); iBadCh++) {
          print(lChVec.at(iBadCh),debugStream);
        }
        debugStream << std::endl;
        debugStream << "[FEDErrors] Active (have been unlocked in at least one event) cabled channels which had errors: ";
	for (unsigned int iBadCh(0); iBadCh < lChVec.size(); iBadCh++) {
          if ((lChVec.at(iBadCh)).IsActive) print(lChVec.at(iBadCh),debugStream);
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

bool FEDErrors::fillFEErrors(const sistrip::FEDBuffer* aBuffer)
{
  bool foundOverflow = false;
  bool foundBadMajority = false;
  bool foundMissing = false;
  for (unsigned int iFE = 0; iFE < sistrip::FEUNITS_PER_FED; iFE++) {
    
    FEDErrors::FELevelErrors lFeErr;
    lFeErr.FeID = iFE;
    lFeErr.Overflow = false;
    lFeErr.Missing = false;
    lFeErr.BadMajorityAddress = false;

    if (aBuffer->feOverflow(iFE)) {
      lFeErr.Overflow = true;
      foundOverflow = true;
      addBadFE(lFeErr);
      //if FE overflowed then address isn't valid
      continue;
    }
    if (!aBuffer->feEnabled(iFE)) continue;
    //check for cabled channels
    bool hasCabledChannels = false;
    for (unsigned int feUnitCh = 0; feUnitCh < sistrip::FEDCH_PER_FEUNIT; feUnitCh++) {
      if (connected_[iFE*sistrip::FEDCH_PER_FEUNIT+feUnitCh]) {
        hasCabledChannels = true;
        break;
      }
    }
    //check for missing data
    if (!aBuffer->fePresent(iFE)) {
      if (hasCabledChannels) {
	lFeErr.Missing = true;
        foundMissing = true;
	addBadFE(lFeErr);
      }
      continue;
    }
    if (aBuffer->majorityAddressErrorForFEUnit(iFE)) {
      foundBadMajority = true;
      addBadFE(lFeErr);
    }
  }

  return !(foundOverflow || foundMissing || foundBadMajority);
}

bool FEDErrors::fillChannelErrors(const sistrip::FEDBuffer* aBuffer, 
				  bool & aFullDebug,
				  const unsigned int aPrintDebug
				  )
{
  bool foundError = false;

  const sistrip::FEDFEHeader* header = aBuffer->feHeader();
  const sistrip::FEDFullDebugHeader* debugHeader = dynamic_cast<const sistrip::FEDFullDebugHeader*>(header);

  aFullDebug = debugHeader;

  //this method is not called if there was anyFEDerrors(), so only corruptBuffer check is useful.
  bool lPassedMonitoringFEDcheck = !fedErrors_.CorruptBuffer;
  
  for (unsigned int iCh = 0; iCh < sistrip::FEDCH_PER_FED; iCh++) {//loop on channels
    bool isGood = true;

    bool lFailUnpackerChannelCheck = (!aBuffer->channelGood(iCh) && connected_[iCh]) || failUnpackerFEDCheck_;
    bool lFailMonitoringChannelCheck = !lPassedMonitoringFEDcheck;

    if (!connected_[iCh]) isGood = false;
    if (!aBuffer->feGood(static_cast<unsigned int>(iCh/sistrip::FEDCH_PER_FEUNIT))) {
      isGood = false;
      //if (lPassedMonitoringFEDcheck)
      if (connected_[iCh]) lFailMonitoringChannelCheck = true;
    }

    bool activeChannel = false;

    if (debugHeader) {
      if (!debugHeader->unlocked(iCh)) activeChannel = true;
    } else {
      if (header->checkChannelStatusBits(iCh)) activeChannel = true;
    }

    FEDErrors::ChannelLevelErrors lChErr;
    lChErr.ChannelID = iCh;
    lChErr.Connected = connected_[iCh];
    lChErr.IsActive = activeChannel;
    lChErr.Unlocked = false;
    lChErr.OutOfSync = false;


    bool lFirst = true;

    for (unsigned int iAPV = 0; iAPV < 2; iAPV++) {//loop on APVs

      FEDErrors::APVLevelErrors lAPVErr;
      lAPVErr.APVID = 2*iCh+iAPV;
      lAPVErr.ChannelID = iCh;
      lAPVErr.Connected = connected_[iCh];
      lAPVErr.IsActive = activeChannel;
      lAPVErr.APVStatusBit = false;
      lAPVErr.APVError = false;
      lAPVErr.APVAddressError = false;

      if (!header->checkStatusBits(iCh,iAPV) && isGood) {
	//if (lPassedMonitoringFEDcheck) 
	lFailMonitoringChannelCheck = true;
	lAPVErr.APVStatusBit = true;
	foundError = true;
      }

      if (debugHeader) {
	if (debugHeader->apvError(iCh,iAPV)) {
	  lAPVErr.APVError = true;
	}
        if (debugHeader->apvAddressError(iCh,iAPV)) {
          lAPVErr.APVAddressError = true;
        }
      }

      if ( (lAPVErr.APVStatusBit && isGood) || 
	   lAPVErr.APVError || 
	   lAPVErr.APVAddressError
	   ) addBadAPV(lAPVErr, lFirst);
    }//loop on APVs

    if (debugHeader) {
      if (debugHeader->unlocked(iCh)) {
	lChErr.Unlocked = true;
      }
      if (debugHeader->outOfSync(iCh)) {
	lChErr.OutOfSync = true;
      }
      if (lChErr.Unlocked || lChErr.OutOfSync) addBadChannel(lChErr);
    }


    if (lFailUnpackerChannelCheck != lFailMonitoringChannelCheck && aPrintDebug) {
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

  }//loop on channels

  return !foundError;
}

void FEDErrors::fillBadChannelList(std::map<unsigned int,std::pair<unsigned short,unsigned short> > & aMap,
				   const SiStripFedCabling* aCabling,
				   unsigned int & aNBadChannels,
				   unsigned int & aNBadActiveChannels,
				   const bool aFillAll)
{

  std::pair<std::map<unsigned int,std::pair<unsigned short,unsigned short> >::iterator,bool> alreadyThere;

  //unsigned int nBadChans = 0;
  for (unsigned int iCh = 0; 
       iCh < sistrip::FEDCH_PER_FED; 
       iCh++) {//loop on channels
    const FedChannelConnection & lConnection = aCabling->connection(fedID_,iCh);
    if (!lConnection.isConnected()) continue;
    
    unsigned int feNumber = static_cast<unsigned int>(iCh/sistrip::FEDCH_PER_FEUNIT);
	
    bool isBadFE = false;
    for (unsigned int badfe(0); badfe<feErrors_.size(); badfe++) {
      if ((feErrors_.at(badfe)).FeID == feNumber) {
	isBadFE = true;
	break;
      }
    }

    bool isBadChan = false;
    bool isActiveChan = false;
    for (unsigned int badCh(0); badCh<chErrors_.size(); badCh++) {
      if (chErrors_.at(badCh).first == iCh) {
	if (chErrors_.at(badCh).second) isActiveChan = true;
	isBadChan = true;
	break;
      }
    }

    unsigned int detid = lConnection.detId();
    if (!detid || detid == sistrip::invalid32_) continue;
    unsigned short nChInModule = lConnection.nApvPairs();

    if (failMonitoringFEDCheck() || isBadFE || isBadChan) {
      alreadyThere = aMap.insert(std::pair<unsigned int,std::pair<unsigned short,unsigned short> >(detid,std::pair<unsigned short,unsigned short>(nChInModule,1)));
      if (!alreadyThere.second) ((alreadyThere.first)->second).second += 1;
      //nBadChans++;
      aNBadChannels++;
      if ((isBadChan && isActiveChan) || failMonitoringFEDCheck() || isBadFE) aNBadActiveChannels++;
    }
    else {
      if (aFillAll) alreadyThere = aMap.insert(std::pair<unsigned int,std::pair<unsigned short,unsigned short> >(detid,std::pair<unsigned short,unsigned short>(nChInModule,0)));
    }

  }//loop on channels


  //if (nBadChans>0) std::cout << "-------- FED " << fedId << ", " << nBadChans << " bad channels." << std::endl;

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
	   fedErrors_.BadActiveChannelStatusBit
	   );

}

const unsigned int FEDErrors::fedID(){
  return fedID_;
}


FEDErrors::FEDCounters & FEDErrors::getFEDErrorsCounters()
{
  static FEDCounters lFedCounter;
  return lFedCounter;
}

FEDErrors::ChannelCounters & FEDErrors::getChannelErrorsCounters()
{
  static ChannelCounters lChCounter;
  return lChCounter;
}

FEDErrors::FECounters & FEDErrors::getFEErrorsCounters()
{
  return feCounter_;
}

FEDErrors::FEDLevelErrors & FEDErrors::getFEDLevelErrors()
{
  return fedErrors_;
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
}

void FEDErrors::addBadChannel(const FEDErrors::ChannelLevelErrors & aChannel)
{
  chErrorsDetailed_.push_back(aChannel);
  incrementChannelCounters(aChannel);
}

void FEDErrors::addBadAPV(const FEDErrors::APVLevelErrors & aAPV, bool & aFirst)
{
  apvErrors_.push_back(aAPV);
  incrementAPVCounters(aAPV);
  if (aAPV.APVStatusBit && aFirst) {
    fedErrors_.BadChannelStatusBit = true;
    (FEDErrors::getFEDErrorsCounters().nBadChannels)++;
    chErrors_.push_back(std::pair<unsigned int, bool>(aAPV.ChannelID,aAPV.IsActive));
    if (aAPV.IsActive) {
      //print(aAPV);
      fedErrors_.BadActiveChannelStatusBit = true;
      (FEDErrors::getFEDErrorsCounters().nBadActiveChannels)++;
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
    (FEDErrors::getFEDErrorsCounters().nDAQProblems)++;
    (FEDErrors::getFEDErrorsCounters().nFEDErrors)++;
  }

  //FElevel errors
  if (fedErrors_.FEsOverflow){
    (FEDErrors::getFEDErrorsCounters().nFEDsWithFEOverflows)++;
  }
  else if (fedErrors_.FEsBadMajorityAddress){
    (FEDErrors::getFEDErrorsCounters().nFEDsWithFEBadMajorityAddresses)++;
  }
  else if (fedErrors_.FEsMissing){
    (FEDErrors::getFEDErrorsCounters().nFEDsWithMissingFEs)++;
  }

  if (fedErrors_.FEsOverflow ||
      fedErrors_.FEsBadMajorityAddress ||
      fedErrors_.FEsMissing
      ){
    (FEDErrors::getFEDErrorsCounters().nFEDsWithFEProblems)++;
    (FEDErrors::getFEDErrorsCounters().nFEDErrors)++;
  }
  else if (fedErrors_.CorruptBuffer) {
    (FEDErrors::getFEDErrorsCounters().nCorruptBuffers)++;
    (FEDErrors::getFEDErrorsCounters().nFEDErrors)++;
  }


}


void FEDErrors::incrementChannelCounters(const FEDErrors::ChannelLevelErrors & aChannel)
{
  if (aChannel.Unlocked && aChannel.Connected) (FEDErrors::getChannelErrorsCounters().nUnlocked)++; 
  if (aChannel.OutOfSync && aChannel.Connected) (FEDErrors::getChannelErrorsCounters().nOutOfSync)++; 
}

void FEDErrors::incrementAPVCounters(const FEDErrors::APVLevelErrors & aAPV)
{
  if (aAPV.Connected && aAPV.IsActive){
    if (aAPV.APVStatusBit) (FEDErrors::getChannelErrorsCounters().nAPVStatusBit)++; 
    if (aAPV.APVAddressError) (FEDErrors::getChannelErrorsCounters().nAPVAddressError)++; 
    if (aAPV.APVError) (FEDErrors::getChannelErrorsCounters().nAPVError)++; 
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
      << "[FEDErrors]======== FEOverflow = " << aErr.Overflow << std::endl
      << "[FEDErrors]======== FEMissing = " << aErr.Missing << std::endl
      << "[FEDErrors]======== BadMajorityAddresses = " << aErr.BadMajorityAddress << std::endl
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
      << "[FEDErrors]============ statusBit = " << aErr.Unlocked << std::endl
      << "[FEDErrors]============ statusBit = " << aErr.OutOfSync << std::endl
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
      << "[FEDErrors]============ statusBit = " << aErr.APVStatusBit << std::endl
      << "[FEDErrors]============ statusBit = " << aErr.APVError << std::endl
      << "[FEDErrors]============ statusBit = " << aErr.APVAddressError << std::endl
      << "[FEDErrors]=================================================" << std::endl;
}


