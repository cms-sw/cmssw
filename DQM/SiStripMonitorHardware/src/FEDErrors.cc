#include "DQM/SiStripMonitorHardware/interface/FEDErrors.hh"


FEDErrors::FEDErrors()
{
  fedID_ = 0;

  FEDCounters & lFedCounter = FEDErrors::getFEDErrorsCounters();
  lFedCounter.nFEDErrors = 0;
  lFedCounter.nDAQProblems = 0;
  lFedCounter.nFEDsWithFEProblems = 0;
  lFedCounter.nCorruptBuffers = 0;
  lFedCounter.nBadActiveChannels = 0;
  lFedCounter.nFEDsWithFEOverflows = 0;
  lFedCounter.nFEDsWithFEBadMajorityAddresses = 0;
  lFedCounter.nFEDsWithMissingFEs = 0;

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

void FEDErrors::initialise(const unsigned int aFedID)
{
  fedID_ = aFedID;
 
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

void FEDErrors::hasCabledChannels(const bool isCabled)
{
  fedErrors_.HasCabledChannels = isCabled;
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
}

void FEDErrors::addBadAPV(const FEDErrors::APVLevelErrors & aAPV, bool & aFirst)
{
  apvErrors_.push_back(aAPV);
  if (aAPV.APVStatusBit && aFirst) {
    fedErrors_.BadChannelStatusBit = true;
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
  aOs << "============================================" << std::endl
      << "==== Printing FEDCounters information : ====" << std::endl
      << "============================================" << std::endl
      << "======== nFEDErrors = " << aFEDCounter.nFEDErrors << std::endl
      << "======== nDAQProblems = " << aFEDCounter.nDAQProblems << std::endl
      << "======== nFEDsWithFEProblems = " << aFEDCounter.nFEDsWithFEProblems << std::endl
      << "======== nCorruptBuffers = " << aFEDCounter.nCorruptBuffers << std::endl
      << "======== nBadActiveChannels = " << aFEDCounter.nBadActiveChannels << std::endl
      << "======== nFEDsWithFEOverflows = " << aFEDCounter.nFEDsWithFEOverflows << std::endl
      << "======== nFEDsWithFEBadMajorityAddresses = " << aFEDCounter.nFEDsWithFEBadMajorityAddresses << std::endl
      << "======== nFEDsWithMissingFEs = " << aFEDCounter.nFEDsWithMissingFEs << std::endl
      << "============================================" << std::endl;
    

}
  
void FEDErrors::print(const FEDErrors::FECounters & aFECounter, std::ostream & aOs)
{

  aOs << std::endl;
  aOs << "============================================" << std::endl
      << "==== Printing FECounters information :  ====" << std::endl
      << "============================================" << std::endl
      << "======== nFEOverflows = " << aFECounter.nFEOverflows << std::endl
      << "======== nFEBadMajorityAddresses = " << aFECounter.nFEBadMajorityAddresses << std::endl
      << "======== nFEMissing = " << aFECounter.nFEMissing << std::endl
      << "============================================" << std::endl;
    

}

void FEDErrors::print(const FEDErrors::FEDLevelErrors & aFEDErr, std::ostream & aOs)
{

  aOs << std::endl;
  aOs << "============================================" << std::endl
      << "==== Printing FED errors information :  ====" << std::endl
      << "============================================" << std::endl
      << "======== HasCabledChannels = " << aFEDErr.HasCabledChannels << std::endl
      << "======== DataPresent = " << aFEDErr.DataPresent << std::endl
      << "======== DataMissing = " << aFEDErr.DataMissing << std::endl
      << "======== InvalidBuffers = " << aFEDErr.InvalidBuffers << std::endl
      << "======== BadFEDCRCs = " << aFEDErr.BadFEDCRCs << std::endl
      << "======== BadDAQCRCs = " << aFEDErr.BadDAQCRCs << std::endl
      << "======== BadIDs = " << aFEDErr.BadIDs << std::endl
      << "======== BadDAQPacket = " << aFEDErr.BadDAQPacket << std::endl
      << "======== CorruptBuffer = " << aFEDErr.CorruptBuffer << std::endl
      << "======== FEOverflows = " << aFEDErr.FEsOverflow << std::endl
      << "======== FEMissing = " << aFEDErr.FEsMissing << std::endl
      << "======== BadMajorityAddresses = " << aFEDErr.FEsBadMajorityAddress << std::endl
      << "============================================" << std::endl;

}

void FEDErrors::print(const FEDErrors::FELevelErrors & aErr, std::ostream & aOs)
{

  aOs << std::endl;
  aOs << "============================================" << std::endl
      << "==== Printing FE errors information :   ====" << std::endl
      << "============================================" << std::endl
      << "======== FE #" << aErr.FeID << std::endl
      << "======== FEOverflow = " << aErr.Overflow << std::endl
      << "======== FEMissing = " << aErr.Missing << std::endl
      << "======== BadMajorityAddresses = " << aErr.BadMajorityAddress << std::endl
      << "============================================" << std::endl;

}

void FEDErrors::print(const FEDErrors::ChannelLevelErrors & aErr, std::ostream & aOs)
{
  aOs << std::endl;
  aOs << "=================================================" << std::endl
      << "==== Printing channel errors information :   ====" << std::endl
      << "=================================================" << std::endl
      << "============ Channel #" << aErr.ChannelID  << std::endl
      << "============ isActive  = " << aErr.IsActive << std::endl
      << "============ statusBit = " << aErr.Unlocked << std::endl
      << "============ statusBit = " << aErr.OutOfSync << std::endl
      << "=================================================" << std::endl;
}


void FEDErrors::print(const FEDErrors::APVLevelErrors & aErr, std::ostream & aOs)
{
  aOs << std::endl;
  aOs << "=================================================" << std::endl
      << "==== Printing APV errors information :       ====" << std::endl
      << "=================================================" << std::endl
      << "============ APV #" << aErr.APVID  << std::endl
      << "============ Channel #" << aErr.ChannelID  << std::endl
      << "============ isActive  = " << aErr.IsActive << std::endl
      << "============ statusBit = " << aErr.APVStatusBit << std::endl
      << "============ statusBit = " << aErr.APVError << std::endl
      << "============ statusBit = " << aErr.APVAddressError << std::endl
      << "=================================================" << std::endl;
}


