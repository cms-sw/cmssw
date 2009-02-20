#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


void HcalSeverityLevelComputer::getChStBit(HcalSeverityDefinition& mydef, 
					   const std::string& mybit)
{
  if (mybit == "HcalCellOff") setBit(HcalChannelStatus::HcalCellOff, mydef.chStatusMask);
  else if (mybit == "HcalCellL1Mask") setBit(HcalChannelStatus::HcalCellL1Mask, mydef.chStatusMask);
  else if (mybit == "HcalCellDead") setBit(HcalChannelStatus::HcalCellDead, mydef.chStatusMask);
  else if (mybit == "HcalCellHot") setBit(HcalChannelStatus::HcalCellHot, mydef.chStatusMask);
  else if (mybit == "HcalCellStabErr") setBit(HcalChannelStatus::HcalCellStabErr, mydef.chStatusMask);
  else if (mybit == "HcalCellTimErr") setBit(HcalChannelStatus::HcalCellTimErr, mydef.chStatusMask);
  else 
    { // error: unrecognized channel status name
      edm::LogWarning  ("HcalSeverityLevelComputer") 
	<< "HcalSeverityLevelComputer: Error: ChannelStatusFlag >>" << mybit 
	<< "<< unknown. Ignoring.";
    }
}

void HcalSeverityLevelComputer::getRecHitFlag(HcalSeverityDefinition& mydef, 
					      const std::string& mybit)
{
  // HB, HE ++++++++++++++++++++
  if (mybit == "HBHEHpdHitMultiplicity")
    setBit(HcalCaloFlagLabels::HBHEHpdHitMultiplicity, mydef.HBHEFlagMask);
  else if (mybit == "HBHEPulseShape")
    setBit(HcalCaloFlagLabels::HBHEPulseShape, mydef.HBHEFlagMask);

  // HO ++++++++++++++++++++
  else if (mybit == "HOBit")
    setBit(HcalCaloFlagLabels::HOBit, mydef.HOFlagMask);
  
  // HF ++++++++++++++++++++
  else if (mybit == "HFDigiTime")
    setBit(HcalCaloFlagLabels::HFDigiTime, mydef.HFFlagMask);
  else if (mybit == "HFLongShort")
    setBit(HcalCaloFlagLabels::HFLongShort, mydef.HFFlagMask);
  
  // ZDC ++++++++++++++++++++
  else if (mybit == "ZDCBit")
    setBit(HcalCaloFlagLabels::ZDCBit, mydef.ZDCFlagMask);
  
  // Calib ++++++++++++++++++++
  else if (mybit == "CalibrationBit")
    setBit(HcalCaloFlagLabels::CalibrationBit, mydef.CalibFlagMask);

  else
    {
      // error: unrecognized flag name
      edm::LogWarning  ("HcalSeverityLevelComputer") 
	<< "HcalSeverityLevelComputer: Error: RecHitFlag >>" << mybit 
	<< "<< unknown. Ignoring.";
    }
}

HcalSeverityLevelComputer::HcalSeverityLevelComputer( const edm::ParameterSet& iConfig)
{
  std::cout << "HcalSeverityLevelComputer - initializing" << std::endl;

  // initialize: get the levels and masks from the cfg:
  typedef std::vector< edm::ParameterSet > myParameters;
  myParameters myLevels = iConfig.getParameter<myParameters>((std::string)"SeverityLevels");

  // now run through the parameter set vector:
  for ( myParameters::iterator itLevels = myLevels.begin(); itLevels != myLevels.end(); ++itLevels)
    {
      // create the basic object
      HcalSeverityDefinition mydef;

      // get the level:
      mydef.sevLevel = itLevels->getParameter<int>("Level");

      // get the RecHitFlags:
      std::vector<std::string> myRecHitFlags = 
	itLevels->getParameter<std::vector <std::string> > ("RecHitFlags");

      // get channel statuses:
      std::vector<std::string> myChStatuses = 
	itLevels->getParameter<std::vector <std::string> > ("ChannelStatus");

      // now translate the RecHitFlags and the ChannelStatuses into a mask each:
      // channel status:
      for (unsigned k=0; k < myChStatuses.size(); k++)
	{
	  if (myChStatuses[k].empty()) break;
	  getChStBit(mydef, myChStatuses[k]);
	}
      // RecHitFlag:
      //      HBHEStatusFlag, HOStatusFlag, HFStatusFlag, ZDCStatusFlag, CalibrationFlag
      for (unsigned k=0; k < myRecHitFlags.size(); k++)
	{
	  if (myRecHitFlags[k].empty()) break;
	  getRecHitFlag(mydef, myRecHitFlags[k]);
	}

      //      std::cout << "Made Severity Level:" << std::endl;
      //      std::cout << mydef << std::endl;

      // finally, append the masks to the mask vectors, sorting them according to level   
      std::vector<HcalSeverityDefinition>::iterator it = SevDef.begin();

      do
	{
	  if (it == SevDef.end()) { SevDef.push_back(mydef); break; }
	  
	  if (it->sevLevel == mydef.sevLevel)
	    {
	      edm::LogWarning  ("HcalSeverityLevelComputer") 
		<< "HcalSeverityLevelComputer: Error: level " << mydef.sevLevel 
		<< " already defined. Ignoring new definition.";
	      break;
	    }

	  if (it->sevLevel < mydef.sevLevel) 
	    if (( (it+1) == SevDef.end()) || ( (it+1)->sevLevel > mydef.sevLevel ))
	      {
		SevDef.insert(it+1, mydef);
		break;
	      }

	  it++;
	}
      while(it != SevDef.end());

    } //for (myParameters::iterator itLevels=myLevels.begin(); itLevels != myLevels.end(); ++itLevels)

  edm::LogInfo("HcalSeverityLevelComputer") 
    << "HcalSeverityLevelComputer - Summary of Severity Levels:" << std::endl;
  for (std::vector<HcalSeverityDefinition>::iterator it = SevDef.begin(); it !=SevDef.end(); it++)
    {
      // debug: write the levels definitions on screen:
      edm::LogInfo("HcalSeverityLevelComputer") 
	<< (*it) << std::endl;
    }

  //
  // Now make the definition for recoveredRecHit
  //
  std::vector<std::string> myRecovered = 
	iConfig.getParameter<std::vector <std::string> > ("RecoveredRecHitBits");
  RecoveredRecHit_ = new HcalSeverityDefinition();
  for (unsigned k=0; k < myRecovered.size(); k++)
    {
      if (myRecovered[k].empty()) break;
      getRecHitFlag( (*RecoveredRecHit_), myRecovered[k]);
    }

  //
  // Now make the definition for dropChannel
  //
  std::vector<std::string> myDrop = 
	iConfig.getParameter<std::vector <std::string> > ("DropChannelStatusBits");
  DropChannel_ = new HcalSeverityDefinition();
  for (unsigned k=0; k < myDrop.size(); k++)
    {
      if (myDrop[k].empty()) break;
      getChStBit( (*DropChannel_), myDrop[k]);
    }

  edm::LogInfo("HcalSeverityLevelComputer")
    << "HcalSeverityLevelComputer - Summary for Recovered RecHit bits: \n"
    << (*RecoveredRecHit_) << std::endl
    << "HcalSeverityLevelComputer - Summary for Drop the Channel bits: \n"
    << (*DropChannel_) << std::endl;


} // HcalSeverityLevelComputer::HcalSeverityLevelComputer


HcalSeverityLevelComputer::~HcalSeverityLevelComputer() {}

  
int HcalSeverityLevelComputer::getSeverityLevel(const DetId& myid, const uint32_t& myflag, 
						const uint32_t& mystatus) const
{
  uint32_t myRecHitMask;
  HcalGenericDetId myId(myid);
  HcalGenericDetId::HcalGenericSubdetector mysubdet = myId.genericSubdet();

  for (unsigned i=(SevDef.size()-1); i >= 0; i--)
    {
      switch (mysubdet)
	{
	case HcalGenericDetId::HcalGenBarrel : case HcalGenericDetId::HcalGenEndcap : 
	  myRecHitMask = SevDef[i].HBHEFlagMask; break;
	case HcalGenericDetId::HcalGenOuter : myRecHitMask = SevDef[i].HOFlagMask; break;
	case HcalGenericDetId::HcalGenForward : myRecHitMask = SevDef[i].HFFlagMask; break;
	case HcalGenericDetId::HcalGenZDC : myRecHitMask = SevDef[i].ZDCFlagMask; break;
	case HcalGenericDetId::HcalGenCalibration : myRecHitMask = SevDef[i].CalibFlagMask; break;
	default: myRecHitMask = 0;
	}
 
      // for debugging:     
//      std::cout << std::hex << " SLD: RHMask 0x" << myRecHitMask
//		<< " chstmask 0x" << SevDef[i].chStatusMask
//		<< " RHmask & myflag 0x" << (myRecHitMask&myflag)
//		<< " chstmask & mystatus 0x" << (SevDef[i].chStatusMask&mystatus)
//		<< std::dec << " level = " << SevDef[i].sevLevel << std::endl;
      
      // true if:
      // rechitmask empty and chstatusmask empty
      // rechitmask empty and chstatusmask&mychstat true
      // chstatusmask empty and rechitmask&myflag true
      // rechitmask&myflag true OR chstatusmask&mychstat true


      if ( ( ( (!myRecHitMask) || (myRecHitMask & myflag) ) &&
	   ( (!SevDef[i].chStatusMask) || (SevDef[i].chStatusMask & mystatus) ) )
	   || ( (myRecHitMask & myflag) || (SevDef[i].chStatusMask & mystatus) ) )
	return SevDef[i].sevLevel;

    }

  return -1;  
}
  
bool HcalSeverityLevelComputer::recoveredRecHit(const DetId& myid, const uint32_t& myflag) const
{
  uint32_t myRecHitMask;
  HcalGenericDetId myId(myid);
  HcalGenericDetId::HcalGenericSubdetector mysubdet = myId.genericSubdet();

  switch (mysubdet)
    {
    case HcalGenericDetId::HcalGenBarrel : case HcalGenericDetId::HcalGenEndcap : 
      myRecHitMask = RecoveredRecHit_->HBHEFlagMask; break;
    case HcalGenericDetId::HcalGenOuter : myRecHitMask = RecoveredRecHit_->HOFlagMask; break;
    case HcalGenericDetId::HcalGenForward : myRecHitMask = RecoveredRecHit_->HFFlagMask; break;
    case HcalGenericDetId::HcalGenZDC : myRecHitMask = RecoveredRecHit_->ZDCFlagMask; break;
    case HcalGenericDetId::HcalGenCalibration : myRecHitMask = RecoveredRecHit_->CalibFlagMask; break;
    default: myRecHitMask = 0;
    }

  if (myRecHitMask & myflag) 
    return true;

  return false;
}

bool HcalSeverityLevelComputer::dropChannel(const uint32_t& mystatus) const
{
  if (DropChannel_->chStatusMask & mystatus)
    return true;

  return false;
}

void HcalSeverityLevelComputer::setBit(const unsigned bitnumber, uint32_t& where) 
{
  uint32_t statadd = 0x1<<(bitnumber);
  where = where|statadd;
}

std::ostream& operator<<(std::ostream& s, const HcalSeverityLevelComputer::HcalSeverityDefinition& def)
{
  s << "Hcal Severity Level Definition, Level = " << def.sevLevel << std::endl;
  s << std::hex << std::showbase;
  s << "  channel status mask = " <<  def.chStatusMask << std::endl;
  s << "  HBHEFlagMask        = " <<  def.HBHEFlagMask << std::endl;
  s << "  HOFlagMask          = " <<  def.HOFlagMask << std::endl;
  s << "  HFFlagMask          = " <<  def.HFFlagMask << std::endl;
  s << "  ZDCFlagMask         = " <<  def.ZDCFlagMask << std::endl;
  s << "  CalibFlagMask       = " <<  def.CalibFlagMask << std::dec << std::noshowbase << std::endl;
  return s;
} 
