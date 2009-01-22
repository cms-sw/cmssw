#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelProducer.h"
#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


HcalSeverityLevelProducer::HcalSeverityLevelProducer( const edm::ParameterSet& iConfig)
{
  std::cout << "HcalSeverityLevelProducer - initializing" << std::endl;

  // initialize: get the levels and masks from the cfg:
  typedef std::vector< edm::ParameterSet > myParameters;
  myParameters myLevels = iConfig.getParameter<myParameters>((std::string)"Levels");

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
	  if (myChStatuses[k] == "HcalCellOff") 
	    setBit(HcalChannelStatus::HcalCellOff, mydef.chStatusMask);
	  else if (myChStatuses[k] == "HcalCellL1Mask") 
	    setBit(HcalChannelStatus::HcalCellL1Mask, mydef.chStatusMask);
	  else if (myChStatuses[k] == "HcalCellDead") 
	    setBit(HcalChannelStatus::HcalCellDead, mydef.chStatusMask);
	  else if (myChStatuses[k] == "HcalCellHot") 
	    setBit(HcalChannelStatus::HcalCellHot, mydef.chStatusMask);
	  else if (myChStatuses[k] == "HcalCellStabErr") 
	    setBit(HcalChannelStatus::HcalCellStabErr, mydef.chStatusMask);
	  else if (myChStatuses[k] == "HcalCellTimErr") 
	    setBit(HcalChannelStatus::HcalCellTimErr, mydef.chStatusMask);
	  else 
	    {
	      // error: unrecognized channel status name
	      std::cout << "HcalSeverityLevelProducer: Error: ChannelStatusFlag >>" << myChStatuses[k] 
			<< "<< unknown. Ignoring." << std::endl;
	    }
	}
      // RecHitFlag:
      //      HBHEStatusFlag, HOStatusFlag, HFStatusFlag, ZDCStatusFlag, CalibrationFlag
      for (unsigned k=0; k < myRecHitFlags.size(); k++)
	{
        // HB, HE ++++++++++++++++++++
	  if (myRecHitFlags[k] == "HBHEHpdHitMultiplicity")
	    setBit(HcalCaloFlagLabels::HBHEHpdHitMultiplicity, mydef.HBHEFlagMask);
	  else
          if (myRecHitFlags[k] == "HBHEPulseShape")
            setBit(HcalCaloFlagLabels::HBHEPulseShape, mydef.HBHEFlagMask);
          else
        // HO ++++++++++++++++++++
          if (myRecHitFlags[k] == "HOBit")
            setBit(HcalCaloFlagLabels::HOBit, mydef.HOFlagMask);
	  else
        // HF ++++++++++++++++++++
	  if (myRecHitFlags[k] == "HFDigiTime")
	    setBit(HcalCaloFlagLabels::HFDigiTime, mydef.HFFlagMask);
	  else
	  if (myRecHitFlags[k] == "HFLongShort")
	    setBit(HcalCaloFlagLabels::HFLongShort, mydef.HFFlagMask);
	  else
        // ZDC ++++++++++++++++++++
	  if (myRecHitFlags[k] == "ZDCBit")
	    setBit(HcalCaloFlagLabels::ZDCBit, mydef.ZDCFlagMask);
	  else
        // Calib ++++++++++++++++++++
	  if (myRecHitFlags[k] == "CalibrationBit")
	    setBit(HcalCaloFlagLabels::CalibrationBit, mydef.CalibFlagMask);
	  else
	    {
	      // error: unrecognized flag name
	      std::cout << "HcalSeverityLevelProducer: Error: RecHitFlag >>" << myRecHitFlags[k] 
			<< "<< unknown. Ignoring." << std::endl;
	    }
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
	      std::cout << "HcalSeverityLevelProducer: Error: level " << mydef.sevLevel 
			<< " already defined. Ignoring new definition." << std::endl;
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

  std::cout << "HcalSeverityLevelProducer - Summary of Severity Levels:" << std::endl;
  for (std::vector<HcalSeverityDefinition>::iterator it = SevDef.begin(); it !=SevDef.end(); it++)
    {
      // debug: write the levels definitions on screen:
      std::cout << (*it) << std::endl;
    }

} // HcalSeverityLevelProducer::HcalSeverityLevelProducer


HcalSeverityLevelProducer::~HcalSeverityLevelProducer() {}

  
int HcalSeverityLevelProducer::getSeverityLevel(const DetId mydetid, const uint32_t myflag, 
						const uint32_t mystatus)
{
  uint32_t myRecHitMask;
  HcalGenericDetId myId(mydetid);
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
  

void HcalSeverityLevelProducer::setBit(const unsigned bitnumber, uint32_t& where) 
{
  uint32_t statadd = 0x1<<(bitnumber);
  where = where|statadd;
}

std::ostream& operator<<(std::ostream& s, const HcalSeverityLevelProducer::HcalSeverityDefinition& def)
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
