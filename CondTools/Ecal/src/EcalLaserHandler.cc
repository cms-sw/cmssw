#include "CondTools/Ecal/interface/EcalLaserHandler.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/LMFSextuple.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"

#include<iostream>
#include <sstream>

popcon::EcalLaserHandler::EcalLaserHandler(const edm::ParameterSet & ps) 
  : m_name(ps.getUntrackedParameter<std::string>("name","EcalLaserHandler")) {
  
  std::cout << "EcalLaser Source handler constructor\n" << std::endl;
  
  m_sid= ps.getParameter<std::string>("OnlineDBSID");
  m_user= ps.getParameter<std::string>("OnlineDBUser");
  m_pass= ps.getParameter<std::string>("OnlineDBPassword");
  m_locationsource= ps.getParameter<std::string>("LocationSource");
  m_location=ps.getParameter<std::string>("Location");
  m_gentag=ps.getParameter<std::string>("GenTag");
  m_debug=ps.getParameter<bool>("debug");
  
  std::cout << "Starting O2O process on DB: " << m_sid
	    << " User: "<< m_user << " Location: " << m_location 
	    << " Tag: " << m_gentag << std::endl;
}

popcon::EcalLaserHandler::~EcalLaserHandler()
{
  // do nothing
}

/*
bool popcon::EcalLaserHandler::checkAPDPN(float x, float old_x)
{
  bool result=true;
  if(x<=0 || x>20) result=false;
  if((old_x!=1.000 && old_x!=0) && std::abs(x-old_x)/old_x>100.00) result=false; 
  return result;
}
*/

void popcon::EcalLaserHandler::getNewObjects()
{
  std::cerr << "------- " << m_name 
	    << " ---> getNewObjects" << std::endl;
  
  std::cout << "------- Ecal -> getNewObjects\n";
  
  
  unsigned long long max_since= 1;
  Ref payload= lastPayload();
  
  // here popcon tells us which is the last since of the last object in the 
  // offline DB
  max_since=tagInfo().lastInterval.first;
  Tm max_since_tm(max_since);

  // get the last object in the orcoff
  edm::Timestamp t_min= edm::Timestamp(18446744073709551615ULL);

  const EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap& laserRatiosMap = 
    payload->getLaserMap(); 
  std::cout << "payload->getLaserMap():  OK " << std::endl;
  std::cout << "Its size is " << laserRatiosMap.size() << std::endl;
  const EcalLaserAPDPNRatios::EcalLaserTimeStampMap& laserTimeMap = 
    payload->getTimeMap();
  std::cout << "payload->getTimeMap():  OK " << std::endl;
  std::cout << "Last Object in Offline DB has SINCE = "  << max_since
	    << " (" << max_since_tm.str() << ")"
	    << " and  SIZE = " << tagInfo().size
	    << std::endl;
  // loop through light modules and determine the minimum date among the
  // available channels
  for (int i=0; i<92; i++) {
    EcalLaserAPDPNRatios::EcalLaserTimeStamp timestamp = laserTimeMap[i];
    if( t_min > timestamp.t1) {
      t_min=timestamp.t1;
    }
  }
  
  std::cout <<"WOW: we just retrieved the last valid record from DB "
	    << std::endl;
  std::cout <<"Its tmin is "<< Tm(t_min.value()).str() << std::endl;

  // connect to the database 
  try {
    std::cout << "Making connection..." << std::flush;
    econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
    std::cout << "Done." << std::endl;
  } catch (std::runtime_error &e) {
    std::cout << " connection parameters " << m_sid << "/" << m_user;
    if (m_debug) {
      std::cout << "/" << m_pass <<std::endl;
    } else {
      std::cout << "/**********" <<std::endl;
    }
    std::cerr << e.what() << std::endl;
    throw cms::Exception("OMDS not available");
  } 

  // retrieve the lists of logic_ids, to build the detids
  std::vector<EcalLogicID> crystals_EB  = 
    econn->getEcalLogicIDSetOrdered( "EB_crystal_angle",
				     -85,85,1,360,
				     EcalLogicID::NULLID,EcalLogicID::NULLID,
				     "EB_crystal_number", 4 );
  std::vector<EcalLogicID> crystals_EE  = 
    econn->getEcalLogicIDSetOrdered( "EE_crystal_number",
				     -1,1,1,100,
				     1,100,
				     "EE_crystal_number", 4 );
  
  std::vector<EcalLogicID>::const_iterator ieb = crystals_EB.begin();
  std::vector<EcalLogicID>::const_iterator eeb = crystals_EB.end();

  std::cout << "Got list of " << crystals_EB.size() << " crystals in EB" 
	    << std::endl;
  std::cout << "Got list of " << crystals_EE.size() << " crystals in EE" 
	    << std::endl;
  // loop through barrel
  int count = 0;
  // prepare a map to associate EB logic id's to detids
  std::map<int, int> detids;
  while (ieb != eeb) {
    int iEta = ieb->getID1();
    int iPhi = ieb->getID2();
    count++;
    EBDetId ebdetid(iEta,iPhi);
    unsigned int hieb = ebdetid.hashedIndex();    
    detids[ieb->getLogicID()] = hieb;
    ieb++;
  }
  std::cout << "Validated " << count << " logic ID's for EB" << std::endl;
  
  // do the same for EE
  
  std::vector<EcalLogicID>::const_iterator iee = crystals_EE.begin();
  std::vector<EcalLogicID>::const_iterator eee = crystals_EE.end();

  count = 0;
  while (iee != eee) {
    int iSide = iee->getID1();
    int iX    = iee->getID2();
    int iY    = iee->getID3();
    EEDetId eedetidpos(iX,iY,iSide);
    int hi = eedetidpos.hashedIndex();
    detids[iee->getLogicID()] = hi;
    count ++;
    iee++;
  }
  std::cout << "Validated " << count << " logic ID's for EE" << std::endl;

  // get association between ecal logic id and LMR
  std::map<int, int> logicId2Lmr = econn->getEcalLogicID2LmrMap();

  std::cout << "Retrieving corrections from ONLINE DB ... " << std::endl;

  LMFCorrCoefDat data(econn);
  if (m_debug) {
    data.debug();
  }
  // get all data in the database taken after the last available time in ORCOFF
  // we associate another map, whose key is the crystal ID and whose value is a
  // sextuple (p1, p2, p3, t1, t2, t3)
  std::map<int, std::map<int, LMFSextuple> > d = 
    data.getCorrections(Tm(t_min.value()), 16);
  // sice must be equal to the number of different SEQ_ID's found
  std::cout << "Data organized into " << d.size() << " sequences" << std::endl;
  // iterate over sequences
  std::map<int, std::map<int, LMFSextuple> >::const_iterator iseq = d.begin();
  std::map<int, std::map<int, LMFSextuple> >::const_iterator eseq = d.end();
  std::cout << "===== Looping on Sequences" << std::endl;
  while (iseq != eseq) {
    std::cout << "==== SEQ_ID: " << iseq->first 
	      << " contains " << iseq->second.size() << " crystals" 
	      << std::endl << std::flush;
    // iterate over crystals
    std::map<int, LMFSextuple>::const_iterator is = iseq->second.begin();
    std::map<int, LMFSextuple>::const_iterator es = iseq->second.end();
    EcalLaserAPDPNRatios* apdpns_popcon = new EcalLaserAPDPNRatios();         
    Time_t t_last = t_min.value(); 
    while (is != es) {
      EcalLaserAPDPNRatios::EcalLaserAPDPNpair apdpnpair_temp;
      apdpnpair_temp.p1 = is->second.p[0];
      apdpnpair_temp.p2 = is->second.p[1];
      apdpnpair_temp.p3 = is->second.p[2];
      EcalLaserAPDPNRatios::EcalLaserTimeStamp timestamp_temp;
      timestamp_temp.t1 = edm::Timestamp(is->second.t[0].microsTime());
      timestamp_temp.t2 = edm::Timestamp(is->second.t[1].microsTime());
      timestamp_temp.t3 = edm::Timestamp(is->second.t[2].microsTime());
      apdpns_popcon->setValue(detids[is->first], apdpnpair_temp);
      if (logicId2Lmr.find(is->first) != logicId2Lmr.end()) {
	int hashedIndex = logicId2Lmr[is->first] - 1;
	if ((hashedIndex >= 0) && (hashedIndex <= 91)) {
	  apdpns_popcon->setTime( hashedIndex , timestamp_temp);
	  t_last = timestamp_temp.t1.value();
	} else {
	  std::stringstream ss;
	  ss << "LOGIC_ID: " << is->first << " LMR: " << hashedIndex + 1
	     << " Out of range";
	  throw(std::runtime_error("[EcalLaserHandler::getNewObjects]" +
				   ss.str()));
	}
      } else {
	  std::stringstream ss;
	  ss << "LOGIC_ID: " << is->first << " Cannot determine LMR";
	  throw(std::runtime_error("[EcalLaserHandler::getNewObjects]" +
				   ss.str()));
      }
      is++;
    }
    if (iseq->second.size() > 0) {
      m_to_transfer.push_back(std::make_pair(apdpns_popcon, t_last));
    }
    iseq++;
  }
  std::cout << "==== END OF LOOP ON SEQUENCES" << std::endl << std::flush;
  delete econn;
  std::cout << "Ecal -> end of getNewObjects -----------\n";
	
	
}
