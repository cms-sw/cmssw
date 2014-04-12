#include "CondTools/Ecal/interface/EcalLaserHandler.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/LMFSextuple.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"

#include<iostream>
#include<iomanip>
#include <sstream>

popcon::EcalLaserHandler::EcalLaserHandler(const edm::ParameterSet & ps) 
  : m_name(ps.getUntrackedParameter<std::string>("name","EcalLaserHandler")) {
  
  std::cout << "EcalLaser Source handler constructor\n" << std::endl;

  m_sequences = 1;
  m_fake = true;

  m_sid= ps.getParameter<std::string>("OnlineDBSID");
  m_user= ps.getParameter<std::string>("OnlineDBUser");
  m_pass= ps.getParameter<std::string>("OnlineDBPassword");
  m_debug=ps.getParameter<bool>("debug");
  m_fake=ps.getParameter<bool>("fake");
  m_sequences=static_cast<unsigned int>(atoi( ps.getParameter<std::string>("sequences").c_str()));
  m_maxtime=ps.getParameter<std::string>("maxtime").c_str();
  std::cout << "Starting O2O process on DB: " << m_sid
	    << " User: "<< m_user << std::endl;
  if (m_fake) {
    std::cout << "*******************************************" << std::endl;
    std::cout << "This is a fake run. No change to offline DB" << std::endl;
    std::cout << "*******************************************" << std::endl;
  }
}

popcon::EcalLaserHandler::~EcalLaserHandler()
{
  // do nothing
}

double popcon::EcalLaserHandler::diff(float a, float b) {
  return std::abs(b- a)/a;
}

void popcon::EcalLaserHandler::notifyProblems(const EcalLaserAPDPNRatios::EcalLaserAPDPNpair &old, 
					      const EcalLaserAPDPNRatios::EcalLaserAPDPNpair &current,
					      int hashedIndex, const std::string &reason) {
  std::cout << "===== " << reason << " =====" << std::endl;
  if (hashedIndex < 0) {
    EEDetId ee;
    std::cout << "Triplets for " << ee.unhashIndex(-hashedIndex) << " bad: [" << old.p1 << ", "
	      << old.p2 << ", " << old.p3 << "] ==> [" << current.p1 << ", "
	      << current.p2 << ", " << current.p3 << "]" << std::endl;
  } else {
    EBDetId eb;
    std::cout << "Triplets for " << eb.unhashIndex(hashedIndex) << " bad: [" << old.p1 << ", "
	      << old.p2 << ", " << old.p3 << "] ==> [" << current.p1 << ", "
	      << current.p2 << ", " << current.p3 << "]" << std::endl;
  }
} 

bool popcon::EcalLaserHandler::checkAPDPN(const EcalLaserAPDPNRatios::EcalLaserAPDPNpair &old, 
					  const EcalLaserAPDPNRatios::EcalLaserAPDPNpair &current,
					  int hashedIndex) {
  bool ret = true;
  if ((current.p1 < 0) || (current.p2 < 0) || (current.p3 < 0)) {
    ret = false;
    notifyProblems(old, current, hashedIndex, "Negative values");
  } else if ((current.p1 > 10) || (current.p2 > 10) || (current.p3 > 10)) {
    ret = false;
    notifyProblems(old, current, hashedIndex, "Values too large");
  } else if (((diff(old.p1, current.p1) > 0.2) && (old.p1 != 0) && (old.p1 != 1)) ||
	     ((diff(old.p2, current.p2) > 0.2) && (old.p2 != 0) && (old.p1 != 2)) ||
	     ((diff(old.p3, current.p3) > 0.2) && (old.p3 != 0) && (old.p1 != 3))) {
    ret = false;
    notifyProblems(old, current, hashedIndex, "Difference w.r.t. previous too large");
  }
  return ret;
}

bool popcon::EcalLaserHandler::checkAPDPNs(const EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap &laserMap,
                                           const EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap &apdpns_popcon) {
  bool ret = true;
  for (int hashedIndex = 0; hashedIndex < 61200; hashedIndex++) {
    EcalLaserAPDPNRatios::EcalLaserAPDPNpair old = laserMap.barrel(hashedIndex);
    EcalLaserAPDPNRatios::EcalLaserAPDPNpair current = apdpns_popcon.barrel(hashedIndex);
    ret = checkAPDPN(old, current, hashedIndex);
  }
  for (int hashedIndex = 0; hashedIndex < 14648; hashedIndex++) {
    EcalLaserAPDPNRatios::EcalLaserAPDPNpair old = laserMap.endcap(hashedIndex);
    EcalLaserAPDPNRatios::EcalLaserAPDPNpair current = apdpns_popcon.endcap(hashedIndex);
    ret = checkAPDPN(old, current, -hashedIndex);
  }
  return ret;
}

void popcon::EcalLaserHandler::dumpBarrelPayload(EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap const &laserMap) {
  int c = 0;
  EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap::const_iterator i = laserMap.barrelItems().begin();
  EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap::const_iterator e = laserMap.barrelItems().end();
  EBDetId eb;
  try {
    EcalCondDBInterface *econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
    while (i != e) {
      if (c % 1000 == 0) {
	std::cout << std::setw(5) << c << ": " << eb.unhashIndex(c) << " "   
		  << econn->getEcalLogicID("EB_crystal_angle", eb.unhashIndex(c).ieta(), 
					   eb.unhashIndex(c).iphi(), EcalLogicID::NULLID, 
					   "EB_crystal_number").getLogicID() 
		  << " " << std::setiosflags(std::ios::fixed) << std::setprecision(9) 
		  << i->p1 << " " << i->p2 << " " << i->p3 << std::endl;
      }
      i++;
      c++;
    }
    delete econn;
  }
  catch (std::runtime_error &e) {
    std::cerr << e.what() << std::endl;
    delete econn;
    throw cms::Exception("OMDS not available");
  }
}

void popcon::EcalLaserHandler::dumpEndcapPayload(EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap const &laserMap) {
  int c = 0;
  EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap::const_iterator i = laserMap.endcapItems().begin();
  EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap::const_iterator e = laserMap.endcapItems().end();
  EEDetId ee;
  try {
    EcalCondDBInterface *econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
    while (i != e) {
      if (c % 1000 == 0) {
	std::cout << std::setw(5) << c << ": " << ee.unhashIndex(c) << " "   
		  << econn->getEcalLogicID("EE_crystal_number", ee.unhashIndex(c).zside(), 
					   ee.unhashIndex(c).ix(), ee.unhashIndex(c).iy(),
					   "EE_crystal_number").getLogicID() 
		  << " " << std::setiosflags(std::ios::fixed) << std::setprecision(9) 
		  << i->p1 << " " << i->p2 << " " << i->p3 << std::endl;
      }
      i++;
      c++;
    }
    delete econn;
  }
  catch (std::runtime_error &e) {
    std::cerr << e.what() << std::endl;
    delete econn;
    throw cms::Exception("OMDS not available");
  }
}

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
  //  Tm max_since_tm((max_since >> 32)*1000000);
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
	    << " -> " << max_since_tm.cmsNanoSeconds() 
	    << " (" << max_since_tm << ")"
	    << " and  SIZE = " << tagInfo().size
	    << std::endl;
  // loop through light modules and determine the minimum date among the
  // available channels
  if (m_debug) {
    dumpBarrelPayload(laserRatiosMap);
    dumpEndcapPayload(laserRatiosMap);
  }
  for (int i=0; i<92; i++) {
    EcalLaserAPDPNRatios::EcalLaserTimeStamp timestamp = laserTimeMap[i];
    if( t_min > timestamp.t1) {
      t_min=timestamp.t1;
    }
  }

  std::cout <<"WOW: we just retrieved the last valid record from DB "
	    << std::endl;
  //std::cout <<"Its tmin is "<< Tm((t_min.value() >> 32)*1000000)
  std::cout <<"Its tmin is "<< Tm(t_min.value())
  	    << std::endl;

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
    //    unsigned int hieb = ebdetid.hashedIndex();    
    detids[ieb->getLogicID()] = ebdetid;
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
    //    int hi = eedetidpos.hashedIndex();
    detids[iee->getLogicID()] = eedetidpos;
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
  Tm tmax;
  if (m_maxtime[0] == '-') {
    // this is a time relative to now
    tmax.setToCurrentLocalTime();
    if (m_debug) {
      std::cout << "Subtracting " << m_maxtime.substr(1) << " hours "
		<< "to " << tmax.str() << std::endl;
      std::cout << "tmax was " << tmax.microsTime() << " ns" << std::endl;
    }
    tmax -= atoi(m_maxtime.substr(1).c_str())*3600;//
    if (m_debug) {
      std::cout << "tmax is  " << tmax.microsTime() << " ns" << std::endl;
    }
  } else {
    if (m_debug) {
      std::cout << "Setting t_max to " << m_maxtime << std::endl; 
    }
    tmax.setToString(m_maxtime);
  }
  //  Tm tmin = Tm((t_min.value() >> 32)*1000000);
  Tm tmin = Tm(t_min.value());
  /*
  Tm strunz;
  strunz.setToString("2011-04-11 20:50:08");
  if (tmin < strunz) {
    tmin = strunz;
  }
  */

  if (m_debug) {
    std::cout << "Tmin: " << tmin << std::endl;
    std::cout << "Tmax: " << tmax << std::endl;
  }

  std::map<int, std::map<int, LMFSextuple> > d = 
    data.getCorrections(tmin, tmax, m_sequences);
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
    // iterate over crystals, but skip those sequences with wrong number of crystals
    if (iseq->second.size() == (61200 + 14648)) {
      std::map<int, LMFSextuple>::const_iterator is = iseq->second.begin();
      std::map<int, LMFSextuple>::const_iterator es = iseq->second.end();
      EcalLaserAPDPNRatios* apdpns_popcon = new EcalLaserAPDPNRatios();         
      Time_t t_last = 18446744073709551615ULL;
      while (is != es) {
	EcalLaserAPDPNRatios::EcalLaserAPDPNpair apdpnpair_temp;
	apdpnpair_temp.p1 = is->second.p[0];
	apdpnpair_temp.p2 = is->second.p[1];
	apdpnpair_temp.p3 = is->second.p[2];
	EcalLaserAPDPNRatios::EcalLaserTimeStamp timestamp_temp;
	timestamp_temp.t1 = edm::Timestamp(is->second.t[0].cmsNanoSeconds());
	timestamp_temp.t2 = edm::Timestamp(is->second.t[1].cmsNanoSeconds());
	timestamp_temp.t3 = edm::Timestamp(is->second.t[2].cmsNanoSeconds());
	apdpns_popcon->setValue(detids[is->first], apdpnpair_temp);
	if (logicId2Lmr.find(is->first) != logicId2Lmr.end()) {
	  int hashedIndex = logicId2Lmr[is->first] - 1;
	  if ((hashedIndex >= 0) && (hashedIndex <= 91)) {
	    apdpns_popcon->setTime( hashedIndex , timestamp_temp);
	    if (t_last > timestamp_temp.t1.value()) {
	      t_last = timestamp_temp.t1.value();
	    }
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
      if (m_fake) {
	delete apdpns_popcon;
      }
      if ((iseq->second.size() > 0) && (!m_fake)) {
	m_to_transfer.push_back(std::make_pair(apdpns_popcon, 
					       Tm(t_last).cmsNanoSeconds()));
      } 
    } else {
      // Here we should put a warning
    }
    iseq++;
  }
  std::cout << "==== END OF LOOP ON SEQUENCES" << std::endl << std::flush;
  delete econn;
  std::cout << "Ecal -> end of getNewObjects -----------\n";
	
	
}
