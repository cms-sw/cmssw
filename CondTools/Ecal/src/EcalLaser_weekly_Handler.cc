#include "CondTools/Ecal/interface/EcalLaser_weekly_Handler.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/LMFSextuple.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"

#include<iostream>
#include<iomanip>
#include <sstream>

popcon::EcalLaser_weekly_Handler::EcalLaser_weekly_Handler(const edm::ParameterSet & ps) 
  : m_name(ps.getUntrackedParameter<std::string>("name","EcalLaser_weekly_Handler")) {
  //  wrongBy = ps.getUntrackedParameter<double>("WrongBy",1.0);
  std::cout << "EcalLaser Source handler constructor\n" << std::endl;
}

popcon::EcalLaser_weekly_Handler::~EcalLaser_weekly_Handler() {
  // do nothing
}

void popcon::EcalLaser_weekly_Handler::getNewObjects() {
  //  uint64_t iov[1] = {5726925116361670656};
  //  int file[1] = {190111};
  //  int file[1] = {190708};
  const size_t nLmes = 92;
  //  cond::Time_t iovStart = 0;
  uint64_t t1, t2, t3;

  int iIov = 0;  

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
  for (int i=0; i<92; i++) {
    EcalLaserAPDPNRatios::EcalLaserTimeStamp timestamp = laserTimeMap[i];
    if( t_min > timestamp.t1) {
      t_min=timestamp.t1;
    }
  }

  std::cout <<"WOW: we just retrieved the last valid record from DB "
	    << std::endl;
  //std::cout <<"Its tmin is "<< Tm((t_min.value() >> 32)*1000000)
  std::cout <<"Its tmin is "<< Tm(t_min.value()) << std::endl;

  //  for(int week = 0; week < 1; week++) {
  EcalLaserAPDPNRatios* corrSet = new EcalLaserAPDPNRatios;  

  EcalLaserAPDPNRatios::EcalLaserAPDPNpair corr = {0, 0, 0};
  int fileIOV;
  std::cout << " which input IOV do you want " << std::endl;
  std::cin >> fileIOV; 
  std::ifstream fWeek;
  std::ostringstream oss;
  oss << fileIOV;
  std::string fname = "../../../Tools/DBDump/bin/weekly_" + oss.str();
  fWeek.open(fname.c_str());
  if(!fWeek.is_open()) {
    std::cout << "ERROR : can't open file 'weekly_" << oss.str() << std::endl;
    exit(-1);
  }
  std::cout << " file weekly_" << oss.str() << " opened" << std::endl;

  // find the timestamp for this run
  std::ifstream fRunStartTime;
  fRunStartTime.open("RunStartTime");
  if(!fRunStartTime.is_open()) {
    std::cout << "ERROR : cannot open file RunStartTime" << std::endl;
    exit (1);
  }
  uint64_t iov = 0;
  while(!fRunStartTime.eof()) {
    int run;
    uint64_t start;
    fRunStartTime >> run >> start;
    if(run == fileIOV) {
      iov = start;
      std::cout << "run " << run << " timestamp " << start << "\n";
      break;
    }
    else if (run == fileIOV) {
      std::cout << " run " << fileIOV << " not found in RunStartTime. Let us give up" << std::endl;
      exit(-1);
    }
  }
  fRunStartTime.close();
  EcalLaserAPDPNRatios::EcalLaserTimeStamp t;
  
  t1 = iov;
  //    t3 = t1 + 2597596220620800; // 1 week << 32
  t3 = t1 + 7792788661862400; // 3 weeks << 32
  t2 = t1 + (t3 - t1)/2;
  //    iovStart = t1;
  for(size_t i = 0; i < nLmes; ++i){
    t.t1 = edm::Timestamp(t1);
    t.t2 = edm::Timestamp(t2);
    t.t3 = edm::Timestamp(t3);
    corrSet->setTime(i, t);
  }

    int rawId;
    float corrp;
    for (int iChannel = 0; iChannel < kEBChannels; iChannel++) {
      EBDetId ebId = EBDetId::unhashIndex(iChannel);
      fWeek >> rawId >> corrp;
      corr.p1=corr.p2=corr.p3 = corrp;
      corrSet->setValue((int)ebId, corr );
    }
    for (int iChannel = 0; iChannel < kEEChannels; iChannel++) {
      EEDetId eeId = EEDetId::unhashIndex(iChannel);
      fWeek >> rawId >> corrp;
      corr.p1=corr.p2=corr.p3 = corrp;
      corrSet->setValue((int)eeId, corr );
    }
    fWeek.close();

    std::cout << "Write IOV " << iIov << " starting from " <<  fileIOV << "... "<< std::endl;
      //      db_->writeOne(corrSet, iovStart, "EcalLaserAPDPNRatiosRcd");
    m_to_transfer.push_back(std::make_pair(corrSet, fileIOV));
    iIov++;
    //  }   // end loop over week
  std::cout << "Ecal -> end of getNewObjects -----------\n";	
}
