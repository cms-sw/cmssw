#include "CondTools/Ecal/interface/EcalLaser_weekly_Linearization.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/LMFSextuple.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"

#include<iostream>
#include<iomanip>
#include <sstream>

popcon::EcalLaser_weekly_Linearization::EcalLaser_weekly_Linearization(const edm::ParameterSet & ps) 
  : m_name(ps.getUntrackedParameter<std::string>("name","EcalLaser_weekly_Handler")) {
  std::cout << "EcalLaser Source handler constructor\n" << std::endl;
}

popcon::EcalLaser_weekly_Linearization::~EcalLaser_weekly_Linearization() {
  // do nothing
}

void popcon::EcalLaser_weekly_Linearization::getNewObjects() {
  //  int file[1] = {190708};

  int iIov = 0;  

  std::cout << "------- Ecal -> getNewObjects\n";
  
  
  unsigned long long max_since= 1;
  Ref payload= lastPayload();
  
  // here popcon tells us which is the last since of the last object in the offline DB
  max_since=tagInfo().lastInterval.first;
  Tm max_since_tm(max_since);

  //  const EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap& laserRatiosMap = 
  //    payload->getLaserMap(); 
  //  std::cout << "payload->getLaserMap():  OK " << std::endl;
  //  std::cout << "Its size is " << laserRatiosMap.size() << std::endl;
  //  const EcalLaserAPDPNRatios::EcalLaserTimeStampMap& laserTimeMap = 
  //    payload->getTimeMap();
  //  std::cout << "payload->getTimeMap():  OK " << std::endl;
  //  std::cout << "Last Object in Offline DB has SINCE = "  << max_since
  //	    << " -> " << max_since_tm.cmsNanoSeconds() 
  //	    << " (" << max_since_tm << ")"
  //	    << " and  SIZE = " << tagInfo().size
  //	    << std::endl;

  for(int week = 0; week < 1; week++) {
    int fileIOV;
    std::cout << " which input IOV do you want " << std::endl;
    std::cin >> fileIOV; 
    std::ifstream fWeek;
    std::ostringstream oss;
    oss << fileIOV;
    std::string fname = "/afs/cern.ch/cms/ECAL/triggerTransp/TPG_beamv6_trans_" + oss.str() + "_spikekill.txt";
    /*
    oss << file[week];
    //    std::string fname = "/afs/cern.ch/cms/ECAL/triggerTransp/TPG_beamv5_trans_" + oss.str() + "_";
    std::string fname = "/afs/cern.ch/cms/ECAL/triggerTransp/TPG_beamv6_trans_" + oss.str() + "_";
    oss.str("");
    //   if(week == 32) oss << 200000;
    //   else oss << file[week + 1] - 1;
    oss << 200000;
    fname += oss.str() + ".txt";
    */
    fWeek.open(fname.c_str());
    if(!fWeek.is_open()) {
      std::cout << "ERROR : can't open file '" << fname << std::endl;
      break;
    }
    std::cout << " file " << fname << " opened" << std::endl;
    //    int rawId;
    //    float corrp;
    std::string line;
    for(int i = 0; i < 85; i++) getline (fWeek, line);
    char cryst[10];
    uint32_t  ped[kGains], mult[kGains], shift[kGains];
    uint32_t id;

    EcalTPGLinearizationConst *linC = new EcalTPGLinearizationConst;

    for (int iChannel = 0; iChannel < kEBChannels; iChannel++) {
      getline (fWeek, line);
      //     std::cout << " line " << line << std::endl;
      //      fWeek >> cryst >> id;
      sscanf(line.c_str(), "%s %u", cryst, &id);
      //      std::cout << cryst << " id " << id << std::endl; 
      // EBDetId ebId = DetId(id);
      for (int gain = 0; gain < kGains; gain++) {
	//	fWeek >> std::hex >> ped[gain] >> mult[gain] >> shift[gain];
	getline (fWeek, line);
	//	std::cout << " line g " << line << std::endl;
	sscanf(line.c_str(), "%X %X %X", &ped[gain], &mult[gain], &shift[gain]);
	//	std::cout << " gain " << gain << " ped " << ped[gain] << " mult " << mult[gain] << " shift " << shift[gain]<< std::endl;
      }
      EcalTPGLinearizationConst::Item item;
      item.mult_x1   = mult[2];
      item.mult_x6   = mult[1];
      item.mult_x12  = mult[0];
      item.shift_x1  = shift[2];
      item.shift_x6  = shift[1];
      item.shift_x12 = shift[0];
    	  
      linC->insert(std::make_pair(id, item));
      //      corr.p1=corr.p2=corr.p3 = corrp;
      //      corrSet->setValue((int)ebId, corr );
      //      cryst ="";
    }   // end loop over EB channels
    getline (fWeek, line);  // cmment before EE crystals
    std::cout << " comment line " << line << std::endl;
    for (int iChannel = 0; iChannel < kEEChannels; iChannel++) {
      getline (fWeek, line);
      //      std::cout << " line " << line << std::endl;
      sscanf(line.c_str(), "%s %u", cryst, &id);
      //      std::cout << cryst << " id " << id << std::endl; 
      // EEDetId eeId = DetId(id);
      for (int gain =0; gain < kGains; gain++) {
	getline (fWeek, line);
	//	std::cout << " line g " << line << std::endl;
	sscanf(line.c_str(), "%X %X %X", &ped[gain], &mult[gain], &shift[gain]);
	//	std::cout << " gain " << gain << " ped " << ped[gain] << " mult " << mult[gain] << " shift " << shift[gain]<< std::endl;

      }      //      corr.p1=corr.p2=corr.p3 = corrp;
      //      corrSet->setValue((int)eeId, corr );
      EcalTPGLinearizationConst::Item item;
      item.mult_x1   = mult[2];
      item.mult_x6   = mult[1];
      item.mult_x12  = mult[0];
      item.shift_x1  = shift[2];
      item.shift_x6  = shift[1];
      item.shift_x12 = shift[0];
    	  
      linC->insert(std::make_pair(id, item));
    }   // end loop over EE channels
    fWeek.close();
    // special tag for Stephanie
    //    m_to_transfer.push_back(std::make_pair((EcalTPGLinearizationConst*)linC, file[week]));
    m_to_transfer.push_back(std::make_pair(linC, fileIOV));
    // end special
    iIov++;
  }   // end loop over week
  std::cout << "Ecal -> end of getNewObjects -----------\n";	
}
