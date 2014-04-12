#include "CondTools/Ecal/interface/EcalLaser_weekly_Linearization_Check.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/LMFSextuple.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"

#include<iostream>
#include<iomanip>
#include <sstream>
#include "TFile.h"
#include <string>

popcon::EcalLaser_weekly_Linearization_Check::EcalLaser_weekly_Linearization_Check(const edm::ParameterSet & ps) 
  : m_name(ps.getUntrackedParameter<std::string>("name","EcalLaser_weekly_Handler")) {
  std::cout << "EcalLaser Source handler constructor\n" << std::endl;
}

popcon::EcalLaser_weekly_Linearization_Check::~EcalLaser_weekly_Linearization_Check() {
  // do nothing
}

void popcon::EcalLaser_weekly_Linearization_Check::getNewObjects() {
  std::cout << "------- Ecal -> getNewObjects\n";
  
  std::ifstream fin;
  int conf, idEB, crysEB[61200];
  uint32_t mult12EB[61200], mult6EB[61200], mult1EB[61200], shift12EB[61200], shift6EB[61200], shift1EB[61200];
  int idEE, crysEE[14648];
  uint32_t mult12EE[14648], mult6EE[14648], mult1EE[14648], shift12EE[14648], shift6EE[14648], shift1EE[14648];
  int ansDB, ans;
  std::string fDB = "";
  std::cout << "Check from OMDS (1) or Orcon prod DB (2)? ";
  std::cin >> ansDB;
  if(ansDB == 1) {
    std::cout << "Which LIN_DATA_CONF_ID? ";
    std::cin >> ans;
    fDB = Form("/afs/cern.ch/cms/ECAL/triggerTransp/LIN_DATA_CONF_ID_%i.dat",ans);
    fin.open(fDB.c_str());
    if(!fin) {
      std::cout << "Error: file LIN_DATA_CONF_ID_"<< ans << ".dat could not be opened" << std::endl;
      exit(1);
    }
    std::cout << "file LIN_DATA_CONF_ID_"<< ans << ".dat opened" << std::endl;
    std::string c, log, m1, m2, m3, s1, s2, s3;
    fin >> c >> log >> m1 >> m2 >> m3 >> s1 >> s2 >> s3;
    // EB
    for (int ich = 0; ich < 61200; ich++) {
      fin >> conf >> idEB >> mult12EB[ich] >> mult6EB[ich] >> mult1EB[ich] >> shift12EB[ich] >> shift6EB[ich] >> shift1EB[ich];
      int chinSM = idEB%10000;
      int SM = (idEB/10000)%100;
      if(SM < 1 || SM > 36 || chinSM < 1 || chinSM > 1700) std::cout << idEB << " EB channel " << chinSM << " SM " << SM << std::endl; 
      EBDetId EBId(SM, chinSM, EBDetId::SMCRYSTALMODE);
      crysEB[ich] = EBId.hashedIndex();
    }
    // EE
    for (int ich = 0; ich < 14648; ich++) {
      fin >> conf >> idEE >> mult12EE[ich] >> mult6EE[ich] >> mult1EE[ich] >> shift12EE[ich] >> shift6EE[ich] >> shift1EE[ich];
      int ix = (idEE/1000)%1000;
      int iy = idEE%1000;
      int iz = (idEE/1000000)%10;
      int side = -1;
      if(iz == 2) side = 1;
      if(ix < 1 || ix > 100 || iy < 1 || iy > 100 || (iz != 0 && iz != 2)) 
	std::cout << idEE << " ix " << ix << " iy " << iy << " iz " << iz << std::endl; 
      EEDetId EEId(ix, iy, side, EEDetId::XYMODE);
      crysEE[ich] = EEId.hashedIndex();
    }
    fin.close();
  }   // end OMDS
  else if(ansDB == 2) {
    std::cout << "Which Linearization? ";
    std::cin >> ans;
    fDB = Form("./Linearization_%i.txt",ans);
    fin.open(fDB.c_str());
    if(!fin) {
      std::cout << "Error: file " << fDB << " could not be opened" << std::endl;
      exit(1);
    }
    std::cout << "prod DB file " << fDB << " opened" << std::endl;
    // EB
    for (int ich = 0; ich < 61200; ich++) {
      fin >> idEB >> mult12EB[ich] >> mult6EB[ich] >> mult1EB[ich] >> shift12EB[ich] >> shift6EB[ich] >> shift1EB[ich];
      EBDetId ebId = DetId(idEB);
      crysEB[ich] = ebId.hashedIndex();
    }
    // EE
    for (int ich = 0; ich < 14648; ich++) {
      fin >> idEE >> mult12EE[ich] >> mult6EE[ich] >> mult1EE[ich] >> shift12EE[ich] >> shift6EE[ich] >> shift1EE[ich];
      EEDetId eeId = DetId(idEE);
      crysEE[ich] = eeId.hashedIndex();
    }
    fin.close();
  }   // end Orcon
  else {
    std::cout << ansDB << " is not a right answer. Sorry let us give up!" << std::endl;
    exit(-1);
  }

  //    EcalTPGLinearizationConst *linC = new EcalTPGLinearizationConst;
  int fileIOV;
  std::cout << " Which input file IOV? ";
  std::cin >> fileIOV; 
  std::ifstream fWeek;
  std::ostringstream oss;
  oss << fileIOV;
  std::string fname = "/afs/cern.ch/cms/ECAL/triggerTransp/TPG_beamv6_trans_" + oss.str() + "_spikekill.txt";
  fWeek.open(fname.c_str());
  if(!fWeek.is_open()) {
    std::cout << "ERROR : can't open file '" << fname << std::endl;
    exit(-1);
  }
  std::cout << " file " << fname << " opened" << std::endl;
  std::string line;
  for(int i = 0; i < 85; i++) getline (fWeek, line);
  char cryst[10];
  uint32_t  ped[kGains], mult[kGains], shift[kGains];
  uint32_t id;
  for (int iChannel = 0; iChannel < kEBChannels; iChannel++) {
    getline (fWeek, line);
    sscanf(line.c_str(), "%s %u", cryst, &id);
    EBDetId ebId = DetId(id);
    int crys = ebId.hashedIndex();
    bool found = false;
    int ich = -1;
    for (int icc = 0; icc < 61200; icc++) {
      if(crys == crysEB[icc]) {
	found = true;
	ich = icc;
	break;
      }
    }
    if(!found) std::cout << " ***** EB crystal not found in DB " << crys << std::endl;
    for (int gain = 0; gain < kGains; gain++) {
      getline (fWeek, line);
      sscanf(line.c_str(), "%X %X %X", &ped[gain], &mult[gain], &shift[gain]);
    }
    if(mult[0] != mult12EB[ich]) 
      std::cout << " mult12 file " << fDB << " " << mult12EB[ich] << " file " << fname << " " << mult[0] << "\n";
    if(mult[1] != mult6EB[ich]) 
      std::cout << " mult6 file " << fDB << " " << mult6EB[ich] << " file " << fname << " " << mult[1] << "\n";
    if(mult[2] != mult1EB[ich]) 
      std::cout << " mult1 file " << fDB << " " << mult1EB[ich] << " file " << fname << " " << mult[2] << "\n";
    if(shift[0] != shift12EB[ich]) 
      std::cout << " shift12 file " << fDB << " " << shift12EB[ich] << " file " << fname << " " << shift[0] << "\n";
    if(shift[1] != shift6EB[ich]) 
      std::cout << " shift6 file " << fDB << " " << shift6EB[ich] << " file " << fname << " " << shift[1] << "\n";
    if(shift[2] != shift1EB[ich]) {
      std::cout << " ***** EB crystal " << id << " hashed " << crys << std::endl;
      std::cout << " shift1 file " << fDB << " " << shift1EB[ich] << " file " << fname << " " << shift[2] << "\n";
      exit(-1);
    }
  }   // end loop over EB channels
  getline (fWeek, line);  // comment before EE crystals
  std::cout << " comment line " << line << std::endl;
  int chm12 = 0, chm6 = 0, chm1 = 0, chs12 = 0, chs6 = 0, chs1 = 0;
  for (int iChannel = 0; iChannel < kEEChannels; iChannel++) {
    getline (fWeek, line);
    sscanf(line.c_str(), "%s %u", cryst, &id);
    EEDetId eeId = DetId(id);
    int crys = eeId.hashedIndex();
    bool found = false;
    int ich = -1;
    for (int icc = 0; icc < 14648; icc++) {
      if(crys == crysEE[icc]) {
	found = true;
	ich = icc;
	break;
      }
    }
    if(!found) std::cout << " ***** EE crystal not found in DB " << crys << std::endl;
    for (int gain =0; gain < kGains; gain++) {
      getline (fWeek, line);
      sscanf(line.c_str(), "%X %X %X", &ped[gain], &mult[gain], &shift[gain]);
      //	std::cout << " gain " << gain << " ped " << ped[gain] << " mult " << mult[gain] << " shift " << shift[gain]<< std::endl;
    } 
    if(mult[0] != mult12EE[ich]) chm12++;
    if(mult[1] != mult6EE[ich]) chm6++;
    if(mult[2] != mult1EE[ich]) chm1++;
    if(shift[0] != shift12EE[ich])chs12++;
    if(shift[1] != shift6EE[ich]) chs6++;
    if(shift[2] != shift1EE[ich]) chs1++;
  }   // end loop over EE channels
  fWeek.close();
  if(chm12 != 0 || chm6 != 0 || chm1 != 0 || chs12 != 0 || chs6 != 0 || chs1 != 0)
    std::cout << " different files "<< fDB << " and TPG_beamv6_trans_" << fileIOV << "_spikekill.txt" << "\n"
	      << " mult12 " << chm12 << " mult6 " << chm6 << " mult1 " << chm1 
	      << " shift12 " << chs12 << " shift6 " << chs6 << " shift1 " << chs1 << std::endl;
  else std::cout << "identical files "<< fDB 
		 << " and TPG_beamv6_trans_" << fileIOV << "_spikekill.txt" <<std:: endl;
  std::cout << "Ecal -> end of getNewObjects -----------\n";	
}
