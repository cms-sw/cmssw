#include "CondTools/Ecal/plugins/StoreESCondition.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <time.h>
#include <unistd.h> 

using std::string;

StoreESCondition::StoreESCondition(const edm::ParameterSet& iConfig) {

  prog_name_ = "StoreESCondition";

  logfile_ = iConfig.getParameter< std::string >("logfile");

  esgain_ =  iConfig.getParameter< unsigned int >("gain");

  typedef std::vector< edm::ParameterSet > Parameters;
  Parameters toPut=iConfig.getParameter<Parameters>("toPut");
  for (Parameters::iterator itToPut = toPut.begin(); itToPut != toPut.end(); ++itToPut) {
    inpFileName_.push_back(itToPut->getUntrackedParameter<std::string>("inputFile"));
    objectName_.push_back(itToPut->getUntrackedParameter<std::string>("conditionType"));
    since_.push_back(itToPut->getUntrackedParameter<unsigned int>("since"));
  }

}

void StoreESCondition::endJob() {
  
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if ( !mydbservice.isAvailable() ) {
    edm::LogError("StoreESCondition")<<"PoolDBOutputService is unavailable"<< "\n";
    return;
  }
  
  bool toAppend = false;
  // copy a string to the char *
  std::string message = "finished OK\n";
  size_t messageSize = message.size() + 1;
  char * messChar = new char [messageSize];
  strncpy(messChar, message.c_str(), messageSize);
  
  for (unsigned int i=0;i<objectName_.size();i++) {
    cond::Time_t newTime;
    
    if ( mydbservice->isNewTagRequest( objectName_[i]+std::string("Rcd") ) ) {
      // This is the first object for this tag.
      // Append mode should be off.
      // newTime is the end of this new objects IOV.
      newTime = mydbservice->beginOfTime();
    } else {
      // There should already be an object in the DB for this tag.
      // Append IOV mode should be on.
      // newTime is the beginning of this new objects IOV.
      toAppend=true;
      newTime = (cond::Time_t)since_[i];
    }
    edm::LogInfo("StoreESCondition") << "Reading " << objectName_[i] 
	      << " from file and writing to DB with newTime " << newTime << "\n";
    if (objectName_[i] == "ESChannelStatus") {
      edm::LogInfo("StoreESCondition") << " ESChannelStatus file "  << inpFileName_[i] << "\n";
      ESChannelStatus* mycali = readESChannelStatusFromFile(inpFileName_[i].c_str());
      edm::LogInfo("StoreESCondition") << " ESChannelStatus file read " << "\n";
      if(!toAppend){
	edm::LogInfo("StoreESCondition") << " before create " << "\n";
	mydbservice->createNewIOV<ESChannelStatus>(mycali,newTime,mydbservice->endOfTime(),"ESChannelStatusRcd");
	edm::LogInfo("StoreESCondition") << " after create " << "\n";
      }else{
	edm::LogInfo("StoreESCondition") << " before append " << "\n";
	mydbservice->appendSinceTime<ESChannelStatus>(mycali,newTime,"ESChannelStatusRcd");
	edm::LogInfo("StoreESCondition") << " after append " << "\n";
      } 
    } else if (objectName_[i] == "ESIntercalibConstants") {
      ESIntercalibConstants* myintercalib = readESIntercalibConstantsFromFile(inpFileName_[i].c_str());
      if(!toAppend){
	mydbservice->createNewIOV<ESIntercalibConstants>(myintercalib, newTime, mydbservice->endOfTime(), "ESIntercalibConstantsRcd");
      }else{
	mydbservice->appendSinceTime<ESIntercalibConstants>(myintercalib, newTime, "ESIntercalibConstantsRcd");
      } 
    } else if (objectName_[i] == "ESTimeSampleWeights") {
      ESTimeSampleWeights* myintercalib = readESTimeSampleWeightsFromFile(inpFileName_[i].c_str());
      if(!toAppend){
	mydbservice->createNewIOV<ESTimeSampleWeights>(myintercalib, newTime, mydbservice->endOfTime(), "ESTimeSampleWeightsRcd");
      }else{
	mydbservice->appendSinceTime<ESTimeSampleWeights>(myintercalib, newTime, "ESTimeSampleWeightsRcd");
      } 
    } else if (objectName_[i] == "ESGain") {
      ESGain* myintercalib = readESGainFromFile(inpFileName_[i].c_str());
      if(!toAppend){
	mydbservice->createNewIOV<ESGain>(myintercalib, newTime, mydbservice->endOfTime(), "ESGainRcd");
      }else{
	mydbservice->appendSinceTime<ESGain>(myintercalib, newTime, "ESGainRcd");
      } 
    } else if (objectName_[i] == "ESMissingEnergyCalibration") {
      ESMissingEnergyCalibration* myintercalib = readESMissingEnergyFromFile(inpFileName_[i].c_str());
      if(!toAppend){
	mydbservice->createNewIOV<ESMissingEnergyCalibration>(myintercalib, newTime, mydbservice->endOfTime(), "ESMissingEnergyCalibrationRcd");
      }else{
	mydbservice->appendSinceTime<ESMissingEnergyCalibration>(myintercalib, newTime, "ESMissingEnergyCalibrationRcd");
      } 
    } else if (objectName_[i] == "ESRecHitRatioCuts") {
      ESRecHitRatioCuts* myintercalib = readESRecHitRatioCutsFromFile(inpFileName_[i].c_str());
      if(!toAppend){
	mydbservice->createNewIOV<ESRecHitRatioCuts>(myintercalib, newTime, mydbservice->endOfTime(), "ESRecHitRatioCutsRcd");
      }else{
	mydbservice->appendSinceTime<ESRecHitRatioCuts>(myintercalib, newTime, "ESRecHitRatioCutsRcd");
      } 
    } else if (objectName_[i] == "ESThresholds") {
      ESThresholds* myintercalib = readESThresholdsFromFile(inpFileName_[i].c_str());
      if(!toAppend){
	mydbservice->createNewIOV<ESThresholds>(myintercalib, newTime, mydbservice->endOfTime(), "ESThresholdsRcd");
      }else{
	mydbservice->appendSinceTime<ESThresholds>(myintercalib, newTime, "ESThresholdsRcd");
      } 
    } else if (objectName_[i] == "ESPedestals") {
      ESPedestals* myintercalib = readESPedestalsFromFile(inpFileName_[i].c_str());
      if(!toAppend){
	mydbservice->createNewIOV<ESPedestals>(myintercalib, newTime, mydbservice->endOfTime(), "ESPedestalsRcd");
      }else{
	mydbservice->appendSinceTime<ESPedestals>(myintercalib, newTime, "ESPedestalsRcd");
      } 
    } else if (objectName_[i] == "ESEEIntercalibConstants") {
      ESEEIntercalibConstants *myintercalib = readESEEIntercalibConstantsFromFile(inpFileName_[i].c_str());
      if(!toAppend){
        mydbservice->createNewIOV<ESEEIntercalibConstants>(myintercalib, newTime, mydbservice->endOfTime(), "ESEEIntercalibConstantsRcd");
      }else{
        mydbservice->appendSinceTime<ESEEIntercalibConstants>(myintercalib, newTime, "ESEEIntercalibConstantsRcd");
      }
    } else {
      edm::LogError("StoreESCondition")<< "Object " << objectName_[i]  << " is not supported by this program." << "\n";
    }
    // if more records write here else if .... 
    
    writeToLogFileResults(messChar);
    
    edm::LogInfo("StoreESCondition") << "Finished endJob" << "\n";
  }
  
  delete [] messChar;
}

StoreESCondition::~StoreESCondition() {

}

void StoreESCondition::analyze( const edm::Event& evt, const edm::EventSetup& evtSetup) {
}

void StoreESCondition::writeToLogFile(string a, string b, unsigned long long since) {
  
  FILE *outFile; // output log file for appending 
  outFile = fopen(logfile_.c_str(),"a");  
  if (!outFile) {
    edm::LogError("StoreESCondition") <<"*** Can not open file: " << logfile_;
    return;
  }
  char header[256];
  fillHeader(header);
  char appendMode[10];
  if (since != 0)
    strcpy(appendMode,"append");
  else
    strcpy(appendMode,"create");
  
  //fprintf(outFile, "%s %s condition from file %s written into DB for SM %d (mapped to SM %d) in %s mode (since run %u)\n", 
  //header, a.c_str(), b.c_str(),  appendMode, (unsigned int)since);
  
  fclose(outFile);           // close out file
  
}

void StoreESCondition::writeToLogFileResults(char* arg) {
  
  FILE *outFile; // output log file for appending 
  outFile = fopen(logfile_.c_str(),"a");  
  if (!outFile) {
    edm::LogError("StoreESCondition")<<"*** Can not open file: " << logfile_;
    return;
  }
  char header[256];
  fillHeader(header);
  fprintf(outFile, "%s %s\n", header,arg);
  fclose(outFile);           // close out file
}

void StoreESCondition::fillHeader(char* header) {
  time_t rawtime;
  struct tm * timeinfo;
  time ( &rawtime );
  timeinfo = localtime ( &rawtime );
  char user[50];
  strcpy(user,getlogin());
  strcpy(header,asctime(timeinfo));
  strcpy(header,user);
}

ESThresholds* StoreESCondition::readESThresholdsFromFile(const char* inputFile) {

  std::ifstream ESThresholdsFile(edm::FileInPath(inputFile).fullPath().c_str());
  float ts2, zs;   //2nd time sample, ZS threshold
  ESThresholdsFile >> ts2;
  ESThresholdsFile >> zs;
  ESThresholds* esThresholds = new ESThresholds(ts2, zs);

  return esThresholds;
}

ESEEIntercalibConstants* StoreESCondition::readESEEIntercalibConstantsFromFile(const char* inputFile) {

  std::ifstream ESEEIntercalibFile(edm::FileInPath(inputFile).fullPath().c_str());
  float gammaLow0, alphaLow0, gammaHigh0, alphaHigh0, gammaLow1, alphaLow1, gammaHigh1, alphaHigh1,
    gammaLow2, alphaLow2, gammaHigh2, alphaHigh2, gammaLow3, alphaLow3, gammaHigh3, alphaHigh3;
  //  const float ESEEIntercalibValue[16];
  //  for (int i = 0; i < 16; ++i) {
  //    ESEEIntercalibFile >> ESEEIntercalibValue[i];
  //  }
  //  ESEEIntercalibConstants* eseeIntercalibConstants = new ESEEIntercalibConstants(ESEEIntercalibValue);
  ESEEIntercalibFile >> gammaLow0;
  ESEEIntercalibFile >> alphaLow0;
  ESEEIntercalibFile >> gammaHigh0;
  ESEEIntercalibFile >> alphaHigh0;
  ESEEIntercalibFile >> gammaLow1;
  ESEEIntercalibFile >> alphaLow1;
  ESEEIntercalibFile >> gammaHigh1;
  ESEEIntercalibFile >> alphaHigh1;
  ESEEIntercalibFile >> gammaLow2;
  ESEEIntercalibFile >> alphaLow2;
  ESEEIntercalibFile >> gammaHigh2;
  ESEEIntercalibFile >> alphaHigh2;
  ESEEIntercalibFile >> gammaLow3;
  ESEEIntercalibFile >> alphaLow3;
  ESEEIntercalibFile >> gammaHigh3;
  ESEEIntercalibFile >> alphaHigh3;
  ESEEIntercalibConstants* eseeIntercalibConstants = new ESEEIntercalibConstants(gammaLow0, alphaLow0, gammaHigh0, alphaHigh0, 
										 gammaLow1, alphaLow1, gammaHigh1, alphaHigh1,
										 gammaLow2, alphaLow2, gammaHigh2, alphaHigh2, 
										 gammaLow3, alphaLow3, gammaHigh3, alphaHigh3);

  return eseeIntercalibConstants;

}

ESMissingEnergyCalibration* StoreESCondition::readESMissingEnergyFromFile(const char* inputFile) {

  std::ifstream ESMissingEnergyFile(edm::FileInPath(inputFile).fullPath().c_str());
  float ConstAEta0, ConstBEta0, ConstAEta1, ConstBEta1, ConstAEta2, ConstBEta2, ConstAEta3, ConstBEta3;
  ESMissingEnergyFile >> ConstAEta0;
  ESMissingEnergyFile >> ConstBEta0;
  ESMissingEnergyFile >> ConstAEta1;
  ESMissingEnergyFile >> ConstBEta1;
  ESMissingEnergyFile >> ConstAEta2;
  ESMissingEnergyFile >> ConstBEta2;
  ESMissingEnergyFile >> ConstAEta3;
  ESMissingEnergyFile >> ConstBEta3;
  ESMissingEnergyCalibration* esMissingEnergy = new ESMissingEnergyCalibration(ConstAEta0, ConstBEta0, ConstAEta1, ConstBEta1, 
									       ConstAEta2, ConstBEta2, ConstAEta3, ConstBEta3); 

  return esMissingEnergy;
}

ESPedestals* StoreESCondition::readESPedestalsFromFile(const char* inputFile) {

  ESPedestals* esPedestals = new ESPedestals();

  // int ped[2][2][40][40][32];
  // for (int i=0; i<2; ++i) 
  //  for (int j=0; j<2; ++j)
  //    for (int k=0; k<40; ++k)
  //	for (int m=0; m<40; ++m)
  //	  for (int n=0; n<32; ++n) 
  //	    ped[i][j][k][m][n] = 0;

  int ped[ESDetId::IZ_NUM][ESDetId::PLANE_MAX][ESDetId::IX_MAX][ESDetId::IY_MAX][ESDetId::ISTRIP_MAX]={};

  int iz, ip, ix, iy, is, ped_, zside;
  std::ifstream pedestalFile(edm::FileInPath(inputFile).fullPath().c_str());
  
  for (int i=0; i<137216; ++i) {
    pedestalFile >> iz >> ip >> ix >> iy >> is >> ped_;
    
    zside = (iz==-1) ? 1 : 0; 
    ped[zside][ip-1][ix-1][iy-1][is-1] = ped_;
  }
  
  for (int iz=-1; iz<=1; ++iz) {
    
    if (iz==0) continue;
    zside = (iz==-1) ? 1 : 0; 
    
    for (int iplane=ESDetId::PLANE_MIN; iplane<=ESDetId::PLANE_MAX; ++iplane) 
      for (int ix=ESDetId::IX_MIN; ix<=ESDetId::IX_MAX; ++ix) 
	for (int iy=ESDetId::IY_MIN; iy<=ESDetId::IY_MAX; ++iy)   
	  for (int istrip=ESDetId::ISTRIP_MIN; istrip<=ESDetId::ISTRIP_MAX; ++istrip) {
	    
	    ESPedestals::Item ESitem;
	    ESitem.mean = ped[zside][iplane-1][ix-1][iy-1][istrip-1];
	    ESitem.rms  = 3; // LG : 3, HG : 6
	    
	      if(ESDetId::validDetId(istrip, ix, iy, iplane, iz)) {
		ESDetId esId(istrip, ix, iy, iplane, iz);
		esPedestals->insert(std::make_pair(esId.rawId(), ESitem));
	      }
	  }
  }
  
  return esPedestals;
}

ESRecHitRatioCuts* StoreESCondition::readESRecHitRatioCutsFromFile(const char* inputFile) {

  std::ifstream ESRecHitRatioCutsFile(edm::FileInPath(inputFile).fullPath().c_str());

  float r12Low, r23Low, r12High, r23High;
  ESRecHitRatioCutsFile >> r12Low;
  ESRecHitRatioCutsFile >> r23Low;
  ESRecHitRatioCutsFile >> r12High;
  ESRecHitRatioCutsFile >> r23High;
  ESRecHitRatioCuts* esRecHitRatioCuts= new ESRecHitRatioCuts(r12Low, r23Low, r12High, r23High);
  //  cout<<"We are in RH ratio cut : gain : " << esgain_<<endl;

  // HG (R12Low, R23Low, R12High, R23High)
  //  ESRecHitRatioCuts* esRecHitRatioCuts;
  //if (esgain_ == 2) esRecHitRatioCuts = new ESRecHitRatioCuts(-99999., 0.24, 0.61, 2.23);
  //  if (esgain_ == 2) esRecHitRatioCuts = new ESRecHitRatioCuts(-99999., 0.61, 0.24, 2.23); // HG
  //  else esRecHitRatioCuts = new ESRecHitRatioCuts(-99999., 0.2, 0.39, 2.8);
  // LG (R12Low, R23Low, R12High, R23High) 
  //ESRecHitRatioCuts* esRecHitRatioCuts = new ESRecHitRatioCuts(-99999., 99999., -99999., 99999.); // first running 
  //ESRecHitRatioCuts* esRecHitRatioCuts = new ESRecHitRatioCuts(-99999., 0.2, 0.39, 2.8);

  return esRecHitRatioCuts;
}

ESGain* StoreESCondition::readESGainFromFile(const char* inputFile) {

  std::ifstream amplFile(edm::FileInPath(inputFile).fullPath().c_str());

  int gain;
  amplFile >> gain;
  edm::LogInfo("StoreESCondition")  << "gain : "<< gain << "\n";

  ESGain* esGain = new ESGain(gain); // 1: LG, 2: HG
  return esGain;
}

ESTimeSampleWeights* StoreESCondition::readESTimeSampleWeightsFromFile(const char* inputFile) {

  std::ifstream amplFile(edm::FileInPath(inputFile).fullPath().c_str());

  float w[3];
  for (int k = 0; k < 3; ++k) {
    float ww;
    amplFile >> ww;
    w[k] = ww;
    edm::LogInfo("StoreESCondition") <<"weight : "<<k<<" "<<w[k]<< "\n";
  }
  
  ESTimeSampleWeights* esWeights = new ESTimeSampleWeights(w[0], w[1], w[2]);
  return esWeights;

}

ESIntercalibConstants* StoreESCondition::readESIntercalibConstantsFromFile(const char* inputFile) {

  ESIntercalibConstants* ical = new ESIntercalibConstants();

  std::ifstream mipFile(edm::FileInPath(inputFile).fullPath().c_str());

  for (int i=0; i<137216; ++i) {
    int iz, ip, ix, iy, is;
    double mip;
    mipFile >> iz >> ip >> ix >> iy >> is >> mip;
    //if (mip <20 || mip> 70) cout<<iz<<" "<<ip<<" "<<ix<<" "<<iy<<" "<<is<<" "<<mip<<endl; // HG
    // LG : HG MIP/6/1.14
    //mip = mip/6/1.14;
    // LG : HG MIP/6
    if (esgain_ == 1) mip = mip/6.; // LG
    if (mip <20 || mip> 70) edm::LogInfo("StoreESCondition") <<iz<<" "<<ip<<" "<<ix<<" "<<iy<<" "<<is<<" "<<mip<< "\n"; // LG

    if(ESDetId::validDetId(is, ix, iy, ip, iz)) {
      ESDetId esId(is, ix, iy, ip, iz);
      ical->setValue(esId.rawId(), mip);
    }
  }
  
  return ical;
}

ESChannelStatus* StoreESCondition::readESChannelStatusFromFile(const char* inputFile) {
  

  int z[1000], p[1000], x[1000], y[1000], nsensors;
  std::ifstream statusFile(edm::FileInPath(inputFile).fullPath().c_str());
  statusFile >> nsensors;
  edm::LogInfo("StoreESCondition")  << " nsensors " << nsensors << "\n";
  if(nsensors >= 1000) {
    edm::LogInfo("StoreESCondition")  << " *** value too high, modify the method!***" << "\n";
    exit(-1);
  }
  for (int i = 0; i < nsensors; ++i) {
    statusFile >> z[i] >> p[i] >> x[i] >> y[i];
  }
  ESChannelStatus* ecalStatus = new ESChannelStatus();
  int Nbstatus = 0, Nbstrip = 0;
  for (int istrip=ESDetId::ISTRIP_MIN;istrip<=ESDetId::ISTRIP_MAX;istrip++) {
    for (int ix=ESDetId::IX_MIN;ix<=ESDetId::IX_MAX;ix++) {
      for (int iy=ESDetId::IY_MIN;iy<=ESDetId::IY_MAX;iy++) {
	for (int iplane = 1; iplane <= 2; iplane++) {
	  for (int izeta = -1; izeta <= 1; izeta = izeta + 2) {
	    //	    if (izeta==0) continue;
	    //	    try {

	      //ESDetId Plane iplane Zside izeta
	    //	    if(!ESDetId::validDetId(istrip,ix,iy,iplane,izeta)) cout << " Unvalid DetId" << endl;
	    //	    else {
	    if(ESDetId::validDetId(istrip,ix,iy,iplane,izeta)) {
	      ESDetId anESId(istrip,ix,iy,iplane,izeta);
	      int status = 0;
	      //	      std::ifstream statusFile(edm::FileInPath(inputFile).fullPath().c_str());
	      Nbstrip++;
	      for (int i=0; i<nsensors; ++i) {
		if (izeta == z[i] && iplane == p[i] && ix == x[i] && iy == y[i]) status = 1; 
	      }
	      if(status == 1) {
		Nbstatus++;
		if(istrip == 1) edm::LogInfo("StoreESCondition") << " Bad channel ix " << ix << " iy " << iy << " iplane " << iplane << " iz " << izeta << "\n";  // print only once
	      }
	      ecalStatus->setValue(anESId, status);
	      //	      statusFile.close();
	    }    // valid DetId
	  //	    catch ( cms::Exception &e ) { }
	  }   // loop over z
	}    //  loop over plane
      }     //   loop over y
    }      //    loop over x
  }       //     loop over strips
  edm::LogInfo("StoreESCondition") << " Nb of strips " << Nbstrip << " Number of bad channels " << Nbstatus << "\n";
  statusFile.close();

  // overwrite the statuses which are in the file                                                                                
  return ecalStatus;
  
}

