#include "CondTools/Ecal/plugins/StoreEcalCondition.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <fstream>
#include <string>
#include <cstring>
#include <ctime>
#include <unistd.h>

using std::string;

StoreEcalCondition::StoreEcalCondition(const edm::ParameterSet& iConfig) {
  prog_name_ = "StoreEcalCondition";

  logfile_ = iConfig.getParameter<std::string>("logfile");
  sm_slot_ = iConfig.getUntrackedParameter<unsigned int>("smSlot", 1);

  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters toPut = iConfig.getParameter<Parameters>("toPut");
  for (Parameters::iterator itToPut = toPut.begin(); itToPut != toPut.end(); ++itToPut) {
    inpFileName_.push_back(itToPut->getUntrackedParameter<std::string>("inputFile"));
    inpFileNameEE_.push_back(itToPut->getUntrackedParameter<std::string>("inputFileEE"));
    objectName_.push_back(itToPut->getUntrackedParameter<std::string>("conditionType"));
    since_.push_back(itToPut->getUntrackedParameter<unsigned int>("since"));
  }

  sm_constr_ = -1;
}

void StoreEcalCondition::endJob() {
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    edm::LogError("StoreEcalCondition") << "PoolDBOutputService is unavailable";
    return;
  }

  bool toAppend = false;
  // copy a string to the char *
  std::string message = "finished OK\n";
  size_t messageSize = message.size() + 1;
  char* messChar = new char[messageSize];
  strncpy(messChar, message.c_str(), messageSize);

  for (unsigned int i = 0; i < objectName_.size(); i++) {
    cond::Time_t newTime;

    if (mydbservice->isNewTagRequest(objectName_[i] + std::string("Rcd"))) {
      // This is the first object for this tag.
      // Append mode should be off.
      // newTime is the end of this new objects IOV.
      newTime = mydbservice->beginOfTime();
    } else {
      // There should already be an object in the DB for this tag.
      // Append IOV mode should be on.
      // newTime is the beginning of this new objects IOV.
      toAppend = true;
      newTime = (cond::Time_t)since_[i];
    }
    edm::LogInfo("StoreEcalCondition") << "Reading " << objectName_[i] << " from file and writing to DB with newTime "
                                       << newTime;

    if (objectName_[i] == "EcalWeightXtalGroups") {
      const auto mycali = readEcalWeightXtalGroupsFromFile(inpFileName_[i].c_str());
      if (!toAppend) {
        mydbservice->createOneIOV<EcalWeightXtalGroups>(*mycali, newTime, "EcalWeightXtalGroupsRcd");
      } else {
        mydbservice->appendOneIOV<EcalWeightXtalGroups>(*mycali, newTime, "EcalWeightXtalGroupsRcd");
      }
    } else if (objectName_[i] == "EcalTBWeights") {
      const auto mycali = readEcalTBWeightsFromFile(inpFileName_[i].c_str());
      if (!toAppend) {
        mydbservice->createOneIOV<EcalTBWeights>(*mycali, newTime, "EcalTBWeightsRcd");
      } else {
        mydbservice->appendOneIOV<EcalTBWeights>(*mycali, newTime, "EcalTBWeightsRcd");
      }
    } else if (objectName_[i] == "EcalADCToGeVConstant") {
      const auto mycali = readEcalADCToGeVConstantFromFile(inpFileName_[i].c_str());
      if (!toAppend) {
        mydbservice->createOneIOV<EcalADCToGeVConstant>(*mycali, newTime, "EcalADCToGeVConstantRcd");
      } else {
        mydbservice->appendOneIOV<EcalADCToGeVConstant>(*mycali, newTime, "EcalADCToGeVConstantRcd");
      }
    } else if (objectName_[i] == "EcalIntercalibConstants") {
      const auto mycali = readEcalIntercalibConstantsFromFile(inpFileName_[i].c_str(), inpFileNameEE_[i].c_str());
      if (!toAppend) {
        mydbservice->createOneIOV<EcalIntercalibConstants>(*mycali, newTime, "EcalIntercalibConstantsRcd");
      } else {
        mydbservice->appendOneIOV<EcalIntercalibConstants>(*mycali, newTime, "EcalIntercalibConstantsRcd");
      }
    } else if (objectName_[i] == "EcalPFRecHitThresholds") {
      const auto mycali = readEcalPFRecHitThresholdsFromFile(inpFileName_[i].c_str(), inpFileNameEE_[i].c_str());
      if (!toAppend) {
        mydbservice->createOneIOV<EcalPFRecHitThresholds>(*mycali, newTime, "EcalPFRecHitThresholdsRcd");
      } else {
        mydbservice->appendOneIOV<EcalPFRecHitThresholds>(*mycali, newTime, "EcalPFRecHitThresholdsRcd");
      }
    } else if (objectName_[i] == "EcalIntercalibConstantsMC") {
      const auto mycali = readEcalIntercalibConstantsMCFromFile(inpFileName_[i].c_str(), inpFileNameEE_[i].c_str());
      if (!toAppend) {
        mydbservice->createOneIOV<EcalIntercalibConstantsMC>(*mycali, newTime, "EcalIntercalibConstantsMCRcd");
      } else {
        mydbservice->appendOneIOV<EcalIntercalibConstantsMC>(*mycali, newTime, "EcalIntercalibConstantsMCRcd");
      }
    } else if (objectName_[i] == "EcalGainRatios") {
      const auto mycali = readEcalGainRatiosFromFile(inpFileName_[i].c_str());
      if (!toAppend) {
        mydbservice->createOneIOV<EcalGainRatios>(*mycali, newTime, "EcalGainRatiosRcd");
      } else {
        mydbservice->appendOneIOV<EcalGainRatios>(*mycali, newTime, "EcalGainRatiosRcd");
      }
    } else if (objectName_[i] == "EcalChannelStatus") {
      const auto mycali = readEcalChannelStatusFromFile(inpFileName_[i].c_str());
      if (!toAppend) {
        mydbservice->createOneIOV<EcalChannelStatus>(*mycali, newTime, "EcalChannelStatusRcd");
      } else {
        mydbservice->appendOneIOV<EcalChannelStatus>(*mycali, newTime, "EcalChannelStatusRcd");
      }
    } else {
      edm::LogError("StoreEcalCondition") << "Object " << objectName_[i] << " is not supported by this program.";
    }

    //      writeToLogFile(objectName_[i], inpFileName_[i], since_[i]);
    //writeToLogFileResults("finished OK\n");
    writeToLogFileResults(messChar);

    edm::LogInfo("StoreEcalCondition") << "Finished endJob";
  }

  delete[] messChar;
}

StoreEcalCondition::~StoreEcalCondition() {}

//-------------------------------------------------------------
void StoreEcalCondition::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  //-------------------------------------------------------------
}

//------------------------------------------------------------
void StoreEcalCondition::writeToLogFile(string a, string b, unsigned long long since) {
  //-------------------------------------------------------------

  FILE* outFile;  // output log file for appending
  outFile = fopen(logfile_.c_str(), "a");
  if (!outFile) {
    edm::LogError("StoreEcalCondition") << "*** Can not open file: " << logfile_;
    return;
  }
  char header[256];
  fillHeader(header);
  char appendMode[10];
  if (since != 0)
    sprintf(appendMode, "append");
  else
    sprintf(appendMode, "create");

  fprintf(outFile,
          "%s %s condition from file %s written into DB for SM %d (mapped to SM %d) in %s mode (since run %u)\n",
          header,
          a.c_str(),
          b.c_str(),
          sm_constr_,
          sm_slot_,
          appendMode,
          (unsigned int)since);

  fclose(outFile);  // close out file
}
//------------------------------------------------------------
void StoreEcalCondition::writeToLogFileResults(char* arg) {
  //-------------------------------------------------------------

  FILE* outFile;  // output log file for appending
  outFile = fopen(logfile_.c_str(), "a");
  if (!outFile) {
    edm::LogError("StoreEcalCondition") << "*** Can not open file: " << logfile_;
    return;
  }
  char header[256];
  fillHeader(header);
  fprintf(outFile, "%s %s\n", header, arg);
  fclose(outFile);  // close out file
}

//------------------------------------------------------------
void StoreEcalCondition::fillHeader(char* header)
//------------------------------------------------------------
{
  time_t rawtime;
  struct tm* timeinfo;
  time(&rawtime);
  timeinfo = localtime(&rawtime);
  char user[50];
  sprintf(user, "%s", getlogin());
  sprintf(header, "%s %s:", asctime(timeinfo), user);
}

/*
 * Format for ASCII groups & weights file
 * Updated by Alex Zabi Imperial College
 * 03/07/06: implementing final weights format
 * Accepted format:
 * groupId nSamples nTDCbins (introductory line)
 *  and then nTDCbins x (3 + nSamples) lines of nSamples numbers containing
 *  For TDCbin1
 *  gain 12 weights (before gain switch)
 *  ===========================================
 *  ampWeight[0] ampWeight[1] .............                           |
 *  pedWeight[0] pedWeight[1] .............                           |
 *  jitWeight[0] jitWeight[1] .............                           |
 *  chi2Matri[0][0] chi2Matrix[0][1] ..........                       |
 *  chi2Matrix[1][0] chi2Matrix[1][1] ..........                      |
 *   .....                                                            |
 *  chi2Matrix[nsamples-1][0] chi2Matrix[nsamples-1][1] ..........    |
 *  gain 6 and 1 weights (after gain switch)
 *  ===========================================
 *  ampWeight[0] ampWeight[1] .............                           |
 *  pedWeight[0] pedWeight[1] .............                           |
 *  jitWeight[0] jitWeight[1] .............                           |
 *  chi2Matri[0][0] chi2Matrix[0][1] ..........                       |
 *  chi2Matrix[1][0] chi2Matrix[1][1] ..........                      |
 *   .....                                                            |
 *  chi2Matrix[nsamples-1][0] chi2Matrix[nsamples-1][1] ..........    |
 *  ===========================================
 *  For TDCbin nTDCBins
 *  ............
 */

//-------------------------------------------------------------
std::shared_ptr<EcalWeightXtalGroups> StoreEcalCondition::readEcalWeightXtalGroupsFromFile(const char* inputFile) {
  //-------------------------------------------------------------

  // Code taken from EcalWeightTools/test/MakeOfflineDbFromAscii.cpp

  auto xtalGroups = std::make_shared<EcalWeightXtalGroups>();
  std::ifstream groupid_in(inputFile);

  if (!groupid_in.is_open()) {
    edm::LogError("StoreEcalCondition") << "*** Can not open file: " << inputFile;
    return nullptr;
  }

  int smnumber = -99999;

  std::ostringstream str;
  groupid_in >> smnumber;
  if (smnumber == -99999) {
    edm::LogError("StoreEcalCondition") << "ERROR: SM number not found in file";
    return nullptr;
  }
  str << "sm= " << smnumber << "\n";
  sm_constr_ = smnumber;

  char temp[256];
  //Reading the other 5 header lines containing various informations
  for (int i = 0; i <= 5; i++) {
    groupid_in.getline(temp, 255);
    str << temp << "\n";
  }

  // Skip the nGroup/Mean line
  groupid_in.getline(temp, 255);
  str << temp << "\n";

  edm::LogInfo("StoreEcalCondition") << "GROUPID file " << str.str();

  int xtals = 0;
  int xtal, ietaf, iphif, groupID;
  while (groupid_in.good()) {
    groupid_in >> xtal >> ietaf >> iphif >> groupID;
    if (groupid_in.eof()) {
      break;
    }

    LogDebug("StoreEcalCondition") << "XTAL=" << xtal << " ETA=" << ietaf << " PHI=" << iphif << " GROUP=" << groupID;

    //EBDetId ebid(ieta,iphi);
    EBDetId ebid(sm_slot_, xtal, EBDetId::SMCRYSTALMODE);
    // xtalGroups->setValue(ebid.rawId(), EcalXtalGroupId( ebid.hashedIndex()) );
    xtalGroups->setValue(ebid.rawId(), EcalXtalGroupId(groupID));
    xtals++;
  }  //loop iphi

  if (xtals != 1700) {
    edm::LogError("StoreEcalCondition") << "ERROR:  GROUPID file did not contain data for 1700 crystals";
    return nullptr;
  }

  edm::LogInfo("StoreEcalCondition") << "Groups for " << xtals << " xtals written into DB";
  sm_constr_ = smnumber;

  return xtalGroups;
}

//-------------------------------------------------------------
std::shared_ptr<EcalTBWeights> StoreEcalCondition::readEcalTBWeightsFromFile(const char* inputFile) {
  //-------------------------------------------------------------

  // Zabi code to be written here

  auto tbwgt = std::make_shared<EcalTBWeights>();

  std::ifstream WeightsFileTB(inputFile);
  if (!WeightsFileTB.is_open()) {
    edm::LogError("StoreEcalCondition") << "*** Can not open file: " << inputFile;
    return nullptr;
  }

  int smnumber = -99999;

  std::ostringstream str;
  WeightsFileTB >> smnumber;
  if (smnumber == -99999)
    return nullptr;

  str << "sm= " << smnumber << "\n";

  char temp[256];
  //Reading the other 5 header lines containing various informations
  for (int i = 0; i <= 5; i++) {
    WeightsFileTB.getline(temp, 255);
    str << temp << "\n";
  }

  edm::LogInfo("StoreEcalCondition") << "Weights file " << str.str();

  int ngroups = 0;
  while (WeightsFileTB.good()) {
    int igroup_ID = -99999;
    int nSamples = -99999;
    int nTdcBins = -99999;

    WeightsFileTB >> igroup_ID >> nSamples >> nTdcBins;
    if (igroup_ID == -99999 || nSamples == -99999 || nTdcBins == -99999)
      break;

    std::ostringstream str;
    str << "Igroup=" << igroup_ID << " Nsamples=" << nSamples << " NTdcBins=" << nTdcBins << "\n";

    for (int iTdcBin = 0; iTdcBin < nTdcBins; iTdcBin++) {
      EcalWeightSet wgt;  // one set of weights
      EcalWeightSet::EcalWeightMatrix& wgt1 = wgt.getWeightsBeforeGainSwitch();
      EcalWeightSet::EcalWeightMatrix& wgt2 = wgt.getWeightsAfterGainSwitch();
      EcalWeightSet::EcalChi2WeightMatrix& chisq1 = wgt.getChi2WeightsBeforeGainSwitch();
      EcalWeightSet::EcalChi2WeightMatrix& chisq2 = wgt.getChi2WeightsAfterGainSwitch();

      // 	std::vector<EcalWeight> wamp, wped, wtime; //weights before gain switch
      // 	std::vector<EcalWeight> wamp2, wped2, wtime2; //weights after gain switch

      //WEIGHTS BEFORE GAIN SWITCH
      //Amplitude weights
      for (int j = 0; j < nSamples; ++j) {
        double ww = 0.0;
        WeightsFileTB >> ww;
        wgt1(0, j) = ww;
        str << ww << " ";
      }  // loop Samples
      str << "\n";

      //Pedestal weights
      for (int j = 0; j < nSamples; ++j) {
        double ww = 0.0;
        WeightsFileTB >> ww;
        wgt1(1, j) = ww;
        str << ww << " ";
      }  //loop Samples
      str << "\n";

      //Timing weights
      for (int j = 0; j < nSamples; ++j) {
        double ww = 0.0;
        WeightsFileTB >> ww;
        wgt1(2, j) = ww;
        str << ww << " ";
      }  //loop Samples
      str << "\n";

      for (int j = 0; j < nSamples; ++j) {
        // fill chi2 matrix
        //std::vector<EcalWeight> vChi2; // row of chi2 matrix
        for (int k = 0; k < nSamples; ++k) {
          double ww = 0.0;
          WeightsFileTB >> ww;
          chisq1(j, k) = ww;
          str << ww << " ";
        }  //loop samples
        str << "\n";
      }  //loop lines

      //WEIGHTS AFTER GAIN SWITCH
      for (int j = 0; j < nSamples; ++j) {
        double ww = 0.0;
        WeightsFileTB >> ww;
        wgt2(0, j) = ww;
        str << ww << " ";
      }  // loop Samples
      str << "\n";

      //Pedestal weights
      for (int j = 0; j < nSamples; ++j) {
        double ww = 0.0;
        WeightsFileTB >> ww;
        wgt2(1, j) = ww;
        str << ww << " ";
      }  //loop Samples
      str << "\n";

      //Timing weights
      for (int j = 0; j < nSamples; ++j) {
        double ww = 0.0;
        WeightsFileTB >> ww;
        wgt2(2, j) = ww;
        str << ww << " ";
      }  //loop Samples
      str << "\n";

      for (int j = 0; j < nSamples; ++j) {
        // fill chi2 matrix
        //std::vector<EcalWeight> vChi2; // row of chi2 matrix
        for (int k = 0; k < nSamples; ++k) {
          double ww = 0.0;
          WeightsFileTB >> ww;
          chisq2(j, k) = ww;
          str << ww << " ";
        }  //loop samples
        str << "\n";
      }  //loop lines

      LogDebug("StoreEcalCondition") << str.str();

      //modif-27-07-06 tdcid should start from 1
      tbwgt->setValue(std::make_pair(igroup_ID, iTdcBin + 1), wgt);
    }  //loop Tdc bins
    ngroups++;
  }  //loop groupID

  sm_constr_ = smnumber;

  edm::LogInfo("StoreEcalCondition") << "Weights for " << ngroups << " groups written into DB";
  return tbwgt;
}

//-------------------------------------------------------------
std::shared_ptr<EcalADCToGeVConstant> StoreEcalCondition::readEcalADCToGeVConstantFromFile(const char* inputFile) {
  //-------------------------------------------------------------

  FILE* inpFile;  // input file
  inpFile = fopen(inputFile, "r");
  if (!inpFile) {
    edm::LogError("StoreEcalCondition") << "*** Can not open file: " << inputFile;
    return nullptr;
  }

  char line[256];

  std::ostringstream str;

  fgets(line, 255, inpFile);
  int sm_number = atoi(line);
  str << "sm= " << sm_number << "\n";

  fgets(line, 255, inpFile);
  //int nevents=atoi(line); // not necessary here just for online conddb

  fgets(line, 255, inpFile);
  string gen_tag = to_string(line);
  str << "gen tag " << gen_tag << "\n";  // should I use this?

  fgets(line, 255, inpFile);
  string cali_method = to_string(line);
  str << "cali method " << cali_method << "\n";  // not important

  fgets(line, 255, inpFile);
  string cali_version = to_string(line);
  str << "cali version " << cali_version << "\n";  // not important

  fgets(line, 255, inpFile);
  string cali_type = to_string(line);
  str << "cali type " << cali_type << "\n";  // not important

  edm::LogInfo("StoreEcalCondition") << "ADCToGeV file " << str.str();

  fgets(line, 255, inpFile);
  float adc_to_gev = 0;
  sscanf(line, "%f", &adc_to_gev);
  LogDebug("StoreEcalCondition") << " calib=" << adc_to_gev;
  fgets(line, 255, inpFile);
  float adc_to_gev_ee = 0;
  sscanf(line, "%f", &adc_to_gev_ee);
  LogDebug("StoreEcalCondition") << " calib=" << adc_to_gev_ee;

  fclose(inpFile);  // close inp. file

  sm_constr_ = sm_number;

  // barrel and endcaps the same
  auto agc = std::make_shared<EcalADCToGeVConstant>(adc_to_gev, adc_to_gev_ee);
  edm::LogInfo("StoreEcalCondition") << "ADCtoGeV scale written into the DB";
  return agc;
}

//-------------------------------------------------------------
std::shared_ptr<EcalPFRecHitThresholds> StoreEcalCondition::readEcalPFRecHitThresholdsFromFile(
    const char* inputFile, const char* inputFileEE) {
  //-------------------------------------------------------------

  auto ical = std::make_shared<EcalPFRecHitThresholds>();

  FILE* inpFile;  // input file
  inpFile = fopen(inputFile, "r");
  if (!inpFile) {
    edm::LogError("StoreEcalCondition") << "*** Can not open file: " << inputFile;
    return nullptr;
  }

  char line[256];

  int ieta = 0;
  int iphi = 0;
  int ix = 0;
  int iy = 0;
  int iz = 0;

  float thresh = 0;

  int ii = 0;
  while (fgets(line, 255, inpFile)) {
    sscanf(line, "%d %d %f ", &ieta, &iphi, &thresh);
    if (ii == 0)
      edm::LogVerbatim("StoreEcalCondition") << "crystal " << ieta << "/" << iphi << " Thresh= " << thresh << "\n";

    if (EBDetId::validDetId(ieta, iphi)) {
      EBDetId ebid(ieta, iphi);
      ical->setValue(ebid.rawId(), thresh);
      ii++;
    }
  }

  //    inf.close();           // close inp. file
  fclose(inpFile);  // close inp. file

  edm::LogInfo("StoreEcalCondition") << "Read PF RecHits for " << ii << " xtals ";

  FILE* inpFileEE;  // input file
  inpFileEE = fopen(inputFileEE, "r");
  if (!inpFileEE) {
    edm::LogError("StoreEcalCondition") << "*** Can not open file: " << inputFileEE;
    return nullptr;
  }
  ii = 0;
  while (fgets(line, 255, inpFileEE)) {
    sscanf(line, "%d %d %d %f ", &ix, &iy, &iz, &thresh);
    if (ii == 0)
      edm::LogVerbatim("StoreEcalCondition")
          << "crystal " << ix << "/" << iy << "/" << iz << " Thresh= " << thresh << "\n";
    if (EEDetId::validDetId(ix, iy, iz)) {
      EEDetId eeid(ix, iy, iz);
      ical->setValue(eeid.rawId(), thresh);
      ii++;
    }
  }

  //    inf.close();           // close inp. file
  fclose(inpFileEE);  // close inp. file

  edm::LogInfo("StoreEcalCondition") << "loop on EE channels done - number of crystals =" << ii;

  return ical;
}

//-------------------------------------------------------------
std::shared_ptr<EcalIntercalibConstants> StoreEcalCondition::readEcalIntercalibConstantsFromFile(
    const char* inputFile, const char* inputFileEE) {
  //-------------------------------------------------------------

  auto ical = std::make_shared<EcalIntercalibConstants>();

  FILE* inpFile;  // input file
  inpFile = fopen(inputFile, "r");
  if (!inpFile) {
    edm::LogError("StoreEcalCondition") << "*** Can not open file: " << inputFile;
    return nullptr;
  }

  char line[256];

  std::ostringstream str;

  fgets(line, 255, inpFile);
  string sm_or_all = to_string(line);
  int sm_number = 0;
  int nchan = 1700;
  sm_number = atoi(line);
  str << "sm= " << sm_number << "\n";
  if (sm_number != -1) {
    nchan = 1700;
  } else {
    nchan = 61200;
  }

  fgets(line, 255, inpFile);
  //int nevents=atoi(line); // not necessary here just for online conddb

  fgets(line, 255, inpFile);
  string gen_tag = to_string(line);
  str << "gen tag " << gen_tag << "\n";  // should I use this?

  fgets(line, 255, inpFile);
  string cali_method = to_string(line);
  str << "cali method " << cali_method << "\n";  // not important

  fgets(line, 255, inpFile);
  string cali_version = to_string(line);
  str << "cali version " << cali_version << "\n";  // not important

  fgets(line, 255, inpFile);
  string cali_type = to_string(line);
  str << "cali type " << cali_type << "\n";  // not important

  edm::LogInfo("StoreEcalCondition") << "Intercalibration file " << str.str();

  int sm_num[61200] = {0};
  int cry_num[61200] = {0};
  float calib[61200] = {0};
  float calib_rms[61200] = {0};
  int calib_nevents[61200] = {0};
  int calib_status[61200] = {0};

  int ii = 0;
  if (sm_number != -1) {
    while (fgets(line, 255, inpFile)) {
      sscanf(line, "%d %f %f %d %d", &cry_num[ii], &calib[ii], &calib_rms[ii], &calib_nevents[ii], &calib_status[ii]);
      //       if(ii<10) { // print out only the first ten channels
      // 	cout << "cry="<<cry_num[ii]<<" calib="<<calib[ii]<<endl;
      //       }
      sm_num[ii] = sm_number;
      ii++;
    }
  } else {
    // this is for the whole Barrel
    edm::LogInfo("StoreEcalCondition") << "mode ALL BARREL";
    while (fgets(line, 255, inpFile)) {
      sscanf(line, "%d %d %f %f %d", &sm_num[ii], &cry_num[ii], &calib[ii], &calib_rms[ii], &calib_status[ii]);
      if (ii == 0)
        edm::LogVerbatim("StoreEcalCondition")
            << "crystal " << cry_num[ii] << " of sm " << sm_num[ii] << " cali= " << calib[ii] << "\n";
      ii++;
    }
  }

  //    inf.close();           // close inp. file
  fclose(inpFile);  // close inp. file

  edm::LogInfo("StoreEcalCondition") << "Read intercalibrations for " << ii << " xtals ";

  if (ii != nchan)
    edm::LogWarning("StoreEcalCondition") << "Some crystals missing. Missing channels will be set to 0";

  // Get channel ID

  sm_constr_ = sm_number;

  // Set the data
  for (int i = 0; i < nchan; i++) {
    // EBDetId(int index1, int index2, int mode = ETAPHIMODE)
    // sm and crys index SMCRYSTALMODE index1 is SM index2 is crystal number a la H4

    int slot_num = convertFromConstructionSMToSlot(sm_num[i], -1);
    EBDetId ebid(slot_num, cry_num[i], EBDetId::SMCRYSTALMODE);

    ical->setValue(ebid.rawId(), calib[i]);

    if (i == 0)
      edm::LogVerbatim("StoreEcalCondition") << "crystal " << cry_num[i] << " of sm " << sm_num[i] << " in slot "
                                             << slot_num << " calib= " << calib[i] << "\n";

  }  // loop over channels

  edm::LogInfo("StoreEcalCondition") << "loop on channels done";

  FILE* inpFileEE;  // input file
  inpFileEE = fopen(inputFileEE, "r");
  if (!inpFileEE) {
    edm::LogError("StoreEcalCondition") << "*** Can not open file: " << inputFile;

    // dummy endcap data

    for (int iX = EEDetId::IX_MIN; iX <= EEDetId::IX_MAX; ++iX) {
      for (int iY = EEDetId::IY_MIN; iY <= EEDetId::IY_MAX; ++iY) {
        // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
        if (EEDetId::validDetId(iX, iY, 1)) {
          EEDetId eedetidpos(iX, iY, 1);
          ical->setValue(eedetidpos, 1.0);
        }
        if (EEDetId::validDetId(iX, iY, -1)) {
          EEDetId eedetidneg(iX, iY, -1);
          ical->setValue(eedetidneg, 1.0);
        }
      }
    }

  } else {
    edm::LogInfo("StoreEcalCondition") << "... now reading EE file ...";

    int ii = 0;
    while (fgets(line, 255, inpFileEE)) {
      int iz, ix, iy;
      float calibee;
      sscanf(line, "%d %d %d %f", &iz, &ix, &iy, &calibee);
      if (ii <= 0)
        edm::LogVerbatim("StoreEcalCondition")
            << "crystal " << iz << "/" << ix << "/" << iy << " cali=" << calibee << "\n";

      if (EEDetId::validDetId(ix, iy, iz)) {
        EEDetId eedetid(ix, iy, iz);
        ical->setValue(eedetid, calibee);
      }

      ii++;
    }

    fclose(inpFileEE);  // close inp. file
  }

  edm::LogInfo("StoreEcalCondition") << "loop on EE channels done";

  return ical;
}

//-------------------------------------------------------------
std::shared_ptr<EcalIntercalibConstantsMC> StoreEcalCondition::readEcalIntercalibConstantsMCFromFile(
    const char* inputFile, const char* inputFileEE) {
  //-------------------------------------------------------------

  auto ical = std::make_shared<EcalIntercalibConstantsMC>();

  FILE* inpFile;  // input file
  inpFile = fopen(inputFile, "r");
  if (!inpFile) {
    edm::LogError("StoreEcalCondition") << "*** Can not open file: " << inputFile;
    return nullptr;
  }

  char line[256];

  std::ostringstream str;

  fgets(line, 255, inpFile);
  string sm_or_all = to_string(line);
  int sm_number = 0;
  int nchan = 1700;
  sm_number = atoi(line);
  str << "sm= " << sm_number << "\n";
  if (sm_number != -1) {
    nchan = 1700;
  } else {
    nchan = 61200;
  }

  fgets(line, 255, inpFile);
  //int nevents=atoi(line); // not necessary here just for online conddb

  fgets(line, 255, inpFile);
  string gen_tag = to_string(line);
  str << "gen tag " << gen_tag << "\n";  // should I use this?

  fgets(line, 255, inpFile);
  string cali_method = to_string(line);
  str << "cali method " << cali_method << "\n";  // not important

  fgets(line, 255, inpFile);
  string cali_version = to_string(line);
  str << "cali version " << cali_version << "\n";  // not important

  fgets(line, 255, inpFile);
  string cali_type = to_string(line);
  str << "cali type " << cali_type << "\n";  // not important

  edm::LogInfo("StoreEcalCondition") << "Intercalibration file " << str.str();

  int sm_num[61200] = {0};
  int cry_num[61200] = {0};
  float calib[61200] = {0};
  float calib_rms[61200] = {0};
  int calib_nevents[61200] = {0};
  int calib_status[61200] = {0};

  int ii = 0;
  if (sm_number != -1) {
    while (fgets(line, 255, inpFile)) {
      sscanf(line, "%d %f %f %d %d", &cry_num[ii], &calib[ii], &calib_rms[ii], &calib_nevents[ii], &calib_status[ii]);
      //       if(ii<10) { // print out only the first ten channels
      // 	cout << "cry="<<cry_num[ii]<<" calib="<<calib[ii]<<endl;
      //       }
      sm_num[ii] = sm_number;
      ii++;
    }
  } else {
    // this is for the whole Barrel
    edm::LogInfo("StoreEcalCondition") << "mode ALL BARREL";
    while (fgets(line, 255, inpFile)) {
      sscanf(line, "%d %d %f %f %d", &sm_num[ii], &cry_num[ii], &calib[ii], &calib_rms[ii], &calib_status[ii]);
      if (ii == 0)
        edm::LogVerbatim("StoreEcalCondition")
            << "crystal " << cry_num[ii] << " of sm " << sm_num[ii] << " cali= " << calib[ii] << "\n";
      ii++;
    }
  }

  //    inf.close();           // close inp. file
  fclose(inpFile);  // close inp. file

  edm::LogInfo("StoreEcalCondition") << "Read intercalibrations for " << ii << " xtals ";

  if (ii != nchan)
    edm::LogWarning("StoreEcalCondition") << "Some crystals missing. Missing channels will be set to 0";

  // Get channel ID

  sm_constr_ = sm_number;

  // Set the data
  for (int i = 0; i < nchan; i++) {
    // EBDetId(int index1, int index2, int mode = ETAPHIMODE)
    // sm and crys index SMCRYSTALMODE index1 is SM index2 is crystal number a la H4

    int slot_num = convertFromConstructionSMToSlot(sm_num[i], -1);
    EBDetId ebid(slot_num, cry_num[i], EBDetId::SMCRYSTALMODE);

    ical->setValue(ebid.rawId(), calib[i]);

    if (i == 0)
      edm::LogVerbatim("StoreEcalCondition") << "crystal " << cry_num[i] << " of sm " << sm_num[i] << " in slot "
                                             << slot_num << " calib= " << calib[i] << "\n";

  }  // loop over channels

  edm::LogInfo("StoreEcalCondition") << "loop on channels done";

  FILE* inpFileEE;  // input file
  inpFileEE = fopen(inputFileEE, "r");
  if (!inpFileEE) {
    edm::LogError("StoreEcalCondition") << "*** Can not open file: " << inputFile;

    // dummy endcap data

    for (int iX = EEDetId::IX_MIN; iX <= EEDetId::IX_MAX; ++iX) {
      for (int iY = EEDetId::IY_MIN; iY <= EEDetId::IY_MAX; ++iY) {
        // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
        if (EEDetId::validDetId(iX, iY, 1)) {
          EEDetId eedetidpos(iX, iY, 1);
          ical->setValue(eedetidpos, 1.0);
        }
        if (EEDetId::validDetId(iX, iY, -1)) {
          EEDetId eedetidneg(iX, iY, -1);
          ical->setValue(eedetidneg, 1.0);
        }
      }
    }

  } else {
    edm::LogInfo("StoreEcalCondition") << "... now reading EE file ...";

    int ii = 0;
    while (fgets(line, 255, inpFileEE)) {
      int iz, ix, iy;
      float calibee;
      sscanf(line, "%d %d %d %f", &iz, &ix, &iy, &calibee);
      if (ii <= 0)
        edm::LogVerbatim("StoreEcalCondition")
            << "crystal " << iz << "/" << ix << "/" << iy << " cali=" << calibee << "\n";

      if (EEDetId::validDetId(ix, iy, iz)) {
        EEDetId eedetid(ix, iy, iz);
        ical->setValue(eedetid, calibee);
      }

      ii++;
    }

    fclose(inpFileEE);  // close inp. file
  }

  edm::LogInfo("StoreEcalCondition") << "loop on EE channels done";

  return ical;
}

//-------------------------------------------------------------
int StoreEcalCondition::convertFromConstructionSMToSlot(int sm_constr, int sm_slot) {
  // input either cosntruction number or slot number and returns the other
  // the slots are numbered first EB+ slot 1 ...18 then EB- 1... 18
  // the slots start at 1 and the SM start at 0
  //-------------------------------------------------------------
  int slot_to_constr[37] = {-1, 12, 17, 10, 1,  8,  4, 27, 20, 23, 25, 6,  34, 35, 15, 18, 30, 21, 9,
                            24, 22, 13, 31, 26, 16, 2, 11, 5,  0,  29, 28, 14, 33, 32, 3,  7,  19};
  int constr_to_slot[36] = {28, 4,  25, 34, 6,  27, 11, 35, 5,  18, 3,  26, 1,  21, 31, 14, 24, 2,
                            15, 36, 8,  17, 20, 9,  19, 10, 23, 7,  30, 29, 16, 22, 33, 32, 12, 13};

  int result = 0;
  if (sm_constr != -1) {
    result = constr_to_slot[sm_constr];
  } else if (sm_slot != -1) {
    result = slot_to_constr[sm_slot];
  }
  return result;
}

//-------------------------------------------------------------
std::shared_ptr<EcalGainRatios> StoreEcalCondition::readEcalGainRatiosFromFile(const char* inputFile) {
  //-------------------------------------------------------------

  // create gain ratios
  auto gratio = std::make_shared<EcalGainRatios>();

  FILE* inpFile;  // input file
  inpFile = fopen(inputFile, "r");
  if (!inpFile) {
    edm::LogError("StoreEcalCondition") << "*** Can not open file: " << inputFile;
    return nullptr;
  }

  char line[256];
  std::ostringstream str;

  fgets(line, 255, inpFile);
  string sm_or_all = to_string(line);
  int sm_number = 0;
  sm_number = atoi(line);
  str << "sm= " << sm_number << "\n";

  fgets(line, 255, inpFile);
  //int nevents=atoi(line);

  fgets(line, 255, inpFile);
  string gen_tag = to_string(line);
  str << "gen tag " << gen_tag << "\n";

  fgets(line, 255, inpFile);
  string cali_method = to_string(line);
  str << "cali method " << cali_method << "\n";

  fgets(line, 255, inpFile);
  string cali_version = to_string(line);
  str << "cali version " << cali_version << "\n";

  fgets(line, 255, inpFile);
  string cali_type = to_string(line);

  str << "cali type " << cali_type << "\n";

  edm::LogInfo("StoreEcalCondition") << "GainRatio file " << str.str();

  int cry_num[61200] = {0};
  float g1_g12[61200] = {0};
  float g6_g12[61200] = {0};
  int calib_status[61200] = {0};
  int dummy1 = 0;
  int dummy2 = 0;
  int hash1 = 0;

  int ii = 0;

  if (sm_number != -1) {
    while (fgets(line, 255, inpFile)) {
      sscanf(line, "%d %d %d %f %f %d", &dummy1, &dummy2, &cry_num[ii], &g1_g12[ii], &g6_g12[ii], &calib_status[ii]);
      ii++;
    }

    fclose(inpFile);  // close inp. file

    edm::LogInfo("StoreEcalCondition") << "Read gainRatios for " << ii << " xtals ";
    if (ii != 1700)
      edm::LogWarning("StoreEcalCondition") << " Missing crystals:: missing channels will be set to 0";

    // Get channel ID
    sm_constr_ = sm_number;

    for (int i = 0; i < 1700; i++) {
      // EBDetId(int index1, int index2, int mode = ETAPHIMODE)
      // sm and crys index SMCRYSTALMODE index1 is SM index2 is crystal number a la H4
      EBDetId ebid(sm_slot_, cry_num[i], EBDetId::SMCRYSTALMODE);
      EcalMGPAGainRatio gr;
      gr.setGain12Over6(g6_g12[i]);
      gr.setGain6Over1(g1_g12[i] / g6_g12[i]);
      gratio->setValue(ebid.rawId(), gr);
    }  // loop over channels

  } else {
    // this is for the whole Barrel
    edm::LogInfo("StoreEcalCondition") << "mode ALL BARREL";
    while (fgets(line, 255, inpFile)) {
      int eta = 0;
      int phi = 0;
      sscanf(line, "%d %d %d %f %f", &hash1, &eta, &phi, &g1_g12[ii], &g6_g12[ii]);
      if (ii < 20)
        edm::LogVerbatim("StoreEcalCondition")
            << "crystal eta/phi=" << eta << "/" << phi << " g1_12/g6_12= " << g1_g12[ii] << "/" << g6_g12[ii] << "\n";

      if (g1_g12[ii] < 9 || g1_g12[ii] > 15)
        g1_g12[ii] = 12.0;
      if (g6_g12[ii] < 1 || g6_g12[ii] > 3)
        g6_g12[ii] = 2.0;

      if (eta < -85 || eta > 85 || eta == 0)
        edm::LogVerbatim("StoreEcalCondition") << "error!!!\n";
      if (phi < 1 || phi > 360)
        edm::LogVerbatim("StoreEcalCondition") << "error!!!\n";

      EBDetId ebid(eta, phi, EBDetId::ETAPHIMODE);
      EcalMGPAGainRatio gr;
      gr.setGain12Over6(g6_g12[ii]);
      gr.setGain6Over1(g1_g12[ii] / g6_g12[ii]);
      gratio->setValue(ebid.rawId(), gr);

      ii++;
    }

    fclose(inpFile);  // close inp. file
    if (ii != 61200)
      edm::LogWarning("StoreEcalCondition") << " Missing crystals !!!!!!!";

    edm::LogInfo("StoreEcalCondition") << "number of crystals read:" << ii;
  }

  for (int iX = EEDetId::IX_MIN; iX <= EEDetId::IX_MAX; ++iX) {
    for (int iY = EEDetId::IY_MIN; iY <= EEDetId::IY_MAX; ++iY) {
      // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
      EcalMGPAGainRatio gr;
      gr.setGain12Over6(2.);
      gr.setGain6Over1(6.);

      if (EEDetId::validDetId(iX, iY, 1)) {
        EEDetId eedetidpos(iX, iY, 1);
        gratio->setValue(eedetidpos.rawId(), gr);
      }
      if (EEDetId::validDetId(iX, iY, -1)) {
        EEDetId eedetidneg(iX, iY, -1);
        gratio->setValue(eedetidneg.rawId(), gr);
      }
    }
  }

  edm::LogInfo("StoreEcalCondition") << " gratio pointer=" << gratio;

  edm::LogInfo("StoreEcalCondition") << "now leaving";

  return gratio;
}

std::shared_ptr<EcalChannelStatus> StoreEcalCondition::readEcalChannelStatusFromFile(const char* inputFile) {
  auto status = std::make_shared<EcalChannelStatus>();
  // barrel
  for (int ieta = -EBDetId::MAX_IETA; ieta <= EBDetId::MAX_IETA; ++ieta) {
    if (ieta == 0)
      continue;
    for (int iphi = EBDetId::MIN_IPHI; iphi <= EBDetId::MAX_IPHI; ++iphi) {
      if (EBDetId::validDetId(ieta, iphi)) {
        EBDetId ebid(ieta, iphi);
        status->setValue(ebid, 0);
      }
    }
  }
  // endcap
  for (int iX = EEDetId::IX_MIN; iX <= EEDetId::IX_MAX; ++iX) {
    for (int iY = EEDetId::IY_MIN; iY <= EEDetId::IY_MAX; ++iY) {
      // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
      if (EEDetId::validDetId(iX, iY, 1)) {
        EEDetId eedetidpos(iX, iY, 1);
        status->setValue(eedetidpos, 0);
      }
      if (EEDetId::validDetId(iX, iY, -1)) {
        EEDetId eedetidneg(iX, iY, -1);
        status->setValue(eedetidneg, 0);
      }
    }
  }

  edm::LogInfo("StoreEcalCondition") << "Reading channel status from file " << inputFile;
  FILE* ifile = fopen(inputFile, "r");
  if (!ifile)
    throw cms::Exception("Cannot open ECAL channel status file");

  char line[256];

  fgets(line, 255, ifile);
  std::string gen_tag = line;
  edm::LogVerbatim("StoreEcalCondition") << "Gen tag " << gen_tag << "\n";

  fgets(line, 255, ifile);
  std::string comment = line;
  edm::LogVerbatim("StoreEcalCondition") << "Gen comment " << comment << "\n";

  int iovRunStart(0);
  fgets(line, 255, ifile);
  sscanf(line, "%d", &iovRunStart);
  edm::LogVerbatim("StoreEcalCondition") << "IOV START " << iovRunStart << "\n";
  //if -1 start of time

  int iovRunEnd(0);
  fgets(line, 255, ifile);
  sscanf(line, "%d", &iovRunEnd);
  edm::LogVerbatim("StoreEcalCondition") << "IOV END " << iovRunEnd << "\n";
  //if -1 end of time

  while (fgets(line, 255, ifile)) {
    std::string EBorEE;
    int hashedIndex(0);
    int chStatus(0);
    std::stringstream aStrStream;
    aStrStream << line;
    aStrStream >> EBorEE >> hashedIndex >> chStatus;
    edm::LogVerbatim("StoreEcalCondition")
        << EBorEE << " hashedIndex " << hashedIndex << " status " << chStatus << "\n";

    if (EBorEE == "EB") {
      EBDetId aEBDetId = EBDetId::unhashIndex(hashedIndex);
      status->setValue(aEBDetId, chStatus);
    } else if (EBorEE == "EE") {
      //		chStatus=1;
      EEDetId aEEDetId = EEDetId::unhashIndex(hashedIndex);
      status->setValue(aEEDetId, chStatus);
    } else if (EBorEE == "EBTT") {
      int ism = hashedIndex;
      int itt = chStatus;

      int ixtt = (itt - 1) % 4;
      int iytt = (itt - 1) / 4;
      int ixmin = ixtt * 5;
      int iymin = iytt * 5;
      int ixmax = (ixtt + 1) * 5 - 1;
      int iymax = (iytt + 1) * 5 - 1;
      for (int ieta = iymin; ieta <= iymax; ieta++) {
        for (int iphi = ixmin; iphi <= ixmax; iphi++) {
          int ixt = ieta * 20 + iphi + 1;
          edm::LogVerbatim("StoreEcalCondition") << "killing crystal " << ism << "/" << ixt << "\n";
          EBDetId ebid(ism, ixt, EBDetId::SMCRYSTALMODE);
          status->setValue(ebid, 1);
        }
      }
    }
  }

  fclose(ifile);

  /*
	std::cout <<"KILLING CHANNELS FOR CRAFT EB+16 AND EB+7"<<endl; 

	int ism=7;
	for(int ixt=1; ixt<=500; ixt++) {
	  EBDetId ebid(ism,ixt,EBDetId::SMCRYSTALMODE);
	  status->setValue( ebid, 1 );
	}
	for(int ixt=501; ixt<=900; ixt++) {
	  EBDetId ebid(ism,ixt,EBDetId::SMCRYSTALMODE);
	  if( ((ixt)%20==0) || ((ixt)%20>10) ){  
	    status->setValue( ebid, 1 );
	  }
	}
	ism=16;
	for(int ixt=501; ixt<=900; ixt++) {
	  EBDetId ebid(ism,ixt,EBDetId::SMCRYSTALMODE);
	  if( ((ixt)%20==0) || ((ixt)%20>10) ){  
	    status->setValue( ebid, 1 );
	  }
	}

	*/

  return status;
}
