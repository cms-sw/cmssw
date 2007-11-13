#include "CondTools/Ecal/plugins/StoreEcalCondition.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <fstream>
#include <iostream>
#include <time.h>
#include <unistd.h> 

using std::string;

StoreEcalCondition::StoreEcalCondition(const edm::ParameterSet& iConfig) {

  prog_name_ = "StoreEcalCondition";

  logfile_ = iConfig.getParameter< std::string >("logfile");
  sm_slot_ = iConfig.getUntrackedParameter< unsigned int >("smSlot", 1);

  typedef std::vector< edm::ParameterSet > Parameters;
  Parameters toPut=iConfig.getParameter<Parameters>("toPut");
  for(Parameters::iterator itToPut = toPut.begin(); itToPut != toPut.end(); ++itToPut) 
    {
      inpFileName_.push_back(itToPut->getUntrackedParameter<std::string>("inputFile"));
      objectName_.push_back(itToPut->getUntrackedParameter<std::string>("conditionType"));
      //      since_.push_back(itToPut->getUntrackedParameter<unsigned int>("since"));
    }

  sm_constr_ = -1;
}

void StoreEcalCondition::endJob() {
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( !mydbservice.isAvailable() ){
    edm::LogError("StoreEcalCondition")<<"PoolDBOutputService is unavailable"<<std::endl;
    return;
  }

  bool toAppend=false;
  for (unsigned int i=0;i<objectName_.size();i++) {
    try{
      cond::Time_t newTime;
      
      if ( mydbservice->isNewTagRequest( objectName_[i]+std::string("Rcd") ) ) {
	// This is the first object for this tag.
	// Append mode should be off.
	// newTime is the end of this new objects IOV.
	newTime = mydbservice->endOfTime();
      } else {
	// There should already be an object in the DB for this tag.
	// Append IOV mode should be on.
	// newTime is the beginning of this new objects IOV.
	toAppend=true;
	newTime = (cond::Time_t)since_[i];
      }
      edm::LogInfo("StoreEcalCondition") << "Reading " << objectName_[i] 
					 << " from file and writing to DB with newTime " << newTime << endl;
      
      if (objectName_[i] == "EcalWeightXtalGroups") {
	EcalWeightXtalGroups* mycali = readEcalWeightXtalGroupsFromFile(inpFileName_[i].c_str());
	if(!toAppend){
	  mydbservice->createNewIOV<EcalWeightXtalGroups>(mycali,newTime,"EcalWeightXtalGroupsRcd");
	}else{
	  mydbservice->appendSinceTime<EcalWeightXtalGroups>(mycali,newTime,"EcalWeightXtalGroupsRcd");
	}
      }else if (objectName_[i]  =="EcalTBWeights") {
	EcalTBWeights* mycali=readEcalTBWeightsFromFile(inpFileName_[i].c_str());
	if(!toAppend){
	  mydbservice->createNewIOV<EcalTBWeights>(mycali,newTime,"EcalTBWeightsRcd");
	}else{
	  mydbservice->appendSinceTime<EcalTBWeights>(mycali,newTime,"EcalTBWeightsRcd");
	}
      } else if (objectName_[i]  == "EcalADCToGeVConstant") {
	EcalADCToGeVConstant* mycali=readEcalADCToGeVConstantFromFile(inpFileName_[i].c_str());
	if(!toAppend){
	  mydbservice->createNewIOV<EcalADCToGeVConstant>(mycali,newTime,"EcalADCToGeVConstantRcd");
	}else{
	  mydbservice->appendSinceTime<EcalADCToGeVConstant>(mycali,newTime,"EcalADCToGeVConstantRcd");
	}
      } else if (objectName_[i]  ==  "EcalIntercalibConstants") {
	EcalIntercalibConstants* mycali=readEcalIntercalibConstantsFromFile(inpFileName_[i].c_str());
	if(!toAppend){
	  mydbservice->createNewIOV<EcalIntercalibConstants>(mycali,newTime,"EcalIntercalibConstantsRcd");
	}else{
	  mydbservice->appendSinceTime<EcalIntercalibConstants>(mycali,newTime,"EcalIntercalibConstantsRcd");
	}
      } else if (objectName_[i]  ==  "EcalGainRatios") {
	EcalGainRatios* mycali=readEcalGainRatiosFromFile(inpFileName_[i].c_str());
	if(!toAppend){
	  mydbservice->createNewIOV<EcalGainRatios>(mycali,newTime,"EcalGainRatiosRcd");
	}else{
	  mydbservice->appendSinceTime<EcalGainRatios>(mycali,newTime,"EcalGainRatiosRcd");
	}
      } else {
	edm::LogError("StoreEcalCondition")<< "Object " << objectName_[i]  << " is not supported by this program." << endl;
      }
      

      //      writeToLogFile(objectName_[i], inpFileName_[i], since_[i]);

    } catch(cond::Exception &e) {
      writeToLogFileResults("finished with exception\n");
      edm::LogError("StoreEcalCondition")<< e.what() ;
    } catch(std::exception &e) {
      writeToLogFileResults("finished with exception\n");
      edm::LogError("StoreEcalCondition")<< e.what() ;
    } catch(...) {
      writeToLogFileResults("finished with exception\n");
      edm::LogError("StoreEcalCondition") << "Unknown exception" ;
    }
  }

  writeToLogFileResults("finished OK\n");
  edm::LogInfo("StoreEcalCondition") << "Finished endJob" << endl;
}


StoreEcalCondition::~StoreEcalCondition() {

}

//-------------------------------------------------------------
void StoreEcalCondition::analyze( const edm::Event& evt, const edm::EventSetup& evtSetup) {
//-------------------------------------------------------------

}


//------------------------------------------------------------
void StoreEcalCondition::writeToLogFile(string a, string b, unsigned long long since) {
//-------------------------------------------------------------
  
    FILE *outFile; // output log file for appending 
    outFile = fopen(logfile_.c_str(),"a");  
    if(!outFile) {
      edm::LogError("StoreEcalCondition") <<"*** Can not open file: " << logfile_;
      return;
    }
    char header[256];
    fillHeader(header);
    char appendMode[10];
    if (since != 0)
      sprintf(appendMode,"append");
    else
      sprintf(appendMode,"create");

    fprintf(outFile, "%s %s condition from file %s written into DB for SM %d (mapped to SM %d) in %s mode (since run %u)\n", 
	    header, a.c_str(),b .c_str(), sm_constr_, sm_slot_, appendMode, (unsigned int)since);

    fclose(outFile);           // close out file

}
//------------------------------------------------------------
void StoreEcalCondition::writeToLogFileResults(char* arg) {
//-------------------------------------------------------------
  
    FILE *outFile; // output log file for appending 
    outFile = fopen(logfile_.c_str(),"a");  
    if(!outFile) {
      edm::LogError("StoreEcalCondition")<<"*** Can not open file: " << logfile_;
      return;
    }
    char header[256];
    fillHeader(header);
    fprintf(outFile, "%s %s\n", header,arg);
    fclose(outFile);           // close out file
}

//------------------------------------------------------------
void StoreEcalCondition::fillHeader(char* header)
//------------------------------------------------------------
{
  time_t rawtime;
  struct tm * timeinfo;
  time ( &rawtime );
  timeinfo = localtime ( &rawtime );
  char user[50];
  sprintf(user,"%s",getlogin());
  sprintf(header,"%s %s:",asctime(timeinfo),user);
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
EcalWeightXtalGroups*
StoreEcalCondition::readEcalWeightXtalGroupsFromFile(const char* inputFile) {
//-------------------------------------------------------------

// Code taken from EcalWeightTools/test/MakeOfflineDbFromAscii.cpp

  EcalWeightXtalGroups* xtalGroups = new EcalWeightXtalGroups();
  std::ifstream groupid_in(inputFile);

  if(!groupid_in.is_open()) {
    edm::LogError("StoreEcalCondition")<< "*** Can not open file: "<< inputFile ;
    return 0;
  }  

  int smnumber=-99999;

  std::ostringstream str;
  groupid_in >> smnumber;
  if (smnumber == -99999) {
    edm::LogError("StoreEcalCondition") << "ERROR: SM number not found in file" << endl;
    return 0;
  }
  str << "sm= " << smnumber << endl;
  sm_constr_ = smnumber;

  char temp[256];
  //Reading the other 5 header lines containing various informations
  for (int i=0;i<=5;i++) {
    groupid_in.getline(temp,255);
    str << temp << endl;
  }

  // Skip the nGroup/Mean line
  groupid_in.getline(temp, 255);
  str << temp << endl;

  edm::LogInfo("StoreEcalCondition") << "GROUPID file " << str.str() ;

  int xtals = 0;
  int xtal, ietaf, iphif, groupID;
  while (groupid_in.good()) {
    groupid_in >> xtal >> ietaf >> iphif >> groupID;
    if (groupid_in.eof()) { break; }
    
    LogDebug("StoreEcalCondition") << "XTAL=" << xtal << " ETA=" << ietaf << " PHI=" << iphif 
				       << " GROUP=" << groupID ;

    //EBDetId ebid(ieta,iphi);	
    EBDetId ebid(sm_slot_,xtal,EBDetId::SMCRYSTALMODE);	
    // xtalGroups->setValue(ebid.rawId(), EcalXtalGroupId( ebid.hashedIndex()) );
    xtalGroups->setValue(ebid.rawId(), EcalXtalGroupId( groupID ) );
    xtals++;
  }//loop iphi

  if (xtals != 1700) {
    edm::LogError("StoreEcalCondition") << "ERROR:  GROUPID file did not contain data for 1700 crystals" << endl;
    return 0;
  }
  
  edm::LogInfo("StoreEcalCondition") << "Groups for " << xtals << " xtals written into DB" ;
  sm_constr_ = smnumber;
  
  return xtalGroups;
}

//-------------------------------------------------------------
EcalTBWeights*
StoreEcalCondition::readEcalTBWeightsFromFile(const char* inputFile) {
  //-------------------------------------------------------------
  
  // Zabi code to be written here 
  
  EcalTBWeights* tbwgt = new EcalTBWeights();
  
  std::ifstream WeightsFileTB(inputFile);
  if(!WeightsFileTB.is_open()) {
    edm::LogError("StoreEcalCondition")<< "*** Can not open file: "<< inputFile ;
    return 0;
  }

  int smnumber=-99999;

  std::ostringstream str;
  WeightsFileTB >> smnumber;
  if (smnumber == -99999)
    return 0;

  str << "sm= " << smnumber << endl;

  char temp[256];
  //Reading the other 5 header lines containing various informations
  for (int i=0;i<=5;i++)
    {
      WeightsFileTB.getline(temp,255);
      str << temp << endl;
    }

  edm::LogInfo("StoreEcalCondition") << "Weights file " << str.str() ;

  int ngroups=0;
  while (WeightsFileTB.good())
    {
      int igroup_ID = -99999;
      int nSamples  = -99999;
      int nTdcBins  = -99999;
      
      WeightsFileTB >> igroup_ID >> nSamples >> nTdcBins;
      if (igroup_ID == -99999 || nSamples == -99999 || nTdcBins == -99999)
	break;
      
      std::ostringstream str;
      str << "Igroup=" << igroup_ID << " Nsamples=" << nSamples << " NTdcBins=" << nTdcBins <<"\n" ;
      
      for (int iTdcBin = 0; iTdcBin < nTdcBins; iTdcBin++) {
	
	EcalWeightSet wgt; // one set of weights
	EcalWeightSet::EcalWeightMatrix& wgt1   = wgt.getWeightsBeforeGainSwitch();
	EcalWeightSet::EcalWeightMatrix& wgt2   = wgt.getWeightsAfterGainSwitch();	
	EcalWeightSet::EcalChi2WeightMatrix& chisq1 = wgt.getChi2WeightsBeforeGainSwitch();
	EcalWeightSet::EcalChi2WeightMatrix& chisq2 = wgt.getChi2WeightsAfterGainSwitch();
	
// 	std::vector<EcalWeight> wamp, wped, wtime; //weights before gain switch
// 	std::vector<EcalWeight> wamp2, wped2, wtime2; //weights after gain switch
	
	//WEIGHTS BEFORE GAIN SWITCH
	//Amplitude weights 
	for(int j = 0; j < nSamples; ++j) {
	  double ww = 0.0; WeightsFileTB >> ww;
	  wgt1(0,j)=ww;
	  str << ww << " ";
	}// loop Samples
	str << std::endl;
	
	//Pedestal weights
	for(int j = 0; j < nSamples; ++j) {
	  double ww = 0.0; WeightsFileTB >> ww;
	  wgt1(1,j)=ww;
	  str << ww << " ";
	}//loop Samples
	str << std::endl;
	
	//Timing weights
	for(int j = 0; j < nSamples; ++j) {
	  double ww = 0.0; WeightsFileTB >> ww;
	  wgt1(2,j)=ww;
	  str << ww << " ";
	}//loop Samples
	str << std::endl;
	
	for(int j = 0; j < nSamples; ++j) {
	  // fill chi2 matrix
	  //std::vector<EcalWeight> vChi2; // row of chi2 matrix
	  for(int k = 0; k < nSamples; ++k) {
	    double ww = 0.0; WeightsFileTB >> ww;
	    chisq1(j,k)=ww;
	    str << ww << " ";
	  } //loop samples
	  str << std::endl;
	}//loop lines
	
	//WEIGHTS AFTER GAIN SWITCH
	for(int j = 0; j < nSamples; ++j) {
	  double ww = 0.0; WeightsFileTB >> ww;
	  wgt2(0,j)=ww;
	  str << ww << " ";
	}// loop Samples
	str << std::endl;
	
	//Pedestal weights
	for(int j = 0; j < nSamples; ++j) {
	  double ww = 0.0; WeightsFileTB >> ww;
	  wgt2(1,j)=ww;
	  str << ww << " ";
	}//loop Samples
	str << std::endl;
	
	//Timing weights
	for(int j = 0; j < nSamples; ++j) {
	  double ww = 0.0; WeightsFileTB >> ww;
	  wgt2(2,j)=ww;
	  str << ww << " ";
	}//loop Samples
	str << std::endl;
	
	for(int j = 0; j < nSamples; ++j) {
	  // fill chi2 matrix
	  //std::vector<EcalWeight> vChi2; // row of chi2 matrix
	  for(int k = 0; k < nSamples; ++k) {
	    double ww = 0.0; WeightsFileTB >> ww;
	    chisq2(j,k)=ww;
	    str << ww << " ";
	  } //loop samples
	  str << std::endl;
	}//loop lines

	LogDebug("StoreEcalCondition") << str.str();
      
	//modif-27-07-06 tdcid should start from 1
	tbwgt->setValue(std::make_pair( igroup_ID , iTdcBin+1 ), wgt);
      }//loop Tdc bins
      ngroups++;
    }//loop groupID

  sm_constr_ = smnumber;


  edm::LogInfo("StoreEcalCondition") << "Weights for " << ngroups << " groups written into DB" ;
  return tbwgt;
}


//-------------------------------------------------------------
EcalADCToGeVConstant*
StoreEcalCondition::readEcalADCToGeVConstantFromFile(const char* inputFile) {
//-------------------------------------------------------------
  
    
    FILE *inpFile; // input file
    inpFile = fopen(inputFile,"r");
    if(!inpFile) {
      edm::LogError("StoreEcalCondition")<<"*** Can not open file: "<<inputFile;
      return 0;
    }

    char line[256];
    
    std::ostringstream str;

    fgets(line,255,inpFile);
    int sm_number=atoi(line);
    str << "sm= " << sm_number << endl ;  

    fgets(line,255,inpFile);
    //int nevents=atoi(line); // not necessary here just for online conddb
    
    fgets(line,255,inpFile);
    string gen_tag=to_string(line);
    str << "gen tag " << gen_tag << endl ;  // should I use this? 

    fgets(line,255,inpFile);
    string cali_method=to_string(line);
    str << "cali method " << cali_method << endl ; // not important 

    fgets(line,255,inpFile);
    string cali_version=to_string(line);
    str << "cali version " << cali_version << endl ; // not important 

    fgets(line,255,inpFile);
    string cali_type=to_string(line);
    str << "cali type " << cali_type << endl ; // not important

    edm::LogInfo("StoreEcalCondition") << "ADCToGeV file " << str.str() ;

    fgets(line,255,inpFile);
    float adc_to_gev=0;
    sscanf(line, "%f", &adc_to_gev );
    LogDebug("StoreEcalCondition") <<" calib="<< adc_to_gev ;
    
    fclose(inpFile);           // close inp. file

    sm_constr_ = sm_number;

    // barrel and endcaps the same 
    EcalADCToGeVConstant* agc = new EcalADCToGeVConstant(adc_to_gev,adc_to_gev );
    edm::LogInfo("StoreEcalCondition") << "ADCtoGeV scale written into the DB";
    return agc;
}


//-------------------------------------------------------------
EcalIntercalibConstants*
StoreEcalCondition::readEcalIntercalibConstantsFromFile(const char* inputFile) {
//-------------------------------------------------------------

  EcalIntercalibConstants* ical = new EcalIntercalibConstants();

    
    FILE *inpFile; // input file
    inpFile = fopen(inputFile,"r");
    if(!inpFile) {
      edm::LogError("StoreEcalCondition")<<"*** Can not open file: "<<inputFile;
      return 0;
    }

    char line[256];
 
    std::ostringstream str;

    fgets(line,255,inpFile);
    string sm_or_all=to_string(line);
    int sm_number=0;
    int nchan=1700;
    sm_number=atoi(line);
    str << "sm= " << sm_number << endl ;  
    if(sm_number!=-1){
      nchan=1700;
    } else {
      nchan=61200;
    }
    

    fgets(line,255,inpFile);
    //int nevents=atoi(line); // not necessary here just for online conddb
    
    fgets(line,255,inpFile);
    string gen_tag=to_string(line);
    str << "gen tag " << gen_tag << endl ;  // should I use this? 

    fgets(line,255,inpFile);
    string cali_method=to_string(line);
    str << "cali method " << cali_method << endl ; // not important 

    fgets(line,255,inpFile);
    string cali_version=to_string(line);
    str << "cali version " << cali_version << endl ; // not important 

    fgets(line,255,inpFile);
    string cali_type=to_string(line);
    str << "cali type " << cali_type << endl ; // not important

    edm::LogInfo("StoreEcalCondition") << "Intercalibration file " << str.str() ;

    int sm_num[61200]={0};
    int cry_num[61200]={0};
    float calib[61200]={0};
    float calib_rms[61200]={0};
    int calib_nevents[61200]={0};
    int calib_status[61200]={0};

    int ii = 0;
    if(sm_number!=-1){
      while(fgets(line,255,inpFile)) {
	sscanf(line, "%d %f %f %d %d", &cry_num[ii], &calib[ii], &calib_rms[ii], &calib_nevents[ii], &calib_status[ii] );
//       if(ii<10) { // print out only the first ten channels 
// 	cout << "cry="<<cry_num[ii]<<" calib="<<calib[ii]<<endl;
//       }
	sm_num[ii]=sm_number;
	ii++ ;
      }
    } else {
      // this is for the whole Barrel 
      cout<<"mode ALL BARREL" <<endl; 
      while(fgets(line,255,inpFile)) {
	sscanf(line, "%d %d %f %f %d", &sm_num[ii], &cry_num[ii], &calib[ii], &calib_rms[ii], &calib_status[ii] );
	if(ii==0) cout<<"crystal "<<cry_num[ii]<<" of sm "<<sm_num[ii]<<" cali= "<< calib[ii]<<endl;
	ii++ ;
      }
    }

    //    inf.close();           // close inp. file
    fclose(inpFile);           // close inp. file

    edm::LogInfo("StoreEcalCondition") << "Read intercalibrations for " << ii << " xtals " ; 

    cout << " I read the calibrations for "<< ii<< " crystals " << endl;
    if(ii!=nchan) edm::LogWarning("StoreEcalCondition") << "Some crystals missing. Missing channels will be set to 0" << endl;

    // Get channel ID 
    
    sm_constr_ = sm_number;


    // Set the data
    for(int i=0; i<nchan; i++){
    
    // EBDetId(int index1, int index2, int mode = ETAPHIMODE)
    // sm and crys index SMCRYSTALMODE index1 is SM index2 is crystal number a la H4
    
      int slot_num=convertFromConstructionSMToSlot(sm_num[i],-1);
      EBDetId ebid(slot_num,cry_num[i],EBDetId::SMCRYSTALMODE);

      ical->setValue( ebid.rawId(), calib[i]  );

      if(i==0) cout<<"crystal "<<cry_num[i]<<" of sm "<<sm_num[i]<< " in slot " <<slot_num<<" calib= "<< calib[i]<<endl;

    } // loop over channels 

    cout<<"loop on channels done" << endl;
  return ical;
}

//-------------------------------------------------------------
int StoreEcalCondition::convertFromConstructionSMToSlot(int sm_constr,int sm_slot){
  // input either cosntruction number or slot number and returns the other
  // the slots are numbered first EB+ slot 1 ...18 then EB- 1... 18 
  // the slots start at 1 and the SM start at 0
//-------------------------------------------------------------
  int slot_to_constr[37]={-1,12,17,10,1,8,4,27,20,23,25,6,34,35,15,18,30,21,9 
			  ,24,22,13,31,26,16,2,11,5,0,29,28,14,33,32,3,7,19};
  int constr_to_slot[36]={28,4,25,34,6,27,11,35,5,18,3,26,1,21,31,14,24,2,15,
			  36,8,17,20,9,19,10,23,7,30,29,16,22,33,32,12,13  };
  
  int result=0;
  if(sm_constr!=-1) {
    result=constr_to_slot[sm_constr];
  } else if(sm_slot!=-1) {
    result=slot_to_constr[sm_slot];
  }
  return result;
}


//-------------------------------------------------------------
EcalGainRatios*
StoreEcalCondition::readEcalGainRatiosFromFile(const char* inputFile) {
//-------------------------------------------------------------

  // create gain ratios
  EcalGainRatios* gratio = new EcalGainRatios;
    
    FILE *inpFile; // input file
    inpFile = fopen(inputFile,"r");
    if(!inpFile) {
      edm::LogError("StoreEcalCondition")<<"*** Can not open file: "<<inputFile;
      return 0;
    }

    char line[256];
    std::ostringstream str;
              
    fgets(line,255,inpFile);
    int sm_number=atoi(line);
    str << "sm= " << sm_number << endl ;  

    fgets(line,255,inpFile);
    //int nevents=atoi(line);
    
    fgets(line,255,inpFile);
    string gen_tag=to_string(line);
    str << "gen tag " << gen_tag << endl ;    

    fgets(line,255,inpFile);
    string cali_method=to_string(line);
    str << "cali method " << cali_method << endl ;

    fgets(line,255,inpFile);
    string cali_version=to_string(line);
    str << "cali version " << cali_version << endl ;


    fgets(line,255,inpFile);
    string cali_type=to_string(line);

    str << "cali type " << cali_type << endl ;

    edm::LogInfo("StoreEcalCondition") << "GainRatio file " << str.str() ;

    int cry_num[1700]={0};
    float g1_g12[1700]={0};
    float g6_g12[1700]={0};
    int calib_status[1700]={0};
    int dummy1=0;
    int dummy2=0;

    int ii = 0;


    while(fgets(line,255,inpFile)) {
      sscanf(line, "%d %d %d %f %f %d", &dummy1, &dummy2, &cry_num[ii], &g1_g12[ii], &g6_g12[ii], &calib_status[ii] );
//       if(ii<10){
// 	cout << "cry="<<cry_num[ii]<<" gains="
// 	     <<g1_g12[ii]<< "and "<< g6_g12[ii]<<endl;
//       }
      ii++ ;
    }
    
    
    fclose(inpFile);           // close inp. file

    edm::LogInfo("StoreEcalCondition") << "Read gainRatios for " << ii << " xtals " ;
    if(ii!=1700) edm::LogWarning("StoreEcalCondition") << " Missing crystals:: missing channels will be set to 0" << endl;

    // Get channel ID 
    sm_constr_ = sm_number;

    for(int i=0; i<1700; i++){
      // EBDetId(int index1, int index2, int mode = ETAPHIMODE)
      // sm and crys index SMCRYSTALMODE index1 is SM index2 is crystal number a la H4
      EBDetId ebid(sm_slot_,cry_num[i],EBDetId::SMCRYSTALMODE);
      // cout << "ebid.rawId()"<< ebid.rawId()<< endl;
      EcalMGPAGainRatio gr;
      gr.setGain12Over6( g6_g12[i] );
      gr.setGain6Over1(g1_g12[i]/g6_g12[i]);
      gratio->setValue( ebid.rawId(), gr );
    } // loop over channels 


    return gratio;

}
