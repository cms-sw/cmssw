#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/Calibration/interface/Pedestals.h"
#include "CalibCalorimetry/EcalTBCondTools/interface/StoreEcalCondition.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <fstream>

using std::string;

StoreEcalCondition::StoreEcalCondition(const edm::ParameterSet& iConfig) {

  prog_name_ = "StoreEcalCondition";
  typedef std::vector< edm::ParameterSet > Parameters;
  Parameters toPut=iConfig.getParameter<Parameters>("toPut");

  for(Parameters::iterator itToPut = toPut.begin(); itToPut != toPut.end(); ++itToPut) 
    {
      inpFileName_.push_back(itToPut->getUntrackedParameter<std::string>("inputFile"));
      objectName_.push_back(itToPut->getUntrackedParameter<std::string>("conditionType"));
      appendCondition_.push_back(itToPut->getUntrackedParameter<bool>("appendCondition"));
    }

  sm_constr_=0;
  firstrun_=0;
  lastrun_=0;

}

void StoreEcalCondition::endJob() {
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( !mydbservice.isAvailable() ){
    edm::LogError("StoreEcalCondition")<<"PoolDBOutputService is unavailable"<<std::endl;
    return;
  }

  // writeToLogFile(prog_name_, inpFileName_);
  for (unsigned int i=0;i<objectName_.size();i++)
    {

      size_t callbackToken=mydbservice->callbackToken( objectName_[i]);

      //   unsigned int irun=evt.id().run();

      // FC I think we must change the mydbservice->currentTime() in 
      //here 	mydbservice->newValidityForNewPayload<EcalWeightXtalGroups>(mycali,mydbservice->endOfTime(),callbackToken);
      // to start from the firstrun


      unsigned long long tillTime;
      if(appendCondition_[i])   tillTime = (unsigned long long)(firstrun_ - 1); // close exisiting payload at first run of new payload; new payload valid until endOfTime
      else                   tillTime = (unsigned long long)lastrun_; // create new payload for 1st time until last run

      try{
	if (objectName_[i] == "EcalWeightXtalGroups") {
	  EcalWeightXtalGroups* mycali = readEcalWeightXtalGroupsFromFile(inpFileName_[i].c_str());
      
	  try{
	    edm::LogInfo("StoreEcalCondition") <<"current time "<<mydbservice->currentTime();
	    mydbservice->newValidityForNewPayload<EcalWeightXtalGroups>(mycali,tillTime,callbackToken);

	  }catch(const cond::Exception& er){
	    edm::LogError("StoreEcalCondition")<<er.what();
	  }catch(const std::exception& er){
	    edm::LogError("StoreEcalCondition")<<"caught std::exception "<<er.what();
	  }catch(...){
	    edm::LogError("StoreEcalCondition")<<"Funny error";
	  }
	} else if (objectName_[i]  =="EcalTBWeights") {

	  EcalTBWeights* mycali=readEcalTBWeightsFromFile(inpFileName_[i].c_str());

	  try{
	    edm::LogInfo("StoreEcalCondition")<<"current time "<<mydbservice->currentTime();
	    mydbservice->newValidityForNewPayload<EcalTBWeights>(mycali,tillTime,callbackToken);

	  }catch(const cond::Exception& er){
	    edm::LogError("StoreEcalCondition")<<er.what();
	  }catch(const std::exception& er){
	    edm::LogError("StoreEcalCondition")<<"caught std::exception "<<er.what();
	  }catch(...){
	    edm::LogError("StoreEcalCondition")<<"Funny error";
	  }

	}else if (objectName_[i]  == "EcalADCToGeVConstant") {


	  EcalADCToGeVConstant* mycali=readEcalADCToGeVConstantFromFile(inpFileName_[i].c_str());
	  try{
	    edm::LogInfo("StoreEcalCondition")<<"current time "<<mydbservice->currentTime();
	    mydbservice->newValidityForNewPayload<EcalADCToGeVConstant>(mycali,tillTime,callbackToken);

	  }catch(const cond::Exception& er){
	    edm::LogError("StoreEcalCondition")<<er.what();
	  }catch(const std::exception& er){
	    edm::LogError("StoreEcalCondition")<<"caught std::exception "<<er.what();
	  }catch(...){
	    edm::LogError("StoreEcalCondition")<<"Funny error";
	  }

	} else if (objectName_[i]  ==  "EcalIntercalibConstants") {
	  EcalIntercalibConstants* mycali=readEcalIntercalibConstantsFromFile(inpFileName_[i].c_str());
	  try{
	    edm::LogInfo("StoreEcalCondition")<<"current time "<<mydbservice->currentTime();
	    mydbservice->newValidityForNewPayload<EcalIntercalibConstants>(mycali,tillTime,callbackToken);

	  }catch(const cond::Exception& er){
	    edm::LogError("StoreEcalCondition")<<er.what();
	  }catch(const std::exception& er){
	    edm::LogError("StoreEcalCondition")<<"caught std::exception "<<er.what();
	  }catch(...){
	    edm::LogError("StoreEcalCondition")<<"Funny error";
	  }

	} else if (objectName_[i]  ==  "EcalGainRatios") {

	  EcalGainRatios* mycali=readEcalGainRatiosFromFile(inpFileName_[i].c_str());
	  try{
	    edm::LogInfo("StoreEcalCondition")<<"current time "<<mydbservice->currentTime();
	    mydbservice->newValidityForNewPayload<EcalGainRatios>(mycali,tillTime,callbackToken);

	  }catch(const cond::Exception& er){
	    edm::LogError("StoreEcalCondition")<<er.what();
	  }catch(const std::exception& er){
	    edm::LogError("StoreEcalCondition")<<er.what();
	  }catch(...){
	    edm::LogError("StoreEcalCondition")<<"Funny error";
	  }

	} else {
	  edm::LogError("StoreEcalCondition")<< "Object " << objectName_[i]  << " is not supported by this program." << endl;
	}


      } catch(cond::Exception &e) {

	//    writeToLogFileResults("finished with exception\n");
	edm::LogError("StoreEcalCondition")<< e.what() ;
      } catch(std::exception &e) {

	// writeToLogFileResults("finished with exception\n");
	edm::LogError("StoreEcalCondition")<< e.what() ;
      } catch(...) {

	//writeToLogFileResults("finished with exception\n");
	edm::LogError("StoreEcalCondition") << "Unknown exception" ;
      }
    }
  //cout << "about to write to logfile " << endl ;
  // writeToLogFileResults("finished OK\n");
  //cout << "done write to logfile " << endl ;


}


StoreEcalCondition::~StoreEcalCondition() {

}

//-------------------------------------------------------------
void StoreEcalCondition::analyze( const edm::Event& evt, const edm::EventSetup& evtSetup) {
//-------------------------------------------------------------

}


//-------------------------------------------------------------
void StoreEcalCondition::readSMRunFile() {
//-------------------------------------------------------------

  int sm_num[100]={0};
  int start_run[100]={0};
  int end_run[100]={0};
  
  FILE *inpFile; // input file
  inpFile = fopen("/afs/cern.ch/cms/ECAL/testbeam/pedestal/2006/sm_run.txt","r");
  if(!inpFile) {
    edm::LogError("StoreEcalCondition")<<"*** Can not open file: /afs/cern.ch/cms/ECAL/testbeam/pedestal/2006/sm_run.txt"<<endl<<endl;
    return;
  }

  char line[256];
  
  
  int ii = 0;
  //  cout << "reading file ...  " << endl;
  
  std::ostringstream str;
  while(fgets(line,255,inpFile)) {
    sscanf(line, "%d %d %d",  &sm_num[ii], &start_run[ii],&end_run[ii] );
     str << "sm="<<sm_num[ii]<<" start run " << start_run[ii] << " end run"  << end_run[ii] <<endl;
    
    ii++ ;
  }

  edm::LogInfo("StoreEcalCondition") << str.str() ;
  fclose(inpFile);           // close inp. file
  
  for (int ism=0; ism<ii; ism++){
    if(sm_constr_ ==sm_num[ism]) {
      firstrun_=start_run[ism];
      lastrun_=end_run[ism];
      edm::LogInfo("StoreEcalCondition")  << "sm required is ="<<sm_num[ism]<<" start run " << start_run[ism] << " end run"  << end_run[ism] ;
    } 
  }
  
  if(firstrun_ ==0 && lastrun_==0) {
    edm::LogError("StoreEcalCondition") << "no run number found in file for SM="<<endl;	
  }
  
}

//------------------------------------------------------------
void StoreEcalCondition::writeToLogFile(string a, string b) {
//-------------------------------------------------------------
  
    FILE *outFile; // output log file for appending 
    outFile = fopen("/afs/cern.ch/cms/ECAL/testbeam/pedestal/2006/LOGFILE/offdb.log","a");  
    if(!outFile) {
      edm::LogError("StoreEcalCondition") <<"*** Can not open file: /afs/cern.ch/cms/ECAL/testbeam/pedestal/2006/LOGFILE/offdb.log";
      return;
    }
    fprintf(outFile, "%s %s \n", a.c_str(),b .c_str());

    fclose(outFile);           // close out file

}
//------------------------------------------------------------
void StoreEcalCondition::writeToLogFileResults(char* arg) {
//-------------------------------------------------------------
  
    FILE *outFile; // output log file for appending 
    outFile = fopen("/afs/cern.ch/cms/ECAL/testbeam/pedestal/2006/LOGFILE/offdb.log","a");  
    if(!outFile) {
      edm::LogError("StoreEcalCondition")<<"*** Can not open file: /afs/cern.ch/cms/ECAL/testbeam/pedestal/2006/LOGFILE/offdb.log";
      return;
    }
    fprintf(outFile, "%s \n", arg );

    fclose(outFile);           // close out file

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

  edm::LogInfo("StoreEcalCondition") << "LOADING GROUPID FILE" ;

  int xtals=0;
  while (groupid_in.good())
    {
      int xtal_  = -99999;
      int ietaf   = -99999;
      int iphif   = -99999;
      int groupID = -99999;
      groupid_in >> xtal_ >> ietaf >> iphif >> groupID;

      if (xtal_  == -99999 || groupID == -99999 || iphif  == -99999 || ietaf  == -99999 )
	  break;

      LogDebug("StoreEcalCondition") << "XTAL=" << xtal_ << " ETA=" << ietaf << " PHI=" << iphif 
				     << " GROUP=" << groupID ;
      
        //EBDetId ebid(ieta,iphi);	
	// Creating DetId for SM #1
      EBDetId ebid(1,xtal_,EBDetId::SMCRYSTALMODE);	
      //         xtalGroups->setValue(ebid.rawId(), EcalXtalGroupId( ebid.hashedIndex()) );
      xtalGroups->setValue(ebid.rawId(), EcalXtalGroupId( groupID ) );
      xtals++;
    }//loop iphi
  
  edm::LogInfo("StoreEcalCondition") << "Groups for " << xtals << " xtals written into DB" ;
  sm_constr_ = 22;
  readSMRunFile() ; // this to read the correct IOV from the official file 
  
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
	EcalWeightSet::EcalWeightMatrix& chisq1 = wgt.getChi2WeightsBeforeGainSwitch();
	EcalWeightSet::EcalWeightMatrix& chisq2 = wgt.getChi2WeightsAfterGainSwitch();
	
	std::vector<EcalWeight> wamp, wped, wtime; //weights before gain switch
	std::vector<EcalWeight> wamp2, wped2, wtime2; //weights after gain switch
	
	//WEIGHTS BEFORE GAIN SWITCH
	//Amplitude weights 
	for(int j = 0; j < nSamples; ++j) {
	  double ww = 0.0; WeightsFileTB >> ww;
	  wamp.push_back(EcalWeight(ww));
	  str << ww << " ";
	}// loop Samples
	wgt1.push_back(wamp); //vector<vector>
	str << std::endl;
	
	//Pedestal weights
	for(int j = 0; j < nSamples; ++j) {
	  double ww = 0.0; WeightsFileTB >> ww;
	  wped.push_back(EcalWeight(ww));
	  str << ww << " ";
	}//loop Samples
	wgt1.push_back(wped); //vector<vector>
	str << std::endl;
	
	//Timing weights
	for(int j = 0; j < nSamples; ++j) {
	  double ww = 0.0; WeightsFileTB >> ww;
	  wtime.push_back(EcalWeight(ww));
	  str << ww << " ";
	}//loop Samples
	wgt1.push_back(wtime);//vector<vector>
	str << std::endl;
	
	for(int j = 0; j < nSamples; ++j) {
	  // fill chi2 matrix
	  std::vector<EcalWeight> vChi2; // row of chi2 matrix
	  for(int iRow = 0; iRow < nSamples; ++iRow) {
	    double ww = 0.0; WeightsFileTB >> ww;
	    vChi2.push_back(ww);
	    str << ww << " ";
	  } //loop samples
	  chisq1.push_back(vChi2);
	  str << std::endl;
	}//loop lines
	
	//WEIGHTS AFTER GAIN SWITCH
	for(int j = 0; j < nSamples; ++j) {
	  double ww = 0.0; WeightsFileTB >> ww;
	  wamp2.push_back(EcalWeight(ww));
	  str << ww << " ";
	}// loop Samples
	wgt2.push_back(wamp2); //vector<vector>
	str << std::endl;
	
	//Pedestal weights
	for(int j = 0; j < nSamples; ++j) {
	  double ww = 0.0; WeightsFileTB >> ww;
	  wped2.push_back(EcalWeight(ww));
	  str << ww << " ";
	}//loop Samples
	wgt2.push_back(wped2); //vector<vector>
	str << std::endl;
	
	//Timing weights
	for(int j = 0; j < nSamples; ++j) {
	  double ww = 0.0; WeightsFileTB >> ww;
	  wtime2.push_back(EcalWeight(ww));
	  str << ww << " ";
	}//loop Samples
	wgt2.push_back(wtime2);//vector<vector>
	str << std::endl;
	
	for(int j = 0; j < nSamples; ++j) {
	  // fill chi2 matrix
	  std::vector<EcalWeight> vChi2; // row of chi2 matrix
	  for(int iRow = 0; iRow < nSamples; ++iRow) {
	    double ww = 0.0; WeightsFileTB >> ww;
	    vChi2.push_back(ww);
	    str << ww << " ";
	  } //loop samples
	  chisq2.push_back(vChi2);
	  str << std::endl;
	}//loop lines

	LogDebug("StoreEcalCondition") << str.str();
      
	//modif-27-07-06 tdcid should start from 1
	tbwgt->setValue(std::make_pair( igroup_ID , iTdcBin+1 ), wgt);
      }//loop Tdc bins
      ngroups++;
    }//loop groupID

  sm_constr_ = 22;
  readSMRunFile() ; // this to read the correct IOV from the official file 

  edm::LogInfo("StoreEcalCondition") << "Weights for " << ngroups << " xtals written into DB" ;
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

    sm_constr_ =sm_number;
    readSMRunFile() ; // this to read the correct IOV from the official file 


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
    int sm_number=atoi(line);

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

    int cry_num[1700]={0};
    float calib[1700]={0};
    float calib_rms[1700]={0};
    int calib_nevents[1700]={0};
    int calib_status[1700]={0};

    int ii = 0;

    while(fgets(line,255,inpFile)) {
      sscanf(line, "%d %f %f %d %d", &cry_num[ii], &calib[ii], &calib_rms[ii], &calib_nevents[ii], &calib_status[ii] );
//       if(ii<10) { // print out only the first ten channels 
// 	cout << "cry="<<cry_num[ii]<<" calib="<<calib[ii]<<endl;
//       }
      ii++ ;
    }
    

    //    inf.close();           // close inp. file
    fclose(inpFile);           // close inp. file

    edm::LogInfo("StoreEcalCondition") << "Read intercalibrations for " << ii << " xtals " ; 

    //    cout << " I read the calibrations for "<< ii<< " crystals " << endl;
    if(ii!=1700) edm::LogWarning("StoreEcalCondition") << "Some crystals missing. Missing channels will be set to 0" << endl;

    // Get channel ID 
    
    sm_constr_ =sm_number;

    readSMRunFile() ; // this to read the correct IOV from the official file 


    // Set the data
    
 
    // DB supermodule always set to 1 
    
    int sm_db=1;

    for(int i=0; i<1700; i++){
    
    // EBDetId(int index1, int index2, int mode = ETAPHIMODE)
    // sm and crys index SMCRYSTALMODE index1 is SM index2 is crystal number a la H4
    
      EBDetId ebid(sm_db,cry_num[i],EBDetId::SMCRYSTALMODE);

      ical->setValue( ebid.rawId(), calib[i]  );
   
  } // loop over channels 

  return ical;
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
    sm_constr_ =sm_number;
    readSMRunFile() ; // this to read the correct IOV from the official file 

    // DB supermodule always set to 1 
    int sm_db=1;
    for(int i=0; i<1700; i++){
      // EBDetId(int index1, int index2, int mode = ETAPHIMODE)
      // sm and crys index SMCRYSTALMODE index1 is SM index2 is crystal number a la H4
      EBDetId ebid(sm_db,cry_num[i],EBDetId::SMCRYSTALMODE);
      // cout << "ebid.rawId()"<< ebid.rawId()<< endl;
      EcalMGPAGainRatio gr;
      gr.setGain12Over6( g6_g12[ii] );
      gr.setGain6Over1(g1_g12[ii]/g6_g12[ii]);
      gratio->setValue( ebid.rawId(), gr );
    } // loop over channels 


    return gratio;

}
