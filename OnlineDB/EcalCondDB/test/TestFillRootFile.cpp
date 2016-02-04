#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <time.h>

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/all_lmf_types.h"
#include "OnlineDB/EcalCondDB/interface/RunDat.h"
#include "TROOT.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TH2F.h"
#include "TF1.h"


using namespace std;

class CondDBApp {
public:

  /**
   *   App constructor; Makes the database connection
   */
  CondDBApp(string host, string sid, string user, string pass)
  {
    try {
      cout << "Making connection..." << flush;
      econn = new EcalCondDBInterface( host, sid, user, pass );
      cout << "Done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
      exit(-1);
    }

 
  }



  /**
   *  App destructor;  Cleans up database connection
   */
  ~CondDBApp() 
  {
    delete econn;
  }

inline std::string to_string( char value[])
                                                                                
{
                                                                                
    std::ostringstream streamOut;
                                                                                
    streamOut << value;
                                                                                
    return streamOut.str();
                                                                                
}
                                                                                
                                                                                

  RunIOV makeRunIOV(std::string run_num, std::string& location)
  {
    cout << "entering retrieveRunIOV for run " << run_num<< " and location= "<< location << endl;
    // The objects necessary to identify a dataset
    LocationDef locdef;
    RunTypeDef rundef;
    RunTag runtag;
 
    locdef.setLocation(location);
 
    rundef.setRunType("LASER");
     
    runtag.setLocationDef(locdef);
    runtag.setRunTypeDef(rundef);
    runtag.setGeneralTag("LASER"); 
    cout << " set the run tag and location " << endl;

    // Set the properties of the iov
    RunIOV runiov;


    int run=atoi(run_num.c_str());

    //    string run_string=to_string(run_num);
    // sscanf(run_num, "%d", &run );

     runiov.setRunTag(runtag); 
     runiov.setRunNumber(run);
    cout << " before " << endl;
    //  int runiov_id=runiov.fetchIDNoTimeInfo();
    cout << " after " << endl;
    // runiov.setByID(runiov_id);
     
    cout << "Setting run " << run << endl;
    
    return runiov;

  }

  
  LMFRunIOV makeLMFRunIOV(RunIOV* runiov)
  {
    // LMF Tag and IOV
    
    LMFRunTag lmftag;
    
    LMFRunIOV lmfiov;
    lmfiov.setLMFRunTag(lmftag);
    lmfiov.setRunIOV(*runiov);
    lmfiov.setSubRunNumber(0);
    lmfiov.setSubRunStart(runiov->getRunStart());

    return lmfiov;
  }



  /**
   *  Write LMFLaserBlueRawDat objects with associated RunTag and RunIOVs
   *  IOVs are written using automatic updating of the 'RunEnd', as if
   *  the time of the end of the run was not known at the time of writing.
   */
  void testWrite(std::string& run_number)
  {
    std::string location="H4B"; // this is H4 beam 
    cout << "Writing LMFLaserBlueDat to database..." << endl;
    RunIOV runiov = this->makeRunIOV(run_number, location);
    RunTag runtag = runiov.getRunTag();
    run_t run = runiov.getRunNumber();
    cout << "Fetching run by tag just used..." << flush;
    RunIOV runiov_prime = econn->fetchRunIOV(&runtag, run);

    // LMF Tag and IOV
    LMFRunIOV lmfiov = this->makeLMFRunIOV(&runiov_prime);

    // retrieve run dat => get the sm in this run
    EcalLogicID ecid_sm;
    RunDat rd_sm;

  
    RunDat rd;
    map<EcalLogicID, RunDat> dataset;
    econn->fetchDataSet(&dataset, &runiov_prime);


    int sm_num=ecid_sm.getID1();  
    int count = 0;
    typedef map< EcalLogicID, RunDat >::const_iterator CI;
    for (CI p = dataset.begin(); p != dataset.end(); p++) {
      count++;
      ecid_sm = p->first;
      rd_sm  = p->second;
      sm_num=ecid_sm.getID1();
      cout << "SM:                     " << ecid_sm.getID1() << endl;
      cout << "========================" << endl;
    }
    cout << endl;


     




    // Get channel ID for SM sm_num, crystal c      
    map<EcalLogicID, LMFLaserBlueNormDat> dataset_lmf;
    map<EcalLogicID, LMFLaserBlueRawDat> dataset_lraw;
    map<EcalLogicID, LMFPNBlueDat> dataset_lpn;
    map<EcalLogicID, LMFMatacqBlueDat> dataset_mq;

    Int_t cx, cy;

    TFile *f;
    TFile *f_mq;
    
    ostringstream Run;
    ostringstream sROOT; 
    ostringstream sROOT_mq; 

    Run.str("");
    Run << "0000" << run;

    sROOT.str("");
    sROOT_mq.str("");
    // this is for when they will store the file with the SM number 
    // for now all are stored fas if Sm22
    //    sROOT << "/data1/daq/crogan/ROOT_files/SM" << sm_num << "/SM" << sm_num  << "-0000";

    sROOT << "/data1/daq/crogan/ROOT_files/SM22/SM22-0000";
    sROOT_mq << "/data1/daq/crogan/ROOT_files/SM22/MATACQ-SM22-0000";
    sROOT << run << ".root";
    sROOT_mq << run << ".root";

    cout << "opening file "<< sROOT.str().c_str() << endl;
    f = (TFile*) new TFile(sROOT.str().c_str());
    TDirectory *DIR = (TDirectory*) f->GetDirectory(Run.str().c_str());

    //    f = (TFile*) new TFile("/data1/daq/crogan/ROOT_files/SM22/SM22-000017400.root");
    //  TDirectory *DIR = (TDirectory*) f->GetDirectory("000017400");

    if(DIR != NULL){
      TH2F *APDPN = (TH2F*) DIR->Get("APDPN");
      TH2F *APD = (TH2F*) DIR->Get("APD");
      TH2F *APD_RMS = (TH2F*) DIR->Get("APD_RMS");
      TH2F *APDPN_RMS = (TH2F*) DIR->Get("APDPN_RMS");
      TH2F *PN = (TH2F*) DIR->Get("PN");
      if(APDPN !=NULL && APD !=NULL && APD_RMS !=NULL && APDPN_RMS!=NULL && PN !=NULL ){

	int pn_chan[]={1,101, 501, 901, 1301, 11, 111, 511, 911, 1311 };

	for (int c=1; c<1701; c++){
	  // the channels are turned in phi and eta 
	  // with respect to the standard numbering scheme
	  cx = 85-(c-1)/20;
	  cy = 20-(c-1)%20;
	  float apdpn = (float) APDPN->GetBinContent(cx,cy);
	  float apd = (float) APD->GetBinContent(cx,cy);
	  float apd_rms = (float) APD_RMS->GetBinContent(cx,cy);
	  float apdpn_rms = (float) APDPN_RMS->GetBinContent(cx,cy);
	  float pn = (float) PN->GetBinContent(cx,cy);
	  
	  if(c%20==0) cout << "channel "<< c<< " value "<< apdpn << endl; 
	  EcalLogicID ecid;
	  ecid = econn->getEcalLogicID("EB_crystal_number", sm_num, c);

	  // Set the data
	  LMFLaserBlueNormDat bluelaser;
	  LMFLaserBlueRawDat bluelaserraw;
	  
	  int ispn=-1;
	  for(int j=0; j<10; j++){
	    if(pn_chan[j]==c) ispn=j;
	  }

	  if(ispn!=-1){
	    LMFPNBlueDat bluepn;

	    bluepn.setPNPeak(pn);
	    bluepn.setPNErr(pn/100.);
	    EcalLogicID ecid_pn;
	    ecid_pn = econn->getEcalLogicID("EB_LM_PN", sm_num, ispn);
	    dataset_lpn[ecid_pn] = bluepn;
	  }

	  bluelaser.setAPDOverPNMean(apdpn);
	  bluelaser.setAPDOverPNAMean(apdpn);
	  bluelaser.setAPDOverPNBMean(apdpn);
	  bluelaser.setAPDOverPNARMS(apdpn_rms);
	  bluelaser.setAPDOverPNBRMS(apdpn_rms);
	  bluelaser.setAPDOverPNRMS(apdpn_rms);
    
	  bluelaserraw.setAPDPeak(apd);
	  bluelaserraw.setAPDErr(apd_rms);
	  
	  // Fill the dataset
	  dataset_lmf[ecid] = bluelaser;
	  dataset_lraw[ecid] = bluelaserraw;
	  

	}
      }

    }
    f->Close();
    cout << "finished processing the APD and PN data"  << endl;

    cout << "opening MATACQ file "<< sROOT_mq.str().c_str() << endl;

    try {
 

    f_mq = (TFile*) new TFile(sROOT_mq.str().c_str());

    TH1F *mq_height = (TH1F*) f_mq->Get("Height_Channel_0_blue");
    TH1F *mq_width = (TH1F*) f_mq->Get("Width_Channel_0_blue");
    TH1F *mq_timing = (TH1F*) f_mq->Get("Timing_Channel_0_blue");
    
    mq_height->Fit("gaus");   
    TF1 *fit_height= mq_height->GetFunction("gaus");
    float par_height= (float) fit_height ->GetParameter(1);

    mq_width->Fit("gaus");   
    TF1 *fit_width= mq_width->GetFunction("gaus");
    float par_width=(float) fit_width ->GetParameter(1);

    mq_timing->Fit("gaus");   
    TF1 *fit_timing= mq_timing->GetFunction("gaus");
    float par_timing=(float) fit_timing ->GetParameter(1);

    // siamo arrivati qui ... 
    // aggiustare il ecid del matacq e sistemare i dati del mq. 

    LMFMatacqBlueDat lmf_mq;
    EcalLogicID ecid_mq;
    ecid_mq = econn->getEcalLogicID("EB");
    lmf_mq.setAmplitude(par_height);
    lmf_mq.setWidth(par_width);
    lmf_mq.setTimeOffset(par_timing);
    dataset_mq[ecid_mq] = lmf_mq;
    econn->insertDataSet(&dataset_mq, &lmfiov);

    f_mq->Close();


    cout << "Done Matacq "  << endl;

    } catch (exception &e) { 
      cout << "TestLMF>> error with MATACQ " << e.what() ;
    } 

    // Insert the dataset, identifying by iov
    cout << "Inserting dataset..." << flush;
    econn->insertDataSet(&dataset_lmf, &lmfiov);
    econn->insertDataSet(&dataset_lraw, &lmfiov);
    econn->insertDataSet(&dataset_lpn, &lmfiov);
    cout << "Done." << endl;

 
  };



  /**
   *  Write a data set
   */
  template<class DATT, class IOVT>
  void testTable(DATT* data, IOVT* iov, EcalLogicID* ecid)
  {
    tablesTried++;
    try {
      cout << "Table " << tablesTried << "..." << flush;
      map<EcalLogicID, DATT> dataset;
      dataset[*ecid] = *data;
      econn->insertDataSet(&dataset, iov);
      cout << "write OK..." << flush;
      dataset.clear();
      econn->fetchDataSet(&dataset, iov);
      if (!dataset.size()) {
	throw(runtime_error("Zero rows read back"));
      }
      cout << "read OK" << endl;
      tablesOK++;
    } catch (runtime_error &e) {
      cout << "testTable FAILED:  " << e.what() << endl;
    } catch (...) {
      cout << "testTable FAILED:  unknown exception" << endl;
    }
  }





private:
  CondDBApp();  // hidden default constructor
  EcalCondDBInterface* econn;
  string locations[4];
  uint64_t startmicros;
  uint64_t endmicros;
  run_t startrun;
  run_t endrun;

  int tablesTried;
  int tablesOK;

  /**
   *   Iterate through the dataset and print some data
   */
  void printDataSet( const map<EcalLogicID, LMFLaserBlueRawDat>* dataset, int limit = 0 ) const
  {
    cout << "==========printDataSet()" << endl;
    if (dataset->size() == 0) {
      cout << "No data in map!" << endl;
    }
    EcalLogicID ecid;
    LMFLaserBlueRawDat bluelaser;

    int count = 0;
    typedef map< EcalLogicID, LMFLaserBlueRawDat >::const_iterator CI;
    for (CI p = dataset->begin(); p != dataset->end(); p++) {
      count++;
      if (limit && count > limit) { return; }
      ecid = p->first;
      bluelaser  = p->second;
     
      cout << "SM:                     " << ecid.getID1() << endl;
      cout << "Xtal:                   " << ecid.getID2() << endl;
      cout << "APD Peak:               " << bluelaser.getAPDPeak() << endl;
      cout << "========================" << endl;
    }
    cout << endl;
  }


  /**
   *   Print out a RunTag
   */
  void printTag( const RunTag* tag) const
  {
    cout << endl;
    cout << "=============RunTag:" << endl;
    cout << "GeneralTag:         " << tag->getGeneralTag() << endl;
    cout << "Location:           " << tag->getLocationDef().getLocation() << endl;
    cout << "Run Type:           " << tag->getRunTypeDef().getRunType() << endl;
    cout << "====================" << endl;
  }

  void printIOV( const RunIOV* iov) const
  {
    cout << endl;
    cout << "=============RunIOV:" << endl;
    RunTag tag = iov->getRunTag();
    printTag(&tag);
    cout << "Run Number:         " << iov->getRunNumber() << endl;
    cout << "Run Start:          " << iov->getRunStart().str() << endl;
    cout << "Run End:            " << iov->getRunEnd().str() << endl;
    cout << "====================" << endl;
  }
};



int main (int argc, char* argv[])
{
  string host;
  string sid;
  string user;
  string pass;
  string run_num;

  if (argc != 6) {
    cout << "Usage:" << endl;
    cout << "  " << argv[0] << " <host> <SID> <user> <pass> <run_num>" << endl;
    exit(-1);
  }

  host = argv[1];
  sid = argv[2];
  user = argv[3];
  pass = argv[4];
  run_num= argv[5];

  try {
    CondDBApp app(host, sid, user, pass);

    app.testWrite(run_num);
  } catch (exception &e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }

  cout << "All Done." << endl;

  return 0;
}
