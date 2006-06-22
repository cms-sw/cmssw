/** 
 * Analyzer for calculating CFEB SCA pedestal.
 * author O.Boeriu 23/06/06 
 * runs over multiple DDUs
 * takes variable size chambers & layers 
 * produces histograms & ntuple 
 */

#include <iostream>
#include <time.h>
#include <sys/stat.h>	
#include <unistd.h>
#include <fstream>

#include "OnlineDB/CSCCondDB/interface/cscmap.h"
#include "OnlineDB/CSCCondDB/interface/CSCOnlineDB.h"
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"

class CSCscaAnalyzer : public edm::EDAnalyzer {
 public:
  explicit CSCscaAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
   
#define CHAMBERS_sca 18
#define LAYERS_sca 6
#define STRIPS_sca 80
#define TIMEBINS_sca 8
#define DDU_sca 36
#define TOTALSTRIPS_sca 480
#define TOTALEVENTS_sca 320

  ~CSCscaAnalyzer(){

    //get time of Run file for DB transfer
    filein.open("../test/CSCsca.cfg");
    filein.ignore(1000,'\n');
    
    while(filein != NULL){
      lines++;
      getline(filein,PSet);
      
      if (lines==3){
	name=PSet;  
      }
    }
    
    //get name of run file from .cfg and name root output after that
    string::size_type runNameStart = name.find("RunNum",0);
    string::size_type runNameEnd   = name.find("bin",0);
    string::size_type rootStart    = name.find("Evs",0);
    int nameSize = runNameEnd+3-runNameStart;
    int myRootSize = rootStart-runNameStart;
    std::string myname= name.substr(runNameStart,nameSize);
    std::string myRootName= name.substr(runNameStart,myRootSize);
    std::string myRootType = "SCA";
    std::string myRootEnd = ".root";
    std::string runFile= myRootName;
    std::string myRootFileName = myRootType+runFile+myRootEnd;
    //const char *myNewName=myRootFileName.c_str();
    
    struct tm* clock;			    
    struct stat attrib;			    
    stat(myname.c_str(), &attrib);          
    clock = localtime(&(attrib.st_mtime));  
    std::string myTime=asctime(clock);
    
    //DB object and map
    //CSCobject *cn = new CSCobject();
    //CSCobject *cn1 = new CSCobject();
    //cscmap *map = new cscmap();
    //condbon *dbon = new condbon();
  }

 private:

  int eventNumber,evt,strip,misMatch,fff,ret_code,length,Nddu,myevt;
  int chamber,layer,reportedChambers,chamber_num,sector,run,NChambers ;
  int dmbID[CHAMBERS_sca],crateID[CHAMBERS_sca],size[CHAMBERS_sca];
  std::vector<int> adc;
  std::string chamber_id;
  int lines;
  std::ifstream filein;
  string PSet,name;
  bool debug;
  int flag;
  float pedMean;
  int scaBlock,trigTime,lctPhase,power,cap,scaNumber;
};
