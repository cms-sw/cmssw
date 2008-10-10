/*
 *  Fabian Stoeckli
 *  Feb. 2007
 */


#include "GeneratorInterface/MCatNLOInterface/interface/MCatNLOSource.h"
#include "GeneratorInterface/MCatNLOInterface/interface/HWRGEN.h"
#include "GeneratorInterface/MCatNLOInterface/interface/Dummies.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/HepMCProduct/interface/GenInfoProduct.h"
#include "FWCore/Framework/interface/Run.h"

#include <iostream>
#include <fstream>
#include "time.h"
#include <ctype.h>

// include Herwig stuff
#include "HepMC/HerwigWrapper6_4.h"
#include "HepMC/IO_HERWIG.h"
#include "herwig.h"

extern"C" {
  void setpdfpath_(char*);
  void mysetpdfpath_(char*);
  void setlhaparm_(char*);
  void setherwpdf_(void);
  // function to chatch 'STOP' in original HWWARN:
//void cmsending_(int*);
}

#define setpdfpath setpdfpath_
#define mysetpdfpath mysetpdfpath_
#define setlhaparm setlhaparm_
#define setherwpdf setherwpdf_
//#define cmsending cmsending_

using namespace edm;
using namespace std;

MCatNLOSource::MCatNLOSource( const ParameterSet & pset, InputSourceDescription const& desc ) :
  GeneratedInputSource(pset, desc), evt(0), 
  doHardEvents_(pset.getUntrackedParameter<bool>("doHardEvents",true)),
  mcatnloVerbosity_(pset.getUntrackedParameter<int>("mcatnloVerbosity",0)),
  herwigVerbosity_ (pset.getUntrackedParameter<int>("herwigVerbosity",0)),
  herwigHepMCVerbosity_ (pset.getUntrackedParameter<bool>("herwigHepMCVerbosity",false)),
  maxEventsToPrint_ (pset.getUntrackedParameter<int>("maxEventsToPrint",0)),
  comenergy(pset.getUntrackedParameter<double>("comEnergy",14000.)),
  processNumber_(pset.getUntrackedParameter<int>("processNumber",0)),
  numEvents_(pset.getUntrackedParameter<int>("numHardEvents",maxEvents())),
  stringFileName_(pset.getUntrackedParameter<string>("stringFileName",std::string("stringInput"))),
  lhapdfSetPath_(pset.getUntrackedParameter<string>("lhapdfSetPath",std::string(""))),
  useJimmy_(pset.getUntrackedParameter<bool>("useJimmy",true)),
  doMPInteraction_(pset.getUntrackedParameter<bool>("doMPInteraction",true)),
  printCards_(pset.getUntrackedParameter<bool>("printCards",true)),
  eventCounter_(0),
  extCrossSect(pset.getUntrackedParameter<double>("crossSection", -1.)),
  extFilterEff(pset.getUntrackedParameter<double>("filterEfficiency", -1.))
{
   std::ostringstream header_str;

  header_str << "----------------------------------------------" << "\n";
  header_str << "Initializing MCatNLOSource" << "\n";
  header_str << "----------------------------------------------" << "\n";
  /*check for MC@NLO verbosity mode:
                     0 :  print default info
		     1 :  + print MC@NLO output
		     2 :  + print bases integration information
		     3 :  + print spring event generation information
    herwigVerbosity Level IPRINT
    valid argumets are:  0: print title only
                         1: + print selected input parameters
                         2: + print table of particle codes and properties
			 3: + tables of Sudakov form factors  
  */
  if(numEvents_<1)
    throw edm::Exception(edm::errors::Configuration,"MCatNLOError") 
      <<" Number of input events not set: Either use maxEvents input > 0 or numHardEvents > maxEvents output."; 

  fstbases.basesoutput = mcatnloVerbosity_;
  header_str << "   MC@NLO verbosity level         = " << fstbases.basesoutput << "\n";
  header_str << "   Herwig verbosity level         = " << herwigVerbosity_ << "\n";
  header_str << "   HepMC verbosity                = " << herwigHepMCVerbosity_ << "\n";
  header_str << "   Number of events to be printed = " << maxEventsToPrint_ << "\n";
  if(useJimmy_) {
    header_str << "   HERWIG will be using JIMMY for UE/MI." << "\n";
    if(doMPInteraction_) 
      header_str << "   JIMMY trying to generate multiple interactions." << "\n";
  }

  // setting up lhapdf path name from environment varaible (***)
  char* lhaPdfs = NULL;
  header_str<<"   Trying to find LHAPATH in environment ...";
  lhaPdfs = getenv("LHAPATH");
  if(lhaPdfs != NULL) {
    header_str<<" done.\n";
    lhapdfSetPath_=std::string(lhaPdfs);
  }
  else
    header_str<<" failed.\n";

  // set some MC@NLO parameters ...
  params.mmmaxevt = numEvents_;
  params.mmiproc=processNumber_;
  params.mmit1 = 10;
  params.mmit2 = 10;
  // we only allow for proton-proton collision
  params.mmpart1[0]='P';
  params.mmpart2[0]='P';
  for(int k=1;k<4; ++k) {
    params.mmpart1[k]=' ';
    params.mmpart2[k]=' ';
  }
  
  // Set MC@NLO parameters in a single ParameterSet
  ParameterSet mcatnlo_params = pset.getParameter<ParameterSet>("MCatNLOParameters") ;
  vector<string> setNames1 = mcatnlo_params.getParameter<vector<string> >("parameterSets");  
  // Loop over the sets
  for ( unsigned i=0; i<setNames1.size(); ++i ) {  
    string mySet = setNames1[i];
    // Read the MC@NLO parameters for each set of parameters
    vector<string> pars = mcatnlo_params.getParameter<vector<string> >(mySet);
    header_str << "----------------------------------------------" << "\n";
    header_str << "Read MC@NLO parameter set " << mySet << "\n";
    header_str << "----------------------------------------------" << "\n";

    // set parameters for string input ...
    directory[0]='\0';
    prefix_bases[0]='\0';
    prefix_events[0]='\0';

    // Loop over all parameters and stop in case of mistake
    for( vector<string>::const_iterator itPar = pars.begin(); itPar != pars.end(); ++itPar ) {
      if(!(this->give(*itPar))) {
	throw edm::Exception(edm::errors::Configuration,"MCatNLOError") 
	  <<" MCatNLO did not accept the following \""<<*itPar<<"\""; 
      }
      else if(printCards_) {
	header_str << "   " << *itPar << "\n";
      }
    }
  }
  
  header_str << "----------------------------------------------" << "\n";
  header_str << "Setting MCatNLO random number generator seed." << "\n";
  header_str << "----------------------------------------------" << "\n";
  edm::Service<RandomNumberGenerator> rng;
  randomEngine = &(rng->getEngine());
  int seed = rng->mySeed();
  double x[5];
  int s = seed;
  for (int i=0; i<5; i++) {
    s = s * 29943829 - 1;
    x[i] = s * (1./(65536.*65536.));
  }
  // get main seed
  long double c;
  c = (long double)2111111111.0 * x[3] +
    1492.0 * (x[3] = x[2]) +
    1776.0 * (x[2] = x[1]) +
    5115.0 * (x[1] = x[0]) +
    x[4];
  x[4] = floorl(c);
  x[0] = c - x[4];
  x[4] = x[4] * (1./(65536.*65536.));
  params.mmiseed = int(x[0]*99999);
  header_str << "   RNDEVSEED = "<<params.mmiseed<<"\n";

  // get helper seeds for decay etc ...
  c = (long double)2111111111.0 * x[3] +
    1492.0 * (x[3] = x[2]) +
    1776.0 * (x[2] = x[1]) +
    5115.0 * (x[1] = x[0]) +
    x[4];
  x[4] = floorl(c);
  x[0] = c - x[4];
  x[4] = x[4] * (1./(65536.*65536.));
  params.mmidecseed=int(x[0]*99999);
  c = (long double)2111111111.0 * x[3] +
    1492.0 * (x[3] = x[2]) +
    1776.0 * (x[2] = x[1]) +
    5115.0 * (x[1] = x[0]) +
    x[4];
  x[4] = floorl(c);
  x[0] = c - x[4];
  x[4] = x[4] * (1./(65536.*65536.));
  params.mmifk88seed=int(x[0]*99999);


  // only LHAPDF available
  params.mmgname[0]='L';
  params.mmgname[1]='H';
  params.mmgname[2]='A';
  params.mmgname[3]='P';
  params.mmgname[4]='D';
  params.mmgname[5]='F';
  for(int k=6;k<20; ++k) params.mmgname[k]=' ';
    

  params.mmxrenmc=params.mmxren;
  params.mmxfhmc=params.mmxfh;


  // we only allow for proton-proton collision
  params.mmpart1[0]='P';
  params.mmpart2[0]='P';
  for(int k=1;k<4; ++k) {
    params.mmpart1[k]=' ';
    params.mmpart2[k]=' ';
  }
  
  createStringFile(stringFileName_);

  char pdfpath[232];
  int pathlen = lhapdfSetPath_.length();
  for(int i=0; i<pathlen; ++i) 
  pdfpath[i]=lhapdfSetPath_.at(i);
  for(int i=pathlen; i<232; ++i) 
  pdfpath[i]=' ';
  mysetpdfpath(pdfpath);

  // decide which process to call ...
  if(doHardEvents_) {
    header_str << "----------------------------------------------" << "\n";
    header_str << "Starting hard event generation." << "\n";
    header_str << "----------------------------------------------" << "\n";
    
    if(processNumber_>0) processUnknown(true);
  
    switch(abs(processNumber_)) {
    case(1705):case(1706):case(11705):case(11706): 
      processQQ(); 
      break;
    case(2850):case(2860):case(2870):case(2880):case(12850):case(12860):case(12870):case(12880): 
      processVV(); 
      break;
    case(1600):case(1601):case(1602):case(1603):case(1604):case(1605):case(1606):case(1607):case(1608):case(1609):case(11600):case(11601):
    case(11602):case(11603):case(11604):case(11605):case(11606):case(11607):case(11608):case(11609):case(1610):case(1611):case(1612):
    case(11610):case(11611):case(11612):case(1699):case(11699): 
      processHG(); 
      break;
    case(1396):case(1397):case(1497):case(1498):case(11396):case(11397):case(11497):case(11498): 
      processSB(); 
      break;
    case(1351):case(1352):case(1353):case(1354):case(1355):case(1356):case(1361):case(1362):case(1363):case(1364):case(1365):case(1366):
    case(1371):case(1372):case(1373):case(1374):case(1375):case(1376):case(1461):case(1462):case(1463):case(1471):case(1472):case(1473):
    case(11351):case(11352):case(11353):case(11354):case(11355):case(11356):case(11361):case(11362):case(11363):case(11364):case(11365):
    case(11366):case(11371):case(11372):case(11373):case(11374):case(11375):case(11376):case(11461):case(11462):case(11463):case(11471):
    case(11472):case(11473):
      processLL(); 
      break;
    case(2600):case(2601):case(2602):case(2603):case(2604):case(2605):case(2606):case(2607):case(2608):case(2609):case(2610):case(2611):case(2612):case(2699):
    case(12600):case(12601):case(12602):case(12603):case(12604):case(12605):case(12606):case(12607):case(12608):case(12609):case(12610):case(12611):case(12612):
    case(12699):case(2700):case(2701):case(2702):case(2703):case(2704):case(2705):case(2706):case(2707):case(2708):case(2709):case(2710):case(2711):
    case(2712):case(2799):case(12700):case(12701):case(12702):case(12703):case(12704):case(12705):case(12706):case(12707):case(12708):case(12709):case(12710):
    case(12711):case(12712):case(12799): 
      processVH(); 
      break;
    case(2000):case(2001):case(2004):case(2010):case(2011):case(2014):case(2020):case(2021):case(2024):case(12000):case(12001):
    case(12004):case(12010):case(12011):case(12014):case(12020):case(12021):case(12024): 
      processST(); 
      break;
    default: 
      processUnknown(false); 
      break;
    }
  }
  else {
    header_str << "----------------------------------------------" << "\n";
    header_str << "SKipping hard event generation." << "\n";
    header_str << "----------------------------------------------" << "\n";
  }
  
  edm::LogInfo("")<<header_str.str();  

  std::ostringstream header2_str;

  // ==============================  HERWIG PART =========================================

  header2_str << "----------------------------------------------" << "\n";
  header2_str << "Initializing Herwig" << "\n";
  header2_str << "----------------------------------------------" << "\n";

  hwudat();
    
  // setting basic parameters ...
  hwproc.PBEAM1 = comenergy/2.;
  hwproc.PBEAM2 = comenergy/2.;
  hwbmch.PART1[0]  = 'P';
  hwbmch.PART2[0]  = 'P';
  for(int i=1;i<8;++i){
    hwbmch.PART1[i]  = ' ';
    hwbmch.PART2[i]  = ' ';}
  
  if(useJimmy_ && doMPInteraction_) jmparm.MSFLAG = 1;

  // initialize other common block ...
  hwigin();

  // seting maximum errrors allowed
  hwevnt.MAXER = numEvents_/10;
  if(hwevnt.MAXER<100) hwevnt.MAXER=100;


  if(useJimmy_) jimmin();

  // set some 'non-herwig' defaults
  hwevnt.MAXPR =  maxEventsToPrint_;           // no printing out of events
  hwpram.IPRINT = herwigVerbosity_;            // HERWIG print out mode
  hwprop.RMASS[6] = params.mmxmt;              // top mass 
  hwproc.IPROC = processNumber_;

  // set HERWIG PDF's to LHAPDF
  setherwpdf();

  // setting pdfs to MCatNLO pdf's
  hwpram.MODPDF[0]=params.mmidpdfset;
  hwpram.MODPDF[1]=params.mmidpdfset;
  

  // check process code and set necessary HERWIG parameters
  int jpr0 = (abs(hwproc.IPROC)%10000);
  int jpr = jpr0/100;
  if(jpr == 13 || jpr == 14) {
    if(jpr0 == 1396) {
      hwhard.EMMIN = params.mmv1massinf;
      hwhard.EMMAX = params.mmv1masssup;
    }
    else if(jpr0 == 1397) {
      hwprop.RMASS[200] = params.mmxzm;
      hwpram.GAMZ = params.mmxzw;
      hwbosc.GAMMAX = params.mmv1gammax;
    }
    else if(jpr0 == 1497 || jpr0 == 1498) {
      hwprop.RMASS[198] = params.mmxwm;
      hwpram.GAMW = params.mmxww;
      hwbosc.GAMMAX = params.mmv1gammax;
    }
    else if((jpr0 >= 1350 && jpr0 <= 1356) || (jpr0 >= 1361 && jpr0 <= 1366)) {
      hwprop.RMASS[200] = params.mmxzm;
      hwpram.GAMZ = params.mmxzw;
      hwbosc.GAMMAX = params.mmv1gammax;
      hwhard.EMMIN = params.mmv1massinf;
      hwhard.EMMAX = params.mmv1masssup;
    }
    else if(jpr0 >= 1371 && jpr0 <= 1373) {
      hwhard.EMMIN = params.mmv1massinf;
      hwhard.EMMAX = params.mmv1masssup;
    }
    else if((jpr0 >= 1450 && jpr0 <= 1453) 
	    || (jpr0 >= 1461 && jpr0 <= 1463)
	    || (jpr0 >= 1471 && jpr0 <= 1473)) {
      hwprop.RMASS[198] = params.mmxwm;
      hwpram.GAMW = params.mmxww;
      hwbosc.GAMMAX = params.mmv1gammax;    
      hwhard.EMMIN = params.mmv1massinf;
      hwhard.EMMAX = params.mmv1masssup;
    }
  }
  else if(jpr == 28) {
    hwprop.RMASS[198] = params.mmxwm;
    hwpram.GAMW = params.mmxww;
    hwprop.RMASS[200] = params.mmxzm;
    hwpram.GAMZ = params.mmxzw;
    if(params.mmv1gammax>params.mmv2gammax)
      hwbosc.GAMMAX = params.mmv1gammax;
    else
      hwbosc.GAMMAX = params.mmv2gammax;
  }
  else if(jpr == 16) {
    hwprop.RMASS[201] = params.mmxmh0;
    hwprop.RMASS[6] = params.mmxmt;
  }
  else if(jpr == 17) {
    if(abs(hwproc.IPROC)==1705 || abs(hwproc.IPROC)==11705) 
      hwprop.RMASS[5]= params.mmxmt;
    else if (abs(hwproc.IPROC)==1706 || abs(hwproc.IPROC)==11706) {
      hwprop.RMASS[6]= params.mmxmt;
      hwprop.RMASS[198] = params.mmxwm;
    }
  }
  else if(jpr == 26) {
    hwprop.RMASS[198] = params.mmxwm;
    hwpram.GAMW = params.mmxww;
    hwprop.RMASS[201] = params.mmxmh0;
  }
  else if(jpr == 27) {
    hwprop.RMASS[200] = params.mmxzm;
    hwpram.GAMZ = params.mmxzw;
    hwprop.RMASS[201] = params.mmxmh0;
  }
  else if(jpr == 20) {
    hwprop.RMASS[6] = params.mmxmt;
    hwprop.RMASS[198] = params.mmxwm;
  }
  else {
    throw edm::Exception(edm::errors::Configuration,"MCatNLOError")
      <<" bad process ID IPROC "<<hwproc.IPROC<<"!";
  }
  hwprop.RMASS[1] = params.mmxmass1;
  hwprop.RMASS[2] = params.mmxmass2;
  hwprop.RMASS[3] = params.mmxmass3;
  hwprop.RMASS[4] = params.mmxmass4;
  hwprop.RMASS[5] = params.mmxmass5;
  hwprop.RMASS[13] = params.mmxmass21;

  // some sensitive defaults
  hwpram.SOFTME = false;  
  hwevnt.NOWGT = false;
  hw6203.NEGWTS = true;
  hwpram.LRSUD = 0;
  hwpram.LWSUD = 77;
  hwpram.NSTRU = 8;
  hwpram.PRVTX = false;
  hwhard.PTMIN = 0.5;
  if(!hwevnt.NOWGT) {
    hwevnt.WGTMAX = 1.000001;
    hw6203.AVABW = 1.000001;
  }
  hwprop.RLTIM[6]=1.0e-23;
  hwprop.RLTIM[12]=1.0e-23;
  if(abs(hwproc.IPROC)==1705 || abs(hwproc.IPROC)==11705) 
    hwpram.PSPLT[1] = 0.5;
  
  // Set HERWIG parameters in a single ParameterSet
  ParameterSet herwig_params = 
    pset.getParameter<ParameterSet>("HerwigParameters") ;

  // The parameter sets to be read (default, min bias, user ...) in the
  // proper order.
  vector<string> setNames2 = 
    herwig_params.getParameter<vector<string> >("parameterSets");  

  // Loop over the sets
  for ( unsigned i=0; i<setNames2.size(); ++i ) {
    
    string mySet = setNames2[i];

    // Read the HERWIG parameters for each set of parameters
    vector<string> pars = 
      herwig_params.getParameter<vector<string> >(mySet);

    header2_str << "----------------------------------------------" << "\n";
    header2_str << "Read HERWIG parameter set " << mySet << "\n";
    header2_str << "----------------------------------------------" << "\n";

    // Loop over all parameters and stop in case of mistake
    for( vector<string>::const_iterator  
	   itPar = pars.begin(); itPar != pars.end(); ++itPar ) {
      static string sRandomValueSetting1("NRN(1)");
      static string sRandomValueSetting2("NRN(2)");
      if( (0 == itPar->compare(0,sRandomValueSetting1.size(),sRandomValueSetting1) )||(0 == itPar->compare(0,sRandomValueSetting2.size(),sRandomValueSetting2) )) {
	throw edm::Exception(edm::errors::Configuration,"HerwigError")
	  <<" attempted to set random number using HERWIG command 'NRN(.)'. This is not allowed.\n  Please use the RandomNumberGeneratorService to set the random number seed.";
      }

      if( ! hwgive(*itPar) ) {
	throw edm::Exception(edm::errors::Configuration,"HerwigError") 
	  <<" herwig did not accept the following \""<<*itPar<<"\"";
      }
      else if(printCards_){
		header2_str << "   " << *itPar << "\n";
      }
    }
  }


  if(vvjin.QQIN[0]!='\0') {
    header2_str<<"   HERWIG will be reading hard events from file: ";
    for(int i=0; i<50; ++i) header2_str<<vvjin.QQIN[i];
    header2_str<<"\n";
  }
  else {
    throw edm::Exception(edm::errors::Configuration,"MCatNLOError")
      <<" <prefix>.events file must be provided.\n Set the QQIN-parameter in the config file.";
  }

  if(abs(hwproc.IPROC)<=10000) {
    hwgupr.LHSOFT = true;
    header2_str <<"   HERWIG will produce underlying event."<<"\n";
  }
  else {
    hwgupr.LHSOFT = false;
    header2_str <<"   HERWIG will *not* produce underlying event."<<"\n";
  }
    
  /// make sure all quarks-antiquarks have the same mass
  for(int i=1; i<6; ++i) 
  hwprop.RMASS[i+6]=hwprop.RMASS[i];

  // set W+ to W-
  hwprop.RMASS[199]=hwprop.RMASS[198];
  
  // setting up herwig RNG seeds NRN(.)
  header2_str << "----------------------------------------------" << "\n";
  header2_str << "Setting Herwig random number generator seeds" << "\n";
  header2_str << "----------------------------------------------" << "\n";
  c = (long double)2111111111.0 * x[3] +
    1492.0 * (x[3] = x[2]) +
    1776.0 * (x[2] = x[1]) +
    5115.0 * (x[1] = x[0]) +
    x[4];
  x[4] = floorl(c);
  x[0] = c - x[4];
  x[4] = x[4] * (1./(65536.*65536.));
  hwevnt.NRN[0]=int(x[0]*99999);
  header2_str << "   NRN(1) = "<<hwevnt.NRN[0]<<"\n";
  c = (long double)2111111111.0 * x[3] +
    1492.0 * (x[3] = x[2]) +
    1776.0 * (x[2] = x[1]) +
    5115.0 * (x[1] = x[0]) +
    x[4];
  x[4] = floorl(c);
  x[0] = c - x[4];
  hwevnt.NRN[1]=int(x[0]*99999);
  header2_str << "   NRN(2) = "<<hwevnt.NRN[1]<<"\n";

  hwuinc();

  // *** commented out the seeting stables for PI0 and B hadrons
  /*
  hwusta("PI0     ",1);
  if(jpr == 20) {
    hwusta("B+      ",1);
    hwusta("B-      ",1);
    hwusta("B_D0    ",1);
    hwusta("B_DBAR0 ",1);
    hwusta("B_S0    ",1);
    hwusta("B_SBAR0 ",1);
    hwusta("SIGMA_B+",1);
    hwusta("LMBDA_B0",1);
    hwusta("SIGMA_B-",1);
    hwusta("XI_B0   ",1);
    hwusta("XI_B-   ",1);
    hwusta("OMEGA_B-",1);
    hwusta("B_C-    ",1);
    hwusta("UPSLON1S",1);
    hwusta("SGM_BBR-",1);
    hwusta("LMD_BBR0",1);
    hwusta("SGM_BBR+",1);
    hwusta("XI_BBAR0",1);
    hwusta("XI_B+   ",1);
    hwusta("OMG_BBR+",1);
    hwusta("B_C+    ",1);
  }
  */

  hweini();

  if(useJimmy_) jminit();

  produces<HepMCProduct>();
  produces<GenInfoProduct, edm::InRun>();

  header2_str << "\n----------------------------------------------" << "\n";
  header2_str << "Starting event generation" << "\n";
  header2_str << "----------------------------------------------" << "\n";
  
  LogInfo("")<<header2_str.str();

}


MCatNLOSource::~MCatNLOSource()
{
  std::ostringstream footer_str;

  footer_str << "----------------------------------------------" << "\n";
  footer_str << "Event generation done" << "\n";
  footer_str << "----------------------------------------------" << "\n";

  LogInfo("")<<footer_str.str();
  
  clear();
}

void MCatNLOSource::clear() 
{

  if(useJimmy_) jmefin();
}

void MCatNLOSource::processHG() 
{
  hgmain();
}

void MCatNLOSource::processLL()
{
  getVpar();
  llmain();
}

void MCatNLOSource::processVH()
{
  getVpar();
  vhmain();
}

void MCatNLOSource::processVV()
{
  vbmain();
}

void MCatNLOSource::processQQ()
{
  qqmain();
}

void MCatNLOSource::processSB()
{
  if(processNumber_ != -1396) getVpar();
  sbmain();
}

void MCatNLOSource::processST()
{
  stmain();
}

void MCatNLOSource::processUnknown(bool positive)
{
  if(positive)
    throw edm::Exception(edm::errors::Configuration,"MCatNLOError")
      <<" Unsupported process "<<processNumber_<<". Use Herwig6Interface for positively valued process ID.";
  else
    throw edm::Exception(edm::errors::Configuration,"MCatNLOError")
      <<" Unsupported process "<<processNumber_<<". Check MCatNLO manuel for allowed process ID.";
}


bool MCatNLOSource::produce(Event & e) {

  // check if we run out of hard-events. If yes, throw exception...
  eventCounter_++;
  if(eventCounter_>numEvents_)
    throw edm::Exception(edm::errors::Configuration,"MCatNLOError") <<" No more hard events left. Either increase numHardEvents or use maxEvents { untracked int32 input = N }.";
  
  hwuine();
  hwepro();
  hwbgen();
  
  if(useJimmy_ && doMPInteraction_ && hwevnt.IERROR == 0) {
    double eventok = 0.0;
    eventok=hwmsct_dummy(&eventok);
    if(eventok > 0.5) 
      return true;
  }
  
  hwdhob();
  hwcfor();
  hwcdec();
  hwdhad();
  hwdhvy();
  hwmevt();
  hwufne();

  if(hwevnt.IERROR != 0)
    return true;

  // herwig common block conversion
  HepMC::IO_HERWIG conv;
  
  HepMC::GenEvent* evt = new HepMC::GenEvent();
  bool ok = conv.fill_next_event( evt );
  if(!ok) throw edm::Exception(edm::errors::EventCorruption,"HerwigError")
    <<" Conversion problems in event nr."<<numberEventsInRun() - remainingEvents() - 1<<".";  

  evt->set_signal_process_id(hwproc.IPROC);  
  evt->weights().push_back(hwevnt.EVWGT);
  evt->set_event_number(numberEventsInRun() - remainingEvents() - 1);
  

  if (herwigHepMCVerbosity_) {
    LogInfo("")<< "Event process = " << evt->signal_process_id() <<"\n"
	 << "----------------------" << "\n";
    evt->print();
  }

  if(evt) {
    auto_ptr<HepMCProduct> bare_product(new HepMCProduct());    
    bare_product->addHepMCData(evt );
    e.put(bare_product);
  }
  
  return true;
}

bool MCatNLOSource::hwgive(const std::string& ParameterString) {

  bool accepted = 1;

  if(!strncmp(ParameterString.c_str(),"QQIN",4))
    {
      int tostart=0;
      while(ParameterString.c_str()[tostart]!='=') tostart++;
      tostart++;
      while(ParameterString.c_str()[tostart]==' ') tostart++;
      int todo = 0;
      while(ParameterString.c_str()[todo+tostart]!='\0') {
	vvjin.QQIN[todo]=ParameterString.c_str()[todo+tostart];
	todo++;
      }
      for(int i=todo ;i <50+todo; ++i) vvjin.QQIN[i]=' ';
    }
  else if(!strncmp(ParameterString.c_str(),"IPROC",5)) {
    LogWarning("")<<" WARNING: IPROC parameter will be ignored. Use 'untracked int32 processNumber = xxx' to set IPROC.\n";
  }
  else if(!strncmp(ParameterString.c_str(),"MAXEV",5)) {
    LogWarning("")<<" WARNING: MAXEV parameter will be ignored. Use 'untracked int32 maxEvents = xxx' to set the number of events to be generated.\n";
  }  
  else if(!strncmp(ParameterString.c_str(),"AUTPDF(",7)){
    LogWarning("")<<" WARNING: AUTPDF parameter *not* suported. HERWIG will use LHAPDF only.\n";
  }
  else if(!strncmp(ParameterString.c_str(),"TAUDEC",6)){
    int tostart=0;
    while(ParameterString.c_str()[tostart]!='=') tostart++;
    tostart++;
    while(ParameterString.c_str()[tostart]==' ') tostart++;
    int todo = 0;
    while(ParameterString.c_str()[todo+tostart]!='\0') {
      hwdspn.TAUDEC[todo]=ParameterString.c_str()[todo+tostart];
      todo++;
    }
    if(todo != 6) {
      throw edm::Exception(edm::errors::Configuration,"HerwigError")
	<<" Attempted to set TAUDEC to "<<hwdspn.TAUDEC<<". This is not allowed.\n Options for TAUDEC are HERWIG and TAUOLA.";
    }
  }
  else if(!strncmp(ParameterString.c_str(),"BDECAY",6)){
    LogWarning("")<<" WARNING: BDECAY parameter *not* suported. HERWIG will use default b decay.\n";
      }
  else if(!strncmp(ParameterString.c_str(),"QCDLAM",6))
    hwpram.QCDLAM = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);
  else if(!strncmp(ParameterString.c_str(),"VQCUT",5))
    hwpram.VQCUT = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"VGCUT",5))
    hwpram.VGCUT = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"VPCUT",5))
    hwpram.VPCUT = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"CLMAX",5))
    hwpram.CLMAX = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"CLPOW",5))
    hwpram.CLPOW = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"PSPLT(1)",8))
    hwpram.PSPLT[0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"PSPLT(2)",8))
    hwpram.PSPLT[1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"QDIQK",5))
    hwpram.QDIQK = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"PDIQK",5))
    hwpram.PDIQK = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"QSPAC",5))
    hwpram.QSPAC = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"PTRMS",5))
    hwpram.PTRMS = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"IPRINT",6))
    hwpram.IPRINT = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"PRVTX",5))
    hwpram.PRVTX = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"NPRFMT",6))
    hwpram.NPRFMT = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"PRNDEC",6))
    hwpram.PRNDEC = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"PRNDEF",6))
    hwpram.PRNDEF = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"PRNTEX",6))
    hwpram.PRNTEX = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"PRNWEB",6))
    hwpram.PRNWEB = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"MAXPR",5))
    hwevnt.MAXPR = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"MAXER",5))
    hwevnt.MAXER = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"LWEVT",5))
    hwevnt.LWEVT = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"LRSUD",5))
    hwpram.LRSUD = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"LWSUD",5))
    hwpram.LWSUD = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"NRN(1)",6))
    hwevnt.NRN[0] = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"NRN(2)",6))
    hwevnt.NRN[1] = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"WGTMAX",6))
    hwevnt.WGTMAX = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"NOWGT",5))
    hwevnt.NOWGT = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"AVWGT",5))
    hwevnt.AVWGT = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"AZSOFT",6))
    hwpram.AZSOFT = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"AZSPIN",6))
    hwpram.AZSPIN = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"HARDME",6))
    hwpram.HARDME = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"SOFTME",6))
    hwpram.SOFTME = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"GCUTME",6))
    hwpram.GCUTME = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"NCOLO",5))
    hwpram.NCOLO = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"NFLAV",5))
    hwpram.NFLAV = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"MODPDF(1)",9))
    hwpram.MODPDF[0] = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);
  else if(!strncmp(ParameterString.c_str(),"MODPDF(2)",9))
    hwpram.MODPDF[1] = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);
  else if(!strncmp(ParameterString.c_str(),"NSTRU",5))
    hwpram.NSTRU = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"PRSOF",5))
    hwpram.PRSOF = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"ENSOF",5))
    hwpram.ENSOF = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"IOPREM",6))
    hwpram.IOPREM = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"BTCLM",5))
    hwpram.BTCLM = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"ETAMIX",6))
    hwpram.ETAMIX = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"PHIMIX",6))
    hwpram.PHIMIX = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"H1MIX",5))
    hwpram.H1MIX = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"F0MIX",5))
    hwpram.F0MIX = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"F1MIX",5))
    hwpram.F1MIX = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"F2MIX",5))
    hwpram.F2MIX = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"ET2MIX",6))
    hwpram.ET2MIX = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"OMHMIX",6))
    hwpram.OMHMIX = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"PH3MIX",6))
    hwpram.PH3MIX = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"B1LIM",5))
    hwpram.B1LIM = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"CLDIR(1)",8))
    hwpram.CLDIR[0] = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"CLDIR(2)",8))
    hwpram.CLDIR[1] = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);   
  else if(!strncmp(ParameterString.c_str(),"CLSMR(1)",8))
    hwpram.CLSMR[0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"CLSMR(2)",8))
    hwpram.CLSMR[1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"RMASS(1)",8))
    hwprop.RMASS[1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"RMASS(2)",8))
    hwprop.RMASS[2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"RMASS(3)",8))
    hwprop.RMASS[3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"RMASS(4)",8))
    hwprop.RMASS[4] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"RMASS(5)",8))
    hwprop.RMASS[5] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"RMASS(6)",8))
    hwprop.RMASS[6] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"RMASS(13)",9))
    hwprop.RMASS[13] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"SUDORD",6))
    hwusud.SUDORD = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"INTER",5))
    hwusud.INTER = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"NEGWTS",6))
    hw6203.NEGWTS = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"AVABW",5))
    hw6203.AVABW = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"PMBN1",5))
    hwminb.PMBN1 = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"PMBN2",5))
    hwminb.PMBN2 = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"PMBN3",5))
    hwminb.PMBN3 = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"PMBK1",5))
    hwminb.PMBK1 = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"PMBK2",5))
    hwminb.PMBK2 = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"PMBM1",5))
    hwminb.PMBM1 = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"PMBM2",5))
    hwminb.PMBM2 = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"PMBP1",5))
    hwminb.PMBP1 = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"PMBP2",5))
    hwminb.PMBP2 = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"PMBP3",5))
    hwminb.PMBP3 = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"VMIN2",5))
    hwdist.VMIN2 = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"EXAG",4))
    hwdist.EXAG = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"PRECO",5))
    hwuclu.PRECO = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"CLRECO",6))
    hwuclu.CLRECO = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"PWT(1)",6))
    hwuwts.PWT[0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"PWT(2)",6))
    hwuwts.PWT[1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"PWT(3)",6))
    hwuwts.PWT[2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"PWT(4)",6))
    hwuwts.PWT[3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"PWT(5)",6))
    hwuwts.PWT[4] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"PWT(6)",6))
    hwuwts.PWT[5] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"PWT(7)",6))
    hwuwts.PWT[6] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,0,0)",12))
    hwuwts.REPWT[0][0][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,0,1)",12))
    hwuwts.REPWT[0][0][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,0,2)",12))
    hwuwts.REPWT[0][0][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,0,3)",12))
    hwuwts.REPWT[0][0][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,0,4)",12))
    hwuwts.REPWT[0][0][4] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,1,0)",12))
    hwuwts.REPWT[0][1][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,1,1)",12))
    hwuwts.REPWT[0][1][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,1,2)",12))
    hwuwts.REPWT[0][1][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,1,3)",12))
    hwuwts.REPWT[0][1][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,1,4)",12))
    hwuwts.REPWT[0][1][4] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,2,0)",12))
    hwuwts.REPWT[0][2][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,2,1)",12))
    hwuwts.REPWT[0][2][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,2,2)",12))
    hwuwts.REPWT[0][2][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,2,3)",12))
    hwuwts.REPWT[0][2][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,2,4)",12))
    hwuwts.REPWT[0][2][4] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,3,0)",12))
    hwuwts.REPWT[0][3][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,3,1)",12))
    hwuwts.REPWT[0][3][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,3,2)",12))
    hwuwts.REPWT[0][3][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,3,3)",12))
    hwuwts.REPWT[0][3][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,3,4)",12))
    hwuwts.REPWT[0][3][4] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,4,0)",12))
    hwuwts.REPWT[0][4][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,4,1)",12))
    hwuwts.REPWT[0][4][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,4,2)",12))
    hwuwts.REPWT[0][4][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,4,3)",12))
    hwuwts.REPWT[0][4][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,4,4)",12))
    hwuwts.REPWT[0][4][4] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,0,0)",12))
    hwuwts.REPWT[1][0][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,0,1)",12))
    hwuwts.REPWT[1][0][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,0,2)",12))
    hwuwts.REPWT[1][0][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,0,3)",12))
    hwuwts.REPWT[1][0][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,0,4)",12))
    hwuwts.REPWT[1][0][4] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,1,0)",12))
    hwuwts.REPWT[1][1][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,1,1)",12))
    hwuwts.REPWT[1][1][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,1,2)",12))
    hwuwts.REPWT[1][1][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,1,3)",12))
    hwuwts.REPWT[1][1][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,1,4)",12))
    hwuwts.REPWT[1][1][4] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,2,0)",12))
    hwuwts.REPWT[1][2][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,2,1)",12))
    hwuwts.REPWT[1][2][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,2,2)",12))
    hwuwts.REPWT[1][2][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,2,3)",12))
    hwuwts.REPWT[1][2][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,2,4)",12))
    hwuwts.REPWT[1][2][4] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,3,0)",12))
    hwuwts.REPWT[1][3][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,3,1)",12))
    hwuwts.REPWT[1][3][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,3,2)",12))
    hwuwts.REPWT[1][3][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,3,3)",12))
    hwuwts.REPWT[1][3][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,3,4)",12))
    hwuwts.REPWT[1][3][4] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,4,0)",12))
    hwuwts.REPWT[1][4][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,4,1)",12))
    hwuwts.REPWT[1][4][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,4,2)",12))
    hwuwts.REPWT[1][4][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,4,3)",12))
    hwuwts.REPWT[1][4][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,4,4)",12))
    hwuwts.REPWT[1][4][4] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,0,0)",12))
    hwuwts.REPWT[2][0][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,0,1)",12))
    hwuwts.REPWT[2][0][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,0,2)",12))
    hwuwts.REPWT[2][0][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,0,3)",12))
    hwuwts.REPWT[2][0][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,0,4)",12))
    hwuwts.REPWT[2][0][4] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,1,0)",12))
    hwuwts.REPWT[2][1][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,1,1)",12))
    hwuwts.REPWT[2][1][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,1,2)",12))
    hwuwts.REPWT[2][1][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,1,3)",12))
    hwuwts.REPWT[2][1][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,1,4)",12))
    hwuwts.REPWT[2][1][4] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,2,0)",12))
    hwuwts.REPWT[2][2][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,2,1)",12))
    hwuwts.REPWT[2][2][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,2,2)",12))
    hwuwts.REPWT[2][2][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,2,3)",12))
    hwuwts.REPWT[2][2][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,2,4)",12))
    hwuwts.REPWT[2][2][4] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,3,0)",12))
    hwuwts.REPWT[2][3][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,3,1)",12))
    hwuwts.REPWT[2][3][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,3,2)",12))
    hwuwts.REPWT[2][3][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,3,3)",12))
    hwuwts.REPWT[2][3][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,3,4)",12))
    hwuwts.REPWT[2][3][4] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,4,0)",12))
    hwuwts.REPWT[2][4][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,4,1)",12))
    hwuwts.REPWT[2][4][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,4,2)",12))
    hwuwts.REPWT[2][4][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,4,3)",12))
    hwuwts.REPWT[2][4][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,4,4)",12))
    hwuwts.REPWT[2][4][4] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,0,0)",12))
    hwuwts.REPWT[3][0][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,0,1)",12))
    hwuwts.REPWT[3][0][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,0,2)",12))
    hwuwts.REPWT[3][0][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,0,3)",12))
    hwuwts.REPWT[3][0][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,0,4)",12))
    hwuwts.REPWT[3][0][4] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,1,0)",12))
    hwuwts.REPWT[3][1][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,1,1)",12))
    hwuwts.REPWT[3][1][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,1,2)",12))
    hwuwts.REPWT[3][1][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,1,3)",12))
    hwuwts.REPWT[3][1][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,1,4)",12))
    hwuwts.REPWT[3][1][4] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,2,0)",12))
    hwuwts.REPWT[3][2][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,2,1)",12))
    hwuwts.REPWT[3][2][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,2,2)",12))
    hwuwts.REPWT[3][2][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,2,3)",12))
    hwuwts.REPWT[3][2][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,2,4)",12))
    hwuwts.REPWT[3][2][4] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,3,0)",12))
    hwuwts.REPWT[3][3][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,3,1)",12))
    hwuwts.REPWT[3][3][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,3,2)",12))
    hwuwts.REPWT[3][3][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,3,3)",12))
    hwuwts.REPWT[3][3][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,3,4)",12))
    hwuwts.REPWT[3][3][4] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,4,0)",12))
    hwuwts.REPWT[3][4][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,4,1)",12))
    hwuwts.REPWT[3][4][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,4,2)",12))
    hwuwts.REPWT[3][4][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,4,3)",12))
    hwuwts.REPWT[3][4][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,4,4)",12))
    hwuwts.REPWT[3][4][4] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"SNGWT",5))
    hwuwts.SNGWT = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"DECWT",5))
    hwuwts.DECWT = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"PLTCUT",6))
    hwdist.PLTCUT = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"VTOCDK(",7)){
    // we find the index ...
    int ind = atoi(&ParameterString[7]);  
    hwprop.VTOCDK[ind] = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);}
  else if(!strncmp(ParameterString.c_str(),"VTORDK(",7)){
    // we find the index ...
    int ind = atoi(&ParameterString[7]);  
    hwprop.VTORDK[ind] = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);}
  else if(!strncmp(ParameterString.c_str(),"PIPSMR",6))
    hwdist.PIPSMR = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"VIPWID(1)",9))
    hw6202.VIPWID[0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"VIPWID(2)",9))
    hw6202.VIPWID[1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"VIPWID(3)",9))
    hw6202.VIPWID[2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"MAXDKL",6))
    hwdist.MAXDKL = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"IOPDKL",6))
    hwdist.IOPDKL = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"DXRCYL",6))
    hw6202.DXRCYL = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"DXZMAX",6))
    hw6202.DXZMAX = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"DXRSPH",6))
    hw6202.DXRSPH = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  //  else if(!strncmp(ParameterString.c_str(),"BDECAY",6))
  //    hwprch.BDECAY = ParameterString[strcspn(ParameterString.c_str(),"=")+1];  
  else if(!strncmp(ParameterString.c_str(),"MIXING",6))
    hwdist.MIXING = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"XMIX(1)",7))
    hwdist.XMIX[0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"XMIX(2)",7))
    hwdist.XMIX[1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"YMIX(1)",7))
    hwdist.YMIX[0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"YMIX(2)",7))
    hwdist.YMIX[1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"RMASS(198)",10))
    hwprop.RMASS[198] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);      
  else if(!strncmp(ParameterString.c_str(),"RMASS(199)",10))
    hwprop.RMASS[199] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);      
  else if(!strncmp(ParameterString.c_str(),"GAMW",4))
    hwpram.GAMW = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);      
  else if(!strncmp(ParameterString.c_str(),"GAMZ",4))
    hwpram.GAMZ = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);      
  else if(!strncmp(ParameterString.c_str(),"RMASS(200)",10))
    hwprop.RMASS[200] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"WZRFR",5))
    hw6202.WZRFR = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"MODBOS(",7)) {
    int ind = atoi(&ParameterString[7]);
    hwbosc.MODBOS[ind-1] = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"RMASS(201)",10))
    hwprop.RMASS[201] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"IOPHIG",6))
    hwbosc.IOPHIG = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"GAMMAX",6))
    hwbosc.GAMMAX = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"ENHANC(",7)) {
    int ind = atoi(&ParameterString[7]);
    hwbosc.ENHANC[ind-1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"RMASS(209)",10))
    hwprop.RMASS[209] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"RMASS(215)",10))
    hwprop.RMASS[215] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"ALPHEM",6))
    hwpram.ALPHEM = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"SWEIN",5))
    hwpram.SWEIN = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"QFCH(",5)){
    int ind = atoi(&ParameterString[5]);
    hwpram.QFCH[ind-1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"AFCH(1,",7)){
    int ind = atoi(&ParameterString[7]); if(ind<1||ind>2) return 0;
    hwpram.AFCH[ind-1][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"AFCH(2,",7)){
    int ind = atoi(&ParameterString[7]);if(ind<1||ind>2) return 0;
    hwpram.AFCH[ind-1][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"AFCH(3,",7)){
    int ind = atoi(&ParameterString[7]);if(ind<1||ind>2) return 0;
    hwpram.AFCH[ind-1][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"AFCH(4,",7)){
    int ind = atoi(&ParameterString[7]);if(ind<1||ind>2) return 0;
    hwpram.AFCH[ind-1][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"AFCH(5,",7)){
    int ind = atoi(&ParameterString[7]);if(ind<1||ind>2) return 0;
    hwpram.AFCH[ind-1][4] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"AFCH(6,",7)){
    int ind = atoi(&ParameterString[7]);if(ind<1||ind>2) return 0;
    hwpram.AFCH[ind-1][5] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"AFCH(7,",7)){
    int ind = atoi(&ParameterString[7]);if(ind<1||ind>2) return 0;
    hwpram.AFCH[ind-1][6] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"AFCH(8,",7)){
    int ind = atoi(&ParameterString[7]);if(ind<1||ind>2) return 0;
    hwpram.AFCH[ind-1][7] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"AFCH(9,",7)){
    int ind = atoi(&ParameterString[7]);if(ind<1||ind>2) return 0;
    hwpram.AFCH[ind-1][8] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"AFCH(10,",8)){
    int ind = atoi(&ParameterString[8]);if(ind<1||ind>2) return 0;
    hwpram.AFCH[ind-1][9] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"AFCH(11,",8)){
    int ind = atoi(&ParameterString[8]);if(ind<1||ind>2) return 0;
    hwpram.AFCH[ind-1][10] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"AFCH(12,",8)){
    int ind = atoi(&ParameterString[8]);if(ind<1||ind>2) return 0;
    hwpram.AFCH[ind-1][11] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"AFCH(13,",8)){
    int ind = atoi(&ParameterString[8]);if(ind<1||ind>2) return 0;
    hwpram.AFCH[ind-1][12] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"AFCH(14,",8)){
    int ind = atoi(&ParameterString[8]);if(ind<1||ind>2) return 0;
    hwpram.AFCH[ind-1][13] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"AFCH(15,",8)){
    int ind = atoi(&ParameterString[8]);if(ind<1||ind>2) return 0;
    hwpram.AFCH[ind-1][14] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"AFCH(16,",8)){
    int ind = atoi(&ParameterString[8]);if(ind<1||ind>2) return 0;
    hwpram.AFCH[ind-1][15] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"VFCH(1,",7)){
    int ind = atoi(&ParameterString[7]); if(ind<1||ind>2) return 0;
    hwpram.VFCH[ind-1][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"VFCH(2,",7)){
    int ind = atoi(&ParameterString[7]);if(ind<1||ind>2) return 0;
    hwpram.VFCH[ind-1][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"VFCH(3,",7)){
    int ind = atoi(&ParameterString[7]);if(ind<1||ind>2) return 0;
    hwpram.VFCH[ind-1][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"VFCH(4,",7)){
    int ind = atoi(&ParameterString[7]);if(ind<1||ind>2) return 0;
    hwpram.VFCH[ind-1][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"VFCH(5,",7)){
    int ind = atoi(&ParameterString[7]);if(ind<1||ind>2) return 0;
    hwpram.VFCH[ind-1][4] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"VFCH(6,",7)){
    int ind = atoi(&ParameterString[7]);if(ind<1||ind>2) return 0;
    hwpram.VFCH[ind-1][5] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"VFCH(7,",7)){
    int ind = atoi(&ParameterString[7]);if(ind<1||ind>2) return 0;
    hwpram.VFCH[ind-1][6] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"VFCH(8,",7)){
    int ind = atoi(&ParameterString[7]);if(ind<1||ind>2) return 0;
    hwpram.VFCH[ind-1][7] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"VFCH(9,",7)){
    int ind = atoi(&ParameterString[7]);if(ind<1||ind>2) return 0;
    hwpram.VFCH[ind-1][8] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"VFCH(10,",8)){
    int ind = atoi(&ParameterString[8]);if(ind<1||ind>2) return 0;
    hwpram.VFCH[ind-1][9] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"VFCH(11,",8)){
    int ind = atoi(&ParameterString[8]);if(ind<1||ind>2) return 0;
    hwpram.VFCH[ind-1][10] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"VFCH(12,",8)){
    int ind = atoi(&ParameterString[8]);if(ind<1||ind>2) return 0;
    hwpram.VFCH[ind-1][11] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"VFCH(13,",8)){
    int ind = atoi(&ParameterString[8]);if(ind<1||ind>2) return 0;
    hwpram.VFCH[ind-1][12] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"VFCH(14,",8)){
    int ind = atoi(&ParameterString[8]);if(ind<1||ind>2) return 0;
    hwpram.VFCH[ind-1][13] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"VFCH(15,",8)){
    int ind = atoi(&ParameterString[8]);if(ind<1||ind>2) return 0;
    hwpram.VFCH[ind-1][14] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"VFCH(16,",8)){
    int ind = atoi(&ParameterString[8]);if(ind<1||ind>2) return 0;
    hwpram.VFCH[ind-1][15] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"ZPRIME",6))
    hwpram.ZPRIME = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"RMASS(202)",10))
    hwprop.RMASS[202] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"GAMZP",5))
    hwpram.GAMZP = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"VCKM(",5)) {
    int ind1 = atoi(&ParameterString[5]);
    if(ind1<1||ind1>3) return 0;
    int ind2 = atoi(&ParameterString[7]);
    if(ind2<1||ind2>3) return 0;
    hwpram.VCKM[ind2][ind1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"SCABI",5))
    hwpram.SCABI = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"EPOLN(",6)) {
    int ind = atoi(&ParameterString[6]);
    if(ind<1||ind>3) return 0;
    hwhard.EPOLN[ind-1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"PPOLN(",6)) {
    int ind = atoi(&ParameterString[6]);
    if(ind<1||ind>3) return 0;
    hwhard.PPOLN[ind-1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); }
  else if(!strncmp(ParameterString.c_str(),"QLIM",4))
    hwhard.QLIM = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);         
  else if(!strncmp(ParameterString.c_str(),"THMAX",5))
    hwhard.THMAX = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);         
  else if(!strncmp(ParameterString.c_str(),"Y4JT",4))
    hwhard.Y4JT = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);         
  else if(!strncmp(ParameterString.c_str(),"DURHAM",6))
    hwhard.DURHAM = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);         
  else if(!strncmp(ParameterString.c_str(),"IOP4JT(1)",9))
    hwpram.IOP4JT[0] = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);         
  else if(!strncmp(ParameterString.c_str(),"IOP4JT(2)",9))
    hwpram.IOP4JT[1] = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);         
  else if(!strncmp(ParameterString.c_str(),"BGSHAT",6))
    hwhard.BGSHAT = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);         
  else if(!strncmp(ParameterString.c_str(),"BREIT",5))
    hwbrch.BREIT = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);         
  else if(!strncmp(ParameterString.c_str(),"USECMF",6))
    hwbrch.USECMF = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);         
  else if(!strncmp(ParameterString.c_str(),"NOSPAC",6))
    hwpram.NOSPAC = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);         
  else if(!strncmp(ParameterString.c_str(),"ISPAC",5))
    hwpram.ISPAC = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);         
  else if(!strncmp(ParameterString.c_str(),"TMNISR",6))
    hwhard.TMNISR = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);         
  else if(!strncmp(ParameterString.c_str(),"ZMXISR",6))
    hwhard.ZMXISR = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);         
  else if(!strncmp(ParameterString.c_str(),"ASFIXD",6))
    hwhard.ASFIXD = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);         
  else if(!strncmp(ParameterString.c_str(),"OMEGA0",6))
    hwhard.OMEGA0 = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);         
  else if(!strncmp(ParameterString.c_str(),"IAPHIG",6))
    hwhard.IAPHIG = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);         
  else if(!strncmp(ParameterString.c_str(),"PHOMAS",6))
    hwhard.PHOMAS = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);         
  else if(!strncmp(ParameterString.c_str(),"PRESPL",6))
    hw6500.PRESPL = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);            
  else if(!strncmp(ParameterString.c_str(),"PTMIN",5))
    hwhard.PTMIN = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"PTMAX",5))
    hwhard.PTMAX = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"PTPOW",5))
    hwhard.PTPOW = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"YJMIN",5))
    hwhard.YJMIN = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"YJMAX",5))
    hwhard.YJMAX = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"EMMIN",5))
    hwhard.EMMIN = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"EMMAX",5))
    hwhard.EMMAX = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"EMPOW",5))
    hwhard.EMPOW = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"Q2MIN",5))
    hwhard.Q2MIN = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"Q2MAX",5))
    hwhard.Q2MAX = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"Q2POW",5))
    hwhard.Q2POW = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"YBMIN",5))
    hwhard.YBMIN = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"YBMAX",5))
    hwhard.YBMAX = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"WHMIN",5))
    hwhard.WHMIN = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"ZJMAX",5))
    hwhard.ZJMAX = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"Q2WWMN",6))
    hwhard.Q2WWMN = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"Q2WWMX",6))
    hwhard.Q2WWMX = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"YWWMIN",6))
    hwhard.YWWMIN = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"YWWMAX",6))
    hwhard.YWWMAX = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"CSPEED",6))
    hwpram.CSPEED = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"GEV2NB",6))
    hwpram.GEV2NB = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"IBSH",4))
    hwhard.IBSH = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"IBRN(1)",7))
    hwhard.IBRN[0] = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"IBRN(2)",7))
    hwhard.IBRN[1] = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"NQEV",4))
    hwusud.NQEV = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"ZBINM",5))
    hwpram.ZBINM = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"NZBIN",5))
    hwpram.NZBIN = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"NBTRY",5))
    hwpram.NBTRY = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"NCTRY",5))
    hwpram.NCTRY = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"NETRY",5))
    hwpram.NETRY = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"NSTRY",5))
    hwpram.NSTRY = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"ACCUR",5))
    hwusud.ACCUR = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"RPARTY",6))
    hwrpar.RPARTY = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"SUSYIN",6))
    hwsusy.SUSYIN = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"LRSUSY",6))
    hw6202.LRSUSY = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"SYSPIN",6))
    hwdspn.SYSPIN = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"THREEB",6))
    hwdspn.THREEB = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"FOURB",5))
    hwdspn.FOURB = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"LHSOFT",6))
    hwgupr.LHSOFT = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"LHGLSF",6))
    hwgupr.LHGLSF = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"OPTM",4))
    hw6300.OPTM = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"IOPSTP",6))
    hw6300.IOPSTP = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"IOPSH",5))
    hw6300.IOPSH = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"JMUEO",5))
    jmparm.JMUEO = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"PTJIM",5))
    jmparm.PTJIM = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  else if(!strncmp(ParameterString.c_str(),"JMRAD(73)",9))
    jmparm.JMRAD[72] = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]); 
  
  else accepted = 0;
  
  return accepted;
}


#ifdef NEVER
//-------------------------------------------------------------------------------
// dummy hwaend (has to be REMOVED from herwig)
#define hwaend hwaend_

extern "C" {
  void hwaend(){/*dummy*/}
}
//-------------------------------------------------------------------------------
#endif


bool MCatNLOSource::give(const std::string& iParm )
{
  bool accepted = 1;
  if(!strncmp(iParm.c_str(),"ECM",3))
    params.mmecm = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"TWIDTH",6))
    params.mmtwidth = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"FREN",4))
    params.mmxren = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"FFACT",5))
    params.mmxfh = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"HVQMASS",7))
    params.mmxmt = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);  
  else if(!strncmp(iParm.c_str(),"WMASS",5))
    params.mmxwm = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"ZMASS",5))
    params.mmxzm = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"WWIDTH",6))
    params.mmxww = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"ZWIDTH",6))
    params.mmxzw = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"HGGMASS",7))
    params.mmxmh0 = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"HGGWIDTH",8))
    params.mmgah = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"IBORNHGG",8))
    params.mmibornex = atoi(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"V1GAMMAX",8))
    params.mmv1gammax = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"V1MASSINF",9))
    params.mmv1massinf = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"V1MASSSUP",9))
    params.mmv1masssup = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"V2GAMMAX",8))
    params.mmv2gammax = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"V2MASSINF",9))
    params.mmv2massinf = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"V2MASSSUP",9))
    params.mmv2masssup = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"HGAMMAX",7))
    params.mmgammax = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"HMASSINF",8))
    params.mmxmhl = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"HMASSSUP",8))
    params.mmxmhu = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"T1GAMMAX",8))
    para331.mmgammay1 = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"T1MASSINF",9))
    para331.mmym1low = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"T1MASSSUP",9))
    para331.mmym1upp = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"T2GAMMAX",8))
    para331.mmgammay2 = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"T2MASSINF",9))
    para331.mmym2low = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"T2MASSSUP",9))
    para331.mmym2upp = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"UMASS",5))
    params.mmxmass1 = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"DMASS",5))
    params.mmxmass2 = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"SMASS",5))
    params.mmxmass3 = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"CMASS",5))
    params.mmxmass4 = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"BMASS",5))
    params.mmxmass5 = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"GMASS",5))
    params.mmxmass21 = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"VUD",3))
    params.mmvud = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"VUS",3))
    params.mmvus = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"VUB",3))
    params.mmvub = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"VCD",3))
    params.mmvcd = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"VCS",3))
    params.mmvcs = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"VCB",3))
    params.mmvcb = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"VTD",3))
    params.mmvtd = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"VTS",3))
    params.mmvts = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"VTB",3))
    params.mmvtb = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"AEMRUN",6))
    {
      int tostart=0;
      while(iParm.c_str()[tostart]!='=') tostart++;
      tostart++;
      while(iParm.c_str()[tostart]==' ') tostart++;

      if(!strncmp(&iParm.c_str()[tostart],"YES",3))
	params.mmaemrun = 0;
      else if(!strncmp(&iParm.c_str()[tostart],"NO",2))
	params.mmaemrun = 1;
      else
	return false;
    }
  else if(!strncmp(iParm.c_str(),"IPROC",5)) 
    LogWarning("")<<" WARNING: IPROC parameter will be ignored. Use 'untracked int32 processNumber = xxx' to set IPROC.\n";
  else if(!strncmp(iParm.c_str(),"IVCODE",6))
    params.mmivcode = atoi(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"IL1CODE",7))
    params.mmil1code = atoi(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"IL2CODE",7))
    params.mmil2code = atoi(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"PART1",5) || !strncmp(iParm.c_str(),"PART2",5))
    LogWarning("")<<" WARNING: PARTi parameter will be ignored. Only proton-proton collisions supported. For proton-antiproton please go to Batavia (but hurry).\n";
  else if(!strncmp(iParm.c_str(),"PDFGROUP",8)) {
    /*
    int tostart=0;
    while(iParm.c_str()[tostart]!='=') tostart++;
    tostart++;
    while(iParm.c_str()[tostart]==' ') tostart++;
    int todo = 0;
    while(iParm.c_str()[todo+tostart]!='\0' && todo < 20) {
      params.mmgname[todo]=iParm.c_str()[todo+tostart];
      
      todo++;
      }
    for(int i=todo ;i <20; ++i) params.mmgname[i]=' ';
    */
    LogWarning("")<<" WARNING: PDFGROUP parameter will be ignored. Only LHAPDF sets supported.\n";
  }
  else if(!strncmp(iParm.c_str(),"PDFSET",6))
    params.mmidpdfset = atoi(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"LAMBDAFIVE",10))
    params.mmxlam = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"SCHEMEOFPDF",11)) {
    int tostart=0;
    while(iParm.c_str()[tostart]!='=') tostart++;
    tostart++;
    while(iParm.c_str()[tostart]==' ') tostart++;
    params.mmscheme[0]=iParm.c_str()[tostart];
    params.mmscheme[1]=iParm.c_str()[tostart+1];
  }  
  else if(!strncmp(iParm.c_str(),"LAMBDAHERW",10))
    params.mmxlamherw = atof(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"FPREFIX",7)) {
    int tostart=0;
    while(iParm.c_str()[tostart]!='=') tostart++;
    tostart++;
    while(iParm.c_str()[tostart]==' ') tostart++;
    int todo = 0;
    while(iParm.c_str()[todo+tostart]!='\0' && todo < 10) {
      prefix_bases[todo]=iParm.c_str()[todo+tostart];
      todo++;
    }
    if(todo<10) prefix_bases[todo]='\0';
  }  
  else if(!strncmp(iParm.c_str(),"EVPREFIX",8)) {
    int tostart=0;
    while(iParm.c_str()[tostart]!='=') tostart++;
    tostart++;
    while(iParm.c_str()[tostart]==' ') tostart++;
    int todo = 0;
    while(iParm.c_str()[todo+tostart]!='\0' && todo < 10) {
      prefix_events[todo]=iParm.c_str()[todo+tostart];
      todo++;
    }
    if(todo<10) prefix_events[todo]='\0';
  }  
  else if(!strncmp(iParm.c_str(),"NEVENTS",7)) 
    LogWarning("")<<" WARNING: NEVENTS parameter will be ignored. Use 'untracked int32 maxEvents = xxx' to set NEVENTS."<<"\n";
  else if(!strncmp(iParm.c_str(),"WGTTYPE",7))
    params.mmiwgtnorm = atoi(&iParm[strcspn(iParm.c_str(),"=")+1]);
  else if(!strncmp(iParm.c_str(),"RNDEVSEED",9))
    //    params.mmiseed = atoi(&iParm[strcspn(iParm.c_str(),"=")+1]);
    LogWarning("")<<" WARNING: RNDEVSEED will be ignored. Use the RandomNumberGeneratorService to set RNG seed."<<"\n";    
  else if(!strncmp(iParm.c_str(),"BASES",5)) 
    LogWarning("")<<" WARNING: BASES parameter will be ignored."<<"\n";
  else if(!strncmp(iParm.c_str(),"PDFLIBRARY",10)) 
    LogWarning("")<<" WARNING: PDFLIBRARY parameter will be ignored. Only LHAPDF is supported."<<"\n";
  else if(!strncmp(iParm.c_str(),"HERPDF",6)) 
    LogWarning("")<<" WARNING: HERPDF parameter will be ignored. Use the same PDF as for hard event generation."<<"\n";
  else if(!strncmp(iParm.c_str(),"HWPATH",6)) 
    LogWarning("")<<" WARNING: HWPATH parameter is not needed and will be ignored." << "\n"; 
  else if(!strncmp(iParm.c_str(),"HWUTI",5)) 
    LogWarning("")<<" WARNING: HWUTI parameter will be ignored. Herwig utilities not needed."<<"\n";
  else if(!strncmp(iParm.c_str(),"HERWIGVER",9)) 
    LogWarning("")<<" WARNING: HERWIGVER parameter will be ignored. Herwig library not needed."<<"\n";
  else if(!strncmp(iParm.c_str(),"LHAPATH",7)) 
    LogWarning("")<<" WARNING: LHAPATH parameter will be ignored. Use the <untracked string lhapdfSetPath> parameter in order to set LHAPDF path."<<"\n";
  else if(!strncmp(iParm.c_str(),"LHAOFL",6)) 
    LogWarning("")<<" WARNING: LHAOFL parameter will be ignored. *** THIS WILL CHANGE IN FURTHER RELEASE ***"<<"\n";
  else if(!strncmp(iParm.c_str(),"PDFPATH",6)) 
    LogWarning("")<<" WARNING: PDFPATH parameter will be ignored. Only LHAPDF available."<<"\n";
    else if(!strncmp(iParm.c_str(),"SCRTCH",6)) {
    int tostart=0;
    while(iParm.c_str()[tostart]!='=') tostart++;
    tostart++;
    while(iParm.c_str()[tostart]==' ') tostart++;
    int todo = 0;
    while(iParm.c_str()[todo+tostart]!='\0' && todo < 70) {
      directory[todo]=iParm.c_str()[todo+tostart];
      todo++;
    }
    if(todo<70) directory[todo]='\0';
    }  
  
  else accepted = false;
  return accepted;
}

void MCatNLOSource::getVpar()
{
  switch(abs(processNumber_)) {
  case(1397):case(11397):
  case(1351):case(1352):case(1353):case(1354):case(1355):case(1356):
  case(1361):case(1362):case(1363):case(1364):case(1365):case(1366):
  case(1371):case(1372):case(1373):case(1374):case(1375):case(1376):
  case(11351):case(11352):case(11353):case(11354):case(11355):case(11356):
  case(11361):case(11362):case(11363):case(11364):case(11365):case(11366):
  case(11371):case(11372):case(11373):case(11374):case(11375):case(11376):
    params.mmxm0 = params.mmxzm;
    params.mmgah = params.mmxzw;
    break;

  case(2700):case(2701):case(2702):case(2703):case(2704):case(2705):case(2706):
  case(2707):case(2708):case(2709):case(2710):case(2711):case(2712):case(2799):
  case(12700):case(12701):case(12702):case(12703):case(12704):case(12705):case(12706):
  case(12707):case(12708):case(12709):case(12710):case(12711):case(12712):case(12799):
    params.mmxm0v = params.mmxzm;
    params.mmgav = params.mmxzw;
    break;
  case(1497):case(11497):
  case(1498):case(11498):
  case(1461):case(1462):case(1463):case(1471):case(1472):case(1473):
  case(11461):case(11462):case(11463):case(11471):case(11472):case(11473):
    params.mmxm0 = params.mmxwm;
    params.mmgah = params.mmxww;
    break;

  case(2600):case(2601):case(2602):case(2603):case(2604):case(2605):case(2606):
  case(2607):case(2608):case(2609):case(2610):case(2611):case(2612):case(2699):
  case(12600):case(12601):case(12602):case(12603):case(12604):case(12605):case(12606):
  case(12607):case(12608):case(12609):case(12610):case(12611):case(12612):case(12699):
    params.mmxm0v = params.mmxwm;
    params.mmgav = params.mmxww;
    break;
  default:
    throw edm::Exception(edm::errors::Configuration,"MCatNLOError") <<" No such option in getVpar.";
  }
}

void MCatNLOSource::createStringFile(const std::string& fileName)
{

  bool endone = false;
  for(int i=0; i<100; ++i) {
    if(fileName.c_str()[i]=='\0') endone = true;
    if(!endone) fstbases.stfilename[i]=fileName.c_str()[i];
    else fstbases.stfilename[i]=' ';
  }

  // put together ouput-file-strings ...
  char string1[81];
  char string2[81];
  std::ofstream output;
  output.open(fileName.c_str());
  int position = 0;
  while(directory[position]!='\0' && position < 70) {
    string1[position]=directory[position];
    string2[position]=directory[position];
    position++;
  }
  int position3 = position;
  int position2 = 0;
  while(prefix_bases[position2]!='\0' && position2<10) {
    string1[position3]=prefix_bases[position2];
    position3++;
    position2++;
  }
  string1[position3]='\0';
  position3 = position;
  position2 = 0;
  while(prefix_events[position2]!='\0' && position2<10) {
    string2[position3]=prefix_events[position2];
    position3++;
    position2++;
  }
  string2[position3]='\0';
  output.put('\'');
  for(int i=0; ;++i) {
    if(string1[i]=='\0') {
      vvjin.QQIN[i]='.';
      vvjin.QQIN[i+1]='e';
      vvjin.QQIN[i+2]='v';
      vvjin.QQIN[i+3]='e';
      vvjin.QQIN[i+4]='n';
      vvjin.QQIN[i+5]='t';
      vvjin.QQIN[i+6]='s';
      vvjin.QQIN[i+7]='\0';
      break;
    }
    else {
      output.put(string1[i]);
      vvjin.QQIN[i]=string1[i];
    }
  }
  output.put('\'');
  output.put('\n');
  output.put('\'');
  for(int i=0; ;++i) {
    if(string2[i]=='\0') break;
    else output.put(string2[i]);
  }
  output.put('\'');
  output.put('\n');  
  output.close();
}

void MCatNLOSource::endRun(Run & r) {
  hwefin();
  auto_ptr<GenInfoProduct> giprod (new GenInfoProduct());
  intCrossSect = 1000.0*hwevnt.AVWGT;
  giprod->set_cross_section(intCrossSect);
  giprod->set_external_cross_section(extCrossSect);
  giprod->set_filter_efficiency(extFilterEff);
  r.put(giprod);

}

#ifdef NEVER
extern "C" {
  void cmsending_(int* ecode) {
    throw edm::Exception(edm::errors::LogicError,"Herwig6Error") <<" Herwig stoped run with error code "<<*ecode<<".";
  }
}
#endif
