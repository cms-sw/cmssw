/*
 *  Original Author: Fabian Stoeckli 
 *  26/09/06
 *  Modified for Pomwig
 *  03/2007 Antonio.Vilela.Pereira@cern.ch
 */

#include "GeneratorInterface/PomwigInterface/interface/PomwigSource.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/JamesRandom.h"
#include "CLHEP/Random/RandFlat.h"

#include <iostream>
#include "time.h"
#include <ctype.h>

using namespace edm;
using namespace std;

#include "HepMC/HerwigWrapper6_4.h"
#include "HepMC/IO_HERWIG.h"
#include "HepMC/HEPEVT_Wrapper.h"

// INCLUDE JIMMY,HERWIG,LHAPDF,POMWIG COMMON BLOCKS AND FUNTIONS
#include "herwig.h"


extern"C" {
  void setpdfpath_(char*,int*);
  void mysetpdfpath_(char*);
  void setlhaparm_(char*);
  void setherwpdf_(void);
  // function to chatch 'STOP' in original HWWARN
  void cmsending_(int*);

  // struct to check wheter HERWIG killed an event
  extern struct {
    double eventisok;
  } eventstat_;
}

#define eventstat eventstat_

#define setpdfpath setpdfpath_
#define mysetpdfpath mysetpdfpath_
#define setlhaparm setlhaparm_
#define setherwpdf setherwpdf_
#define cmsending cmsending_


// -----------------  used for defaults --------------------------------------
  static const unsigned long kNanoSecPerSec = 1000000000;
  static const unsigned long kAveEventPerSec = 200;

// -----------------  Source Code -----------------------------------------
PomwigSource::PomwigSource( const ParameterSet & pset, 
			    InputSourceDescription const& desc ) :
  GeneratedInputSource(pset, desc), evt(0), 
  herwigVerbosity_ (pset.getUntrackedParameter<int>("herwigVerbosity",0)),
  herwigHepMCVerbosity_ (pset.getUntrackedParameter<bool>("herwigHepMCVerbosity",false)),
  herwigLhapdfVerbosity_ (pset.getUntrackedParameter<int>("herwigLhapdfVerbosity",0)),
  maxEventsToPrint_ (pset.getUntrackedParameter<int>("maxEventsToPrint",0)),
  comenergy(pset.getUntrackedParameter<double>("comEnergy",14000.)),
  lhapdfSetPath_(pset.getUntrackedParameter<string>("lhapdfSetPath","")),
  printCards_(pset.getUntrackedParameter<bool>("printCards",true)),
  diffTopology(pset.getParameter<int>("diffTopology"))
{
  cout << "----------------------------------------------" << endl;
  cout << "Initializing PomwigSource" << endl;
  cout << "----------------------------------------------" << endl;
  /* herwigVerbosity Level IPRINT
     valid argumets are: 0: print title only
                         1: + print selected input parameters
                         2: + print table of particle codes and properties
			 3: + tables of Sudakov form factors  
              *** NOT IMPLEMENTED ***
      LHA vebosity:      0=silent 
                         1=lowkey (default) 
			 2=all 
  */

  cout << "   Herwig verbosity level         = " << herwigVerbosity_ << endl;
  cout << "   LHAPDF verbosity level         = " << herwigLhapdfVerbosity_ << endl;
  cout << "   HepMC verbosity                = " << herwigHepMCVerbosity_ << endl;
  cout << "   Number of events to be printed = " << maxEventsToPrint_ << endl;
  
  // setting up lhapdf path name from environment varaible (***)
  char* lhaPdfs = NULL;
  std::cout<<"   Trying to find LHAPATH in environment ...";
  lhaPdfs = getenv("LHAPATH");
  if(lhaPdfs != NULL) {
    std::cout<<" done."<<std::endl;
    lhapdfSetPath_=std::string(lhaPdfs);
    std::cout<<"   Using "<< lhapdfSetPath_ << std::endl;	
  }
  else{
    std::cout<<" failed."<<std::endl;
    std::cout<<"   Using "<< lhapdfSetPath_ << std::endl;
  }	

  // Call hwudat to set up HERWIG block data
  hwudat();
  
  // Setting basic parameters ...
  hwproc.PBEAM1 = comenergy/2.;
  hwproc.PBEAM2 = comenergy/2.;
  // Choose beam particles for POMWIG depending on topology
  switch (diffTopology){
        case 0: //DPE
                hwbmch.PART1[0]  = 'E';
                hwbmch.PART1[1]  = '-';
                hwbmch.PART2[0]  = 'E';
                hwbmch.PART2[1]  = '-';
                break;
        case 1: //SD survive PART1
                hwbmch.PART1[0]  = 'E';
                hwbmch.PART1[1]  = '-';
                hwbmch.PART2[0]  = 'P';
                hwbmch.PART2[1]  = ' ';
                break;
        case 2: //SD survive PART2
                hwbmch.PART1[0]  = 'P';
                hwbmch.PART1[1]  = ' ';
                hwbmch.PART2[0]  = 'E';
                hwbmch.PART2[1]  = '-';
                break;
        case 3: //Non diffractive
                hwbmch.PART1[0]  = 'P';
                hwbmch.PART1[1]  = ' ';
                hwbmch.PART2[0]  = 'P';
                hwbmch.PART2[1]  = ' ';
                break;
        default:
                throw edm::Exception(edm::errors::Configuration,"PomwigError")
          <<" Invalid Diff. Topology. Must be DPE(diffTopology = 0), SD particle 1 (diffTopology = 1), SD particle 2 (diffTopology = 2) and Non diffractive (diffTopology = 3)";
                break;
  }
  for(int i=2;i<8;++i){
    hwbmch.PART1[i]  = ' ';
    hwbmch.PART2[i]  = ' ';}
  int numEvents = desc.maxEvents_;
  //hwproc.MAXEV = pset.getUntrackedParameter<int>("maxEvents",10);

  // initialize other common blocks ...
  hwigin();
 
  double fracErrors_ = pset.getUntrackedParameter<double>("fracErrors",0.1);
  int maxerrors = int(fracErrors_*numEvents);
  hwevnt.MAXER = maxerrors;
  if(hwevnt.MAXER<100) hwevnt.MAXER = 100;
  std::cout<<"   MAXER set to "<< hwevnt.MAXER << std::endl;

  // set some 'non-herwig' defaults
  hwevnt.MAXPR =  maxEventsToPrint_;           // no printing out of events
  hwpram.IPRINT = herwigVerbosity_;            // HERWIG print out mode
  hwprop.RMASS[6] = 175.0;

  // Set HERWIG parameters in a single ParameterSet
  ParameterSet herwig_params = 
    pset.getParameter<ParameterSet>("HerwigParameters") ;
  
  // The parameter sets to be read (default, min bias, user ...) in the proper order.
  vector<string> setNames = 
    herwig_params.getParameter<vector<string> >("parameterSets");  
  
  // Loop over the sets
  for ( unsigned i=0; i<setNames.size(); ++i ) {    
    string mySet = setNames[i];
    vector<string> pars = 
      herwig_params.getParameter<vector<string> >(mySet);
    
    cout << "----------------------------------------------" << endl;
    cout << "Read HERWIG parameter set " << mySet << endl;
    cout << "----------------------------------------------" << endl;
    
    // Loop over all parameters and stop in case of mistake
    for( vector<string>::const_iterator  
	   itPar = pars.begin(); itPar != pars.end(); ++itPar ) {
      static string sRandomValueSetting1("NRN(1)");
      static string sRandomValueSetting2("NRN(2)");
      if( (0 == itPar->compare(0,sRandomValueSetting1.size(),sRandomValueSetting1) )||(0 == itPar->compare(0,sRandomValueSetting2.size(),sRandomValueSetting2) )) {
	throw edm::Exception(edm::errors::Configuration,"HerwigError")
	  <<" attempted to set random number using pythia command 'NRN(.)'. This is not allowed.\n  Please use the RandomNumberGeneratorService to set the random number seed.";
      }
      
      if( ! hwgive(*itPar) ) {
	throw edm::Exception(edm::errors::Configuration,"HerwigError") 
	  <<" herwig did not accept the following \""<<*itPar<<"\"";
      }
      else if(printCards_)
	cout << "   " << *itPar << endl;
    }
  }

  // setting up herwgi RNG seeds NRN(.)
  cout << "----------------------------------------------" << endl;
  cout << "Setting Herwig random number generator seeds" << endl;
  cout << "----------------------------------------------" << endl;
  edm::Service<RandomNumberGenerator> rng;
  int wwseed = rng->mySeed();
  bool rngok = setRngSeeds(wwseed);
  if(!rngok)
    throw edm::Exception(edm::errors::Configuration,"HerwigError")
      <<" Impossible error in setting 'NRN(.)'.";
  cout << "   NRN(1) = "<<hwevnt.NRN[0]<<endl;
  cout << "   NRN(2) = "<<hwevnt.NRN[1]<<endl;

  // set the LHAPDF grid directory and path
  setherwpdf();
  char pdfpath[232];
  int pathlen = lhapdfSetPath_.length();
  for(int i=0; i<pathlen; ++i) 
  pdfpath[i]=lhapdfSetPath_.at(i);
  for(int i=pathlen; i<232; ++i) 
  pdfpath[i]=' ';
  mysetpdfpath(pdfpath);

  // HERWIG preparations ...
  hwuinc();
  hwusta("PI0     ",1);

  // Initialize H1 pomeron
  if(diffTopology != 3){
        int nstru = hwpram.NSTRU;
        int ifit = pset.getParameter<int>("h1fit");
        if(nstru == 9){
                if((ifit <= 0)||(ifit >= 7)){
                        throw edm::Exception(edm::errors::Configuration,"PomwigError")
                        <<" Attempted to set non existant H1 1997 fit index. Has to be 1...6";
                }
                cout << "   H1 1997 pdf's" << endl;
                cout << "   IFIT = "<< ifit << endl;
                double xp = 0.1;
                double Q2 = 75.0;
                double xpq[13];
                qcd_1994(xp,Q2,xpq,ifit);
        } else if(nstru == 12){
                /*if(ifit != 1){
                        throw edm::Exception(edm::errors::Configuration,"PomwigError")
                        <<" Attempted to set non existant H1 2006 A fit index. Only IFIT=1";
                }*/
                ifit = 1;
                cout << "   H1 2006 A pdf's" << endl;
                cout << "   IFIT = "<< ifit <<endl;
                double xp = 0.1;
                double Q2 = 75.0;
                double xpq[13];
                double f2[2];
                double fl[2];
                double c2[2];
                double cl[2];
                qcd_2006(xp,Q2,ifit,xpq,f2,fl,c2,cl);
        } else if(nstru == 14){
                /*if(ifit != 2){
                        throw edm::Exception(edm::errors::Configuration,"PomwigError")
                        <<" Attempted to set non existant H1 2006 B fit index. Only IFIT=2";
                }*/
                ifit = 2;
                cout << "   H1 2006 B pdf's" << endl;
                cout << "   IFIT = "<< ifit <<endl;
                double xp = 0.1;
                double Q2 = 75.0;
                double xpq[13];
                double f2[2];
                double fl[2];
                double c2[2];
                double cl[2];
                qcd_2006(xp,Q2,ifit,xpq,f2,fl,c2,cl);
        } else{
                throw edm::Exception(edm::errors::Configuration,"PomwigError")
                <<" Only running Pomeron H1 1997 (NSTRU=9), H1 2006 fit A (NSTRU=12) and H1 2006 fit B (NSTRU=14)";
        }
  }

  hweini();

  cout << endl; // Stetically add for the output
  produces<HepMCProduct>();

  cout << "----------------------------------------------" << endl;
  cout << "Starting event generation" << endl;
  cout << "----------------------------------------------" << endl;
}


PomwigSource::~PomwigSource(){
  cout << "----------------------------------------------" << endl;
  cout << "Event generation done" << endl;
  cout << "----------------------------------------------" << endl;
  clear();
}

void PomwigSource::clear() {
  // teminate elementary process
  hwefin();
}


bool PomwigSource::produce(Event & e) {

  eventstat.eventisok = 0.0;

  // call herwig routines to create HEPEVT
  hwuine();
  hwepro();
  hwbgen();  
  
  hwdhob();
  hwcfor();
  hwcdec();
  hwdhad();
  hwdhvy();
  hwmevt();
  hwufne();
  
  // if event was killed by HERWIG; skip 
  if(eventstat.eventisok > 0.5) return true;

  // -----------------  HepMC converter --------------------
  HepMC::IO_HERWIG conv;

  // HEPEVT is ok, create new HepMC event
  evt = new HepMC::GenEvent();
  bool ok = conv.fill_next_event( evt );
  // if conversion failed; throw excpetion and stop processing
  if(!ok) throw cms::Exception("HerwigError")
    <<" Conversion problems in event nr."<<numberEventsInRun() - remainingEvents() - 1<<".";  

  // set process id and event number
  evt->set_signal_process_id(hwproc.IPROC);  
  evt->set_event_number(numberEventsInRun() - remainingEvents() - 1);
  
  if (herwigHepMCVerbosity_) {
    cout << "Event process = " << evt->signal_process_id() <<endl
	 << "----------------------" << endl;
    evt->print();
  }
  
  // dummy if: event MUST be there
  if(evt)  {
    auto_ptr<HepMCProduct> bare_product(new HepMCProduct());  
    bare_product->addHepMCData(evt );
    e.put(bare_product);
  }
  
  return true;
}

// -------------------------------------------------------------------------------------------------
// function to pass parameters to common blocks
bool PomwigSource::hwgive(const std::string& ParameterString) {
  bool accepted = 1;

 
  if(!strncmp(ParameterString.c_str(),"IPROC",5)) {
    hwproc.IPROC = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);
    if(hwproc.IPROC<0) {
      throw cms::Exception("HerwigError")
	<<" Attempted to set IPROC to a negative value. This is not allowed.\n Please use the McatnloInterface to cope with negative valued IPROCs.";
    }
  }
  else if(!strncmp(ParameterString.c_str(),"AUTPDF(",7)){
    cout<<"   WARNING: AUTPDF parameter *not* suported. HERWIG will use LHAPDF."<<endl;
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
      throw cms::Exception("HerwigError")
	<<" Attempted to set TAUDEC to "<<hwdspn.TAUDEC<<". This is not allowed.\n Options for TAUDEC are HERWIG and TAUOLA.";
    }
  }
  else if(!strncmp(ParameterString.c_str(),"BDECAY",6)){
    cout<<"   WARNING: BDECAY parameter *not* suported. HERWIG will use default b decay."<<endl;
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

  else accepted = 0;

  return accepted;
}


//-------------------------------------------------------------------------------
// dummy hwaend (to be REMOVED from herwig)
#define hwaend hwaend_

extern "C" {
  void hwaend(){/*dummy*/}
}
//-------------------------------------------------------------------------------

bool PomwigSource::setRngSeeds(int mseed)
{
  double temx[5];
  for (int i=0; i<5; i++) {
    mseed = mseed * 29943829 - 1;
    temx[i] = mseed * (1./(65536.*65536.));
  }
  long double c;
  c = (long double)2111111111.0 * temx[3] +
    1492.0 * (temx[3] = temx[2]) +
    1776.0 * (temx[2] = temx[1]) +
    5115.0 * (temx[1] = temx[0]) +
    temx[4];
  temx[4] = floorl(c);
  temx[0] = c - temx[4];
  temx[4] = temx[4] * (1./(65536.*65536.));
  hwevnt.NRN[0]=int(temx[0]*99999);
  c = (long double)2111111111.0 * temx[3] +
    1492.0 * (temx[3] = temx[2]) +
    1776.0 * (temx[2] = temx[1]) +
    5115.0 * (temx[1] = temx[0]) +
    temx[4];
  temx[4] = floorl(c);
  temx[0] = c - temx[4];
  hwevnt.NRN[1]=int(temx[0]*99999);

  return true;
}

extern "C" {
  void cmsending_(int* ecode) {
    cout<<"   ERROR: Herwig stoped run after recieving error code "<<*ecode<<"."<<endl;
    throw cms::Exception("HerwigError") <<" Herwig stoped run with error code "<<*ecode<<".";
  }
}
