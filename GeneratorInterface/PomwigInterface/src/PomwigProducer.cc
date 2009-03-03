/*
 *  Original Author: Fabian Stoeckli 
 *  26/09/06
 * 
 *  Modified for PomwigProducer: Antonio.Vilela.Pereira@cern.ch
 */

#include "GeneratorInterface/PomwigInterface/interface/PomwigProducer.h"
#include "GeneratorInterface/PomwigInterface/interface/Dummies.h"
#include "GeneratorInterface/PomwigInterface/interface/HWRGEN.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/JamesRandom.h"
#include "CLHEP/Random/RandFlat.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/GeneratorProducts/interface/GenInfoProduct.h"
#include "FWCore/Framework/interface/Run.h"

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
}

#define setpdfpath setpdfpath_
#define mysetpdfpath mysetpdfpath_
#define setlhaparm setlhaparm_
#define setherwpdf setherwpdf_
#define cmsending cmsending_

// -----------------  Source Code -----------------------------------------
PomwigProducer::PomwigProducer( const ParameterSet & pset) :
  EDProducer(), evt(0), 
  herwigVerbosity_ (pset.getUntrackedParameter<int>("herwigVerbosity",0)),
  herwigHepMCVerbosity_ (pset.getUntrackedParameter<bool>("herwigHepMCVerbosity",false)),
  herwigLhapdfVerbosity_ (pset.getUntrackedParameter<int>("herwigLhapdfVerbosity",0)),
  maxEventsToPrint_ (pset.getUntrackedParameter<int>("maxEventsToPrint",0)),
  comenergy(pset.getUntrackedParameter<double>("comEnergy",14000.)),
  lhapdfSetPath_(pset.getUntrackedParameter<string>("lhapdfSetPath","")),
  printCards_(pset.getUntrackedParameter<bool>("printCards",false)),
  extCrossSect(pset.getUntrackedParameter<double>("crossSection", -1.)),
  extFilterEff(pset.getUntrackedParameter<double>("filterEfficiency", -1.)),
  survivalProbability(pset.getUntrackedParameter<double>("survivalProbability", 0.05)),
  diffTopology(pset.getParameter<int>("diffTopology")),
  enableForcedDecays(pset.getUntrackedParameter<bool>("enableForcedDecays",false)),
  maxEvents_ (pset.getUntrackedParameter<int>("numberOfEvents",999999999)),
  eventNumber_(0)
{
  useJimmy_ = false;
  numTrials_ = 100;
  doMPInteraction_ = false;

  std::ostringstream header_str;

  header_str << "----------------------------------------------\n";
  header_str << "Initializing PomwigProducer\n";
  header_str << "----------------------------------------------\n";

  
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

  header_str << "   Herwig verbosity level         = " << herwigVerbosity_ << "\n";
  header_str << "   LHAPDF verbosity level         = " << herwigLhapdfVerbosity_ << "\n";
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
    header_str<<"   Using "<< lhapdfSetPath_ << "\n";	
  }
  else{
    header_str<<" failed.\n";
    header_str<<"   Using "<< lhapdfSetPath_ << "\n";
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

  int numEvents = maxEvents_;  
  if(useJimmy_) jmparm.MSFLAG = 1;

  // initialize other common blocks ...
  hwigin();

  double fracErrors_ = pset.getUntrackedParameter<double>("fracErrors",0.1);
  int maxerrors = int(fracErrors_*numEvents);
  hwevnt.MAXER = maxerrors;
  if(hwevnt.MAXER<100) hwevnt.MAXER = 100;
  header_str << "   MAXER set to " << hwevnt.MAXER << "\n";

  if(useJimmy_) jimmin();
  
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
    
    header_str << "----------------------------------------------" << "\n";
    header_str << "Read HERWIG parameter set " << mySet << "\n";
    header_str << "----------------------------------------------" << "\n";
    
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
	header_str << "   " << *itPar << "\n";
    }
  }

  // setting up herwgi RNG seeds NRN(.)
  header_str << "----------------------------------------------" << "\n";
  header_str << "Setting Herwig random number generator seeds" << "\n";
  header_str << "----------------------------------------------" << "\n";
  edm::Service<RandomNumberGenerator> rng;
  int wwseed = rng->mySeed();
  randomEngine = fRandomEngine = &(rng->getEngine());
  bool rngok = setRngSeeds(wwseed);
  if(!rngok)
    throw edm::Exception(edm::errors::Configuration,"HerwigError")
      <<" Impossible error in setting 'NRN(.)'.";
  header_str << "   NRN(1) = "<<hwevnt.NRN[0]<<"\n";
  header_str << "   NRN(2) = "<<hwevnt.NRN[1]<<"\n";

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

  // Function for force QQ decay is hwmodk

  /*
  C-----------------------------------------------------------------------
        SUBROUTINE HWMODK(IDKTMP,BRTMP,IMETMP,
       & IATMP,IBTMP,ICTMP,IDTMP,IETMP)
  C-----------------------------------------------------------------------
  C     Takes the decay, IDKTMP -> I-(A+B+C+D+E)-TMP, and simply stores it
  C     if internal pointers not set up (.NOT.DKPSET) else if pre-existing
  C     mode updates branching ratio BRTMP and matrix element code IMETMP,
  C     if -ve leaves as is. If a new mode adds to table and if consistent
  C     adjusts pointers,  sets CMMOM (for two-body mode) and resets RSTAB
  C     if necessary.  The branching ratios of any other IDKTMP decays are
  C     scaled by (1.-BRTMP)/(1.-BR_OLD)
  C-----------------------------------------------------------------------
  */

  // Initialization

  if(enableForcedDecays){
    header_str << "\n----------------------------------------------" << "\n";
    header_str << "HWMODK will be called to force decays" << "\n";
    header_str << "----------------------------------------------" << "\n";
    // Get ParameterSet with settings for forcing decays
    ParameterSet fdecays_pset = pset.getParameter<ParameterSet>("ForcedDecaysParameters") ; 
    vector<int> defidktmp ;
    defidktmp.push_back(0) ;
    std::vector<int> idktmp = fdecays_pset.getUntrackedParameter< vector<int> >("Idktmp", defidktmp);
    vector<double> defbrtmp ;
    defbrtmp.push_back(0) ;
    std::vector<double> brtmp = fdecays_pset.getUntrackedParameter< vector<double> >("Brtmp",defbrtmp);
    vector<int> defimetmp ;
    defimetmp.push_back(0) ;
    std::vector<int> imetmp = fdecays_pset.getUntrackedParameter< vector<int> >("Imetmp",defimetmp);
    vector<int> defiatmp ;
    defiatmp.push_back(0) ;
    std::vector<int> iatmp = fdecays_pset.getUntrackedParameter< vector<int> >("Iatmp",defiatmp);
    vector<int> defibtmp ;
    defibtmp.push_back(0) ;
    std::vector<int> ibtmp = fdecays_pset.getUntrackedParameter< vector<int> >("Ibtmp",defibtmp);
    vector<int> defictmp ;
    defictmp.push_back(0) ;
    std::vector<int> ictmp = fdecays_pset.getUntrackedParameter< vector<int> >("Ictmp",defictmp);
    vector<int> defidtmp ;
    defidtmp.push_back(0) ;
    std::vector<int> idtmp = fdecays_pset.getUntrackedParameter< vector<int> >("Idtmp",defidtmp);
    vector<int> defietmp ;
    defietmp.push_back(0) ;
    std::vector<int> ietmp = fdecays_pset.getUntrackedParameter< vector<int> >("Ietmp",defietmp);

    for (unsigned int i = 0; i < idktmp.size(); i++){
        int idktmp1 = idktmp[i];
        double brtmp1 = brtmp[i];
        int imetmp1 = imetmp[i];
        int iatmp1 = iatmp[i];
        int ibtmp1 = ibtmp[i];
        int ictmp1 = ictmp[i];
        int idtmp1 = idtmp[i];
        int ietmp1 = ietmp[i];
        // Call Herwig function HWMODK
	header_str << "   Forcing decay " << idktmp1 << "->" << iatmp1 << "+" 
						       	     << ibtmp1 << "+" 
						             << ictmp1 << "+"
						             << idtmp1 << "+"
						             << ietmp1 << "  with BR " << brtmp1 << "\n";
        hwmodk(idktmp1, brtmp1, imetmp1, iatmp1, ibtmp1, ictmp1, idtmp1, ietmp1);
    }
  }   

  hwusta("PI0     ",1);

  // Initialize H1 pomeron/reggeon
  if(diffTopology != 3){
        int nstru = hwpram.NSTRU;
        int ifit = pset.getParameter<int>("h1fit");
        if(nstru == 9){
                if((ifit <= 0)||(ifit >= 7)){
                        throw edm::Exception(edm::errors::Configuration,"PomwigError")
                        <<" Attempted to set non existant H1 1997 fit index. Has to be 1...6";
                }
                cout << "   H1 1997 pomeron pdf's" << endl;
                cout << "   IFIT = "<< ifit << endl;
                double xp = 0.1;
                double Q2 = 75.0;
                double xpq[13];
                qcd_1994(xp,Q2,xpq,ifit);
	} else if(nstru == 10){
                if((ifit <= 0)||(ifit >= 7)){
                        throw edm::Exception(edm::errors::Configuration,"PomwigError")
                        <<" Attempted to set non existant H1 1997 fit index. Has to be 1...6";
                }
                cout << "   H1 1997 reggeon pdf's" << endl;
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
                cout << "   H1 2006 A pomeron pdf's" << endl;
                cout << "   IFIT = "<< ifit <<endl;
                double xp = 0.1;
                double Q2 = 75.0;
                double xpq[13];
                double f2[2];
                double fl[2];
                double c2[2];
                double cl[2];
                qcd_2006(xp,Q2,ifit,xpq,f2,fl,c2,cl);
	} else if(nstru == 13){
                /*if(ifit != 1){
                        throw edm::Exception(edm::errors::Configuration,"PomwigError")
                        <<" Attempted to set non existant H1 2006 A fit index. Only IFIT=1";
                }*/
                ifit = 1;
                cout << "   H1 2006 A reggeon pdf's" << endl;
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
                cout << "   H1 2006 B pomeron pdf's" << endl;
                cout << "   IFIT = "<< ifit <<endl;
                double xp = 0.1;
                double Q2 = 75.0;
                double xpq[13];
                double f2[2];
                double fl[2];
                double c2[2];
                double cl[2];
                qcd_2006(xp,Q2,ifit,xpq,f2,fl,c2,cl);
	} else if(nstru == 15){
                /*if(ifit != 2){
                        throw edm::Exception(edm::errors::Configuration,"PomwigError")
                        <<" Attempted to set non existant H1 2006 B fit index. Only IFIT=2";
                }*/
                ifit = 2;
                cout << "   H1 2006 B reggeon pdf's" << endl;
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
                <<" Only running Pomeron H1 1997 (NSTRU=9), H1 2006 fit A (NSTRU=12) and H1 2006 fit B (NSTRU=14) or Reggeon H1 1997 (NSTRU=10), H1 2006 fit A (NSTRU=13) and H1 2006 fit B (NSTRU=15)";
        }
  }

  hweini();
  if(useJimmy_) jminit();

  produces<HepMCProduct>();
  produces<GenInfoProduct, edm::InRun>();

  header_str << "\n----------------------------------------------" << "\n";
  header_str << "Starting event generation" << "\n";
  header_str << "----------------------------------------------" << "\n";

  edm::LogInfo("")<<header_str.str(); 


}


PomwigProducer::~PomwigProducer(){

  std::ostringstream footer_str;

  footer_str << "----------------------------------------------" << "\n";
  footer_str << "Event generation done" << "\n";
  footer_str << "----------------------------------------------" << "\n";

  LogInfo("") << footer_str.str();

  clear();
}

void PomwigProducer::clear() {
  // teminate elementary process
  hwefin();
  if(useJimmy_) jmefin();
}


void PomwigProducer::produce(Event & e, const EventSetup& es) {

  int counter = 0;
  double mpiok = 1.0;

  while(mpiok > 0.5 && counter < numTrials_) {

    // call herwig routines to create HEPEVT
    hwuine();
    hwepro();
    hwbgen();  

    // call jimmy ... only if event is not killed yet by HERWIG
    if(useJimmy_ && doMPInteraction_ && hwevnt.IERROR==0)
      mpiok = hwmsct_dummy(1.1);
    else mpiok = 0.0;
    counter++;
  }
  
  // event after numTrials MP is not ok -> skip event
  if(mpiok > 0.5) {
    LogWarning("") <<"   JIMMY could not produce MI in "<<numTrials_<<" trials.\n"<<"   Event will be skipped to prevent from deadlock.\n";

// Throw an exception if generation fails.  Use the EventCorruption
// exception since it maps onto SkipEvent which is what we want to do here.

    std::ostringstream sstr;
    sstr << "PomwigProducer: JIMMY could not produce MI in " << numTrials_ << " trials.\n";
    edm::Exception except(edm::errors::EventCorruption, sstr.str());
    throw except;
  }  
  
  hwdhob();
  hwcfor();
  hwcdec();
  hwdhad();
  hwdhvy();
  hwmevt();
  hwufne();
  
  // if event was killed by HERWIG; skip 
  if(hwevnt.IERROR!=0) {
    std::ostringstream sstr;
    sstr << "PomwigProducer: HERWIG indicates a failure. Abandon the event.\n";
    edm::Exception except(edm::errors::EventCorruption, sstr.str());
    throw except;
  }

  intCrossSect = 1000.0*survivalProbability*hwevnt.AVWGT;

  // -----------------  HepMC converter --------------------
  HepMC::IO_HERWIG conv;

  // HEPEVT is ok, create new HepMC event
  evt = new HepMC::GenEvent();
  bool ok = conv.fill_next_event( evt );
  // if conversion failed; throw excpetion and stop processing
  if(!ok) throw cms::Exception("HerwigError")
    <<" Conversion problems in event nr."<< eventNumber_ << ".";  

  // set process id and event number
  evt->set_signal_process_id(hwproc.IPROC);  
  evt->set_event_number(eventNumber_);
  
  if (herwigHepMCVerbosity_) {
    LogWarning("") << "Event process = " << evt->signal_process_id() << "\n----------------------\n";
    evt->print();
  }
  
  // dummy if: event MUST be there
  if(evt)  {
    auto_ptr<HepMCProduct> bare_product(new HepMCProduct());  
    bare_product->addHepMCData(evt );
    e.put(bare_product);
  }
  
  return;
}

// -------------------------------------------------------------------------------------------------
// function to pass parameters to common blocks
bool PomwigProducer::hwgive(const std::string& ParameterString) {
  bool accepted = 1;

 
  if(!strncmp(ParameterString.c_str(),"IPROC",5)) {
    hwproc.IPROC = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);
    if(hwproc.IPROC<0) {
      throw cms::Exception("HerwigError")
	<<" Attempted to set IPROC to a negative value. This is not allowed.\n Please use the McatnloProducer to cope with negative valued IPROCs.";
    }
  }
  else if(!strncmp(ParameterString.c_str(),"AUTPDF(",7)){
    LogWarning("") <<"   WARNING: AUTPDF parameter *not* suported. HERWIG will use LHAPDF."<<"\n";
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
    LogWarning("")<<"   WARNING: BDECAY parameter *not* suported. HERWIG will use default b decay."<<"\n";
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
    hwuwts.REPWT[1][0][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,0,2)",12))
    hwuwts.REPWT[2][0][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,0,3)",12))
    hwuwts.REPWT[3][0][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,0,4)",12))
    hwuwts.REPWT[4][0][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,1,0)",12))
    hwuwts.REPWT[0][1][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,1,1)",12))
    hwuwts.REPWT[1][1][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,1,2)",12))
    hwuwts.REPWT[2][1][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,1,3)",12))
    hwuwts.REPWT[3][1][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,1,4)",12))
    hwuwts.REPWT[4][1][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,2,0)",12))
    hwuwts.REPWT[0][2][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,2,1)",12))
    hwuwts.REPWT[1][2][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,2,2)",12))
    hwuwts.REPWT[2][2][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,2,3)",12))
    hwuwts.REPWT[3][2][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,2,4)",12))
    hwuwts.REPWT[4][2][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,3,0)",12))
    hwuwts.REPWT[0][3][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,3,1)",12))
    hwuwts.REPWT[1][3][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,3,2)",12))
    hwuwts.REPWT[2][3][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,3,3)",12))
    hwuwts.REPWT[3][3][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,3,4)",12))
    hwuwts.REPWT[4][3][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,4,0)",12))
    hwuwts.REPWT[0][4][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,4,1)",12))
    hwuwts.REPWT[1][4][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,4,2)",12))
    hwuwts.REPWT[2][4][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,4,3)",12))
    hwuwts.REPWT[3][4][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(0,4,4)",12))
    hwuwts.REPWT[4][4][0] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,0,0)",12))
    hwuwts.REPWT[0][0][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,0,1)",12))
    hwuwts.REPWT[1][0][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,0,2)",12))
    hwuwts.REPWT[2][0][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,0,3)",12))
    hwuwts.REPWT[3][0][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,0,4)",12))
    hwuwts.REPWT[4][0][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,1,0)",12))
    hwuwts.REPWT[0][1][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,1,1)",12))
    hwuwts.REPWT[1][1][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,1,2)",12))
    hwuwts.REPWT[2][1][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,1,3)",12))
    hwuwts.REPWT[3][1][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,1,4)",12))
    hwuwts.REPWT[4][1][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,2,0)",12))
    hwuwts.REPWT[0][2][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,2,1)",12))
    hwuwts.REPWT[1][2][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,2,2)",12))
    hwuwts.REPWT[2][2][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,2,3)",12))
    hwuwts.REPWT[3][2][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,2,4)",12))
    hwuwts.REPWT[4][2][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,3,0)",12))
    hwuwts.REPWT[0][3][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,3,1)",12))
    hwuwts.REPWT[1][3][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,3,2)",12))
    hwuwts.REPWT[2][3][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,3,3)",12))
    hwuwts.REPWT[3][3][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,3,4)",12))
    hwuwts.REPWT[4][3][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,4,0)",12))
    hwuwts.REPWT[0][4][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,4,1)",12))
    hwuwts.REPWT[1][4][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,4,2)",12))
    hwuwts.REPWT[2][4][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,4,3)",12))
    hwuwts.REPWT[3][4][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(1,4,4)",12))
    hwuwts.REPWT[4][4][1] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,0,0)",12))
    hwuwts.REPWT[0][0][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,0,1)",12))
    hwuwts.REPWT[1][0][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,0,2)",12))
    hwuwts.REPWT[2][0][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,0,3)",12))
    hwuwts.REPWT[3][0][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,0,4)",12))
    hwuwts.REPWT[4][0][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,1,0)",12))
    hwuwts.REPWT[0][1][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,1,1)",12))
    hwuwts.REPWT[1][1][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,1,2)",12))
    hwuwts.REPWT[2][1][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,1,3)",12))
    hwuwts.REPWT[3][1][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,1,4)",12))
    hwuwts.REPWT[4][1][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,2,0)",12))
    hwuwts.REPWT[0][2][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,2,1)",12))
    hwuwts.REPWT[1][2][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,2,2)",12))
    hwuwts.REPWT[2][2][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,2,3)",12))
    hwuwts.REPWT[3][2][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,2,4)",12))
    hwuwts.REPWT[4][2][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,3,0)",12))
    hwuwts.REPWT[0][3][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,3,1)",12))
    hwuwts.REPWT[1][3][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,3,2)",12))
    hwuwts.REPWT[2][3][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,3,3)",12))
    hwuwts.REPWT[3][3][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,3,4)",12))
    hwuwts.REPWT[4][3][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,4,0)",12))
    hwuwts.REPWT[0][4][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,4,1)",12))
    hwuwts.REPWT[1][4][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,4,2)",12))
    hwuwts.REPWT[2][4][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,4,3)",12))
    hwuwts.REPWT[3][4][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(2,4,4)",12))
    hwuwts.REPWT[4][4][2] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,0,0)",12))
    hwuwts.REPWT[0][0][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,0,1)",12))
    hwuwts.REPWT[1][0][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,0,2)",12))
    hwuwts.REPWT[2][0][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,0,3)",12))
    hwuwts.REPWT[3][0][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,0,4)",12))
    hwuwts.REPWT[4][0][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,1,0)",12))
    hwuwts.REPWT[0][1][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,1,1)",12))
    hwuwts.REPWT[1][1][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,1,2)",12))
    hwuwts.REPWT[2][1][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,1,3)",12))
    hwuwts.REPWT[3][1][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,1,4)",12))
    hwuwts.REPWT[4][1][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,2,0)",12))
    hwuwts.REPWT[0][2][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,2,1)",12))
    hwuwts.REPWT[1][2][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,2,2)",12))
    hwuwts.REPWT[2][2][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,2,3)",12))
    hwuwts.REPWT[3][2][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,2,4)",12))
    hwuwts.REPWT[4][2][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,3,0)",12))
    hwuwts.REPWT[0][3][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,3,1)",12))
    hwuwts.REPWT[1][3][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,3,2)",12))
    hwuwts.REPWT[2][3][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,3,3)",12))
    hwuwts.REPWT[3][3][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,3,4)",12))
    hwuwts.REPWT[4][3][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,4,0)",12))
    hwuwts.REPWT[0][4][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,4,1)",12))
    hwuwts.REPWT[1][4][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,4,2)",12))
    hwuwts.REPWT[2][4][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,4,3)",12))
    hwuwts.REPWT[3][4][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
  else if(!strncmp(ParameterString.c_str(),"REPWT(3,4,4)",12))
    hwuwts.REPWT[4][4][3] = atof(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);  
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
// dummy hwaend (to be REMOVED from herwig)
#define hwaend hwaend_

extern "C" {
  void hwaend(){/*dummy*/}
}
//-------------------------------------------------------------------------------
#endif

bool PomwigProducer::setRngSeeds(int mseed)
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

void PomwigProducer::endRun(Run & r) {
 
 auto_ptr<GenInfoProduct> giprod (new GenInfoProduct());
 giprod->set_cross_section(intCrossSect);
 cout<<"cross section = "<<intCrossSect<<std::endl;
 giprod->set_external_cross_section(extCrossSect);
 giprod->set_filter_efficiency(extFilterEff);
 r.put(giprod);

}




#ifdef NEVER
extern "C" {
  void cmsending_(int* ecode) {
    LogError("")<<"   ERROR: Herwig stoped run after recieving error code "<<*ecode<<".\n";
    throw cms::Exception("HerwigError") <<" Herwig stoped run with error code "<<*ecode<<".";
  }
}
#endif
