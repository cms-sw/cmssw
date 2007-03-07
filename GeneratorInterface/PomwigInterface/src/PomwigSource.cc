/*
 *  Original Author: Fabian Stoeckli 
 *  26/09/06
 *  Modified for Pomwig interface
 *  02/2007
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

// Generator modifications
// ***********************
#include "HerwigWrapper6_4.h"
#include "IO_HERWIG.h"
#include "HEPEVT_Wrapper.h"

//-------------------------------------------------------------------------------
// COMMON block stuff, that doesn't come with the HerwigWrapper6_4.h ....

/*C Arrays for particle properties (NMXRES = max no of particles defined)
      PARAMETER(NMXRES=500)
      COMMON/HWPROP/RLTIM(0:NMXRES),RMASS(0:NMXRES),RSPIN(0:NMXRES),
     & ICHRG(0:NMXRES),IDPDG(0:NMXRES),IFLAV(0:NMXRES),NRES,
     & VTOCDK(0:NMXRES),VTORDK(0:NMXRES),
     & QORQQB(0:NMXRES),QBORQQ(0:NMXRES) */
const int nmxres = 500+1; // we need NMXRES+1 entries ...
extern struct {
  double RLTIM[nmxres], RMASS[nmxres], RSPIN[nmxres];
  int ICHRG[nmxres], IDPDG[nmxres],IFLAV[nmxres], NRES;
  int VTOCDK[nmxres], VTORDK[nmxres], QORQQB[nmxres], QBORQQ[nmxres];    
} hwprop_;
#define hwprop hwprop_

/*C Parameters for Sudakov form factors
C (NMXSUD= max no of entries in lookup table)
      PARAMETER (NMXSUD=1024)
      COMMON/HWUSUD/ACCUR,QEV(NMXSUD,6),SUD(NMXSUD,6),INTER,NQEV,NSUD,
      & SUDORD*/
const int nmxsud = 1024;
extern struct {
  double ACCUR, QEV[nmxsud][6],SUD[nmxsud][6];
  int INTER, NQEV, NSUD, SUDORD;
} hwusud_;
#define hwusud hwusud_

/*C  New parameters for version 6.203
      DOUBLE PRECISION ABWGT,ABWSUM,AVABW
      INTEGER NNEGWT,NNEGEV
      LOGICAL NEGWTS
      COMMON/HW6203/ABWGT,ABWSUM,AVABW,NNEGWT,NNEGEV,NEGWTS */
extern struct {
  double ABWGT, ABWSUM, AVABW;
  int NNEGWT,NNEGEV,NEGWTS;
} hw6203_;
#define hw6203 hw6203_


/*CHARACTER*20
     & AUTPDF
     COMMON/HWPRCH/AUTPDF(2),BDECAY   */
extern struct {
  char AUTPDF[2][20],BDECAY;
} hwprch_;
#define hwprch hwprch_

/*C Parameters for minimum bias/soft underlying event
      COMMON/HWMINB/
      & PMBN1,PMBN2,PMBN3,PMBK1,PMBK2,PMBM1,PMBM2,PMBP1,PMBP2,PMBP3  */
extern struct {
  double PMBN1,PMBN2,PMBN3,PMBK1,PMBK2,PMBM1,PMBM2,PMBP1,PMBP2,PMBP3;
} hwminb_;
#define hwminb hwminb_

/*C Variables controling mixing and vertex information
C--VTXPIP should have been a 5-vector, problems with NAG compiler
      COMMON/HWDIST/EXAG,GEV2MM,HBAR,PLTCUT,VMIN2,VTXPIP(5),XMIX(2),
      & XMRCT(2),YMIX(2),YMRCT(2),IOPDKL,MAXDKL,MIXING,PIPSMR */
extern struct {
  double EXAG,GEV2MM,HBAR,PLTCUT,VMIN2,VTXPIP[5],XMIX[2],XMRCT[2],YMIX[2],YMRCT[2];
  int IOPDKL,MAXDKL,MIXING,PIPSMR;
} hwdist_;
#define hwdist hwdist_

/*      PARAMETER(NMXCDK=4000)
      COMMON/HWUCLU/CLDKWT(NMXCDK),CTHRPW(12,12),PRECO,RESN(12,12),
      & RMIN(12,12),LOCN(12,12),NCLDK(NMXCDK),NRECO,CLRECO  */

const int nmxcdk=4000;
extern struct {
  double CLDKWT[nmxcdk],CTHRPW[12][12],PRECO,RESN[12][12], RMIN[12][12];
  int LOCN[12][12],NCLDK[nmxcdk], NRECO,CLRECO;
} hwuclu_;
#define hwuclu hwuclu_ 

/*C Weights used in cluster decays
      COMMON/HWUWTS/REPWT(0:3,0:4,0:4),SNGWT,DECWT,QWT(3),PWT(12),
      & SWTEF(NMXRES)  */
extern struct {
  double REPWT[4][5][5],SNGWT,DECWT,QWT[3],PWT[12],SWTEF[nmxres-1];
} hwuwts_;
#define hwuwts hwuwts_

/*C  Other new parameters for version 6.2
      DOUBLE PRECISION VIPWID,DXRCYL,DXZMAX,DXRSPH
      LOGICAL WZRFR,FIX4JT
      INTEGER IMSSM,IHIGGS,PARITY,LRSUSY
      COMMON/HW6202/VIPWID(3),DXRCYL,DXZMAX,DXRSPH,WZRFR,FIX4JT,
      & IMSSM,IHIGGS,PARITY,LRSUSY   */
extern struct {
  double VIPWID[3], DXRCYL,DXZMAX,DXRSPH;
  int WZRFR,FIX4JT,IMSSM,IHIGGS,PARITY,LRSUSY;
} hw6202_;
#define hw6202 hw6202_

/*      PARAMETER (MODMAX=50)
      COMMON/HWBOSC/ALPFAC,BRHIG(12),ENHANC(12),GAMMAX,RHOHEP(3,NMXHEP),
      & IOPHIG,MODBOS(MODMAX)  */
const int hepevt_size = 4000; // check in HerwigWrapper
const int modmax = 50;
extern struct {
  double ALPFAC, BRHIG[12], ENHANC[12], GAMMAX, RHOHEP[3][hepevt_size];
  int IOPHIG, MODBOS[modmax];
} hwbosc_;
#define hwbosc hwbosc_

/*      COMMON/HWHARD/ASFIXD,CLQ(7,6),COSS,COSTH,CTMAX,DISF(13,2),EMLST,
     & EMMAX,EMMIN,EMPOW,EMSCA,EPOLN(3),GCOEF(7),GPOLN,OMEGA0,PHOMAS,
     & PPOLN(3),PTMAX,PTMIN,PTPOW,Q2MAX,Q2MIN,Q2POW,Q2WWMN,Q2WWMX,QLIM,
     & SINS,THMAX,Y4JT,TMNISR,TQWT,XX(2),XLMIN,XXMIN,YBMAX,YBMIN,YJMAX,
     & YJMIN,YWWMAX,YWWMIN,WHMIN,ZJMAX,ZMXISR,IAPHIG,IBRN(2),IBSH,
     & ICO(10),IDCMF,IDN(10),IFLMAX,IFLMIN,IHPRO,IPRO,MAPQ(6),MAXFL,
     & BGSHAT,COLISR,FSTEVT,FSTWGT,GENEV,HVFCEN,TPOL,DURHAM   */
extern struct {
  double ASFIXD,CLQ[6][7],COSS,COSTH,CTMAX,DISF[2][13],EMLST, EMMAX,EMMIN,EMPOW,EMSCA,EPOLN[3],GCOEF[7],GPOLN,OMEGA0,PHOMAS, PPOLN[3],PTMAX,PTMIN,PTPOW,Q2MAX,Q2MIN,Q2POW,Q2WWMN,Q2WWMX,QLIM, SINS,THMAX,Y4JT,TMNISR,TQWT,XX[2],XLMIN,XXMIN,YBMAX,YBMIN,YJMAX,YJMIN,YWWMAX,YWWMIN,WHMIN,ZJMAX,ZMXISR;
  int IAPHIG,IBRN[2],IBSH, ICO[10],IDCMF,IDN[10],IFLMAX,IFLMIN,IHPRO,IPRO,MAPQ[6],MAXFL,BGSHAT,COLISR,FSTEVT,FSTWGT,GENEV,HVFCEN,TPOL,DURHAM;
} hwhard_;
#define hwhard hwhard_

/*C other HERWIG branching, event and hard subprocess common blocks
  COMMON/HWBRCH/ANOMSC(2,2),HARDST,PTINT(3,2),XFACT,INHAD,JNHAD,
  & NSPAC(7),ISLENT,BREIT,FROST,USECMF */
extern struct {
  double ANOMSC[2][2],HARDST,PTINT[2][3],XFACT;
  int INHAD,JNHAD,NSPAC[7],ISLENT,BREIT,FROST,USECMF;
} hwbrch_;
#define hwbrch hwbrch_

/*      LOGICAL PRESPL
	COMMON /HW6500/ PRESPL   */
extern struct {
  int PRESPL;
} hw6500_;
#define hw6500 hw6500_

/*C R-Parity violating parameters and colours
      COMMON /HWRPAR/ LAMDA1(3,3,3),LAMDA2(3,3,3),
      &                LAMDA3(3,3,3),HRDCOL(2,5),RPARTY,COLUPD   */
extern struct {
  double LAMDA1[3][3][3],LAMDA2[3][3][3],LAMDA3[3][3][3];
  int HRDCOL[5][2],RPARTY,COLUPD;
} hwrpar_;
#define hwrpar hwrpar_

/*C SUSY parameters
      COMMON/HWSUSY/
     & TANB,ALPHAH,COSBPA,SINBPA,COSBMA,SINBMA,COSA,SINA,COSB,SINB,COTB,
     & ZMIXSS(4,4),ZMXNSS(4,4),ZSGNSS(4), LFCH(16),RFCH(16),
     & SLFCH(16,4),SRFCH(16,4), WMXUSS(2,2),WMXVSS(2,2), WSGNSS(2),
     & QMIXSS(6,2,2),LMIXSS(6,2,2),
     & THETAT,THETAB,THETAL,ATSS,ABSS,ALSS,MUSS,FACTSS,
     & GHWWSS(3),GHZZSS(3),GHDDSS(4),GHUUSS(4),GHWHSS(3),
     & GHSQSS(4,6,2,2),XLMNSS,RMMNSS,DMSSM,SENHNC(24),SSPARITY,SUSYIN  */
extern struct {
  double TANB,ALPHAH,COSBPA,SINBPA,COSBMA,SINBMA,COSA,SINA,COSB,SINB,COTB,ZMIXSS[4][4],ZMXNSS[4][4],ZSGNSS[4], LFCH[16],RFCH[16],SLFCH[4][16],SRFCH[4][16], WMXUSS[2][2],WMXVSS[2][2], WSGNSS[2],QMIXSS[2][2][6],LMIXSS[2][2][6],THETAT,THETAB,THETAL,ATSS,ABSS,ALSS,MUSS,FACTSS,GHWWSS[3],GHZZSS[3],GHDDSS[4],GHUUSS[4],GHWHSS[3],GHSQSS[2][2][6][4],XLMNSS,RMMNSS,DMSSM,SENHNC[24],SSPARITY;
  int SUSYIN;
} hwsusy_;
#define hwsusy hwsusy_

/*INTEGER NDECSY,NSEARCH,LRDEC,LWDEC
      LOGICAL SYSPIN,THREEB,FOURB
      CHARACTER *6 TAUDEC
      COMMON /HWDSPN/NDECSY,NSEARCH,LRDEC,LWDEC,SYSPIN,THREEB,
      &	FOURB,TAUDEC */
extern struct {
  int NDECSY,NSEARCH,LRDEC,LWDEC,SYSPIN,THREEB,FOURB;
  char TAUDEC[6];
} hwdspn_;
#define hwdspn hwdspn_

/*C--common block for Les Houches interface to store information we need
C
      INTEGER MAXHRP
      PARAMETER (MAXHRP=100)
      DOUBLE PRECISION LHWGT(MAXHRP),LHWGTS(MAXHRP),LHMXSM,
     &     LHXSCT(MAXHRP),LHXERR(MAXHRP),LHXMAX(MAXHRP)
      INTEGER LHIWGT(MAXHRP),ITYPLH,LHNEVT(MAXHRP)
      LOGICAL LHSOFT,LHGLSF	
      COMMON /HWGUPR/LHWGT,LHWGTS,LHXSCT,LHXERR,LHXMAX,LHMXSM,LHIWGT,
      &     LHNEVT,ITYPLH,LHSOFT,LHGLSF  */
const int maxhrp = 100;
extern struct {
  double LHWGT[maxhrp],LHWGTS[maxhrp],LHMXSM,LHXSCT[maxhrp],LHXERR[maxhrp],LHXMAX[maxhrp];
  int LHIWGT,LHNEVT,ITYPLH,LHSOFT,LHGLSF;
} hwgupr_;
#define hwgupr hwgupr_

/*C  New parameters for version 6.3
      INTEGER IMAXCH,IMAXOP
      PARAMETER (IMAXCH=20,IMAXOP=40)
      DOUBLE PRECISION MJJMIN,CHNPRB(IMAXCH)
      INTEGER IOPSTP,IOPSH
      LOGICAL OPTM,CHON(IMAXCH)
      COMMON/HW6300/MJJMIN,CHNPRB,IOPSTP,IOPSH,OPTM,CHON   */
const int imaxch = 20;
extern struct {
  double MJJMIN,CHNPRB[imaxch];
  int IOPSTP,IOPSH,OPTM,CHON[imaxch];
} hw6300_;
#define hw6300 hw6300_
//-------------------------------------------------------------------------------

HepMC::IO_HERWIG conv;
// ***********************

extern"C" {
  void setpdfpath_(char*);
  void setlhaparm_(char*);
}

#define setpdfpath setpdfpath_
#define setlhaparm setlhaparm_

// Subroutine inside H1QCD
#define qcd_1994 qcd_1994_
extern "C" {
    void qcd_1994(double&,double&,double*,int);
}
// HWABEG to initialize H1 pomeron structure function
#define hwabeg hwabeg_
extern "C" {
    void hwabeg();
}

//used for defaults
  static const unsigned long kNanoSecPerSec = 1000000000;
  static const unsigned long kAveEventPerSec = 200;

PomwigSource::PomwigSource( const ParameterSet & pset, 
			    InputSourceDescription const& desc ) :
  GeneratedInputSource(pset, desc), evt(0), 
  herwigVerbosity_ (pset.getUntrackedParameter<int>("herwigVerbosity",0)),
  herwigHepMCVerbosity_ (pset.getUntrackedParameter<bool>("herwigHepMCVerbosity",false)),
  herwigLhapdfVerbosity_ (pset.getUntrackedParameter<int>("herwigLhapdfVerbosity",0)),
  maxEventsToPrint_ (pset.getUntrackedParameter<int>("maxEventsToPrint",0)),
  comenergy(pset.getUntrackedParameter<double>("comEnergy",14000.)),
  diffTopology(pset.getUntrackedParameter<int>("diffTopology",0)),
  lhapdfSetPath_(pset.getUntrackedParameter<string>("lhapdfSetPath",""))
{
  
  cout << "PomwigSource: initializing Pomwig/Herwig. " << endl;

  // Call hwudat to set up HERWIG block data
  hwudat();

  // herwigVerbosity Level IPRINT
  /* valid argumets are: 0: print title only
                         1: + print selected input parameters
                         2: + print table of particle codes and properties
			 3: + tables of Sudakov form factors                */

  cout << "Herwig verbosity level = " << herwigVerbosity_ << endl;

  // LHA vebosity: 0=silent, 1=lowkey, 2=defaul(all)

  cout << "LHAPDF verbosity level = " << herwigLhapdfVerbosity_ << endl;
  
  //Max number of events printed on verbosity level 
  //maxEventsToPrint_ = pset.getUntrackedParameter<int>("maxEventsToPrint",0);
  cout << "Number of events to be printed = " << maxEventsToPrint_ << endl;
  
  // setting basic parameters ...
  hwproc.PBEAM1 = comenergy/2.;
  hwproc.PBEAM2 = comenergy/2.;
  // Choose beam particles for POMWIG depending on topology
  switch (diffTopology){
        case 0: //DPE
                hwbmch.PART1[0]  = 'E';
                hwbmch.PART1[1]  = '-';
                hwbmch.PART2[0]  = 'E';
                hwbmch.PART2[1]  = '-';
                //hwpram.MODPDF[0] = -1;
                //hwpram.MODPDF[1] = -1;
                break;
        case 1: //SD survive PART1
                hwbmch.PART1[0]  = 'E';
                hwbmch.PART1[1]  = '-';
                hwbmch.PART2[0]  = 'P';
                hwbmch.PART2[1]  = ' ';
                //hwpram.MODPDF[0] = -1;
		//hwpram.MODPDF[1] = 20060;
                break;
        case 2: //SD survive PART2
                hwbmch.PART1[0]  = 'P';
                hwbmch.PART1[1]  = ' ';
                hwbmch.PART2[0]  = 'E';
                hwbmch.PART2[1]  = '-';
		//hwpram.MODPDF[0] = 20060;
                //hwpram.MODPDF[1] = -1;
                break;
        case 3: //Non diffractive
                hwbmch.PART1[0]  = 'P';
                hwbmch.PART1[1]  = ' ';
                hwbmch.PART2[0]  = 'P';
                hwbmch.PART2[1]  = ' ';
		//hwpram.MODPDF[0] = 20060;
		//hwpram.MODPDF[1] = 20060;
                break;
        default:
                throw edm::Exception(edm::errors::Configuration,"HerwigError")
          <<" Invalid Diff. Topology. Must be DPE(diffTopology = 0), SD particle 1 (diffTopology = 1), SD particle 2 (diffTop
ology = 2) and Non diffractive (diffTopology = 3)";
                break;
  }
  for(int i=2;i<8;++i){
    hwbmch.PART1[i]  = ' ';
    hwbmch.PART2[i]  = ' ';}

  // process number (should be changed ... )
  //  hwproc.IPROC = 1500;
  // this is not used anymore ..
  hwproc.MAXEV = 10;

  // initialize other common block ...
  hwigin();

  // set some 'non-herwig' defaults
  hwevnt.MAXPR =  maxEventsToPrint_;           // no printing out of events
  hwpram.IPRINT = herwigVerbosity_;            // HERWIG print out mode
  hwprop.RMASS[6] = 175.0;

  // Set HERWIG parameters in a single ParameterSet
  ParameterSet herwig_params = 
    pset.getParameter<ParameterSet>("HerwigParameters") ;

  // The parameter sets to be read (default, min bias, user ...) in the
  // proper order.
  vector<string> setNames = 
    herwig_params.getParameter<vector<string> >("parameterSets");  

  // Loop over the sets
  for ( unsigned i=0; i<setNames.size(); ++i ) {
    
    string mySet = setNames[i];

    // Read the HERWIG parameters for each set of parameters
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
    }
  }


  //In the future, we will get the random number seed on each event and tell 
  // HERWIG to use that new seed  
  cout << "----------------------------------------------" << endl;
  cout << "Setting Herwig random number seeds *blank*" << endl;
  cout << "----------------------------------------------" << endl;
  /*edm::Service<RandomNumberGenerator> rng;
  uint32_t seed1 = rng->mySeed();
  uint32_t seed2 = rng->mySeed();
  hwevnt.NRN[0] = seed1;
  hwevnt.NRN[1] = seed2;*/

  // set the LHSPDF grid directory
  hwprch.AUTPDF[0][0]='H';
  hwprch.AUTPDF[0][1]='W';
  hwprch.AUTPDF[0][2]='L';
  hwprch.AUTPDF[0][3]='H';
  hwprch.AUTPDF[0][4]='A';
  hwprch.AUTPDF[0][5]='P';
  hwprch.AUTPDF[0][6]='D';
  hwprch.AUTPDF[0][7]='F';
  hwprch.AUTPDF[1][0]='H';
  hwprch.AUTPDF[1][1]='W';
  hwprch.AUTPDF[1][2]='L';
  hwprch.AUTPDF[1][3]='H';
  hwprch.AUTPDF[1][4]='A';
  hwprch.AUTPDF[1][5]='P';
  hwprch.AUTPDF[1][6]='D';
  hwprch.AUTPDF[1][7]='F';
  for(int i=8; i<20; ++i) {
    hwprch.AUTPDF[0][i]=' ';
    hwprch.AUTPDF[1][i]=' ';
  }


  char pdfpath[232];
  bool dot=false;
  for(int i=0; i<232; ++i) {
    if(lhapdfSetPath_.c_str()[i]=='\0') dot=true;
    if(!dot) pdfpath[i]=lhapdfSetPath_.c_str()[i];
    else pdfpath[i]=' ';
  }

  setpdfpath(pdfpath);


  hwuinc();

  // callung HWUSTA to make any particle stable (PI0 by default)
  hwusta("PI0     ",1);

  // Initialize H1 pomeron structure function
  /*int ifit = 5;
  double xp = 0.1;
  double Q2 = 75.0;
  double xpq[13];
  qcd_1994(xp,Q2,xpq,ifit);*/
  hwabeg();

  // initialize elemetary process ...
  hweini();
  
  cout << endl; // Stetically add for the output
  //********                                      
  
  produces<HepMCProduct>();
  cout << "PomwigSource: starting event generation ... " << endl<<endl;
}


PomwigSource::~PomwigSource(){
  cout << "PomwigSource: event generation done. " << endl;
  clear();
}

void PomwigSource::clear() {
  // teminate elementary process
  hwefin();
}


bool PomwigSource::produce(Event & e) {
  
  auto_ptr<HepMCProduct> bare_product(new HepMCProduct());  
  
  //********                                         
  //

  // initialize event
  hwuine();

  // generate hard subprocess
  hwepro();

  // generate parton cascades
  hwbgen();

  // do heavy object decays
  hwdhob();

  // do cluster formation
  hwcfor();

  // do cluster decays
  hwcdec();

  // do unstable particle decays
  hwdhad();

  // do heavy flavour hadron decays
  hwdhvy();

  // add soft underlying event if needed
  hwmevt();

  // finish event
  hwufne();

  HepMC::GenEvent* evt = new HepMC::GenEvent();
  bool ok = conv.fill_next_event( evt );
  if(!ok) throw cms::Exception("HerwigError")
    <<" Conversion problems in event nr."<<numberEventsInRun() - remainingEvents() - 1<<".";  
  evt->set_signal_process_id(hwproc.IPROC);  
  evt->set_event_number(numberEventsInRun() - remainingEvents() - 1);
  
  if (herwigHepMCVerbosity_) {
    cout << "Event process = " << evt->signal_process_id() <<endl
	 << "----------------------" << endl;
    evt->print();
  }
  
  //evt = reader_->fillCurrentEventData(); 
  //********                                      
  
  if(evt)  bare_product->addHepMCData(evt );
  
  e.put(bare_product);
  
  return true;
}


bool 
PomwigSource::hwgive(const std::string& ParameterString) {

  bool accepted = 1;
  
  if(!strncmp(ParameterString.c_str(),"IPROC",5)) {
    hwproc.IPROC = atoi(&ParameterString[strcspn(ParameterString.c_str(),"=")+1]);
    if(hwproc.IPROC<0) {
      throw cms::Exception("HerwigError")
	<<" Attempted to set IPROC to a negative value. This is not allowed.\n Please use the McatnloInterface to cope with negative valued IPROCs.";
    }
  }
  else if(!strncmp(ParameterString.c_str(),"AUTPDF(",7)){
    cout<<" WARNING: AUTPDF parameter *not* suported. HERWIG will use LHAPDF."<<endl;
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
    cout<<" WARNING: BDECAY parameter *not* suported. HERWIG will use default b decay."<<endl;
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
  void hwaend(){/*dummy*/};
}
//-------------------------------------------------------------------------------
