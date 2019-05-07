#ifndef FPGACONSTANTS_H
#define FPGACONSTANTS_H

//Uncomment if you want root output
//#define USEROOT

//Uncomment to run the hybrid algorithm
//#ifdef CMSSW_GIT_HASH
//#define USEHYBRID
//#endif

//Uncomment to run the HLS version of the KF if using the Hybrid (instead of the C++ KF).
//(Please also follow the instructions in L1Trigger/TrackFindingTMTT/README_HLS.txt).
//#define USE_HLS

static unsigned int nHelixPar = 4; // 4 or 5 param helix fit.

static bool doKF=false; //true => use KF (assumes USEHYBRID is defined)
static bool printDebugKF=false; // if true print lots of debugging statements related to the KF fit
static bool bookHistos=false;

static bool hourglassExtended=false; // This is turn on Displaced Tracking. Also change the file in Tracklet_cfi from hourglass to hourglassExtended ****************

//Gemetry extensions
static std::string geomext=hourglassExtended?"hourglassExtended":"hourglass";  

static int TMUX = 6;

static std::string fitpatternfile="../data/fitpattern.txt";

//If this string is non-empty we will write ascii file with
//processed events
static std::string skimfile="";
//static std::string skimfile="evlist_skim.txt";

//Debug options (should be false for 'normal' operation)
static bool dumppars=false;
static bool dumpproj=false;

static bool writeVerilog=false;     //Write out Verilog mudules for TCs
static bool writeHLS=false;         //Write out HLS mudules for TCs
static bool writeInvTable=false;    //Write out tables of drinv and invt in tracklet calculator for Verilog module
static bool writeHLSInvTable=false; //Write out tables of drinv and invt in tracklet calculator for HLS module

static bool writeFitDerTable=false; //Write out track derivative tables


//static bool writeDTCLinks=false;
static bool writeIL=false;
static bool writeStubsLayer=false;
static bool writeStubsLayerperSector=false;
static bool writeAllStubs=false;
static bool writeVMOccupancyME=false;
static bool writeVMOccupancyTE=false;
static bool writeSeeds=false;
static bool writeTE=false;
static bool writeTED=false;
static bool writeTRE=false;
static bool writeTrackletCalculator=false;
static bool writeTrackletCalculatorDisplaced=false;
static bool writeTrackletPars=false;
static bool writeAllProjections=false;
static bool writeVMProjections=false;
static bool writeTrackProjOcc=false;
static bool writeME=false;
static bool writeMatchCalculator=false;
static bool writeResiduals=false;
static bool writeProjectionTransceiver=false;
static bool writeMatchTransceiver=false;
static bool writeFitTrack=false;
static bool writeChiSq=false;

static bool writeNMatches=false;
static bool writeHitEff=false;

static bool writeTETables=false;
static bool writeVMTables=false;
static bool writeMETables=false;
static bool writeMCcuts=false;

static bool writeCabling=false;

static bool writeHitPattern=false;
static bool writeTrackletParsOverlap=false;
static bool writeTrackletParsDisk=false;

static bool writeAllCT=false; //write out .dat file containing all output tracks in bitwise format

static bool writeVariance=false; //write out residuals for variand matrix determination
static bool writeResEff=false; //write files for making resolution & efficiency plots for standable code version
static bool writePars=false; //write files for making plots of track parameters

static bool writeMatchEff=false; //write files for making plots with truth matched efficiency



static bool writestubs=false;  // write input stubs in the normal format
static bool writestubs_in2=false;  // write input stubs in hardware-ready format
static bool writeifit=false;
static bool padding=true;

static bool useMSFit=false;
static bool tcorrection=true;
static bool exactderivatives=false;  //for both the integer and float
static bool exactderivativesforfloating=true; //only for the floating point
static bool useapprox=true; //use approximate postion based on integer representation for floating point
static int alphashift=12;  
static int nbitsalpha=4;  //bits used to store alpha
static int alphaBitsTable=2; //For number of bits in track derivative table
static int nrinvBitsTable=3; //number of bits for tabulating rinv dependence
static bool writetrace=false; //Print out details about startup
static bool debug1=false; //Print detailed debug information about tracking
static bool writeoutReal = false; 
static bool writememLinks = false; //Write files for dtc links
static bool writemem=false; //Note that for 'full' detector this will open
                            //a LOT of files, and the program will run excruciatingly slow
static unsigned int writememsect=3;  //writemem only for this sector

static bool writeVMRTables = false; //write tables used by VMRouter
static bool writeTripletTables=false; //Train and write the TED and TRE tables. N.B.: the tables
                                      //cannot be applied while they are being trained, i.e.,
                                      //this flag effectively turns off the cuts in
                                      //FPGATrackletEngineDisplaced and FPGATripletEngine


static bool warnNoMem=false;  //If true will print out warnings about missing projection memories

//Program flow (should be true for normal operation)
//enables the stub finding in these layer/disk combinations
static bool doL1L2=true;
static bool doL2L3=true;
static bool doL3L4=true;
static bool doL5L6=true;

static bool doD1D2=true; 
static bool doD3D4=true;

static bool doL1D1=true;
static bool doL2D1=true;

static bool doL3L4L2=true; // only run if hourglassExtended is true
static bool doL5L6L4=true; // only run if hourglassExtended is true
static bool doL2L3D1=true; // only run if hourglassExtended is true
static bool doD1D2L2=true; // only run if hourglassExtended is true

static bool allSector=false; //if true duplicate stubs in all sectors

static bool doProjections=true;

static const int MEBinsBits=3;
static const int MEBins=(1<<MEBinsBits);

static const int MEBinsDisks=8; //on each side

//Geometry 

//These define the length scale for both r and z
static double zlength=120.0;
static double rmaxdisk=120.0;

// can automatically determine the above values using script plotstub.cc:
// root -b -q 'plotstub.cc("evlist_MuPlus_1to10_D11_PU0")'
// Values are commented out and use discrete values consistent with integer positions

// these assume D11 geometry!
static double rmeanL1=(rmaxdisk*858)/4096; //25.1493;
static double rmeanL2=(rmaxdisk*1279)/4096; //37.468;
static double rmeanL3=(rmaxdisk*1795)/4096; //52.5977;
static double rmeanL4=(rmaxdisk*2347)/4096; //68.7737;
static double rmeanL5=(rmaxdisk*2937)/4096; //86.0591;
static double rmeanL6=(rmaxdisk*3783)/4096; //110.844;

static double zmeanD1=(zlength*2239)/2048; //131.18;
static double zmeanD2=(zlength*2645)/2048; //155.0;
static double zmeanD3=(zlength*3163)/2048; //185.34;
static double zmeanD4=(zlength*3782)/2048; //221.619;
static double zmeanD5=(zlength*4523)/2048; //265.0;



static double rmindiskvm=22.5;
static double rmaxdiskvm=67.0;

static double rmaxdiskl1overlapvm=45.0;
static double rmindiskl2overlapvm=40.0;
static double rmindiskl3overlapvm=50.0;

static double half2SmoduleWidth=4.57;

// need separate lookup values for inner two vs outer three disks for 2S modules
// these assume D11 geometry!
static double rDSSinner[10] = {66.7728, 71.7967, 77.5409, 82.5584, 84.8736, 89.8953, 95.7791, 100.798, 102.495, 107.52};  // <=== these 10 are for inner 2 disks
static double rDSSouter[10] = {65.1694, 70.1936, 75.6641, 80.6908, 83.9581, 88.9827, 94.6539, 99.6772, 102.494, 107.519}; // <=== these 10 are for outer 3 disks


static double drmax=rmaxdisk/32.0;

static double dzmax=zlength/32.0;

static double drdisk=rmaxdisk;

static double rmean[6]={rmeanL1,rmeanL2,rmeanL3,rmeanL4,rmeanL5,rmeanL6};

static double zmean[5]={zmeanD1,zmeanD2,zmeanD3,zmeanD4,zmeanD5};


static unsigned int nallstubslayers[6]={8,4,4,4,4,4};
static unsigned int nvmtelayers[6]={4,8,4,8,4,8};
static unsigned int nvmteextralayers[6]={0,4,4,0,0,0};

static unsigned int nallprojlayers[6]={8,4,4,4,4,4};
static unsigned int nvmmelayers[6]={4,8,8,8,8,8};

static unsigned int nallstubsdisks[5]={4,4,4,4,4};
static unsigned int nvmtedisks[5]={4,4,4,4,4};

static unsigned int nallprojdisks[5]={4,4,4,4,4};
static unsigned int nvmmedisks[5]={8,4,4,4,4};
//for seeding in L1D1 L2D1
static unsigned int nallstubsoverlaplayers[3] = {8, 4, 4}; 
static unsigned int nvmteoverlaplayers[3] = {2, 2, 2};

static unsigned int nallstubsoverlapdisks[2] = {4, 4}; 
static unsigned int nvmteoverlapdisks[2] = {4, 4};

static double rcrit=55.0;


static double rminL1=rmeanL1-drmax; 
static double rmaxL1=rmeanL1+drmax; 
static double rminL2=rmeanL2-drmax; 
static double rmaxL2=rmeanL2+drmax; 
static double rminL3=rmeanL3-drmax; 
static double rmaxL3=rmeanL3+drmax; 
static double rminL4=rmeanL4-drmax; 
static double rmaxL4=rmeanL4+drmax; 
static double rminL5=rmeanL5-drmax; 
static double rmaxL5=rmeanL5+drmax; 
static double rminL6=rmeanL6-drmax; 
static double rmaxL6=rmeanL6+drmax; 

static double zminD1=zmeanD1-dzmax; 
static double zmaxD1=zmeanD1+dzmax; 
static double zminD2=zmeanD2-dzmax; 
static double zmaxD2=zmeanD2+dzmax; 
static double zminD3=zmeanD3-dzmax; 
static double zmaxD3=zmeanD3+dzmax; 
static double zminD4=zmeanD4-dzmax; 
static double zmaxD4=zmeanD4+dzmax; 
static double zminD5=zmeanD5-dzmax; 
static double zmaxD5=zmeanD5+dzmax; 

static double two_pi=2*M_PI;

static double ptcut=1.91; //Minimum pt
static double rinvcut=0.01*0.3*3.8/ptcut; //0.01 to convert to cm-1
static double ptcutte=1.6; //Minimum pt in TE
static double rinvcutte=0.01*0.3*3.8/ptcutte; //0.01 to convert to cm-1 in TE
static double bendcut=1.5;
static double bendcutdisk=2.0;
static double z0cut=15.0;


static unsigned int NSector=9; 
static int Nphibits=2;         //Number of bits required to label the phi VM
static int L1Nphi=(1<<Nphibits)-1; //Number of odd layer VMs
static int Nzbits=3;         //Number of bits required to label the z VM
static int L1Nz=(1<<Nzbits); //Number of z VMs in odd layers
//static int VMzbits=4;        //Number of bits for the z position in VM
static int L2Nphi=(1<<Nphibits); //Number of even layer VMs
static int L2Nz=(1<<Nzbits); //Number of z VMs in even layers
//static int VMrbits=2;        //Number of bits for r position 'in VM'
static int VMphibits=3;      //Number of bits for phi position in VM

static double rinvmax=0.01*0.3*3.8/2.0; //0.01 to convert to cm-1

static double dphisectorHG=2*M_PI/NSector+2*fmax(fabs(asin(0.5*rinvmax*rmean[0])-asin(0.5*rinvmax*rcrit)),
						fabs(asin(0.5*rinvmax*rmean[5])-asin(0.5*rinvmax*rcrit)));

static double phicritmin=0.5*dphisectorHG-M_PI/NSector;
static double phicritmax=dphisectorHG-0.5*dphisectorHG+M_PI/NSector;

static double dphicritmc=0.005; //lose for MC
static double phicritminmc=phicritmin-dphicritmc;
static double phicritmaxmc=phicritmax+dphicritmc;


static const unsigned int NLONGVMBITS=3; 
static const unsigned int NLONGVMRBITS=3;   //4 bins on each side (+-z)
static const unsigned int NLONGVMBINS=(1<<NLONGVMBITS);
static const unsigned int NLONGVMRBINS=(1<<NLONGVMRBITS);
static const unsigned int NLONGVMODDLAYERBITS=6;
static const unsigned int NLONGVMODDDISKBITS=6;
static const double rinnerdisk=22.0;



//limits per FED region
//static int NMAXstub  = 250;
//static int NMAXroute = 250;

static unsigned int MAXOFFSET=10000; //set to 0 for regular truncation

static unsigned int MAXSTUBSLINK = 108 + MAXOFFSET; //Max stubs per link
static unsigned int MAXLAYERROUTER = 108 + MAXOFFSET; //Max stubs handled by layer router
static unsigned int MAXDISKROUTER = 108 + MAXOFFSET; //Max stubs handled by disk router
static unsigned int MAXVMROUTER = 108 + MAXOFFSET; //Max stubs handled by VM router
static unsigned int MAXTE = 108 + MAXOFFSET; //Maximum number of stub pairs to try in TE 
static unsigned int MAXTRE = 108 + MAXOFFSET; //Maximum number of stub pairs to try in TRE 
static unsigned int MAXTC = 108 + MAXOFFSET; //Maximum number of tracklet parameter calculations
static unsigned int MAXPROJROUTER = 108 + MAXOFFSET; //Maximum number of projections to route
static unsigned int MAXME = 108 + MAXOFFSET; //Maximum number of stub-projection matches to try
static unsigned int MAXMC = 108 + MAXOFFSET; //Maximum number of match calculations
static unsigned int MAXFIT = 108 + MAXOFFSET; //Maximum number of track fits


static double dphisector=2*M_PI/NSector;

//Constants for defining stub representations
static int nbitsrL123=7;
static int nbitsrL456=7;

static int nbitszL123=12;
static int nbitszL456=8;

static int nbitsphistubL123=14;
static int nbitsphistubL456=17;

static int nrbitsdisk=12;
static int nzbitsdisk=7;

static int nrbitsprojdisk=12;
static int nrbitsprojderdisk=9;

static int nbitsphiprojL123=nbitsphistubL123;
static int nbitsphiprojL456=nbitsphistubL456;

static int nbitszprojL123=12;
static int nbitszprojL456=hourglassExtended?12:8;

static int nbitsphiprojderL123=hourglassExtended?16:8+2;
static int nbitsphiprojderL456=hourglassExtended?16:8+2;

static int nbitszprojderL123=8+2;
static int nbitszprojderL456=7+2;

//vm stubs
static int nfinephibarrelinner=2;
static int nfinephibarrelouter=3;

static int nfinephidiskinner=1;
static int nfinephidiskouter=2;

static int nfinephioverlapinner=1;
static int nfinephioverlapouter=2;


//Bits used to store track parameter in tracklet
static int nbitsrinv=14;
static int nbitsphi0=18;
static int nbitsd0=13;
static int nbitst=14;
static int nbitsz0=10;

//Minimal ranges for track parameters
static double maxrinv=0.006;
//static double maxphi0=0.59;
//static double maxt=9.0;
//static double maxz0=28.0;
static double maxd0=10.;

static double rmin[6]={rminL1,rminL2,rminL3,rminL4,rminL5,rminL6};

//These are constants defining global coordinate system

static double kphi=dphisectorHG/(1<<nbitsphistubL123);
static double kphi1=dphisectorHG/(1<<nbitsphistubL456);
static double kz=2*zlength/(1<<nbitszL123);
//static double kr=2*drmax/(1<<nbitsrL456);
static double kr=rmaxdisk/(1<<nrbitsdisk);
static double kd0 = 2*maxd0/(1<<nbitsd0);

//track and tracklet parameters
const int rinv_shift = -8;  // Krinv = 2^shift * Kphi/Kr
const int phi0_shift = 1;   // Kphi0 = 2^shift * Kphi
const int t_shift    = -10; // Kt    = 2^shift * Kz/Kr
const int z0_shift   = 0;   // Kz0   = 2^shift * kz

//projections are coarsened from global to stub precision  

//projection to R parameters
const int PS_phiL_shift = 0;   // phi projections have global precision in ITC
const int SS_phiL_shift = 0;   
const int PS_zL_shift   = 0;   // z projections have global precision in ITC
const int SS_zL_shift   = 0;

const int PS_phiderL_shift = -5;   // Kderphi = 2^shift * Kphi/Kr
const int SS_phiderL_shift = -5; 
const int PS_zderL_shift   = -7;  // Kderz = 2^shift * Kz/Kr
const int SS_zderL_shift   = -7;  
  
//projection to Z parameters
const int PS_phiD_shift = 3;   
const int SS_phiD_shift = 3;   
const int PS_rD_shift   = 1;   // a bug?! coarser by a factor of two then stubs??
const int SS_rD_shift   = 1;

const int PS_phiderD_shift = -4; //Kderphidisk = 2^shift * Kphi/Kz
const int SS_phiderD_shift = -4; 
const int PS_rderD_shift   = -6;  //Kderrdisk = 2^shift * Kr/Kz
const int SS_rderD_shift   = -6;  

//constants derivative from the above
static double krinvpars, kphi0pars, kd0pars, ktpars, kz0pars;
static double kphiproj123, kphiproj456, kzproj, kphider, kzder;
static double krprojshiftdisk, kphiprojdisk,krprojderdisk;
static double krdisk,krprojderdiskshift, kzpars;

//numbers needed for matches & fit, unclear what they are.
static int idrinvbits=19;
static int phi0bitshift=1;
static int rinvbitshift=13;
static int tbitshift=9;
static int z0bitshift=0;
static int phiderbitshift=7;
static int zderbitshift=6;
static int t2bits=23;
static int t3shift=8;
//static int t4shift=8;
//static int t4shift2=8;
//static int t6bits=12;
static int rinvbitshiftdisk=13; 
//static int phi0bitshiftdisk=1;
static int rprojdiskbitshift=6;
static int phiderdiskbitshift=20;
static int rderdiskbitshift=7;


static int phiresidbits=hourglassExtended?16:12; 
static int zresidbits=hourglassExtended?16:9;
static int rresidbits=hourglassExtended?16:7;

//Trackfit
static int fitrinvbitshift=9;  //6 OK?
static int fitphi0bitshift=6;  //4 OK?
static int fittbitshift=10;     //4 OK? //lower number gives rounding problems
static int fitz0bitshift=8;    //6 OK?

//r correction bits
static int rcorrbits=6;

static int chisqphifactbits=14;
static int chisqzfactbits=14;

//Duplicate Removal
static int minIndStubs=3; // Not for merge removal

#ifdef USEHYBRID
static std::string RemovalType="ichi"; //"merge";
#else
static std::string RemovalType="ichi";
#endif

//"ichi" (pairwise, keep track with best ichisq), "nstub" (pairwise, keep track with more stubs), "grid" (TMTT-like removal), "" (no removal)
static bool fakefit_5par=false; //if true, this would use KF 5-parameter fit for displaced tracking, false means use tracklet parameters instead (i.e. no fit)

#endif




