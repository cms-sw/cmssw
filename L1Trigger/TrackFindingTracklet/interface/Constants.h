#ifndef CONSTANTS_H
#define CONSTANTS_H

//Uncomment if you want root output
//#define USEROOT

//Uncomment to run the hybrid algorithm
#ifdef CMSSW_GIT_HASH
#define USEHYBRID
#endif

//Uncomment to use HLS version of KF. Also read TrackFindingTMTT/README_HLS.txt
#ifdef USEHYBRID
//#define USE_HLS
#endif

//Uncomment to run the HLS version of the KF if using the Hybrid (instead of the C++ KF).
//(Please also follow the instructions in L1Trigger/TrackFindingTMTT/README_HLS.txt).
//#define USE_HLS

static const bool doKF=true; //true => use KF (assumes USEHYBRID is defined)
static const bool printDebugKF=false; // if true print lots of debugging statements related to the KF fit
static const bool bookHistos=false; //set to true/false to turn on/off histogram booking internal to the tracking (class "HistImp")

static unsigned int nHelixPar = 4; // 4 or 5 param helix fit.
static bool hourglassExtended=false; // turn on displaced tracking, also edit L1TrackNtupleMaker_cfg.py (search for "Extended" on several lines)

//Gemetry extensions -- used only by stand-alone code.
static std::string geomext=hourglassExtended?"hourglassExtended":"hourglass";  

static const bool geomTkTDR=false; // false => newest T14 tracker, true => "TDR" (T5/T6 tracker, D21/D11/D17 CMS geometries)

//static const double cSpeed=2.99792458e10; // Speed of light (cm/s) => these are currently not used, comment out
//static double bField=3.81120228767395;    // Assumed B-field

static const int TMUX = 6;

static std::string fitpatternfile="../data/fitpattern.txt";

//If this string is non-empty we will write ascii file with
//processed events
static const std::string skimfile="";
//static const std::string skimfile="evlist_skim.txt";

//Debug options (should be false for 'normal' operation)
static const bool dumppars=false;
static const bool dumpproj=false;

static const bool writeInvTable=false;    //Write out tables of drinv and invt in tracklet calculator for Verilog module
static const bool writeVerilog=false;     //Write out auto-generated Verilog mudules used by TCs

//static const bool writeDTCLinks=false;
static const bool writeIL=false;
static const bool writeStubsLayer=false;
static const bool writeStubsLayerperSector=false;
static const bool writeAllStubs=false;
static const bool writeVMOccupancyME=false;
static const bool writeVMOccupancyTE=false;
static const bool writeSeeds=false;
static const bool writeTE=false;
static const bool writeTED=false;
static const bool writeTRE=false;
static const bool writeTrackletProcessor=false;
static const bool writeTrackletCalculator=false;
static const bool writeTrackletCalculatorDisplaced=false;
static const bool writeTrackletPars=false;
static const bool writeAllProjections=false;
static const bool writeVMProjections=false;
static const bool writeTrackProjOcc=false;
static const bool writeME=false;
static const bool writeMatchCalculator=false;
static const bool writeResiduals=false;
static const bool writeFitTrack=false;
static const bool writeChiSq=false;

static const bool writeTC=false; //if true write out which memories track projetions will fill

static const bool writeNMatches=false;
static const bool writeHitEff=false;

// For HLS LUTs: makes test/*.txt files used to load LUTs in HLS code (for sector 0 only).
static const bool writeVMRTables = false; //write tables used by VMRouter
static const bool writeTETables=false;
static const bool writeVMTables=false;
static const bool writeMETables=false;
static const bool writeMCcuts=false;
static const bool writeHLSInvTable=false; //Write out tables of drinv and invt in tracklet calculator for HLS module
static const bool writeFitDerTable=false; //Write out track derivative tables
static const bool writeTripletTables=false; //Train and write the TED and TRE tables. N.B.: the tables
                                      //cannot be applied while they are being trained, i.e.,
                                      //this flag effectively turns off the cuts in
                                      //TrackletEngineDisplaced and TripletEngine
static const bool writeHLS=false;         //Write out auto-generated HLS modules used by TCs

// For HLS testing: produce data/MemPrints/*/*.dat files of input/output data of processing modules.
static const unsigned int writememsect=3;  // Restricts output to a single sector to speed up/reduce output size.
static const bool writememLinks = false; //Write files corresponding to data arriving on dtc links.
static const bool writemem = false; // Write files corresponding to memory modules.

static const bool writeCabling=false;
static const bool writeHitPattern=false;
static const bool writeTrackletParsOverlap=false;
static const bool writeTrackletParsDisk=false;

static const bool writeAllCT=false; //write out .dat file containing all output tracks in bitwise format

static const bool writeVariance=false; //write out residuals for variand matrix determination
static const bool writeResEff=false; //write files for making resolution & efficiency plots for standable code version
static const bool writePars=false; //write files for making plots of track parameters

static const bool writeMatchEff=true; //write files for making plots with truth matched efficiency



static const bool writestubs=false;  // write input stubs in the normal format
static const bool writestubs_in2=false;  // write input stubs in hardware-ready format
static const bool writeifit=false;
static const bool padding=true;

static const bool useMSFit=false;
static const bool tcorrection=true;
static const bool exactderivatives=false;  //for both the integer and float
static const bool exactderivativesforfloating=true; //only for the floating point
static const bool useapprox=true; //use approximate postion based on integer representation for floating point
static const bool usephicritapprox=true; //use approximate version of phicrit cut
static const int alphashift=12;  
static const int nbitsalpha=4;  //bits used to store alpha
static const int alphaBitsTable=2; //For number of bits in track derivative table
static const int nrinvBitsTable=3; //number of bits for tabulating rinv dependence
static const bool writetrace=false; //Print out details about startup
static const bool debug1=false; //Print detailed debug information about tracking
static const bool writeoutReal = false; 

static const bool warnNoMem=false;  //If true will print out warnings about missing projection memories

//Program flow (should be true for normal operation)
//enables the stub finding in these layer/disk combinations
static const bool doL1L2=true;
static const bool doL2L3=true;
static const bool doL3L4=true;
static const bool doL5L6=true;

static const bool doD1D2=true; 
static const bool doD3D4=true;

static const bool doL1D1=true;
static const bool doL2D1=true;

static const bool doL3L4L2=true; // only run if hourglassExtended is true
static const bool doL5L6L4=true; // only run if hourglassExtended is true
static const bool doL2L3D1=true; // only run if hourglassExtended is true
static const bool doD1D2L2=true; // only run if hourglassExtended is true

static const bool allSector=false; //if true duplicate stubs in all sectors

static const bool doProjections=true;

static const int MEBinsBits=3;
static const int MEBins=(1<<MEBinsBits);

static const int MEBinsDisks=8; //on each side

//Geometry 

//These define the length scale for both r and z
static const double zlength=120.0;
static const double rmaxdisk=120.0;


// these assume either "TDR" tracker geometry (T5 or T6), or otherwise most recent T14 tracker 
// T5: http://cms-tklayout.web.cern.ch/cms-tklayout/layouts/recent-layouts/OT616_200_IT404/layout.html
// T14: http://cms-tklayout.web.cern.ch/cms-tklayout/layouts/recent-layouts/OT616_200_IT404/layout.html

static const double rmeanL1=geomTkTDR?(rmaxdisk*858)/4096:(rmaxdisk*851)/4096;
static const double rmeanL2=geomTkTDR?(rmaxdisk*1279)/4096:(rmaxdisk*1269)/4096;
static const double rmeanL3=geomTkTDR?(rmaxdisk*1795)/4096:(rmaxdisk*1784)/4096;
static const double rmeanL4=geomTkTDR?(rmaxdisk*2347)/4096:(rmaxdisk*2347)/4096;
static const double rmeanL5=geomTkTDR?(rmaxdisk*2937)/4096:(rmaxdisk*2936)/4096;
static const double rmeanL6=geomTkTDR?(rmaxdisk*3783)/4096:(rmaxdisk*3697)/4096;

static const double zmeanD1=(zlength*2239)/2048;
static const double zmeanD2=(zlength*2645)/2048;
static const double zmeanD3=(zlength*3163)/2048;
static const double zmeanD4=(zlength*3782)/2048;
static const double zmeanD5=(zlength*4523)/2048;


static const double rmindiskvm=22.5;
static const double rmaxdiskvm=67.0;

static const double rmaxdiskl1overlapvm=45.0;
static const double rmindiskl2overlapvm=40.0;
static const double rmindiskl3overlapvm=50.0;

static const double half2SmoduleWidth=4.57;

// need separate lookup values for inner two vs outer three disks for 2S modules

// T5 tracker geometry (= D11, D17, D21, ... CMS geometry)!
// http://cms-tklayout.web.cern.ch/cms-tklayout/layouts/recent-layouts/OT616_200_IT404/layout.html
//static const double rDSSinner[10] = {66.7728, 71.7967, 77.5409, 82.5584, 84.8736, 89.8953, 95.7791, 100.798, 102.495, 107.52};  // <=== these 10 are for inner 2 disks
//static const double rDSSouter[10] = {65.1694, 70.1936, 75.6641, 80.6908, 83.9581, 88.9827, 94.6539, 99.6772, 102.494, 107.519}; // <=== these 10 are for outer 3 disks

// T14 tracker geometry (= D41 CMS geometry) 
// http://cms-tklayout.web.cern.ch/cms-tklayout/layouts/recent-layouts/OT616_200_IT404/layout.html
//static const double rDSSinner[10] = {66.4391, 71.4391, 76.275, 81.275, 82.9550, 87.9550, 93.815, 98.815, 99.816, 104.816};
//static const double rDSSouter[10] = {63.9903, 68.9903, 74.275, 79.275, 81.9562, 86.9562, 92.492, 97.492, 99.816, 104.816};

static const double rDSSinner_mod1 = geomTkTDR?69.2345:68.9391;
static const double rDSSinner_mod2 = geomTkTDR?80.0056:78.7750;
static const double rDSSinner_mod3 = geomTkTDR?87.3444:85.4550;
static const double rDSSinner_mod4 = geomTkTDR?98.2515:96.3150;
static const double rDSSinner_mod5 = geomTkTDR?104.9750:102.3160;

static const double rDSSouter_mod1 = geomTkTDR?67.6317:66.4903;
static const double rDSSouter_mod2 = geomTkTDR?78.1300:76.7750;
static const double rDSSouter_mod3 = geomTkTDR?86.4293:84.4562;
static const double rDSSouter_mod4 = geomTkTDR?97.1316:94.9920;
static const double rDSSouter_mod5 = geomTkTDR?104.9750:102.3160;

static const double halfstrip = 2.5; //we want the center of the two strip positions in a module, not just the center of a module 

static const double rDSSinner[10] = {rDSSinner_mod1-halfstrip, rDSSinner_mod1+halfstrip, rDSSinner_mod2-halfstrip, rDSSinner_mod2+halfstrip, rDSSinner_mod3-halfstrip, rDSSinner_mod3+halfstrip,
			       rDSSinner_mod4-halfstrip, rDSSinner_mod4+halfstrip, rDSSinner_mod5-halfstrip, rDSSinner_mod5+halfstrip};
static const double rDSSouter[10] = {rDSSouter_mod1-halfstrip, rDSSouter_mod1+halfstrip, rDSSouter_mod2-halfstrip, rDSSouter_mod2+halfstrip, rDSSouter_mod3-halfstrip, rDSSouter_mod3+halfstrip, 
			       rDSSouter_mod4-halfstrip, rDSSouter_mod4+halfstrip, rDSSouter_mod5-halfstrip, rDSSouter_mod5+halfstrip};


static const double drmax=rmaxdisk/32.0;

static const double dzmax=zlength/32.0;

static const double drdisk=rmaxdisk;

static const double rmean[6]={rmeanL1,rmeanL2,rmeanL3,rmeanL4,rmeanL5,rmeanL6};

static const double zmean[5]={zmeanD1,zmeanD2,zmeanD3,zmeanD4,zmeanD5};


static const unsigned int nallstubslayers[6]={8,4,4,4,4,4};
static const unsigned int nvmtelayers[6]={4,8,4,8,4,8};
static const unsigned int nvmteextralayers[6]={0,4,4,0,0,0};

static const unsigned int nallprojlayers[6]={8,4,4,4,4,4};
static const unsigned int nvmmelayers[6]={4,8,8,8,8,8};

static const unsigned int nallstubsdisks[5]={4,4,4,4,4};
static const unsigned int nvmtedisks[5]={4,4,4,4,4};

static const unsigned int nallprojdisks[5]={4,4,4,4,4};
static const unsigned int nvmmedisks[5]={8,4,4,4,4};
//for seeding in L1D1 L2D1
static const unsigned int nallstubsoverlaplayers[3] = {8, 4, 4}; 
static const unsigned int nvmteoverlaplayers[3] = {2, 2, 2};

static const unsigned int nallstubsoverlapdisks[2] = {4, 4}; 
static const unsigned int nvmteoverlapdisks[2] = {4, 4};

static const double rcrit=55.0;


static const double rmaxL1=rmeanL1+drmax; 
static const double rmaxL2=rmeanL2+drmax; 
static const double rmaxL3=rmeanL3+drmax; 
static const double rmaxL4=rmeanL4+drmax; 
static const double rmaxL5=rmeanL5+drmax; 
static const double rmaxL6=rmeanL6+drmax; 


static const double zminD1=zmeanD1-dzmax; 
static const double zmaxD1=zmeanD1+dzmax; 
static const double zminD2=zmeanD2-dzmax; 
static const double zmaxD2=zmeanD2+dzmax; 
static const double zminD3=zmeanD3-dzmax; 
static const double zmaxD3=zmeanD3+dzmax; 
static const double zminD4=zmeanD4-dzmax; 
static const double zmaxD4=zmeanD4+dzmax; 
static const double zminD5=zmeanD5-dzmax; 
static const double zmaxD5=zmeanD5+dzmax; 

static const double two_pi=2*M_PI;

static const double ptcut=1.91; //Minimum pt
static const double rinvcut=0.01*0.3*3.8/ptcut; //0.01 to convert to cm-1
static const double ptcutte=1.8; //Minimum pt in TE
static const double rinvcutte=0.01*0.3*3.8/ptcutte; //0.01 to convert to cm-1 in TE
static const double bendcut=1.25;
static const double bendcutdisk=1.25;
static const double z0cut=15.0;
static const double mecut=2.0;
static const double mecutdisk=1.5;


static const unsigned int NSector=9; 
static const int Nphibits=2;         //Number of bits required to label the phi VM
static const int L1Nphi=(1<<Nphibits)-1; //Number of odd layer VMs
static const int Nzbits=3;         //Number of bits required to label the z VM
static const int L1Nz=(1<<Nzbits); //Number of z VMs in odd layers
//static const int VMzbits=4;        //Number of bits for the z position in VM
static const int L2Nphi=(1<<Nphibits); //Number of even layer VMs
static const int L2Nz=(1<<Nzbits); //Number of z VMs in even layers
//static const int VMrbits=2;        //Number of bits for r position 'in VM'
static const int VMphibits=3;      //Number of bits for phi position in VM

static const double rinvmax=0.01*0.3*3.8/2.0; //0.01 to convert to cm-1

static const double dphisectorHG=2*M_PI/NSector+2*fmax(fabs(asin(0.5*rinvmax*rmean[0])-asin(0.5*rinvmax*rcrit)),
						fabs(asin(0.5*rinvmax*rmean[5])-asin(0.5*rinvmax*rcrit)));

static const double phicritmin=0.5*dphisectorHG-M_PI/NSector;
static const double phicritmax=dphisectorHG-0.5*dphisectorHG+M_PI/NSector;

static const double dphicritmc=0.005; //lose for MC
static const double phicritminmc=phicritmin-dphicritmc;
static const double phicritmaxmc=phicritmax+dphicritmc;

// these are tuned such that all tracklets passing the exact phicrit cut also
// pass the approximate version in high-pileup events
// these two numbers are not used in Jack's solution : search phicritmaxmc or phicritmaxmc in TC for more details 
static const int phicritapproxminmc=9253;
static const int phicritapproxmaxmc=56269;

static const unsigned int NLONGVMBITS=3; 
static const unsigned int NLONGVMRBITS=3;   //4 bins on each side (+-z)
static const unsigned int NLONGVMBINS=(1<<NLONGVMBITS);
static const unsigned int NLONGVMRBINS=(1<<NLONGVMRBITS);
static const unsigned int NLONGVMODDLAYERBITS=6;
static const unsigned int NLONGVMODDDISKBITS=6;
static const double rinnerdisk=22.0;



//limits per FED region
//static const int NMAXstub  = 250;
//static const int NMAXroute = 250;

static const unsigned int MAXOFFSET=10000; //set to 0 to enable regular truncation or 10000 to disable it.

static const unsigned int MAXSTUBSLINK = 108 + MAXOFFSET; //Max stubs per link
static const unsigned int MAXLAYERROUTER = 108 + MAXOFFSET; //Max stubs handled by layer router
static const unsigned int MAXDISKROUTER = 108 + MAXOFFSET; //Max stubs handled by disk router
static const unsigned int MAXVMROUTER = 108 + MAXOFFSET; //Max stubs handled by VM router
static const unsigned int MAXTE = 108 + MAXOFFSET; //Maximum number of stub pairs to try in TE 
static const unsigned int MAXTRE = 108 + MAXOFFSET; //Maximum number of stub pairs to try in TRE 
static const unsigned int MAXTC = 108 + MAXOFFSET; //Maximum number of tracklet parameter calculations
static const unsigned int MAXPROJROUTER = 108 + MAXOFFSET; //Maximum number of projections to route
static const unsigned int MAXME = 108 + MAXOFFSET; //Maximum number of stub-projection matches to try
static const unsigned int MAXMC = 108 + MAXOFFSET; //Maximum number of match calculations
static const unsigned int MAXFIT = 108 + MAXOFFSET; //Maximum number of track fits


static const double dphisector=2*M_PI/NSector;

//Constants for defining stub representations
static const int nbitsrL123=7;
static const int nbitsrL456=7;

static const int nbitszL123=12;
static const int nbitszL456=8;

static const int nbitsphistubL123=14;
static const int nbitsphistubL456=17;

static const int nrbitsdisk=12;
static const int nzbitsdisk=7;

static const int nrbitsprojdisk=12;
static const int nrbitsprojderdisk=9;

static const int nbitsphiprojL123=nbitsphistubL123;
static const int nbitsphiprojL456=nbitsphistubL456;

static const int nbitszprojL123=12;
static const int nbitszprojL456=8;

static const int nbitsphiprojderL123=8+2;
static const int nbitsphiprojderL456=8+2;

static const int nbitszprojderL123=8+2;
static const int nbitszprojderL456=7+2;

//vm stubs
static const int nfinephibarrelinner=2;
static const int nfinephibarrelouter=3;

static const int nfinephidiskinner=2; //too small!
static const int nfinephidiskouter=3;

static const int nfinephioverlapinner=2;
static const int nfinephioverlapouter=3;


//Bits used to store track parameter in tracklet
static const int nbitsrinv=14;
static const int nbitsphi0=18;
static const int nbitsd0=13;
static const int nbitst=14;
static const int nbitsz0=10;

//Minimal ranges for track parameters
static const double maxrinv=0.006;
//static const double maxphi0=0.59;
//static const double maxt=9.0;
//static const double maxz0=28.0;
static const double maxd0=10.;


//These are constants defining global coordinate system

static const double kphi=dphisectorHG/(1<<nbitsphistubL123);
static const double kphi1=dphisectorHG/(1<<nbitsphistubL456);
static const double kz=2*zlength/(1<<nbitszL123);
//static const double kr=2*drmax/(1<<nbitsrL456);
static const double kr=rmaxdisk/(1<<nrbitsdisk);
static const double kd0 = 2*maxd0/(1<<nbitsd0);

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
static const int idrinvbits=19;
static const int phi0bitshift=1;
static const int rinvbitshift=13;
static const int tbitshift=9;
static const int z0bitshift=0;
static const int phiderbitshift=7;
static const int zderbitshift=6;
static const int t2bits=23;
static const int t3shift=8;
//static const int t4shift=8;
//static const int t4shift2=8;
//static const int t6bits=12;
static const int rinvbitshiftdisk=13; 
//static const int phi0bitshiftdisk=1;
static const int rprojdiskbitshift=6;
static const int phiderdiskbitshift=20;
static const int rderdiskbitshift=7;


static const int phiresidbits=12; 
static const int zresidbits=9;
static const int rresidbits=7;

//Trackfit
static const int fitrinvbitshift=9;  //6 OK?
static const int fitphi0bitshift=6;  //4 OK?
static const int fittbitshift=10;     //4 OK? //lower number gives rounding problems
static const int fitz0bitshift=8;    //6 OK?

//r correction bits
static const int rcorrbits=6;

static const int chisqphifactbits=14;
static const int chisqzfactbits=14;

//Duplicate Removal
static const int minIndStubs=3; // not used with merge removal
//"ichi" (pairwise, keep track with best ichisq), "nstub" (pairwise, keep track with more stubs), "grid" (TMTT-like removal), "" (no removal), "merge" (hybrid dup removal)

#ifdef USEHYBRID
static const std::string RemovalType="merge";
// "CompareBest" (recommended) Compares only the best stub in each track for each region (best = smallest phi residual) and will merge the two tracks if stubs are shared in three or more regions
// "CompareAll" Compares all stubs in a region, looking for matches, and will merge the two tracks if stubs are shared in three or more regions
static const std::string MergeComparison="CompareBest";
#else
static const std::string RemovalType="ichi";
#endif
//static const std::string RemovalType=""; // Run without duplicate removal

static const bool fakefit=false; //if true, run a dummy fit, producing TTracks directly from output of tracklet pattern reco stage. (Not working for Hybrid)

//projection layers by seed index
static const int projlayers[12][4] = {
  {3, 4, 5, 6},  //0 L1L2
  {1, 2, 5, 6},  //1 L3L4
  {1, 2, 3, 4},  //2 L5L6    
  {1, 2},  //3 D1D2    
  {1, 2},  //4 D3D4    
  {1, 2},  //5 L1D1    
  {1, 2},  //6 L2D1    
  {1, 4, 5, 6},  //7 L2L3
  {1, 5, 6}, //8 L2L3L4
  {1, 2, 3}, //9 L4L5L6
  {1}, //10 L2L3D1
  {} //11 D1D2L2
};

//projection disks by seed index
static const int projdisks[12][5] = {
  {1, 2, 3, 4, 5}, //0 L1L2
  {1, 2, 3, 4, 5}, //1 L3L4
  {1, 2, 3, 4, 5}, //2 L5L6
  {3, 4, 5},       //3 D1D2    
  {1, 2, 5},       //4 D3D4    
  {2, 3, 4, 5}, //5 L1D1    
  {2, 3, 4, 5}, //6 L2D1    
  {1, 2, 3, 4, 5}, //7 L2L3    
  {1, 2},          //8 L2L3L4
  {},              //9 L4L5L6
  {2, 3, 4},              //10 L2L3D1
  {}               //11 D1D2L2
};
  
  
#endif




