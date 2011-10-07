#ifndef ME_hh
#define ME_hh

#include <vector>
#include <map>
#include <string>
#include <TString.h>

class MEChannel;

class ME
{
public:

  // ECAL regions
  enum EcalRegion { iEEM=0, iEBM, iEBP, iEEP, iSizeE };

  enum EcalUnit   { iEcalRegion=0, iSector, iLMRegion, iLMModule, 
		    iSuperCrystal, iCrystal, iSizeG };

  enum EcalElec   { iElectronicChannel=iSizeG, 
		    iHVChannel, iLVChannel, iSizeU };

  // ECAL region from Laser Monitoring Region
  static int ecalRegion( int ilmr );
  static bool isBarrel( int ilmr );
  
  // Laser Monitoring Region from dcc and side
  static int lmr( int idcc, int iside );

  // ECAL Region, Sector, dcc and side from Laser Monitoring Region
  static void regionAndSector( int ilmr, int& ireg, int& isect, int& idcc, int& iside );

  // dcc and side from the laser monitoring region
  static std::pair<int, int> dccAndSide( int ilmr );

  // get trees of channels
  static MEChannel* regTree( int ireg );
  static MEChannel* lmrTree( int ilmr );
  static bool useElectronicNumbering;

  typedef int DCCid;
  typedef int LMRid;
  typedef int LMMid;
  typedef int PNid;
  enum PN { iPNA=0, iPNB };
  enum PNNORM { iPNNORM=0, iPNANORM, iPNBNORM };
  static std::pair<ME::DCCid,ME::PNid> pn( ME::LMRid ilmr, 
					   ME::LMMid ilmmod, 
					   ME::PN    ipn );
  static std::pair<ME::DCCid,ME::DCCid>  memFromLmr( ME::LMRid ilmr );
  static std::vector<ME::LMMid>        lmmodFromLmr( ME::LMRid ilmr );
  static std::vector<ME::DCCid>          memFromDcc( ME::DCCid idcc ); 
  static std::vector<ME::LMMid>        lmmodFromDcc( ME::DCCid idcc );
  static std::vector< int>             apdRefChannels( ME::LMMid ilmmod , ME::LMRid ilmr);

  // ECAL Region names
  static TString region[4];

  // unit names
  static TString granularity[iSizeG];

  // Super-Module name from Laser Monitoring Region
  static TString smNameFromDcc( int idcc );
  static TString smName( int ilmr );

  // Super-Module name from ECAL Region and Super-Module
  static TString smName( int ireg, int ism );
  
  enum RunType    { iLaser=0, iTestPulse, iLED, iSizeT };  
  enum Color      { iBlue=0, iGreen, iRed, iIRed, iSizeC };   
  enum Gain       { iVfeGain12=1, iVfeGain6, iVfeGain1, iSizeVfeGain };  
  enum PNGain     { iPnGain1=0, iPnGain16, iSizePnGain };  

  typedef unsigned long      Time;
  typedef unsigned long long TimeStamp;
  static const TimeStamp kLowMask;

  typedef struct 
  {
    std::string rundir; int dcc; int side;  int run; int lb; int events;
    TimeStamp ts_beg; TimeStamp ts_end;
  } Header;
  
  typedef struct
  {
    int type; int wavelength; int power; int filter; int delay; 
    int mgpagain; int memgain; 
  } Settings;

  // Database tables
  enum { iLmfLaserRun,
	 iLmfLaserConfig,  
	 iLmfLaserPulse,  
	 iLmfLaserPrim,   
	 iLmfLaserPnPrim,
	 iLmfTestPulseRun,
	 iLmfTestPulseConfig,  
	 iLmfTestPulsePrim,   
	 iLmfTestPulsePnPrim,
	 iLmfLEDRun,
	 iLmfLEDPrim,   
	 iLmfLEDPnPrim,
	 iLmfNLSRun,  
	 iLmfNLS,   
	 iLmfNLSRef,   
	 iSizeLmf   }; 

  // Laser primitive variables (table LmfLaserPrim)

  enum { iAPD_FLAG, iAPD_MEAN, iAPD_RMS, iAPD_M3, iAPD_NEVT, 
	 iAPD_OVER_PNA_MEAN, iAPD_OVER_PNA_RMS, iAPD_OVER_PNA_M3, iAPD_OVER_PNA_NEVT, 
	 iAPD_OVER_PNB_MEAN, iAPD_OVER_PNB_RMS, iAPD_OVER_PNB_M3, iAPD_OVER_PNB_NEVT,
	 iAPD_OVER_PN_MEAN, iAPD_OVER_PN_RMS, iAPD_OVER_PN_M3, iAPD_OVER_PN_NEVT, 
	 iAPD_OVER_PNACOR_MEAN, iAPD_OVER_PNACOR_RMS, iAPD_OVER_PNACOR_M3, iAPD_OVER_PNACOR_NEVT, 
	 iAPD_OVER_PNBCOR_MEAN, iAPD_OVER_PNBCOR_RMS, iAPD_OVER_PNBCOR_M3, iAPD_OVER_PNBCOR_NEVT,
	 iAPD_OVER_PNCOR_MEAN, iAPD_OVER_PNCOR_RMS, iAPD_OVER_PNCOR_M3, iAPD_OVER_PNCOR_NEVT,	
	 iAPD_OVER_APDA_MEAN, iAPD_OVER_APDA_RMS, iAPD_OVER_APDA_M3, iAPD_OVER_APDA_NEVT, // JM
	 iAPD_OVER_APDB_MEAN, iAPD_OVER_APDB_RMS, iAPD_OVER_APDB_M3, iAPD_OVER_APDB_NEVT, // JM
	 //iAPDABFIT_OVER_PNACOR_MEAN, iAPDABFIT_OVER_PNACOR_RMS, iAPDABFIT_OVER_PNACOR_M3, iAPDABFIT_OVER_PNACOR_NEVT, 
	 //iAPDABFIT_OVER_PNBCOR_MEAN, iAPDABFIT_OVER_PNBCOR_RMS, iAPDABFIT_OVER_PNBCOR_M3, iAPDABFIT_OVER_PNBCOR_NEVT,
	 //iAPDABFIT_OVER_PNCOR_MEAN, iAPDABFIT_OVER_PNCOR_RMS, iAPDABFIT_OVER_PNCOR_M3, iAPDABFIT_OVER_PNCOR_NEVT,
	 //iAPDABFIX_OVER_PNACOR_MEAN, iAPDABFIX_OVER_PNACOR_RMS, iAPDABFIX_OVER_PNACOR_M3, iAPDABFIX_OVER_PNACOR_NEVT, 
	 //iAPDABFIX_OVER_PNBCOR_MEAN, iAPDABFIX_OVER_PNBCOR_RMS, iAPDABFIX_OVER_PNBCOR_M3, iAPDABFIX_OVER_PNBCOR_NEVT,
	 //iAPDABFIX_OVER_PNCOR_MEAN, iAPDABFIX_OVER_PNCOR_RMS, iAPDABFIX_OVER_PNCOR_M3, iAPDABFIX_OVER_PNCOR_NEVT,
	 iAPD_SHAPE_COR, iAPD_ALPHA, iAPD_BETA,
	 iAPD_TIME_MEAN, iAPD_TIME_RMS, iAPD_TIME_M3, iAPD_TIME_NEVT,
	 iSizeAPD };
  static TString APDPrimVar[ iSizeAPD ];


  // Intermediate Laser primitive variables (table LmfLaserPrimCorr)

  enum { iMID_MEAN, iMID_RMS, iMID_NEVT, 
	 iMIDA_MEAN, iMIDA_RMS, iMIDA_NEVT, 
	 iMIDB_MEAN, iMIDB_RMS, iMIDB_NEVT,
	 iAPD_OVER_PNTMPCOR_MEAN, iAPD_OVER_PNTMPCOR_RMS, iAPD_OVER_PNTMPCOR_NEVT, 
	 iAPD_OVER_PNATMPCOR_MEAN, iAPD_OVER_PNATMPCOR_RMS, iAPD_OVER_PNATMPCOR_NEVT, 
	 iAPD_OVER_PNBTMPCOR_MEAN, iAPD_OVER_PNBTMPCOR_RMS, iAPD_OVER_PNBTMPCOR_NEVT,
	 iSizeMID }; // JM
  
  static TString MIDPrimVar[ iSizeMID ];
  

  // Corrected Laser primitive variables (table LmfNLS)

  enum { iNLS_MEAN, iNLS_RMS, iNLS_NEVT, iNLS_NORM, iNLS_NMEAN, iNLS_ENORM, iNLS_FLAG,
		iCLS_MEAN, iCLS_RMS, iCLS_NEVT, iCLS_NORM, iCLS_NMEAN, iCLS_ENORM, iCLS_FLAG,
		iSizeNLS };


  static TString NLSVar[ iSizeNLS ]; // JM


  // Variable for CLS files
  
  //enum { iRefRun, iRefLB, iRefStartLow,
  //	 iRefStartHigh, iSizeNLSRef }; // JM

  // static TString CLSRefVar[ iSizeNLSRef ]; // JM



  // PN primitive variables (table LmfLaserPnPrim)
  enum { iPN_FLAG, iPN_MEAN, iPN_RMS, iPN_M3, iPN_NEVT, // JM 
	 iPNA_OVER_PNB_MEAN, iPNA_OVER_PNB_RMS, iPNA_OVER_PNB_M3,iPN_SHAPE_COR,
	 iSizePN };
  static TString PNPrimVar[ iSizePN ];
  
  // MATAQ Primitive variables (table iLmfLaserPulse)
  enum { iMTQ_FIT_METHOD, 
	 iMTQ_AMPL, iMTQ_TIME, iMTQ_RISE, 
	 iMTQ_FWHM, iMTQ_FW10, iMTQ_FW05, iMTQ_SLIDING,   
	 iSizeMTQ };
  static TString MTQPrimVar[ iSizeMTQ ];
  
  // TP-APD Primitive variables (table iLmfTestPulsePrim)
  enum { iTPAPD_FLAG, iTPAPD_MEAN, iTPAPD_RMS, iTPAPD_M3, iTPAPD_NEVT,
	 iSizeTPAPD };
  static TString TPAPDPrimVar[ iSizeTPAPD ];

  // TP-PN Primitive variables (table iLmfTestPulsePnPrim)
  enum { iTPPN_GAIN, iTPPN_MEAN, iTPPN_RMS, iTPPN_M3, 
	 iSizeTPPN };
  static TString TPPNPrimVar[ iSizeTPPN ];

  // Time functions
  enum TUnit { iDay, iHour, iMinute, iSecond, iSizeTU };
  static std::vector< Time > timeDiff( Time t1, Time t2, 
				       short int& sign );
  static float timeDiff( Time t1, Time t0, int tunit=iHour );
  static Time  time( float dt, Time t0, int tunit=iHour );
  static Time  time_low( TimeStamp t );
  static Time  time_high( TimeStamp t );

  static TString type[iSizeT];
  static TString color[iSizeC];

  // get file names
  static TString path();               // MusEcal main working directory
  static TString primPath(  int lmr );   // where the primitives are
  static TString nlsPath(  int lmr );   // where the corrected values are
  static TString lmdataPath(  int lmr ); // where the LM data are
  static TString rootFileName( ME::Header header, ME::Settings settings );
  static TString rootNLSFileName( ME::Header header, ME::Settings settings );
  static TString rootNormFileName( ME::Header header, ME::Settings settings );
  static TString runListName( int lmr, int type, int color );
  static TString runListNLSName( int lmr );

  virtual ~ME() {}

  static std::vector<MEChannel*> _trees;
  //GHM  ClassDef(ME,0) // ME -- MusEcal name space
};

#endif
