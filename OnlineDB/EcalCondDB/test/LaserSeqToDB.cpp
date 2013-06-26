#include <time.h>
#include <stdio.h>
#include <cmath>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cassert>

using namespace std;

#include <TString.h>
#include <TObjString.h>

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/all_lmf_types.h"
#include "OnlineDB/EcalCondDB/interface/RunDat.h"
#include "OnlineDB/EcalCondDB/interface/RunLaserRunDat.h"

// fixme
#include "OnlineDB/EcalCondDB/interface/LMFSeqDat.h"
#include "OnlineDB/EcalCondDB/interface/LMFLaserPulseDat.h"
#include "OnlineDB/EcalCondDB/interface/LMFPrimDat.h"
#include "OnlineDB/EcalCondDB/interface/LMFPnPrimDat.h"

#include "CalibCalorimetry/EcalLaserAnalyzer/interface/ME.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MEGeom.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MEEBGeom.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MEEEGeom.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MELaserPrim.h"

class CondDBApp 
{
public:

  CondDBApp( string sid, string user, string pass ) : _debug(0), _insert(true),_run(0), _seq(0), _type(0), _color(0), _t0(0), _t1(0)
  {
    _location="P5_Co";
    _runtype ="TEST";     //???
    _gentag  ="default";  //???

    try 
      {
	if( _debug>0 )
	  {
	    cout << "Making connection..." << endl;
	    cout << "->  sid="  << sid << endl;
	    cout << "-> user="  << user << endl;
	    cout << "-> pass="  << pass << endl;
	  }
	econn = new EcalCondDBInterface( sid, user, pass );
      } 
    catch (runtime_error &e ) 
      {
	cerr << e.what() << endl;
	exit(-1);
      } 
  }
  
  ~CondDBApp() 
  {
    delete econn;
  }
  
  void init( int irun, int iseq, int type, int color, time_t t0, time_t t1 )
  {
    _run   = irun;
    _seq   = iseq;
    _type  = type;
    _color = color;
    _t0    = t0;
    _t1    = t1;

    cout << endl;
    cout << "--- Transfering Laser Monitoring data to OMDS ---\n";
    cout << "---   for run/seq(t0/t1/Dt) " 
 	 << _run << "/" << _seq << "(" << _t0 << "/" << _t1 << "/" << _t1-_t0 << ")" << endl;
    
    // set the RunIOV
    setRun();

    // set trigger type and color
    setTriggerAndColor();

    // set the sequence
    setSequence();

    // set the map of runIOVs
    //setMapOfRunIOV();

  }

  void setRun()
  {
    LocationDef locdef;
    RunTypeDef rundef;
    
    locdef.setLocation(_location);
    rundef.setRunType(_runtype);
    
    _runtag.setLocationDef(locdef);
    _runtag.setRunTypeDef(rundef);
    _runtag.setGeneralTag(_gentag);
    
    run_t run = _run;
    
    if( _debug>0 ) cout << "Fetching run by tag..." << endl;
    _runiov = econn->fetchRunIOV(&_runtag, run);    
    printRunIOV(_runiov);

    // get logic id for all ECAL
    ecid_allEcal = econn->getEcalLogicID("ECAL");    
    
    //
    // Laser Run
    //

    map< EcalLogicID, RunLaserRunDat >    dataset;
    
    econn->fetchDataSet( &dataset, &_runiov );
    int n_ =  dataset.count( ecid_allEcal );
    if( n_==0 )
      {
    	if( _debug>0 ) cout << "RunLaserRunDat -- not created yet: create & insert it" << endl;
    	RunLaserRunDat rd;
    	rd.setLaserSequenceType("STANDARD");
    	rd.setLaserSequenceCond("IN_THE_GAP");
    	dataset[ecid_allEcal]=rd;
    	if( _debug>0 ) cout << "Inserting LaserRunDat " << flush;
    	if( _insert )
    	  {
    	    econn->insertDataSet( &dataset, &_runiov );
    	    if( _debug>0 ) cout << "...dummy..." << endl;
    	  }
    	if( _debug>0 ) cout << "Done." << endl;
      }
    else
      {
    	if( _debug>0 ) cout << "RunLaserRunDat -- already inserted" << endl;
      }    
  }

  void setTriggerAndColor()
  {
    //  Color id     : 0  blue - blue laser (440 nm) or blue led
    //  Color id     : 1  green - green (495 nm)
    //  Color id     : 2  red/orange - red laser (706 nm) or orange led
    //  Color id     : 3  IR - infrared (796 nm)
    int colorID = _color;

    //Trig id     : 0  las - laser
    //Trig id     : 1  led - led
    //Trig id     : 2  tp - test pulse
    //Trig id     : 3  ped - pedestal
    int trigID = _type;

    // LMF Run tag
    int tagID = 1;    
    {
      _lmfcol.setConnection(econn->getEnv(), econn->getConn());
      _lmfcol.setColor(colorID);
      _lmfcol.dump();
      //GO      _lmfcol.fetchID();
    }
    
    {
      _lmftrig.setConnection(econn->getEnv(), econn->getConn());
      //GO      _lmftrig.setByID(trigID);
      _lmftrig.setByID(trigID + 1);
      _lmftrig.fetchID();
    }
    
    {
      _lmftag.setConnection( econn->getEnv(), econn->getConn() );
      _lmftag.setByID(tagID);
      _lmftag.fetchID();
    }
  }

  void setSequence()
  {
    _lmfseq.setConnection(econn->getEnv(), econn->getConn());
    std::map<int, LMFSeqDat> l = _lmfseq.fetchByRunIOV(_runiov);
    cout << "Run has " << l.size() << " sequences in database" << endl;
    std::map<int, LMFSeqDat>::const_iterator b = l.begin();
    std::map<int, LMFSeqDat>::const_iterator e = l.end();
    bool ok_(false);
    while( b!=e )
      {
	_lmfseq = b->second;
	int iseq_ = (b->second).getSequenceNumber();
	if( iseq_ == _seq ) 
	  {
	    ok_ = true;
	    break;
	  }
	b++;
      }
    if( !ok_ )
      {
	LMFSeqDat seq_;
	cout << "Sequence does not exist -- create it " << endl;
	//	seq_.debug();
	seq_.setRunIOV( _runiov );
	seq_.setSequenceNumber( _seq );	
	// run times
	Tm startTm;
	Tm endTm;
	uint64_t microsStart = _t0;
	uint64_t microsEnd   = _t1;
	microsStart *= 1000000;
	microsEnd   *= 1000000;
	startTm.setToMicrosTime(microsStart);
	endTm.setToMicrosTime(microsEnd);
	seq_.setSequenceStart(startTm);
	seq_.setSequenceStop(endTm);
	seq_.setVersions(1,1);
	if( _insert ) {
	  econn->insertLmfSeq( &seq_ );
	}	
	_lmfseq = seq_;
      }
    printLMFSeqDat( _lmfseq ); 
  }

  void setMapOfRunIOV()
  {
    LMFRunIOV lmfiov;
    lmfiov.setConnection( econn->getEnv(), econn->getConn() ); 
    std::list< LMFRunIOV > iov_l = lmfiov.fetchBySequence( _lmfseq );
    std::list<LMFRunIOV>::const_iterator iov_b = iov_l.begin();
    std::list<LMFRunIOV>::const_iterator iov_e = iov_l.end();
    while( iov_b != iov_e ) 
      {
      if( iov_b->getTriggerType()==_lmftrig &&  iov_b->getColor()==_lmfcol )
	{
	  int ilmr = iov_b->getLmr();
	  _lmfRunIOV[ilmr] = *iov_b;
	}        
      iov_b++;
    }
  }

  void storeLaserPrimitive( const char* fname, ME::Header header, ME::Settings settings, time_t dt )
  {

    LMFRunIOV lmfiov;
    lmfiov.setConnection( econn->getEnv(), econn->getConn() ); 
    
    //     // Rustine
    TString normStr_ = "APD_OVER_PN";
    bool useAPDnorm_ = false;
    if( useAPDnorm_ ) normStr_ = "APD_OVER_APD";
    
    int table_id_(0);
    int id1_(0);
    int id2_(0);
    int id3_(EcalLogicID::NULLID);

    bool ok_(false);
    TString channelViewName_;
    
    int logic_id;
    // LMF Tag and IOV
    int type  = settings.type;
    int color = settings.wavelength;
    assert( type==_type && color==_color );
    time_t t_beg = _t0 + dt;
    time_t t_end = _t0 + dt + 5; //!!!GHM!!!
    
    int dcc   = header.dcc;
    int side  = header.side;    
    int  ilmr = ME::lmr( dcc, side );
    
    if( existsLMFRunIOV(ilmr) )
      {
	lmfiov = _lmfRunIOV[ilmr];
      }
    else
      {
	LMFRunIOV lmfiov_;

	lmfiov_.setConnection(econn->getEnv(), econn->getConn());
	lmfiov_.debug();
	lmfiov_.startProfiling();
	cout << "LMRunIOV not found -- create it " << endl;  
	
	// fill the fields
	lmfiov_.setLMFRunTag(   _lmftag  );
	lmfiov_.setSequence(    _lmfseq  );
	lmfiov_.setTriggerType( _lmftrig );
	lmfiov_.setColor(       _lmfcol  );
	lmfiov_.setLmr( ilmr );
	lmfiov_.setSubRunType("test");  // ?

	// run times
	Tm startTm;
	Tm endTm;
	uint64_t microsStart = t_beg;
	uint64_t microsEnd   = t_end;
	microsStart *= 1000000;
	microsEnd   *= 1000000;
	startTm.setToMicrosTime(microsStart);
	endTm.setToMicrosTime(microsEnd);
	lmfiov_.setSubRunStart(startTm);
	lmfiov_.setSubRunEnd(endTm);

	// insert
	if( _insert )
	  {
	    econn->insertLmfRunIOV(&lmfiov_);
	  }
	else
	  {
	    cout << "fake insert" << endl;
	  }

	// keep in the map 
	_lmfRunIOV[ilmr] = lmfiov_;
	lmfiov = _lmfRunIOV[ilmr];
      }
    {
      printLMFRunIOV( lmfiov );
    }


    bool isEB = ME::isBarrel( ilmr ); 

    //    econn->fetchDataSet( );
    

    //     // datasets
    //     map< EcalLogicID, LMFLaserPrimDat   >    dataset_prim;
    //     map< EcalLogicID, LMFLaserPNPrimDat >    dataset_pnprim; 
    //     map< EcalLogicID, LMFLaserPulseDat  >    dataset_pulse;

    //  Open the ROOT file
    TString rootfile( fname );
    TDirectory* f = TFile::Open( rootfile );
    if( f==0 ) 
      {
     	cout << "ERROR -- file=" << rootfile << " not found! " << endl;
	//	abort();  // this must not happen
     	return;
      }
    if( _debug>0 ) 
      cout << "File=" << rootfile << " found. " << endl;
	    
    TString sep("__");
    TString hname,  vname;
    TString table;
    TH2I* i_h;

    // LMF Run
    //    map< EcalLogicID, LMFRunDat         >    dataset_lmfrun;
    table = "LMF_RUN_DAT";
    TTree* lmfrun_t = (TTree*) f->Get(table);
    map< TString, unsigned int    > lmfrun_i;
    vname = "LOGIC_ID";      lmfrun_t->SetBranchAddress( vname, &lmfrun_i[vname] );
    vname = "NEVENTS";       lmfrun_t->SetBranchAddress( vname, &lmfrun_i[vname] );
    vname = "QUALITY_FLAG";  lmfrun_t->SetBranchAddress( vname, &lmfrun_i[vname] );
    lmfrun_t->LoadTree( 0 );
    lmfrun_t->GetEntry( 0 );
    logic_id =  (int)  lmfrun_i["LOGIC_ID"];
    int nevents = (int) lmfrun_i["NEVENTS"];
    int flag    = (int) lmfrun_i["QUALITY_FLAG"];
    if( _debug>0 ) 
      {
	cout << "nevents=" << nevents;
	cout << "\tquality flag=" << flag;
	cout << endl;
      }

    // build the logicID for the "fanout"
    table_id_ = MELaserPrim::iECAL_LMR;
    id1_ = 0;
    id2_ = 0;
    ok_ = MELaserPrim::getViewIds( logic_id, table_id_, id1_, id2_ );
    if( !ok_ )
      {
	cout << "warning -- inconsistent table_id [1] --> " << table_id_  << endl;
     	table_id_ = MELaserPrim::iECAL_LMR;
      }
    if( id1_!=ilmr )
      {
	cout << "warning -- inconsistent id1/lmr " << id1_ << "/" << ilmr << endl;
     	id1_ = ilmr;
      }
    //    table_id_ = logic_id/1000000;
    //    assert( table_id_==MELaserPrim::iECAL_LMR );
    //    id1_ = (logic_id%1000000)/10000;
    //    id2_ = 0;
    //    EcalLogicID ecid_lmfrun = econn->getEcalLogicID( "EB_LM_side", id1_, id2_ );    
    EcalLogicID ecid_lmfrun = econn->getEcalLogicID( "ECAL_LMR", id1_ );    
    if( _debug ) cout << ecid_lmfrun.getLogicID() << endl;
      {
     	// Set the data
     	LMFRunDat lmf_lmfrun(econn);
	lmf_lmfrun.setLMFRunIOV( lmfiov );

	// DOES NOT WORK
	lmf_lmfrun.setEvents(      ecid_lmfrun,  nevents );
	lmf_lmfrun.setQualityFlag( ecid_lmfrun,  flag );

	// WORKS
	//	vector< float > data(2);
	//	data[0] = nevents;
	//	data[1] = flag;
	//	lmf_lmfrun.setData( ecid_lmfrun, data );

	// DOES NOT WORK
	// lmf_lmfrun.setData( ecid_lmfrun, string("NEVENTS"), (float) nevents );
	if( _insert )
	  {
	    econn->insertLmfDat(&lmf_lmfrun);
	  }
      }	

      //
      // Laser Configuration
      //
    //     map< EcalLogicID, LMFLaserConfigDat >    dataset_config;
         table = "LMF_LASER_CONFIG_DAT";
         TTree* config_t = (TTree*) f->Get(table);
         map< TString, unsigned int    > config_i;
         vname = "LOGIC_ID";        config_t->SetBranchAddress( vname, &config_i[vname] );
         vname = "WAVELENGTH";      config_t->SetBranchAddress( vname, &config_i[vname] );
         vname = "VFE_GAIN";        config_t->SetBranchAddress( vname, &config_i[vname] );
         vname = "PN_GAIN";         config_t->SetBranchAddress( vname, &config_i[vname] );
         vname = "LSR_POWER";       config_t->SetBranchAddress( vname, &config_i[vname] );
         vname = "LSR_ATTENUATOR";  config_t->SetBranchAddress( vname, &config_i[vname] );
         vname = "LSR_CURRENT";     config_t->SetBranchAddress( vname, &config_i[vname] );
         vname = "LSR_DELAY_1";     config_t->SetBranchAddress( vname, &config_i[vname] );
         vname = "LSR_DELAY_2";     config_t->SetBranchAddress( vname, &config_i[vname] );
         config_t->LoadTree( 0 );
         config_t->GetEntry( 0 );
         assert( logic_id ==  (int)  config_i["LOGIC_ID"] );
         EcalLogicID ecid_config = econn->getEcalLogicID( ecid_lmfrun.getLogicID() );
	 //     econn->fetchDataSet( &dataset_config, &lmfiov );
	 //    n_ =  dataset_config.size();
    //     if( n_==0 )
    //       {   
    // 	// Set the data
	 LMFLaserConfigDat lmf_config(econn);
	 lmf_config.setLMFRunIOV(lmfiov);
     	lmf_config.setWavelength( ecid_config, (int) config_i["WAVELENGTH"]  );
     	lmf_config.setVFEGain(    ecid_config, (int) config_i["VFE_GAIN"]    );
     	lmf_config.setPNGain(     ecid_config, (int) config_i["PN_GAIN"]     );
     	lmf_config.setLSRPower(      ecid_config, (int) config_i["LSR_POWER"]   );
     	lmf_config.setLSRAttenuator( ecid_config, (int) config_i["LSR_ATTENUATOR"] );
     	lmf_config.setLSRCurrent(    ecid_config, (int) config_i["LSR_CURRENT"] );
     	lmf_config.setLSRDelay1(     ecid_config, (int) config_i["LSR_DELAY_1"] );
     	lmf_config.setLSRDelay2(     ecid_config, (int) config_i["LSR_DELAY_2"] );
	if (_insert) {
	  econn->insertLmfDat(&lmf_config);
	}
	
  
	 //
	 // Laser MATACQ Primitives
	 //
	 table = MELaserPrim::lmfLaserName( ME::iLmfLaserPulse, type, color );
	 TTree* pulse_t = (TTree*) f->Get(table);
	 map< TString, unsigned int    > pulse_i;
	 map< TString, float > pulse_f;
	 vname = "LOGIC_ID";        pulse_t->SetBranchAddress( vname, &pulse_i[vname] );
	 vname = "FIT_METHOD";      pulse_t->SetBranchAddress( vname, &pulse_i[vname] );
	 vname = "MTQ_AMPL";        pulse_t->SetBranchAddress( vname, &pulse_f[vname] );
	 vname = "MTQ_TIME";        pulse_t->SetBranchAddress( vname, &pulse_f[vname] );
	 vname = "MTQ_RISE";        pulse_t->SetBranchAddress( vname, &pulse_f[vname] );
	 vname = "MTQ_FWHM";        pulse_t->SetBranchAddress( vname, &pulse_f[vname] );
	 vname = "MTQ_FW20";        pulse_t->SetBranchAddress( vname, &pulse_f[vname] );
	 vname = "MTQ_FW80";        pulse_t->SetBranchAddress( vname, &pulse_f[vname] );
	 vname = "MTQ_SLIDING";     pulse_t->SetBranchAddress( vname, &pulse_f[vname] );
	 pulse_t->LoadTree( 0 );
	 pulse_t->GetEntry( 0 );
	 assert(  logic_id == (int)  pulse_i["LOGIC_ID"] );
	 EcalLogicID ecid_pulse = econn->getEcalLogicID( ecid_lmfrun.getLogicID() );
	 // 	econn->fetchDataSet( &dataset_pulse, &lmfiov );
	 // 	n_ =  dataset_pulse.size();
    // 	if( n_==0 )
    // 	  {   
    // 	    // Set the data
    // 	    LMFLaserPulseDat::setColor( color ); // set the color
	 LMFLaserPulseDat lmf_pulse(econn);
	 lmf_pulse.setColor(color);
	 lmf_pulse.setLMFRunIOV(lmfiov);
	 lmf_pulse.setFitMethod( ecid_pulse, 0 );  
	 //	 lmf_pulse.setAmplitude( ecid_pulse, (float) pulse_f["MTQ_AMPL"] );
	 lmf_pulse.setMTQTime(      ecid_pulse, (float) pulse_f["MTQ_TIME"] );
	 lmf_pulse.setMTQRise(      ecid_pulse, (float) pulse_f["MTQ_RISE"] ); 
	 lmf_pulse.setMTQFWHM(      ecid_pulse, (float) pulse_f["MTQ_FWHM"] ); 
	 lmf_pulse.setMTQFW80(      ecid_pulse, (float) pulse_f["MTQ_FW80"] );
	 lmf_pulse.setMTQFW20(      ecid_pulse, (float) pulse_f["MTQ_FW20"] );
	 lmf_pulse.setMTQSliding(   ecid_pulse, (float) pulse_f["MTQ_SLIDING"] );
	 if (_insert) {
	   econn->insertLmfDat(&lmf_pulse);
	 }
         
         // Laser Primitives
         //
         table = MELaserPrim::lmfLaserName( ME::iLmfLaserPrim, type, color );
         map< TString, TH2* > h_;
         hname = "LOGIC_ID";          h_[hname] = (TH2*) f->Get(table+sep+hname);
         hname = "FLAG";              h_[hname] = (TH2*) f->Get(table+sep+hname);
         hname = "MEAN";              h_[hname] = (TH2*) f->Get(table+sep+hname);
         hname = "RMS";               h_[hname] = (TH2*) f->Get(table+sep+hname);
         hname = "M3";                h_[hname] = (TH2*) f->Get(table+sep+hname);
         hname = normStr_+"A_MEAN";   h_[hname] = (TH2*) f->Get(table+sep+hname);
         hname = normStr_+"A_RMS";    h_[hname] = (TH2*) f->Get(table+sep+hname);
         hname = normStr_+"A_M3";     h_[hname] = (TH2*) f->Get(table+sep+hname);
         hname = normStr_+"B_MEAN";   h_[hname] = (TH2*) f->Get(table+sep+hname);
         hname = normStr_+"B_RMS";    h_[hname] = (TH2*) f->Get(table+sep+hname);
         hname = normStr_+"B_M3";     h_[hname] = (TH2*) f->Get(table+sep+hname);
         hname = normStr_+"_MEAN";    h_[hname] = (TH2*) f->Get(table+sep+hname);
         hname = normStr_+"_RMS";     h_[hname] = (TH2*) f->Get(table+sep+hname);
         hname = normStr_+"_M3";      h_[hname] = (TH2*) f->Get(table+sep+hname);
         hname = "ALPHA";             h_[hname] = (TH2*) f->Get(table+sep+hname);
         hname = "BETA";              h_[hname] = (TH2*) f->Get(table+sep+hname);
         hname = "SHAPE_COR";         h_[hname] = (TH2*) f->Get(table+sep+hname);


         i_h = (TH2I*) h_["LOGIC_ID"];
         TAxis* ax = i_h->GetXaxis();
         TAxis* ay = i_h->GetYaxis();      
         int ixbmin = ax->GetFirst();
         int ixbmax = ax->GetLast();
         int iybmin = ay->GetFirst();
         int iybmax = ay->GetLast();
	 //     LMFLaserPrimDat::setColor( color ); // set the color
	 
         if( _debug>0 ) cout << "Filling laser primitive dataset" << endl;
	 
         if( _debug>1 ) 
           {
	     cout << "ixbmin/ixbmax " << ixbmin << "/" << ixbmax << endl;
	     cout << "iybmin/iybmax " << iybmin << "/" << iybmax << endl;
           }
	 
         // OK This is new...
         // Fetch an ordered list of crystal EcalLogicIds
         vector<EcalLogicID> channels_;
         TString view_;
         int id1_min =EcalLogicID::NULLID;
         int id2_min =EcalLogicID::NULLID;
         int id3_min =EcalLogicID::NULLID;
         int id1_max =EcalLogicID::NULLID;
         int id2_max =EcalLogicID::NULLID;
         int id3_max =EcalLogicID::NULLID;
         int sm_(0);
         if( isEB )
           {
     	view_ = "EB_crystal_number";
     	sm_ = MEEBGeom::smFromDcc( dcc );
     	id1_min = sm_;
     	id1_max = sm_;
     	id2_min = 1;
     	id2_max = 1700;
           }
         else
           {
     	view_ = "EE_crystal_number";
     	id1_min = 1;
     	if( ME::ecalRegion( ilmr )==ME::iEEM ) id1_min=-1;
     	id1_max =  id1_min;
     	id2_min =  0;
     	id2_max =  101;
     	id3_min =  0;
     	id3_max =  101;
           }
         channels_ = econn->getEcalLogicIDSet( view_.Data(),
     					  id1_min, id1_max,
     					  id2_min, id2_max,
     					  id3_min, id3_max,
     					  view_.Data() );      
	 LMFPrimDat laser_(econn, color, "LASER");
	 laser_.setLMFRunIOV(lmfiov);
	 laser_.dump();
         for( unsigned ii=0; ii<channels_.size(); ii++ )
           {
	     EcalLogicID ecid_prim = channels_[ii];
	     logic_id = ecid_prim.getLogicID();
	     id1_ = ecid_prim.getID1();
	     id2_ = ecid_prim.getID2();
	     id3_ = ecid_prim.getID3();
	     
	     int ix_(0);
	     int iy_(0);
	     if( isEB )
	       {
		 assert( id1_==sm_ );
		 MEEBGeom::XYCoord xy_ = MEEBGeom::localCoord( id2_ );
		 ix_ = xy_.first;
		 iy_ = xy_.second;
	       }
	     else
	       {
		 ix_ = id2_;
		 iy_ = id3_;
	       }
	     int ixb = ax->FindBin( ix_ );
	     int iyb = ay->FindBin( iy_ );	
	     int logic_id_  = (int) i_h->GetCellContent( ixb, iyb );
	     
	     if( logic_id_<=0 )
	       {
		 if( _debug>1 )
		   {
		     cout << "LogicID(" << view_ << "," << id1_ << "," << id2_ << "," << id3_ << ")=" << logic_id << " -->  no entry " << endl;
		   }
		 continue;
	       }
	     
	     if( _debug>2 )
	       {
		 //     sanity check
		 
		 table_id_ = 0;
		 
		 int jd1_(0);
		 int jd2_(0);
		 int jd3_(0);
		 MELaserPrim::getViewIds( logic_id_, table_id_, jd1_, jd2_ );
		 
		 int jx_ = (int)ax->GetBinCenter( ixb );
		 int jy_ = (int)ay->GetBinCenter( iyb );
		 //int jx_id_(0), jy_id_(0);
		 if( isEB ) 
		   {
		     if( table_id_!=MELaserPrim::iEB_crystal_number )
		       {
			 //		    cout << "warning -- inconsistent table_id [2] --> " << table_id_  << endl;
			 table_id_=MELaserPrim::iEB_crystal_number;
		       }
		     assert( jd1_>=1 && jd1_<=36 );
		     if( sm_!=jd1_ )
		       {
			 // 		    //		    cout << "Warning -- inconsistency in SM numbering " 
			 //			 << jd1_ << "/" << sm_ << " logic_id/id2=" << logic_id_ << "/" << jd2_ << endl;
		       }
		     assert( jd2_>=1 && jd2_<=1700 ); // !!!
		     // MEEBGeom::XYCoord xy_ = MEEBGeom::localCoord( jd2_ );
		     //jx_id_ = xy_.first;
		     //jy_id_ = xy_.second;
		     
		     jd1_ = sm_;
		     jd2_ = MEEBGeom::crystal_channel( jx_, jy_ );
		     jd3_ = EcalLogicID::NULLID;
		   }
		 else       
		   {
		     if( table_id_!=MELaserPrim::iEE_crystal_number )
		       {
			 //		    cout << "warning -- inconsistent table_id [3] --> " << table_id_  << endl;
			 table_id_=MELaserPrim::iEE_crystal_number;
		       }
		     //jx_id_ = jd1_;
		     //jy_id_ = jd2_;
		     
		     jd1_ = 1;
		     if( ME::ecalRegion( ilmr )==ME::iEEM ) jd1_=-1;
		     jd2_ = jx_;
		     jd3_ = jy_;
		   }
		 //  	    if( jx_!=jx_id_ || jy_!=jy_id_ )
		 //  	      {
		 //  		cout << "Warning inconsistency ix=" << jx_ << "/" << jx_id_ 
		 //  		     << " iy=" << jy_ << "/" << jy_id_ << endl; 
		 //  	      }
		 if( jd1_!=id1_ || jd2_!=id2_  || jd3_!=id3_  )
		   {
		     cout << "Warning inconsistency " 
			  << " -- jd1/id1 " << jd1_ << "/" << id1_ 
			  << " -- jd2/id2 " << jd2_ << "/" << id2_ 
			  << " -- jd3/id3 " << jd3_ << "/" << id3_ 
			  << endl;	      
		   }
	       }
	     if( _debug>1 )
	       cout << "LogicID(" << view_ << "," << id1_ << "," << id2_ << "," << id3_ << ")=" << logic_id << " --> val(ixb=" << ixb << ",iyb=" << iyb << ")=" <<  (float) h_["MEAN"] -> GetCellContent( ixb, iyb )  << endl; 
	      	// Set the data
	     cout << "Filling laser_ for " << ecid_prim.getLogicID() << endl;
	     laser_.setFlag( ecid_prim,  (int) h_["FLAG"] -> GetCellContent( ixb, iyb ) );
	     cout << "Filling laser_ for " << ecid_prim.getLogicID() << endl;
	     laser_.setMean( ecid_prim, (float) h_["MEAN"] -> GetCellContent( ixb, iyb ) );
	     cout << "Filling laser_ for " << ecid_prim.getLogicID() << endl;
	     laser_.setRMS(  ecid_prim, (float) h_["RMS"]  -> GetCellContent( ixb, iyb ) );
	     cout << "Filling laser_ for " << ecid_prim.getLogicID() << endl;
	     laser_.setM3( ecid_prim, (float) h_["M3"] -> GetCellContent( ixb, iyb ) );
	     cout << "Filling laser_ for " << ecid_prim.getLogicID() << endl;
	     laser_.setAPDoverAMean(  ecid_prim, (float) h_[normStr_+"A_MEAN"] -> GetCellContent( ixb, iyb ) );
	     cout << "Filling laser_ for " << ecid_prim.getLogicID() << endl;
	     laser_.setAPDoverARMS(   ecid_prim, (float) h_[normStr_+"A_RMS"]  -> GetCellContent( ixb, iyb ) );
	     cout << "Filling laser_ for " << ecid_prim.getLogicID() << endl;
	     laser_.setAPDoverAM3(  ecid_prim, (float) h_[normStr_+"A_M3"] -> GetCellContent( ixb, iyb ) );
	     cout << "Filling laser_ for " << ecid_prim.getLogicID() << endl;
	     laser_.setAPDoverBMean(  ecid_prim, (float) h_[normStr_+"B_MEAN"] -> GetCellContent( ixb, iyb ) );
	     cout << "Filling laser_ for " << ecid_prim.getLogicID() << endl;
	     laser_.setAPDoverBRMS(   ecid_prim, (float) h_[normStr_+"B_RMS"]  -> GetCellContent( ixb, iyb ) );
	     cout << "Filling laser_ for " << ecid_prim.getLogicID() << endl;
	     laser_.setAPDoverBM3(  ecid_prim, (float) h_[normStr_+"B_M3"] -> GetCellContent( ixb, iyb ) );	  
	     cout << "Filling laser_ for " << ecid_prim.getLogicID() << endl;
	     laser_.setAPDoverPnMean(   ecid_prim, (float) h_[normStr_+"_MEAN"]  -> GetCellContent( ixb, iyb ) );
	     cout << "Filling laser_ for " << ecid_prim.getLogicID() << endl;
	     laser_.setAPDoverPnRMS(    ecid_prim, (float) h_[normStr_+"_RMS"]   -> GetCellContent( ixb, iyb ) );
	     cout << "Filling laser_ for " << ecid_prim.getLogicID() << endl;
	     laser_.setAPDoverPnM3(   ecid_prim, (float) h_[normStr_+"_M3"]  -> GetCellContent( ixb, iyb ) );	 
	     cout << "Filling laser_ for " << ecid_prim.getLogicID() << endl;
	     laser_.setAlpha(           ecid_prim, (float) h_["ALPHA"] -> GetCellContent( ixb, iyb ) ); 
	     cout << "Filling laser_ for " << ecid_prim.getLogicID() << endl;
	     laser_.setBeta(            ecid_prim, (float) h_["BETA"]  -> GetCellContent( ixb, iyb )); 
	     cout << "Filling laser_ for " << ecid_prim.getLogicID() << endl;
	     //	     laser_.setShapeCorr(        ecid_prim, (float) h_["SHAPE_COR"] -> GetCellContent( ixb, iyb ) );
	     laser_.setShapeCorr(        ecid_prim, 0. );
	     cout << "---- Filling laser_ for " << ecid_prim.getLogicID() << endl;
	     // 	// Fill the dataset
    // 	dataset_prim[ecid_prim] = laser_;
           }  
         if( _debug ) cout << "done. " << endl;

         //
         // Laser PN Primitives
         //
         table = MELaserPrim::lmfLaserName( ME::iLmfLaserPnPrim, type, color );
         TTree* pn_t = (TTree*) f->Get(table);
         map< TString, unsigned int    > pn_i;
         map< TString, float > pn_f;
         vname = "LOGIC_ID";              pn_t->SetBranchAddress( vname, &pn_i[vname] );
         vname = "FLAG";                  pn_t->SetBranchAddress( vname, &pn_i[vname] );
         vname = "MEAN";                  pn_t->SetBranchAddress( vname, &pn_f[vname] );
         vname = "RMS";                   pn_t->SetBranchAddress( vname, &pn_f[vname] );
         vname = "M3";                    pn_t->SetBranchAddress( vname, &pn_f[vname] );
         vname = "PNA_OVER_PNB_MEAN";     pn_t->SetBranchAddress( vname, &pn_f[vname] );
         vname = "PNA_OVER_PNB_RMS";      pn_t->SetBranchAddress( vname, &pn_f[vname] );
         vname = "PNA_OVER_PNB_M3";       pn_t->SetBranchAddress( vname, &pn_f[vname] );
         Long64_t pn_n = pn_t->GetEntries();
	 //     LMFLaserPNPrimDat::setColor( color ); // set the color
	 LMFPnPrimDat pn_(econn, color, "LASER");
	 pn_.setLMFRunIOV(lmfiov);
         for( Long64_t jj=0; jj<pn_n; jj++ )
           {
	     pn_t->LoadTree( jj );
	     pn_t->GetEntry( jj );
	     logic_id =  (int)  pn_i["LOGIC_ID"];
	     
	     //	EcalLogicID ecid_pn = econn->getEcalLogicID( logic_id );
	     table_id_ = 0;
	     id1_      = 0;
	     id2_      = 0;
	     MELaserPrim::getViewIds( logic_id, table_id_, id1_, id2_ );
	     if( isEB )
	       {
		 if( table_id_!=MELaserPrim::iEB_LM_PN )
		   {
		     //		cout << "warning -- inconsistent table_id [4] --> " << table_id_  << endl;
		     table_id_=MELaserPrim::iEB_LM_PN;
		   }
	       }
	     else
	       {
		 if( table_id_!=MELaserPrim::iEE_LM_PN )
		   {
		     //		cout << "warning -- inconsistent table_id [4] --> " << MELaserPrim::channelViewName( table_id_ ) << endl;
		     table_id_=MELaserPrim::iEE_LM_PN;
		   }
	       }
	     
	     channelViewName_ = MELaserPrim::channelViewName( table_id_ );
	     EcalLogicID ecid_pn = econn->getEcalLogicID( channelViewName_.Data(), id1_, id2_ ); 
	     
	     if( _debug>1 )
	       cout << "LogicID(" << channelViewName_ << "," << id1_ << "," << id2_ << "," << id3_ << ")=" << ecid_pn.getLogicID() << " --> " <<  (float) pn_f["MEAN"] << endl; 
	     
	     // 	// Set the data
	     // 	LMFLaserPNPrimDat pn_;
	     pn_.setFlag(      ecid_pn,  (int)  pn_i["FLAG"]  );
	     pn_.setMean(           ecid_pn, (float) pn_f["MEAN"]  );
	     pn_.setRMS(            ecid_pn, (float) pn_f["RMS"]   );
	     pn_.setM3(           ecid_pn, (float) pn_f["M3"]  );
	     pn_.setPNAoverBMean( ecid_pn, (float) pn_f["PNA_OVER_PNB_MEAN"] );
	     pn_.setPNAoverBRMS(  ecid_pn, (float) pn_f["PNA_OVER_PNB_RMS "] );
	     pn_.setPNAoverBM3( ecid_pn, (float) pn_f["PNA_OVER_PNB_M3"] );
	     // 	// Fill the dataset
    // 	dataset_pnprim[ecid_pn] = pn_;
           }

    //     // Inserting the dataset, identified by iov
         if( _debug>0 )
           cout << "Inserting _PRIM_DAT and _PN_PRIM_DAT ..." << flush; 
         if( _insert )
           {
    // 	econn->insertDataSet( &dataset_prim,   &lmfiov );
    // 	econn->insertDataSet( &dataset_pnprim, &lmfiov );
	     laser_.debug();
	     econn->insertLmfDat(&laser_);
	     econn->insertLmfDat(&pn_);
           }
         else
           {       
     	if( _debug>0 ) 
     	  cout << ".... dummy.... " << flush;
           }
         cout << "Done inserting " 
     	 << fname 
     	 << "." << endl;

	 //close the root file
         f->Close();

  };
 
  void debug( int dblevel ) {  _debug=dblevel; }
  void insert( bool insert ) { _insert=insert; }
  
private:
  
  CondDBApp();  // hidden default constructor
  EcalCondDBInterface* econn;

  //   uint64_t startmicros;
  //   uint64_t endmicros;
  //   run_t startrun;
  //   run_t endrun;

  int _debug;
  bool _insert;

  RunIOV _runiov;
  RunTag _runtag;
  int _run;
  int _seq;
  int _type;
  int _color;
  time_t _t0;
  time_t _t1;

  LMFColor    _lmfcol;
  LMFRunTag   _lmftag;
  LMFTrigType _lmftrig;
  LMFSeqDat   _lmfseq;
  
  EcalLogicID ecid_allEcal;

  std::string _location;
  std::string _runtype;
  std::string _gentag;

  std::map< int, LMFRunIOV > _lmfRunIOV;

  bool existsLMFRunIOV( int ilmr )
  {
    if( _lmfRunIOV.count( ilmr )!=0 )
      {
	//	printLMFRunIOV( _lmfRunIOV[ilmr] ); 
	return true;
      }
    return false;
  }

  LMFSeqDat makeSequence( RunIOV* runiov, int iseq )
  {
    LMFSeqDat lmfseqdat;
    lmfseqdat.setRunIOV(*runiov);
    lmfseqdat.setSequenceNumber(iseq);
    return lmfseqdat;
  }

  LMFRunIOV makeLMFRunIOV( int subr, 
			   time_t t_beg, time_t t_end )
  {
    // trick to get the correct Tm objects (from Francesca Cavalleri)
    //  t_beg and t_end are in number of seconds since the Unix epoch
    //        transforms them first in nanoseconds (uint64_int)

    uint64_t startMuS = (uint64_t) t_beg*1000000;
    Tm startTm( startMuS ); 

    uint64_t endMuS = (uint64_t) t_end*1000000;
    Tm   endTm( endMuS );

    LMFRunTag lmftag;
    LMFRunIOV lmfiov;    
    //     lmfiov.setLMFRunTag(lmftag);
    //     lmfiov.setRunIOV(_runiov);
    //     lmfiov.setSubRunNumber(subr);
    //     lmfiov.setSubRunType("Standard");
    //     lmfiov.setSubRunStart( startTm );
    //     lmfiov.setSubRunEnd( endTm );

    //     printLMFRunIOV( lmfiov );

    //     _lmfRunIOV[subr]=lmfiov;

    return lmfiov;
  }

  void printLMFSeqDat( LMFSeqDat& seq ) const
  {
    RunIOV runiov  = seq.getRunIOV();
    //     int subr = iov.getSubRunNumber();
    Tm tm_start( seq.getSequenceStart().microsTime() );
    Tm tm_end(   seq.getSequenceStop().microsTime() );
    cout 
      << "> LMFSequence("
      << runiov.getRunNumber() << "/"
      << seq.getSequenceNumber() << ") ";    
    cout	  
      << tm_start.str() << " --> " 	  
      << tm_end.str()
      << endl;
  }

  void printLMFRunIOV( LMFRunIOV& iov ) const
  {
    LMFSeqDat seq_ = iov.getSequence();
    RunIOV runiov  = seq_.getRunIOV();
    //     int subr = iov.getSubRunNumber();
    LMFTrigType trig_ = iov.getTriggerType();
    LMFColor    col_  = iov.getColor();
    Tm tm_start( iov.getSubRunStart().microsTime() );
    Tm tm_end(   iov.getSubRunEnd().microsTime() );
    int ilmr = iov.getLmr();
    assert( ilmr>0 && ilmr<=92 );
    //     int iseq, ilmr, type, color;
    //     decodeSubRunNumber( subr, iseq, ilmr, type, color );
    cout 
      << ">> LMFRun("
      << runiov.getRunNumber() << "/"
      << seq_.getSequenceNumber();
    cout << "/LMR";
    if( ilmr<10 ) cout << "0";
    cout << ilmr 
	 << "[" << ME::smName(ilmr) << "]) ";
    cout << trig_.getShortName() << "/" << col_.getShortName() << " ";
    cout	  
      << tm_start.str() << " --> " 	  
      << tm_end.str()
      << endl;
    //iov.dump();
  }
  
  void printRunIOV( RunIOV& iov ) const
  {
    RunTag tag = iov.getRunTag();
    Tm tm_start( iov.getRunStart().microsTime() );
    Tm tm_end(   iov.getRunEnd().microsTime() );
    cout << "RunIOV("
	 << iov.getRunNumber() << "/"
	 << tag.getGeneralTag() << "/"
	 << tag.getLocationDef().getLocation() << "/"
	 << tag.getRunTypeDef().getRunType() << ") "
	 << tm_start.str() << " --> " 
	 << tm_end.str() 
	 << endl;
  }

  int subRunNumber( int iseq, int ilmr )
  {
    return 1000000*iseq + 100*ilmr + 10*_type + _color;  
  }

  void decodeSubRunNumber( int subr, int& iseq, int& ilmr, int& type, int& color ) const
  {
    int subr_ = subr;
    iseq = subr_/1000000;
    subr_ -= 1000000*iseq;
    ilmr = subr_/100;
    subr_ -= 100*ilmr;
    type = subr_/10;
    subr_ -= 10*type;
    color = subr_;
  }

};


int main (int argc, char* argv[])
{
  string sid;
  string user;
  string pass;
  // sqlplus cms_ecal_cond/XXXXX_XXXX@cms_omds_lb

  //GHM sid  ="cms_omds_lb"; 
  //GHM user ="cms_ecal_cond"; 

  sid  ="cmsdevr_lb"; 
  user ="cms_ecal_cond"; 
  pass ="NONE";

  int type  = ME::iLaser; 
  int color = ME::iBlue; 

  int run   = 129909;
  int lb    = 1;
  int seq   = -1;

  int debug = 0;
  bool insert = true;

  TString path   = "/nfshome0/ecallaser/LaserPrim";
  TString period = "CRAFT1";
  
  TString seqstr;

  time_t trun(0);
  time_t t0(0), t1(0);
  int tseq(0);
  int dt(0);
  int lmr(0);

  char c;
  while ( (c = getopt( argc, argv, "t:c:R:B:s:S:T:D:L:D:d:p:P:w:n" ) ) != EOF ) 
    {
      switch (c) 
	{
	case 't': type     = atoi( optarg );    break;
	case 'c': color    = atoi( optarg );    break;
	case 'R': run      = atoi( optarg );    break;
	case 's': seq      = atoi( optarg );    break;
	case 'T': trun     = atol( optarg );    break;
	case 'D': tseq     = atoi( optarg );    break;
	case 'S': seqstr   = TString( optarg );   break;
	case 'd': debug    = atoi( optarg );      break;
	case 'P': period   = TString( optarg );   break;
	case 'p': path     = TString( optarg );   break;
	case 'w': pass     = string( optarg );    break;
	case 'n': insert   = false;    break;
	}
    }

  if( pass=="NONE" )
    {
      cout << "--- A DB password (writer) must be provided !" << endl;
      cout << "Usage: LaserSeqToDB -p <password>" << endl;
      return -1;
    }

  TString mestore = path;
  mestore += "/";
  mestore += period;
  setenv("MESTORE",mestore.Data(),1);

  if( seqstr=="" ) return -1;
  
  try 
    {
      CondDBApp app( sid, user, pass );
      app.debug( debug );
      app.insert( insert );

      vector< int > lmrv;
      vector< int >  dtv;

      TObjArray* array_ = seqstr.Tokenize("-");
      TObjString* token_(0);
      TString str_;
      int nTokens_= array_->GetEntries();
      if( nTokens_==0 ) return -1;
      for( int iToken=0; iToken<nTokens_; iToken++ )
	{
	  token_ = (TObjString*)array_->operator[](iToken);
	  str_ = token_->GetString();
	  TObjArray* dum_ = str_.Tokenize("_");
	  dt = ((TObjString*)dum_->operator[](1))->GetString().Atoi();
	  TString lmr_ = ((TObjString*)dum_->operator[](0))->GetString();
	  lmr_.ReplaceAll("LM","00");
	  lmr_.Remove(TString::kLeading,'0');
	  lmr = lmr_.Atoi();

	  //	  cout << iToken << "->" << str_ << " lmr=" << lmr << " t=" << dt << endl;

	  lmrv.push_back( lmr );
	  dtv.push_back( dt );
	}

      t0 = trun+tseq;
      t1 = t0 + dtv.back() + 5;

      app.init( run, seq, type, color, t0, t1 );

      size_t lmrv_size = lmrv.size();
      // size_t lmrv_size = 0;

      for( unsigned int ii=0; ii<lmrv_size; ii++ )
	{
	  bool ok=false;
	  
	  lmr = lmrv[ii];
	  dt  = dtv[ii];

	  int reg_(0);
	  int dcc_(0);
	  int sect_(0);
	  int side_(0);
	  ME::regionAndSector( lmr, reg_, sect_, dcc_, side_ );
	  
	  TString runlistfile = ME::runListName( lmr, type, color );
	  FILE *test;
	  test = fopen( runlistfile, "r" );
	  if(test)
	    {
	      fclose( test );
	    }
	  else
	    {
	      if( debug>0 )
		cout << "File " << runlistfile << " not found." << endl;
	      return(-1);
	    }
	  ifstream fin;
	  fin.open(runlistfile);
	  
	  //	  cout << "GHM-- runListFile " << runlistfile << endl;
	  
	  while( fin.peek() != EOF )
	    {
	      string rundir;
	      long long tsb, tse;
	      int rr, bb, mpga , mem, pp, ff, dd, evts;      
	      fin >> rundir;
	      fin >> rr >> bb >> evts >> tsb >> tse >> mpga >> mem >> pp >> ff >> dd;      
	      
	      if( rr!=run ) continue;

	      time_t t_beg = ME::time_high( tsb );
	      time_t t_end = ME::time_high( tse );

	      //	      cout << "run " << run << " tbeg/tend " << t_beg << "/" << t_end << endl;

	      int dt0_end = t_end-t0;
	      int dt1_end = t_end-t1;
	      int dt0_beg = t_beg-t0;
	      int dt1_beg = t_beg-t1;
	      if( dt0_beg*dt1_beg>0  && dt0_end*dt1_end>0  )  continue;

	      ok = true;

	      //	      if( std::abs(bb-lb_)>3 ) continue;
	      
	      ME::Header header;
	      header.rundir = rundir;
	      header.dcc    = dcc_;
	      header.side   = side_;
	      header.run=rr;
	      header.lb=lb;
	      header.ts_beg=tsb;
	      header.ts_end=tse;
	      header.events=evts;
	      
	      ME::Settings settings;    
	      settings.type = type;
	      settings.wavelength = color;
	      settings.mgpagain=mpga;
	      settings.memgain=mem;
	      settings.power=pp;
	      settings.filter=ff;
	      settings.delay=dd;
	      
	      TString fname = ME::rootFileName( header, settings );
	      FILE *test;
	      test = fopen( fname, "r" );
	      if(test)
		{
		  fclose( test );
		}
	      else
		{
		  continue;
		}
	      cout << "STORING LASER PRIMITIVES" << endl;
	      app.storeLaserPrimitive( fname, header, settings, dt );
	      cout << "LASER PRIMITIVES STORED" << endl;
	      break;
	    }
	  if( !ok )
	    {
	      cout << "Warning -- " << run << "/" << seq 
		   << " no primitive file found for lmr/dt " 
		   << lmr << "[" << ME::smName(lmr) << "]/" << dt << endl;
	    }
	}
    } catch (exception &e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }
  return 0;
}

