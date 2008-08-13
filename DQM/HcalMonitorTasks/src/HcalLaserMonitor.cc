#include "DQM/HcalMonitorTasks/interface/HcalLaserMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
HcalLaserMonitor::HcalLaserMonitor() {}
HcalLaserMonitor::~HcalLaserMonitor() {}
void HcalLaserMonitor::reset() {}
void HcalLaserMonitor::done() {}

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
void HcalLaserMonitor::setup( const edm::ParameterSet& iConfig, DQMStore* dbe ) {
  if( fVerbosity ) printf( "-=-=-=-=-=HcalLaserMonitor Setup=-=-=-=-=-\n" );

  HcalBaseMonitor::setup( iConfig, dbe );
  baseFolder_ = rootFolder_ + "LaserMonitor";

  printf( "====================================================\n" );

  doPerChannel_ = iConfig.getUntrackedParameter<bool>( "LaserPerChannel", false );
  printf( "Laser Monitor per channel set to %s\n", doPerChannel_ ? "true" : "false" );

  etaMax_ = iConfig.getUntrackedParameter<double>( "MaxEta",  41.5 );
  etaMin_ = iConfig.getUntrackedParameter<double>( "MinEta", -41.5 );
  etaBins_ = (int)(etaMax_ - etaMin_);
  cout << "Laser Monitor eta min/max set to " << etaMin_ << "/" << etaMax_ << endl;

  phiMax_ = iConfig.getUntrackedParameter<double>( "MaxPhi", 73 );
  phiMin_ = iConfig.getUntrackedParameter<double>( "MinPhi",  0 );
  phiBins_ = (int)(phiMax_ - phiMin_);
  cout << "Laser Monitor phi min/max set to " << phiMin_ << "/" << phiMax_ << endl;

  sigS0_ = iConfig.getUntrackedParameter<int>( "FirstSignalBin", 0 );
  sigS1_ = iConfig.getUntrackedParameter<int>( "LastSignalBin",  9 );
  if( sigS0_ < 0 ) {
    printf( "HcalLaserMonitor::setup, illegal range for first sample: %d\n", sigS0_ );
    sigS0_ = 0;
  }
  if( sigS1_ > 9 ) {
    printf( "HcalLaserMonitor::setup, illegal range for last sample: %d\n", sigS1_ );
    sigS1_ = 9;
  }
  if( sigS0_ > sigS1_ ) {
    printf( "HcalLaserMonitor::setup, illegal range for first/last sample: %d/%d\n", sigS0_, sigS1_ );
    sigS0_ = 0; sigS1_ = 9;
  }
  cout << "Laser Monitor signal window set to " << sigS0_ << "-" << sigS1_ << endl;

  adcThresh_ = iConfig.getUntrackedParameter<double>( "Laser_ADC_Thresh", -10000 );
  cout << "Laser Monitor threshold set to " << adcThresh_ << endl;

  printf( "====================================================\n" );

  ievt_ = 0;

  if( m_dbe ) {
    m_dbe->setCurrentFolder( baseFolder_ );
    meEVT_ = m_dbe->bookInt( "Laser Task Event Number" );
    meEVT_->Fill( ievt_ );

    MEAN_MAP_TIME_L1_ = m_dbe->book2D( "Laser Mean Time Depth 1", "Laser Mean Time Depth 1", etaBins_, etaMin_, etaMax_, phiBins_, phiMin_, phiMax_ );
    MEAN_MAP_TIME_L2_ = m_dbe->book2D( "Laser Mean Time Depth 2", "Laser Mean Time Depth 2", etaBins_, etaMin_, etaMax_, phiBins_, phiMin_, phiMax_ );
    MEAN_MAP_TIME_L3_ = m_dbe->book2D( "Laser Mean Time Depth 3", "Laser Mean Time Depth 3", etaBins_, etaMin_, etaMax_, phiBins_, phiMin_, phiMax_ );
    MEAN_MAP_TIME_L4_ = m_dbe->book2D( "Laser Mean Time Depth 4", "Laser Mean Time Depth 4", etaBins_, etaMin_, etaMax_, phiBins_, phiMin_, phiMax_ );
    RMS_MAP_TIME_L1_ = m_dbe->book2D( "Laser RMS Time Depth 1", "Laser RMS Time Depth 1", etaBins_, etaMin_, etaMax_, phiBins_, phiMin_, phiMax_ );
    RMS_MAP_TIME_L2_ = m_dbe->book2D( "Laser RMS Time Depth 2", "Laser RMS Time Depth 2", etaBins_, etaMin_, etaMax_, phiBins_, phiMin_, phiMax_ );    
    RMS_MAP_TIME_L3_ = m_dbe->book2D( "Laser RMS Time Depth 3", "Laser RMS Time Depth 3", etaBins_, etaMin_, etaMax_, phiBins_, phiMin_, phiMax_ );
    RMS_MAP_TIME_L4_ = m_dbe->book2D( "Laser RMS Time Depth 4", "Laser RMS Time Depth 4", etaBins_, etaMin_, etaMax_, phiBins_, phiMin_, phiMax_ );

    MEAN_MAP_ENERGY_L1_ = m_dbe->book2D( "Laser Mean Energy Depth 1", "Laser Mean Energy Depth 1", etaBins_, etaMin_, etaMax_, phiBins_, phiMin_, phiMax_ );
    MEAN_MAP_ENERGY_L2_ = m_dbe->book2D( "Laser Mean Energy Depth 2", "Laser Mean Energy Depth 2", etaBins_, etaMin_, etaMax_, phiBins_, phiMin_, phiMax_ );
    MEAN_MAP_ENERGY_L3_ = m_dbe->book2D( "Laser Mean Energy Depth 3", "Laser Mean Energy Depth 3", etaBins_, etaMin_, etaMax_, phiBins_, phiMin_, phiMax_ );
    MEAN_MAP_ENERGY_L4_ = m_dbe->book2D( "Laser Mean Energy Depth 4", "Laser Mean Energy Depth 4", etaBins_, etaMin_, etaMax_, phiBins_, phiMin_, phiMax_ );
    RMS_MAP_ENERGY_L1_ = m_dbe->book2D( "Laser RMS Energy Depth 1", "Laser RMS Energy Depth 1", etaBins_, etaMin_, etaMax_, phiBins_, phiMin_, phiMax_ );
    RMS_MAP_ENERGY_L2_ = m_dbe->book2D( "Laser RMS Energy Depth 2", "Laser RMS Energy Depth 2", etaBins_, etaMin_, etaMax_, phiBins_, phiMin_, phiMax_ );
    RMS_MAP_ENERGY_L3_ = m_dbe->book2D( "Laser RMS Energy Depth 3", "Laser RMS Energy Depth 3", etaBins_, etaMin_, etaMax_, phiBins_, phiMin_, phiMax_ );
    RMS_MAP_ENERGY_L4_ = m_dbe->book2D( "Laser RMS Energy Depth 4", "Laser RMS Energy Depth 4", etaBins_, etaMin_, etaMax_, phiBins_, phiMin_, phiMax_ );

    m_dbe->setCurrentFolder( baseFolder_ + "/2DShape" );
    MEAN_MAP_SHAPE_L1_ = m_dbe->book2D( "Laser Mean Shape Depth 1", "Laser Mean Shape Depth 1", etaBins_, etaMin_, etaMax_, phiBins_, phiMin_, phiMax_ );
    MEAN_MAP_SHAPE_L2_ = m_dbe->book2D( "Laser Mean Shape Depth 2", "Laser Mean Shape Depth 2", etaBins_, etaMin_, etaMax_, phiBins_, phiMin_, phiMax_ );
    MEAN_MAP_SHAPE_L3_ = m_dbe->book2D( "Laser Mean Shape Depth 3", "Laser Mean Shape Depth 3", etaBins_, etaMin_, etaMax_, phiBins_, phiMin_, phiMax_ );
    MEAN_MAP_SHAPE_L4_ = m_dbe->book2D( "Laser Mean Shape Depth 4", "Laser Mean Shape Depth 4", etaBins_, etaMin_, etaMax_, phiBins_, phiMin_, phiMax_ );
    RMS_MAP_SHAPE_L1_ = m_dbe->book2D( "Laser RMS Shape Depth 1", "Laser RMS Shape Depth 1", etaBins_, etaMin_, etaMax_, phiBins_, phiMin_, phiMax_ );    
    RMS_MAP_SHAPE_L2_ = m_dbe->book2D( "Laser RMS Shape Depth 2", "Laser RMS Shape Depth 2", etaBins_, etaMin_, etaMax_, phiBins_, phiMin_, phiMax_ );
    RMS_MAP_SHAPE_L3_ = m_dbe->book2D( "Laser RMS Shape Depth 3", "Laser RMS Shape Depth 3", etaBins_, etaMin_, etaMax_, phiBins_, phiMin_, phiMax_ );
    RMS_MAP_SHAPE_L4_ = m_dbe->book2D( "Laser RMS Shape Depth 4", "Laser RMS Shape Depth 4", etaBins_, etaMin_, etaMax_, phiBins_, phiMin_, phiMax_ );

    m_dbe->setCurrentFolder( baseFolder_ + "/HB" );
    hbHists.allShapePedSub_ = m_dbe->book1D( "HB Ped Subtracted Pulse Shape", "HB Ped Subtracted Pulse Shape", 10, -0.5, 9.5 );
    hbHists.allShape_ = m_dbe->book1D( "HB Average Pulse Shape", "HB Average Pulse Shape", 10, -0.5, 9.5 );
    hbHists.rms_shape_ = m_dbe->book1D( "HB Laser Shape RMS Values", "HB Laser Shape RMS Values", 25, 0, 3 );
    hbHists.mean_shape_ = m_dbe->book1D( "HB Laser Shape Mean Values", "HB Laser Shape Mean Values", 100, -0.5, 9.5 );

    hbHists.allTime_ = m_dbe->book1D( "HB Average Pulse Time", "HB Average Pulse Time", 100, -0.5, 9.5 );
    hbHists.rms_time_ = m_dbe->book1D( "HB Laser Time RMS Values", "HB Laser Time RMS Values", 25, 0, 3 );
    hbHists.mean_time_ = m_dbe->book1D( "HB Laser Time Mean Values", "HB Laser Time Mean Values", 100, -0.5, 9.5 );

    hbHists.allEnergy_ = m_dbe->book1D( "HB Average Pulse Energy", "HB Average Pulse Energy", 1000, 0, 10000 );
    hbHists.rms_energy_ = m_dbe->book1D( "HB Laser Energy RMS Values", "HB Laser Energy RMS Values", 25, 0, 100 );
    hbHists.mean_energy_ = m_dbe->book1D( "HB Laser Energy Mean Values", "HB Laser Energy Mean Values", 100, 0, 5000 );

    m_dbe->setCurrentFolder( baseFolder_+ "/HE" );
    heHists.allShapePedSub_ = m_dbe->book1D( "HE Ped Subtracted Pulse Shape", "HE Ped Subtracted Pulse Shape", 10, -0.5, 9.5 );
    heHists.allShape_ = m_dbe->book1D( "HE Average Pulse Shape", "HE Average Pulse Shape", 10, -0.5, 9.5 );
    heHists.rms_shape_ = m_dbe->book1D( "HE Laser Shape RMS Values", "HE Laser Shape RMS Values", 25, 0, 3 );
    heHists.mean_shape_ = m_dbe->book1D( "HE Laser Shape Mean Values", "HE Laser Shape Mean Values", 100, -0.5, 9.5 );

    heHists.allTime_ = m_dbe->book1D( "HE Average Pulse Time", "HE Average Pulse Time", 100, -0.5, 9.5 );
    heHists.rms_time_ = m_dbe->book1D( "HE Laser Time RMS Values", "HE Laser Time RMS Values", 25, 0, 3 );
    heHists.mean_time_ = m_dbe->book1D( "HE Laser Time Mean Values", "HE Laser Time Mean Values", 100, -0.5, 9.5 );

    heHists.allEnergy_ = m_dbe->book1D( "HE Average Pulse Energy", "HE Average Pulse Energy", 1000, 0, 10000 );
    heHists.rms_energy_ = m_dbe->book1D( "HE Laser Energy RMS Values", "HE Laser Energy RMS Values", 25, 0, 100 );
    heHists.mean_energy_ = m_dbe->book1D( "HE Laser Energy Mean Values", "HE Laser Energy Mean Values", 100, 0, 5000 );

    m_dbe->setCurrentFolder( baseFolder_+ "/HO" );
    hoHists.allShapePedSub_ = m_dbe->book1D( "HO Ped Subtracted Pulse Shape", "HO Ped Subtracted Pulse Shape", 10, -0.5, 9.5 );
    hoHists.allShape_ = m_dbe->book1D( "HO Average Pulse Shape", "HO Average Pulse Shape", 10, -0.5, 9.5 );
    hoHists.rms_shape_ = m_dbe->book1D( "HO Laser Shape RMS Values", "HO Laser Shape RMS Values", 25, 0, 3 );
    hoHists.mean_shape_ = m_dbe->book1D( "HO Laser Shape Mean Values", "HO Laser Shape Mean Values", 100, -0.5, 9.5 );

    hoHists.allTime_ = m_dbe->book1D( "HO Average Pulse Time", "HO Average Pulse Time", 100, -0.5, 9.5 );
    hoHists.rms_time_ = m_dbe->book1D( "HO Laser Time RMS Values", "HO Laser Time RMS Values", 25, 0, 3 );
    hoHists.mean_time_ = m_dbe->book1D( "HO Laser Time Mean Values", "HO Laser Time Mean Values", 100, -0.5, 9.5 );

    hoHists.allEnergy_ = m_dbe->book1D( "HO Average Pulse Energy", "HO Average Pulse Energy", 1000, 0, 10000 );
    hoHists.rms_energy_ = m_dbe->book1D( "HO Laser Energy RMS Values", "HO Laser Energy RMS Values", 25, 0, 100 );
    hoHists.mean_energy_ = m_dbe->book1D( "HO Laser Energy Mean Values", "HO Laser Energy Mean Values", 100, 0, 5000 );

    m_dbe->setCurrentFolder( baseFolder_ + "/HF" );
    hfHists.allShapePedSub_ = m_dbe->book1D( "HF Ped Subtracted Pulse Shape", "HF Ped Subtracted Pulse Shape", 10, -0.5, 9.5 );
    hfHists.allShape_ = m_dbe->book1D( "HF Average Pulse Shape", "HF Average Pulse Shape", 10, -0.5, 9.5 );
    hfHists.rms_shape_ = m_dbe->book1D( "HF Laser Shape RMS Values", "HF Laser Shape RMS Values", 25, 0, 3 );
    hfHists.mean_shape_ = m_dbe->book1D( "HF Laser Shape Mean Values", "HF Laser Shape Mean Values", 100, -0.5, 9.5 );

    hfHists.allTime_ = m_dbe->book1D( "HF Average Pulse Time", "HF Average Pulse Time", 100, -0.5, 9.5 );
    hfHists.rms_time_ = m_dbe->book1D( "HF Laser Time RMS Values", "HF Laser Time RMS Values", 25, 0, 3 );
    hfHists.mean_time_ = m_dbe->book1D( "HF Laser Time Mean Values", "HF Laser Time Mean Values", 100, -0.5, 9.5 );

    hfHists.allEnergy_ = m_dbe->book1D( "HF Average Pulse Energy", "HF Average Pulse Energy", 1000, 0, 10000 );
    hfHists.rms_energy_ = m_dbe->book1D( "HF Laser Energy RMS Values", "HF Laser Energy RMS Values", 25, 0, 100 );
    hfHists.mean_energy_ = m_dbe->book1D( "HF Laser Energy Mean Values", "HF Laser Energy Mean Values", 100, 0, 5000 );

    m_dbe->setCurrentFolder( baseFolder_ + "/QADCTDC" );
    TDCHists.numChannels_ = m_dbe->book1D( "TDC Number of Channels", "TDC Number of Channels", 4, -0.5, 3.5 );

    TDCHists.trigger_ = m_dbe->book1D( "TDC Trigger", "TDC Trigger", 25, 12700, 12800 );
    TDCHists.clockOptosync_ = m_dbe->book1D( "TDC Clock Optosync", "TDC Clock Optosync", 50, 13900, 14100 );
    TDCHists.rawOptosync_ = m_dbe->book1D( "TDC Raw Optosync", "TDC Raw Optosync", 50, 13800, 14000 );

    TDCHists.rawOptosync_Trigger_ = m_dbe->book1D( "TDC Raw Optosync Minus Trigger", "TDC Raw Optosync Minus Trigger", 50, 1050, 1250 );

    char temp[128];
    for( int i = 0; i < 32; i++ ) {
      sprintf( temp,"QDC %02d", i );
      QADC_[i] = m_dbe->book1D( temp, temp, 41, 0, 4100 );
    }
  }
}

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
void HcalLaserMonitor::processEvent( const HBHEDigiCollection& hbhe, const HODigiCollection& ho, const HFDigiCollection& hf, const HcalLaserDigi& laserDigi, const HcalDbService& cond ) {
  if( fVerbosity ) printf( "-=-=-=-=-=HcalLaserMonitor processEvent=-=-=-=-=-\n" );

  ievt_++;
  meEVT_->Fill( ievt_ );

  if( !m_dbe ) {
    if( fVerbosity ) printf( "HcalLaserMonitor::processEvent - DQMStore not instantiated!\n" );
    return; 
  }
  float pedSubTS[10];

  // TDC
  try {
    for( int i = 0; i < 32; i++ ) {
      QADC_[i]->Fill( laserDigi.qadc(i) );
    }
    
    TDCHists.numChannels_->Fill( laserDigi.tdcHits() );
    double tClockOpto = -1, tTrig = -1, tRawOpto = -1;
    
    for( uint i = 0; i < laserDigi.tdcHits(); ++i ) {
      if( laserDigi.hitChannel(i) == 1 && tClockOpto < 0 ) {
	tClockOpto = laserDigi.hitNS(i);
	TDCHists.clockOptosync_->Fill( tClockOpto );
      }
      if( laserDigi.hitChannel(i) == 2 && tTrig < 0 ) {
	tTrig = laserDigi.hitNS(i);
	TDCHists.trigger_->Fill( tTrig );
      }
      if( laserDigi.hitChannel(i) == 3 && tRawOpto < 0 ) {
	tRawOpto = laserDigi.hitNS(i);
	TDCHists.rawOptosync_->Fill( tRawOpto );
      }
    }
    
    if( tRawOpto > 0 && tTrig > 0 ) TDCHists.rawOptosync_Trigger_->Fill( tRawOpto - tTrig );
  } catch (...) {
    if( fVerbosity ) printf( "HcalLaserMonitor::processEvent - No Laser Digis.\n" );
  }
  
  // HBHE
  try {
    for( HBHEDigiCollection::const_iterator j = hbhe.begin(); j != hbhe.end(); j++ ) {
      const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
      
      calibs_= cond.getHcalCalibrations( digi.id() );
      float en = 0, ts = 0, bs = 0, maxa = 0;
      int maxi = 0;

      for( int i = sigS0_; i <= sigS1_; i++ ) {
	if( digi.sample(i).adc() > maxa ) { maxa = digi.sample(i).adc(); maxi = i; }
      }
      for( int i = sigS0_; i <= sigS1_; i++ ) {
        int adc = digi.sample(i).adc();
        float fc = adc2fc_[adc] + 0.5;
	en += fc - calibs_.pedestal( digi.sample(i).capid() );
	if( i >= (maxi - 1) && i <= (maxi + 1) ) {
	  ts += i * ( fc - calibs_.pedestal( digi.sample(i).capid() ) );
	  bs += fc - calibs_.pedestal( digi.sample(i).capid() );
	}
      }

      if( en > adcThresh_ ) {
	if( digi.id().subdet() == HcalBarrel ) {
	  hbHists.allEnergy_->Fill(en);
	  if( bs != 0 ) hbHists.allTime_->Fill( ts / bs );
	  else if( fVerbosity ) printf( "HcalLaserMonitor::processEvent - Calculation of HB hit time had a zero denominator!\n" );

	  for( int i = 0; i < digi.size(); i++ ) {
	    int adc = digi.sample(i).adc();
	    float fc = adc2fc_[adc] + 0.5;
	    hbHists.allShape_->Fill( i, fc );
	    hbHists.allShapePedSub_->Fill( i, fc - calibs_.pedestal( digi.sample(i).capid() ) );
	    pedSubTS[i] = fc - calibs_.pedestal( digi.sample(i).capid() );
	  }

	  if( doPerChannel_ ) perChanHists( HcalBarrel, digi.id(), pedSubTS, hbHists.perChanShape_, hbHists.perChanTime_, hbHists.perChanEnergy_, baseFolder_ );
	}
	else if( digi.id().subdet() == HcalEndcap ) {
	  heHists.allEnergy_->Fill(en);
	  if( bs != 0 ) heHists.allTime_->Fill( ts / bs );
	  else if( fVerbosity ) printf( "HcalLaserMonitor::processEvent - Calculation of HE hit time had a zero denominator!\n" );

	  for( int i = 0; i < digi.size(); i++ ) {
	    int adc = digi.sample(i).adc();
	    float fc = adc2fc_[adc] + 0.5;
	    heHists.allShape_->Fill( i, fc );
	    heHists.allShapePedSub_->Fill( i, fc - calibs_.pedestal( digi.sample(i).capid() ) );
	    pedSubTS[i] = fc - calibs_.pedestal( digi.sample(i).capid() );
	  }

	  if( doPerChannel_ ) perChanHists( HcalEndcap, digi.id(), pedSubTS, heHists.perChanShape_, heHists.perChanTime_, heHists.perChanEnergy_, baseFolder_ );
	}
      }
    }
  } catch (...) {
    if( fVerbosity ) printf( "HcalLaserMonitor::processEvent - No HBHE Digis.\n" );
  }

  // HO
  try {
    for( HODigiCollection::const_iterator j = ho.begin(); j != ho.end(); j++ ) {
      const HODataFrame digi = (const HODataFrame)(*j);

      calibs_ = cond.getHcalCalibrations( digi.id() );
      float en = 0, ts = 0, bs = 0, maxa = 0;
      int maxi = 0;

      for( int i = sigS0_; i <= sigS1_; i++ ) {
	if( digi.sample(i).adc() > maxa ) { maxa = digi.sample(i).adc(); maxi = i; }
      }
      for( int i = sigS0_; i <= sigS1_; i++ ) {
        int adc = digi.sample(i).adc();
        float fc = adc2fc_[adc] + 0.5;
	en += fc - calibs_.pedestal( digi.sample(i).capid() );
	if( i >= (maxi - 1) && i <= (maxi + 1) ) {
	  ts += i * ( fc - calibs_.pedestal( digi.sample(i).capid() ) );
	  bs += fc - calibs_.pedestal( digi.sample(i).capid() );
	}
      }

      if( en > adcThresh_ ) {
	hoHists.allEnergy_->Fill(en);
	if( bs != 0 ) hoHists.allTime_->Fill( ts / bs );
	else if( fVerbosity ) printf( "HcalLaserMonitor::processEvent - Calculation of HO hit time had a zero denominator!\n" );

	for( int i = 0; i < digi.size(); i++ ) {
	  int adc = digi.sample(i).adc();
	  float fc = adc2fc_[adc] + 0.5;
	  hoHists.allShape_->Fill( i, fc );
	  hoHists.allShapePedSub_->Fill( i, fc - calibs_.pedestal( digi.sample(i).capid() ) );
	  pedSubTS[i] = fc - calibs_.pedestal( digi.sample(i).capid() );
	}
      }

      if( doPerChannel_ ) perChanHists( HcalOuter, digi.id(), pedSubTS, hoHists.perChanShape_, hoHists.perChanTime_, hoHists.perChanEnergy_, baseFolder_ );
    }
  } catch (...) {
    if( fVerbosity ) cout << "HcalLaserMonitor::processEvent - No HO Digis." << endl;
  }

  // HF
  try {
    for( HFDigiCollection::const_iterator j = hf.begin(); j != hf.end(); j++ ) {
      const HFDataFrame digi = (const HFDataFrame)(*j);

      calibs_ = cond.getHcalCalibrations( digi.id() );
      float en = 0, ts = 0, bs = 0, maxa = 0;
      int maxi = 0;

      for( int i = sigS0_; i <= sigS1_; i++ ) {
	if( digi.sample( i).adc() > maxa ) { maxa = digi.sample(i).adc(); maxi = i; }
      }
      for( int i = sigS0_; i <= sigS1_; i++ ) {
        int adc = digi.sample(i).adc();
        float fc = adc2fc_[adc] + 0.5;
	en += fc - calibs_.pedestal( digi.sample(i).capid() );
	if( i >= (maxi - 1) && i <= (maxi + 1) ) {
	  ts += i * ( fc - calibs_.pedestal( digi.sample(i).capid() ) );
	  bs += fc - calibs_.pedestal( digi.sample(i).capid() );
	}
      }

      if( en > adcThresh_) {
	hfHists.allEnergy_->Fill(en);
	if( bs != 0 ) hfHists.allTime_->Fill( ts / bs );
	else if( fVerbosity ) printf( "HcalLaserMonitor::processEvent - Calculation of HF hit time had a zero denominator!\n" );

	for( int i = 0; i < digi.size(); i++ ) {
	  int adc = digi.sample(i).adc();
	  float fc = adc2fc_[adc] + 0.5;
	  hfHists.allShape_->Fill( i, fc );
	  hfHists.allShapePedSub_->Fill( i, fc - calibs_.pedestal( digi.sample(i).capid() ) );
	  pedSubTS[i] = fc - calibs_.pedestal( digi.sample(i).capid() );
	}
      }

      if( doPerChannel_ ) perChanHists( HcalForward, digi.id(), pedSubTS, hfHists.perChanShape_, hfHists.perChanTime_, hfHists.perChanEnergy_, baseFolder_ );
    }
  } catch (...) {
    if( fVerbosity ) cout << "HcalLaserMonitor::processEvent - No HF Digis." << endl;
  }
}

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
inline void HcalLaserMonitor::perChanHists( const int id, const HcalDetId detid, const float* pedSubTS, map<HcalDetId, MonitorElement*>& tShape, map<HcalDetId, MonitorElement*>& tTime,
					    map<HcalDetId, MonitorElement*>& tEnergy, const string baseFolder ) {  
  MonitorElement* _me;
  if( m_dbe == NULL ) {
    printf( "HcalLaserMonitor::perChanHists - Null MonitorElement!\n" );
    return;
  }
  string type;

  if( id == HcalBarrel )       type = "HB";
  else if( id == HcalEndcap )  type = "HE"; 
  else if( id == HcalOuter )   type = "HO";
  else if( id == HcalForward ) type = "HF";
  else {
    printf( "HcalLaserMonitor::perChanHists - ID not understood!\n" );
    return;
  }


  m_dbe->setCurrentFolder( baseFolder + "/" + type + "/Expert" );
 
  meIter_ = tShape.find(detid);
  if( meIter_ != tShape.end() ) {
    _me = meIter_->second;
    if( _me == NULL ) {
      printf( "HcalLaserAnalysis::perChanHists - This histo is NULL!!??\n" );
      return;
    }
    else {
      float en = 0, ts = 0, bs = 0, maxa = 0;
      int maxi = 0;

      for( int i = sigS0_; i <= sigS1_; i++ ) {
	if( pedSubTS[i] > maxa ) { maxa = pedSubTS[i]; maxi = i; }
      }
      for( int i = sigS0_; i <= sigS1_; i++ ) {
	en += pedSubTS[i];
	if( i >= (maxi - 1) && i <= (maxi + 1) ) {
	  ts += i * pedSubTS[i];
	  bs += pedSubTS[i];
	}
	_me->Fill( i, pedSubTS[i] );
      }

      _me = tTime[detid];
      if( bs != 0 ) _me->Fill( ts / bs );
      else if( fVerbosity ) printf( "HcalLaserMonitor::perChanHists - Calculation of hit time had a zero denominator!\n" );

      _me = tEnergy[detid];
      _me->Fill( en );
    }
  }
  else {
    char name[1024];
    float en = 0, ts = 0, bs = 0, maxa = 0;
    int maxi = 0;

    sprintf( name, "%s Laser Shape ieta=%+03d iphi=%02d depth=%d", type.c_str(), detid.ieta(), detid.iphi(), detid.depth() );
    MonitorElement* insertShape = m_dbe->book1D( name, name, 10, -0.5, 9.5 );

    for( int i = sigS0_; i <= sigS1_; i++ ) {
      if( pedSubTS[i] > maxa ) { maxa = pedSubTS[i]; maxi = i; }
      insertShape->Fill( i, pedSubTS[i] );
    }
    for( int i = sigS0_; i <= sigS1_; i++ ) {
      en += pedSubTS[i];
      if( i >= (maxi - 1) && i <= (maxi + 1) ) {
	ts += i * pedSubTS[i];
	bs += pedSubTS[i];
      }
    }
    tShape[detid] = insertShape;
    
    sprintf( name, "%s Laser Time ieta=%+03d iphi=%02d depth=%d", type.c_str(), detid.ieta(), detid.iphi(), detid.depth() );
    MonitorElement* insertTime = m_dbe->book1D( name, name, 100, 0, 10 );
    if( bs != 0 ) insertTime->Fill( ts / bs );
    else if( fVerbosity ) printf( "HcalLaserMonitor::perChanHists - Calculation of hit time had a zero denominator!\n" );
    tTime[detid] = insertTime;

    sprintf( name, "%s Laser Energy ieta=%+03d iphi=%02d depth=%d", type.c_str(), detid.ieta(), detid.iphi(), detid.depth() );
    MonitorElement* insertEnergy = m_dbe->book1D( name, name, 250, 0, 5000 );
    insertEnergy->Fill(en);
    tEnergy[detid] = insertEnergy;
  }
}
