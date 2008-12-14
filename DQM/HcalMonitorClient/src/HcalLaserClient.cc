#include <DQM/HcalMonitorClient/interface/HcalLaserClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
HcalLaserClient::HcalLaserClient(){}
HcalLaserClient::~HcalLaserClient() { this->cleanup(); }
void HcalLaserClient::setup(void) {}
void HcalLaserClient::createTests() {}


//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
void HcalLaserClient::init( const ParameterSet& ps, DQMStore* dbe, const string clientName){
  HcalBaseClient::init( ps, dbe, clientName );
  
  TDCNumChannels_ = 0;
  TDCTrigger_ = 0;
  TDCRawOptosync_ = 0;
  TDCClockOptosync_ = 0;
  TDCRawOptosync_Trigger_ = 0;

  for( int i = 0; i < 32; i ++ ) {
    QADC_[i] = 0;
  }

  for( int i = 0; i < 4; i++ ) {    
    rms_shape_[i] = 0;
    mean_shape_[i] = 0;
    rms_time_[i] = 0;
    mean_time_[i] = 0;
    rms_energy_[i] = 0;
    mean_energy_[i] = 0;

    rms_shapeDep_[i] = 0;
    mean_shapeDep_[i] = 0;
    rms_timeDep_[i] = 0;
    mean_timeDep_[i] = 0;
    rms_energyDep_[i] = 0;
    mean_energyDep_[i] = 0;

    avg_shape_[i] = 0;
    avg_time_[i] = 0;
    avg_energy_[i] = 0;
  }
}

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
void HcalLaserClient::beginJob( const EventSetup& eventSetup ) {
  if ( debug_ ) cout << "HcalLaserClient: beginJob" << endl;
  ievt_ = jevt_ = 0;
  this->resetAllME();
}

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
void HcalLaserClient::beginRun(void){
  if ( debug_ ) cout << "HcalLaserClient: beginRun" << endl;
  jevt_ = 0;
  this->resetAllME();
}

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
void HcalLaserClient::analyze(void) {  
  jevt_++;
  getHistograms();
}

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
void HcalLaserClient::endRun(void) {
  if ( debug_ ) cout << "HcalLaserClient: endRun, jevt = " << jevt_ << endl;
  this->cleanup();
}

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
void HcalLaserClient::endJob(void) {
  if ( debug_ ) cout << "HcalLaserClient: endJob, ievt = " << ievt_ << endl;
  this->cleanup();
}

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
void HcalLaserClient::cleanup(void) {
  if( cloneME_ ){
    if( TDCNumChannels_ )         delete TDCNumChannels_;
    if( TDCTrigger_ )             delete TDCTrigger_;
    if( TDCRawOptosync_ )         delete TDCRawOptosync_;
    if( TDCClockOptosync_ )       delete TDCClockOptosync_;
    if( TDCRawOptosync_Trigger_ ) delete TDCRawOptosync_Trigger_;

    for( int i = 0; i < 32; i ++ ) {
      if( QADC_[i] ) delete QADC_[i];
    }
 
    for( int i = 0; i < 4; i++ ) {
      if( rms_shape_[i] )      delete rms_shape_[i];
      if( mean_shape_[i] )     delete mean_shape_[i];
      if( rms_time_[i] )       delete rms_time_[i];
      if( mean_time_[i] )      delete mean_time_[i];
      if( rms_energy_[i] )     delete rms_energy_[i];
      if( mean_energy_[i] )    delete mean_energy_[i];

      if( rms_shapeDep_[i] )   delete rms_shapeDep_[i];
      if( mean_shapeDep_[i] )  delete mean_shapeDep_[i];
      if( rms_timeDep_[i] )    delete rms_timeDep_[i];
      if( mean_timeDep_[i] )   delete mean_timeDep_[i];
      if( rms_energyDep_[i] )  delete rms_energyDep_[i];
      if( mean_energyDep_[i] ) delete mean_energyDep_[i];

      if( avg_shape_[i] )      delete avg_shape_[i];
      if( avg_time_[i] )       delete avg_time_[i];
      if( avg_energy_[i] )     delete avg_energy_[i];
    }
  }

  TDCNumChannels_ = 0;
  TDCTrigger_ = 0;
  TDCRawOptosync_ = 0;
  TDCClockOptosync_ = 0;
  TDCRawOptosync_Trigger_ = 0;

  for( int i = 0; i < 32; i ++ ) {
    QADC_[i] = 0;
  }
  
  for( int i = 0; i < 4; i++ ) {    
    rms_shape_[i] = 0;
    mean_shape_[i] = 0;
    rms_time_[i] = 0;
    mean_time_[i] = 0;
    rms_energy_[i] = 0;
    mean_energy_[i] = 0;

    rms_shapeDep_[i] = 0;
    mean_shapeDep_[i] = 0;
    rms_timeDep_[i] = 0;
    mean_timeDep_[i] = 0;
    rms_energyDep_[i] = 0;
    mean_energyDep_[i] = 0;

    avg_shape_[i] = 0;
    avg_time_[i] = 0;
    avg_energy_[i] = 0;
  }

  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();
}

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
void HcalLaserClient::report() {
  if( !dbe_ ) return;
  if( debug_ ) cout << "HcalLaserClient: report" << endl;
  
  char name[256];
  sprintf( name, "%sHcal/LaserMonitor/Laser Task Event Number", process_.c_str() );
  MonitorElement* me = dbe_->get(name);
  if( me ) {
    string s = me->valueString();
    ievt_ = -1;
    sscanf( (s.substr( 2, s.length()-2 )).c_str(), "%d", &ievt_ );
    if( debug_ ) cout << "Found '" << name << "'" << endl;
  }
  getHistograms();
}

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
void HcalLaserClient::getHistograms() {
  if( !dbe_ ) return;
  char name[256];

  //Get mean/rms maps by Geometry
  MonitorElement* meDepTimeMean[4];
  MonitorElement* meDepTimeRMS[4];
  MonitorElement* meDepShapeMean[4];
  MonitorElement* meDepShapeRMS[4];
  MonitorElement* meDepEnergyMean[4];
  MonitorElement* meDepEnergyRMS[4];

  for( int i = 0; i < 4; i++ ) {
    sprintf( name, "%sHcal/LaserMonitor/Laser Mean Time Depth %d", process_.c_str(), i+1 );
    meDepTimeMean[i] = dbe_->get(name);
    sprintf( name, "%sHcal/LaserMonitor/Laser RMS Time Depth %d", process_.c_str(), i+1 );
    meDepTimeRMS[i] = dbe_->get(name);
    sprintf( name, "%sHcal/LaserMonitor/2DShape/Laser Mean Shape Depth %d", process_.c_str(), i+1 );
    meDepShapeMean[i] = dbe_->get(name);
    sprintf( name, "%sHcal/LaserMonitor/2DShape/Laser RMS Shape Depth %d", process_.c_str(), i+1 );
    meDepShapeRMS[i] = dbe_->get(name);
    sprintf( name, "%sHcal/LaserMonitor/Laser Mean Energy Depth %d", process_.c_str(), i+1 );
    meDepEnergyMean[i] = dbe_->get(name);
    sprintf( name, "%sHcal/LaserMonitor/Laser RMS Energy Depth %d", process_.c_str(), i+1 );
    meDepEnergyRMS[i] = dbe_->get(name);

    if(meDepTimeMean[i])   dbe_->softReset( meDepTimeMean[i] );
    if(meDepTimeRMS[i])    dbe_->softReset( meDepTimeRMS[i] );
    if(meDepEnergyMean[i]) dbe_->softReset( meDepEnergyMean[i] );
    if(meDepEnergyRMS[i])  dbe_->softReset( meDepEnergyRMS[i] );
    if(meDepShapeMean[i])  dbe_->softReset( meDepShapeMean[i] );
    if(meDepShapeRMS[i])   dbe_->softReset( meDepShapeRMS[i] );
  }

  // Fill Histos
  // TDC
  sprintf( name, "LaserMonitor/QADCTDC/TDC Number of Channels" );
  TDCNumChannels_ = getHisto( name, process_, dbe_, debug_, cloneME_ );
  sprintf( name, "LaserMonitor/QADCTDC/TDC Trigger" );
  TDCTrigger_ = getHisto( name, process_, dbe_, debug_, cloneME_ );
  sprintf( name, "LaserMonitor/QADCTDC/TDC Raw Optosync" );
  TDCRawOptosync_ = getHisto( name, process_, dbe_, debug_, cloneME_ );
  sprintf( name, "LaserMonitor/QADCTDC/TDC Clock Optosync" );
  TDCClockOptosync_ = getHisto( name, process_, dbe_, debug_, cloneME_ );
  sprintf( name, "LaserMonitor/QADCTDC/TDC Raw Optosync Minus Trigger" );
  TDCRawOptosync_Trigger_ = getHisto( name, process_, dbe_, debug_, cloneME_ );

  for( int i = 0; i < 32; i ++ ) {
    sprintf( name, "LaserMonitor/QADCTDC/QDC %02d", i );
    QADC_[i] = getHisto( name, process_, dbe_, debug_, cloneME_ );
  }

  // Subdetectors
  for( int i = 0; i < 4; i++ ) {
    if( !subDetsOn_[i] ) continue;
    string type = "HB";
    if( i==1 ) type = "HE"; 
    else if( i==2 ) type = "HF";
    else if( i==3 ) type = "HO";

    sprintf( name, "LaserMonitor/%s/%s Average Pulse Shape", type.c_str(), type.c_str() );
    avg_shape_[i] = getHisto( name, process_, dbe_, debug_, cloneME_ );
    sprintf( name, "LaserMonitor/%s/%s Average Pulse Time", type.c_str(), type.c_str() );
    avg_time_[i] = getHisto( name, process_, dbe_, debug_, cloneME_ );
    sprintf( name, "LaserMonitor/%s/%s Average Pulse Energy", type.c_str(), type.c_str() );
    avg_energy_[i] = getHisto( name, process_, dbe_, debug_, cloneME_ );
    
    sprintf( name, "%sHcal/LaserMonitor/%s/%s Laser Shape RMS Values", process_.c_str(), type.c_str(), type.c_str() );
    MonitorElement* meShapeRMS = dbe_->get(name);
    sprintf( name, "%sHcal/LaserMonitor/%s/%s Laser Shape Mean Values", process_.c_str(), type.c_str(), type.c_str() );
    MonitorElement* meShapeMean = dbe_->get(name);
    sprintf( name, "%sHcal/LaserMonitor/%s/%s Laser Time RMS Values", process_.c_str(), type.c_str(), type.c_str() );
    MonitorElement* meTimeRMS = dbe_->get(name);
    sprintf( name, "%sHcal/LaserMonitor/%s/%s Laser Time Mean Values", process_.c_str(), type.c_str(), type.c_str() );
    MonitorElement* meTimeMean = dbe_->get(name);
    sprintf( name, "%sHcal/LaserMonitor/%s/%s Laser Energy RMS Values", process_.c_str(), type.c_str(), type.c_str() );
    MonitorElement* meEnergyRMS = dbe_->get(name);
    sprintf( name, "%sHcal/LaserMonitor/%s/%s Laser Energy Mean Values", process_.c_str(), type.c_str(), type.c_str() );
    MonitorElement* meEnergyMean = dbe_->get(name);

    if( !meShapeRMS || !meShapeMean ) continue;
    if( !meTimeRMS || !meTimeMean ) continue;
    if( !meEnergyRMS || !meEnergyMean ) continue;
    dbe_->softReset( meShapeRMS );  dbe_->softReset( meShapeMean );
    dbe_->softReset( meTimeRMS );   dbe_->softReset( meTimeMean );
    dbe_->softReset( meEnergyRMS ); dbe_->softReset( meEnergyMean );
   
    for( int ieta = -42; ieta <= 42; ieta++ ) {
      if( ieta == 0 ) continue;
      for( int iphi = 1; iphi <= 73; iphi++ ) {
	for( int depth = 1; depth <= 4; depth++ ) {
	  if( !isValidGeom(i, ieta, iphi,depth) ) continue;
	  HcalSubdetector subdet = HcalBarrel;
	  if( i == 1 ) subdet = HcalEndcap;	  
	  else if( i == 2 ) subdet = HcalForward;
	  else if( i == 3 ) subdet = HcalOuter;
	  
	  sprintf( name, "%sHcal/LaserMonitor/%s/Expert/%s Laser Shape ieta=%+03d iphi=%02d depth=%d",
		   process_.c_str(), type.c_str(), type.c_str(), ieta, iphi, depth );  
	  MonitorElement* me = dbe_->get(name);
	  
	  if( me ) {
	    meShapeRMS->Fill( me->getRMS() );
	    meShapeMean->Fill( me->getMean() );
	    meDepShapeRMS[depth-1]->Fill( ieta, iphi, me->getRMS() );
	    meDepShapeMean[depth-1]->Fill( ieta, iphi, me->getMean() );
	  }

	  float timeMeanVal = -1; float enMeanVal = -1;
	  float timeRMSVal = -1; float enRMSVal = -1;

	  sprintf( name, "%sHcal/LaserMonitor/%s/Expert/%s Laser Time ieta=%+03d iphi=%02d depth=%d",
		   process_.c_str(), type.c_str(), type.c_str(), ieta, iphi, depth );  
	  me = dbe_->get(name);

	  if( me ) {
	    timeMeanVal = me->getMean();
	    timeRMSVal = me->getRMS();
	    meTimeRMS->Fill( timeRMSVal );
	    meTimeMean->Fill( timeMeanVal );	
	    meDepTimeRMS[depth-1]->Fill( ieta, iphi, timeRMSVal );
	    meDepTimeMean[depth-1]->Fill( ieta, iphi, timeMeanVal );
	  }
	  
	  sprintf( name, "%sHcal/LaserMonitor/%s/Expert/%s Laser Energy ieta=%+03d iphi=%02d depth=%d",
		   process_.c_str(), type.c_str(), type.c_str(), ieta, iphi, depth );  
	  me = dbe_->get(name);
	  
	  if( me ) {
	    enMeanVal = me->getMean();
	    enRMSVal = me->getRMS();
	    meEnergyRMS->Fill( enRMSVal );
	    meEnergyMean->Fill( enMeanVal );
	    meDepEnergyRMS[depth-1]->Fill( ieta, iphi, enRMSVal );
	    meDepEnergyMean[depth-1]->Fill( ieta, iphi, enMeanVal );
	  } // end of if me
	} // end of for depth
      } // end of for iphi
    } // end of ieta
    
    rms_shape_[i] = getHisto( meShapeRMS, debug_, cloneME_ );
    mean_shape_[i] = getHisto( meShapeMean, debug_, cloneME_ );
    rms_time_[i] = getHisto( meTimeRMS, debug_, cloneME_ );
    mean_time_[i] = getHisto( meTimeMean, debug_, cloneME_ );
    rms_energy_[i] = getHisto( meEnergyRMS, debug_, cloneME_ );
    mean_energy_[i] = getHisto( meEnergyMean, debug_, cloneME_ );
  
  }

  for( int i = 0; i < 4; i++ ) {
    rms_shapeDep_[i] = getHisto2( meDepShapeRMS[i], debug_, cloneME_ );
    mean_shapeDep_[i] = getHisto2( meDepShapeMean[i], debug_, cloneME_ );
    rms_timeDep_[i] = getHisto2( meDepTimeRMS[i], debug_, cloneME_ );
    mean_timeDep_[i] = getHisto2( meDepTimeMean[i], debug_, cloneME_ );
    rms_energyDep_[i] = getHisto2( meDepEnergyRMS[i], debug_, cloneME_ );
    mean_energyDep_[i] = getHisto2( meDepEnergyMean[i], debug_, cloneME_ );
  } 
}

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
void HcalLaserClient::resetAllME() {
  if( !dbe_ ) return;
  Char_t name[150];    

  sprintf( name, "%sHcal/LaserMonitor/QADCTDC/TDC Number of Channels", process_.c_str() );
  resetME( name, dbe_ );
  sprintf( name, "%sHcal/LaserMonitor/QADCTDC/TDC Trigger", process_.c_str() );
  resetME( name, dbe_ );
  sprintf( name, "%sHcal/LaserMonitor/QADCTDC/TDC Raw Optosync", process_.c_str() );
  resetME( name, dbe_ );
  sprintf( name, "%sHcal/LaserMonitor/QADCTDC/TDC Clock Optosync", process_.c_str() );
  resetME( name, dbe_ );
  sprintf( name, "%sHcal/LaserMonitor/QADCTDC/TDC Raw Optosync Minus Trigger", process_.c_str() );
  resetME( name, dbe_ );

  for( int i = 0; i < 32; i ++ ) {
    sprintf( name, "%sHcal/LaserMonitor/QADCTDC/QDC %02d", process_.c_str(), i );
    resetME( name, dbe_ );
  }  

  for( int i = 1; i <= 4; i++ ) {
    sprintf( name, "%sHcal/LaserMonitor/Laser Mean Time Depth %d", process_.c_str(), i );
    resetME( name, dbe_ );
    sprintf( name, "%sHcal/LaserMonitor/Laser RMS Time Depth %d", process_.c_str(), i );
    resetME( name, dbe_ );
    sprintf( name, "%sHcal/LaserMonitor/2DShape/Laser Mean Shape Depth %d", process_.c_str(), i );
    resetME( name, dbe_ );
    sprintf( name, "%sHcal/LaserMonitor/2DShape/Laser RMS Shape Depth %d", process_.c_str(), i );
    resetME( name, dbe_ );
    sprintf( name, "%sHcal/LaserMonitor/Laser Mean Energy Depth %d", process_.c_str(), i );
    resetME( name, dbe_ );
    sprintf( name, "%sHcal/LaserMonitor/Laser RMS Energy Depth %d", process_.c_str(), i );
    resetME( name, dbe_ );
  }

  for( int i = 0; i < 4; i++ ) {
    if( !subDetsOn_[i] ) continue;
    string type = "HB";
    if( i == 1 ) type = "HE"; 
    else if( i == 2 ) type = "HF"; 
    else if( i == 3 ) type = "HO"; 

    sprintf( name, "%sHcal/LaserMonitor/%s/%s Ped Subtracted Pulse Shape", process_.c_str(), type.c_str(), type.c_str() );
    resetME( name, dbe_ );
    sprintf( name, "%sHcal/LaserMonitor/%s/%s Average Pulse Shape", process_.c_str(), type.c_str(), type.c_str() );
    resetME( name, dbe_ );
    sprintf( name, "%sHcal/LaserMonitor/%s/%s Laser Shape RMS Values", process_.c_str(), type.c_str(), type.c_str() );
    resetME( name, dbe_ );
    sprintf( name, "%sHcal/LaserMonitor/%s/%s Laser Shape Mean Values", process_.c_str(), type.c_str(), type.c_str() );
    resetME( name, dbe_ );
    sprintf( name, "%sHcal/LaserMonitor/%s/%s Average Pulse Time", process_.c_str(), type.c_str(), type.c_str() );
    resetME( name, dbe_ );
    sprintf( name, "%sHcal/LaserMonitor/%s/%s Laser Time RMS Values", process_.c_str(), type.c_str(), type.c_str() );
    resetME( name, dbe_ );
    sprintf( name, "%sHcal/LaserMonitor/%s/%s Laser Time Mean Values", process_.c_str(), type.c_str(), type.c_str() );
    resetME( name, dbe_ );
    sprintf( name, "%sHcal/LaserMonitor/%s/%s Average Pulse Energy", process_.c_str(), type.c_str(), type.c_str() );
    resetME( name, dbe_ );
    sprintf( name, "%sHcal/LaserMonitor/%s/%s Laser Energy RMS Values", process_.c_str(), type.c_str(), type.c_str() );
    resetME( name, dbe_ );
    sprintf( name, "%sHcal/LaserMonitor/%s/%s Laser Energy Mean Values", process_.c_str(), type.c_str(), type.c_str() );
    resetME( name, dbe_ );
    
    for( int ieta = -42; ieta < 42; ieta++ ) {
      if( ieta == 0 ) continue;
      for( int iphi = 0; iphi < 73; iphi++ ) {
	for( int depth = 1; depth < 4; depth++ ) {
	  if( !isValidGeom(i, ieta, iphi, depth) ) continue;
	  sprintf( name, "%sHcal/LaserMonitor/%s/Expert/%s Laser Shape ieta=%+03d iphi=%02d depth=%d",
		   process_.c_str(), type.c_str(), type.c_str(), ieta, iphi, depth );  
	  resetME( name, dbe_ );
	  sprintf( name, "%sHcal/LaserMonitor/%s/Expert/%s Laser Time ieta=%+03d iphi=%02d depth=%d",
		   process_.c_str(), type.c_str(), type.c_str(), ieta, iphi, depth );  
	  resetME( name, dbe_ );
	  sprintf( name, "%sHcal/LaserMonitor/%s/Expert/%s Laser Energy ieta=%+03d iphi=%02d depth=%d",
		   process_.c_str(), type.c_str(), type.c_str(), ieta, iphi, depth );  
	  resetME( name, dbe_ );
	} // end of for depth
      } // end of for iphi
    } // end of for ieta
  } // end of for i (subdet)
}

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
void HcalLaserClient::htmlOutput( int runNo, string htmlDir, string htmlName ) {
  
  cout << "Preparing HcalLaserClient html output ..." << endl;
  string client = "LaserMonitor";
  htmlErrors( runNo, htmlDir, client, process_, dbe_, dqmReportMapErr_, dqmReportMapWarn_, dqmReportMapOther_ );

  ofstream htmlFile;
  htmlFile.open( (htmlDir + htmlName).c_str() );

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: Hcal Laser Task output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal Laser</span></h2> " << endl;

  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;

  htmlFile << "<hr>" << endl;
  htmlFile << "<table  width=100% border=1><tr>" << endl;
  if( hasErrors() ) htmlFile << "<td bgcolor=red><a href=\"LaserMonitorErrors.html\">Errors in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Errors</td>" << endl;
  if( hasWarnings() ) htmlFile << "<td bgcolor=yellow><a href=\"LaserMonitorWarnings.html\">Warnings in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Warnings</td>" << endl;
  if( hasOther() ) htmlFile << "<td bgcolor=aqua><a href=\"LaserMonitorMessages.html\">Messages in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Messages</td>" << endl;
  htmlFile << "</tr></table>" << endl;
  htmlFile << "<hr>" << endl;
  
  htmlFile << "<h2><strong>Hcal Laser Histograms</strong></h2>" << endl;
  htmlFile << "<h3>" << endl;
  htmlFile << "<a href=\"#QADCTDC_Plots\">QADCTDC Plots </a></br>" << endl;
  htmlFile << "<a href=\"#GEO_Plots\">Geometry Plots </a></br>" << endl;
  if( subDetsOn_[0] ) htmlFile << "<a href=\"#HB_Plots\">HB Plots </a></br>" << endl;
  if( subDetsOn_[1] ) htmlFile << "<a href=\"#HE_Plots\">HE Plots </a></br>" << endl;
  if( subDetsOn_[2] ) htmlFile << "<a href=\"#HF_Plots\">HF Plots </a></br>" << endl;
  if( subDetsOn_[3] ) htmlFile << "<a href=\"#HO_Plots\">HO Plots </a></br>" << endl;
  htmlFile << "</h3>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  
  htmlFile << "<tr align=\"left\">" << endl;
  htmlFile << "<td>&nbsp;&nbsp;&nbsp;<a name=\"QADCTDC_Plots\"><h3>QADCTDC Histograms</h3></td></tr>" << endl;
  histoHTML( runNo, TDCNumChannels_, "Number of Channels", "Events",  92, htmlFile, htmlDir );
  htmlFile << "</tr>" << endl;
  histoHTML( runNo, TDCTrigger_, "Time (ns)", "Events",  92, htmlFile, htmlDir );
  histoHTML( runNo, TDCRawOptosync_Trigger_, "Time (ns)", "Events", 100, htmlFile, htmlDir );
  htmlFile << "</tr>" << endl;
  histoHTML( runNo, TDCRawOptosync_, "Time (ns)", "Events",  92, htmlFile, htmlDir );
  histoHTML( runNo, TDCClockOptosync_, "Time (ns)", "Events", 100, htmlFile, htmlDir );
  htmlFile << "</tr>" << endl;
  histoHTML( runNo, QADC_[3], "ADC Counts", "Events",  92, htmlFile, htmlDir );
  histoHTML( runNo, QADC_[0], "ADC Counts", "Events", 100, htmlFile, htmlDir );
  htmlFile << "</tr>" << endl;
  histoHTML( runNo, QADC_[4], "ADC Counts", "Events",  92, htmlFile, htmlDir );
  histoHTML( runNo, QADC_[1], "ADC Counts", "Events", 100, htmlFile, htmlDir );
  htmlFile << "</tr>" << endl;
  histoHTML( runNo, QADC_[5], "ADC Counts", "Events",  92, htmlFile, htmlDir );
  histoHTML( runNo, QADC_[2], "ADC Counts", "Events", 100, htmlFile, htmlDir );
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  htmlFile << "<td>&nbsp;&nbsp;&nbsp;<a name=\"GEO_Plots\"><h3>Geometry Histograms</h3></td></tr>" << endl;
  histoHTML2( runNo, mean_energyDep_[0], "iEta", "iPhi", 92, htmlFile, htmlDir );
  histoHTML2( runNo, rms_energyDep_[0], "iEta", "iPhi", 100, htmlFile, htmlDir );
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2( runNo, mean_energyDep_[1], "iEta", "iPhi", 92, htmlFile, htmlDir );
  histoHTML2( runNo, rms_energyDep_[1], "iEta", "iPhi", 100, htmlFile, htmlDir );
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2( runNo, mean_energyDep_[2], "iEta", "iPhi", 92, htmlFile, htmlDir );
  histoHTML2( runNo, rms_energyDep_[2], "iEta", "iPhi", 100, htmlFile, htmlDir );
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2( runNo, mean_energyDep_[3], "iEta", "iPhi", 92, htmlFile, htmlDir );
  histoHTML2( runNo, rms_energyDep_[3], "iEta", "iPhi", 100, htmlFile, htmlDir );
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2( runNo, mean_timeDep_[0], "iEta", "iPhi", 92, htmlFile, htmlDir );
  histoHTML2( runNo, rms_timeDep_[0], "iEta", "iPhi", 100, htmlFile, htmlDir );
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2( runNo, mean_timeDep_[1], "iEta", "iPhi", 92, htmlFile, htmlDir );
  histoHTML2( runNo, rms_timeDep_[1], "iEta", "iPhi", 100, htmlFile, htmlDir );
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2( runNo, mean_timeDep_[2], "iEta", "iPhi", 92, htmlFile, htmlDir );
  histoHTML2( runNo, rms_timeDep_[2], "iEta", "iPhi", 100, htmlFile, htmlDir );
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2( runNo, mean_timeDep_[3], "iEta", "iPhi", 92, htmlFile, htmlDir );
  histoHTML2( runNo, rms_timeDep_[3], "iEta", "iPhi", 100, htmlFile, htmlDir );
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2( runNo, mean_shapeDep_[0], "iEta", "iPhi", 92, htmlFile, htmlDir );
  histoHTML2( runNo, rms_shapeDep_[0], "iEta", "iPhi", 100, htmlFile, htmlDir );
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2( runNo, mean_shapeDep_[1], "iEta", "iPhi", 92, htmlFile, htmlDir );
  histoHTML2( runNo, rms_shapeDep_[1], "iEta", "iPhi", 100, htmlFile, htmlDir );
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2( runNo, mean_shapeDep_[2], "iEta", "iPhi", 92, htmlFile, htmlDir );
  histoHTML2( runNo, rms_shapeDep_[2], "iEta", "iPhi", 100, htmlFile, htmlDir );
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2( runNo, mean_shapeDep_[3], "iEta", "iPhi", 92, htmlFile, htmlDir );
  histoHTML2( runNo, rms_shapeDep_[3], "iEta", "iPhi", 100, htmlFile, htmlDir );
  htmlFile << "</tr>" << endl;

  for( int i = 0; i < 4; i++ ) {
    if( !subDetsOn_[i] ) continue; 
    string type = "HB";
    if( i == 1 ) type = "HE"; 
    else if( i == 2 ) type = "HF"; 
    else if( i == 3 ) type = "HO"; 
     
    htmlFile << "<tr align=\"left\">" << endl;  
    htmlFile << "<td>&nbsp;&nbsp;&nbsp;<a name=\""<<type<<"_Plots\"><h3>" << type << " Histograms</h3></td></tr>" << endl;
     
    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML( runNo, avg_shape_[i], "Timeslice (25ns)", "Events", 92, htmlFile, htmlDir );
    histoHTML( runNo, avg_time_[i], "Timeslice (25ns)", "Events", 100, htmlFile, htmlDir );
    htmlFile << "</tr>" << endl;
     
    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML( runNo, avg_energy_[i], "ADC Sum", "Events", 92, htmlFile, htmlDir );
    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML( runNo, mean_shape_[i], "Timeslice (25ns)", "Channels", 100, htmlFile, htmlDir );
    histoHTML( runNo, rms_shape_[i], "Timeslice (25ns)", "Channels", 92, htmlFile, htmlDir );
    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML( runNo, mean_time_[i], "Timeslice (25ns)", "Channels", 100, htmlFile, htmlDir );
    histoHTML( runNo, rms_time_[i], "Timeslice (25ns)", "Channels", 92, htmlFile, htmlDir );
    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML( runNo, mean_energy_[i], "ADC Sum", "Channels", 100, htmlFile, htmlDir );
    histoHTML( runNo, rms_energy_[i], "ADC Sum", "Channels", 92, htmlFile, htmlDir );
    htmlFile << "</tr>" << endl;
  }
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();
}


//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
void HcalLaserClient::loadHistograms( TFile* infile ) {

  TNamed* tnd = (TNamed*)infile->Get( "DQMData/Hcal/LaserMonitor/Laser Task Event Number" );
  if(tnd) {
    string s = tnd->GetTitle();
    ievt_ = -1;
    sscanf( (s.substr(2, s.length()-2)).c_str(), "%d", &ievt_ );
  }
  char name[256];

  sprintf( name, "DQMData/Hcal/LaserMonitor/QADCTDC/TDC Number of Channels" );
  TDCNumChannels_ = (TH1F*)infile->Get(name);
  sprintf( name, "DQMData/Hcal/LaserMonitor/QADCTDC/TDC Trigger" );
  TDCTrigger_ = (TH1F*)infile->Get(name);
  sprintf( name, "DQMData/Hcal/LaserMonitor/QADCTDC/TDC Raw Optosync" );
  TDCRawOptosync_ = (TH1F*)infile->Get(name);
  sprintf( name, "DQMData/Hcal/LaserMonitor/QADCTDC/TDC Clock Optosync" );
  TDCClockOptosync_ = (TH1F*)infile->Get(name);
  sprintf( name, "DQMData/Hcal/LaserMonitor/QADCTDC/TDC Raw Optosync Minus Trigger" );
  TDCRawOptosync_Trigger_ = (TH1F*)infile->Get(name);

  for( int i = 0; i < 4; i++ ) {
    sprintf( name, "%sHcal/LaserMonitor/Laser Mean Time Depth %d", process_.c_str(), i+1 );
    mean_timeDep_[i] = (TH2F*)infile->Get(name);
    sprintf( name, "%sHcal/LaserMonitor/Laser RMS Time Depth %d", process_.c_str(), i+1 );
    rms_timeDep_[i] = (TH2F*)infile->Get(name);
    sprintf( name, "%sHcal/LaserMonitor/2DShape/Laser Mean Shape Depth %d", process_.c_str(), i+1 );
    mean_shapeDep_[i] = (TH2F*)infile->Get(name);
    sprintf( name, "%sHcal/LaserMonitor/2DShape/Laser RMS Shape Depth %d", process_.c_str(), i+1 );
    rms_shapeDep_[i] = (TH2F*)infile->Get(name);
    sprintf( name, "%sHcal/LaserMonitor/Laser Mean Energy Depth %d", process_.c_str(), i+1 );
    mean_energyDep_[i] = (TH2F*)infile->Get(name);
    sprintf( name, "%sHcal/LaserMonitor/Laser RMS Energy Depth %d", process_.c_str(), i+1 );
    rms_energyDep_[i] = (TH2F*)infile->Get(name);
  }

  for( int i = 0; i < 4; i++ ) {
    if( !subDetsOn_[i] ) continue; 
    string type = "HB";
    if( i == 1 ) type = "HE"; 
    else if( i == 2 ) type = "HF"; 
    else if( i == 3 ) type = "HO";

    sprintf( name, "DQMData/Hcal/LaserMonitor/%s/%s Average Pulse Shape", type.c_str(), type.c_str() );      
    avg_shape_[i] = (TH1F*)infile->Get(name);
    sprintf( name, "DQMData/Hcal/LaserMonitor/%s/%s Average Pulse Time", type.c_str(), type.c_str() );      
    avg_time_[i] = (TH1F*)infile->Get(name);
    sprintf( name, "DQMData/Hcal/LaserMonitor/%s/%s Average Pulse Energy", type.c_str(), type.c_str() );      
    avg_energy_[i] = (TH1F*)infile->Get(name);
    
    sprintf( name, "DQMData/Hcal/LaserMonitor/%s/%s Laser Shape RMS Values", type.c_str(), type.c_str() );
    rms_shape_[i] = (TH1F*)infile->Get(name);
    sprintf( name, "DQMData/Hcal/LaserMonitor/%s/%s Laser Shape Mean Values", type.c_str(), type.c_str() );
    mean_shape_[i] = (TH1F*)infile->Get(name);

    sprintf( name, "DQMData/Hcal/LaserMonitor/%s/%s Laser Time RMS Values", type.c_str(), type.c_str() );
    rms_time_[i] = (TH1F*)infile->Get(name);
    sprintf( name, "DQMData/Hcal/LaserMonitor/%s/%s Laser Time Mean Values", type.c_str(), type.c_str() );
    mean_time_[i] = (TH1F*)infile->Get(name);

    sprintf( name, "DQMData/Hcal/LaserMonitor/%s/%s Laser Energy RMS Values", type.c_str(), type.c_str() );
    rms_energy_[i] = (TH1F*)infile->Get(name);
    sprintf( name, "DQMData/Hcal/LaserMonitor/%s/%s Laser Energy Mean Values", type.c_str(), type.c_str() );
    mean_energy_[i] = (TH1F*)infile->Get(name);

    for( int ieta = -42; ieta <= 42; ieta++ ) {
      if( ieta == 0 ) continue;
      for( int iphi = 1; iphi <= 73; iphi++ ) {
	for( int depth = 1; depth <= 4; depth++ ) {
	  if( !isValidGeom(i, ieta, iphi, depth) ) continue;
	  sprintf( name, "DQMData/Hcal/LaserMonitor/%s/Expert/%s Laser Shape ieta=%+03d iphi=%02d depth=%d",
		   type.c_str(), type.c_str(), ieta, iphi, depth );  
	  TH1F* h = (TH1F*)infile->Get(name);
	  if( h ) {
	    rms_shape_[i]->Fill( h->GetRMS() );
	    mean_shape_[i]->Fill( h->GetMean() );
	  }
	  
	  sprintf( name, "DQMData/Hcal/LaserMonitor/%s/Expert/%s Laser Time ieta=%+03d iphi=%02d depth=%d",
		   type.c_str(), type.c_str(), ieta, iphi, depth );  
	  h = (TH1F*)infile->Get(name);
	  if( h ) {
	    rms_time_[i]->Fill( h->GetRMS() );
	    mean_time_[i]->Fill( h->GetMean() );
	  }	  

	  sprintf( name, "DQMData/Hcal/LaserMonitor/%s/Expert/%s Laser Energy ieta=%+03d iphi=%02d depth=%d",
		   type.c_str(), type.c_str(), ieta, iphi, depth );  
	  h = (TH1F*)infile->Get(name);
	  if( h ) {
	    rms_energy_[i]->Fill( h->GetRMS() );
	    mean_energy_[i]->Fill( h->GetMean() );
	  } // end of if h	  
	} // end of for depth
      } // end of for iphi
    } // end of for ieta
  } // end of for i (subdet)
}
