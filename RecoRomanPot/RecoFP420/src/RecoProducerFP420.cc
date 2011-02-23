#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "RecoRomanPot/RecoFP420/interface/RecoProducerFP420.h"
#include "DataFormats/FP420Cluster/interface/RecoFP420.h"
#include "DataFormats/FP420Cluster/interface/RecoCollectionFP420.h"

#include "CLHEP/Vector/LorentzVector.h"

#include <math.h>

//#include <iostream>

RecoProducerFP420::RecoProducerFP420(const edm::ParameterSet& conf):conf_(conf)  { 
  
  // Create LHC beam line
  //  double length  = param.getParameter<double>("BeamLineLength");
  //  std::cout << " BeamLineLength = " << length << std::endl;
  
  verbosity      = conf_.getUntrackedParameter<int>("VerbosityLevel");
  beam1filename  = conf_.getParameter<std::string>("Beam1");
  beam2filename  = conf_.getParameter<std::string>("Beam2");  
  lengthfp420    = conf_.getParameter<double>("BeamLineLengthFP420" );//m
  lengthhps240   = conf_.getParameter<double>("BeamLineLengthHPS240" );//m
  
  edm::LogInfo ("RecoProducerFP420") << "RecoProducerFP420 parameters: \n" 
				     << "   Verbosity: "   << verbosity << "\n"
				     << "   lengthfp420: " << lengthfp420 << "   lengthhps240: " << lengthhps240 << "\n";
  if (verbosity > 1) {
    std::cout << "  RecoProducerFP420: constructor    " << std::endl;
    std::cout << "   BeamLineLengthFP420:    " << lengthfp420 << "   BeamLineLengthHPS240:    " << lengthhps240 << std::endl;
  }
  
  //  edm::FileInPath b1("SimTransport/HectorData/twiss_ip5_b1_v6.5.txt");
  //  edm::FileInPath b2("SimTransport/HectorData/twiss_ip5_b2_v6.5.txt");
  edm::FileInPath b1(beam1filename.c_str());
  edm::FileInPath b2(beam2filename.c_str());
  
  // construct beam line for FP420:                                                                                           .
  if(lengthfp420 > 0. ) {
    m_beamlineFP4201 = new H_BeamLine(  1, lengthfp420 + 0.1 ); // (direction, length)
    m_beamlineFP4202 = new H_BeamLine( -1, lengthfp420 + 0.1 ); //
    m_beamlineFP4201->fill( b1.fullPath(),  1, "IP5" );
    m_beamlineFP4202->fill( b2.fullPath(), -1, "IP5" );
    m_beamlineFP4201->offsetElements( 120, -0.097 );
    m_beamlineFP4202->offsetElements( 120, +0.097 );
    m_beamlineFP4201->calcMatrix();
    m_beamlineFP4202->calcMatrix();
  }  
  else{
    if ( verbosity ) LogDebug("RecoProducerSetup") << "Hector: WARNING: lengthfp420=  " << lengthfp420;
  }
  
  // construct beam line for HPS240:                                                                                           .
  if(lengthhps240 > 0. ) {
    m_beamlineHPS2401 = new H_BeamLine(  1, lengthhps240 + 0.1 ); // (direction, length)
    m_beamlineHPS2402 = new H_BeamLine( -1, lengthhps240 + 0.1 ); //
    m_beamlineHPS2401->fill( b1.fullPath(),  1, "IP5" );
    m_beamlineHPS2402->fill( b2.fullPath(), -1, "IP5" );
    m_beamlineHPS2401->offsetElements( 120, -0.097 );
    m_beamlineHPS2402->offsetElements( 120, +0.097 );
    m_beamlineHPS2401->calcMatrix();
    m_beamlineHPS2402->calcMatrix();
  }  
  else{
    if ( verbosity ) LogDebug("RecoProducerSetup") << "Hector: WARNING: lengthhps240=  " << lengthhps240;
  }
  
  edm::LogInfo ("RecoProducerFP420") << "====== constructor finished ========================\n";
  
}

RecoProducerFP420::~RecoProducerFP420(){
  if(lengthfp420  != 0. ) {
    delete m_beamlineFP4201;
    delete m_beamlineFP4202;
  }
  if(lengthhps240  != 0. ) {
    delete m_beamlineHPS2401;
    delete m_beamlineHPS2402;
  }
  //  delete m_rp420_f ;
  //  delete m_rp420_b ;
  //  delete m_rp240_f ;
  //  delete m_rp240_b ;
}

std::vector<RecoFP420>  RecoProducerFP420::reconstruct(int StIDReco, double x1_trk, double y1_trk, double x2_trk, double y2_trk, double z1_trk, double z2_trk){
  // ==================================================================================  
  // ==================================================================================  
  std::vector<RecoFP420> rhits;
  int restracks = 10;// max # tracks
  rhits.reserve(restracks); 
  rhits.clear();
  // ==================================================================================  
  // trivial (TM), angle compensation (AM) and position compensation (PM) methods
  // #define TM 1    #define AM 2   #define PM 3
  m_tx0=-100., m_ty0=-100., m_x0=-100., m_y0=-100.;
  
  if ( StIDReco == 1 ) {
    m_rp420_f = new H_RecRPObject( z1_trk*0.001, z2_trk*0.001, *m_beamlineFP4201);// m
    if ( m_rp420_f ) {
      if (verbosity >1) {
	std::cout << "  RecoProducerFP420: input coord. in um   " << std::endl;
	std::cout << "   x1_trk:    " << x1_trk << "   y1_trk:    " << y1_trk << std::endl;
	std::cout << "   x2_trk:    " << x2_trk << "   y2_trk:    " << y2_trk << std::endl;
      }
      m_rp420_f->setPositions( x1_trk, y1_trk, x2_trk ,y2_trk );//input coord. in um
      m_e   = m_rp420_f->getE( AM );// GeV
      //  std::cout << "   m_e1:    " << m_rp420_f->getE( TM ) << std::endl;
      //  std::cout << "   m_e2:    " << m_rp420_f->getE( AM ) << std::endl;
      m_tx0 = m_rp420_f->getTXIP();// urad
      m_ty0 = m_rp420_f->getTYIP();// urad
      m_x0  = m_rp420_f->getX0();// um
      m_y0  = m_rp420_f->getY0();// um
      m_q2  = m_rp420_f->getQ2();// GeV^2
    }// if ( m_rp420_f
  }// if ( StIDReco
  
  else if ( StIDReco == 2 ) {
    m_rp420_b = new H_RecRPObject( z1_trk*0.001, z2_trk*0.001, *m_beamlineFP4202);// m
    if ( m_rp420_b ) {
      m_rp420_b->setPositions( x1_trk, y1_trk, x2_trk ,y2_trk );// input coord. in um
      m_e   = m_rp420_b->getE( AM );// GeV
      m_tx0 = m_rp420_b->getTXIP();// urad
      m_ty0 = m_rp420_b->getTYIP();// urad
      m_x0  = m_rp420_b->getX0();// um
      m_y0  = m_rp420_b->getY0();// um
      m_q2  = m_rp420_b->getQ2();// GeV^2
    }// if ( m_rp420_b
  }// StIDReco
  
  else if ( StIDReco == 3 ) {
    m_rp240_f = new H_RecRPObject( z1_trk*0.001, z2_trk*0.001, *m_beamlineHPS2401);// m
    if ( m_rp240_f ) {
      if (verbosity >1) {
	std::cout << "  RecoProducerFP420: input coord. in um   " << std::endl;
	std::cout << "   x1_trk:    " << x1_trk << "   y1_trk:    " << y1_trk << std::endl;
	std::cout << "   x2_trk:    " << x2_trk << "   y2_trk:    " << y2_trk << std::endl;
      }
      m_rp240_f->setPositions( x1_trk, y1_trk, x2_trk ,y2_trk );//input coord. in um
      m_e   = m_rp240_f->getE( AM );// GeV
      //  std::cout << "   m_e1:    " << m_rp240_f->getE( TM ) << std::endl;
      //  std::cout << "   m_e2:    " << m_rp240_f->getE( AM ) << std::endl;
      m_tx0 = m_rp240_f->getTXIP();// urad
      m_ty0 = m_rp240_f->getTYIP();// urad
      m_x0  = m_rp240_f->getX0();// um
      m_y0  = m_rp240_f->getY0();// um
      m_q2  = m_rp240_f->getQ2();// GeV^2
    }// if ( m_rp240_f
  }// if ( StIDReco
  
  else if ( StIDReco == 4 ) {
    m_rp240_b = new H_RecRPObject( z1_trk*0.001, z2_trk*0.001, *m_beamlineHPS2402);// m
    if ( m_rp240_b ) {
      m_rp240_b->setPositions( x1_trk, y1_trk, x2_trk ,y2_trk );// input coord. in um
      m_e   = m_rp240_b->getE( AM );// GeV
      m_tx0 = m_rp240_b->getTXIP();// urad
      m_ty0 = m_rp240_b->getTYIP();// urad
      m_x0  = m_rp240_b->getX0();// um
      m_y0  = m_rp240_b->getY0();// um
      m_q2  = m_rp240_b->getQ2();// GeV^2
    }// if ( m_rp240_b
  }// StIDReco
  
  else{
    return rhits;
  }
  
  // ==============================  
  if (verbosity > 1) {
    std::cout << "  RecoProducerFP420: rhits.push_back    " << std::endl;
    std::cout << "   m_e:    " << m_e << std::endl;
    std::cout << "   m_x0:    " << m_x0 << std::endl;
    std::cout << "   m_y0:    " << m_y0 << std::endl;
    std::cout << "   m_tx0:    " << m_tx0  << std::endl;
    std::cout << "   m_ty0:    " << m_ty0  << std::endl;
    std::cout << "   m_q2:    " << m_q2  << std::endl;
    std::cout << "   StIDReco:    " << StIDReco  << std::endl;
  }
  rhits.push_back( RecoFP420(m_e,m_x0,m_y0,m_tx0,m_ty0,m_q2,StIDReco) );
  // ==============================  
  ///////////////////////////////////////
  return rhits;
  //============
}



