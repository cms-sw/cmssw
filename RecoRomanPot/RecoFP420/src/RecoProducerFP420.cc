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

  length         = conf_.getParameter<double>("BeamLineLength" );//m
  verbosity      = conf_.getUntrackedParameter<int>("VerbosityLevel");
  beam1filename  = conf_.getParameter<std::string>("Beam1");
  beam2filename  = conf_.getParameter<std::string>("Beam2");  

  edm::LogInfo ("RecoProducerFP420") << "RecoProducerFP420 parameters: \n" 
				     << "   Verbosity: " << verbosity << "\n"
				     << "   length:    " << length << "\n";
  if (verbosity > 1) {
    std::cout << "  RecoProducerFP420: constructor    " << std::endl;
    std::cout << "   BeamLineLength:    " << length << std::endl;
  }
  
  //  edm::FileInPath b1("SimTransport/HectorData/twiss_ip5_b1_v6.5.txt");
  //  edm::FileInPath b2("SimTransport/HectorData/twiss_ip5_b2_v6.5.txt");
  
  edm::FileInPath b1(beam1filename.c_str());
  edm::FileInPath b2(beam2filename.c_str());
  
  
  m_beamline1 = new H_BeamLine(  1, length + 0.1 ); // (direction, length)
  m_beamline2 = new H_BeamLine( -1, length + 0.1 ); //
  
  try {
    m_beamline1->fill( b1.fullPath(),  1, "IP5" );
    m_beamline2->fill( b2.fullPath(), -1, "IP5" );
  } catch ( const edm::Exception& e ) {
    std::string msg = e.what();
    msg += " caught in RecoProducerFP420... \nERROR: Could not locate SimTransport/HectorData data files.";
    edm::LogError ("DataNotFound") << msg;
  }
  
  m_beamline1->offsetElements( 120, -0.097 );
  m_beamline2->offsetElements( 120, +0.097 );
  
  m_beamline1->calcMatrix();
  m_beamline2->calcMatrix();

  edm::LogInfo ("RecoProducerFP420") << "==============================\n";

}

RecoProducerFP420::~RecoProducerFP420(){}

std::vector<RecoFP420>  RecoProducerFP420::reconstruct(int direction, double x1_420, double y1_420, double x2_420, double y2_420, double z1_420, double z2_420){
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
  if ( direction == 1 ) {
    m_rp420_f = new H_RecRPObject( z1_420*0.001, z2_420*0.001, *m_beamline1 );// m
    if ( m_rp420_f ) {
      if (verbosity >1) {
	std::cout << "  RecoProducerFP420: input coord. in um   " << std::endl;
	std::cout << "   x1_420:    " << x1_420 << "   y1_420:    " << y1_420 << std::endl;
	std::cout << "   x2_420:    " << x2_420 << "   y2_420:    " << y2_420 << std::endl;
      }
      m_rp420_f->setPositions( x1_420, y1_420, x2_420 ,y2_420 );//input coord. in um
      m_e   = m_rp420_f->getE( AM );// GeV
      //  std::cout << "   m_e1:    " << m_rp420_f->getE( TM ) << std::endl;
      //  std::cout << "   m_e2:    " << m_rp420_f->getE( AM ) << std::endl;
      m_tx0 = m_rp420_f->getTXIP();// urad
      m_ty0 = m_rp420_f->getTYIP();// urad
      m_x0  = m_rp420_f->getX0();// um
      m_y0  = m_rp420_f->getY0();// um
      m_q2  = m_rp420_f->getQ2();// GeV^2
    }// if ( m_rp420_f
  }// if ( dire
  else if ( direction == 2 ) {
    m_rp420_b = new H_RecRPObject( z1_420*0.001, z2_420*0.001, *m_beamline2 );// m
    if ( m_rp420_b ) {
      m_rp420_b->setPositions( x1_420, y1_420, x2_420 ,y2_420 );// input coord. in um
      m_e   = m_rp420_b->getE( AM );// GeV
      m_tx0 = m_rp420_b->getTXIP();// urad
      m_ty0 = m_rp420_b->getTYIP();// urad
      m_x0  = m_rp420_b->getX0();// um
      m_y0  = m_rp420_b->getY0();// um
      m_q2  = m_rp420_b->getQ2();// GeV^2
    }// if ( m_rp420_b
  }
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
    std::cout << "   direction:    " << direction  << std::endl;
  }
  rhits.push_back( RecoFP420(m_e,m_x0,m_y0,m_tx0,m_ty0,m_q2,direction) );
  // ==============================  
  ///////////////////////////////////////
    return rhits;
    //============
}



