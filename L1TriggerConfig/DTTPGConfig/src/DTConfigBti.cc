//-------------------------------------------------
//
//   Class: DTConfigBti
//
//   Description: Configurable parameters and constants for Level1 Mu DT Trigger - BTI chip
//
//
//   Author List:
//   S.Vanini
//-----------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigBti.h"

//---------------
// C++ Headers --
//---------------
#include <cstdlib>
#include <string>  
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iomanip>
               
//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//----------------
// Constructors --
//----------------
DTConfigBti::DTConfigBti(const edm::ParameterSet& ps) { 

  m_ps = &ps;
  setDefaults();
  
}

//--------------
// Destructor --
//--------------
DTConfigBti::~DTConfigBti() {}

//--------------
// Operations --
//--------------

void
DTConfigBti::setDefaults() {

  // Debug flag 
  m_debug  = m_ps->getUntrackedParameter<int>("Debug");

  // Max K param accepted  
  m_kcut = m_ps->getParameter<int>("KMAX");
 
  // BTI angular acceptance in theta view
  m_kacctheta = m_ps->getParameter<int>("KACCTHETA");

  // BTI digi offset in tdc units
  m_digioffset = m_ps->getParameter<int>("DIGIOFFSET");

  // BTI setup time : fine syncronization  
  m_setuptime = m_ps->getParameter<int>("SINCROTIME");

  // Time indep. K equation suppression (XON) 
  m_xon = m_ps->getParameter<bool>("XON");
  // LTS and SET for low trigger suppression 
  m_lts = m_ps->getParameter<int>("LTS");
  m_set = m_ps->getParameter<int>("SET");
  // pattern acceptance AC1, AC2, ACH, ACL
  m_ac1 = m_ps->getParameter<int>("AC1");
  m_ac2 = m_ps->getParameter<int>("AC2");
  m_ach = m_ps->getParameter<int>("ACH");
  m_acl = m_ps->getParameter<int>("ACL");
  // redundant patterns flag RON
  m_ron = m_ps->getParameter<bool>("RON");

  // pattern masks
  for(int p=0; p<32; p++)
  {
        std::string label = "PTMS";
   	char patt0 = (p/10)+'0';
   	char patt1 = (p%10)+'0';
   	if ( patt0 != '0' )
     		label = label + patt0;
   	label = label + patt1;
	
  	m_pattmask[p]  = m_ps->getParameter<int>(label);
  }

  // wire masks
  for(int w=0; w<9; w++)
  {
        std::string label = "WEN";
   	char wname = w+'0';
   	label = label + wname;
	
  	m_wiremask[w]  = m_ps->getParameter<int>(label);
  }

  // angular window limits for traco
  m_ll = m_ps->getParameter<int>("LL");
  m_lh = m_ps->getParameter<int>("LH");
  m_cl = m_ps->getParameter<int>("CL");
  m_ch = m_ps->getParameter<int>("CH");
  m_rl = m_ps->getParameter<int>("RL");
  m_rh = m_ps->getParameter<int>("RH");
  // drift velocity parameter 4ST3
  m_4st3 = m_ps->getParameter<int>("ST43");
  // drift velocity parameter 4RE3
  m_4re3 = m_ps->getParameter<int>("RE43");
  // DEAD parameter
  m_dead = m_ps->getParameter<int>("DEAD");
}

void 
DTConfigBti::print() const {
  std::cout << "******************************************************************************" << std::endl;
  std::cout << "*              DTTrigger configuration : BTI chips                                 *" << std::endl;
  std::cout << "******************************************************************************" << std::endl;
  std::cout << "*                                                                            *" << std::endl;
  std::cout << "Debug flag : " <<  m_debug << std::endl;
  std::cout << "Max K param accepted : " << m_kcut << std::endl; 
  std::cout << "BTI angular acceptance in theta view : " << m_kacctheta << std::endl;
  std::cout << "BTI digi offset in tdc units : " << m_digioffset << std::endl;
  std::cout << "BTI setup time : fine syncronization : " << m_setuptime << std::endl;

  std::cout << "Time indep. K equation suppression (XON) : " << m_xon << std::endl;
  std::cout << "LTS for low trigger suppression : " << m_lts << std::endl;
  std::cout << "SET for low trigger suppression : " << m_set << std::endl;
  std::cout << "pattern acceptance AC1, AC2, ACH, ACL : " << 
	m_ac1 << ", " << m_ac2 << " , " << m_ach << ", " << m_acl << std::endl;
  std::cout << "redundant patterns flag RON : " << m_ron << std::endl;
  std::cout << "pattern masks : "; 
  for(int p=0; p<32; p++)
  	std::cout << m_pattmask[p] << " ";
  std::cout << std::endl;
 
  std::cout << "wire masks : "; 
  for(int w=0; w<9; w++)
  	std::cout << m_wiremask[w] << " ";
  std::cout << std::endl;

  std::cout << "angular window limits for traco : " << m_ll << ", " << m_lh << ", " 
	<< m_cl << ", " << m_ch << ", " << m_rl << ", " << m_rh << std::endl;
  std::cout << "drift velocity parameter 4ST3 : " << m_4st3 << std::endl;
  std::cout << "drift velocity parameter 4RE3 : " << m_4re3 << std::endl;  
  std::cout << "DEAD parameter : " << m_dead << std::endl;

  std::cout << "******************************************************************************" << std::endl;

}
