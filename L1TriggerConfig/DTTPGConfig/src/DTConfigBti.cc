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

//----------------
// Constructors --
//----------------
DTConfigBti::DTConfigBti(const edm::ParameterSet& ps) { 

  setDefaults(ps);
  
}

//--------------
// Destructor --
//--------------
DTConfigBti::~DTConfigBti() {}

//--------------
// Operations --
//--------------

void
DTConfigBti::setDefaults(const edm::ParameterSet& ps) {

  // Debug flag 
  m_debug  = ps.getUntrackedParameter<int>("Debug");

  // Max K param accepted  
  m_kcut = ps.getParameter<int>("KMAX");
 
  // BTI angular acceptance in theta view
  m_kacctheta = ps.getParameter<int>("KACCTHETA");

  // Time indep. K equation suppression (XON) 
  m_xon = ps.getParameter<bool>("XON");
  // LTS and SET for low trigger suppression 
  m_lts = ps.getParameter<int>("LTS");
  m_set = ps.getParameter<int>("SET");
  // pattern acceptance AC1, AC2, ACH, ACL
  m_ac1 = ps.getParameter<int>("AC1");
  m_ac2 = ps.getParameter<int>("AC2");
  m_ach = ps.getParameter<int>("ACH");
  m_acl = ps.getParameter<int>("ACL");
  // redundant patterns flag RON
  m_ron = ps.getParameter<bool>("RON");

  // pattern masks
  for(int p=0; p<32; p++)
  {
        std::string label = "PTMS";
   	char patt0 = (p/10)+'0';
   	char patt1 = (p%10)+'0';
   	if ( patt0 != '0' )
     		label = label + patt0;
   	label = label + patt1;
	
  	m_pattmask.set(p,ps.getParameter<int>(label));
  }

  // wire masks
  for(int w=0; w<9; w++)
  {
        std::string label = "WEN";
   	char wname = w+'0';
   	label = label + wname;
	
  	m_wiremask.set(w,ps.getParameter<int>(label));
  }

  // angular window limits for traco
  m_ll = ps.getParameter<int>("LL");
  m_lh = ps.getParameter<int>("LH");
  m_cl = ps.getParameter<int>("CL");
  m_ch = ps.getParameter<int>("CH");
  m_rl = ps.getParameter<int>("RL");
  m_rh = ps.getParameter<int>("RH");
  // drift velocity parameter 4ST3
  m_4st3 = ps.getParameter<int>("ST43");
  // drift velocity parameter 4RE3
  m_4re3 = ps.getParameter<int>("RE43");
  // DEAD parameter
  m_dead = ps.getParameter<int>("DEAD");
}

void 
DTConfigBti::print() const {
  std::cout << "******************************************************************************" << std::endl;
  std::cout << "*              DTTrigger configuration : BTI chips                                 *" << std::endl;
  std::cout << "******************************************************************************" << std::endl;
  std::cout << "*                                                                            *" << std::endl;
  std::cout << "Debug flag : " <<  debug() << std::endl;
  std::cout << "Max K param accepted : " << KCut() << std::endl; 
  std::cout << "BTI angular acceptance in theta view : " << KAccTheta() << std::endl;
  std::cout << "Time indep. K equation suppression (XON) : " << XON() << std::endl;
  std::cout << "LTS for low trigger suppression : " << LTS() << std::endl;
  std::cout << "SET for low trigger suppression : " << SET() << std::endl;
  std::cout << "pattern acceptance AC1, AC2, ACH, ACL : " << 
	AccPattAC1() << ", " << AccPattAC2() << " , " << AccPattACH() << ", " << AccPattACL() << std::endl;
  std::cout << "redundant patterns flag RON : " << RONflag() << std::endl;
  std::cout << "pattern masks : "; 
  for(int p=0; p<32; p++)
  	std::cout << PTMSflag(p) << " ";
  std::cout << std::endl;
 
  std::cout << "wire masks : "; 
  for(int w=1; w<=9; w++)
  	std::cout << WENflag(w) << " ";
  std::cout << std::endl;

  std::cout << "angular window limits for traco : " << LL() << ", " << LH() << ", " 
	<< CL() << ", " << CH() << ", " << RL() << ", " << RH() << std::endl;
  std::cout << "drift velocity parameter 4ST3 : " << ST43() << std::endl;
  std::cout << "drift velocity parameter 4RE3 : " << RE43() << std::endl;  
  std::cout << "DEAD parameter : " << DEADpar() << std::endl;

  std::cout << "******************************************************************************" << std::endl;

}
