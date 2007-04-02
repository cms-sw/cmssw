//-------------------------------------------------
//
//   Class: DTConfigTraco
//
//   Description: Configurable parameters and constants for Level1 Mu DT Trigger - TRACO chip
//
//
//   Author List:
//   S.Vanini
//-----------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigTraco.h"

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
DTConfigTraco::DTConfigTraco(const edm::ParameterSet& ps) { 

  m_ps = &ps;
  setDefaults();

}

//--------------
// Destructor --
//--------------
DTConfigTraco::~DTConfigTraco() {}

//--------------
// Operations --
//--------------

void
DTConfigTraco::setDefaults() {

  // Debug flag 
  m_debug = m_ps->getUntrackedParameter<int>("Debug");

  // KRAD traco parameter
  m_krad = m_ps->getParameter<int>("KRAD");

  // BTIC traco parameter
  m_btic = m_ps->getParameter<int>("BTIC");
 
  // DD traco parameter: this is fixed
  m_dd = m_ps->getParameter<int>("DD");

  // recycling of TRACO cand. in inner/outer SL : REUSEI/REUSEO
  m_reusei = m_ps->getParameter<int>("REUSEI");
  m_reuseo = m_ps->getParameter<int>("REUSEO");

  // single HTRIG enabling on first/second tracks F(S)HTMSK
  m_fhtmsk = m_ps->getParameter<int>("FHTMSK");
  m_shtmsk = m_ps->getParameter<int>("SHTMSK");

  // single LTRIG enabling on first/second tracks: F(S)LTMSK
  m_fltmsk = m_ps->getParameter<int>("FLTMSK");
  m_sltmsk = m_ps->getParameter<int>("SLTMSK");

  // preference to inner on first/second tracks: F(S)SLMSK
  m_fslmsk = m_ps->getParameter<int>("FSLMSK");
  m_sslmsk = m_ps->getParameter<int>("SSLMSK");

  // preference to HTRIG on first/second tracks: F(S)HTPRF
  m_fhtprf = m_ps->getParameter<int>("FHTPRF");
  m_shtprf = m_ps->getParameter<int>("SHTPRF");

  // ascend. order for K sorting first/second tracks: F(S)HISM
  m_fhism = m_ps->getParameter<int>("FHISM");
  m_shism = m_ps->getParameter<int>("SHISM");

  // K tollerance for correlation in TRACO: F(S)PRGCOMP
  m_fprgcomp = m_ps->getParameter<int>("FPRGCOMP");
  m_sprgcomp = m_ps->getParameter<int>("SPRGCOMP");

  // suppr. of LTRIG in 4 BX before HTRIG: LTS
  m_lts = m_ps->getParameter<int>("LTS");

  // single LTRIG accept enabling on first/second tracks LTF
  m_ltf = m_ps->getParameter<int>("LTF");

  // Connected bti in traco: bti mask
  for(int b=0; b<16; b++)
  {
        std::string label = "TRGENB";
   	char p0 = (b/10)+'0';
   	char p1 = (b%10)+'0';
   	if ( p0 != '0' )
     		label = label + p0;
   	label = label + p1;
	
  	m_trgenb[b]  = m_ps->getParameter<int>(label);
  }

  // IBTIOFF traco parameter
  m_ibtioff = m_ps->getParameter<int>("IBTIOFF");

  // bending angle cut for all stations and triggers : KPRGCOM
  m_kprgcom = m_ps->getParameter<int>("KPRGCOM");
 
  // flag for Low validation parameter
  m_lvalidifh =  m_ps->getParameter<int>("LVALIDIFH");
}

void 
DTConfigTraco::print() const {
  std::cout << "******************************************************************************" << std::endl;
  std::cout << "*              DTTrigger configuration : TRACO chips                                 *" << std::endl;
  std::cout << "******************************************************************************" << std::endl;
  std::cout << "*                                                                            *" << std::endl;
  std::cout << "Debug flag : " <<  m_debug << std::endl;
  std::cout << "KRAD traco parameter : " << m_krad << std::endl;
  std::cout << "BTIC traco parameter : " << m_btic << std::endl;
  std::cout << "DD traco parameter : " << m_dd << std::endl;
  std::cout << "REUSEI, REUSEO : " << m_reusei << ", " << m_reuseo << std::endl;
  std::cout << "FHTMSK, SHTMSK : " << m_fhtmsk << ", " << m_shtmsk << std::endl;
  std::cout << "FLTMSK, SLTMSK: " << m_fltmsk << ", " << m_sltmsk << std::endl;
  std::cout << "FSLMSK, SSLMSK : " << m_fslmsk << ", " << m_sslmsk << std::endl;
  std::cout << "FHTPRF, SHTPRF : " << m_fhtprf << ", " << m_fhtprf << std::endl;
  std::cout << "FHISM, SHISM : " << m_fhism << ", " << m_shism << std::endl;
  std::cout << "FPRGCOMP, SPRGCOMP : " << m_fprgcomp << ", " << m_sprgcomp << std::endl;
  std::cout << "LTS : " << m_lts << std::endl;
  std::cout << "LTF : " << m_ltf << std::endl;
  std::cout << "Connected bti in traco - bti mask : ";
  for(int b=0; b<16; b++)
  	std::cout << m_trgenb[b] << " "; 
  std::cout << std::endl;
  std::cout << "IBTIOFF : " << m_ibtioff << std::endl;
  std::cout << "bending angle cut : " << m_kprgcom << std::endl;
  std::cout << "flag for Low validation parameter : " << m_lvalidifh << std::endl;
  std::cout << "******************************************************************************" << std::endl;

}
