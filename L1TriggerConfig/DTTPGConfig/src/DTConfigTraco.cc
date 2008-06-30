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

//----------------
// Constructors --
//----------------
DTConfigTraco::DTConfigTraco(const edm::ParameterSet& ps) { 

  setDefaults(ps);

}

//--------------
// Destructor --
//--------------
DTConfigTraco::~DTConfigTraco() {}

//--------------
// Operations --
//--------------

void
DTConfigTraco::setDefaults(const edm::ParameterSet& ps) {

  // Debug flag 
  m_debug = ps.getUntrackedParameter<int>("Debug");

  // KRAD traco parameter
  m_krad = ps.getParameter<int>("KRAD");

  // BTIC traco parameter
  m_btic = ps.getParameter<int>("BTIC");
 
  // DD traco parameter: this is fixed
  m_dd = ps.getParameter<int>("DD");

  // recycling of TRACO cand. in inner/outer SL : REUSEI/REUSEO
  m_reusei = ps.getParameter<int>("REUSEI");
  m_reuseo = ps.getParameter<int>("REUSEO");

  // single HTRIG enabling on first/second tracks F(S)HTMSK
  m_fhtmsk = ps.getParameter<int>("FHTMSK");
  m_shtmsk = ps.getParameter<int>("SHTMSK");

  // single LTRIG enabling on first/second tracks: F(S)LTMSK
  m_fltmsk = ps.getParameter<int>("FLTMSK");
  m_sltmsk = ps.getParameter<int>("SLTMSK");

  // preference to inner on first/second tracks: F(S)SLMSK
  m_fslmsk = ps.getParameter<int>("FSLMSK");
  m_sslmsk = ps.getParameter<int>("SSLMSK");

  // preference to HTRIG on first/second tracks: F(S)HTPRF
  m_fhtprf = ps.getParameter<int>("FHTPRF");
  m_shtprf = ps.getParameter<int>("SHTPRF");

  // ascend. order for K sorting first/second tracks: F(S)HISM
  m_fhism = ps.getParameter<int>("FHISM");
  m_shism = ps.getParameter<int>("SHISM");

  // K tollerance for correlation in TRACO: F(S)PRGCOMP
  m_fprgcomp = ps.getParameter<int>("FPRGCOMP");
  m_sprgcomp = ps.getParameter<int>("SPRGCOMP");

  // suppr. of LTRIG in 4 BX before HTRIG: LTS
  m_lts = ps.getParameter<int>("LTS");

  // single LTRIG accept enabling on first/second tracks LTF
  m_ltf = ps.getParameter<int>("LTF");

  // Connected bti in traco: bti mask
  for(int b=0; b<16; b++)
  {
        std::string label = "TRGENB";
   	char p0 = (b/10)+'0';
   	char p1 = (b%10)+'0';
   	if ( p0 != '0' )
     		label = label + p0;
   	label = label + p1;
	
  	m_trgenb.set(b,ps.getParameter<int>(label));
  }

  // IBTIOFF traco parameter
  m_ibtioff = ps.getParameter<int>("IBTIOFF");

  // bending angle cut for all stations and triggers : KPRGCOM
  m_kprgcom = ps.getParameter<int>("KPRGCOM");
 
  // flag for Low validation parameter
  m_lvalidifh =  ps.getParameter<int>("LVALIDIFH");
}

void 
DTConfigTraco::print() const {
  std::cout << "******************************************************************************" << std::endl;
  std::cout << "*              DTTrigger configuration : TRACO chips                                 *" << std::endl;
  std::cout << "******************************************************************************" << std::endl;
  std::cout << "*                                                                            *" << std::endl;
  std::cout << "Debug flag : " << debug()  << std::endl;
  std::cout << "KRAD traco parameter : " << KRAD() << std::endl;
  std::cout << "BTIC traco parameter : " << BTIC() << std::endl;
  std::cout << "DD traco parameter : " << DD() << std::endl;
  std::cout << "REUSEI, REUSEO : " << TcReuse(0) << ", " << TcReuse(1) << std::endl;
  std::cout << "FHTMSK, SHTMSK : " << singleHflag(0) << ", " << singleHflag(1) << std::endl;
  std::cout << "FLTMSK, SLTMSK: " << singleLflag(0) << ", " << singleLflag(1) << std::endl;
  std::cout << "FSLMSK, SSLMSK : " << prefInner(0) << ", " << prefInner(1) << std::endl;
  std::cout << "FHTPRF, SHTPRF : " << prefHtrig(0) << ", " << prefHtrig(1) << std::endl;
  std::cout << "FHISM, SHISM : " << sortKascend(0) << ", " << sortKascend(1) << std::endl;
  std::cout << "FPRGCOMP, SPRGCOMP : " << TcKToll(0) << ", " << TcKToll(1) << std::endl;
  std::cout << "LTS : " << TcBxLts() << std::endl;
  std::cout << "LTF : " << singleLenab(0) << std::endl;
  std::cout << "Connected bti in traco - bti mask : ";
  for(int b=1; b<=16; b++)
  	std::cout << usedBti(b) << " "; 
  std::cout << std::endl;
  std::cout << "IBTIOFF : " << IBTIOFF() << std::endl;
  std::cout << "bending angle cut : " << BendingAngleCut() << std::endl;
  std::cout << "flag for Low validation parameter : " << LVALIDIFH() << std::endl;
  std::cout << "******************************************************************************" << std::endl;

}
