//-------------------------------------------------
//
//   Class: DTConfigBti
//
//   Description: Configurable parameters and constants for Level1 Mu DT Trigger - BTI chip
//
//
//   Author List:
//   S.Vanini
//
//   Modifications:
//   April,10th : set BTI parameters from string
//-----------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigBti.h"
#include "FWCore/Utilities/interface/Exception.h"

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

DTConfigBti::DTConfigBti(int debugBTI, unsigned short int * buffer) {

        m_debug = debugBTI;

 	// check if this is a BTI configuration string
	if (buffer[2]!=0x54){
		throw cms::Exception("DTTPG") << "===> ConfigBti constructor : not a BTI string!" << std::endl;
	}

	// decode
   	unsigned short int memory_bti[31];
   	for(int i=0;i<31;i++){
                memory_bti[i] = buffer[i+5];
                //std::cout << hex << memory_bti[i] << "  ";
        }
	int wmask[9];
        int st43 = memory_bti[0] & 0x3f;
	wmask[6] = memory_bti[1] & 0x01;
        wmask[7] = (memory_bti[1] >> 1 )& 0x01;
        wmask[8] = (memory_bti[1] >> 2 )& 0x01;
	int re43 = (memory_bti[1] >> 4 )& 0x03;
        wmask[0] = memory_bti[2] & 0x01;
        wmask[1] = (memory_bti[2] >> 1) & 0x01;
	wmask[2] = (memory_bti[2] >> 2 )& 0x01;
	wmask[3] = (memory_bti[2] >> 3 )& 0x01;
	wmask[4] = (memory_bti[2] >> 4 )& 0x01;
	wmask[5] = (memory_bti[2] >> 5 )& 0x01;
        int dead = memory_bti[3] & 0x3F;
        int LH =  memory_bti[4] & 0x3F;
        int LL =  memory_bti[5] & 0x3F;
        int CH =  memory_bti[6] & 0x3F;
        int CL =  memory_bti[7] & 0x3F;
        int RH =  memory_bti[8] & 0x3F;
        int RL =  memory_bti[9] & 0x3F;
        int tston = ( memory_bti[10] & 0x20 ) != 0 ;
        int test = ( memory_bti[10] & 0x10 ) != 0 ;
        int ten = ( memory_bti[10] & 0x8 ) != 0 ;
        int xon = ( memory_bti[10] & 0x2 ) != 0 ;
        int ron = ( memory_bti[10] & 0x1 ) != 0 ;
        int set = ( memory_bti[11] & 0x38 ) >> 3 ;
        int lts = ( memory_bti[11] & 0x6 ) >> 1 ;
        int ac1 = ( memory_bti[12] & 0x30 ) >> 4 ;
        int ac2 = ( memory_bti[12] & 0xc ) >> 2 ;
        int acl = ( memory_bti[12] & 0x3 ) ;
        int ach = ( memory_bti[13] & 0x30 ) >> 4 ;

	int pmask[32];
	for(int ir=0; ir<6; ir++)
	{
       		pmask[0+6*ir] = memory_bti[19-ir] & 0x01;
	       	pmask[1+6*ir] = (memory_bti[19-ir] >> 1 )& 0x01;	
		if(ir!=5)
		{
			pmask[2+6*ir] = (memory_bti[19-ir] >> 2 )& 0x01;		
			pmask[3+6*ir] = (memory_bti[19-ir] >> 3 )& 0x01;
			pmask[4+6*ir] = (memory_bti[19-ir] >> 4 )& 0x01;	
			pmask[5+6*ir] = (memory_bti[19-ir] >> 5 )& 0x01;
		}
	}	
	
	// dump
        if(debug()==1){
               std::cout << std::dec << "st43=" << st43
                << " re43=" << re43
                << " dead=" << dead
                << " LH=" << LH
                << " LL=" << LL
                << " CH=" << CH
                << " CL=" << CL
                << " RH=" << RH
                << " RL=" << RL
                << " tston=" << tston
                << " test=" << test
                << " ten=" << ten
                << " xon=" << xon
                << " ron=" << ron
                << " set=" << set
                << " lts=" << lts
                << " ac1=" << ac1
                << " ac2=" << ac2
                << " acl=" << acl
                << " ach=" << ach
                << std::endl;
		std::cout << std::dec << " wire masks= ";
		for(int iw=0; iw<9; iw++)
			std::cout << wmask[iw] << " ";
		std::cout << std::dec << "\n pattern masks= ";
		for(int ip=0; ip<32; ip++)
			std::cout << pmask[ip] << " ";
		std::cout << std::endl;
	}
	
	// set parameters
	// default for KCut and KAccTheta
  	setKCut(64);
  	setKAccTheta(1);

	for(int ip=0; ip<32; ip++)
  		setPTMSflag(pmask[ip],ip);

	for(int iw=0; iw<9; iw++)
  		setWENflag(wmask[iw],iw+1);

  	setST43(st43);
  	setRE43(re43);
  	setDEADpar(dead);
  	setLL(LL);
  	setLH(LH);
  	setCL(CL);
  	setCH(CH);
  	setRL(RL);
  	setRH(RH);
  	setXON(xon);
  	setRONflag(ron);
  	setSET(set);
  	setLTS(lts);
  	setAccPattAC1(ac1);
 	setAccPattAC2(ac2);
  	setAccPattACH(ach);
  	setAccPattACL(acl);
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
