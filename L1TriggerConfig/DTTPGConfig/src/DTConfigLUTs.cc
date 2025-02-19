//-------------------------------------------------
//
//   Class: DTConfigLUTs
//
//   Description: Configurable parameters and constants 
//   for Level1 Mu DT Trigger - LUTs
//
//
//   Author List:
//   S. Vanini
//
//-----------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigLUTs.h"

//---------------
// C++ Headers --
//---------------
#include <math.h> 
#include <cstring>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/Utilities/interface/Exception.h"

//----------------
// Constructors --
//----------------
DTConfigLUTs::DTConfigLUTs(const edm::ParameterSet& ps) { 
  setDefaults(ps);
}

DTConfigLUTs::DTConfigLUTs(bool debugLUTS, unsigned short int * buffer) {

  m_debug = debugLUTS;

  // check if this is a LUT configuration string
  if (buffer[2]!=0xA8){
  	throw cms::Exception("DTTPG") << "===> ConfigLUTs constructor : not a LUT string!" << std::endl;
  }
   
  // decode
  short int memory_lut[7];
  int c=3;
  for(int i=0;i<7;i++){
  	memory_lut[i] = (buffer[c]<<8) | buffer[c+1];
	c += 2;
        //std::cout << hex << memory_bti[i] << "  ";
  }

  // decode
  int btic = memory_lut[0];
  float d;
  DSPtoIEEE32( memory_lut[1],  memory_lut[2], &d );
  float Xcn;     
  DSPtoIEEE32( memory_lut[3],  memory_lut[4], &Xcn );
  int wheel = memory_lut[5];

  // set parameters 
  setBTIC(btic);
  setD(d);
  setXCN(Xcn);
  setWHEEL(wheel);
  
  return; 
}

//--------------
// Destructor --
//--------------
DTConfigLUTs::~DTConfigLUTs() {}

//--------------
// Operations --
//--------------

void
DTConfigLUTs::setDefaults(const edm::ParameterSet& m_ps) {

  // Debug flag 
  m_debug = m_ps.getUntrackedParameter<bool>("Debug");

  // BTIC parameter
  m_btic = m_ps.getUntrackedParameter<int>("BTIC");

  // d: distance vertex to normal, unit cm. 
  m_d = m_ps.getUntrackedParameter<double>("D");
  
  // Xcn: distance vertex to normal, unit cm. 
  m_Xcn = m_ps.getUntrackedParameter<double>("XCN");
    
  // wheel sign (-1 or +1)
  m_wheel = m_ps.getUntrackedParameter<int>("WHEEL");
}

void 
DTConfigLUTs::print() const {

  std::cout << "******************************************************************************" << std::endl;
  std::cout << "*              DTTrigger configuration : LUT parameters             *" << std::endl;
  std::cout << "******************************************************************************" << std::endl << std::endl;
  std::cout << "Debug flag : " <<  debug()     << std::endl;
  std::cout << "BTIC parameter : " << m_btic << std::endl;
  std::cout << "d: distance vertex to normal, unit cm. " << m_d << std::endl;
  std::cout << "Xcn: distance vertex to normal, unit cm. " << m_Xcn << std::endl;
  std::cout << "wheel sign " << m_wheel << std::endl;
  std::cout << "******************************************************************************" << std::endl;
}

void 
DTConfigLUTs::DSPtoIEEE32(short DSPmantissa, short DSPexp, float *f)
{
  DSPexp -= 15;
  *f = DSPmantissa * (float)pow( 2.0, DSPexp );
  return;
}


void
DTConfigLUTs::IEEE32toDSP(float f, short int & DSPmantissa, short int & DSPexp)
{
  long int *pl=0, lm;
  bool sign=false;

  DSPmantissa = 0;
  DSPexp = 0;

  if( f!=0.0 )
  {
        //pl = (long *)&f;
	memcpy(pl,&f,sizeof(float));
        if((*pl & 0x80000000)!=0)
                sign=true;
        lm = ( 0x800000 | (*pl & 0x7FFFFF)); // [1][23bit mantissa]
        lm >>= 9; //reduce to 15bits
        lm &= 0x7FFF;
        DSPexp = ((*pl>>23)&0xFF)-126;
        DSPmantissa = (short)lm;
        if(sign)
                DSPmantissa = - DSPmantissa;  // convert negative value in 2.s complement

  }
  return;
}




