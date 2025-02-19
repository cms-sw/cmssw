//-------------------------------------------------
//
/**  \class DTConfigLUTs
 *
 *   Configurable parameters and constants 
 *   for Level-1 Muon DT Trigger - LUTs
 *
 *
 *   \author S. Vanini
 *
 */
//
//--------------------------------------------------
#ifndef DT_CONFIG_LUTs_H
#define DT_CONFIG_LUTs_H

//---------------
// C++ Headers --
//---------------
#include<iostream>

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfig.h"

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTConfigLUTs : public DTConfig {

  public:

  //! Constructor
  DTConfigLUTs(const edm::ParameterSet& ps);

  //! Empty Constructor
  DTConfigLUTs() {};

  //! Constructor from string
  DTConfigLUTs(bool debug, unsigned short int * buffer);

  //! Destructor
  ~DTConfigLUTs();

  //! Debug flag
  inline bool debug() const { return m_debug; }

  //! BTIC parameter
  inline int BTIC() const { return m_btic; }

  //! d: distance vertex to normal, unit cm. 
  inline float D() const { return  m_d; }
  
  //! Xcn: distance vertex to normal, unit cm. 
  inline float Xcn() const { return  m_Xcn; }
    
  //! wheel sign (-1 or +1)
  inline int Wheel() const { return  m_wheel; }

  //! Set single parameter functions
  inline void setDebug(bool debug) { m_debug=debug; }
  inline void setBTIC(int btic) { m_btic = btic; }
  inline void setD(float d)	{ m_d = d; }
  inline void setXCN(float Xcn) { m_Xcn = Xcn; } 
  inline void setWHEEL(int wheel) { m_wheel = wheel; } 
  
  //! Print the setup
  void print() const ;
  
  //!  DSP to IEEE32 conversion
  void DSPtoIEEE32(short DSPmantissa, short DSPexp, float *f);

  //!  IEEE32 to DSP conversion
  void IEEE32toDSP(float f, short int & DSPmantissa, short int & DSPexp);

 /*  //! Return pointer to parameter set */
/*   const edm::ParameterSet* getParameterSet() { return m_ps; } */

  private:

  //! Load pset values into class variables
  void setDefaults(const edm::ParameterSet& m_ps);

  bool m_debug;
  int m_btic;
  float m_d;
  float m_Xcn;
  int m_wheel;
};

#endif
