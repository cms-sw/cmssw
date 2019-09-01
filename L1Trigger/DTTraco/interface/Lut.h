//---------------------------------------------------------
//
/**  \class Lut
 *
 *   Class for computing single Traco LUT from given parameters
 *
 *
 *   \author S. Vanini
 */
//
//---------------------------------------------------------
#ifndef LUT_H
#define LUT_H

#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigLUTs.h"

//---------------
// C++ Headers --
//---------------
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

#define ANGRESOL 512.0
#define POSRESOL 4096.0
#define SL_D 23.5       //(cm)SL Distance
#define CELL_PITCH 4.2  //(cm)Wire Distance
#define SL_DIFF 11.75   // (cm )Distance of SL from station center

class Lut {
public:
  Lut(){};
  Lut(const DTConfigLUTs *conf, int ntc, float SL_shift);
  ~Lut();

  // set lut parameters methods
  void setForTestBeam(int station, int board, int traco);

  // get luts
  int get_x(int addr) const;
  int get_k(int addr) const;

  // DSP <-> IEEE32 conversions
  void IEEE32toDSP(float f, short *DSPmantissa, short *DSPexp);
  void DSPtoIEEE32(short DSPmantissa, short DSPexp, float *f);

public:
  float m_d;    // distance vertex - normal
  float m_Xcn;  // Distance correlator - normal
  int m_ST;     // TRACO BTIC parameter
  int m_wheel;  // Wheel sign (+1 or -1)

private:
  float m_pitch_d_ST;  //=pitch/ST private:

  const DTConfigLUTs *_conf_luts;
};
#endif
