#ifndef LUT_H
#define LUT_H

#include <iostream>
#include <fstream>
#include <string>
#include <math.h>

class Lut {

 public:

  Lut( int station, int board, int traco );
  ~Lut();

  int get_x( int addr );
  int get_k( int addr );

 private:

  int nStat;
  int nBoard;
  int nTraco;
  float* tracoPos;

  float SL_DIFF;
  float CELL_H;
  float CELL_PITCH;
  int ANGRESOL;
  int POSRESOL;

  float m_d;
  float m_ST;
  float m_Xc;
  float m_Xn;
  float m_shift;
  float m_stsize;
  float m_distp2;

};
#endif
