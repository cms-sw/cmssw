// -*- C++ -*-
//
// This file is a part of the CLHEP - a Class Library for High Energy Physics.
// 
// This software written by Mark Fischler and Steven Haywood
// 

#include <cmath>

#include "SprSymMatrix.hh"

double SprSymMatrix::posDefFraction5x5 = 1.0;
double SprSymMatrix::posDefFraction6x6 = 1.0;
double SprSymMatrix::adjustment5x5 = 0.0;
double SprSymMatrix::adjustment6x6 = 0.0;
const double SprSymMatrix::CHOLESKY_THRESHOLD_5x5 = .5;
const double SprSymMatrix::CHOLESKY_THRESHOLD_6x6 = .2;
const double SprSymMatrix::CHOLESKY_CREEP_5x5 = .005;
const double SprSymMatrix::CHOLESKY_CREEP_6x6 = .002;

// Aij are indices for a 6x6 symmetric matrix.
//     The indices for 5x5 or 4x4 symmetric matrices are the same, 
//     ignoring all combinations with an index which is inapplicable.

#define A00 0
#define A01 1
#define A02 3
#define A03 6
#define A04 10
#define A05 15

#define A10 1
#define A11 2
#define A12 4
#define A13 7
#define A14 11
#define A15 16

#define A20 3
#define A21 4
#define A22 5
#define A23 8
#define A24 12
#define A25 17

#define A30 6
#define A31 7
#define A32 8
#define A33 9
#define A34 13
#define A35 18

#define A40 10
#define A41 11
#define A42 12
#define A43 13
#define A44 14
#define A45 19

#define A50 15
#define A51 16
#define A52 17
#define A53 18
#define A54 19
#define A55 20


void SprSymMatrix::invert5(int & ifail) { 
      if (posDefFraction5x5 >= CHOLESKY_THRESHOLD_5x5) {
	invertCholesky5(ifail);
        posDefFraction5x5 = .9*posDefFraction5x5 + .1*(1-ifail);
        if (ifail!=0) {  // Cholesky failed -- invert using Haywood
          invertHaywood5(ifail);      
        }
      } else {
        if (posDefFraction5x5 + adjustment5x5 >= CHOLESKY_THRESHOLD_5x5) {
	  invertCholesky5(ifail);
          posDefFraction5x5 = .9*posDefFraction5x5 + .1*(1-ifail);
          if (ifail!=0) {  // Cholesky failed -- invert using Haywood
            invertHaywood5(ifail);      
	    adjustment5x5 = 0;
	  }
        } else {
          invertHaywood5(ifail);      
	  adjustment5x5 += CHOLESKY_CREEP_5x5;
        }
      }
      return;
}

void SprSymMatrix::invert6(int & ifail) { 
      if (posDefFraction6x6 >= CHOLESKY_THRESHOLD_6x6) {
	invertCholesky6(ifail);
        posDefFraction6x6 = .9*posDefFraction6x6 + .1*(1-ifail);
        if (ifail!=0) {  // Cholesky failed -- invert using Haywood
          invertHaywood6(ifail);      
        }
      } else {
        if (posDefFraction6x6 + adjustment6x6 >= CHOLESKY_THRESHOLD_6x6) {
	  invertCholesky6(ifail);
          posDefFraction6x6 = .9*posDefFraction6x6 + .1*(1-ifail);
          if (ifail!=0) {  // Cholesky failed -- invert using Haywood
            invertHaywood6(ifail);      
	    adjustment6x6 = 0;
	  }
        } else {
          invertHaywood6(ifail);      
	  adjustment6x6 += CHOLESKY_CREEP_6x6;
        }
      }
      return;
}


void SprSymMatrix::invertHaywood5  (int & ifail) {

  ifail = 0;

  // Find all NECESSARY 2x2 dets:  (25 of them)

  double Det2_23_01 = m[A20]*m[A31] - m[A21]*m[A30];
  double Det2_23_02 = m[A20]*m[A32] - m[A22]*m[A30];
  double Det2_23_03 = m[A20]*m[A33] - m[A23]*m[A30];
  double Det2_23_12 = m[A21]*m[A32] - m[A22]*m[A31];
  double Det2_23_13 = m[A21]*m[A33] - m[A23]*m[A31];
  double Det2_23_23 = m[A22]*m[A33] - m[A23]*m[A32];
  double Det2_24_01 = m[A20]*m[A41] - m[A21]*m[A40];
  double Det2_24_02 = m[A20]*m[A42] - m[A22]*m[A40];
  double Det2_24_03 = m[A20]*m[A43] - m[A23]*m[A40];
  double Det2_24_04 = m[A20]*m[A44] - m[A24]*m[A40];
  double Det2_24_12 = m[A21]*m[A42] - m[A22]*m[A41];
  double Det2_24_13 = m[A21]*m[A43] - m[A23]*m[A41];
  double Det2_24_14 = m[A21]*m[A44] - m[A24]*m[A41];
  double Det2_24_23 = m[A22]*m[A43] - m[A23]*m[A42];
  double Det2_24_24 = m[A22]*m[A44] - m[A24]*m[A42];
  double Det2_34_01 = m[A30]*m[A41] - m[A31]*m[A40];
  double Det2_34_02 = m[A30]*m[A42] - m[A32]*m[A40];
  double Det2_34_03 = m[A30]*m[A43] - m[A33]*m[A40];
  double Det2_34_04 = m[A30]*m[A44] - m[A34]*m[A40];
  double Det2_34_12 = m[A31]*m[A42] - m[A32]*m[A41];
  double Det2_34_13 = m[A31]*m[A43] - m[A33]*m[A41];
  double Det2_34_14 = m[A31]*m[A44] - m[A34]*m[A41];
  double Det2_34_23 = m[A32]*m[A43] - m[A33]*m[A42];
  double Det2_34_24 = m[A32]*m[A44] - m[A34]*m[A42];
  double Det2_34_34 = m[A33]*m[A44] - m[A34]*m[A43];

  // Find all NECESSARY 3x3 dets:   (30 of them)

  double Det3_123_012 = m[A10]*Det2_23_12 - m[A11]*Det2_23_02 
				+ m[A12]*Det2_23_01;
  double Det3_123_013 = m[A10]*Det2_23_13 - m[A11]*Det2_23_03 
				+ m[A13]*Det2_23_01;
  double Det3_123_023 = m[A10]*Det2_23_23 - m[A12]*Det2_23_03 
				+ m[A13]*Det2_23_02;
  double Det3_123_123 = m[A11]*Det2_23_23 - m[A12]*Det2_23_13 
				+ m[A13]*Det2_23_12;
  double Det3_124_012 = m[A10]*Det2_24_12 - m[A11]*Det2_24_02 
				+ m[A12]*Det2_24_01;
  double Det3_124_013 = m[A10]*Det2_24_13 - m[A11]*Det2_24_03 
				+ m[A13]*Det2_24_01;
  double Det3_124_014 = m[A10]*Det2_24_14 - m[A11]*Det2_24_04 
				+ m[A14]*Det2_24_01;
  double Det3_124_023 = m[A10]*Det2_24_23 - m[A12]*Det2_24_03 
				+ m[A13]*Det2_24_02;
  double Det3_124_024 = m[A10]*Det2_24_24 - m[A12]*Det2_24_04 
				+ m[A14]*Det2_24_02;
  double Det3_124_123 = m[A11]*Det2_24_23 - m[A12]*Det2_24_13 
				+ m[A13]*Det2_24_12;
  double Det3_124_124 = m[A11]*Det2_24_24 - m[A12]*Det2_24_14 
				+ m[A14]*Det2_24_12;
  double Det3_134_012 = m[A10]*Det2_34_12 - m[A11]*Det2_34_02 
				+ m[A12]*Det2_34_01;
  double Det3_134_013 = m[A10]*Det2_34_13 - m[A11]*Det2_34_03 
				+ m[A13]*Det2_34_01;
  double Det3_134_014 = m[A10]*Det2_34_14 - m[A11]*Det2_34_04 
				+ m[A14]*Det2_34_01;
  double Det3_134_023 = m[A10]*Det2_34_23 - m[A12]*Det2_34_03 
				+ m[A13]*Det2_34_02;
  double Det3_134_024 = m[A10]*Det2_34_24 - m[A12]*Det2_34_04 
				+ m[A14]*Det2_34_02;
  double Det3_134_034 = m[A10]*Det2_34_34 - m[A13]*Det2_34_04 
				+ m[A14]*Det2_34_03;
  double Det3_134_123 = m[A11]*Det2_34_23 - m[A12]*Det2_34_13 
				+ m[A13]*Det2_34_12;
  double Det3_134_124 = m[A11]*Det2_34_24 - m[A12]*Det2_34_14 
				+ m[A14]*Det2_34_12;
  double Det3_134_134 = m[A11]*Det2_34_34 - m[A13]*Det2_34_14 
				+ m[A14]*Det2_34_13;
  double Det3_234_012 = m[A20]*Det2_34_12 - m[A21]*Det2_34_02 
				+ m[A22]*Det2_34_01;
  double Det3_234_013 = m[A20]*Det2_34_13 - m[A21]*Det2_34_03 
				+ m[A23]*Det2_34_01;
  double Det3_234_014 = m[A20]*Det2_34_14 - m[A21]*Det2_34_04 
				+ m[A24]*Det2_34_01;
  double Det3_234_023 = m[A20]*Det2_34_23 - m[A22]*Det2_34_03 
				+ m[A23]*Det2_34_02;
  double Det3_234_024 = m[A20]*Det2_34_24 - m[A22]*Det2_34_04 
				+ m[A24]*Det2_34_02;
  double Det3_234_034 = m[A20]*Det2_34_34 - m[A23]*Det2_34_04 
				+ m[A24]*Det2_34_03;
  double Det3_234_123 = m[A21]*Det2_34_23 - m[A22]*Det2_34_13 
				+ m[A23]*Det2_34_12;
  double Det3_234_124 = m[A21]*Det2_34_24 - m[A22]*Det2_34_14 
				+ m[A24]*Det2_34_12;
  double Det3_234_134 = m[A21]*Det2_34_34 - m[A23]*Det2_34_14 
				+ m[A24]*Det2_34_13;
  double Det3_234_234 = m[A22]*Det2_34_34 - m[A23]*Det2_34_24 
				+ m[A24]*Det2_34_23;

  // Find all NECESSARY 4x4 dets:   (15 of them)

  double Det4_0123_0123 = m[A00]*Det3_123_123 - m[A01]*Det3_123_023 
				+ m[A02]*Det3_123_013 - m[A03]*Det3_123_012;
  double Det4_0124_0123 = m[A00]*Det3_124_123 - m[A01]*Det3_124_023 
				+ m[A02]*Det3_124_013 - m[A03]*Det3_124_012;
  double Det4_0124_0124 = m[A00]*Det3_124_124 - m[A01]*Det3_124_024 
				+ m[A02]*Det3_124_014 - m[A04]*Det3_124_012;
  double Det4_0134_0123 = m[A00]*Det3_134_123 - m[A01]*Det3_134_023 
				+ m[A02]*Det3_134_013 - m[A03]*Det3_134_012;
  double Det4_0134_0124 = m[A00]*Det3_134_124 - m[A01]*Det3_134_024 
				+ m[A02]*Det3_134_014 - m[A04]*Det3_134_012;
  double Det4_0134_0134 = m[A00]*Det3_134_134 - m[A01]*Det3_134_034 
				+ m[A03]*Det3_134_014 - m[A04]*Det3_134_013;
  double Det4_0234_0123 = m[A00]*Det3_234_123 - m[A01]*Det3_234_023 
				+ m[A02]*Det3_234_013 - m[A03]*Det3_234_012;
  double Det4_0234_0124 = m[A00]*Det3_234_124 - m[A01]*Det3_234_024 
				+ m[A02]*Det3_234_014 - m[A04]*Det3_234_012;
  double Det4_0234_0134 = m[A00]*Det3_234_134 - m[A01]*Det3_234_034 
				+ m[A03]*Det3_234_014 - m[A04]*Det3_234_013;
  double Det4_0234_0234 = m[A00]*Det3_234_234 - m[A02]*Det3_234_034 
				+ m[A03]*Det3_234_024 - m[A04]*Det3_234_023;
  double Det4_1234_0123 = m[A10]*Det3_234_123 - m[A11]*Det3_234_023 
				+ m[A12]*Det3_234_013 - m[A13]*Det3_234_012;
  double Det4_1234_0124 = m[A10]*Det3_234_124 - m[A11]*Det3_234_024 
				+ m[A12]*Det3_234_014 - m[A14]*Det3_234_012;
  double Det4_1234_0134 = m[A10]*Det3_234_134 - m[A11]*Det3_234_034 
				+ m[A13]*Det3_234_014 - m[A14]*Det3_234_013;
  double Det4_1234_0234 = m[A10]*Det3_234_234 - m[A12]*Det3_234_034 
				+ m[A13]*Det3_234_024 - m[A14]*Det3_234_023;
  double Det4_1234_1234 = m[A11]*Det3_234_234 - m[A12]*Det3_234_134 
				+ m[A13]*Det3_234_124 - m[A14]*Det3_234_123;

  // Find the 5x5 det:

  double det =    m[A00]*Det4_1234_1234 
	 	- m[A01]*Det4_1234_0234 
		+ m[A02]*Det4_1234_0134 
		- m[A03]*Det4_1234_0124 
		+ m[A04]*Det4_1234_0123;

  if ( det == 0 ) {  
    ifail = 1;
    return;
  } 

  double oneOverDet = 1.0/det;
  double mn1OverDet = - oneOverDet;

  m[A00] =  Det4_1234_1234 * oneOverDet;
  m[A01] =  Det4_1234_0234 * mn1OverDet;
  m[A02] =  Det4_1234_0134 * oneOverDet;
  m[A03] =  Det4_1234_0124 * mn1OverDet;
  m[A04] =  Det4_1234_0123 * oneOverDet;

  m[A11] =  Det4_0234_0234 * oneOverDet;
  m[A12] =  Det4_0234_0134 * mn1OverDet;
  m[A13] =  Det4_0234_0124 * oneOverDet;
  m[A14] =  Det4_0234_0123 * mn1OverDet;

  m[A22] =  Det4_0134_0134 * oneOverDet;
  m[A23] =  Det4_0134_0124 * mn1OverDet;
  m[A24] =  Det4_0134_0123 * oneOverDet;

  m[A33] =  Det4_0124_0124 * oneOverDet;
  m[A34] =  Det4_0124_0123 * mn1OverDet;

  m[A44] =  Det4_0123_0123 * oneOverDet;

  return;
}

void SprSymMatrix::invertHaywood6  (int & ifail) {

  ifail = 0;

  // Find all NECESSARY 2x2 dets:  (39 of them)

  double Det2_34_01 = m[A30]*m[A41] - m[A31]*m[A40];
  double Det2_34_02 = m[A30]*m[A42] - m[A32]*m[A40];
  double Det2_34_03 = m[A30]*m[A43] - m[A33]*m[A40];
  double Det2_34_04 = m[A30]*m[A44] - m[A34]*m[A40];
  double Det2_34_12 = m[A31]*m[A42] - m[A32]*m[A41];
  double Det2_34_13 = m[A31]*m[A43] - m[A33]*m[A41];
  double Det2_34_14 = m[A31]*m[A44] - m[A34]*m[A41];
  double Det2_34_23 = m[A32]*m[A43] - m[A33]*m[A42];
  double Det2_34_24 = m[A32]*m[A44] - m[A34]*m[A42];
  double Det2_34_34 = m[A33]*m[A44] - m[A34]*m[A43];
  double Det2_35_01 = m[A30]*m[A51] - m[A31]*m[A50];
  double Det2_35_02 = m[A30]*m[A52] - m[A32]*m[A50];
  double Det2_35_03 = m[A30]*m[A53] - m[A33]*m[A50];
  double Det2_35_04 = m[A30]*m[A54] - m[A34]*m[A50];
  double Det2_35_05 = m[A30]*m[A55] - m[A35]*m[A50];
  double Det2_35_12 = m[A31]*m[A52] - m[A32]*m[A51];
  double Det2_35_13 = m[A31]*m[A53] - m[A33]*m[A51];
  double Det2_35_14 = m[A31]*m[A54] - m[A34]*m[A51];
  double Det2_35_15 = m[A31]*m[A55] - m[A35]*m[A51];
  double Det2_35_23 = m[A32]*m[A53] - m[A33]*m[A52];
  double Det2_35_24 = m[A32]*m[A54] - m[A34]*m[A52];
  double Det2_35_25 = m[A32]*m[A55] - m[A35]*m[A52];
  double Det2_35_34 = m[A33]*m[A54] - m[A34]*m[A53];
  double Det2_35_35 = m[A33]*m[A55] - m[A35]*m[A53];
  double Det2_45_01 = m[A40]*m[A51] - m[A41]*m[A50];
  double Det2_45_02 = m[A40]*m[A52] - m[A42]*m[A50];
  double Det2_45_03 = m[A40]*m[A53] - m[A43]*m[A50];
  double Det2_45_04 = m[A40]*m[A54] - m[A44]*m[A50];
  double Det2_45_05 = m[A40]*m[A55] - m[A45]*m[A50];
  double Det2_45_12 = m[A41]*m[A52] - m[A42]*m[A51];
  double Det2_45_13 = m[A41]*m[A53] - m[A43]*m[A51];
  double Det2_45_14 = m[A41]*m[A54] - m[A44]*m[A51];
  double Det2_45_15 = m[A41]*m[A55] - m[A45]*m[A51];
  double Det2_45_23 = m[A42]*m[A53] - m[A43]*m[A52];
  double Det2_45_24 = m[A42]*m[A54] - m[A44]*m[A52];
  double Det2_45_25 = m[A42]*m[A55] - m[A45]*m[A52];
  double Det2_45_34 = m[A43]*m[A54] - m[A44]*m[A53];
  double Det2_45_35 = m[A43]*m[A55] - m[A45]*m[A53];
  double Det2_45_45 = m[A44]*m[A55] - m[A45]*m[A54];

  // Find all NECESSARY 3x3 dets:  (65 of them)

  double Det3_234_012 = m[A20]*Det2_34_12 - m[A21]*Det2_34_02 
						+ m[A22]*Det2_34_01;
  double Det3_234_013 = m[A20]*Det2_34_13 - m[A21]*Det2_34_03 
						+ m[A23]*Det2_34_01;
  double Det3_234_014 = m[A20]*Det2_34_14 - m[A21]*Det2_34_04 
						+ m[A24]*Det2_34_01;
  double Det3_234_023 = m[A20]*Det2_34_23 - m[A22]*Det2_34_03 
						+ m[A23]*Det2_34_02;
  double Det3_234_024 = m[A20]*Det2_34_24 - m[A22]*Det2_34_04 
						+ m[A24]*Det2_34_02;
  double Det3_234_034 = m[A20]*Det2_34_34 - m[A23]*Det2_34_04 
						+ m[A24]*Det2_34_03;
  double Det3_234_123 = m[A21]*Det2_34_23 - m[A22]*Det2_34_13 
						+ m[A23]*Det2_34_12;
  double Det3_234_124 = m[A21]*Det2_34_24 - m[A22]*Det2_34_14 
						+ m[A24]*Det2_34_12;
  double Det3_234_134 = m[A21]*Det2_34_34 - m[A23]*Det2_34_14 
						+ m[A24]*Det2_34_13;
  double Det3_234_234 = m[A22]*Det2_34_34 - m[A23]*Det2_34_24 
						+ m[A24]*Det2_34_23;
  double Det3_235_012 = m[A20]*Det2_35_12 - m[A21]*Det2_35_02 
						+ m[A22]*Det2_35_01;
  double Det3_235_013 = m[A20]*Det2_35_13 - m[A21]*Det2_35_03 
						+ m[A23]*Det2_35_01;
  double Det3_235_014 = m[A20]*Det2_35_14 - m[A21]*Det2_35_04 
						+ m[A24]*Det2_35_01;
  double Det3_235_015 = m[A20]*Det2_35_15 - m[A21]*Det2_35_05 
						+ m[A25]*Det2_35_01;
  double Det3_235_023 = m[A20]*Det2_35_23 - m[A22]*Det2_35_03 
						+ m[A23]*Det2_35_02;
  double Det3_235_024 = m[A20]*Det2_35_24 - m[A22]*Det2_35_04 
						+ m[A24]*Det2_35_02;
  double Det3_235_025 = m[A20]*Det2_35_25 - m[A22]*Det2_35_05 
						+ m[A25]*Det2_35_02;
  double Det3_235_034 = m[A20]*Det2_35_34 - m[A23]*Det2_35_04 
						+ m[A24]*Det2_35_03;
  double Det3_235_035 = m[A20]*Det2_35_35 - m[A23]*Det2_35_05 
						+ m[A25]*Det2_35_03;
  double Det3_235_123 = m[A21]*Det2_35_23 - m[A22]*Det2_35_13 
						+ m[A23]*Det2_35_12;
  double Det3_235_124 = m[A21]*Det2_35_24 - m[A22]*Det2_35_14 
						+ m[A24]*Det2_35_12;
  double Det3_235_125 = m[A21]*Det2_35_25 - m[A22]*Det2_35_15 
						+ m[A25]*Det2_35_12;
  double Det3_235_134 = m[A21]*Det2_35_34 - m[A23]*Det2_35_14 
						+ m[A24]*Det2_35_13;
  double Det3_235_135 = m[A21]*Det2_35_35 - m[A23]*Det2_35_15 
						+ m[A25]*Det2_35_13;
  double Det3_235_234 = m[A22]*Det2_35_34 - m[A23]*Det2_35_24 
						+ m[A24]*Det2_35_23;
  double Det3_235_235 = m[A22]*Det2_35_35 - m[A23]*Det2_35_25 
						+ m[A25]*Det2_35_23;
  double Det3_245_012 = m[A20]*Det2_45_12 - m[A21]*Det2_45_02 
						+ m[A22]*Det2_45_01;
  double Det3_245_013 = m[A20]*Det2_45_13 - m[A21]*Det2_45_03 
						+ m[A23]*Det2_45_01;
  double Det3_245_014 = m[A20]*Det2_45_14 - m[A21]*Det2_45_04 
						+ m[A24]*Det2_45_01;
  double Det3_245_015 = m[A20]*Det2_45_15 - m[A21]*Det2_45_05 
						+ m[A25]*Det2_45_01;
  double Det3_245_023 = m[A20]*Det2_45_23 - m[A22]*Det2_45_03 
						+ m[A23]*Det2_45_02;
  double Det3_245_024 = m[A20]*Det2_45_24 - m[A22]*Det2_45_04 
						+ m[A24]*Det2_45_02;
  double Det3_245_025 = m[A20]*Det2_45_25 - m[A22]*Det2_45_05 
						+ m[A25]*Det2_45_02;
  double Det3_245_034 = m[A20]*Det2_45_34 - m[A23]*Det2_45_04 
						+ m[A24]*Det2_45_03;
  double Det3_245_035 = m[A20]*Det2_45_35 - m[A23]*Det2_45_05 
						+ m[A25]*Det2_45_03;
  double Det3_245_045 = m[A20]*Det2_45_45 - m[A24]*Det2_45_05 
						+ m[A25]*Det2_45_04;
  double Det3_245_123 = m[A21]*Det2_45_23 - m[A22]*Det2_45_13 
						+ m[A23]*Det2_45_12;
  double Det3_245_124 = m[A21]*Det2_45_24 - m[A22]*Det2_45_14 
						+ m[A24]*Det2_45_12;
  double Det3_245_125 = m[A21]*Det2_45_25 - m[A22]*Det2_45_15 
						+ m[A25]*Det2_45_12;
  double Det3_245_134 = m[A21]*Det2_45_34 - m[A23]*Det2_45_14 
						+ m[A24]*Det2_45_13;
  double Det3_245_135 = m[A21]*Det2_45_35 - m[A23]*Det2_45_15 
						+ m[A25]*Det2_45_13;
  double Det3_245_145 = m[A21]*Det2_45_45 - m[A24]*Det2_45_15 
						+ m[A25]*Det2_45_14;
  double Det3_245_234 = m[A22]*Det2_45_34 - m[A23]*Det2_45_24 
						+ m[A24]*Det2_45_23;
  double Det3_245_235 = m[A22]*Det2_45_35 - m[A23]*Det2_45_25 
						+ m[A25]*Det2_45_23;
  double Det3_245_245 = m[A22]*Det2_45_45 - m[A24]*Det2_45_25 
						+ m[A25]*Det2_45_24;
  double Det3_345_012 = m[A30]*Det2_45_12 - m[A31]*Det2_45_02 
						+ m[A32]*Det2_45_01;
  double Det3_345_013 = m[A30]*Det2_45_13 - m[A31]*Det2_45_03 
						+ m[A33]*Det2_45_01;
  double Det3_345_014 = m[A30]*Det2_45_14 - m[A31]*Det2_45_04 
						+ m[A34]*Det2_45_01;
  double Det3_345_015 = m[A30]*Det2_45_15 - m[A31]*Det2_45_05 
						+ m[A35]*Det2_45_01;
  double Det3_345_023 = m[A30]*Det2_45_23 - m[A32]*Det2_45_03 
						+ m[A33]*Det2_45_02;
  double Det3_345_024 = m[A30]*Det2_45_24 - m[A32]*Det2_45_04 
						+ m[A34]*Det2_45_02;
  double Det3_345_025 = m[A30]*Det2_45_25 - m[A32]*Det2_45_05 
						+ m[A35]*Det2_45_02;
  double Det3_345_034 = m[A30]*Det2_45_34 - m[A33]*Det2_45_04 
						+ m[A34]*Det2_45_03;
  double Det3_345_035 = m[A30]*Det2_45_35 - m[A33]*Det2_45_05 
						+ m[A35]*Det2_45_03;
  double Det3_345_045 = m[A30]*Det2_45_45 - m[A34]*Det2_45_05 
						+ m[A35]*Det2_45_04;
  double Det3_345_123 = m[A31]*Det2_45_23 - m[A32]*Det2_45_13 
						+ m[A33]*Det2_45_12;
  double Det3_345_124 = m[A31]*Det2_45_24 - m[A32]*Det2_45_14 
						+ m[A34]*Det2_45_12;
  double Det3_345_125 = m[A31]*Det2_45_25 - m[A32]*Det2_45_15 
						+ m[A35]*Det2_45_12;
  double Det3_345_134 = m[A31]*Det2_45_34 - m[A33]*Det2_45_14 
						+ m[A34]*Det2_45_13;
  double Det3_345_135 = m[A31]*Det2_45_35 - m[A33]*Det2_45_15 
						+ m[A35]*Det2_45_13;
  double Det3_345_145 = m[A31]*Det2_45_45 - m[A34]*Det2_45_15 
						+ m[A35]*Det2_45_14;
  double Det3_345_234 = m[A32]*Det2_45_34 - m[A33]*Det2_45_24 
						+ m[A34]*Det2_45_23;
  double Det3_345_235 = m[A32]*Det2_45_35 - m[A33]*Det2_45_25 
						+ m[A35]*Det2_45_23;
  double Det3_345_245 = m[A32]*Det2_45_45 - m[A34]*Det2_45_25 
						+ m[A35]*Det2_45_24;
  double Det3_345_345 = m[A33]*Det2_45_45 - m[A34]*Det2_45_35 
						+ m[A35]*Det2_45_34;

  // Find all NECESSARY 4x4 dets:  (55 of them)

  double Det4_1234_0123 = m[A10]*Det3_234_123 - m[A11]*Det3_234_023 
			+ m[A12]*Det3_234_013 - m[A13]*Det3_234_012;
  double Det4_1234_0124 = m[A10]*Det3_234_124 - m[A11]*Det3_234_024 
			+ m[A12]*Det3_234_014 - m[A14]*Det3_234_012;
  double Det4_1234_0134 = m[A10]*Det3_234_134 - m[A11]*Det3_234_034 
			+ m[A13]*Det3_234_014 - m[A14]*Det3_234_013;
  double Det4_1234_0234 = m[A10]*Det3_234_234 - m[A12]*Det3_234_034 
			+ m[A13]*Det3_234_024 - m[A14]*Det3_234_023;
  double Det4_1234_1234 = m[A11]*Det3_234_234 - m[A12]*Det3_234_134 
			+ m[A13]*Det3_234_124 - m[A14]*Det3_234_123;
  double Det4_1235_0123 = m[A10]*Det3_235_123 - m[A11]*Det3_235_023 
			+ m[A12]*Det3_235_013 - m[A13]*Det3_235_012;
  double Det4_1235_0124 = m[A10]*Det3_235_124 - m[A11]*Det3_235_024 
			+ m[A12]*Det3_235_014 - m[A14]*Det3_235_012;
  double Det4_1235_0125 = m[A10]*Det3_235_125 - m[A11]*Det3_235_025 
			+ m[A12]*Det3_235_015 - m[A15]*Det3_235_012;
  double Det4_1235_0134 = m[A10]*Det3_235_134 - m[A11]*Det3_235_034 
			+ m[A13]*Det3_235_014 - m[A14]*Det3_235_013;
  double Det4_1235_0135 = m[A10]*Det3_235_135 - m[A11]*Det3_235_035 
			+ m[A13]*Det3_235_015 - m[A15]*Det3_235_013;
  double Det4_1235_0234 = m[A10]*Det3_235_234 - m[A12]*Det3_235_034 
			+ m[A13]*Det3_235_024 - m[A14]*Det3_235_023;
  double Det4_1235_0235 = m[A10]*Det3_235_235 - m[A12]*Det3_235_035 
			+ m[A13]*Det3_235_025 - m[A15]*Det3_235_023;
  double Det4_1235_1234 = m[A11]*Det3_235_234 - m[A12]*Det3_235_134 
			+ m[A13]*Det3_235_124 - m[A14]*Det3_235_123;
  double Det4_1235_1235 = m[A11]*Det3_235_235 - m[A12]*Det3_235_135 
			+ m[A13]*Det3_235_125 - m[A15]*Det3_235_123;
  double Det4_1245_0123 = m[A10]*Det3_245_123 - m[A11]*Det3_245_023 
			+ m[A12]*Det3_245_013 - m[A13]*Det3_245_012;
  double Det4_1245_0124 = m[A10]*Det3_245_124 - m[A11]*Det3_245_024 
			+ m[A12]*Det3_245_014 - m[A14]*Det3_245_012;
  double Det4_1245_0125 = m[A10]*Det3_245_125 - m[A11]*Det3_245_025 
			+ m[A12]*Det3_245_015 - m[A15]*Det3_245_012;
  double Det4_1245_0134 = m[A10]*Det3_245_134 - m[A11]*Det3_245_034 
			+ m[A13]*Det3_245_014 - m[A14]*Det3_245_013;
  double Det4_1245_0135 = m[A10]*Det3_245_135 - m[A11]*Det3_245_035 
			+ m[A13]*Det3_245_015 - m[A15]*Det3_245_013;
  double Det4_1245_0145 = m[A10]*Det3_245_145 - m[A11]*Det3_245_045 
			+ m[A14]*Det3_245_015 - m[A15]*Det3_245_014;
  double Det4_1245_0234 = m[A10]*Det3_245_234 - m[A12]*Det3_245_034 
			+ m[A13]*Det3_245_024 - m[A14]*Det3_245_023;
  double Det4_1245_0235 = m[A10]*Det3_245_235 - m[A12]*Det3_245_035 
			+ m[A13]*Det3_245_025 - m[A15]*Det3_245_023;
  double Det4_1245_0245 = m[A10]*Det3_245_245 - m[A12]*Det3_245_045 
			+ m[A14]*Det3_245_025 - m[A15]*Det3_245_024;
  double Det4_1245_1234 = m[A11]*Det3_245_234 - m[A12]*Det3_245_134 
			+ m[A13]*Det3_245_124 - m[A14]*Det3_245_123;
  double Det4_1245_1235 = m[A11]*Det3_245_235 - m[A12]*Det3_245_135 
			+ m[A13]*Det3_245_125 - m[A15]*Det3_245_123;
  double Det4_1245_1245 = m[A11]*Det3_245_245 - m[A12]*Det3_245_145 
			+ m[A14]*Det3_245_125 - m[A15]*Det3_245_124;
  double Det4_1345_0123 = m[A10]*Det3_345_123 - m[A11]*Det3_345_023 
			+ m[A12]*Det3_345_013 - m[A13]*Det3_345_012;
  double Det4_1345_0124 = m[A10]*Det3_345_124 - m[A11]*Det3_345_024 
			+ m[A12]*Det3_345_014 - m[A14]*Det3_345_012;
  double Det4_1345_0125 = m[A10]*Det3_345_125 - m[A11]*Det3_345_025 
			+ m[A12]*Det3_345_015 - m[A15]*Det3_345_012;
  double Det4_1345_0134 = m[A10]*Det3_345_134 - m[A11]*Det3_345_034 
			+ m[A13]*Det3_345_014 - m[A14]*Det3_345_013;
  double Det4_1345_0135 = m[A10]*Det3_345_135 - m[A11]*Det3_345_035 
			+ m[A13]*Det3_345_015 - m[A15]*Det3_345_013;
  double Det4_1345_0145 = m[A10]*Det3_345_145 - m[A11]*Det3_345_045 
			+ m[A14]*Det3_345_015 - m[A15]*Det3_345_014;
  double Det4_1345_0234 = m[A10]*Det3_345_234 - m[A12]*Det3_345_034 
			+ m[A13]*Det3_345_024 - m[A14]*Det3_345_023;
  double Det4_1345_0235 = m[A10]*Det3_345_235 - m[A12]*Det3_345_035 
			+ m[A13]*Det3_345_025 - m[A15]*Det3_345_023;
  double Det4_1345_0245 = m[A10]*Det3_345_245 - m[A12]*Det3_345_045 
			+ m[A14]*Det3_345_025 - m[A15]*Det3_345_024;
  double Det4_1345_0345 = m[A10]*Det3_345_345 - m[A13]*Det3_345_045 
			+ m[A14]*Det3_345_035 - m[A15]*Det3_345_034;
  double Det4_1345_1234 = m[A11]*Det3_345_234 - m[A12]*Det3_345_134 
			+ m[A13]*Det3_345_124 - m[A14]*Det3_345_123;
  double Det4_1345_1235 = m[A11]*Det3_345_235 - m[A12]*Det3_345_135 
			+ m[A13]*Det3_345_125 - m[A15]*Det3_345_123;
  double Det4_1345_1245 = m[A11]*Det3_345_245 - m[A12]*Det3_345_145 
			+ m[A14]*Det3_345_125 - m[A15]*Det3_345_124;
  double Det4_1345_1345 = m[A11]*Det3_345_345 - m[A13]*Det3_345_145 
			+ m[A14]*Det3_345_135 - m[A15]*Det3_345_134;
  double Det4_2345_0123 = m[A20]*Det3_345_123 - m[A21]*Det3_345_023 
			+ m[A22]*Det3_345_013 - m[A23]*Det3_345_012;
  double Det4_2345_0124 = m[A20]*Det3_345_124 - m[A21]*Det3_345_024 
			+ m[A22]*Det3_345_014 - m[A24]*Det3_345_012;
  double Det4_2345_0125 = m[A20]*Det3_345_125 - m[A21]*Det3_345_025 
			+ m[A22]*Det3_345_015 - m[A25]*Det3_345_012;
  double Det4_2345_0134 = m[A20]*Det3_345_134 - m[A21]*Det3_345_034 
			+ m[A23]*Det3_345_014 - m[A24]*Det3_345_013;
  double Det4_2345_0135 = m[A20]*Det3_345_135 - m[A21]*Det3_345_035 
			+ m[A23]*Det3_345_015 - m[A25]*Det3_345_013;
  double Det4_2345_0145 = m[A20]*Det3_345_145 - m[A21]*Det3_345_045 
			+ m[A24]*Det3_345_015 - m[A25]*Det3_345_014;
  double Det4_2345_0234 = m[A20]*Det3_345_234 - m[A22]*Det3_345_034 
			+ m[A23]*Det3_345_024 - m[A24]*Det3_345_023;
  double Det4_2345_0235 = m[A20]*Det3_345_235 - m[A22]*Det3_345_035 
			+ m[A23]*Det3_345_025 - m[A25]*Det3_345_023;
  double Det4_2345_0245 = m[A20]*Det3_345_245 - m[A22]*Det3_345_045 
			+ m[A24]*Det3_345_025 - m[A25]*Det3_345_024;
  double Det4_2345_0345 = m[A20]*Det3_345_345 - m[A23]*Det3_345_045 
			+ m[A24]*Det3_345_035 - m[A25]*Det3_345_034;
  double Det4_2345_1234 = m[A21]*Det3_345_234 - m[A22]*Det3_345_134 
			+ m[A23]*Det3_345_124 - m[A24]*Det3_345_123;
  double Det4_2345_1235 = m[A21]*Det3_345_235 - m[A22]*Det3_345_135 
			+ m[A23]*Det3_345_125 - m[A25]*Det3_345_123;
  double Det4_2345_1245 = m[A21]*Det3_345_245 - m[A22]*Det3_345_145 
			+ m[A24]*Det3_345_125 - m[A25]*Det3_345_124;
  double Det4_2345_1345 = m[A21]*Det3_345_345 - m[A23]*Det3_345_145 
			+ m[A24]*Det3_345_135 - m[A25]*Det3_345_134;
  double Det4_2345_2345 = m[A22]*Det3_345_345 - m[A23]*Det3_345_245 
			+ m[A24]*Det3_345_235 - m[A25]*Det3_345_234;

  // Find all NECESSARY 5x5 dets:  (19 of them)

  double Det5_01234_01234 = m[A00]*Det4_1234_1234 - m[A01]*Det4_1234_0234 
    + m[A02]*Det4_1234_0134 - m[A03]*Det4_1234_0124 + m[A04]*Det4_1234_0123;
  double Det5_01235_01234 = m[A00]*Det4_1235_1234 - m[A01]*Det4_1235_0234 
    + m[A02]*Det4_1235_0134 - m[A03]*Det4_1235_0124 + m[A04]*Det4_1235_0123;
  double Det5_01235_01235 = m[A00]*Det4_1235_1235 - m[A01]*Det4_1235_0235 
    + m[A02]*Det4_1235_0135 - m[A03]*Det4_1235_0125 + m[A05]*Det4_1235_0123;
  double Det5_01245_01234 = m[A00]*Det4_1245_1234 - m[A01]*Det4_1245_0234 
    + m[A02]*Det4_1245_0134 - m[A03]*Det4_1245_0124 + m[A04]*Det4_1245_0123;
  double Det5_01245_01235 = m[A00]*Det4_1245_1235 - m[A01]*Det4_1245_0235 
    + m[A02]*Det4_1245_0135 - m[A03]*Det4_1245_0125 + m[A05]*Det4_1245_0123;
  double Det5_01245_01245 = m[A00]*Det4_1245_1245 - m[A01]*Det4_1245_0245 
    + m[A02]*Det4_1245_0145 - m[A04]*Det4_1245_0125 + m[A05]*Det4_1245_0124;
  double Det5_01345_01234 = m[A00]*Det4_1345_1234 - m[A01]*Det4_1345_0234 
    + m[A02]*Det4_1345_0134 - m[A03]*Det4_1345_0124 + m[A04]*Det4_1345_0123;
  double Det5_01345_01235 = m[A00]*Det4_1345_1235 - m[A01]*Det4_1345_0235 
    + m[A02]*Det4_1345_0135 - m[A03]*Det4_1345_0125 + m[A05]*Det4_1345_0123;
  double Det5_01345_01245 = m[A00]*Det4_1345_1245 - m[A01]*Det4_1345_0245 
    + m[A02]*Det4_1345_0145 - m[A04]*Det4_1345_0125 + m[A05]*Det4_1345_0124;
  double Det5_01345_01345 = m[A00]*Det4_1345_1345 - m[A01]*Det4_1345_0345 
    + m[A03]*Det4_1345_0145 - m[A04]*Det4_1345_0135 + m[A05]*Det4_1345_0134;
  double Det5_02345_01234 = m[A00]*Det4_2345_1234 - m[A01]*Det4_2345_0234 
    + m[A02]*Det4_2345_0134 - m[A03]*Det4_2345_0124 + m[A04]*Det4_2345_0123;
  double Det5_02345_01235 = m[A00]*Det4_2345_1235 - m[A01]*Det4_2345_0235 
    + m[A02]*Det4_2345_0135 - m[A03]*Det4_2345_0125 + m[A05]*Det4_2345_0123;
  double Det5_02345_01245 = m[A00]*Det4_2345_1245 - m[A01]*Det4_2345_0245 
    + m[A02]*Det4_2345_0145 - m[A04]*Det4_2345_0125 + m[A05]*Det4_2345_0124;
  double Det5_02345_01345 = m[A00]*Det4_2345_1345 - m[A01]*Det4_2345_0345 
    + m[A03]*Det4_2345_0145 - m[A04]*Det4_2345_0135 + m[A05]*Det4_2345_0134;
  double Det5_02345_02345 = m[A00]*Det4_2345_2345 - m[A02]*Det4_2345_0345 
    + m[A03]*Det4_2345_0245 - m[A04]*Det4_2345_0235 + m[A05]*Det4_2345_0234;
  double Det5_12345_01234 = m[A10]*Det4_2345_1234 - m[A11]*Det4_2345_0234 
    + m[A12]*Det4_2345_0134 - m[A13]*Det4_2345_0124 + m[A14]*Det4_2345_0123;
  double Det5_12345_01235 = m[A10]*Det4_2345_1235 - m[A11]*Det4_2345_0235 
    + m[A12]*Det4_2345_0135 - m[A13]*Det4_2345_0125 + m[A15]*Det4_2345_0123;
  double Det5_12345_01245 = m[A10]*Det4_2345_1245 - m[A11]*Det4_2345_0245 
    + m[A12]*Det4_2345_0145 - m[A14]*Det4_2345_0125 + m[A15]*Det4_2345_0124;
  double Det5_12345_01345 = m[A10]*Det4_2345_1345 - m[A11]*Det4_2345_0345 
    + m[A13]*Det4_2345_0145 - m[A14]*Det4_2345_0135 + m[A15]*Det4_2345_0134;
  double Det5_12345_02345 = m[A10]*Det4_2345_2345 - m[A12]*Det4_2345_0345 
    + m[A13]*Det4_2345_0245 - m[A14]*Det4_2345_0235 + m[A15]*Det4_2345_0234;
  double Det5_12345_12345 = m[A11]*Det4_2345_2345 - m[A12]*Det4_2345_1345 
    + m[A13]*Det4_2345_1245 - m[A14]*Det4_2345_1235 + m[A15]*Det4_2345_1234;

  // Find the determinant 

  double det =    m[A00]*Det5_12345_12345 
	     	- m[A01]*Det5_12345_02345 
	     	+ m[A02]*Det5_12345_01345 
		- m[A03]*Det5_12345_01245 
		+ m[A04]*Det5_12345_01235 
		- m[A05]*Det5_12345_01234;

  if ( det == 0 ) {  
    ifail = 1;
    return;
  } 

  double oneOverDet = 1.0/det;
  double mn1OverDet = - oneOverDet;

  m[A00] =  Det5_12345_12345*oneOverDet;
  m[A01] =  Det5_12345_02345*mn1OverDet;
  m[A02] =  Det5_12345_01345*oneOverDet;
  m[A03] =  Det5_12345_01245*mn1OverDet;
  m[A04] =  Det5_12345_01235*oneOverDet;
  m[A05] =  Det5_12345_01234*mn1OverDet;

  m[A11] =  Det5_02345_02345*oneOverDet;
  m[A12] =  Det5_02345_01345*mn1OverDet;
  m[A13] =  Det5_02345_01245*oneOverDet;
  m[A14] =  Det5_02345_01235*mn1OverDet;
  m[A15] =  Det5_02345_01234*oneOverDet;

  m[A22] =  Det5_01345_01345*oneOverDet;
  m[A23] =  Det5_01345_01245*mn1OverDet;
  m[A24] =  Det5_01345_01235*oneOverDet;
  m[A25] =  Det5_01345_01234*mn1OverDet;

  m[A33] =  Det5_01245_01245*oneOverDet;
  m[A34] =  Det5_01245_01235*mn1OverDet;
  m[A35] =  Det5_01245_01234*oneOverDet;

  m[A44] =  Det5_01235_01235*oneOverDet;
  m[A45] =  Det5_01235_01234*mn1OverDet;

  m[A55] =  Det5_01234_01234*oneOverDet;

  return;
}

void SprSymMatrix::invertCholesky5 (int & ifail) {

// Invert by 
//
// a) decomposing M = G*G^T with G lower triangular
//	(if M is not positive definite this will fail, leaving this unchanged)
// b) inverting G to form H
// c) multiplying H^T * H to get M^-1.
//
// If the matrix is pos. def. it is inverted and 1 is returned.
// If the matrix is not pos. def. it remains unaltered and 0 is returned.

  double h10;				// below-diagonal elements of H
  double h20, h21;
  double h30, h31, h32;
  double h40, h41, h42, h43;
	
  double h00, h11, h22, h33, h44;	// 1/diagonal elements of G = 
					// diagonal elements of H

  double g10;				// below-diagonal elements of G
  double g20, g21;
  double g30, g31, g32;
  double g40, g41, g42, g43;

  ifail = 1;  // We start by assuing we won't succeed...

// Form G -- compute diagonal members of H directly rather than of G
//-------

// Scale first column by 1/sqrt(A00)

  h00 = m[A00]; 
  if (h00 <= 0) return;
  h00 = 1.0 / sqrt(h00);

  g10 = m[A10] * h00;
  g20 = m[A20] * h00;
  g30 = m[A30] * h00;
  g40 = m[A40] * h00;

// Form G11 (actually, just h11)

  h11 = m[A11] - (g10 * g10);
  if (h11 <= 0) return;
  h11 = 1.0 / sqrt(h11);

// Subtract inter-column column dot products from rest of column 1 and
// scale to get column 1 of G 

  g21 = (m[A21] - (g10 * g20)) * h11;
  g31 = (m[A31] - (g10 * g30)) * h11;
  g41 = (m[A41] - (g10 * g40)) * h11;

// Form G22 (actually, just h22)

  h22 = m[A22] - (g20 * g20) - (g21 * g21);
  if (h22 <= 0) return;
  h22 = 1.0 / sqrt(h22);

// Subtract inter-column column dot products from rest of column 2 and
// scale to get column 2 of G 

  g32 = (m[A32] - (g20 * g30) - (g21 * g31)) * h22;
  g42 = (m[A42] - (g20 * g40) - (g21 * g41)) * h22;

// Form G33 (actually, just h33)

  h33 = m[A33] - (g30 * g30) - (g31 * g31) - (g32 * g32);
  if (h33 <= 0) return;
  h33 = 1.0 / sqrt(h33);

// Subtract inter-column column dot product from A43 and scale to get G43

  g43 = (m[A43] - (g30 * g40) - (g31 * g41) - (g32 * g42)) * h33;

// Finally form h44 - if this is possible inversion succeeds

  h44 = m[A44] - (g40 * g40) - (g41 * g41) - (g42 * g42) - (g43 * g43);
  if (h44 <= 0) return;
  h44 = 1.0 / sqrt(h44);

// Form H = 1/G -- diagonal members of H are already correct
//-------------

// The order here is dictated by speed considerations

  h43 = -h33 *  g43 * h44;
  h32 = -h22 *  g32 * h33;
  h42 = -h22 * (g32 * h43 + g42 * h44);
  h21 = -h11 *  g21 * h22;
  h31 = -h11 * (g21 * h32 + g31 * h33);
  h41 = -h11 * (g21 * h42 + g31 * h43 + g41 * h44);
  h10 = -h00 *  g10 * h11;
  h20 = -h00 * (g10 * h21 + g20 * h22);
  h30 = -h00 * (g10 * h31 + g20 * h32 + g30 * h33);
  h40 = -h00 * (g10 * h41 + g20 * h42 + g30 * h43 + g40 * h44);

// Change this to its inverse = H^T*H
//------------------------------------

  m[A00] = h00 * h00 + h10 * h10 + h20 * h20 + h30 * h30 + h40 * h40;
  m[A01] = h10 * h11 + h20 * h21 + h30 * h31 + h40 * h41;
  m[A11] = h11 * h11 + h21 * h21 + h31 * h31 + h41 * h41;
  m[A02] = h20 * h22 + h30 * h32 + h40 * h42;
  m[A12] = h21 * h22 + h31 * h32 + h41 * h42;
  m[A22] = h22 * h22 + h32 * h32 + h42 * h42;
  m[A03] = h30 * h33 + h40 * h43;
  m[A13] = h31 * h33 + h41 * h43;
  m[A23] = h32 * h33 + h42 * h43;
  m[A33] = h33 * h33 + h43 * h43;
  m[A04] = h40 * h44;
  m[A14] = h41 * h44;
  m[A24] = h42 * h44;
  m[A34] = h43 * h44;
  m[A44] = h44 * h44;

  ifail = 0;
  return;

}


void SprSymMatrix::invertCholesky6 (int & ifail) {

// Invert by 
//
// a) decomposing M = G*G^T with G lower triangular
//	(if M is not positive definite this will fail, leaving this unchanged)
// b) inverting G to form H
// c) multiplying H^T * H to get M^-1.
//
// If the matrix is pos. def. it is inverted and 1 is returned.
// If the matrix is not pos. def. it remains unaltered and 0 is returned.

  double h10;				// below-diagonal elements of H
  double h20, h21;
  double h30, h31, h32;
  double h40, h41, h42, h43;
  double h50, h51, h52, h53, h54;
	
  double h00, h11, h22, h33, h44, h55;	// 1/diagonal elements of G = 
					// diagonal elements of H

  double g10;				// below-diagonal elements of G
  double g20, g21;
  double g30, g31, g32;
  double g40, g41, g42, g43;
  double g50, g51, g52, g53, g54;

  ifail = 1;  // We start by assuing we won't succeed...

// Form G -- compute diagonal members of H directly rather than of G
//-------

// Scale first column by 1/sqrt(A00)

  h00 = m[A00]; 
  if (h00 <= 0) return;
  h00 = 1.0 / sqrt(h00);

  g10 = m[A10] * h00;
  g20 = m[A20] * h00;
  g30 = m[A30] * h00;
  g40 = m[A40] * h00;
  g50 = m[A50] * h00;

// Form G11 (actually, just h11)

  h11 = m[A11] - (g10 * g10);
  if (h11 <= 0) return;
  h11 = 1.0 / sqrt(h11);

// Subtract inter-column column dot products from rest of column 1 and
// scale to get column 1 of G 

  g21 = (m[A21] - (g10 * g20)) * h11;
  g31 = (m[A31] - (g10 * g30)) * h11;
  g41 = (m[A41] - (g10 * g40)) * h11;
  g51 = (m[A51] - (g10 * g50)) * h11;

// Form G22 (actually, just h22)

  h22 = m[A22] - (g20 * g20) - (g21 * g21);
  if (h22 <= 0) return;
  h22 = 1.0 / sqrt(h22);

// Subtract inter-column column dot products from rest of column 2 and
// scale to get column 2 of G 

  g32 = (m[A32] - (g20 * g30) - (g21 * g31)) * h22;
  g42 = (m[A42] - (g20 * g40) - (g21 * g41)) * h22;
  g52 = (m[A52] - (g20 * g50) - (g21 * g51)) * h22;

// Form G33 (actually, just h33)

  h33 = m[A33] - (g30 * g30) - (g31 * g31) - (g32 * g32);
  if (h33 <= 0) return;
  h33 = 1.0 / sqrt(h33);

// Subtract inter-column column dot products from rest of column 3 and
// scale to get column 3 of G 

  g43 = (m[A43] - (g30 * g40) - (g31 * g41) - (g32 * g42)) * h33;
  g53 = (m[A53] - (g30 * g50) - (g31 * g51) - (g32 * g52)) * h33;

// Form G44 (actually, just h44)

  h44 = m[A44] - (g40 * g40) - (g41 * g41) - (g42 * g42) - (g43 * g43);
  if (h44 <= 0) return;
  h44 = 1.0 / sqrt(h44);

// Subtract inter-column column dot product from M54 and scale to get G54

  g54 = (m[A54] - (g40 * g50) - (g41 * g51) - (g42 * g52) - (g43 * g53)) * h44;

// Finally form h55 - if this is possible inversion succeeds

  h55 = m[A55] - (g50*g50) - (g51*g51) - (g52*g52) - (g53*g53) - (g54*g54);
  if (h55 <= 0) return;
  h55 = 1.0 / sqrt(h55);

// Form H = 1/G -- diagonal members of H are already correct
//-------------

// The order here is dictated by speed considerations

  h54 = -h44 *  g54 * h55;
  h43 = -h33 *  g43 * h44;
  h53 = -h33 * (g43 * h54 + g53 * h55);
  h32 = -h22 *  g32 * h33;
  h42 = -h22 * (g32 * h43 + g42 * h44);
  h52 = -h22 * (g32 * h53 + g42 * h54 + g52 * h55);
  h21 = -h11 *  g21 * h22;
  h31 = -h11 * (g21 * h32 + g31 * h33);
  h41 = -h11 * (g21 * h42 + g31 * h43 + g41 * h44);
  h51 = -h11 * (g21 * h52 + g31 * h53 + g41 * h54 + g51 * h55);
  h10 = -h00 *  g10 * h11;
  h20 = -h00 * (g10 * h21 + g20 * h22);
  h30 = -h00 * (g10 * h31 + g20 * h32 + g30 * h33);
  h40 = -h00 * (g10 * h41 + g20 * h42 + g30 * h43 + g40 * h44);
  h50 = -h00 * (g10 * h51 + g20 * h52 + g30 * h53 + g40 * h54 + g50 * h55);

// Change this to its inverse = H^T*H
//------------------------------------

  m[A00] = h00 * h00 + h10 * h10 + h20 * h20 + h30 * h30 + h40 * h40 + h50*h50;
  m[A01] = h10 * h11 + h20 * h21 + h30 * h31 + h40 * h41 + h50 * h51;
  m[A11] = h11 * h11 + h21 * h21 + h31 * h31 + h41 * h41 + h51 * h51;
  m[A02] = h20 * h22 + h30 * h32 + h40 * h42 + h50 * h52;
  m[A12] = h21 * h22 + h31 * h32 + h41 * h42 + h51 * h52;
  m[A22] = h22 * h22 + h32 * h32 + h42 * h42 + h52 * h52;
  m[A03] = h30 * h33 + h40 * h43 + h50 * h53;
  m[A13] = h31 * h33 + h41 * h43 + h51 * h53;
  m[A23] = h32 * h33 + h42 * h43 + h52 * h53;
  m[A33] = h33 * h33 + h43 * h43 + h53 * h53;
  m[A04] = h40 * h44 + h50 * h54;
  m[A14] = h41 * h44 + h51 * h54;
  m[A24] = h42 * h44 + h52 * h54;
  m[A34] = h43 * h44 + h53 * h54;
  m[A44] = h44 * h44 + h54 * h54;
  m[A05] = h50 * h55;
  m[A15] = h51 * h55;
  m[A25] = h52 * h55;
  m[A35] = h53 * h55;
  m[A45] = h54 * h55;
  m[A55] = h55 * h55;

  ifail = 0;
  return;

}


void SprSymMatrix::invert4  (int & ifail) {

  ifail = 0;

  // Find all NECESSARY 2x2 dets:  (14 of them)

  double Det2_12_01 = m[A10]*m[A21] - m[A11]*m[A20];
  double Det2_12_02 = m[A10]*m[A22] - m[A12]*m[A20];
  double Det2_12_12 = m[A11]*m[A22] - m[A12]*m[A21];
  double Det2_13_01 = m[A10]*m[A31] - m[A11]*m[A30];
  double Det2_13_02 = m[A10]*m[A32] - m[A12]*m[A30];
  double Det2_13_03 = m[A10]*m[A33] - m[A13]*m[A30];
  double Det2_13_12 = m[A11]*m[A32] - m[A12]*m[A31];
  double Det2_13_13 = m[A11]*m[A33] - m[A13]*m[A31];
  double Det2_23_01 = m[A20]*m[A31] - m[A21]*m[A30];
  double Det2_23_02 = m[A20]*m[A32] - m[A22]*m[A30];
  double Det2_23_03 = m[A20]*m[A33] - m[A23]*m[A30];
  double Det2_23_12 = m[A21]*m[A32] - m[A22]*m[A31];
  double Det2_23_13 = m[A21]*m[A33] - m[A23]*m[A31];
  double Det2_23_23 = m[A22]*m[A33] - m[A23]*m[A32];

  // Find all NECESSARY 3x3 dets:   (10 of them)

  double Det3_012_012 = m[A00]*Det2_12_12 - m[A01]*Det2_12_02 
				+ m[A02]*Det2_12_01;
  double Det3_013_012 = m[A00]*Det2_13_12 - m[A01]*Det2_13_02 
				+ m[A02]*Det2_13_01;
  double Det3_013_013 = m[A00]*Det2_13_13 - m[A01]*Det2_13_03
				+ m[A03]*Det2_13_01;
  double Det3_023_012 = m[A00]*Det2_23_12 - m[A01]*Det2_23_02 
				+ m[A02]*Det2_23_01;
  double Det3_023_013 = m[A00]*Det2_23_13 - m[A01]*Det2_23_03
				+ m[A03]*Det2_23_01;
  double Det3_023_023 = m[A00]*Det2_23_23 - m[A02]*Det2_23_03
				+ m[A03]*Det2_23_02;
  double Det3_123_012 = m[A10]*Det2_23_12 - m[A11]*Det2_23_02 
				+ m[A12]*Det2_23_01;
  double Det3_123_013 = m[A10]*Det2_23_13 - m[A11]*Det2_23_03 
				+ m[A13]*Det2_23_01;
  double Det3_123_023 = m[A10]*Det2_23_23 - m[A12]*Det2_23_03 
				+ m[A13]*Det2_23_02;
  double Det3_123_123 = m[A11]*Det2_23_23 - m[A12]*Det2_23_13 
				+ m[A13]*Det2_23_12;

  // Find the 4x4 det:

  double det =    m[A00]*Det3_123_123 
		- m[A01]*Det3_123_023 
		+ m[A02]*Det3_123_013 
		- m[A03]*Det3_123_012;

  if ( det == 0 ) {  
    ifail = 1;
    return;
  } 

  double oneOverDet = 1.0/det;
  double mn1OverDet = - oneOverDet;

  m[A00] =  Det3_123_123 * oneOverDet;
  m[A01] =  Det3_123_023 * mn1OverDet;
  m[A02] =  Det3_123_013 * oneOverDet;
  m[A03] =  Det3_123_012 * mn1OverDet;


  m[A11] =  Det3_023_023 * oneOverDet;
  m[A12] =  Det3_023_013 * mn1OverDet;
  m[A13] =  Det3_023_012 * oneOverDet;

  m[A22] =  Det3_013_013 * oneOverDet;
  m[A23] =  Det3_013_012 * mn1OverDet;

  m[A33] =  Det3_012_012 * oneOverDet;

  return;
}

void SprSymMatrix::invertHaywood4  (int & ifail) {
  invert4(ifail); // For the 4x4 case, the method we use for invert is already
  		  // the Haywood method.
}

