//   COCOA class header file
//Id:  MatrixMeschach.h
//CAT: Model
//
//   Class for matrices
//
//   History: v1.0
//   Pedro Arce

#ifndef _ALIMATRIX_HH
#define _ALIMATRIX_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include <vector>
#include <iostream>

extern "C" {
#include <matrix.h>
#include <matrix2.h>
}

// Meschach external (matrix.h) defines C++-incompatible macros
// which break other code (e.g. standard <limits>).
// Since these are not used here, undef them.
#undef max
#undef min
#undef catch
#undef Real

class MatrixMeschach {
public:
  MatrixMeschach();
  MatrixMeschach(ALIint NoCol, ALIint NoLin);
  MatrixMeschach(const MatrixMeschach& mat);
  ~MatrixMeschach();

  void AddData(ALIuint col, ALIuint lin, ALIdouble data);
  void transpose();
  void inverse();
  void Dump(const ALIstring& mtext);
  void ostrDump(std::ostream& fout, const ALIstring& mtext);
  void EliminateLines(ALIint lin_first, ALIint lin_last);
  void EliminateColumns(ALIint lin_first, ALIint lin_last);
  void SetCorrelation(ALIint i1, ALIint i2, ALIdouble corr);

  MatrixMeschach& operator=(const MatrixMeschach& mat);
  void operator*=(const MatrixMeschach& mat);
  void operator+=(const MatrixMeschach& mat);
  void operator*=(const ALIdouble num);
  ALIdouble operator()(int i, int j) const;

  //ACCESS PRIVATE DATA MEMBERS
  ALIint NoLines() const { return _NoLines; }
  ALIint NoColumns() const { return _NoColumns; }
  void setNoColumns(ALIint ncol) { _NoColumns = ncol; }
  void setNoLines(ALIint nlin) { _NoLines = nlin; }
  const MAT* Mat() const { return _Mat; }
  void setMat(MAT* mat) { _Mat = mat; }
  MAT* MatNonConst() const { return _Mat; }

private:
  // private data members
  ALIint _NoLines;
  ALIint _NoColumns;
  //  vector< ALIdouble> _data;
  MAT* _Mat;

  void copy(const MatrixMeschach& mat);
};

MatrixMeschach operator*(const MatrixMeschach& mat1, const MatrixMeschach& mat2);
MatrixMeschach operator+(const MatrixMeschach& mat1, const MatrixMeschach& mat2);
MatrixMeschach operator-(const MatrixMeschach& mat1, const MatrixMeschach& mat2);
MatrixMeschach operator*(const ALIdouble doub, const MatrixMeschach& mat);
MatrixMeschach operator*(const MatrixMeschach& mat, const ALIdouble doub);

MatrixMeschach* MatrixByMatrix(const MatrixMeschach& mat1, const MatrixMeschach& mat2);
typedef MatrixMeschach ALIMatrix;

#endif
