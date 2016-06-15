#ifndef MILLE_H
#define MILLE_H

#include <fstream>

/**
 * \class Mille
 *
 *  Class to write a C binary (cf. below) file of a given name and to fill it
 *  with information used as input to pede.
 *  Use its member functions mille(...), special(...), kill() and end() as you would
 *  use the fortran MILLE and its entry points MILLSP, KILLE and ENDLE.
 *
 *  For debugging purposes constructor flags enable switching to text output and/or
 *  to write also derivatives and lables which are ==0.
 *  But note that pede will not be able to read text output and has not been tested with
 *  derivatives/labels ==0.
 *
 *  \author    : Gero Flucke
 *  date       : October 2006
 *  $Revision: 1.2 $
 *  $Date: 2007/03/16 16:44:47 $
 *  (last update by $Author: flucke $)
 */

class Mille
{
 public:
  Mille(const char *outFileName, bool asBinary = true, bool writeZero = false);
  ~Mille();

  void mille(int NLC, const float *derLc, int NGL, const float *derGl,
             const int *label, float rMeas, float sigma);
  void special(int nSpecial, const float *floatings, const int *integers);
  void kill();
  void flushOutputFile();
  void resetOutputFile();
  void end();

 private:
  void newSet();
  bool checkBufferSize(int nLocal, int nGlobal);

  const std::ios_base::openmode fileMode_; // file open mode of the binary
  const std::string fileName_;             // file name of the binary
  std::ofstream outFile_; // C-binary for output
  bool asBinary_;         // if false output as text
  bool writeZero_;        // if true also write out derivatives/lables ==0

  enum {bufferSize_ = 5000};
  int   bufferInt_[bufferSize_];   // to collect labels etc.
  float bufferFloat_[bufferSize_]; // to collect derivatives etc.
  int   bufferPos_;
  bool  hasSpecial_; // if true, special(..) already called for this record

  enum {maxLabel_ = (0xFFFFFFFF - (1 << 31))}; // largest label allowed: 2^31 - 1
};
#endif
