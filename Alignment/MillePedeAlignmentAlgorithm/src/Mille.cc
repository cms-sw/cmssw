/**
 * \file Mille.cc
 *  \author    : Gero Flucke
 *  date       : October 2006
 *  $Revision: 1.3 $
 *  $Date: 2007/04/16 17:47:38 $
 *  (last update by $Author: flucke $)
 */

#include "Mille.h"

#include <iostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//___________________________________________________________________________

Mille::Mille(const char *outFileName, bool asBinary, bool writeZero) :
  fileMode_(asBinary ? (std::ios::binary | std::ios::out) : std::ios::out),
  fileName_(outFileName),
  outFile_(fileName_, fileMode_),
  asBinary_(asBinary), writeZero_(writeZero), bufferPos_(-1), hasSpecial_(false)
{
  // opens outFileName, by default as binary file

  // Instead bufferPos_(-1), hasSpecial_(false) and the following two lines
  // we could call newSet() and kill()...
  bufferInt_[0]   = 0;
  bufferFloat_[0] = 0.;

  if (!outFile_.is_open()) {
    edm::LogError("Alignment")
      << "Mille::Mille: Could not open " << fileName_ << " as output file.";
  }
}

//___________________________________________________________________________

Mille::~Mille()
{
  // closes file
  outFile_.close();
}

//___________________________________________________________________________

void Mille::mille(int NLC, const float *derLc,
                  int NGL, const float *derGl, const int *label,
                  float rMeas, float sigma)
{
  if (sigma <= 0.) return;
  if (bufferPos_ == -1) this->newSet(); // start, e.g. new track
  if (!this->checkBufferSize(NLC, NGL)) return;

  // first store measurement
  ++bufferPos_;
  bufferFloat_[bufferPos_] = rMeas;
  bufferInt_  [bufferPos_] = 0;

  // store local derivatives and local 'lables' 1,...,NLC
  for (int i = 0; i < NLC; ++i) {
    if (derLc[i] || writeZero_) { // by default store only non-zero derivatives
      ++bufferPos_;
      bufferFloat_[bufferPos_] = derLc[i]; // local derivatives
      bufferInt_  [bufferPos_] = i+1;      // index of local parameter
    }
  }

  // store uncertainty of measurement in between locals and globals
  ++bufferPos_;
  bufferFloat_[bufferPos_] = sigma;
  bufferInt_  [bufferPos_] = 0;

  // store global derivatives and their lables
  for (int i = 0; i < NGL; ++i) {
    if (derGl[i] || writeZero_) { // by default store only non-zero derivatives
      if ((label[i] > 0 || writeZero_) && label[i] <= maxLabel_) { // and for valid labels
        ++bufferPos_;
        bufferFloat_[bufferPos_] = derGl[i]; // global derivatives
        bufferInt_  [bufferPos_] = label[i]; // index of global parameter
      } else {
        edm::LogError("Alignment")
          << "Mille::mille: Invalid label " << label[i]
          << " <= 0 or > " << maxLabel_;
      }
    }
  }
}

//___________________________________________________________________________
void Mille::special(int nSpecial, const float *floatings, const int *integers)
{
  if (nSpecial == 0) return;
  if (bufferPos_ == -1) this->newSet(); // start, e.g. new track
  if (hasSpecial_) {
    edm::LogError("Alignment")
      << "Mille::special: Special values already stored for this record.";
    return;
  }
  if (!this->checkBufferSize(nSpecial, 0)) return;
  hasSpecial_ = true; // after newSet() (Note: MILLSP sets to buffer position...)

  //  bufferFloat_[.]   | bufferInt_[.]
  // ------------------------------------
  //      0.0           |      0
  //  -float(nSpecial)  |      0
  //  The above indicates special data, following are nSpecial floating and nSpecial integer data.

  ++bufferPos_; // zero pair
  bufferFloat_[bufferPos_] = 0.;
  bufferInt_  [bufferPos_] = 0;

  ++bufferPos_; // nSpecial and zero
  bufferFloat_[bufferPos_] = -nSpecial; // automatic conversion to float
  bufferInt_  [bufferPos_] = 0;

  for (int i = 0; i < nSpecial; ++i) {
    ++bufferPos_;
    bufferFloat_[bufferPos_] = floatings[i];
    bufferInt_  [bufferPos_] = integers[i];
  }
}

//___________________________________________________________________________

void Mille::kill()
{
  // reset buffers, i.e. kill derivatives accumulated for current set
  bufferPos_ = -1;
}

//___________________________________________________________________________


void Mille::flushOutputFile() {
  // flush output file
  outFile_.flush();
}

//___________________________________________________________________________


void Mille::resetOutputFile() {
  // flush output file
  outFile_.close();
  outFile_.open(fileName_, fileMode_);
  if (!outFile_.is_open()) {
    edm::LogError("Alignment")
      << "Mille::resetOutputFile: Could not reopen " << fileName_ << ".";
  }
}

//___________________________________________________________________________

void Mille::end()
{
  // write set of derivatives with same local parameters to file
  if (bufferPos_ > 0) { // only if anything stored...
    const int numWordsToWrite = (bufferPos_ + 1)*2;

    if (asBinary_) {
      outFile_.write(reinterpret_cast<const char*>(&numWordsToWrite),
                     sizeof(numWordsToWrite));
      outFile_.write(reinterpret_cast<char*>(bufferFloat_),
                     (bufferPos_+1) * sizeof(bufferFloat_[0]));
      outFile_.write(reinterpret_cast<char*>(bufferInt_),
                     (bufferPos_+1) * sizeof(bufferInt_[0]));
    } else {
      outFile_ << numWordsToWrite << "\n";
      for (int i = 0; i < bufferPos_+1; ++i) {
        outFile_ << bufferFloat_[i] << " ";
      }
      outFile_ << "\n";

      for (int i = 0; i < bufferPos_+1; ++i) {
        outFile_ << bufferInt_[i] << " ";
      }
      outFile_ << "\n";
    }
  }
  bufferPos_ = -1; // reset buffer for next set of derivatives
}

//___________________________________________________________________________

void Mille::newSet()
{
  // initilise for new set of locals, e.g. new track
  bufferPos_ = 0;
  hasSpecial_ = false;
  bufferFloat_[0] = 0.0;
  bufferInt_  [0] = 0;   // position 0 used as error counter
}

//___________________________________________________________________________

bool Mille::checkBufferSize(int nLocal, int nGlobal)
{
  // enough space for next nLocal + nGlobal derivatives incl. measurement?

  if (bufferPos_ + nLocal + nGlobal + 2 >= bufferSize_) {
    ++(bufferInt_[0]); // increase error count
    edm::LogError("Alignment")
      << "Mille::checkBufferSize: Buffer too short ("
      << bufferSize_ << "),"
      << "\n need space for nLocal (" << nLocal<< ")"
      << "/nGlobal (" << nGlobal << ") local/global derivatives, "
      << bufferPos_ + 1 << " already stored!";
    return false;
  } else {
    return true;
  }
}
