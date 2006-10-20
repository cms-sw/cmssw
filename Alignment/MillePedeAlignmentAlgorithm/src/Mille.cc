/**
 * \file Mille.cc
 *  \author    : Gero Flucke
 *  date       : October 2006
 *  $Revision: 1.11 $
 *  $Date$
 *  (last update by $Author$)
 */

#include "Mille.h"

#include <fstream>
#include <iostream>

//___________________________________________________________________________

Mille::Mille(const char *outFileName, bool asBinary, bool writeZero) : 
  myOutFile(outFileName, (asBinary ? (std::ios::binary | std::ios::out) : std::ios::out)),
  myAsBinary(asBinary), myWriteZero(writeZero), myBufferPos(-1)
{
  // opens outFileName as binary output file

  myBufferInt[0]   = 0;
  myBufferFloat[0] = 0.;

  if (!myOutFile.is_open()) {
    std::cerr << "Mille::Mille: Could not open " << outFileName 
	      << " as output file." << std::endl;
  }
}

//___________________________________________________________________________

Mille::~Mille()
{
  // closes file
  myOutFile.close();
}

//___________________________________________________________________________

void Mille::mille(int NLC, const float *derLc,
		  int NGL, const float *derGl, const int *label,
		  float rMeas, float sigma)
{
  if (sigma <= 0.) return;
  if (myBufferPos == -1) this->newSet(); // start, e.g. new track
  if (!this->checkBufferSize(NLC, NGL)) return;

  // first store measurement
  ++myBufferPos;
  myBufferFloat[myBufferPos] = rMeas;
  myBufferInt  [myBufferPos] = 0;

  // store local derivatives and local 'lables' 1,...,NLC
  for (int i = 0; i < NLC; ++i) {
    if (derLc[i] || myWriteZero) { // by default store only non-zero derivatives
      ++myBufferPos;
      myBufferFloat[myBufferPos] = derLc[i]; // local derivatives
      myBufferInt  [myBufferPos] = i+1;      // index of local parameter
    }
  }

  // store uncertainty of measurement in between locals and globals
  ++myBufferPos;
  myBufferFloat[myBufferPos] = sigma;
  myBufferInt  [myBufferPos] = 0;

  // store global derivatives and their lables
  for (int i = 0; i < NGL; ++i) {
    if (derGl[i] || myWriteZero) { // by default store only non-zero derivatives
      if ((label[i] > 0 || myWriteZero) && label[i] <= 99999999) { // and for valid labels
	++myBufferPos;
	myBufferFloat[myBufferPos] = derGl[i]; // global derivatives
	myBufferInt  [myBufferPos] = label[i]; // index of global parameter
      } else {
	std::cerr << "Mille::mille: Invalid label " << label[i] 
		  << " <= 0 or > " << 99999999 << std::endl; 
      }
    }
  }
}

//___________________________________________________________________________

void Mille::kill()
{
  // reset buffers, i.e. kill derivatives accumulated for current set
  myBufferPos = -1;
}

//___________________________________________________________________________

void Mille::end()
{
  // write set of derivatives with same local parameters to file
  if (myBufferPos > 0) { // only if anything stored...
    const int numWordsToWrite = (myBufferPos + 1)*2;

    if (myAsBinary) {
      myOutFile.write(reinterpret_cast<const char*>(&numWordsToWrite), 
		      sizeof(numWordsToWrite));
      myOutFile.write(reinterpret_cast<char*>(myBufferFloat), 
		      (myBufferPos+1) * sizeof(myBufferFloat[0]));
      myOutFile.write(reinterpret_cast<char*>(myBufferInt), 
		      (myBufferPos+1) * sizeof(myBufferInt[0]));
    } else {
      myOutFile << numWordsToWrite << "\n";
      for (int i = 0; i < myBufferPos+1; ++i) {
	myOutFile << myBufferFloat[i] << " ";
      }
      myOutFile << "\n";
      
      for (int i = 0; i < myBufferPos+1; ++i) {
	myOutFile << myBufferInt[i] << " ";
      }
      myOutFile << "\n";
    }
  }
  myBufferPos = -1; // reset buffer for next set of derivatives
}

//___________________________________________________________________________

void Mille::newSet()
{
  // initilise for new set of locals, i.e. new track
  myBufferPos = 0;
  myBufferFloat[0] = 0.0;
  myBufferInt  [0] = 0;   // position 0 used as error counter
}

//___________________________________________________________________________

bool Mille::checkBufferSize(int nLocal, int nGlobal)
{
  // enough space for next nLocal + nGlobal derivatives incl. measurement?

  if (myBufferPos + nLocal + nGlobal + 2 >= myBufferSize) {
    ++(myBufferInt[0]); // increase error count
    std::cerr << "Mille::checkBufferSize: Buffer too short (" 
	      << myBufferSize << "),"
	      << "\n need space for nLocal (" << nLocal<< ")"
	      << "/nGlobal (" << nGlobal << ") local/global derivatives, " 
	      << myBufferPos + 1 << " already stored!"
	      << std::endl;
    return false;
  } else {
    return true;
  }
}
