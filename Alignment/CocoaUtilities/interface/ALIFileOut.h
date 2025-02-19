//   COCOA class header file
//Id:  ALIFileOut.h
//CAT: Model
//
//   ostream class for handling the output
// 
//   History: v1.0 
//   Pedro Arce

#ifndef FILEOUT_H
#define FILEOUT_H

#include <fstream>
#include <iostream>

#include <vector>
//#include "bstring.h"

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"


class ALIFileOut : public std::ofstream 
{
public:
  ALIFileOut(){};
  ALIFileOut( const ALIstring& name ): std::ofstream(), theName(name){};
  ~ALIFileOut(){};

  // get the instance of file with name filename
  static ALIFileOut& getInstance( const ALIstring& filename );

 // Access data members
  const ALIstring& name() { return theName; }

// private DATA MEMEBERS
private:
  // Class only instance
  static std::vector<ALIFileOut*> theInstances;

  /// Name of file
  ALIstring theName; 
};

#endif 

