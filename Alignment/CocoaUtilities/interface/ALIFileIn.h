//   COCOA class header file
//Id:  ALIFileIn.h
//CAT: Model
//
//   istream class for handling the reading of files
//
//   History: v1.0
//   Pedro Arce

#ifndef FILEIN_H
#define FILEIN_H

#include <fstream>
#include <iostream>

#include <vector>

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"

class ALIFileIn {
public:
  ALIFileIn(){};
  ~ALIFileIn() {}

private:
  ALIFileIn(const ALIstring& name) : theName(name) {}

public:
  // Get the only instance opening the file
  static ALIFileIn& getInstance(const ALIstring& name);
  // Get the only instance when file should be already opened
  static ALIFileIn& getInstanceOpened(const ALIstring& name);

  // Read a line and transform it to a vector of words
  ALIint getWordsInLine(std::vector<ALIstring>& wl);

  // Print out an error message indicating the line being read
  void ErrorInLine();

  // Access data members
  const ALIint nline() { return theLineNo[theCurrentFile]; }

  const ALIstring& name() { return theName; }

  ALIbool eof();
  void close();

private:
  void openNewFile(const char* filename);

private:
  std::vector<std::ifstream*> theFiles;
  // Number of line being read
  std::vector<ALIint> theLineNo;
  std::vector<ALIstring> theNames;
  int theCurrentFile;  // index of file being read in theFiles

  // private DATA MEMEBERS
  // Vector of class instances (each one identified by its name)
  static std::vector<ALIFileIn*> theInstances;

  /// Name of file
  ALIstring theName;
};

#endif
