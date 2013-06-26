//-------------------------------------------------
//
//   Class: DTTPGLutFile
//
//   Description: Auxiliary class for 
//                Look-up table files
//
//
//   $Date: 2007/10/23 13:44:23 $
//   $Revision: 1.1 $
//
//   Author :
//   N. Neumeister            CERN EP
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/DTUtilities/interface/DTTPGLutFile.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>

using namespace std;

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

// --------------------------------
//       class DTTPGLutFile
//---------------------------------

//----------------
// Constructors --
//----------------

DTTPGLutFile::DTTPGLutFile(const string name) : m_file(name) {}

DTTPGLutFile::DTTPGLutFile(const DTTPGLutFile& in) : m_file(in.m_file) {}


//--------------
// Destructor --
//--------------

DTTPGLutFile::~DTTPGLutFile() {}

//--------------
// Operations --
//--------------

DTTPGLutFile& DTTPGLutFile::operator=(const DTTPGLutFile& lut) {

  m_file = lut.m_file;
  return *this;

}


int DTTPGLutFile::open() {

  const char* file_name = m_file.c_str();
  m_fin.open(file_name,ios::in);
  if ( !m_fin ) {
    cout << "can not open file : " << file_name << endl;
    return -1;
  }
  else 	{
    return 0;
  }

}


void DTTPGLutFile::ignoreLines(int n) {

  char buf[256];
  for ( int i = 0; i < n; i++ ) m_fin.getline(buf,256);

}


int DTTPGLutFile::readInteger() { 

  int tmp = 0;
  m_fin >> tmp; 
  return tmp;

}


int DTTPGLutFile::readHex() { 

  int tmp = 0;
  m_fin >> hex >> tmp; 
  return tmp;
    
}


string DTTPGLutFile::readString() {

  string tmp;
  m_fin >> tmp;
  return tmp;

}
