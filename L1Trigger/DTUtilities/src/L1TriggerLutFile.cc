//-------------------------------------------------
//
//   Class: L1TriggerLutFile
//
//   Description: Auxiliary class for 
//                Look-up table files
//
//
//   $Date: 2003/10/17 12:39:42 $
//   $Revision: 1.1 $
//
//   Author :
//   N. Neumeister            CERN EP
//
//--------------------------------------------------
//#include "Utilities/Configuration/interface/Architecture.h"

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/DTUtilities/interface/L1TriggerLutFile.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>

using namespace std;

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

// --------------------------------
//       class L1TriggerLutFile
//---------------------------------

//----------------
// Constructors --
//----------------

L1TriggerLutFile::L1TriggerLutFile(const string name) : m_file(name) {}

L1TriggerLutFile::L1TriggerLutFile(const L1TriggerLutFile& in) : m_file(in.m_file) {}


//--------------
// Destructor --
//--------------
L1TriggerLutFile::~L1TriggerLutFile() {}

//--------------
// Operations --
//--------------

//
// assignment operator
//
L1TriggerLutFile& L1TriggerLutFile::operator=(const L1TriggerLutFile& lut) {

  m_file = lut.m_file;
  return *this;

}


//
// open file
//
int L1TriggerLutFile::open() {

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


//
// read and ignore n lines from file
//
void L1TriggerLutFile::ignoreLines(int n) {

  char buf[256];
  for ( int i = 0; i < n; i++ ) m_fin.getline(buf,256);

}


//
// read one integer from file
//
int L1TriggerLutFile::readInteger() { 

  int tmp = 0;
  m_fin >> tmp; 
  return tmp;

}


//
// read one hex from file
//
int L1TriggerLutFile::readHex() { 

  int tmp = 0;
  m_fin >> hex >> tmp; 
  return tmp;
    
}


//
// read one string from file
//
string L1TriggerLutFile::readString() {

  string tmp;
  m_fin >> tmp;
  return tmp;

}
