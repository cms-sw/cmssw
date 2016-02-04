//-------------------------------------------------
//
//   Class: L1MuDTEtaPattern
//
//   Description: Pattern for Eta for Eta Track Finder
//
//
//   $Date: 2010/01/19 18:39:54 $
//   $Revision: 1.4 $
//
//   Author :
//   N. Neumeister             CERN EP
//   J. Troconiz               UAM Madrid
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "CondFormats/L1TObjects/interface/L1MuDTEtaPattern.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>
#include <bitset>
#include <cstdlib>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

using namespace std;

// --------------------------------
//       class L1MuDTEtaPattern  
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuDTEtaPattern::L1MuDTEtaPattern() :
  m_id(0), m_eta(0), m_qual(0) {
  
  for (int i = 0; i < 3; i++) {
    m_wheel[i] = 0; 
    m_position[i] = 0;
  }
     
}


L1MuDTEtaPattern::L1MuDTEtaPattern(int id, int w1, int w2, int w3, 
                                   int p1, int p2, int p3, 
                                   int eta, int qual) : 
                                
  m_id(id), m_eta(eta), m_qual(qual) { 
  
  m_wheel[0] = w1;
  m_wheel[1] = w2;
  m_wheel[2] = w3; 
  m_position[0] = p1;
  m_position[1] = p2;
  m_position[2] = p3;
  
}			 			 


L1MuDTEtaPattern::L1MuDTEtaPattern(int id, const string& pat, int eta, int qual) :
  m_id(id), m_eta(eta), m_qual(qual) {
  
  for ( int i = 0; i < 3; i++ ) {
    string sub = pat.substr(3*i,3);
    if ( sub == "___" ) {
      m_wheel[i] = 0;
      m_position[i] = 0;
    }
    else {
      m_wheel[i] = atoi(sub.substr(0,2).c_str());
      m_position[i] = atoi(sub.substr(2,1).c_str());
    }
  }
}


L1MuDTEtaPattern::L1MuDTEtaPattern(const L1MuDTEtaPattern& p) :
  m_id(p.m_id), m_eta(p.m_eta), m_qual(p.m_qual) {
  
  for (int i = 0; i < 3; i++) {
    m_wheel[i] = p.m_wheel[i]; 
    m_position[i] = p.m_position[i];
  }   

}


//--------------
// Destructor --
//--------------

L1MuDTEtaPattern::~L1MuDTEtaPattern() {}


//--------------
// Operations --
//--------------

//
// Assignment operator
//
L1MuDTEtaPattern& L1MuDTEtaPattern::operator=(const L1MuDTEtaPattern& p) {

  if ( this != &p ) {
    m_id   = p.m_id;
    m_eta  = p.m_eta;
    m_qual = p.m_qual;
    for (int i = 0; i < 3; i++) {
      m_wheel[i] = p.m_wheel[i];
      m_position[i] = p.m_position[i];
    }
  }
  return *this; 
  
}


//
// Equal operator
//
bool L1MuDTEtaPattern::operator==(const L1MuDTEtaPattern& p) const { 

  if ( m_id   != p.id() )      return false;
  if ( m_eta  != p.eta() )     return false;
  if ( m_qual != p.quality() ) return false;
  for (int i = 0; i < 3; i++) {
    if ( m_wheel[i]    != p.m_wheel[i] )    return false;
    if ( m_position[i] != p.m_position[i] ) return false;
  }
  return true;
  
}


//
// Unequal operator
//
bool L1MuDTEtaPattern::operator!=(const L1MuDTEtaPattern& p) const {

  if ( m_id   != p.id() )      return true;
  if ( m_eta  != p.eta() )     return true;
  if ( m_qual != p.quality() ) return true;
  for (int i = 0; i < 3; i++) {
    if ( m_wheel[i]    != p.m_wheel[i] )    return true;
    if ( m_position[i] != p.m_position[i] ) return true;
  }
  return false;
  
}



//
// output stream operator
//
ostream& operator<<(ostream& s, const L1MuDTEtaPattern& p) {

  s.setf(ios::right,ios::adjustfield);
  s << "ID = " << setw(8) << p.id() << "  " 
    << "quality = "  << setw(2) << p.quality()  << "  " 
    << "eta = " << setw(1) << p.eta() << endl;
    for (int i = 0; i < 3; i++) {
      s << "station = " << i+1 << " : ";
      for (int j = 0; j < 5; j++) {
        bitset<7> pos;
        if ( p.m_position[i] && (p.m_wheel[i] == j-2) ) pos.set(p.m_position[i]-1);
        s <<  pos << " ";
      }
      s << endl;
    }    

  return s;

}


//
// input stream operator
//
istream& operator>>(istream& s, L1MuDTEtaPattern& p) {

  string pat;

  s >> p.m_id >> pat >> p.m_qual >> p.m_eta;

  for ( int i = 0; i < 3; i++ ) {
    string sub = pat.substr(3*i,3);
    if ( sub == "___" ) {
      p.m_wheel[i] = 0;
      p.m_position[i] = 0;
    }
    else {
      p.m_wheel[i] = atoi(sub.substr(0,2).c_str());
      p.m_position[i] = atoi(sub.substr(2,1).c_str());
    }
  }

  return s;

}
