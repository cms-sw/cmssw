//-------------------------------------------------
//
//   Class: DTChambThSegm.cpp
//
//   Description: Muon Chamber Trigger Theta candidate 
//
//
//   Author List:
//   C. Grandi
//   Modifications: 
//
//
//--------------------------------------------------

// #include "Utilities/Configuration/interface/Architecture.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTTriggerServerTheta/interface/DTChambThSegm.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//---------------
// C++ Headers --
//---------------
#include <iostream>

//----------------
// Constructors --
//----------------
DTChambThSegm::DTChambThSegm(DTChamberId chamberid, int step, 
				     int* pos, int* qual)
  : m_chamberid(chamberid),  m_step(step) {

  for(int i=0;i<7;i++) {
    m_outPos[i] = pos[i];
    m_outQual[i] = qual[i];
  }
}

DTChambThSegm::DTChambThSegm(const DTChambThSegm& seg) : 
  m_chamberid(seg.m_chamberid), m_step(seg.m_step)  {

  for(int i=0;i<7;i++) {
    m_outPos[i] = seg.m_outPos[i];
    m_outQual[i] = seg.m_outQual[i];
  }
}

//--------------
// Destructor --
//--------------
DTChambThSegm::~DTChambThSegm(){
}

//--------------
// Operations --
//--------------

DTChambThSegm&
DTChambThSegm::operator=(const DTChambThSegm& seg){
  if(this != &seg){
    m_chamberid = seg.m_chamberid;
    m_step = seg.m_step;
    for(int i=0;i<7;i++) {
      m_outPos[i] = seg.m_outPos[i];
      m_outQual[i] = seg.m_outQual[i];
    }
  }
  return *this;
}

int 
DTChambThSegm::code(const int i) const {
  if(i<0||i>=7){
    std::cout << "DTChambThSegm::code : index out of range: " << i;
    std::cout << "0 returned!" << std::endl;
    return 0;
  }
  return (int)(m_outPos[i]+m_outQual[i]);
}

int 
DTChambThSegm::position(const int i) const {
  if(i<0||i>=7){
    std::cout << "DTChambThSegm::position : index out of range: " << i;
    std::cout << "0 returned!" << std::endl;
    return 0;
  }
  return (int)m_outPos[i];
}

int 
DTChambThSegm::quality(const int i) const {
  if(i<0||i>=7){
    std::cout << "DTChambThSegm::quality : index out of range: " << i;
    std::cout << "0 returned!" << std::endl;
    return 0;
  }
  return (int)m_outQual[i];
}

void
DTChambThSegm::print() const {
  std::cout << "TP at step " << step() << ", in wheel " << wheel();
  std::cout << ", station " << station() << ", sector " << sector();
  std::cout << " : " << std::endl;
  std::cout << "  output codes : ";
  int i=0;
  for(i=0;i<7;i++){
    std::cout << (int)(m_outPos[i]+m_outQual[i]) << " ";
  }
  std::cout << "\n";
  
}
