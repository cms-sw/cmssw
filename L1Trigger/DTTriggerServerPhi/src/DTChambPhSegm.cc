//-------------------------------------------------
//
//   Class: DTChambPhSegm.cpp
//
//   Description: Muon Chamber Trigger Phi candidate 
//
//
//   Author List:
//   C. Grandi
//   Modifications: 
//
//
//--------------------------------------------------
 
//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTBti/interface/DTBtiTrigData.h"
#include "L1Trigger/DTTraco/interface/DTTracoTrig.h"
#include "L1Trigger/DTTriggerServerPhi/interface/DTChambPhSegm.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigTSPhi.h"

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
DTChambPhSegm::DTChambPhSegm(DTChamberId chamberid, int step) : 
                                         m_chamberid(chamberid), m_step(step) {
  clear();
}

DTChambPhSegm::DTChambPhSegm(DTChamberId chamberid, int step, 
				     const DTTracoTrigData* tracotrig, 
				     int isfirst) :
                                     m_chamberid(chamberid), m_step(step),
                                     m_isFirst(isfirst),
                                     m_tracotrig(tracotrig)  {
}
  
DTChambPhSegm::DTChambPhSegm(const DTChambPhSegm& seg) : 
  m_chamberid(seg.m_chamberid), m_step(seg.m_step), m_isFirst(seg.m_isFirst), 
  m_tracotrig(seg.m_tracotrig) {
}

//--------------
// Destructor --
//--------------
DTChambPhSegm::~DTChambPhSegm() {
}

//--------------
// Operations --
//--------------

DTChambPhSegm&
DTChambPhSegm::operator=(const DTChambPhSegm& seg){
  if(this != &seg){
    m_chamberid = seg.m_chamberid;
    m_step = seg.m_step;
    m_tracotrig = seg.m_tracotrig ;
    m_isFirst = seg.m_isFirst ;
  }
  return *this;
}

void 
DTChambPhSegm::clear() { 
  m_tracotrig = 0;
  m_isFirst = 0;
}

void
DTChambPhSegm::print() const {
  std::cout << "TP at step " << step() << ", in wheel " << wheel();
  std::cout << ", station " << station() << ", sector " << sector() << std::endl;
  std::cout << "TSS " << (tracoTrig()->tracoNumber()-1) / DTConfigTSPhi::NTCTSS + 1;
  std::cout << ", TRACO " << tracoNumber() << " : " << std::endl;
  std::cout << "  -->  code " << oldCode() << ", K " << K();
  std::cout << ", X " << X() << ", position mask " << posMask() << std::endl;
  std::cout << "inner bti equation=" << tracoTrig()->eqIn() <<
          "    outer bti equation=" << tracoTrig()->eqOut() << std::endl;
  std::cout << "        psi " << psi() << ", psiR " << psiR();
  std::cout << ", DeltaPsiR " << DeltaPsiR() << std::endl;
}






