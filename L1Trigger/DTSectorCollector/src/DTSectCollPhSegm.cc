//-------------------------------------------------
//
/**   \Class: DTSectCollPhSegm.cc
 *
 *
 *    Muon Sector Collector Trigger Phi candidate 
 *
 *    
 *   Authors: 
 *   S. Marcellini, D. Bonacorsi
 *   Modifications: 
 *   11/11/06 C. Battilana: New Syc Functionalities implemented 
 */
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTSectorCollector/interface/DTSectCollPhSegm.h"

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
DTSectCollPhSegm::DTSectCollPhSegm(DTSectCollId scId, int step) : m_sectcollid(scId), m_step(step) { clear(); }

DTSectCollPhSegm::DTSectCollPhSegm(DTSectCollId scId, int step, const DTChambPhSegm* tsPhiTrig, int isFirst)
    : m_sectcollid(scId), m_step(step), m_isFirst(isFirst), m_tsphitrig(tsPhiTrig) {}

DTSectCollPhSegm::DTSectCollPhSegm(const DTSectCollPhSegm& seg)
    : m_sectcollid(seg.m_sectcollid), m_step(seg.m_step), m_isFirst(seg.m_isFirst), m_tsphitrig(seg.m_tsphitrig) {}

//--------------
// Destructor --
//--------------
DTSectCollPhSegm::~DTSectCollPhSegm() {}

//--------------
// Operations --
//--------------

DTSectCollPhSegm& DTSectCollPhSegm::operator=(const DTSectCollPhSegm& seg) {
  if (this != &seg) {
    m_sectcollid = seg.m_sectcollid;
    m_step = seg.m_step;
    m_tsphitrig = seg.m_tsphitrig;
    m_isFirst = seg.m_isFirst;
  }
  return *this;
}

void DTSectCollPhSegm::clear() {
  m_tsphitrig = nullptr;
  m_isFirst = 0;
}

void DTSectCollPhSegm::print() const {
  std::cout << "TP at step " << step() << ", in wheel " << wheel();
  std::cout << ", station " << station() << ", sector " << sector() << std::endl;
  std::cout << "TSS " << (m_tsphitrig->tracoTrig()->tracoNumber() - 1) / DTConfig::NTCTSS + 1;
  std::cout << ", TRACO " << tracoNumber() << " : " << std::endl;
  std::cout << "  -->  code " << oldCode() << ", K " << K();
  std::cout << ", X " << X() << ", position mask " << posMask() << std::endl;
  std::cout << "inner bti equation=" << m_tsphitrig->tracoTrig()->eqIn()
            << "    outer bti equation=" << m_tsphitrig->tracoTrig()->eqOut() << std::endl;
  std::cout << "        psi " << psi() << ", psiR " << psiR();
  std::cout << ", DeltaPsiR " << DeltaPsiR() << std::endl;
}
