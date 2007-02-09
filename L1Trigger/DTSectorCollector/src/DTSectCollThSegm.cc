//-------------------------------------------------
//
//   Class: DTSectCollThSegm.cpp
//
//   Description: Muon Sector Collector Trigger Theta candidate 
//
//
//   Author List:
//   C. Battilana
//   Modifications: 
//
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTSectorCollector/interface/DTSectCollThSegm.h"

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
DTSectCollThSegm::DTSectCollThSegm(DTSectCollId scid, int step, 
				     const DTChambThSegm* tstheta_seg) : 
  m_sectcollid(scid),  m_step(step), m_tsthetatrig(tstheta_seg) {
}

DTSectCollThSegm::DTSectCollThSegm(const DTSectCollThSegm& seg) : 
  m_sectcollid(seg.m_sectcollid), m_step(seg.m_step), m_tsthetatrig(seg.m_tsthetatrig)  {
}

//--------------
// Destructor --
//--------------
DTSectCollThSegm::~DTSectCollThSegm(){
}

//--------------
// Operations --
//--------------

DTSectCollThSegm&
DTSectCollThSegm::operator=(const DTSectCollThSegm& seg){
  if(this != &seg){
    m_sectcollid = seg.m_sectcollid;
    m_step = seg.m_step;
    m_tsthetatrig = seg.m_tsthetatrig;
  }
  return *this;
}

void
DTSectCollThSegm::print() const {
  std::cout << "TP at step " << step() << ", in wheel " << wheel();
  std::cout << ", station " << station() << ", sector " << sector();
  std::cout << " : " << std::endl;
  std::cout << "  output codes : ";
  int i=0;
  for(i=0;i<7;i++){
    std::cout << (int)(position(i)+quality(i)) << " ";
  }
  std::cout << "\n";
  
}
