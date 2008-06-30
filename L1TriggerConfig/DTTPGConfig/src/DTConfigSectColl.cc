//-------------------------------------------------
//
//   Class: DTConfigSectColl
//
//   Description: Configurable parameters and constants 
//   for Level1 Mu DT Trigger - Sector Collector chip
//
//
//   Author List:
//   C. Battilana
//
//-----------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigSectColl.h"

//---------------
// C++ Headers --
//---------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//----------------
// Constructors --
//----------------
DTConfigSectColl::DTConfigSectColl(const edm::ParameterSet& ps) { 

  setDefaults(ps);
  if(debug()) print();

}

//--------------
// Destructor --
//--------------
DTConfigSectColl::~DTConfigSectColl() {}

//--------------
// Operations --
//--------------

void
DTConfigSectColl::setDefaults(const edm::ParameterSet& ps) {

  // Debug flag 
  m_debug = ps.getUntrackedParameter<bool>("Debug");

  //  Enabling Carry in Sector Collector for MB1 (1 means enabled, 0 disabled)
  m_scecf[0] = ps.getParameter<bool>("SCECF1");

  //  Enabling Carry in Sector Collector for MB2 (1 means enabled, 0 disabled)
  m_scecf[1] = ps.getParameter<bool>("SCECF2");

  //  Enabling Carry in Sector Collector for MB3 (1 means enabled, 0 disabled)
  m_scecf[2] = ps.getParameter<bool>("SCECF3");

  //  Enabling Carry in Sector Collector for MB4  (1 means enabled, 0 disabled)
  m_scecf[3] = ps.getParameter<bool>("SCECF4");

  // Progammable Coars Sync parameter in Sector Collector for MB1 (possible values [0-7])
  int mycsp = ps.getParameter<int>("SCCSP1");
  if (mycsp<0 || mycsp>7){
    std::cout << "DTConfigSectColl::setDefaults: wrong SCCSP1 value! Using Default" << std::endl;
    mycsp = default_csp;
  }
  m_sccsp[0] = mycsp;

  // Progammable Coars Sync parameter in Sector Collector for MB2 (possible values [0-7])
  mycsp = ps.getParameter<int>("SCCSP2");
  if (mycsp<0 || mycsp>7){
    std::cout << "DTConfigSectColl::setDefaults: wrong SCCSP2 value! Using Default" << std::endl;
    mycsp = default_csp;
  }
  m_sccsp[1] = mycsp;

  // Progammable Coars Sync parameter in Sector Collector for MB3 (possible values [0-7])
  mycsp = ps.getParameter<int>("SCCSP3");
  if (mycsp<0 || mycsp>7){
    std::cout << "DTConfigSectColl::setDefaults: wrong SCCSP3 value! Using Default" << std::endl;
    mycsp = default_csp;
  }
  m_sccsp[2] = mycsp;

  // Progammable Coars Sync parameter in Sector Collector for firts MB4 station (possible values [0-7])
  mycsp = ps.getParameter<int>("SCCSP4");
  if (mycsp<0 || mycsp>7){
    std::cout << "DTConfigSectColl::setDefaults: wrong SCCSP4 value! Using Default" << std::endl;
    mycsp = default_csp;
  }
  m_sccsp[3] = mycsp;

  // Progammable Coars Sync parameter in Sector Collector for second MB4 station (sectors 4 & 10) (possible values [0-7])
  mycsp = ps.getParameter<int>("SCCSP5");
  if (mycsp<0 || mycsp>7){
    std::cout << "DTConfigSectColl::setDefaults: wrong SCCSP5 value! Using Default" << std::endl;
    mycsp = default_csp;
  }
  m_sccsp[4] = mycsp;
}

void 
DTConfigSectColl::print() const {

  std::cout << "******************************************************************************" << std::endl;
  std::cout << "*              DTTrigger configuration : SectorCollector chips               *" << std::endl;
  std::cout << "******************************************************************************" << std::endl << std::endl;
  std::cout << "Debug flag : " <<  debug()     << std::endl;
  std::cout << "SCECF1 :" << SCGetCarryFlag(1) << std::endl;
  std::cout << "SCECF2 :" << SCGetCarryFlag(2) << std::endl;
  std::cout << "SCECF3 :" << SCGetCarryFlag(3) << std::endl;
  std::cout << "SCECF4 :" << SCGetCarryFlag(4) << std::endl;
  std::cout << "SCCSP1 :" << CoarseSync(1)     << std::endl;
  std::cout << "SCCSP2 :" << CoarseSync(2)     << std::endl;
  std::cout << "SCCSP3 :" << CoarseSync(3)     << std::endl;
  std::cout << "SCCSP4 :" << CoarseSync(4)     << std::endl;
  std::cout << "SCCSP5 :" << CoarseSync(5)     << std::endl;
  std::cout << "******************************************************************************" << std::endl;

}

