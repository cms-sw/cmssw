#include "EventFilter/RctRawToDigi/src/RctUnpackCollections.h"


RctUnpackCollections::RctUnpackCollections(edm::Event& event):
  m_event(event),
  m_rctEm(new L1CaloEmCollection()),
  m_rctCalo(new L1CaloRegionCollection())
{
  //m_rctIsoEm->reserve(4);
  //m_rctCenJets->reserve(4);
  //m_rctForJets->reserve(4);
  //m_rctTauJets->reserve(4);
  // ** DON'T RESERVE SPACE IN VECTORS FOR DEBUG UNPACK ITEMS! **
}

RctUnpackCollections::~RctUnpackCollections()
{
  // RCT input collections
  m_event.put(m_rctEm);
  m_event.put(m_rctCalo);

}

std::ostream& operator<<(std::ostream& os, const RctUnpackCollections& rhs)
{
     // RCT input collections
  os << "Read " << rhs.rctEm()->size() << " RCT EM candidates\n"
     << "Read " << rhs.rctCalo()->size() << " RCT Calo Regions\n";

  return os;
}
