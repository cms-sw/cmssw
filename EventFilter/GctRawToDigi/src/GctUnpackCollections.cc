#include "EventFilter/GctRawToDigi/src/GctUnpackCollections.h"


GctUnpackCollections::GctUnpackCollections(edm::Event& event, const bool hltMode):
  m_event(event),
  m_hltMode(hltMode),
  m_gctFibres(new L1GctFibreCollection()),  // GCT input collections
  m_rctEm(new L1CaloEmCollection()),
  m_rctCalo(new L1CaloRegionCollection()),
  m_gctInternEm(new L1GctInternEmCandCollection()),  // GCT internal collections
  m_gctInternJets(new L1GctInternJetDataCollection()),
  m_gctInternEtSums(new L1GctInternEtSumCollection()),
  m_gctIsoEm(new L1GctEmCandCollection()),  // GCT output collections
  m_gctNonIsoEm(new L1GctEmCandCollection()),
  m_gctCenJets(new L1GctJetCandCollection()),
  m_gctForJets(new L1GctJetCandCollection()),
  m_gctTauJets(new L1GctJetCandCollection()),
  m_gctHfBitCounts(new L1GctHFBitCountsCollection()),
  m_gctHfRingEtSums(new L1GctHFRingEtSumsCollection()),
  m_gctEtTot(new L1GctEtTotalCollection()),
  m_gctEtHad(new L1GctEtHadCollection()),
  m_gctEtMiss(new L1GctEtMissCollection()),
  m_gctJetCounts(new L1GctJetCountsCollection())  // Deprecated (empty collection still needed by GT)
{
  m_gctIsoEm->reserve(4);
  m_gctCenJets->reserve(4);
  m_gctForJets->reserve(4);
  m_gctTauJets->reserve(4);
  // ** DON'T RESERVE SPACE IN VECTORS FOR DEBUG UNPACK ITEMS! **
}

GctUnpackCollections::~GctUnpackCollections()
{
  // GCT input collections
  if(!m_hltMode) { m_event.put(m_gctFibres); }
  m_event.put(m_rctEm);
  m_event.put(m_rctCalo);

  // GCT internal collections
  if(!m_hltMode)
  {
    m_event.put(m_gctInternEm);
    m_event.put(m_gctInternJets);
    m_event.put(m_gctInternEtSums);
  }

  // GCT output collections
  m_event.put(m_gctIsoEm, "isoEm");
  m_event.put(m_gctNonIsoEm, "nonIsoEm");
  m_event.put(m_gctCenJets,"cenJets");
  m_event.put(m_gctForJets,"forJets");
  m_event.put(m_gctTauJets,"tauJets");
  m_event.put(m_gctHfBitCounts);
  m_event.put(m_gctHfRingEtSums);
  m_event.put(m_gctEtTot);
  m_event.put(m_gctEtHad);
  m_event.put(m_gctEtMiss);
  m_event.put(m_gctJetCounts);  // Deprecated (empty collection still needed by GT)
}

std::ostream& operator<<(std::ostream& os, const GctUnpackCollections& rhs)
{
     // GCT input collections
  os << "Read " << rhs.gctFibres()->size() << " GCT raw fibre data\n"
     << "Read " << rhs.rctEm()->size() << " RCT EM candidates\n"
     << "Read " << rhs.rctCalo()->size() << " RCT Calo Regions\n"

     // GCT internal collections
     << "Read " << rhs.gctInternEm()->size() << " GCT intermediate EM candidates\n"
     << "Read " << rhs.gctInternJets()->size() << " GCT intermediate jet candidates\n"
     << "Read " << rhs.gctInternEtSums()->size() << " GCT intermediate et sums\n"

     // GCT output collections
     << "Read " << rhs.gctIsoEm()->size() << " GCT iso EM candidates\n"
     << "Read " << rhs.gctNonIsoEm()->size() << " GCT non-iso EM candidates\n"
     << "Read " << rhs.gctCenJets()->size() << " GCT central jet candidates\n"
     << "Read " << rhs.gctForJets()->size() << " GCT forward jet candidates\n"
     << "Read " << rhs.gctTauJets()->size() << " GCT tau jet candidates\n"
     << "Read " << rhs.gctHfBitCounts()->size() << " GCT HF ring bit counts\n"
     << "Read " << rhs.gctHfRingEtSums()->size() << " GCT HF ring et sums\n"
     << "Read " << rhs.gctEtTot()->size() << " GCT total et\n"
     << "Read " << rhs.gctEtHad()->size() << " GCT ht\n"
     << "Read " << rhs.gctEtMiss()->size() << " GCT met\n";
     
  return os;
}
