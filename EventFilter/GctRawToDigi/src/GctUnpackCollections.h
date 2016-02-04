#ifndef GctUnpackCollections_h
#define GctUnpackCollections_h

/*!
* \class GctUnpackCollections
* \brief RAII and useful methods for the many dataformat collections required by the GCT unpacker.
* 
*  Deliberately made non-copyable with const members and private copy ctor, etc.
*
* \author Robert Frazier
* $Revision: 1.1 $
* $Date: 2009/04/07 10:51:06 $
*/ 

// CMSSW headers
#include "FWCore/Framework/interface/Event.h"

// DataFormat headers
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1Trigger/interface/L1TriggerError.h"


class GctUnpackCollections
{
public:
  /// Construct with an event. The collections get put into the event when the object instance goes out of scope (i.e. in the destructor).
  GctUnpackCollections(edm::Event& event);

  /// Destructor - the last action of this object is to put the gct collections into the event provided on construction.
  ~GctUnpackCollections();

  // Collections for storing GCT input data.  
  L1GctFibreCollection * const gctFibres() const { return m_gctFibres.get(); }  ///< Raw fibre input to the GCT.
  L1CaloEmCollection * const rctEm() const { return m_rctEm.get(); } ///< Input electrons from the RCT to the GCT.
  L1CaloRegionCollection * const rctCalo() const { return m_rctCalo.get(); } ///< Input calo regions from the RCT to the GCT.

  // GCT intermediate data
  L1GctInternEmCandCollection * const gctInternEm() const { return m_gctInternEm.get(); }  ///< Internal EM candidate collection
  L1GctInternJetDataCollection * const gctInternJets() const { return m_gctInternJets.get(); } ///< Internal Jet candidate collection
  L1GctInternEtSumCollection * const gctInternEtSums() const { return m_gctInternEtSums.get(); } ///< Internal Et Sum collection
  L1GctInternHFDataCollection * const gctInternHFData() const { return m_gctInternHFData.get(); } ///< Internal Hadronic-Forward bit-counts/ring-sums data collection
  L1GctInternHtMissCollection * const gctInternHtMiss() const { return m_gctInternHtMiss.get(); } ///< Internal missing Ht collection

  // GCT output data
  L1GctEmCandCollection * const gctIsoEm() const { return m_gctIsoEm.get(); }  ///< GCT output: Isolated EM candidate collection
  L1GctEmCandCollection * const gctNonIsoEm() const { return m_gctNonIsoEm.get(); } ///< GCT output: Non-isolated EM candidate collection
  L1GctJetCandCollection * const gctCenJets() const { return m_gctCenJets.get(); } ///< GCT output: Central Jets collection
  L1GctJetCandCollection * const gctForJets() const { return m_gctForJets.get(); } ///< GCT output: Forward Jets collection
  L1GctJetCandCollection * const gctTauJets() const { return m_gctTauJets.get(); } ///< GCT output: Tau Jets collection
  L1GctHFBitCountsCollection * const gctHfBitCounts() const { return m_gctHfBitCounts.get(); } ///< GCT output: Hadronic-Forward bit-counts collection
  L1GctHFRingEtSumsCollection * const gctHfRingEtSums() const { return m_gctHfRingEtSums.get(); }  ///< GCT output: Hadronic-Forward ring-sums collection
  L1GctEtTotalCollection * const gctEtTot() const { return m_gctEtTot.get(); }  ///< GCT output: Total Et collection
  L1GctEtHadCollection * const gctEtHad() const { return m_gctEtHad.get(); }  ///< GCT output: Hadronic transverse-energy (Ht) collection
  L1GctEtMissCollection * const gctEtMiss() const { return m_gctEtMiss.get(); }  ///< GCT output: Missing Et collection
  L1GctHtMissCollection * const gctHtMiss() const { return m_gctHtMiss.get(); }  ///< GCT output: Missing Ht collection
  L1GctJetCountsCollection * const gctJetCounts() const { return m_gctJetCounts.get(); } ///< DEPRECATED. ONLY GT NEEDS THIS.

  // Misc
  L1TriggerErrorCollection * const errors() const { return m_errors.get(); }  ///< Unpack error code collection.

private:

  GctUnpackCollections(const GctUnpackCollections&); ///< Copy ctor - deliberately not implemented!
  GctUnpackCollections& operator=(const GctUnpackCollections&); ///< Assignment op - deliberately not implemented!  

  edm::Event& m_event;  ///< The event the collections will be put into on destruction of the GctUnpackCollections instance.

  // Collections for storing GCT input data.  
  std::auto_ptr<L1GctFibreCollection> m_gctFibres;  ///< Raw fibre input to the GCT.
  std::auto_ptr<L1CaloEmCollection> m_rctEm; ///< Input electrons.
  std::auto_ptr<L1CaloRegionCollection> m_rctCalo; ///< Input calo regions.

  // GCT intermediate data
  std::auto_ptr<L1GctInternEmCandCollection> m_gctInternEm; 
  std::auto_ptr<L1GctInternJetDataCollection> m_gctInternJets; 
  std::auto_ptr<L1GctInternEtSumCollection> m_gctInternEtSums; 
  std::auto_ptr<L1GctInternHFDataCollection> m_gctInternHFData; 
  std::auto_ptr<L1GctInternHtMissCollection> m_gctInternHtMiss;

  // GCT output data
  std::auto_ptr<L1GctEmCandCollection> m_gctIsoEm;
  std::auto_ptr<L1GctEmCandCollection> m_gctNonIsoEm;
  std::auto_ptr<L1GctJetCandCollection> m_gctCenJets;
  std::auto_ptr<L1GctJetCandCollection> m_gctForJets;
  std::auto_ptr<L1GctJetCandCollection> m_gctTauJets;
  std::auto_ptr<L1GctHFBitCountsCollection> m_gctHfBitCounts;
  std::auto_ptr<L1GctHFRingEtSumsCollection> m_gctHfRingEtSums;
  std::auto_ptr<L1GctEtTotalCollection> m_gctEtTot;
  std::auto_ptr<L1GctEtHadCollection> m_gctEtHad;
  std::auto_ptr<L1GctEtMissCollection> m_gctEtMiss;
  std::auto_ptr<L1GctHtMissCollection> m_gctHtMiss;
  std::auto_ptr<L1GctJetCountsCollection> m_gctJetCounts;  // DEPRECATED. ONLY GT NEEDS THIS.
  
  // Misc
  std::auto_ptr<L1TriggerErrorCollection> m_errors;

};

// Pretty print for the GctUnpackCollections sub-class
std::ostream& operator<<(std::ostream& os, const GctUnpackCollections& rhs);

#endif /* GctUnpackCollections_h */
