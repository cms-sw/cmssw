#ifndef DQM_L1TMonitor_L1ExtraDQM_h
#define DQM_L1TMonitor_L1ExtraDQM_h

/**
 * \class L1ExtraDQM
 *
 *
 * Description: online DQM module for L1Extra trigger objects.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 *
 * $Date$
 *
 */

// system include files
#include <iosfwd>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>

// user include files
//   base classes
#include "FWCore/Framework/interface/EDAnalyzer.h"

//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// L1Extra objects
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1HFRings.h"
#include "DataFormats/L1Trigger/interface/L1HFRingsFwd.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1PhiConversion.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GetHistLimits.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1RetrieveL1Extra.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "boost/lexical_cast.hpp"
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"

// forward declarations

// class declaration
class L1ExtraDQM : public DQMOneEDAnalyzer<> {
public:
  // constructor(s)
  explicit L1ExtraDQM(const edm::ParameterSet&);

  // destructor
  ~L1ExtraDQM() override;

public:
  template <class CollectionType>
  class L1ExtraMonElement {
  public:
    // constructor
    L1ExtraMonElement(const edm::EventSetup&, const int);

    // destructor
    virtual ~L1ExtraMonElement();

  public:
    typedef typename CollectionType::const_iterator CIterColl;

    void bookhistograms(const edm::EventSetup& evSetup,
                        DQMStore::IBooker& ibooker,
                        const std::string& l1ExtraObject,
                        const std::vector<L1GtObject>& l1GtObj,
                        const bool bookPhi = true,
                        const bool bookEta = true);

    /// number of objects
    void fillNrObjects(const CollectionType* collType, const bool validColl, const bool isL1Coll, const int bxInEvent);

    /// PT, eta, phi
    void fillPtPhiEta(const CollectionType* collType,
                      const bool validColl,
                      const bool bookPhi,
                      const bool bookEta,
                      const bool isL1Coll,
                      const int bxInEvent);

    /// ET, eta, phi
    void fillEtPhiEta(const CollectionType* collType,
                      const bool validColl,
                      const bool bookPhi,
                      const bool bookEta,
                      const bool isL1Coll,
                      const int bxInEvent);

    /// fill ET total in energy sums
    void fillEtTotal(const CollectionType* collType, const bool validColl, const bool isL1Coll, const int bxInEvent);

    /// fill charge
    void fillCharge(const CollectionType* collType, const bool validColl, const bool isL1Coll, const int bxInEvent);

    /// fill bit counts in HFRings collections
    void fillHfBitCounts(const CollectionType* collType,
                         const bool validColl,
                         const int countIndex,
                         const bool isL1Coll,
                         const int bxInEvent);

    /// fill energy sums in HFRings collections
    void fillHfRingEtSums(const CollectionType* collType,
                          const bool validColl,
                          const int countIndex,
                          const bool isL1Coll,
                          const int bxInEvent);

  private:
    std::vector<MonitorElement*> m_monElement;

    /// histogram index for each quantity, set during histogram booking
    int m_indexNrObjects;
    int m_indexPt;
    int m_indexEt;
    int m_indexPhi;
    int m_indexEta;
    int m_indexEtTotal;
    int m_indexCharge;
    int m_indexHfBitCounts;
    int m_indexHfRingEtSums;
  };

protected:
  void analyzeL1ExtraMuon(const edm::Event&, const edm::EventSetup&);
  void analyzeL1ExtraIsoEG(const edm::Event&, const edm::EventSetup&);
  void analyzeL1ExtraNoIsoEG(const edm::Event&, const edm::EventSetup&);
  void analyzeL1ExtraCenJet(const edm::Event&, const edm::EventSetup&);
  void analyzeL1ExtraForJet(const edm::Event&, const edm::EventSetup&);
  void analyzeL1ExtraTauJet(const edm::Event&, const edm::EventSetup&);
  void analyzeL1ExtraIsoTauJet(const edm::Event&, const edm::EventSetup&);
  void analyzeL1ExtraETT(const edm::Event&, const edm::EventSetup&);
  void analyzeL1ExtraETM(const edm::Event&, const edm::EventSetup&);
  void analyzeL1ExtraHTT(const edm::Event&, const edm::EventSetup&);
  void analyzeL1ExtraHTM(const edm::Event&, const edm::EventSetup&);
  void analyzeL1ExtraHfBitCounts(const edm::Event&, const edm::EventSetup&);
  void analyzeL1ExtraHfRingEtSums(const edm::Event&, const edm::EventSetup&);

  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) override;
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void dqmEndRun(const edm::Run& run, const edm::EventSetup& evSetup) override;

private:
  /// input parameters

  L1RetrieveL1Extra m_retrieveL1Extra;
  edm::InputTag L1ExtraIsoTauJetSource;
  /// directory name for L1Extra plots
  std::string m_dirName;
  bool m_stage1_layer2_;

  /// number of bunch crosses in event to be monitored
  int m_nrBxInEventGmt;
  int m_nrBxInEventGct;

  /// internal members

  bool m_resetModule;
  int m_currentRun;

  ///
  int m_nrEvJob;
  int m_nrEvRun;

private:
  edm::EDGetTokenT<l1extra::L1JetParticleCollection> m_tagL1ExtraIsoTauJetTok;

  /// pointers to L1ExtraMonElement for each sub-analysis
  std::vector<L1ExtraMonElement<l1extra::L1MuonParticleCollection>*> m_meAnalysisL1ExtraMuon;

  std::vector<L1ExtraMonElement<l1extra::L1EmParticleCollection>*> m_meAnalysisL1ExtraIsoEG;
  std::vector<L1ExtraMonElement<l1extra::L1EmParticleCollection>*> m_meAnalysisL1ExtraNoIsoEG;

  std::vector<L1ExtraMonElement<l1extra::L1JetParticleCollection>*> m_meAnalysisL1ExtraCenJet;
  std::vector<L1ExtraMonElement<l1extra::L1JetParticleCollection>*> m_meAnalysisL1ExtraForJet;
  std::vector<L1ExtraMonElement<l1extra::L1JetParticleCollection>*> m_meAnalysisL1ExtraTauJet;
  std::vector<L1ExtraMonElement<l1extra::L1JetParticleCollection>*> m_meAnalysisL1ExtraIsoTauJet;

  std::vector<L1ExtraMonElement<l1extra::L1EtMissParticleCollection>*> m_meAnalysisL1ExtraETT;

  std::vector<L1ExtraMonElement<l1extra::L1EtMissParticleCollection>*> m_meAnalysisL1ExtraETM;

  std::vector<L1ExtraMonElement<l1extra::L1EtMissParticleCollection>*> m_meAnalysisL1ExtraHTT;

  std::vector<L1ExtraMonElement<l1extra::L1EtMissParticleCollection>*> m_meAnalysisL1ExtraHTM;

  std::vector<L1ExtraMonElement<l1extra::L1HFRingsCollection>*> m_meAnalysisL1ExtraHfBitCounts;

  std::vector<L1ExtraMonElement<l1extra::L1HFRingsCollection>*> m_meAnalysisL1ExtraHfRingEtSums;
};

#endif
