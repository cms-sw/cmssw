#ifndef GtBoard_h
#define GtBoard_h

/**
 * \class GlobalBoard
 *
 *
 * Description: Global Trigger Logic board.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 */

// system include files
#include <bitset>
#include <cassert>
#include <vector>
#include <cmath>
#include <memory>

// user include files
#include "FWCore/Utilities/interface/typedefs.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/L1TGlobal/interface/GlobalObjectMapRecord.h"

#include "L1Trigger/L1TGlobal/interface/AlgorithmEvaluation.h"

// Trigger Objects
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/MuonShower.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "L1Trigger/L1TGlobal/interface/GlobalScales.h"

// Objects to produce for the output record.
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

// forward declarations
class TriggerMenu;
class L1CaloGeometry;
class L1MuTriggerScales;
//class L1GtEtaPhiConversions;

// class declaration

namespace l1t {

  class GlobalBoard {
  public:
    // constructors
    GlobalBoard();

    // destructor
    virtual ~GlobalBoard();

  public:
    /// receive data from Global Muon Trigger
    void receiveCaloObjectData(const edm::Event&,
                               const edm::EDGetTokenT<BXVector<l1t::EGamma>>&,
                               const edm::EDGetTokenT<BXVector<l1t::Tau>>&,
                               const edm::EDGetTokenT<BXVector<l1t::Jet>>&,
                               const edm::EDGetTokenT<BXVector<l1t::EtSum>>&,
                               const edm::EDGetTokenT<BXVector<l1t::EtSum>>&,
                               const bool receiveEG,
                               const int nrL1EG,
                               const bool receiveTau,
                               const int nrL1Tau,
                               const bool receiveJet,
                               const int nrL1Jet,
                               const bool receiveEtSums,
                               const bool receiveEtSumsZdc);

    void receiveMuonObjectData(const edm::Event&,
                               const edm::EDGetTokenT<BXVector<l1t::Muon>>&,
                               const bool receiveMu,
                               const int nrL1Mu);

    void receiveMuonShowerObjectData(const edm::Event&,
                                     const edm::EDGetTokenT<BXVector<l1t::MuonShower>>&,
                                     const bool receiveMuShower,
                                     const int nrL1MuShower);

    void receiveExternalData(const edm::Event&, const edm::EDGetTokenT<BXVector<GlobalExtBlk>>&, const bool receiveExt);

    /// initialize the class (mainly reserve)
    void init(const int numberPhysTriggers,
              const int nrL1Mu,
              const int nrL1MuShower,
              const int nrL1EG,
              const int nrL1Tau,
              const int nrL1Jet,
              int bxFirst,
              int bxLast);

    /// run the uGT GTL (Conditions and Algorithms)
    void runGTL(const edm::Event& iEvent,
                const edm::EventSetup& evSetup,
                const TriggerMenu* m_l1GtMenu,
                const bool produceL1GtObjectMapRecord,
                const int iBxInEvent,
                std::unique_ptr<GlobalObjectMapRecord>& gtObjectMapRecord,  //GTO
                const unsigned int numberPhysTriggers,
                const int nrL1Mu,
                const int nrL1MuShower,
                const int nrL1EG,
                const int nrL1Tau,
                const int nrL1Jet);

    /// run the uGT FDL (Apply Prescales and Veto)
    void runFDL(const edm::Event& iEvent,
                const int iBxInEvent,
                const int totalBxInEvent,
                const unsigned int numberPhysTriggers,
                const std::vector<double>& prescaleFactorsAlgoTrig,
                const std::vector<unsigned int>& triggerMaskAlgoTrig,
                const std::vector<int>& triggerMaskVetoAlgoTrig,
                const bool algorithmTriggersUnprescaled,
                const bool algorithmTriggersUnmasked);

    /// Fill the Daq Records
    void fillAlgRecord(int iBxInEvent,
                       std::unique_ptr<GlobalAlgBlkBxCollection>& uGtAlgRecord,
                       int prescaleSet,
                       int menuUUID,
                       int firmwareUUID);

    /// clear uGT
    void reset();
    void resetMu();
    void resetMuonShower();
    void resetCalo();
    void resetExternal();

    /// print received Muon dataWord
    void printGmtData(const int iBxInEvent) const;

    /// return decision
    inline const std::bitset<GlobalAlgBlk::maxPhysicsTriggers>& getDecisionWord() const { return m_gtlDecisionWord; }

    /// return algorithm OR decision
    inline const std::bitset<GlobalAlgBlk::maxPhysicsTriggers>& getAlgorithmOR() const { return m_gtlAlgorithmOR; }

    /// return global muon trigger candidate
    inline const BXVector<const l1t::Muon*>* getCandL1Mu() const { return m_candL1Mu; }

    /// return global muon trigger candidate
    inline const BXVector<std::shared_ptr<l1t::MuonShower>>* getCandL1MuShower() const { return m_candL1MuShower; }

    /// pointer to EG data list
    inline const BXVector<const l1t::L1Candidate*>* getCandL1EG() const { return m_candL1EG; }

    /// pointer to Jet data list
    inline const BXVector<const l1t::L1Candidate*>* getCandL1Jet() const { return m_candL1Jet; }

    /// pointer to Tau data list
    inline const BXVector<const l1t::L1Candidate*>* getCandL1Tau() const { return m_candL1Tau; }

    /// pointer to EtSum data list
    inline const BXVector<const l1t::EtSum*>* getCandL1EtSum() const { return m_candL1EtSum; }

    /// pointer to ZDC EtSum data list
    inline const BXVector<const l1t::EtSum*>* getCandL1EtSumZdc() const { return m_candL1EtSumZdc; }

    /// pointer to External data list
    inline const BXVector<const GlobalExtBlk*>* getCandL1External() const { return m_candL1External; }

    /*  Drop individual EtSums for Now
    /// pointer to ETM data list
    inline const l1t::EtSum* getCandL1ETM() const
    {
        return m_candETM;
    }

    /// pointer to ETT data list
    inline const l1t::EtSum* getCandL1ETT() const
    {
        return m_candETT;
    }

    /// pointer to HTT data list
    inline const l1t::EtSum* getCandL1HTT() const
    {
        return m_candHTT;
    }

    /// pointer to HTM data list
    inline const l1t::EtSum* getCandL1HTM() const
    {
        return m_candHTM;
    }
*/

    void setBxFirst(int bx);
    void setBxLast(int bx);

    void setResetPSCountersEachLumiSec(bool val) { m_resetPSCountersEachLumiSec = val; }
    void setSemiRandomInitialPSCounters(bool val) { m_semiRandomInitialPSCounters = val; }

    void setAXOL1TLModelVersion(std::string axol1tlModelVersion);

  public:
    inline void setVerbosity(const int verbosity) { m_verbosity = verbosity; }

  private:
    // cached stuff

    // trigger menu
    const TriggerMenu* m_l1GtMenu;
    unsigned long long m_l1GtMenuCacheID;

    // L1 scales (phi, eta) for Mu, Calo and EnergySum objects
    const L1CaloGeometry* m_l1CaloGeometry;
    unsigned long long m_l1CaloGeometryCacheID;

    const L1MuTriggerScales* m_l1MuTriggerScales;
    unsigned long long m_l1MuTriggerScalesCacheID;

    // conversions for eta and phi
    //    L1GtEtaPhiConversions* m_gtEtaPhiConversions;

  private:
    BXVector<const l1t::Muon*>* m_candL1Mu;
    BXVector<std::shared_ptr<l1t::MuonShower>>* m_candL1MuShower;
    BXVector<const l1t::L1Candidate*>* m_candL1EG;
    BXVector<const l1t::L1Candidate*>* m_candL1Tau;
    BXVector<const l1t::L1Candidate*>* m_candL1Jet;
    BXVector<const l1t::EtSum*>* m_candL1EtSum;
    BXVector<const l1t::EtSum*>* m_candL1EtSumZdc;
    BXVector<const GlobalExtBlk*>* m_candL1External;

    //    BXVector<const l1t::EtSum*>* m_candETM;
    //    BXVector<const l1t::EtSum*>* m_candETT;
    //    BXVector<const l1t::EtSum*>* m_candHTM;
    //    BXVector<const l1t::EtSum*>* m_candHTT;

    int m_bxFirst_;
    int m_bxLast_;

    std::string m_axol1tlModelVersion = "NULL";
    
    std::bitset<GlobalAlgBlk::maxPhysicsTriggers> m_gtlAlgorithmOR;
    std::bitset<GlobalAlgBlk::maxPhysicsTriggers> m_gtlDecisionWord;

    GlobalAlgBlk m_uGtAlgBlk;

    // cache of maps
    std::vector<AlgorithmEvaluation::ConditionEvaluationMap> m_conditionResultMaps;

    unsigned int m_currentLumi;

  private:
    /// verbosity level
    int m_verbosity;
    bool m_isDebugEnabled;

    // Flags for the OR of all algorithms at various stages. (Single bx)
    bool m_algInitialOr;
    bool m_algIntermOr;
    bool m_algPrescaledOr;
    bool m_algFinalOr;
    bool m_algFinalOrVeto;

    // Counter for number of events seen by this board
    unsigned int m_boardEventCount;

    // Information about board
    int m_uGtBoardNumber;
    bool m_uGtFinalBoard;

    // whether we reset the prescales each lumi or not
    bool m_resetPSCountersEachLumiSec = false;

    // start the PS counter from a random value between [1,PS] instead of PS
    bool m_semiRandomInitialPSCounters = false;

    // step-size in prescale counter corresponding to 10^p,
    // where p is the precision allowed for non-integer prescales;
    // since the introduction of L1T fractional prescales, p == 2
    static constexpr size_t m_singlestep = 100;

    // struct to increment the prescale according to fractional prescale logic in firmware
    struct PrescaleCounter {
      size_t const prescale_count;
      size_t trigger_counter;

      PrescaleCounter(double prescale, size_t const initial_counter = 0)
          : prescale_count(std::lround(prescale * m_singlestep)), trigger_counter(initial_counter) {
        if (prescale_count != 0 and (prescale_count < m_singlestep or prescale < 0)) {
          throw cms::Exception("PrescaleCounterConstructor")
              << "invalid initialisation of PrescaleCounter: prescale = " << prescale
              << ", prescale_count = " << prescale_count << " (< " << m_singlestep << " = m_singlestep)";
        }
      }

      // function to increment the prescale counter and return the decision
      bool accept();
    };

    // prescale counters: NumberPhysTriggers counters per bunch cross in event
    std::vector<std::vector<PrescaleCounter>> m_prescaleCounterAlgoTrig;

    // create prescale counters, initialising trigger_counter to zero
    static std::vector<PrescaleCounter> prescaleCounters(std::vector<double> const& prescaleFactorsAlgoTrig);

    // create prescale counters, initialising trigger_counter to a semirandom number between 0 and prescale_count - 1 inclusive
    static std::vector<PrescaleCounter> prescaleCountersWithSemirandomInitialCounter(
        std::vector<double> const& prescaleFactorsAlgoTrig, edm::Event const& iEvent);
  };

}  // namespace l1t

#endif
