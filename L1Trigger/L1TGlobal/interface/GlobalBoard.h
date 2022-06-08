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

// user include files
#include "FWCore/Utilities/interface/typedefs.h"
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
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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
    void receiveCaloObjectData(edm::Event&,
                               const edm::EDGetTokenT<BXVector<l1t::EGamma>>&,
                               const edm::EDGetTokenT<BXVector<l1t::Tau>>&,
                               const edm::EDGetTokenT<BXVector<l1t::Jet>>&,
                               const edm::EDGetTokenT<BXVector<l1t::EtSum>>&,
                               const bool receiveEG,
                               const int nrL1EG,
                               const bool receiveTau,
                               const int nrL1Tau,
                               const bool receiveJet,
                               const int nrL1Jet,
                               const bool receiveEtSums);

    void receiveMuonObjectData(edm::Event&,
                               const edm::EDGetTokenT<BXVector<l1t::Muon>>&,
                               const bool receiveMu,
                               const int nrL1Mu);

    void receiveMuonShowerObjectData(edm::Event&,
                                     const edm::EDGetTokenT<BXVector<l1t::MuonShower>>&,
                                     const bool receiveMuShower,
                                     const int nrL1MuShower);

    void receiveExternalData(edm::Event&, const edm::EDGetTokenT<BXVector<GlobalExtBlk>>&, const bool receiveExt);

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
    void runGTL(edm::Event& iEvent,
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
    void runFDL(edm::Event& iEvent,
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
    inline const BXVector<const l1t::MuonShower*>* getCandL1MuShower() const { return m_candL1MuShower; }

    /// pointer to EG data list
    inline const BXVector<const l1t::L1Candidate*>* getCandL1EG() const { return m_candL1EG; }

    /// pointer to Jet data list
    inline const BXVector<const l1t::L1Candidate*>* getCandL1Jet() const { return m_candL1Jet; }

    /// pointer to Tau data list
    inline const BXVector<const l1t::L1Candidate*>* getCandL1Tau() const { return m_candL1Tau; }

    /// pointer to Tau data list
    inline const BXVector<const l1t::EtSum*>* getCandL1EtSum() const { return m_candL1EtSum; }

    /// pointer to Tau data list
    inline const BXVector<const GlobalExtBlk*>* getCandL1External() const { return m_candL1External; }

    //initializer prescale counter using a semi-random value between [1, prescale value]
    static const std::vector<double> semirandomNumber(const edm::Event& iEvent,
                                                      const std::vector<double>& prescaleFactorsAlgoTrig);

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
    BXVector<const l1t::MuonShower*>* m_candL1MuShower;
    BXVector<const l1t::L1Candidate*>* m_candL1EG;
    BXVector<const l1t::L1Candidate*>* m_candL1Tau;
    BXVector<const l1t::L1Candidate*>* m_candL1Jet;
    BXVector<const l1t::EtSum*>* m_candL1EtSum;
    BXVector<const GlobalExtBlk*>* m_candL1External;

    //    BXVector<const l1t::EtSum*>* m_candETM;
    //    BXVector<const l1t::EtSum*>* m_candETT;
    //    BXVector<const l1t::EtSum*>* m_candHTM;
    //    BXVector<const l1t::EtSum*>* m_candHTT;

    int m_bxFirst_;
    int m_bxLast_;

    std::bitset<GlobalAlgBlk::maxPhysicsTriggers> m_gtlAlgorithmOR;
    std::bitset<GlobalAlgBlk::maxPhysicsTriggers> m_gtlDecisionWord;

    GlobalAlgBlk m_uGtAlgBlk;

    // cache  of maps
    std::vector<AlgorithmEvaluation::ConditionEvaluationMap> m_conditionResultMaps;

    /// prescale counters: NumberPhysTriggers counters per bunch cross in event
    std::vector<std::vector<double>> m_prescaleCounterAlgoTrig;

    bool m_firstEv;
    bool m_firstEvLumiSegment;
    uint m_currentLumi;

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

    //whether we reset the prescales each lumi or not
    bool m_resetPSCountersEachLumiSec = true;

    // start the PS counter from a random value between [1,PS] instead of PS
    bool m_semiRandomInitialPSCounters = false;
  };

}  // namespace l1t
#endif
