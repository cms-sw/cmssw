#ifndef L1GlobalTrigger_L1GlobalTriggerReadoutRecord_h
#define L1GlobalTrigger_L1GlobalTriggerReadoutRecord_h

/**
 * \class L1GlobalTriggerReadoutRecord 
 * 
 * 
 * Description: readout record for L1 Global Trigger 
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: N. Neumeister        - HEPHY Vienna - ORCA version 
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version 
 * 
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <bitset>
#include <string>
#include <vector>

// user include files
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"

// forward declarations

// class interface

class L1GlobalTriggerReadoutRecord
{

public:

    /// constructors
    L1GlobalTriggerReadoutRecord();

    /// copy constructor
    L1GlobalTriggerReadoutRecord(const L1GlobalTriggerReadoutRecord&);

    /// destructor
    virtual ~L1GlobalTriggerReadoutRecord();
      
    /// assignment operator
    L1GlobalTriggerReadoutRecord& operator=(const L1GlobalTriggerReadoutRecord&);
  
    /// equal operator
    bool operator==(const L1GlobalTriggerReadoutRecord&) const;

    /// unequal operator
    bool operator!=(const L1GlobalTriggerReadoutRecord&) const;

public:

    static const unsigned int NumberPhysTriggers = 128;    
    static const unsigned int NumberPhysTriggersExtended = 192;    
    static const unsigned int NumberTechnicalTriggers = 64;    

    static const unsigned int NumberL1Muons = 4;
        
    static const unsigned int NumberL1Electrons = 4;    
    static const unsigned int NumberL1IsolatedElectrons = 4;    
    
    static const unsigned int NumberL1CentralJets = 4;    
    static const unsigned int NumberL1ForwardJets = 4;    
    static const unsigned int NumberL1TauJets = 4;    
    
    static const unsigned int NumberL1JetCounts = 12;

public:

    /// typedefs          

    /// algorithm bits: 128 bits
    typedef std::vector<bool> DecisionWord;

    /// extended decision word: 192 bits (extend DecisionWord with 64 bits)
    /// need a new FDL chip :-)  
    typedef std::vector<bool> DecisionWordExtended;

    /// technical trigger bits (64 bits)
    typedef std::vector<bool> TechnicalTriggerWord;

    // muons are represented as 32 bits (actually 26 bits)     
    typedef unsigned MuonDataWord;
    static const unsigned int NumberMuonBits = 32;

    // e-gamma, jet objects have 16 bits
    typedef uint16_t CaloDataWord;
    static const unsigned int NumberCaloBits = 16;

    // missing Et has 32 bits
    typedef uint32_t CaloMissingEtWord;
    static const unsigned int NumberMissingEtBits = 32;

    // twelve jet counts, encoded in five bits per count; six jets per 32-bit word 
    // code jet count = 31 indicate overflow condition 
    typedef std::vector<unsigned> CaloJetCountsWord;
    static const unsigned int NumberJetCountsBits = 32;
    static const unsigned int NumberJetCountsWords = 2;
    static const unsigned int NumberCountBits = 5;
    

    typedef std::vector< std::bitset<64> > L1GlobalTriggerDaqWord; 
    typedef std::vector< std::bitset<64> > L1GlobalTriggerEvmWord;
    typedef std::vector< std::bitset<64> > L1GlobalTriggerPsbWord;
      

    /// get Global Trigger decision and the decision word
    inline const bool decision() const { return m_gtGlobalDecision; }
    inline const DecisionWord decisionWord() const { return m_gtDecision; }
  
    /// set global decision
    void setDecision(bool t) { m_gtGlobalDecision = t; }
  
    /// set decision
    void setDecisionWord(const DecisionWord& decision) { m_gtDecision = decision; }
    
// TODO after inserting the configuration
//    ///  set decision for given configuration
//    ///  disable trigger item(s)


    /// return bunch-crossing identifier from header
    int bxId() const { return m_gtBxId; }

    //**************************************************************************
    // get/set physical candidates 
    //     indexCand    index of candidate
    //     bxInEvent    bunch cross in the event (3bx records, 5bx records) 
    //**************************************************************************

    /// get candidate 
    ///     two arguments:   candidate index, bunch cross index
    ///     one argument:    candidate index, for bunch cross with L1Accept

    /// get all non-empty candidates
    ///     one argument:  bunch cross index
    ///     no argument:   for bunch cross with L1Accept

    /// muon 

    const L1MuGMTCand muonCand(unsigned int indexCand, unsigned int bxInEvent) const;
    const L1MuGMTCand muonCand(unsigned int indexCand) const;

    std::vector<L1MuGMTCand> muonCands(unsigned int bxInEvent) const;
    std::vector<L1MuGMTCand> muonCands() const;

    /// electron

    const L1GctEmCand electronCand(unsigned int indexCand, unsigned int bxInEvent) const;
    const L1GctEmCand electronCand(unsigned int indexCand) const;

    std::vector<L1GctEmCand> electronCands(unsigned int bxInEvent) const;
    std::vector<L1GctEmCand> electronCands() const;

    /// isolated electron
    const L1GctEmCand isolatedElectronCand(unsigned int indexCand, unsigned int bxInEvent) const;
    const L1GctEmCand isolatedElectronCand(unsigned int indexCand) const;

    std::vector<L1GctEmCand> isolatedElectronCands(unsigned int bxInEvent) const;
    std::vector<L1GctEmCand> isolatedElectronCands() const;

    /// central jet
    const L1GctJetCand centralJetCand(unsigned int indexCand, unsigned int bxInEvent) const;
    const L1GctJetCand centralJetCand(unsigned int indexCand) const;

    std::vector<L1GctJetCand> centralJetCands(unsigned int bxInEvent) const;
    std::vector<L1GctJetCand> centralJetCands() const;

    /// forward jet
    const L1GctJetCand forwardJetCand(unsigned int indexCand, unsigned int bxInEvent) const;
    const L1GctJetCand forwardJetCand(unsigned int indexCand) const;

    std::vector<L1GctJetCand> forwardJetCands(unsigned int bxInEvent) const;
    std::vector<L1GctJetCand> forwardJetCands() const;

    /// tau jet
    const L1GctJetCand tauJetCand(unsigned int indexCand, unsigned int bxInEvent) const;
    const L1GctJetCand tauJetCand(unsigned int indexCand) const;

    std::vector<L1GctJetCand> tauJetCands(unsigned int bxInEvent) const;
    std::vector<L1GctJetCand> tauJetCands() const;

    /// missing Et
    const L1GctEtMiss missingEt(unsigned int bxInEvent) const;
    const L1GctEtMiss missingEt() const;

    /// total Et
    const L1GctEtTotal totalEt(unsigned int bxInEvent) const;
    const L1GctEtTotal totalEt() const;

    /// total calibrated Et in jets
    const L1GctEtHad totalHt(unsigned int bxInEvent) const;
    const L1GctEtHad totalHt() const;

    /// jet counts
    const L1GctJetCounts jetCounts(unsigned int bxInEvent) const;
    const L1GctJetCounts jetCounts() const;
  
    /// set candidate data words (all non-empty candidates)
    ///     two arguments:   candidate index, bunch cross index
    ///     one argument:    candidate index, for bunch cross with L1Accept
  
    /// muon
    void setMuons(const std::vector<MuonDataWord>&, unsigned int bxInEvent);
    void setMuons(const std::vector<MuonDataWord>&);

    /// electron
    void setElectrons(const std::vector<CaloDataWord>&, unsigned int bxInEvent);
    void setElectrons(const std::vector<CaloDataWord>&);
  
    /// isolated electron
    void setIsolatedElectrons(const std::vector<CaloDataWord>&, unsigned int bxInEvent); 
    void setIsolatedElectrons(const std::vector<CaloDataWord>&); 
  
    /// central jets
    void setCentralJets(const std::vector<CaloDataWord>&, unsigned int bxInEvent); 
    void setCentralJets(const std::vector<CaloDataWord>&); 
  
    /// forward jets
    void setForwardJets(const std::vector<CaloDataWord>&, unsigned int bxInEvent);
    void setForwardJets(const std::vector<CaloDataWord>&);
  
    /// tau jets
    void setTauJets(const std::vector<CaloDataWord>&, unsigned int bxInEvent);
    void setTauJets(const std::vector<CaloDataWord>&);
  
    /// missing Et
    void setMissingEt(const CaloMissingEtWord&, unsigned int bxInEvent);
    void setMissingEt(const CaloMissingEtWord&);
  
    /// total Et
    void setTotalEt(const CaloDataWord&, unsigned int bxInEvent);
    void setTotalEt(const CaloDataWord&);

    /// total calibrated Et
    void setTotalHt(const CaloDataWord&, unsigned int bxInEvent);
    void setTotalHt(const CaloDataWord&);
  
    /// jet count
    void setJetCounts(const CaloJetCountsWord&, unsigned int bxInEvent);
    void setJetCounts(const CaloJetCountsWord&);


    /// print global decision and algorithm decision word
    void print() const;
    
    /// print technical triggers
    void printTechnicalTrigger() const;
    
    /// print all L1 Trigger Objects
    void printL1Objects() const;


    //**************************************************************************
    // get/set hardware-related words
    // 
    // Board description: file GlobalTriggerBoardsMapper.dat // TODO xml file instead?
    //**************************************************************************


    // get/set DAQ and EVM words. Beware setting the words: they are correlated!
    
    /// get / set DAQ readout record
    const L1GlobalTriggerDaqWord daqWord() const;
    void setDaqWord(const std::vector< std::bitset<64> >&);
    
    /// get / set EVM readout record
    const L1GlobalTriggerEvmWord evmWord() const;
    void setEvmWord(const L1GlobalTriggerEvmWord&);

    // TODO: type for boardId; temporary unsigned int 

    /// get / set word for PSB board     
    const L1GlobalTriggerPsbWord psbWord(unsigned int boardId, unsigned int bxInEvent) const;
    const L1GlobalTriggerPsbWord psbWord(unsigned int boardId) const;
    void setPsbWord(const L1GlobalTriggerPsbWord&, unsigned int bxInEvent);
    void setPsbWord(const L1GlobalTriggerPsbWord&);
    
    // other methods
    
    /// clear the record
    void reset(); 
  
    /// output stream operator
    friend std::ostream& operator<<(std::ostream&, const L1GlobalTriggerReadoutRecord&);


private:

    
    int m_gtBxId;               // reference bunch cross number from CMS header 
    unsigned int m_bxInEvent;   // bunch cross from GT event record 
        
    DecisionWord m_gtDecision;
    bool m_gtGlobalDecision;

    TechnicalTriggerWord m_gtTechnicalTrigger;
       
    MuonDataWord m_gtMuon[NumberL1Muons];
    
    CaloDataWord m_gtElectron[NumberL1Electrons];  
    CaloDataWord m_gtIsoElectron[NumberL1IsolatedElectrons];
  
    CaloDataWord m_gtCJet[NumberL1CentralJets];  
    CaloDataWord m_gtFJet[NumberL1ForwardJets];  
    CaloDataWord m_gtTJet[NumberL1TauJets];
  
    CaloMissingEtWord m_gtMissingEt;
    
    CaloDataWord m_gtTotalEt;    
    CaloDataWord m_gtTotalHt;
  
    CaloJetCountsWord m_gtJetNr;
      
};


#endif
