#ifndef L1Trigger_L1GlobalTriggerReadoutRecord_h
#define L1Trigger_L1GlobalTriggerReadoutRecord_h

/**
 * \class L1GlobalTriggerReadoutRecord 
 * 
 * 
 * Description: readout record for L1 Global Trigger 
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date:$
 * $Revision:$
 *
 */

// system include files
#include <bitset>
#include <string>

// user include files

//// forward declarations

class L1MuGMTCand;

class L1GctCand;
class L1GctEmCand;

class L1GctCenJet;
class L1GctForJet;
class L1GctTauJet;

class L1GctEtTotal;
class L1GctEtHad;
class L1GctEtMiss;

// TODO uncomment when class available in GCT
//class L1GctJetCount;


class L1GlobalTriggerReadoutRecord
{

public:

    // constructors
    L1GlobalTriggerReadoutRecord();

    // copy constructor
    L1GlobalTriggerReadoutRecord(const L1GlobalTriggerReadoutRecord&);

    // destructor
    virtual ~L1GlobalTriggerReadoutRecord();
      
    /// assignment operator
    L1GlobalTriggerReadoutRecord& operator=(const L1GlobalTriggerReadoutRecord&);
  
    /// equal operator
    bool operator==(const L1GlobalTriggerReadoutRecord&) const;

    /// unequal operator
    bool operator!=(const L1GlobalTriggerReadoutRecord& res) const;

public:

    // typedefs      
    static const unsigned int NumberPhysTriggers = 128;    
    typedef std::bitset<NumberPhysTriggers> DecisionWord;

    // TODO what about 192 bits: extended decision word?
    //     
    typedef unsigned MuonDataWord;
    typedef std::bitset<32>  CaloDataWord;

    typedef std::vector<std::bitset<64>> L1GlobalTriggerDaqWord; // TODO keep this format?
    typedef std::vector<std::bitset<64>> L1GlobalTriggerEvmWord; // TODO 
      

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
    ///     one arguments:   candidate index, for bunch cross with L1Accept

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
    const L1GctCenJet centralJetCand(unsigned int indexCand, unsigned int bxInEvent) const;
    const L1GctCenJet centralJetCand(unsigned int indexCand) const;

    std::vector<L1GctCenJet> centralJetCands(unsigned int bxInEvent) const;
    std::vector<L1GctCenJet> centralJetCands() const;

    /// forward jet
    const L1GctForJet forwardJetCand(unsigned int indexCand, unsigned int bxInEvent) const;
    const L1GctForJet forwardJetCand(unsigned int indexCand) const;

    std::vector<L1GctForJet> forwardJetCands(unsigned int bxInEvent) const;
    std::vector<L1GctForJet> forwardJetCands() const;

    /// tau jet
    const L1GctJet tauJetCand(unsigned int indexCand, unsigned int bxInEvent) const;
    const L1GctJet tauJetCand(unsigned int indexCand) const;

    std::vector<L1GctJet> tauJetCands(unsigned int bxInEvent) const;
    std::vector<L1GctJet> tauJetCands() const;

    /// missing Et
    const L1GctEtMiss missingEt(unsigned int bxInEvent) const;
    const L1GctEtMiss missingEt() const;

    /// total Et
    const L1GctEtTotal totalEt(unsigned int bxInEvent) const;
    const L1GctEtTotal totalEt() const;

    /// total calibrated Et in jets
    const L1GctEtHad totalHt(unsigned int bxInEvent) const;
    const L1GctEtHad totalHt() const;

//    /// jet counts
//    const L1GctJetCount jetCount(unsigned int bxInEvent) const;
//    const L1GctJetCount jetCount() const;
  
// TODO  review the format

    //
    // set candidate data words (all non-empty candidates)
    //
  
    /// muon
    void setMuons(const std::vector<MuonDataWord>&, unsigned int bxInEvent);

    /// electron
    void setElectrons(const std::vector<CaloDataWord>, unsigned int bxInEvent);
  
    /// isolated electron
    void setIsolatedElectrons(const std::vector<CaloDataWord>&, unsigned int bxInEvent); 
  
    /// central jets
    void setCentralJets(const std::vector<CaloDataWord>&, unsigned int bxInEvent); 
  
    /// forward jets
    void setForwardJets(const std::vector<CaloDataWord>&, unsigned int bxInEvent);
  
    /// tau jets
    void setTauJets(const std::vector<CaloDataWord>&, unsigned int bxInEvent);
  
    /// missing Et
    void setMissingEt(const CaloDataWord&, unsigned int bxInEvent);
  
    /// total Et
    void setTotalEt(const CaloDataWord&, unsigned int bxInEvent);

    /// total calibrated Et
    void setTotalHt(const CaloDataWord&, unsigned int bxInEvent);
  
// TODO un-comment when JetNr available
//    /// jet count
//    void setJetNr(const CaloDataWord&, unsigned int bxInEvent);


    /// print result
    void print() const;

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
    void setDaqWord(const L1GlobalTriggerDaqWord&);
    
    /// get / set EVM readout record
    const L1GlobalTriggerEvmWord evmWord() const;
    void setEvmWord(const L1GlobalTriggerEvmWord&);

    // TODO: do I introduce PsbWord, FdlWord or I use a templated word... 
    // TODO: type for boardId; temporary unsigned int 

//    /// get word for board     
//    const L1GlobalTriggerBoardWord boardWord(unsigned int boardId, unsigned int bxInEvent) const;
//
//    /// set word for board:
//    void setBoardWord(const L1GlobalTriggerBoardWord&, unsigned int bxInEvent);
    
    // other methods
    
    /// clear the record
    void reset(); 
  
    /// output stream operator
    friend std::ostream& operator<<(ostream&, const L1GlobalTriggerReadoutRecord&);


private:

    int m_gtBxId;
    
    DecisionWord m_gtDecision;

    bool m_gtGlobalDecision;
  
    MuonDataWord m_gtMuon[4];
  
    CaloDataWord m_gtElectron[4];  
    CaloDataWord m_gtIsoElectron[4];
  
    CaloDataWord m_gtCJet[4];  
    CaloDataWord m_gtFJet[4];  
    CaloDataWord m_gtTJet[4];
  
    CaloDataWord m_gtMissingEt;
    CaloDataWord m_gtTotalEt;    
    CaloDataWord m_gtTotalHt;
  
    CaloDataWord m_gtJetNr;
  

};


#endif
