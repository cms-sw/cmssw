#ifndef L1GlobalTrigger_L1GlobalTriggerReadoutSetup_h
#define L1GlobalTrigger_L1GlobalTriggerReadoutSetup_h

/**
 * \class L1GlobalTriggerReadoutSetup
 * 
 * 
 * 
 * Description: group static constants and typedefs for GT readout record 
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date:$
 * $Revision:$
 *
 */

// this class header
// system include files
#include <string>
#include <vector>

#include <boost/cstdint.hpp>

// user include files
//   base class
// forward declarations

// class interface

class L1GlobalTriggerReadoutSetup
{

public:
	L1GlobalTriggerReadoutSetup();
	virtual ~L1GlobalTriggerReadoutSetup();

public:

    static const unsigned int NumberPhysTriggers = 128;    
    static const unsigned int NumberPhysTriggersExtended = 64; // in addition to 128    
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

    /// extend DecisionWord with 64 bits
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

    // TODO FIXME 
    // bunch cross in event: order and number as written by hardware     
//    static const std::vector<int> BxInEventNumber = {0xE, 0xF, 0x0, 0x1, 0x2};
        
};

#endif /*L1GlobalTrigger_L1GlobalTriggerReadoutSetup_h*/
