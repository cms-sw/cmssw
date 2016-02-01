#ifndef L1TGlobalUtil_h
#define L1TGlobalUtil_h

/**
 * \class L1TGlobalUtil
 *
 *
 * Description: Accessor Class for uGT Result
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 */

// system include files
#include <vector>

#include "L1Trigger/L1TGlobal/interface/TriggerMenu.h"

// Objects to produce for the output record.
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/EventSetup.h"

// forward declarations
//class TriggerMenu;


// class declaration

namespace l1t {

class L1TGlobalUtil
{

public:

    // constructors
  L1TGlobalUtil(std::string preScaleFileName, unsigned int psColumn);

    // destructor
    virtual ~L1TGlobalUtil();

public:

    /// initialize the class (mainly reserve)
    void retrieveL1(const edm::Event& iEvent, const edm::EventSetup& evSetup,
                    edm::EDGetToken gtAlgToken);


public:

    inline void setVerbosity(const int verbosity) {
        m_verbosity = verbosity;
    }

 
    inline bool getFinalOR() {return m_finalOR;} 
    
    // get the trigger bit from the name
    const bool getAlgBitFromName(const std::string& AlgName, int& bit) const;

    // get the name from the trigger bit
    const bool getAlgNameFromBit(int& bit, std::string& AlgName) const;
				 
    // Access results for particular trigger bit 
    const bool getInitialDecisionByBit(int& bit,   bool& decision) const;
    const bool getPrescaledDecisionByBit(int& bit, bool& decision) const;
    const bool getFinalDecisionByBit(int& bit,     bool& decision) const;

    // Access Prescale
    const bool getPrescaleByBit(int& bit,           int& prescale) const;

    // Access Masks:
    // follows logic of uGT board:
    //       finalDecision[AlgBit] = ( prescaledDecision[AlgBit] & mask[AlgBit] ) implying:
    //    If mask = true, algorithm bit (true/false) keeps its value  
    //    If mask = false, algorithm bit is forced to false for the finalDecision
    //
    //    If vetoMask = true and Algorithm is true, the FINOR (final global decision) is forced to false (ie. event is vetoed)
    //    If vetoMask = false, algorithm cannot veto FINOR (final global decision)
    const bool getMaskByBit(int& bit,              bool&     mask) const;
    const bool getVetoMaskByBit(int& bit,          bool&     veto) const;

    // Access results for particular trigger name
    const bool getInitialDecisionByName(const std::string& algName,   bool& decision) const;
    const bool getPrescaledDecisionByName(const std::string& algName, bool& decision) const;
    const bool getFinalDecisionByName(const std::string& algName,     bool& decision) const;

    // Access Prescales
    const bool getPrescaleByName(const std::string& algName,           int& prescale) const;

    // Access Masks (see note) above
    const bool getMaskByName(const std::string& algName,              bool&     mask) const;
    const bool getVetoMaskByName(const std::string& algName,          bool&     veto) const;

    // Some inline commands to return the full vectors
    inline const std::vector<std::pair<std::string, bool> >& decisionsInitial()   { return m_decisionsInitial; }
    inline const std::vector<std::pair<std::string, bool> >& decisionsPrescaled() { return m_decisionsPrescaled; }
    inline const std::vector<std::pair<std::string, bool> >& decisionsFinal()     { return m_decisionsFinal; }

    // Access all prescales
    inline const std::vector<std::pair<std::string, int> >&  prescales()          { return m_prescales; }

    // Access Masks (see note) above
    inline const std::vector<std::pair<std::string, bool> >& masks()              { return m_masks; }
    inline const std::vector<std::pair<std::string, bool> >& vetoMasks()          { return m_vetoMasks; }
    
private:

    /// clear decision vectors on a menu change
    void resetDecisionVectors();
    void resetPrescaleVectors();
    void resetMaskVectors();
    void loadPrescalesAndMasks();

    // trigger menu
    const TriggerMenu* m_l1GtMenu;
    unsigned long long m_l1GtMenuCacheID;

    // prescales and masks
    bool m_filledPrescales;

    // algorithm maps
    const AlgorithmMap* m_algorithmMap;
    
private:

    // Number of physics triggers
    unsigned int m_numberPhysTriggers;
    
    //file  and container for prescale factors
    std::string m_preScaleFileName;
    unsigned int m_PreScaleColumn;
    const std::vector<std::vector<int> >* m_prescaleFactorsAlgoTrig;
    const std::vector<unsigned int>* m_triggerMaskAlgoTrig;
    const std::vector<unsigned int>* m_triggerMaskVetoAlgoTrig;
    
    // access to the results block from uGT 
    edm::Handle<BXVector<GlobalAlgBlk>>  m_uGtAlgBlk;

    // final OR
    bool m_finalOR;

    // Vectors containing the trigger name and information about that trigger
    std::vector<std::pair<std::string, bool> > m_decisionsInitial;
    std::vector<std::pair<std::string, bool> > m_decisionsPrescaled;
    std::vector<std::pair<std::string, bool> > m_decisionsFinal;
    std::vector<std::pair<std::string, int> >  m_prescales;
    std::vector<std::pair<std::string, bool> > m_masks;
    std::vector<std::pair<std::string, bool> > m_vetoMasks;
    
    /// verbosity level
    int m_verbosity;
    

};

}
#endif
