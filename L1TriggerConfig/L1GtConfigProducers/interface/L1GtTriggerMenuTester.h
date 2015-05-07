#ifndef L1GtConfigProducers_L1GtTriggerMenuTester_h
#define L1GtConfigProducers_L1GtTriggerMenuTester_h

/**
 * \class L1GtTriggerMenuTester
 * 
 * 
 * Description: test analyzer for L1 GT trigger menu.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtTriggerMenuTester.h"

// system include files
#include <string>
#include <map>

// user include files
//   base class
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/L1TObjects/interface/L1GtAlgorithm.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

// forward declarations
class L1GtStableParameters;
class L1GtPrescaleFactors;
class L1GtTriggerMask;
class L1GtTriggerMenu;

// class declaration
class L1GtTriggerMenuTester: public edm::EDAnalyzer {

public:

    // constructor
    explicit L1GtTriggerMenuTester(const edm::ParameterSet&);

    // destructor
    virtual ~L1GtTriggerMenuTester();

private:

    /// begin job
    void beginJob();

    /// begin run
    void beginRun(const edm::Run&, const edm::EventSetup&);

    /// begin luminosity block
    void beginLuminosityBlock(const edm::LuminosityBlock&,
            const edm::EventSetup&);

    /// analyze
    void analyze(const edm::Event&, const edm::EventSetup&);

    /// end luminosity block
    void
    endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);

    /// end run
    void endRun(const edm::Run&, const edm::EventSetup&);

    /// end job
    void endJob();

private:

    /// retrieve all the relevant L1 trigger event setup records
    void retrieveL1EventSetup(const edm::EventSetup&);

    /// L1 seed - HLT path association
    void associateL1SeedsHltPath(const edm::Run&, const edm::EventSetup&);

    /// printing template for a trigger group
    void printTriggerGroup(const std::string& trigGroupName,
            const std::map<std::string, const L1GtAlgorithm*>& trigGroup,
            const bool compactPrint, const bool printPfsRates);

    /// printing in Wiki format
    void printWiki();

private:

    /// constant iterator
    typedef std::map<std::string, const L1GtAlgorithm*>::const_iterator
            CItAlgoP;

private:

    /// input parameters

    /// overwrite name of the HTML file containing the detailed L1 menu
    /// with the name given in m_htmlFile
    bool m_overwriteHtmlFile;

    /// name of HTML file attached to the wiki page
    std::string m_htmlFile;

    /// use a HLT menu for L1 seed - HLT path association
    bool m_useHltMenu;

    /// process name of HLT process for which to get HLT configuration
    std::string m_hltProcessName;

    /// do not throw an exceptions if a L1 trigger requested as seed is not available in the L1 menu,
    /// just report this
    bool m_noThrowIncompatibleMenu;

    /// print prescale factors and rates
    bool m_printPfsRates;

    /// index of prescale factor set to be printed
    int m_indexPfSet;

private:

    /// event setup cached stuff

    /// stable parameters
    const L1GtStableParameters* m_l1GtStablePar;

    /// number of algorithm triggers
    unsigned int m_numberAlgorithmTriggers;

    /// number of technical triggers
    unsigned int m_numberTechnicalTriggers;

    /// prescale factors
    const L1GtPrescaleFactors* m_l1GtPfAlgo;

    const L1GtPrescaleFactors* m_l1GtPfTech;

    const std::vector<std::vector<int> >* m_prescaleFactorsAlgoTrig;
    const std::vector<std::vector<int> >* m_prescaleFactorsTechTrig;

    /// trigger masks & veto masks
    const L1GtTriggerMask* m_l1GtTmAlgo;
    const L1GtTriggerMask* m_l1GtTmTech;

    const L1GtTriggerMask* m_l1GtTmVetoAlgo;
    const L1GtTriggerMask* m_l1GtTmVetoTech;

    const std::vector<unsigned int>* m_triggerMaskAlgoTrig;
    const std::vector<unsigned int>* m_triggerMaskTechTrig;

    const std::vector<unsigned int>* m_triggerMaskVetoAlgoTrig;
    const std::vector<unsigned int>* m_triggerMaskVetoTechTrig;

    // trigger menu
    const L1GtTriggerMenu* m_l1GtMenu;

    const AlgorithmMap* m_algorithmMap;
    const AlgorithmMap* m_algorithmAliasMap;
    const AlgorithmMap* m_technicalTriggerMap;

private:

    /// The instance of the HLTConfigProvider as a data member
    HLTConfigProvider m_hltConfig;

    /// HLT menu was used to associate the HLT path to the L1 algorithm triggers
    std::string m_hltTableName;

    /// vector of HLT paths seeded by a L1 algorithm trigger - vector index corresponds to the bit number
    std::vector<std::vector<std::string> > m_hltPathsForL1AlgorithmTrigger;

    /// vector of HLT paths seeded by a L1 technical trigger - vector index corresponds to the bit number
    std::vector<std::vector<std::string> > m_hltPathsForL1TechnicalTrigger;

    /// vector of algorithm or technical triggers not in the L1 menu
    std::vector<std::string> m_algoTriggerSeedNotInL1Menu;
    std::vector<std::string> m_techTriggerSeedNotInL1Menu;

};

#endif /*L1GtConfigProducers_L1GtTriggerMenuTester_h*/
