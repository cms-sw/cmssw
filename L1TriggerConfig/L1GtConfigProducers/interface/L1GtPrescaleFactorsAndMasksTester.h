#ifndef L1GtConfigProducers_L1GtPrescaleFactorsAndMasksTester_h
#define L1GtConfigProducers_L1GtPrescaleFactorsAndMasksTester_h

/**
 * \class L1GtPrescaleFactorsAndMasksTester
 * 
 * 
 * Description: test analyzer for L1 GT prescale factors and masks.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date$
 * $Revision$
 *
 */

// system include files

// user include files
//   base class
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// forward declarations
class L1GtPrescaleFactors;
class L1GtTriggerMask;


// class declaration
class L1GtPrescaleFactorsAndMasksTester: public edm::EDAnalyzer {

public:

    // constructor
    explicit L1GtPrescaleFactorsAndMasksTester(const edm::ParameterSet&);

    // destructor
    virtual ~L1GtPrescaleFactorsAndMasksTester();

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

    /// print the requred records
    void printL1EventSetup(const edm::EventSetup&);

private:

    /// input parameters

    /// analyze prescale factors, trigger masks and trigger veto masks, respectively
    bool m_testerPrescaleFactors;
    bool m_testerTriggerMask;
    bool m_testerTriggerVetoMask;

    /// retrieve the records in beginRun, beginLuminosityBlock, analyze, respectively
    bool m_retrieveInBeginRun;
    bool m_retrieveInBeginLuminosityBlock;
    bool m_retrieveInAnalyze;

    /// print the records in beginRun, beginLuminosityBlock, analyze, respectively
    bool m_printInBeginRun;
    bool m_printInBeginLuminosityBlock;
    bool m_printInAnalyze;

    /// print output
     int m_printOutput;



private:

    /// prescale factors
    const L1GtPrescaleFactors* m_l1GtPfAlgo;
    const L1GtPrescaleFactors* m_l1GtPfTech;

    /// trigger masks & veto masks
    const L1GtTriggerMask* m_l1GtTmAlgo;
    const L1GtTriggerMask* m_l1GtTmTech;

    const L1GtTriggerMask* m_l1GtTmVetoAlgo;
    const L1GtTriggerMask* m_l1GtTmVetoTech;

};

#endif /*L1GtConfigProducers_L1GtPrescaleFactorsAndMasksTester_h*/
