#ifndef DQM_L1TMonitor_L1GtHwValidation_h
#define DQM_L1TMonitor_L1GtHwValidation_h

/**
 * \class L1GtHwValidation
 * 
 * 
 * Description: compare hardware records with emulator records for L1 GT records.
 *
 * Implementation:
 *    Get the L1 GT records from data and from emulator.   
 *    Compare every board between data and emulator.
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// forward declarations
class L1GtfeWord;
class L1GtFdlWord;
class L1GtPsbWord;
class L1TcsWord;
class L1GtTriggerMenu;
class L1GtPrescaleFactors;
class L1GtTriggerMask;

// class declaration

class L1GtHwValidation: public edm::EDAnalyzer
{

public:
    explicit L1GtHwValidation(const edm::ParameterSet&);
    virtual ~L1GtHwValidation();

private:

    /// compare the GTFE board
    virtual void compareGTFE(const edm::Event&, const edm::EventSetup&,
            const L1GtfeWord&, const L1GtfeWord&, const int);

    /// compare the FDL board
    virtual void compareFDL(const edm::Event&, const edm::EventSetup&,
            const L1GtFdlWord&, const L1GtFdlWord&, const int);

    /// compare the PSB board
    virtual void comparePSB(const edm::Event&, const edm::EventSetup&,
            const L1GtPsbWord&, const L1GtPsbWord&);

    /// compare the TCS board
    virtual void compareTCS(const edm::Event&, const edm::EventSetup&,
            const L1TcsWord&, const L1TcsWord&);

    /// L1 GT DAQ record comparison
    virtual void compareDaqRecord(const edm::Event&, const edm::EventSetup&);

    /// L1 GT EVM record comparison
    virtual void compareEvmRecord(const edm::Event&, const edm::EventSetup&);

    /// compare the GCT collections obtained from L1 GT PSB with the input 
    /// GCT collections
    virtual void compareGt_Gct(const edm::Event&, const edm::EventSetup&);

    /// book all histograms for the module
    void bookHistograms();

    virtual void beginJob();
    void beginRun(const edm::Run& run, const edm::EventSetup& c);

    virtual void analyze(const edm::Event&, const edm::EventSetup&);

    void endRun(const edm::Run& run, const edm::EventSetup& c);
    virtual void endJob();

private:

    /// input tag for the L1 GT hardware DAQ record
    edm::InputTag m_l1GtDataDaqInputTag;

    /// input tag for the L1 GT hardware EVM record
    edm::InputTag m_l1GtDataEvmInputTag;

    /// input tag for the L1 GT emulator DAQ record
    edm::InputTag m_l1GtEmulDaqInputTag;

    /// input tag for the L1 GT emulator EVM record
    edm::InputTag m_l1GtEmulEvmInputTag;

    /// input tag for the L1 GCT hardware record 
    edm::InputTag m_l1GctDataInputTag;

private:

    /// directory name for L1Extra plots
    std::string m_dirName;

    /// an output stream to print into
    /// it can then be directed to whatever log level is desired
    std::ostringstream m_myCoutStream;

    /// counters
    int m_nrDataEventError;
    int m_nrEmulEventError;

    // cached stuff

    /// trigger menu
    const L1GtTriggerMenu* m_l1GtMenu;
    unsigned long long m_l1GtMenuCacheID;

    /// prescale factors
    const L1GtPrescaleFactors* m_l1GtPfAlgo;
    unsigned long long m_l1GtPfAlgoCacheID;

    const L1GtPrescaleFactors* m_l1GtPfTech;
    unsigned long long m_l1GtPfTechCacheID;

    const std::vector<std::vector<int> >* m_prescaleFactorsAlgoTrig;
    const std::vector<std::vector<int> >* m_prescaleFactorsTechTrig;

    /// trigger masks
    const L1GtTriggerMask* m_l1GtTmAlgo;
    unsigned long long m_l1GtTmAlgoCacheID;

    const L1GtTriggerMask* m_l1GtTmTech;
    unsigned long long m_l1GtTmTechCacheID;

    std::vector<unsigned int> m_triggerMaskAlgoTrig;
    std::vector<unsigned int> m_triggerMaskTechTrig;

private:
    /// internal members

    DQMStore* m_dbe;

private:

    static const int TotalBxInEvent = 5;
    static const int NumberOfGtRecords = 2; // DAQ and EVM

    /// histograms

    /// GTFE    
    MonitorElement* m_gtfeDataEmul[NumberOfGtRecords];

    /// FDL (0 for DAQ, 1 for EVM record)
    MonitorElement* m_fdlDataEmul[TotalBxInEvent][NumberOfGtRecords];
    //
    MonitorElement* m_fdlDataAlgoDecision[TotalBxInEvent][NumberOfGtRecords];
    MonitorElement* m_fdlDataAlgoDecisionPrescaled[TotalBxInEvent][NumberOfGtRecords];
    MonitorElement* m_fdlDataAlgoDecisionUnprescaled[TotalBxInEvent][NumberOfGtRecords];
    MonitorElement* m_fdlDataAlgoDecisionMask[TotalBxInEvent][NumberOfGtRecords];
    MonitorElement* m_fdlDataAlgoDecision_NoMatch[TotalBxInEvent][NumberOfGtRecords];
    MonitorElement* m_fdlDataAlgoDecisionPrescaled_NoMatch[TotalBxInEvent][NumberOfGtRecords];
    MonitorElement* m_fdlDataAlgoDecisionUnprescaled_NoMatch[TotalBxInEvent][NumberOfGtRecords];
    MonitorElement* m_fdlDataAlgoDecision_Err[NumberOfGtRecords];

    MonitorElement* m_fdlEmulAlgoDecision[TotalBxInEvent][NumberOfGtRecords];
    MonitorElement* m_fdlEmulAlgoDecisionPrescaled[TotalBxInEvent][NumberOfGtRecords];
    MonitorElement* m_fdlEmulAlgoDecisionUnprescaled[TotalBxInEvent][NumberOfGtRecords];
    MonitorElement* m_fdlEmulAlgoDecisionMask[TotalBxInEvent][NumberOfGtRecords];
    MonitorElement* m_fdlEmulAlgoDecision_NoMatch[TotalBxInEvent][NumberOfGtRecords];
    MonitorElement* m_fdlEmulAlgoDecisionPrescaled_NoMatch[TotalBxInEvent][NumberOfGtRecords];
    MonitorElement* m_fdlEmulAlgoDecisionUnprescaled_NoMatch[TotalBxInEvent][NumberOfGtRecords];
    MonitorElement* m_fdlEmulAlgoDecision_Err[NumberOfGtRecords];

    //
    MonitorElement* m_fdlDataEmulAlgoDecision[TotalBxInEvent][NumberOfGtRecords];
    MonitorElement* m_fdlDataEmulAlgoDecisionPrescaled[TotalBxInEvent][NumberOfGtRecords];
    MonitorElement* m_fdlDataEmulAlgoDecisionUnprescaled[TotalBxInEvent][NumberOfGtRecords];
    MonitorElement* m_fdlDataEmulAlgoDecisionMask[TotalBxInEvent][NumberOfGtRecords];
    MonitorElement* m_fdlDataEmulAlgoDecision_Err[NumberOfGtRecords];
    MonitorElement* m_fdlDataEmul_Err[NumberOfGtRecords];

    //
    MonitorElement* m_fdlDataTechDecision[TotalBxInEvent][NumberOfGtRecords];
    MonitorElement* m_fdlDataTechDecisionMask[TotalBxInEvent][NumberOfGtRecords];
    MonitorElement* m_fdlDataTechDecision_Err[NumberOfGtRecords];

    MonitorElement* m_fdlEmulTechDecision[TotalBxInEvent][NumberOfGtRecords];
    MonitorElement* m_fdlEmulTechDecisionMask[TotalBxInEvent][NumberOfGtRecords];
    MonitorElement* m_fdlEmulTechDecision_Err[NumberOfGtRecords];

    MonitorElement* m_fdlDataEmulTechDecision[TotalBxInEvent][NumberOfGtRecords];
    MonitorElement* m_fdlDataEmulTechDecisionMask[TotalBxInEvent][NumberOfGtRecords];
    MonitorElement* m_fdlDataEmulTechDecision_Err[NumberOfGtRecords];

    /// PSB

    ///
    int m_nrEvJob;
    int m_nrEvRun;

};

#endif /*DQM_L1TMonitor_L1GtHwValidation_h*/
