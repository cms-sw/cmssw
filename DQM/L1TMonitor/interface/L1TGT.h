#ifndef L1TGT_H
#define L1TGT_H

/**
 * \class L1TGT
 *
 *
 * Description: DQM for L1 Global Trigger.
 *
 * \author J. Berryhill, I. Mikulec
 * \author Vasile Mihai Ghete - HEPHY Vienna
 *
 * $Date: 2012/05/31 16:52:42 $
 * $Revision: 1.12 $
 *
 */

// system include files
#include <memory>
#include <unistd.h>
#include <vector>
#include <utility>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// class declaration
//

class L1TGT: public edm::EDAnalyzer {

public:

    // constructor
    L1TGT(const edm::ParameterSet& ps);

    // destructor
    virtual ~L1TGT();

private:

    virtual void beginJob();
    virtual void beginRun(const edm::Run&, const edm::EventSetup&);
    virtual void beginLuminosityBlock(const edm::LuminosityBlock&,
            const edm::EventSetup&);

    virtual void analyze(const edm::Event&, const edm::EventSetup&);

    /// end section
    virtual void endLuminosityBlock(const edm::LuminosityBlock&,
            const edm::EventSetup&);
    virtual void endRun(const edm::Run&, const edm::EventSetup&);

    virtual void endJob();

private:

    /// book all histograms for the module
    void bookHistograms();

    bool isActive(int word, int bit);
    // Active boards DAQ record bit number:
    // 0 FDL
    // 1 PSB_0 9 Techn.Triggers for FDL
    // 2 PSB_1 13 Calo data for GTL
    // 3 PSB_2 14 Calo data for GTL
    // 4 PSB_3 15 Calo data for GTL
    // 5 PSB_4 19 M/Q bits for GMT
    // 6 PSB_5 20 M/Q bits for GMT
    // 7 PSB_6 21 M/Q bits for GMT
    // 8 GMT
    enum activeDAQ {
        FDL = 0, PSB9, PSB13, PSB14, PSB15, PSB19, PSB20, PSB21, GMT
    };
    // Active boards EVM record bit number:
    // 0 TCS
    // 1 FDL
    enum activeEVM {
        TCS, FDLEVM
    };

    // count the number of indices per Ls for prescale factor sets
    // if no errors, it must be 1
    void countPfsIndicesPerLs();


private:

    /// input parameters

    /// input tag for L1 GT DAQ readout record
    edm::InputTag gtSource_;

    /// input tag for L1 GT EVM readout record
    edm::InputTag gtEvmSource_;

    /// switches to choose the running of various methods
    bool m_runInEventLoop;
    bool m_runInEndLumi;
    bool m_runInEndRun;
    bool m_runInEndJob;


    /// verbosity switch
    bool verbose_;

private:

    MonitorElement* algo_bits;
    MonitorElement* algo_bits_corr;
    MonitorElement* tt_bits;
    MonitorElement* tt_bits_corr;
    MonitorElement* algo_tt_bits_corr;
    MonitorElement* algo_bits_lumi;
    MonitorElement* tt_bits_lumi;
    MonitorElement* event_type;

    MonitorElement* event_number;
    MonitorElement* event_lumi;
    MonitorElement* trigger_number;
    MonitorElement* trigger_lumi;
    MonitorElement* evnum_trignum_lumi;
    MonitorElement* orbit_lumi;
    MonitorElement* setupversion_lumi;

    MonitorElement* gtfe_bx;
    MonitorElement* dbx_module;

    MonitorElement* BST_MasterStatus;
    MonitorElement* BST_turnCountNumber;
    MonitorElement* BST_lhcFillNumber;
    MonitorElement* BST_beamMode;
    MonitorElement* BST_beamMomentum;
    MonitorElement* BST_intensityBeam1;
    MonitorElement* BST_intensityBeam2;
    MonitorElement* gpsfreq;
    MonitorElement* gpsfreqwide;
    MonitorElement* gpsfreqlum;

    MonitorElement* m_monL1PrescaleFactorSet;
    MonitorElement* m_monL1PfIndicesPerLs;

    MonitorElement* m_monOrbitNrDiffTcsFdlEvm;
    MonitorElement* m_monLsNrDiffTcsFdlEvm;
    // maximum difference in orbit number, luminosity number
    // histogram range: -(MaxOrbitNrDiffTcsFdlEvm+1), (MaxOrbitNrDiffTcsFdlEvm+1)
    //   if value is greater than the maximum difference, fill an entry in the last but one bin
    //   if value is smaller than the negative value of maximum difference, fill an entry
    //     in the second bin
    //   if no value can be retrieved for TCS, fill an entry in the first bin
    //   if no value can be retrieved for FDL, fill an entry in the last bin
    static const int MaxOrbitNrDiffTcsFdlEvm;
    static const int MaxLsNrDiffTcsFdlEvm;

    MonitorElement* m_monOrbitNrDiffTcsFdlEvmLs;
    MonitorElement* m_monLsNrDiffTcsFdlEvmLs;

    //MonitorElement* m_monDiffEvmDaqFdl;

private:

    /// internal members

    DQMStore* m_dbe;

    /// number of events processed
    int m_nrEvJob;
    int m_nrEvRun;

    /// histogram folder for L1 GT plots
    std::string m_histFolder;

    boost::uint64_t preGps_;
    boost::uint64_t preOrb_;


    std::vector<std::pair<int,int> > m_pairLsNumberPfIndex;
    typedef std::vector<std::pair<int, int> >::const_iterator CItVecPair;


};

#endif
