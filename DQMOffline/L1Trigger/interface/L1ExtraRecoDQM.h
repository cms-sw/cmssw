#ifndef DQMOffline_L1Trigger_L1ExtraRecoDQM_h
#define DQMOffline_L1Trigger_L1ExtraRecoDQM_h

/**
 * \class L1ExtraRecoDQM
 *
 *
 * Description: online DQM module for L1Extra versus Reco trigger objects.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 *
 * $Date: 2011/12/05 10:20:56 $
 * $Revision: 1.2 $
 *
 */

// system include files
#include <iosfwd>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>

// user include files
//   base classes
#include "FWCore/Framework/interface/EDAnalyzer.h"

//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

// L1Extra objects
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1HFRings.h"
#include "DataFormats/L1Trigger/interface/L1HFRingsFwd.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1PhiConversion.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GetHistLimits.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1RetrieveL1Extra.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "boost/lexical_cast.hpp"

// forward declarations


// class declaration
class L1ExtraRecoDQM: public edm::EDAnalyzer {

public:

    // constructor(s)
    explicit L1ExtraRecoDQM(const edm::ParameterSet&);

    // destructor
    virtual ~L1ExtraRecoDQM();

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

    /// input parameters

    L1RetrieveL1Extra m_retrieveL1Extra;

    /// directory name for L1Extra plots
    std::string m_dirName;

    /// number of bunch crosses in event to be monitored
    int m_nrBxInEventGmt;
    int m_nrBxInEventGct;

    /// internal members

    DQMStore* m_dbe;

    bool m_resetModule;
    int m_currentRun;

    ///
    int m_nrEvJob;
    int m_nrEvRun;

};

#endif
