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

// this class header
#include "DQMOffline/L1Trigger/interface/L1ExtraRecoDQM.h"

// system include files
#include <iostream>
#include <iomanip>
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/MakerMacros.h"

// constructor
L1ExtraRecoDQM::L1ExtraRecoDQM(const edm::ParameterSet& paramSet) :
            //
            m_retrieveL1Extra(
                    paramSet.getParameter<edm::ParameterSet> (
                            "L1ExtraInputTags")),
            m_dirName(
                    paramSet.getUntrackedParameter("DirName",
                            std::string("L1T/L1ExtraRecoDQM"))),
            //
            m_nrBxInEventGmt(paramSet.getParameter<int> ("NrBxInEventGmt")),
            m_nrBxInEventGct(paramSet.getParameter<int> ("NrBxInEventGct")),
            //
            m_dbe(0), m_resetModule(true), m_currentRun(-99),
            //
            m_nrEvJob(0), m_nrEvRun(0)

{

    //
    if ((m_nrBxInEventGmt > 0) && ((m_nrBxInEventGmt % 2) == 0)) {
        m_nrBxInEventGmt = m_nrBxInEventGmt - 1;

        edm::LogInfo("L1ExtraRecoDQM")
                << "\nWARNING: Number of bunch crossing to be monitored for GMT rounded to: "
                << m_nrBxInEventGmt
                << "\n         The number must be an odd number!\n"
                << std::endl;
    }

    if ((m_nrBxInEventGct > 0) && ((m_nrBxInEventGct % 2) == 0)) {
        m_nrBxInEventGct = m_nrBxInEventGct - 1;

        edm::LogInfo("L1ExtraRecoDQM")
                << "\nWARNING: Number of bunch crossing to be monitored for GCT rounded to: "
                << m_nrBxInEventGct
                << "\n         The number must be an odd number!\n"
                << std::endl;
    }

    m_dbe = edm::Service<DQMStore>().operator->();
    if (m_dbe == 0) {
        edm::LogInfo("L1ExtraRecoDQM") << "\n Unable to get DQMStore service.";
    } else {

        if (paramSet.getUntrackedParameter<bool> ("DQMStore", false)) {
            m_dbe->setVerbose(0);
        }

        m_dbe->setCurrentFolder(m_dirName);

    }

}

// destructor
L1ExtraRecoDQM::~L1ExtraRecoDQM() {

    // empty

}

//
void L1ExtraRecoDQM::beginJob() {

}

void L1ExtraRecoDQM::beginRun(const edm::Run& iRun,
        const edm::EventSetup& evSetup) {

    m_nrEvRun = 0;

    DQMStore* dbe = 0;
    dbe = edm::Service<DQMStore>().operator->();

    // clean up directory
    if (dbe) {
        dbe->setCurrentFolder(m_dirName);
        if (dbe->dirExists(m_dirName)) {
            dbe->rmdir(m_dirName);
        }
        dbe->setCurrentFolder(m_dirName);
    }

}

void L1ExtraRecoDQM::beginLuminosityBlock(const edm::LuminosityBlock& iLumi,
        const edm::EventSetup& evSetup) {

    //


}


//
void L1ExtraRecoDQM::analyze(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    ++m_nrEvJob;
    ++m_nrEvRun;

    //
    m_retrieveL1Extra.retrieveL1ExtraObjects(iEvent, evSetup);

}

// end section
void L1ExtraRecoDQM::endLuminosityBlock(const edm::LuminosityBlock& iLumi,
        const edm::EventSetup& evSetup) {

    // empty

}

void L1ExtraRecoDQM::endRun(const edm::Run& run, const edm::EventSetup& evSetup) {

    //

}

void L1ExtraRecoDQM::endJob() {

    edm::LogInfo("L1ExtraRecoDQM")
            << "\n\nTotal number of events analyzed in this job: " << m_nrEvJob
            << "\n" << std::endl;

    return;
}

//define this as a plug-in
DEFINE_FWK_MODULE( L1ExtraRecoDQM);
