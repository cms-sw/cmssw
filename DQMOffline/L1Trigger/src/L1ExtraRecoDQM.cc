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
							      "L1ExtraInputTags"),consumesCollector()),
            m_dirName(
                    paramSet.getUntrackedParameter("DirName",
                            std::string("L1T/L1ExtraRecoDQM"))),
            //
            m_nrBxInEventGmt(paramSet.getParameter<int> ("NrBxInEventGmt")),
            m_nrBxInEventGct(paramSet.getParameter<int> ("NrBxInEventGct")),
            //
	    m_resetModule(true), m_currentRun(-99),
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

}

// destructor
L1ExtraRecoDQM::~L1ExtraRecoDQM() {

    // empty

}


void L1ExtraRecoDQM::bookHistograms(DQMStore::IBooker &ibooker, const edm::Run& iRun, const edm::EventSetup& evSetup) {

    m_nrEvRun = 0;
    ibooker.setCurrentFolder(m_dirName);

}

void L1ExtraRecoDQM::beginLuminosityBlock(const edm::LuminosityBlock& iLumi,
        const edm::EventSetup& evSetup) {

    //
}

void L1ExtraRecoDQM::dqmBeginRun(const edm::Run&, const edm::EventSetup&){

  
}

//
void L1ExtraRecoDQM::analyze(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    ++m_nrEvJob;
    ++m_nrEvRun;

    //
    m_retrieveL1Extra.retrieveL1ExtraObjects(iEvent, evSetup);

}

DEFINE_FWK_MODULE( L1ExtraRecoDQM);
