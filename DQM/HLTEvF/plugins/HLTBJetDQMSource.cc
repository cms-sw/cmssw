#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "HLTBJetDQMSource.h"

/** \class HLTBJetDQMSource
 *
 *  DQM source for BJet HLT paths
 *
 *  $Date: 2008/05/13 13:50:06 $
 *  $Revision: 1.1 $
 *  \author Andrea Bocci, Pisa
 *
 */

HLTBJetDQMSource::HLTBJetDQMSource(const edm::ParameterSet & config) { }

HLTBJetDQMSource::~HLTBJetDQMSource() { }

void HLTBJetDQMSource::beginJob(const edm::EventSetup & setup) { }
void HLTBJetDQMSource::endJob() { }

void HLTBJetDQMSource::beginRun(const edm::Run & run, const edm::EventSetup & setup) { }
void HLTBJetDQMSource::endRun(const edm::Run & run, const edm::EventSetup & setup) { }

void HLTBJetDQMSource::beginLuminosityBlock(const edm::LuminosityBlock & lumi, const edm::EventSetup & setup) { }
void HLTBJetDQMSource::endLuminosityBlock(const edm::LuminosityBlock & lumi, const edm::EventSetup & setup) { }

void HLTBJetDQMSource::analyze(const edm::Event & event, const edm::EventSetup & setup)  { }

