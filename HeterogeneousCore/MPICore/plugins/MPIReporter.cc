#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Guid.h"
#include "HeterogeneousCore/MPICore/interface/MPIToken.h"

/* MPIReporter class
 *
 */

class MPIReporter : public edm::stream::EDAnalyzer<> {
public:
  explicit MPIReporter(edm::ParameterSet const& config);
  ~MPIReporter() override = default;

  void analyze(edm::Event const& event, edm::EventSetup const& setup) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<MPIToken> token_;
};

MPIReporter::MPIReporter(edm::ParameterSet const& config) : token_(consumes<MPIToken>(edm::InputTag("source"))) {}

void MPIReporter::analyze(edm::Event const& event, edm::EventSetup const& setup) {
  {
    edm::LogAbsolute log("MPI");
    log << "stream " << event.streamID() << ": processing run " << event.run() << ", lumi " << event.luminosityBlock()
        << ", event " << event.id().event();
    log << "\nprocess history:    " << event.processHistory();
    log << "\nprocess history id: " << event.processHistory().id();
    log << "\nprocess history id: " << event.eventAuxiliary().processHistoryID() << " (from eventAuxiliary)";
    log << "\nisRealData " << event.eventAuxiliary().isRealData();
    log << "\nexperimentType " << event.eventAuxiliary().experimentType();
    log << "\nbunchCrossing " << event.eventAuxiliary().bunchCrossing();
    log << "\norbitNumber " << event.eventAuxiliary().orbitNumber();
    log << "\nstoreNumber " << event.eventAuxiliary().storeNumber();
    log << "\nprocessHistoryID " << event.eventAuxiliary().processHistoryID();
    log << "\nprocessGUID " << edm::Guid(event.eventAuxiliary().processGUID(), true).toString();
  }

  auto const& token = event.get(token_);
  {
    edm::LogAbsolute log("MPI");
    log << "got the MPIToken opaque wrapper around the MPIChannel at " << &token;
  }
}

void MPIReporter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MPIReporter);
