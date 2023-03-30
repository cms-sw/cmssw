#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Guid.h"
#include "HeterogeneousCore/MPICore/interface/MPIOrigin.h"

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
  edm::EDGetTokenT<MPIOrigin> origin_;
};

MPIReporter::MPIReporter(edm::ParameterSet const& config) : origin_(consumes<MPIOrigin>(edm::InputTag("source"))) {}

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

  auto const& origin = event.get(origin_);
  {
    edm::LogAbsolute log("MPI");
    log << "original process rank: " << origin.rank();
    log << "\noriginal process stream: " << origin.stream();
  }
}

void MPIReporter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MPIReporter);
