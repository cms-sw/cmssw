#include <iostream>
#include <sstream>

#include <mpi.h>

#include <TBufferFile.h>
#include <TClass.h>

#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/GenericHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Reflection/interface/ObjectWithDict.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Guid.h"
#include "HeterogeneousCore/MPIServices/interface/MPIService.h"

#include "api.h"
#include "messages.h"

/* MPIDriver class
 *
 * This module runs inside a CMSSW job (the "driver") and connects to an "MPISource" in a separate CMSSW job (the "follower").
 * The follower is informed of all stream transitions seen by the driver, and can replicate them in it
 *
 * Current limitations:
 *   - support a single "follower"
 *
 * Future work:
 *   - support multiple "followers"
 */

class MPIDriver : public edm::stream::EDProducer<> {
public:
  explicit MPIDriver(edm::ParameterSet const& config);
  ~MPIDriver() override;

  void beginStream(edm::StreamID sid) override;
  void endStream() override;

  void beginRun(edm::Run const& run, edm::EventSetup const& setup) override;
  void endRun(edm::Run const& run, edm::EventSetup const& setup) override;

  void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) override;
  void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) override;

  void produce(edm::Event& event, edm::EventSetup const& setup) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::StreamID sid_ = edm::StreamID::invalidStreamID();

  MPISender sender_;
  MPI_Comm comm_ = MPI_COMM_NULL;

  /* FIXME move to MPISend ?
  std::vector<std::string> eventLabels_;
  std::vector<edm::BranchDescription> branches_;
  std::vector<edm::EDGetToken> tokens_;
  */
};

MPIDriver::MPIDriver(edm::ParameterSet const& config)
/* FIXME move to MPISend ?
    : eventLabels_(config.getUntrackedParameter<std::vector<std::string>>("eventProducts")) {
  // sort the labels, so they can be used with binary_search
  std::sort(eventLabels_.begin(), eventLabels_.end());
  */
{
  // make sure that MPI is initialised
  MPIService::required();

  // FIXME move into the MPIService ?
  // make sure the EDM MPI types are available
  EDM_MPI_build_types();

  // look up the "server" port
  char port[MPI_MAX_PORT_NAME];
  MPI_Lookup_name("server", MPI_INFO_NULL, port);
  edm::LogAbsolute("MPI") << "Trying to connect to the MPI server on port " << port;

  // connect to the server
  int size;
  MPI_Comm_connect(port, MPI_INFO_NULL, 0, MPI_COMM_SELF, &comm_);
  MPI_Comm_remote_size(comm_, &size);
  edm::LogAbsolute("MPI") << "Client connected to " << size << (size == 1 ? " server" : " servers");
  if (size > 1) {
    throw cms::Exception("UnsupportedFeature")
        << "MPIDriver supports only a single follower, but it was connected to " << size << " followers";
  }
  sender_ = MPISender(comm_, 0);

  /* FIXME move to MPISend ?
  // register a dependency on all the event products described by the "eventProducts"
  callWhenNewProductsRegistered([this](edm::BranchDescription const& branch) {
    static const std::string kWildcard("*");
    static const std::string kPathStatus("edm::PathStatus");
    static const std::string kEndPathStatus("edm::EndPathStatus");

    switch (branch.branchType()) {
      case edm::InEvent:
        if (std::binary_search(eventLabels_.begin(), eventLabels_.end(), branch.moduleLabel()) or
            (std::binary_search(eventLabels_.begin(), eventLabels_.end(), kWildcard) and
             branch.className() != kPathStatus and branch.className() != kEndPathStatus)) {
          tokens_.push_back(
              this->consumes(edm::TypeToGet{branch.unwrappedTypeID(), edm::PRODUCT_TYPE},
                             edm::InputTag{branch.moduleLabel(), branch.productInstanceName(), branch.processName()}));
          branches_.push_back(branch);
        }
        break;

      default:
          // ignore the other product types
          ;
    }
  });
  */
}

MPIDriver::~MPIDriver() {
  // close the intercommunicator
  MPI_Comm_disconnect(&comm_);
}

void MPIDriver::beginStream(edm::StreamID sid) {
  // store this stream's id
  sid_ = sid;

  // signal the connection
  sender_.sendConnect(sid_);

  /* is there a way to access all known process histories ?
  edm::ProcessHistoryRegistry const& registry = * edm::ProcessHistoryRegistry::instance();
  edm::LogAbsolute("MPI") << "ProcessHistoryRegistry:";
  for (auto const& keyval: registry) {
    edm::LogAbsolute("MPI") << keyval.first << ": " << keyval.second;
  }
  */

  /* FIXME move to MPISend ?
  // send the branch descriptions for all event products
  for (auto& branch : branches_) {
    sender_.sendSerializedProduct(sid_, branch);
  }
  // indicate that all branches have been sent
  sender_.sendComplete(sid_);
  */
  // signal the begin stream
  sender_.sendBeginStream(sid_);
}

void MPIDriver::endStream() {
  // signal the end stream
  sender_.sendEndStream(sid_);
  // signal the disconnection
  sender_.sendDisconnect(sid_);
}

void MPIDriver::beginRun(edm::Run const& run, edm::EventSetup const& setup) {
  // signal a new run, and transmit the RunAuxiliary
  /* FIXME
   * Ideally the ProcessHistoryID stored in the run.runAuxiliary() should be the correct one, and
   * we could simply do

  sender_.sendBeginRun(sid_, run.runAuxiliary());

   * Instead, it looks like the ProcessHistoryID stored in the run.runAuxiliary() is that of the
   * _parent_ process.
   * So, we make a copy of the RunAuxiliary, set the ProcessHistoryID to the correct value, and
   * transmit the modified RunAuxiliary.
   */
  auto aux = run.runAuxiliary();
  aux.setProcessHistoryID(run.processHistory().id());
  sender_.sendBeginRun(sid_, aux);
  // transmit the ProcessHistory
  sender_.sendSerializedProduct(sid_, run.processHistory());
}

void MPIDriver::endRun(edm::Run const& run, edm::EventSetup const& setup) {
  // signal the end of run
  /* FIXME
   * Ideally the ProcessHistoryID stored in the run.runAuxiliary() should be the correct one, and
   * we could simply do

  sender_.sendEndRun(sid_, run.runAuxiliary());

   * Instead, it looks like the ProcessHistoryID stored in the run.runAuxiliary() is that of the
   * _parent_ process.
   * So, we make a copy of the RunAuxiliary, set the ProcessHistoryID to the correct value, and
   * transmit the modified RunAuxiliary.
   */
  auto aux = run.runAuxiliary();
  aux.setProcessHistoryID(run.processHistory().id());
  sender_.sendEndRun(sid_, aux);
}

void MPIDriver::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) {
  // signal a new luminosity block, and transmit the LuminosityBlockAuxiliary
  /* FIXME
   * Ideally the ProcessHistoryID stored in the lumi.luminosityBlockAuxiliary() should be the
   * correct one, and we could simply do

  sender_.sendBeginLuminosityBlock(sid_, lumi.luminosityBlockAuxiliary());

   * Instead, it looks like the ProcessHistoryID stored in the lumi.luminosityBlockAuxiliary() is
   * that of the _parent_ process.
   * So, we make a copy of the LuminosityBlockAuxiliary, set the ProcessHistoryID to the correct
   * value, and transmit the modified LuminosityBlockAuxiliary.
   */
  auto aux = lumi.luminosityBlockAuxiliary();
  aux.setProcessHistoryID(lumi.processHistory().id());
  sender_.sendBeginLuminosityBlock(sid_, aux);
}

void MPIDriver::endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) {
  // signal the end of luminosity block
  /* FIXME
   * Ideally the ProcessHistoryID stored in the lumi.luminosityBlockAuxiliary() should be the
   * correct one, and we could simply do

  sender_.sendEndLuminosityBlock(sid_, lumi.luminosityBlockAuxiliary());

   * Instead, it looks like the ProcessHistoryID stored in the lumi.luminosityBlockAuxiliary() is
   * that of the _parent_ process.
   * So, we make a copy of the LuminosityBlockAuxiliary, set the ProcessHistoryID to the correct
   * value, and transmit the modified LuminosityBlockAuxiliary.
   */
  auto aux = lumi.luminosityBlockAuxiliary();
  aux.setProcessHistoryID(lumi.processHistory().id());
  sender_.sendEndLuminosityBlock(sid_, aux);
}

void MPIDriver::produce(edm::Event& event, edm::EventSetup const& setup) {
  /*
  {
    edm::LogAbsolute log("MPI");
    log << "stream " << sid_ << ": processing run " << event.run() << ", lumi " << event.luminosityBlock() << ", event "
        << event.id().event();
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
  */

  // signal a new event, and transmit the EventAuxiliary
  sender_.sendEvent(sid_, event.eventAuxiliary());

  /* FIXME move to MPISend ?
  // transmit the event data products
  unsigned int size = tokens_.size();
  assert(branches_.size() == size);
  for (unsigned int i = 0; i < size; ++i) {
    auto const& token = tokens_[i];
    auto const& branch = branches_[i];
    auto const& type = branch.unwrappedType();
    edm::GenericHandle handle(type);
    event.getByToken(token, handle);
    if (handle.isValid()) {
      // transmit the BranchKey in order to reconstruct the BranchDescription on the receiving side
      sender_.sendSerializedProduct(sid_, edm::BranchKey(branch));
      // transmit the ProductProvenance
      sender_.sendSerializedProduct(sid_, *handle.provenance()->productProvenance());
      // transmit the ProductID
      sender_.sendSerializedProduct(sid_, handle.id());
      // transmit the wrapped product
      sender_.sendSerializedProduct(sid_, *handle.product());
    }
  }

  // indicate that all products have been sent
  sender_.sendComplete(sid_);
  */

  // FIXME produce the MPI "token"
  //event.emplace(...);
}

void MPIDriver::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  descriptions.setComment(
      "This module connects to an \"MPISource\" in a separate CMSSW job, and transmits all "
      "Runs, LuminosityBlocks and Events from the current process to the remote one.");

  edm::ParameterSetDescription desc;
  /* FIXME move to MPISend ?
  desc.addUntracked<std::vector<std::string>>("eventProducts", {})
      ->setComment(
          "List of modules whose event products will be sent to the remote process. "
          "Use \"*\" to send all event products.");
  */
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(MPIDriver);
