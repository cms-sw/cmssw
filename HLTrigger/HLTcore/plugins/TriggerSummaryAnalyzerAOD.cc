/** \class TriggerSummaryAnalyzerAOD
 *
 *  This class is an EDAnalyzer analyzing the HLT summary object for AOD
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//
class TriggerSummaryAnalyzerAOD : public edm::global::EDAnalyzer<> {
public:
  explicit TriggerSummaryAnalyzerAOD(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override;

private:
  /// InputTag of TriggerEvent to analyze
  const edm::InputTag inputTag_;
  const edm::EDGetTokenT<trigger::TriggerEvent> inputToken_;
};

//
// constructors and destructor
//
TriggerSummaryAnalyzerAOD::TriggerSummaryAnalyzerAOD(const edm::ParameterSet& ps)
    : inputTag_(ps.getParameter<edm::InputTag>("inputTag")), inputToken_(consumes<trigger::TriggerEvent>(inputTag_)) {}

//
// member functions
//

void TriggerSummaryAnalyzerAOD::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputTag", edm::InputTag("hltTriggerSummaryAOD"));
  descriptions.add("triggerSummaryAnalyzerAOD", desc);
}

// ------------ method called to produce the data  ------------
void TriggerSummaryAnalyzerAOD::analyze(edm::StreamID, const edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  LogVerbatim("TriggerSummaryAnalyzerAOD") << endl;
  LogVerbatim("TriggerSummaryAnalyzerAOD")
      << "TriggerSummaryAnalyzerAOD: content of TriggerEvent: " << inputTag_.encode() << endl;

  Handle<TriggerEvent> handle;
  iEvent.getByToken(inputToken_, handle);
  if (handle.isValid()) {
    LogVerbatim("TriggerSummaryAnalyzerAOD") << "Used Processname: " << handle->usedProcessName() << endl;
    const size_type nC(handle->sizeCollections());
    LogVerbatim("TriggerSummaryAnalyzerAOD") << "Number of packed Collections: " << nC << endl;
    LogVerbatim("TriggerSummaryAnalyzerAOD") << "The Collections: #, tag, 1-past-end index" << endl;
    for (size_type iC = 0; iC != nC; ++iC) {
      LogVerbatim("TriggerSummaryAnalyzerAOD")
          << iC << " " << handle->collectionTag(iC).encode() << " " << handle->collectionKey(iC) << endl;
    }
    const size_type nO(handle->sizeObjects());
    LogVerbatim("TriggerSummaryAnalyzerAOD") << "Number of TriggerObjects: " << nO << endl;
    LogVerbatim("TriggerSummaryAnalyzerAOD") << "The TriggerObjects: #, id, pt, eta, phi, mass" << endl;
    const TriggerObjectCollection& TOC(handle->getObjects());
    for (size_type iO = 0; iO != nO; ++iO) {
      const TriggerObject& TO(TOC[iO]);
      LogVerbatim("TriggerSummaryAnalyzerAOD")
          << iO << " " << TO.id() << " " << TO.pt() << " " << TO.eta() << " " << TO.phi() << " " << TO.mass() << endl;
    }
    const size_type nF(handle->sizeFilters());
    LogVerbatim("TriggerSummaryAnalyzerAOD") << "Number of TriggerFilters: " << nF << endl;
    LogVerbatim("TriggerSummaryAnalyzerAOD") << "The Filters: #, tag, #ids/#keys, the id/key pairs" << endl;
    for (size_type iF = 0; iF != nF; ++iF) {
      const Vids& VIDS(handle->filterIds(iF));
      const Keys& KEYS(handle->filterKeys(iF));
      const size_type nI(VIDS.size());
      const size_type nK(KEYS.size());
      LogVerbatim("TriggerSummaryAnalyzerAOD")
          << iF << " " << handle->filterTag(iF).encode() << " " << nI << "/" << nK << " the pairs: ";
      const size_type n(max(nI, nK));
      for (size_type i = 0; i != n; ++i) {
        LogVerbatim("TriggerSummaryAnalyzerAOD") << " " << VIDS[i] << "/" << KEYS[i];
      }
      LogVerbatim("TriggerSummaryAnalyzerAOD") << endl;
      assert(nI == nK);
    }
  } else {
    LogVerbatim("TriggerSummaryAnalyzerAOD") << "Handle invalid! Check InputTag provided." << endl;
  }
  LogVerbatim("TriggerSummaryAnalyzerAOD") << endl;

  return;
}

DEFINE_FWK_MODULE(TriggerSummaryAnalyzerAOD);
