#ifndef TOPSINGLELEPTONDQM_MINIAOD
#define TOPSINGLELEPTONDQM_MINIAOD

#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include <string>
#include <vector>
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DQM/Physics/interface/TopDQMHelpers.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"

namespace TopSingleLepton_miniAOD {
  using dqm::legacy::DQMStore;
  using dqm::legacy::MonitorElement;

  class MonitorEnsemble {
  public:
    /// different verbosity levels
    enum Level { STANDARD, VERBOSE, DEBUG };

  public:
    /// default contructor
    MonitorEnsemble(const char* label, const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC);
    /// default destructor
    ~MonitorEnsemble(){};

    /// book histograms in subdirectory _directory_
    void book(DQMStore::IBooker& ibooker);
    /// fill monitor histograms with electronId and jetCorrections
    void fill(const edm::Event& event, const edm::EventSetup& setup);

  private:
    /// deduce monitorPath from label, the label is expected
    /// to be of type 'selectionPath:monitorPath'
    std::string monitorPath(const std::string& label) const { return label.substr(label.find(':') + 1); };
    /// deduce selectionPath from label, the label is
    /// expected to be of type 'selectionPath:monitorPath'
    std::string selectionPath(const std::string& label) const { return label.substr(0, label.find(':')); };

    /// set configurable labels for trigger monitoring histograms
    void triggerBinLabels(std::string channel, const std::vector<std::string> labels);
    /// fill trigger monitoring histograms
    void fill(const edm::Event& event,
              const edm::TriggerResults& triggerTable,
              std::string channel,
              const std::vector<std::string> labels) const;

    /// check if histogram was booked
    bool booked(const std::string histName) const { return hists_.find(histName) != hists_.end(); };
    /// fill histogram if it had been booked before
    void fill(const std::string histName, double value) const {
      if (booked(histName))
        hists_.find(histName)->second->Fill(value);
    };
    /// fill histogram if it had been booked before (2-dim version)
    void fill(const std::string histName, double xValue, double yValue) const {
      if (booked(histName))
        hists_.find(histName)->second->Fill(xValue, yValue);
    };
    /// fill histogram if it had been booked before (2-dim version)
    void fill(const std::string histName, double xValue, double yValue, double zValue) const {
      if (booked(histName))
        hists_.find(histName)->second->Fill(xValue, yValue, zValue);
    };

  private:
    /// verbosity level for booking
    Level verbosity_;
    /// instance label
    std::string label_;
    /// considers a vector of METs
    std::vector<edm::EDGetTokenT<edm::View<pat::MET> > > mets_;
    /// input sources for monitoring
    edm::EDGetTokenT<edm::View<pat::Jet> > jets_;
    edm::EDGetTokenT<edm::View<pat::Muon> > muons_;
    edm::EDGetTokenT<edm::View<pat::Electron> > elecs_;
    edm::EDGetTokenT<edm::View<reco::Vertex> > pvs_;
    /// trigger table
    edm::EDGetTokenT<edm::TriggerResults> triggerTable_;
    /// trigger paths for monitoring, expected
    /// to be of form signalPath:MonitorPath
    std::vector<std::string> triggerPaths_;

    edm::InputTag rhoTag;

    /// electronId label
    edm::EDGetTokenT<edm::ValueMap<float> > electronId_;

    double eidCutValue_;
    /// extra isolation criterion on electron
    std::unique_ptr<StringCutObjectSelector<pat::Electron> > elecIso_;
    /// extra selection on electrons
    std::unique_ptr<StringCutObjectSelector<pat::Electron> > elecSelect_;

    /// extra selection on primary vertices; meant to investigate the pile-up
    /// effect
    std::unique_ptr<StringCutObjectSelector<reco::Vertex> > pvSelect_;

    /// extra isolation criterion on muon
    std::unique_ptr<StringCutObjectSelector<pat::Muon> > muonIso_;

    /// extra selection on muons
    std::unique_ptr<StringCutObjectSelector<pat::Muon> > muonSelect_;

    /// jetID as an extra selection type
    edm::EDGetTokenT<reco::JetIDValueMap> jetIDLabel_;
    /// extra jetID selection on calo jets
    std::unique_ptr<StringCutObjectSelector<reco::JetID> > jetIDSelect_;
    /// extra selection on jets (here given as std::string as it depends
    /// on the the jet type, which selections are valid and which not)
    std::string jetSelect_;
    std::unique_ptr<StringCutObjectSelector<pat::Jet> > jetSelect;
    /// include btag information or not
    /// to be determined from the cfg
    bool includeBTag_;
    /// btag discriminator labels
    edm::EDGetTokenT<reco::JetTagCollection> btagEff_, btagPur_, btagVtx_, btagCSV_;
    /// btag working points
    double btagEffWP_, btagPurWP_, btagVtxWP_, btagCSVWP_;
    /// mass window upper and lower edge
    double lowerEdge_, upperEdge_;

    /// number of logged interesting events
    int logged_;

    /// histogram container
    std::map<std::string, MonitorElement*> hists_;
    edm::EDConsumerBase tmpConsumerBase;
    std::string directory_;
  };

  inline void MonitorEnsemble::triggerBinLabels(std::string channel, const std::vector<std::string> labels) {
    for (unsigned int idx = 0; idx < labels.size(); ++idx) {
      hists_[channel + "Mon_"]->setBinLabel(idx + 1, "[" + monitorPath(labels[idx]) + "]", 1);
      hists_[channel + "Eff_"]->setBinLabel(
          idx + 1, "[" + selectionPath(labels[idx]) + "]|[" + monitorPath(labels[idx]) + "]", 1);
    }
  }

  inline void MonitorEnsemble::fill(const edm::Event& event,
                                    const edm::TriggerResults& triggerTable,
                                    std::string channel,
                                    const std::vector<std::string> labels) const {
    for (unsigned int idx = 0; idx < labels.size(); ++idx) {
      if (accept(event, triggerTable, monitorPath(labels[idx]))) {
        fill(channel + "Mon_", idx + 0.5);
        // take care to fill triggerMon_ before evts is being called
        int evts = hists_.find(channel + "Mon_")->second->getBinContent(idx + 1);
        double value = hists_.find(channel + "Eff_")->second->getBinContent(idx + 1);
        fill(
            channel + "Eff_", idx + 0.5, 1. / evts * (accept(event, triggerTable, selectionPath(labels[idx])) - value));
      }
    }
  }
}  // namespace TopSingleLepton_miniAOD

#include <utility>

#include "DQM/Physics/interface/TopDQMHelpers.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Common/interface/TriggerResults.h"

class TopSingleLeptonDQM_miniAOD : public DQMOneEDAnalyzer<> {
public:
  /// default constructor
  TopSingleLeptonDQM_miniAOD(const edm::ParameterSet& cfg);
  /// default destructor
  ~TopSingleLeptonDQM_miniAOD() override{};

  /// do this during the event loop
  void analyze(const edm::Event& event, const edm::EventSetup& setup) override;

protected:
  //Book histograms
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  /// deduce object type from ParameterSet label, the label
  /// is expected to be of type 'objectType:selectionStep'
  std::string objectType(const std::string& label) { return label.substr(0, label.find(':')); };
  /// deduce selection step from ParameterSet label, the
  /// label is expected to be of type 'objectType:selectionStep'
  std::string selectionStep(const std::string& label) { return label.substr(label.find(':') + 1); };

private:
  /// trigger table
  edm::EDGetTokenT<edm::TriggerResults> triggerTable__;
  /// trigger paths
  std::vector<std::string> triggerPaths_;
  /// string cut selector
  std::unique_ptr<StringCutObjectSelector<reco::Vertex> > vertexSelect_;

  /// beamspot
  edm::InputTag beamspot_;
  edm::EDGetTokenT<reco::BeamSpot> beamspot__;
  /// string cut selector
  std::unique_ptr<StringCutObjectSelector<reco::BeamSpot> > beamspotSelect_;

  /// needed to guarantee the selection order as defined by the order of
  /// ParameterSets in the _selection_ vector as defined in the config
  std::vector<std::string> selectionOrder_;
  /// this is the heart component of the plugin; std::string keeps a label
  /// the selection step for later identification, edm::ParameterSet keeps
  /// the configuration of the selection for the SelectionStep class,
  /// MonitoringEnsemble keeps an instance of the MonitorEnsemble class to
  /// be filled _after_ each selection step
  std::map<std::string, std::pair<edm::ParameterSet, std::unique_ptr<TopSingleLepton_miniAOD::MonitorEnsemble> > >
      selection_;
  std::unique_ptr<SelectionStep<pat::Muon> > MuonStep;
  std::unique_ptr<SelectionStep<pat::Electron> > ElectronStep;
  std::unique_ptr<SelectionStep<reco::Vertex> > PvStep;
  std::unique_ptr<SelectionStep<pat::MET> > METStep;
  std::vector<std::unique_ptr<SelectionStep<pat::Jet> > > JetSteps;

  std::vector<edm::ParameterSet> sel_;
  edm::ParameterSet setup_;
};

#endif

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
