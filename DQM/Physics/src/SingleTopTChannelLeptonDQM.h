#ifndef SINGLETOPTCHANNELLEPTONDQM
#define SINGLETOPTCHANNELLEPTONDQM

#include <string>
#include <vector>

#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DQM/Physics/interface/TopDQMHelpers.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"

/**
   \class   MonitorEnsemble TopDQMHelpers.h
   "DQM/Physics/interface/TopDQMHelpers.h"

   \brief   Helper class to define histograms for monitoring of
   muon/electron/jet/met quantities.

   Helper class to contain histograms for the monitoring of
   muon/electron/jet/met quantities.
   This class can be instantiated several times after several event selection
   steps. It can
   be used to fill histograms in three different granularity levels according to
   STANDARD
   (<10 histograms), VERBOSE(<20 histograms), DEBUG(<30 histgorams). Note that
   for the sake
   of simplicity and to force the analyst to keep the number of histograms to be
   monitored
   small the MonitorEnsemble class contains the histograms for all objects at
   once. It should
   not contain much more than 10 histograms though in the STANDARD
   configuration, as these
   histograms will be monitored at each SelectionStep. Monitoring of histograms
   after selec-
   tion steps within the same object collection needs to be implemented within
   the Monitor-
   Ensemble. It will not be covered by the SelectionStep class.
*/

namespace SingleTopTChannelLepton {
  using dqm::legacy::DQMStore;
  using dqm::legacy::MonitorElement;

  class MonitorEnsemble {
  public:
    /// different verbosity levels
    enum Level { STANDARD, VERBOSE, DEBUG };

  public:
    /// default contructor
    MonitorEnsemble(const char* label,
                    const edm::ParameterSet& cfg,
                    const edm::VParameterSet& vcfg,
                    edm::ConsumesCollector&& iC);
    /// default destructor
    ~MonitorEnsemble() {}

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
    std::vector<edm::EDGetTokenT<edm::View<reco::MET> > > mets_;
    //    std::vector<edm::InputTag> mets_;
    /// input sources for monitoring
    edm::EDGetTokenT<edm::View<reco::Jet> > jets_;
    edm::EDGetTokenT<edm::View<reco::PFCandidate> > muons_;
    edm::EDGetTokenT<edm::View<reco::GsfElectron> > elecs_gsf_;
    edm::EDGetTokenT<edm::View<reco::PFCandidate> > elecs_;
    edm::EDGetTokenT<edm::View<reco::Vertex> > pvs_;

    //    edm::InputTag elecs_, elecs_gsf_, muons_, muons_reco_, jets_, pvs_;

    /// trigger table
    //    edm::InputTag triggerTable_;
    edm::EDGetTokenT<edm::TriggerResults> triggerTable_;

    /// trigger paths for monitoring, expected
    /// to be of form signalPath:MonitorPath
    std::vector<std::string> triggerPaths_;

    /// electronId label
    //    edm::InputTag electronId_;
    edm::EDGetTokenT<edm::ValueMap<float> > electronId_;
    /// electronId pattern we expect the following pattern:
    ///  0: fails
    ///  1: passes electron ID only
    ///  2: passes electron Isolation only
    ///  3: passes electron ID and Isolation only
    ///  4: passes conversion rejection
    ///  5: passes conversion rejection and ID
    ///  6: passes conversion rejection and Isolation
    ///  7: passes the whole selection
    /// As described on
    /// https://twiki.cern.ch/twiki/bin/view/CMS/SimpleCutBasedEleID
    // int eidPattern_;
    // the cut for the MVA Id
    double eidCutValue_;
    /// extra isolation criterion on electron
    std::string elecIso_;
    /// extra selection on electrons
    std::unique_ptr<StringCutObjectSelector<reco::PFCandidate> > elecSelect_;
    edm::InputTag rhoTag;

    /// extra selection on primary vertices; meant to investigate the pile-up
    /// effect
    std::unique_ptr<StringCutObjectSelector<reco::Vertex> > pvSelect_;

    /// extra isolation criterion on muon
    std::unique_ptr<StringCutObjectSelector<reco::PFCandidate> > muonIso_;

    /// extra selection on muons
    std::unique_ptr<StringCutObjectSelector<reco::PFCandidate> > muonSelect_;
    /// jetCorrector
    edm::EDGetTokenT<reco::JetCorrector> jetCorrector_;
    /// jetID as an extra selection type
    edm::EDGetTokenT<reco::JetIDValueMap> jetIDLabel_;

    /// extra jetID selection on calo jets
    std::unique_ptr<StringCutObjectSelector<reco::JetID> > jetIDSelect_;
    std::unique_ptr<StringCutObjectSelector<reco::PFJet> > jetlooseSelection_;
    std::unique_ptr<StringCutObjectSelector<reco::PFJet> > jetSelection_;
    /// extra selection on jets (here given as std::string as it depends
    /// on the the jet type, which selections are valid and which not)
    std::string jetSelect_;
    /// include btag information or not
    /// to be determined from the cfg
    bool includeBTag_;
    /// btag discriminator labels
    //    edm::InputTag btagEff_, btagPur_, btagVtx_, btagCombVtx_;
    edm::EDGetTokenT<reco::JetTagCollection> btagEff_, btagPur_, btagVtx_, btagCSV_, btagCombVtx_;

    /// btag working points
    double btagEffWP_, btagPurWP_, btagVtxWP_, btagCSVWP_, btagCombVtxWP_;
    /// mass window upper and lower edge
    double lowerEdge_, upperEdge_;

    /// number of logged interesting events
    int logged_;
    /// storage manager
    /// histogram container
    std::map<std::string, MonitorElement*> hists_;
    edm::EDConsumerBase tmpConsumerBase;

    std::string directory_;

    std::unique_ptr<StringCutObjectSelector<reco::PFCandidate, true> > muonSelect;
    std::unique_ptr<StringCutObjectSelector<reco::PFCandidate, true> > muonIso;

    std::unique_ptr<StringCutObjectSelector<reco::PFCandidate, true> > elecSelect;
    std::unique_ptr<StringCutObjectSelector<reco::PFCandidate, true> > elecIso;
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
}  // namespace SingleTopTChannelLepton

#include <utility>

#include "DQM/Physics/interface/TopDQMHelpers.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Common/interface/TriggerResults.h"

/**
   \class   SingleTopTChannelLeptonDQM SingleTopTChannelLeptonDQM.h
   "DQM/Physics/plugins/SingleTopTChannelLeptonDQM.h"

   \brief   Module to apply a monitored selection of top like events in the
   semi-leptonic channel

   Plugin to apply a monitored selection of top like events with some minimal
   flexibility in the number and definition of the selection steps. To achieve
   this flexibility it employes the SelectionStep class. The MonitorEnsemble
   class is used to provide a well defined set of histograms to be monitored
   after each selection step. The SelectionStep class provides a flexible and
   intuitive selection via the StringCutParser.  SelectionStep and
   MonitorEnsemble classes are interleaved. The monitoring starts after a
   preselection step (which is not monitored in the context of this module) with
   an instance of the MonitorEnsemble class. The following objects are supported
   for selection:

    - jets  : of type reco::Jet (jets), reco::CaloJet (jets/calo) or reco::PFJet
   (jets/pflow)
    - elecs : of type reco::GsfElectron
    - muons : of type reco::Muon
    - met   : of type reco::MET

   These types have to be present as prefix of the selection step paramter
   _label_ separated from the rest of the label by a ':' (e.g. in the form
   "jets:step0"). The class expects selection labels of this type. They will be
   disentangled by the private helper functions _objectType_ and _seletionStep_
   as declared below.
*/

/// define MonitorEnsembple to be used
// using SingleTopTChannelLepton::MonitorEnsemble;

class SingleTopTChannelLeptonDQM : public DQMOneEDAnalyzer<> {
public:
  /// default constructor
  SingleTopTChannelLeptonDQM(const edm::ParameterSet& cfg);
  /// default destructor
  ~SingleTopTChannelLeptonDQM() override{};

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
  /// primary vertex
  edm::InputTag vertex_;
  edm::EDGetTokenT<reco::Vertex> vertex__;
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
  std::map<std::string, std::pair<edm::ParameterSet, std::unique_ptr<SingleTopTChannelLepton::MonitorEnsemble> > >
      selection_;

  std::unique_ptr<SelectionStep<reco::Muon> > MuonStep;
  std::unique_ptr<SelectionStep<reco::PFCandidate> > PFMuonStep;
  std::unique_ptr<SelectionStep<reco::GsfElectron> > ElectronStep;
  std::unique_ptr<SelectionStep<reco::PFCandidate> > PFElectronStep;
  std::unique_ptr<SelectionStep<reco::Vertex> > PvStep;

  std::vector<std::unique_ptr<SelectionStep<reco::Jet> > > JetSteps;
  std::vector<std::unique_ptr<SelectionStep<reco::CaloJet> > > CaloJetSteps;
  std::vector<std::unique_ptr<SelectionStep<reco::PFJet> > > PFJetSteps;

  std::unique_ptr<SelectionStep<reco::MET> > METStep;
  std::vector<edm::ParameterSet> sel;
};

#endif
