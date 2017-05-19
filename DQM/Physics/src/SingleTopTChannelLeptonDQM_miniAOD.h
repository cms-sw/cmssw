#ifndef SINGLETOPTCHANNELLEPTONDQM_MINIAOD
#define SINGLETOPTCHANNELLEPTONDQM_MINIAOD

#include <string>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DQM/Physics/interface/TopDQMHelpers.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/PatCandidates/interface/Electron.h"


namespace SingleTopTChannelLepton_miniAOD {

class MonitorEnsemble {
 public:
  /// different verbosity levels
  enum Level {
    STANDARD,
    VERBOSE,
    DEBUG
  };

 public:
  /// default contructor
  MonitorEnsemble(const char* label, const edm::ParameterSet& cfg,
                  const edm::VParameterSet& vcfg, edm::ConsumesCollector&& iC);
  /// default destructor
  ~MonitorEnsemble() {}

  /// book histograms in subdirectory _directory_
  void book(DQMStore::IBooker & ibooker);
  /// fill monitor histograms with electronId and jetCorrections
  void fill(const edm::Event& event, const edm::EventSetup& setup);

 private:
  /// deduce monitorPath from label, the label is expected
  /// to be of type 'selectionPath:monitorPath'
  std::string monitorPath(const std::string& label) const {
    return label.substr(label.find(':') + 1);
  };
  /// deduce selectionPath from label, the label is
  /// expected to be of type 'selectionPath:monitorPath'
  std::string selectionPath(const std::string& label) const {
    return label.substr(0, label.find(':'));
  };

  /// set configurable labels for trigger monitoring histograms
  void triggerBinLabels(std::string channel,
                        const std::vector<std::string> labels);
  /// fill trigger monitoring histograms
  void fill(const edm::Event& event, const edm::TriggerResults& triggerTable,
            std::string channel, const std::vector<std::string> labels) const;

  /// check if histogram was booked
  bool booked(const std::string histName) const {
    return hists_.find(histName.c_str()) != hists_.end();
  };
  /// fill histogram if it had been booked before
  void fill(const std::string histName, double value) const {
    if (booked(histName.c_str()))
      hists_.find(histName.c_str())->second->Fill(value);
  };
  /// fill histogram if it had been booked before (2-dim version)
  void fill(const std::string histName, double xValue, double yValue) const {
    if (booked(histName.c_str()))
      hists_.find(histName.c_str())->second->Fill(xValue, yValue);
  };
  /// fill histogram if it had been booked before (2-dim version)
  void fill(const std::string histName, double xValue, double yValue,
            double zValue) const {
    if (booked(histName.c_str()))
      hists_.find(histName.c_str())->second->Fill(xValue, yValue, zValue);
  };

 private:
  /// verbosity level for booking
  Level verbosity_;
  /// instance label
  std::string label_;
  /// considers a vector of METs
  std::vector<edm::EDGetTokenT<edm::View<pat::MET> > > mets_;
  //    std::vector<edm::InputTag> mets_;
  /// input sources for monitoring
  edm::EDGetTokenT<edm::View<pat::Jet> > jets_;
  edm::EDGetTokenT<edm::View<pat::Muon> > muons_;
  edm::EDGetTokenT<edm::View<pat::Electron> > elecs_gsf_;
  edm::EDGetTokenT<edm::View<pat::Muon> > elecs_;
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


  double eidCutValue_;
  /// extra isolation criterion on electron
  std::string elecIso_;
  /// extra selection on electrons
  std::string elecSelect_;

  /// extra selection on primary vertices; meant to investigate the pile-up
  /// effect
  std::unique_ptr<StringCutObjectSelector<reco::Vertex> > pvSelect_;

  /// extra isolation criterion on muon
  std::string muonIso_;
  /// extra selection on muons
  std::string muonSelect_;

  /// jetCorrector
  std::string jetCorrector_;
  /// jetID as an extra selection type
  edm::EDGetTokenT<reco::JetIDValueMap> jetIDLabel_;

  /// extra jetID selection on calo jets
  std::unique_ptr<StringCutObjectSelector<reco::JetID> > jetIDSelect_;

  /// extra selection on jets (here given as std::string as it depends
  /// on the the jet type, which selections are valid and which not)
  std::string jetSelect_;
  /// include btag information or not
  /// to be determined from the cfg
  bool includeBTag_;
  /// btag discriminator labels
  //    edm::InputTag btagEff_, btagPur_, btagVtx_, btagCombVtx_;
  edm::EDGetTokenT<reco::JetTagCollection> btagEff_, btagPur_, btagVtx_,
      btagCombVtx_;

  /// btag working points
  double btagEffWP_, btagPurWP_, btagVtxWP_, btagCombVtxWP_;
  /// mass window upper and lower edge
  double lowerEdge_, upperEdge_;

  /// number of logged interesting events
  int logged_;
  /// storage manager
  /// histogram container
  std::map<std::string, MonitorElement*> hists_;
  edm::EDConsumerBase tmpConsumerBase;

  std::string directory_;

  std::unique_ptr<StringCutObjectSelector<pat::Muon, true> > muonSelect;
  std::unique_ptr<StringCutObjectSelector<pat::Muon, true> > muonIso;
  
  std::unique_ptr<StringCutObjectSelector<reco::CaloJet> > jetSelectCalo;
  std::unique_ptr<StringCutObjectSelector<reco::PFJet> > jetSelectPF;
  std::unique_ptr<StringCutObjectSelector<pat::Jet> > jetSelectJet;
  
  std::unique_ptr<StringCutObjectSelector<pat::Electron, true> > elecSelect;
  std::unique_ptr<StringCutObjectSelector<pat::Electron, true> > elecIso;


};

inline void MonitorEnsemble::triggerBinLabels(
    std::string channel, const std::vector<std::string> labels) {
  for (unsigned int idx = 0; idx < labels.size(); ++idx) {
    hists_[(channel + "Mon_").c_str()]
        ->setBinLabel(idx + 1, "[" + monitorPath(labels[idx]) + "]", 1);
    hists_[(channel + "Eff_").c_str()]
        ->setBinLabel(idx + 1, "[" + selectionPath(labels[idx]) + "]|[" +
                                   monitorPath(labels[idx]) + "]",
                      1);
  }
}

inline void MonitorEnsemble::fill(const edm::Event& event,
                                  const edm::TriggerResults& triggerTable,
                                  std::string channel,
                                  const std::vector<std::string> labels) const {
  for (unsigned int idx = 0; idx < labels.size(); ++idx) {
    if (accept(event, triggerTable, monitorPath(labels[idx]))) {
      fill((channel + "Mon_").c_str(), idx + 0.5);
      // take care to fill triggerMon_ before evts is being called
      int evts = hists_.find((channel + "Mon_").c_str())
                     ->second->getBinContent(idx + 1);
      double value = hists_.find((channel + "Eff_").c_str())
                         ->second->getBinContent(idx + 1);
      fill(
          (channel + "Eff_").c_str(), idx + 0.5,
          1. / evts * (accept(event, triggerTable, selectionPath(labels[idx])) -
                       value));
    }
  }
}
}

#include <utility>

#include "DQM/Physics/interface/TopDQMHelpers.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"



class SingleTopTChannelLeptonDQM_miniAOD : public DQMEDAnalyzer {
 public:
  /// default constructor
  SingleTopTChannelLeptonDQM_miniAOD(const edm::ParameterSet& cfg);
  /// default destructor
  ~SingleTopTChannelLeptonDQM_miniAOD() {};

  /// do this during the event loop
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup) override;
 
 protected:
  //Book histograms
  void bookHistograms(DQMStore::IBooker &,
    edm::Run const &, edm::EventSetup const &) override;

 private:
  /// deduce object type from ParameterSet label, the label
  /// is expected to be of type 'objectType:selectionStep'
  std::string objectType(const std::string& label) {
    return label.substr(0, label.find(':'));
  };
  /// deduce selection step from ParameterSet label, the
  /// label is expected to be of type 'objectType:selectionStep'
  std::string selectionStep(const std::string& label) {
    return label.substr(label.find(':') + 1);
  };

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


  std::vector<std::string> selectionOrder_;

  std::map<
      std::string,
      std::pair<edm::ParameterSet,
                std::unique_ptr<SingleTopTChannelLepton_miniAOD::MonitorEnsemble> > >
      selection_;

  std::unique_ptr<SelectionStep<pat::Muon> > muonStep_;
  std::unique_ptr<SelectionStep<pat::Electron> > electronStep_;
  std::unique_ptr<SelectionStep<reco::Vertex> > pvStep_;

  std::vector<std::unique_ptr<SelectionStep<pat::Jet> > > jetSteps_;

  std::unique_ptr<SelectionStep<pat::MET> > metStep_;
  std::vector<edm::ParameterSet> sel;
};

#endif
