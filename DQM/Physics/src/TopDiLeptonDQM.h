#ifndef TopDiLeptonDQM_H
#define TopDiLeptonDQM_H

#include <string>
#include <vector>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h" 
#include "DataFormats/MuonReco/interface/MuonEnergy.h"
#include "DataFormats/MuonReco/interface/MuonIsolation.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "HLTrigger/HLTfilters/interface/HLTHighLevel.h"

class TH1F;
class TH2F;
class TopDiLeptonDQM : public edm::EDAnalyzer {

  public:

    explicit TopDiLeptonDQM(const edm::ParameterSet&);
    ~TopDiLeptonDQM();

  protected:

    void beginRun(const edm::Run&, const edm::EventSetup&);
    void endRun(const edm::Run&, const edm::EventSetup&);

  private:

    void initialize();
    virtual void beginJob(const edm::EventSetup&);
    virtual void analyze(const edm::Event&, const edm::EventSetup&);
    virtual void endJob();
       
    edm::ParameterSet parameters_;
    DQMStore * dbe_;

    std::string moduleName_;
    edm::InputTag triggerResults_;
    std::vector<std::string> hltPaths_L3_;
    std::vector<std::string> hltPaths_L3_mu_;
    std::vector<std::string> hltPaths_L3_el_;

    edm::InputTag muons_;
    double muon_pT_cut_;
    double muon_eta_cut_;

    edm::InputTag elecs_;
    double elec_pT_cut_;
    double elec_eta_cut_;

    MonitorElement * Trigs_;

    MonitorElement * Muon_Trigs_;
    MonitorElement * Nmuons_;
    MonitorElement * pT_muons_;
    MonitorElement * eta_muons_;
    MonitorElement * phi_muons_;

    MonitorElement * Elec_Trigs_;
    MonitorElement * Nelecs_;
    MonitorElement * pT_elecs_;
    MonitorElement * eta_elecs_;
    MonitorElement * phi_elecs_;

    MonitorElement * MuIso_emEt03_;
    MonitorElement * MuIso_hadEt03_;
    MonitorElement * MuIso_hoEt03_;
    MonitorElement * MuIso_nJets03_;
    MonitorElement * MuIso_nTracks03_;
    MonitorElement * MuIso_sumPt03_;

    MonitorElement * ElecIso_cal_;
    MonitorElement * ElecIso_trk_;

    MonitorElement * dimassRC_LOG_;
    MonitorElement * dimassWC_LOG_;
    MonitorElement * dimassRC_;
    MonitorElement * dimassWC_;
    MonitorElement * D_eta_muons_;
    MonitorElement * D_phi_muons_;

    MonitorElement * isoDimassCorrelation_;

    MonitorElement * absCount_;
    MonitorElement * relCount_;
    MonitorElement * combCount_;
    MonitorElement * diCombCount_;
};

#endif
