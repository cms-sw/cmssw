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
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h" 
#include "DataFormats/MuonReco/interface/MuonEnergy.h"

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

    edm::InputTag muons_;
    double pT_cut_;
    double eta_cut_;
    std::string moduleName_;

    MonitorElement * Nmuons_;
    MonitorElement * pT_muons_;
    MonitorElement * eta_muons_;
    MonitorElement * phi_muons_;

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
