#ifndef DQM_L1TMonitor_L1TStage2uGMT_h
#define DQM_L1TMonitor_L1TStage2uGMT_h


#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTConfiguration.h"

#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DQMServices/Core/interface/ConcurrentMonitorElement.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace ugmtdqm {
  struct Histograms {
    ConcurrentMonitorElement ugmtBMTFBX;
    ConcurrentMonitorElement ugmtBMTFnMuons;
    ConcurrentMonitorElement ugmtBMTFhwPt;
    ConcurrentMonitorElement ugmtBMTFhwEta;
    ConcurrentMonitorElement ugmtBMTFhwPhi;
    ConcurrentMonitorElement ugmtBMTFglbPhi;
    ConcurrentMonitorElement ugmtBMTFProcvshwPhi;
    ConcurrentMonitorElement ugmtBMTFhwSign;
    ConcurrentMonitorElement ugmtBMTFhwSignValid;
    ConcurrentMonitorElement ugmtBMTFhwQual;
    ConcurrentMonitorElement ugmtBMTFlink;
    ConcurrentMonitorElement ugmtBMTFMuMuDEta;
    ConcurrentMonitorElement ugmtBMTFMuMuDPhi;
    ConcurrentMonitorElement ugmtBMTFMuMuDR;

    ConcurrentMonitorElement ugmtOMTFBX;
    ConcurrentMonitorElement ugmtOMTFnMuons;
    ConcurrentMonitorElement ugmtOMTFhwPt;
    ConcurrentMonitorElement ugmtOMTFhwEta;
    ConcurrentMonitorElement ugmtOMTFhwPhiPos;
    ConcurrentMonitorElement ugmtOMTFhwPhiNeg;
    ConcurrentMonitorElement ugmtOMTFglbPhiPos;
    ConcurrentMonitorElement ugmtOMTFglbPhiNeg;
    ConcurrentMonitorElement ugmtOMTFProcvshwPhiPos;
    ConcurrentMonitorElement ugmtOMTFProcvshwPhiNeg;
    ConcurrentMonitorElement ugmtOMTFhwSign;
    ConcurrentMonitorElement ugmtOMTFhwSignValid;
    ConcurrentMonitorElement ugmtOMTFhwQual;
    ConcurrentMonitorElement ugmtOMTFlink;
    ConcurrentMonitorElement ugmtOMTFMuMuDEta;
    ConcurrentMonitorElement ugmtOMTFMuMuDPhi;
    ConcurrentMonitorElement ugmtOMTFMuMuDR;

    ConcurrentMonitorElement ugmtEMTFBX;
    ConcurrentMonitorElement ugmtEMTFnMuons;
    ConcurrentMonitorElement ugmtEMTFhwPt;
    ConcurrentMonitorElement ugmtEMTFhwEta;
    ConcurrentMonitorElement ugmtEMTFhwPhiPos;
    ConcurrentMonitorElement ugmtEMTFhwPhiNeg;
    ConcurrentMonitorElement ugmtEMTFglbPhiPos;
    ConcurrentMonitorElement ugmtEMTFglbPhiNeg;
    ConcurrentMonitorElement ugmtEMTFProcvshwPhiPos;
    ConcurrentMonitorElement ugmtEMTFProcvshwPhiNeg;
    ConcurrentMonitorElement ugmtEMTFhwSign;
    ConcurrentMonitorElement ugmtEMTFhwSignValid;
    ConcurrentMonitorElement ugmtEMTFhwQual;
    ConcurrentMonitorElement ugmtEMTFlink;
    ConcurrentMonitorElement ugmtEMTFMuMuDEta;
    ConcurrentMonitorElement ugmtEMTFMuMuDPhi;
    ConcurrentMonitorElement ugmtEMTFMuMuDR;

    ConcurrentMonitorElement ugmtBOMTFposMuMuDEta;
    ConcurrentMonitorElement ugmtBOMTFposMuMuDPhi;
    ConcurrentMonitorElement ugmtBOMTFposMuMuDR;
    ConcurrentMonitorElement ugmtBOMTFnegMuMuDEta;
    ConcurrentMonitorElement ugmtBOMTFnegMuMuDPhi;
    ConcurrentMonitorElement ugmtBOMTFnegMuMuDR;

    ConcurrentMonitorElement ugmtEOMTFposMuMuDEta;
    ConcurrentMonitorElement ugmtEOMTFposMuMuDPhi;
    ConcurrentMonitorElement ugmtEOMTFposMuMuDR;
    ConcurrentMonitorElement ugmtEOMTFnegMuMuDEta;
    ConcurrentMonitorElement ugmtEOMTFnegMuMuDPhi;
    ConcurrentMonitorElement ugmtEOMTFnegMuMuDR;

    ConcurrentMonitorElement ugmtBMTFBXvsProcessor;
    ConcurrentMonitorElement ugmtOMTFBXvsProcessor;
    ConcurrentMonitorElement ugmtEMTFBXvsProcessor;
    ConcurrentMonitorElement ugmtBXvsLink;

    ConcurrentMonitorElement ugmtMuonBX;
    ConcurrentMonitorElement ugmtnMuons;
    ConcurrentMonitorElement ugmtMuonIndex;
    ConcurrentMonitorElement ugmtMuonhwPt;
    ConcurrentMonitorElement ugmtMuonhwEta;
    ConcurrentMonitorElement ugmtMuonhwPhi;
    ConcurrentMonitorElement ugmtMuonhwEtaAtVtx;
    ConcurrentMonitorElement ugmtMuonhwPhiAtVtx;
    ConcurrentMonitorElement ugmtMuonhwCharge;
    ConcurrentMonitorElement ugmtMuonhwChargeValid;
    ConcurrentMonitorElement ugmtMuonhwQual;
    ConcurrentMonitorElement ugmtMuonhwIso;

    ConcurrentMonitorElement ugmtMuonPt;
    ConcurrentMonitorElement ugmtMuonEta;
    ConcurrentMonitorElement ugmtMuonPhi;
    ConcurrentMonitorElement ugmtMuonEtaAtVtx;
    ConcurrentMonitorElement ugmtMuonPhiAtVtx;
    ConcurrentMonitorElement ugmtMuonCharge;

    ConcurrentMonitorElement ugmtMuonPhiBmtf;
    ConcurrentMonitorElement ugmtMuonPhiOmtf;
    ConcurrentMonitorElement ugmtMuonPhiEmtf;
    ConcurrentMonitorElement ugmtMuonDEtavsPtBmtf;
    ConcurrentMonitorElement ugmtMuonDPhivsPtBmtf;
    ConcurrentMonitorElement ugmtMuonDEtavsPtOmtf;
    ConcurrentMonitorElement ugmtMuonDPhivsPtOmtf;
    ConcurrentMonitorElement ugmtMuonDEtavsPtEmtf;
    ConcurrentMonitorElement ugmtMuonDPhivsPtEmtf;

    ConcurrentMonitorElement ugmtMuonPtvsEta;
    ConcurrentMonitorElement ugmtMuonPtvsPhi;
    ConcurrentMonitorElement ugmtMuonPhivsEta;
    ConcurrentMonitorElement ugmtMuonPhiAtVtxvsEtaAtVtx;

    ConcurrentMonitorElement ugmtMuonBXvsLink;
    ConcurrentMonitorElement ugmtMuonBXvshwPt;
    ConcurrentMonitorElement ugmtMuonBXvshwEta;
    ConcurrentMonitorElement ugmtMuonBXvshwPhi;
    ConcurrentMonitorElement ugmtMuonBXvshwCharge;
    ConcurrentMonitorElement ugmtMuonBXvshwChargeValid;
    ConcurrentMonitorElement ugmtMuonBXvshwQual;
    ConcurrentMonitorElement ugmtMuonBXvshwIso;

    // muon correlations
    ConcurrentMonitorElement ugmtMuMuInvMass;
    ConcurrentMonitorElement ugmtMuMuInvMassAtVtx;

    ConcurrentMonitorElement ugmtMuMuDEta;
    ConcurrentMonitorElement ugmtMuMuDPhi;
    ConcurrentMonitorElement ugmtMuMuDR;

    ConcurrentMonitorElement ugmtMuMuDEtaBOpos;
    ConcurrentMonitorElement ugmtMuMuDPhiBOpos;
    ConcurrentMonitorElement ugmtMuMuDRBOpos;
    ConcurrentMonitorElement ugmtMuMuDEtaBOneg;
    ConcurrentMonitorElement ugmtMuMuDPhiBOneg;
    ConcurrentMonitorElement ugmtMuMuDRBOneg;

    ConcurrentMonitorElement ugmtMuMuDEtaEOpos;
    ConcurrentMonitorElement ugmtMuMuDPhiEOpos;
    ConcurrentMonitorElement ugmtMuMuDREOpos;
    ConcurrentMonitorElement ugmtMuMuDEtaEOneg;
    ConcurrentMonitorElement ugmtMuMuDPhiEOneg;
    ConcurrentMonitorElement ugmtMuMuDREOneg;

    ConcurrentMonitorElement ugmtMuMuDEtaB;
    ConcurrentMonitorElement ugmtMuMuDPhiB;
    ConcurrentMonitorElement ugmtMuMuDRB;

    ConcurrentMonitorElement ugmtMuMuDEtaOpos;
    ConcurrentMonitorElement ugmtMuMuDPhiOpos;
    ConcurrentMonitorElement ugmtMuMuDROpos;
    ConcurrentMonitorElement ugmtMuMuDEtaOneg;
    ConcurrentMonitorElement ugmtMuMuDPhiOneg;
    ConcurrentMonitorElement ugmtMuMuDROneg;

    ConcurrentMonitorElement ugmtMuMuDEtaEpos;
    ConcurrentMonitorElement ugmtMuMuDPhiEpos;
    ConcurrentMonitorElement ugmtMuMuDREpos;
    ConcurrentMonitorElement ugmtMuMuDEtaEneg;
    ConcurrentMonitorElement ugmtMuMuDPhiEneg;
    ConcurrentMonitorElement ugmtMuMuDREneg;
  };
}

class L1TStage2uGMT : public DQMGlobalEDAnalyzer<ugmtdqm::Histograms> {

 public:

  L1TStage2uGMT(const edm::ParameterSet& ps);
  ~L1TStage2uGMT() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 protected:

  void dqmBeginRun(const edm::Run&, const edm::EventSetup&, ugmtdqm::Histograms &) const override;
  void bookHistograms(DQMStore::ConcurrentBooker&, const edm::Run&, const edm::EventSetup&, ugmtdqm::Histograms &) const override;
  void dqmAnalyze(const edm::Event&, const edm::EventSetup&, ugmtdqm::Histograms const&) const override;

 private:  

  l1t::tftype getTfOrigin(int tfMuonIndex) const;

  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> ugmtBMTFToken;
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> ugmtOMTFToken;
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> ugmtEMTFToken;
  edm::EDGetTokenT<l1t::MuonBxCollection> ugmtMuonToken;
  std::string monitorDir;
  bool emul;
  bool verbose;

  const float etaScale_;
  const float phiScale_;
};

#endif
