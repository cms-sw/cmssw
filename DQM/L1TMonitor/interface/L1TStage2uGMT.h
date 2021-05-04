#ifndef DQM_L1TMonitor_L1TStage2uGMT_h
#define DQM_L1TMonitor_L1TStage2uGMT_h

#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTConfiguration.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class L1TStage2uGMT : public DQMEDAnalyzer {
public:
  L1TStage2uGMT(const edm::ParameterSet& ps);
  ~L1TStage2uGMT() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  l1t::tftype getTfOrigin(const int tfMuonIndex);

  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> ugmtBMTFToken;
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> ugmtOMTFToken;
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> ugmtEMTFToken;
  edm::EDGetTokenT<l1t::MuonBxCollection> ugmtMuonToken;
  std::string monitorDir;
  bool emul;
  bool verbose;
  bool displacedQuantities_;

  const float etaScale_;
  const float phiScale_;

  MonitorElement* ugmtBMTFBX;
  MonitorElement* ugmtBMTFnMuons;
  MonitorElement* ugmtBMTFhwPt;
  MonitorElement* ugmtBMTFhwPtUnconstrained;
  MonitorElement* ugmtBMTFhwDXY;
  MonitorElement* ugmtBMTFhwEta;
  MonitorElement* ugmtBMTFhwPhi;
  MonitorElement* ugmtBMTFglbPhi;
  MonitorElement* ugmtBMTFProcvshwPhi;
  MonitorElement* ugmtBMTFhwSign;
  MonitorElement* ugmtBMTFhwSignValid;
  MonitorElement* ugmtBMTFhwQual;
  MonitorElement* ugmtBMTFlink;
  MonitorElement* ugmtBMTFMuMuDEta;
  MonitorElement* ugmtBMTFMuMuDPhi;
  MonitorElement* ugmtBMTFMuMuDR;

  MonitorElement* ugmtOMTFBX;
  MonitorElement* ugmtOMTFnMuons;
  MonitorElement* ugmtOMTFhwPt;
  MonitorElement* ugmtOMTFhwEta;
  MonitorElement* ugmtOMTFhwPhiPos;
  MonitorElement* ugmtOMTFhwPhiNeg;
  MonitorElement* ugmtOMTFglbPhiPos;
  MonitorElement* ugmtOMTFglbPhiNeg;
  MonitorElement* ugmtOMTFProcvshwPhiPos;
  MonitorElement* ugmtOMTFProcvshwPhiNeg;
  MonitorElement* ugmtOMTFhwSign;
  MonitorElement* ugmtOMTFhwSignValid;
  MonitorElement* ugmtOMTFhwQual;
  MonitorElement* ugmtOMTFlink;
  MonitorElement* ugmtOMTFMuMuDEta;
  MonitorElement* ugmtOMTFMuMuDPhi;
  MonitorElement* ugmtOMTFMuMuDR;

  MonitorElement* ugmtEMTFBX;
  MonitorElement* ugmtEMTFnMuons;
  MonitorElement* ugmtEMTFhwPt;
  MonitorElement* ugmtEMTFhwPtUnconstrained;
  MonitorElement* ugmtEMTFhwDXY;
  MonitorElement* ugmtEMTFhwEta;
  MonitorElement* ugmtEMTFhwPhiPos;
  MonitorElement* ugmtEMTFhwPhiNeg;
  MonitorElement* ugmtEMTFglbPhiPos;
  MonitorElement* ugmtEMTFglbPhiNeg;
  MonitorElement* ugmtEMTFProcvshwPhiPos;
  MonitorElement* ugmtEMTFProcvshwPhiNeg;
  MonitorElement* ugmtEMTFhwSign;
  MonitorElement* ugmtEMTFhwSignValid;
  MonitorElement* ugmtEMTFhwQual;
  MonitorElement* ugmtEMTFlink;
  MonitorElement* ugmtEMTFMuMuDEta;
  MonitorElement* ugmtEMTFMuMuDPhi;
  MonitorElement* ugmtEMTFMuMuDR;

  MonitorElement* ugmtBOMTFposMuMuDEta;
  MonitorElement* ugmtBOMTFposMuMuDPhi;
  MonitorElement* ugmtBOMTFposMuMuDR;
  MonitorElement* ugmtBOMTFnegMuMuDEta;
  MonitorElement* ugmtBOMTFnegMuMuDPhi;
  MonitorElement* ugmtBOMTFnegMuMuDR;

  MonitorElement* ugmtEOMTFposMuMuDEta;
  MonitorElement* ugmtEOMTFposMuMuDPhi;
  MonitorElement* ugmtEOMTFposMuMuDR;
  MonitorElement* ugmtEOMTFnegMuMuDEta;
  MonitorElement* ugmtEOMTFnegMuMuDPhi;
  MonitorElement* ugmtEOMTFnegMuMuDR;

  MonitorElement* ugmtBMTFBXvsProcessor;
  MonitorElement* ugmtOMTFBXvsProcessor;
  MonitorElement* ugmtEMTFBXvsProcessor;
  MonitorElement* ugmtBXvsLink;

  MonitorElement* ugmtMuonBX;
  MonitorElement* ugmtnMuons;
  MonitorElement* ugmtMuonIndex;
  MonitorElement* ugmtMuonhwPt;
  MonitorElement* ugmtMuonhwPtUnconstrained;
  MonitorElement* ugmtMuonhwDXY;
  MonitorElement* ugmtMuonhwEta;
  MonitorElement* ugmtMuonhwPhi;
  MonitorElement* ugmtMuonhwEtaAtVtx;
  MonitorElement* ugmtMuonhwPhiAtVtx;
  MonitorElement* ugmtMuonhwCharge;
  MonitorElement* ugmtMuonhwChargeValid;
  MonitorElement* ugmtMuonhwQual;
  MonitorElement* ugmtMuonhwIso;

  MonitorElement* ugmtMuonPt;
  MonitorElement* ugmtMuonPtUnconstrained;
  MonitorElement* ugmtMuonEta;
  MonitorElement* ugmtMuonPhi;
  MonitorElement* ugmtMuonEtaAtVtx;
  MonitorElement* ugmtMuonPhiAtVtx;
  MonitorElement* ugmtMuonCharge;

  MonitorElement* ugmtMuonPhiBmtf;
  MonitorElement* ugmtMuonPhiOmtf;
  MonitorElement* ugmtMuonPhiEmtf;
  MonitorElement* ugmtMuonDEtavsPtBmtf;
  MonitorElement* ugmtMuonDPhivsPtBmtf;
  MonitorElement* ugmtMuonDEtavsPtOmtf;
  MonitorElement* ugmtMuonDPhivsPtOmtf;
  MonitorElement* ugmtMuonDEtavsPtEmtf;
  MonitorElement* ugmtMuonDPhivsPtEmtf;

  MonitorElement* ugmtMuonPtvsEta;
  MonitorElement* ugmtMuonPtvsPhi;
  MonitorElement* ugmtMuonPhivsEta;
  MonitorElement* ugmtMuonPhiAtVtxvsEtaAtVtx;

  MonitorElement* ugmtMuonBXvsLink;
  MonitorElement* ugmtMuonBXvshwPt;
  MonitorElement* ugmtMuonBXvshwEta;
  MonitorElement* ugmtMuonBXvshwPhi;
  MonitorElement* ugmtMuonBXvshwCharge;
  MonitorElement* ugmtMuonBXvshwChargeValid;
  MonitorElement* ugmtMuonBXvshwQual;
  MonitorElement* ugmtMuonBXvshwIso;
  MonitorElement* ugmtMuonChargevsLink;

  // muon correlations
  MonitorElement* ugmtMuMuInvMass;
  MonitorElement* ugmtMuMuInvMassAtVtx;

  MonitorElement* ugmtMuMuDEta;
  MonitorElement* ugmtMuMuDPhi;
  MonitorElement* ugmtMuMuDR;

  MonitorElement* ugmtMuMuDEtaBOpos;
  MonitorElement* ugmtMuMuDPhiBOpos;
  MonitorElement* ugmtMuMuDRBOpos;
  MonitorElement* ugmtMuMuDEtaBOneg;
  MonitorElement* ugmtMuMuDPhiBOneg;
  MonitorElement* ugmtMuMuDRBOneg;

  MonitorElement* ugmtMuMuDEtaEOpos;
  MonitorElement* ugmtMuMuDPhiEOpos;
  MonitorElement* ugmtMuMuDREOpos;
  MonitorElement* ugmtMuMuDEtaEOneg;
  MonitorElement* ugmtMuMuDPhiEOneg;
  MonitorElement* ugmtMuMuDREOneg;

  MonitorElement* ugmtMuMuDEtaB;
  MonitorElement* ugmtMuMuDPhiB;
  MonitorElement* ugmtMuMuDRB;

  MonitorElement* ugmtMuMuDEtaOpos;
  MonitorElement* ugmtMuMuDPhiOpos;
  MonitorElement* ugmtMuMuDROpos;
  MonitorElement* ugmtMuMuDEtaOneg;
  MonitorElement* ugmtMuMuDPhiOneg;
  MonitorElement* ugmtMuMuDROneg;

  MonitorElement* ugmtMuMuDEtaEpos;
  MonitorElement* ugmtMuMuDPhiEpos;
  MonitorElement* ugmtMuMuDREpos;
  MonitorElement* ugmtMuMuDEtaEneg;
  MonitorElement* ugmtMuMuDPhiEneg;
  MonitorElement* ugmtMuMuDREneg;
};

#endif
