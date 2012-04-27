#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"
#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCPIsolation.h"
#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCPCaloInfo.h"
#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCPDeDxInfo.h"
#include "AnalysisDataFormats/SUSYBSMObjects/interface/MuonSegment.h"

namespace {
 namespace {
  susybsm::HSCParticle pa;
/*  susybsm::DriftTubeTOF dtitof;

  susybsm::TimeMeasurement tm;
  std::vector<susybsm::TimeMeasurement> tmv;
  susybsm::MuonTOFCollection mtc; 
  susybsm::MuonTOF mt;
  susybsm::MuonTOFRef mtr;
  susybsm::MuonTOFRefProd mtrp;
  susybsm::MuonTOFRefVector mtrv;
  edm::Wrapper<susybsm::MuonTOFCollection> wr;
  std::vector<std::pair<edm::Ref<std::vector<reco::Muon>,reco::Muon,edm::refhelper::FindUsingAdvance<std::vector<reco::Muon>,reco::Muon> >,susybsm::DriftTubeTOF> > a;
  std::vector<susybsm::DriftTubeTOF> b;
  */
//  susybsm::CaloBetaMeasurement calobeta;
  susybsm::RPCBetaMeasurement rpcbeta;
//susybsm::DeDxBeta dedxbeta;
  susybsm::HSCParticleCollection hc;
  susybsm::HSCParticleRef hr;
  susybsm::HSCParticleRefProd hp;
  susybsm::HSCParticleRefVector hv;
  edm::Wrapper<susybsm::HSCParticleCollection> wr1;

  susybsm::MuonSegment ms;
  susybsm::MuonSegmentCollection msc;
  susybsm::MuonSegmentRef msr;
  susybsm::MuonSegmentRefProd msp;
  susybsm::MuonSegmentRefVector msv;
  edm::Wrapper<susybsm::MuonSegmentCollection> mswr1;
  
  susybsm::TracksEcalRecHitsMap terhm;
  edm::Wrapper<susybsm::TracksEcalRecHitsMap> wr2;
  edm::helpers::KeyVal<edm::RefProd<std::vector<reco::Track> >,edm::RefProd<edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> > > > hlpr1;

  susybsm::HSCPIsolation hscpI;
  susybsm::HSCPIsolationCollection hscpIc;
  susybsm::HSCPIsolationValueMap hscpIvm; 
  edm::Wrapper<susybsm::HSCPIsolation> hscpIW; 
  edm::Wrapper<susybsm::HSCPIsolationCollection> hscpIcW;
  edm::Wrapper<susybsm::HSCPIsolationValueMap> hscpIvmW;

  susybsm::HSCPCaloInfo hscpC;
  susybsm::HSCPCaloInfoCollection hscpCc;
  susybsm::HSCPCaloInfoRef hscpCr;
  susybsm::HSCPCaloInfoRefProd hscpCp;
  susybsm::HSCPCaloInfoRefVector hscpCv;
  susybsm::HSCPCaloInfoValueMap hscpCvm;
  edm::Wrapper<susybsm::HSCPCaloInfo> hscpCW;
  edm::Wrapper<susybsm::HSCPCaloInfoCollection> hscpCcW;
  edm::Wrapper<susybsm::HSCPCaloInfoValueMap> hscpCvmW;

  susybsm::HSCPDeDxInfo hscpDEDX;
  susybsm::HSCPDeDxInfoCollection hscpDEDXc;
  susybsm::HSCPDeDxInfoRef hscpDEDXr;
  susybsm::HSCPDeDxInfoRefProd hscpDEDXp;
  susybsm::HSCPDeDxInfoRefVector hscpDEDXv;
  susybsm::HSCPDeDxInfoValueMap hscpDEDXvm;
  edm::Wrapper<susybsm::HSCPDeDxInfo> hscpDEDXW;
  edm::Wrapper<susybsm::HSCPDeDxInfoCollection> hscpDEDXcW;
  edm::Wrapper<susybsm::HSCPDeDxInfoValueMap> hscpDEDXvmW;

  
 }
}
