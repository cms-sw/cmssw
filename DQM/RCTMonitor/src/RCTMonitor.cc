#include "DQM/RCTMonitor/interface/RCTMonitor.h"
#include "DQM/RCTMonitor/interface/somedefinitions.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <iostream>

RCTMonitor::RCTMonitor(const edm::ParameterSet& iConfig) {
  // set Token(-s)
  m_rctSourceToken_ = consumes<L1CaloEmCollection>(
      iConfig.getUntrackedParameter<edm::InputTag>("rctSource"));
}

RCTMonitor::~RCTMonitor() {}

void RCTMonitor::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const&,
                                  edm::EventSetup const&) {
  // Book RCT histograms
  iBooker.setCurrentFolder("RCT");

  m_rctIsoEmRankEtaPhi1 =
      iBooker.book2D("RctIsoEmRankEtaPhi", "ISO EM RANK", PHIBINS, PHIMIN,
                     PHIMAX, ETABINS, ETAMIN, ETAMAX);
  m_rctIsoEmOccEtaPhi1 =
      iBooker.book2D("RctIsoEmOccEtaPhi", "ISO EM OCCUPANCY", PHIBINS, PHIMIN,
                     PHIMAX, ETABINS, ETAMIN, ETAMAX);
  m_rctIsoEmRank1 =
      iBooker.book1D("RctIsoEmRank", "ISO EM RANK", R6BINS, R6MIN, R6MAX);
  m_rctIsoEmRankEtaPhi10 =
      iBooker.book2D("RctIsoEmRankEtaPhi10", "ISO EM RANK", PHIBINS, PHIMIN,
                     PHIMAX, ETABINS, ETAMIN, ETAMAX);
  m_rctIsoEmOccEtaPhi10 =
      iBooker.book2D("RctIsoEmOccEtaPhi10", "ISO EM OCCUPANCY", PHIBINS, PHIMIN,
                     PHIMAX, ETABINS, ETAMIN, ETAMAX);
  m_rctIsoEmRank10 =
      iBooker.book1D("RctIsoEmRank10", "ISO EM RANK", R6BINS, R6MIN, R6MAX);

  m_rctNonIsoEmRankEtaPhi1 =
      iBooker.book2D("RctNonIsoEmRankEtaPhi", "NON-ISO EM RANK", PHIBINS,
                     PHIMIN, PHIMAX, ETABINS, ETAMIN, ETAMAX);
  m_rctNonIsoEmOccEtaPhi1 =
      iBooker.book2D("RctNonIsoEmOccEtaPhi", "NON-ISO EM OCCUPANCY", PHIBINS,
                     PHIMIN, PHIMAX, ETABINS, ETAMIN, ETAMAX);
  m_rctNonIsoEmRank1 = iBooker.book1D("RctNonIsoEmRank", "NON-ISO EM RANK",
                                      R6BINS, R6MIN, R6MAX);
  m_rctNonIsoEmRankEtaPhi10 =
      iBooker.book2D("RctNonIsoEmRankEtaPhi10", "NON-ISO EM RANK", PHIBINS,
                     PHIMIN, PHIMAX, ETABINS, ETAMIN, ETAMAX);
  m_rctNonIsoEmOccEtaPhi10 =
      iBooker.book2D("RctNonIsoEmOccEtaPhi10", "NON-ISO EM OCCUPANCY", PHIBINS,
                     PHIMIN, PHIMAX, ETABINS, ETAMIN, ETAMAX);
  m_rctNonIsoEmRank10 = iBooker.book1D("RctNonIsoEmRank10", "NON-ISO EM RANK",
                                       R6BINS, R6MIN, R6MAX);

  m_rctRelaxedEmRankEtaPhi1 =
      iBooker.book2D("RctRelaxedEmRankEtaPhi", "RELAXED EM RANK", PHIBINS,
                     PHIMIN, PHIMAX, ETABINS, ETAMIN, ETAMAX);
  m_rctRelaxedEmOccEtaPhi1 =
      iBooker.book2D("RctRelaxedEmOccEtaPhi", "RELAXED EM OCCUPANCY", PHIBINS,
                     PHIMIN, PHIMAX, ETABINS, ETAMIN, ETAMAX);
  m_rctRelaxedEmRank1 = iBooker.book1D("RctRelaxedEmRank", "RELAXED EM RANK",
                                       R6BINS, R6MIN, R6MAX);
  m_rctRelaxedEmRankEtaPhi10 =
      iBooker.book2D("RctRelaxedEmRankEtaPhi", "RELAXED EM RANK", PHIBINS,
                     PHIMIN, PHIMAX, ETABINS, ETAMIN, ETAMAX);
  m_rctRelaxedEmOccEtaPhi10 =
      iBooker.book2D("RctRelaxedEmOccEtaPhi10", "RELAXED EM OCCUPANCY", PHIBINS,
                     PHIMIN, PHIMAX, ETABINS, ETAMIN, ETAMAX);
  m_rctRelaxedEmRank10 = iBooker.book1D("RctRelaxedEmRank", "RELAXED EM RANK",
                                        R6BINS, R6MIN, R6MAX);

  m_rctRegionsEtEtaPhi =
      iBooker.book2D("RctRegionsEtEtaPhi", "REGION E_{T}", PHIBINS, PHIMIN,
                     PHIMAX, ETABINS, ETAMIN, ETAMAX);
  m_rctRegionsOccEtaPhi =
      iBooker.book2D("RctRegionsOccEtaPhi", "REGION OCCUPANCY", PHIBINS, PHIMIN,
                     PHIMAX, ETABINS, ETAMIN, ETAMAX);
  m_rctTauVetoEtaPhi =
      iBooker.book2D("RctTauVetoEtaPhi", "TAU VETO OCCUPANCY", PHIBINS, PHIMIN,
                     PHIMAX, ETABINS, ETAMIN, ETAMAX);
  m_rctRegionEt =
      iBooker.book1D("RctRegionEt", "REGION E_{T}", R10BINS, R10MIN, R10MAX);
}

void RCTMonitor::analyze(const edm::Event& iEvent,
                         const edm::EventSetup& iSetup) {
  // Fill histograms
  FillRCT(iEvent, iSetup);
}

float DynamicScale(int EtaStamp) {
  // This function weights bin elements according to spatial extent of
  // calorimeter tower.
  if (EtaStamp >= 6 && EtaStamp <= 15) {
    return ScaleINNER;
  } else if (EtaStamp == 5 || EtaStamp == 16) {
    return ScaleIN;
  } else if (EtaStamp == 4 || EtaStamp == 17) {
    return ScaleOUT;
  } else {
    return 0.000000;
  }
}

void RCTMonitor::FillRCT(const edm::Event& iEvent,
                         const edm::EventSetup& iSetup) {
  // Get the RCT digis
  edm::Handle<L1CaloEmCollection> em;

  iEvent.getByToken(m_rctSourceToken_, em);

  // Isolated and non-isolated EM with cut at >1 GeV
  for (L1CaloEmCollection::const_iterator iem = em->begin(); iem != em->end();
       iem++) {
    if (iem->rank() > 1.) {   // applies the 1 GeV cut
      if (iem->isolated()) {  // looks for isolated EM candidates only
        m_rctIsoEmRank1->Fill(iem->rank());
        // std::cout << "Just to show what is there " << iem->rank() <<
        // std::endl ;
        m_rctIsoEmRankEtaPhi1->Fill(iem->regionId().iphi(),
                                    iem->regionId().ieta(), iem->rank());
        m_rctIsoEmOccEtaPhi1->Fill(iem->regionId().iphi(),
                                   iem->regionId().ieta(),
                                   DynamicScale(iem->regionId().ieta()));
        m_rctRelaxedEmRankEtaPhi1->Fill(iem->regionId().iphi(),
                                        iem->regionId().ieta(), iem->rank());
        m_rctRelaxedEmOccEtaPhi1->Fill(iem->regionId().iphi(),
                                       iem->regionId().ieta(),
                                       DynamicScale(iem->regionId().ieta()));
        m_rctRelaxedEmRank1->Fill(iem->rank());
      } else {  // instructions for Non-isolated EM candidates
        m_rctNonIsoEmRank1->Fill(iem->rank());
        m_rctNonIsoEmRankEtaPhi1->Fill(iem->regionId().iphi(),
                                       iem->regionId().ieta(), iem->rank());
        m_rctNonIsoEmOccEtaPhi1->Fill(iem->regionId().iphi(),
                                      iem->regionId().ieta(),
                                      DynamicScale(iem->regionId().ieta()));
        m_rctRelaxedEmRankEtaPhi1->Fill(iem->regionId().iphi(),
                                        iem->regionId().ieta(), iem->rank());
        m_rctRelaxedEmOccEtaPhi1->Fill(iem->regionId().iphi(),
                                       iem->regionId().ieta(),
                                       DynamicScale(iem->regionId().ieta()));
        m_rctRelaxedEmRank1->Fill(iem->rank());
      }
    }
    if (iem->rank() > 10.) {  // applies the 10 GeV cut
      if (iem->isolated()) {  // looks for isolated EM candidates only
        m_rctIsoEmOccEtaPhi10->Fill(iem->regionId().iphi(),
                                    iem->regionId().ieta(),
                                    DynamicScale(iem->regionId().ieta()));
        m_rctRelaxedEmOccEtaPhi10->Fill(iem->regionId().iphi(),
                                        iem->regionId().ieta(),
                                        DynamicScale(iem->regionId().ieta()));
      } else {  // instructions for Non-isolated EM candidates
        m_rctNonIsoEmOccEtaPhi10->Fill(iem->regionId().iphi(),
                                       iem->regionId().ieta(),
                                       DynamicScale(iem->regionId().ieta()));
        m_rctRelaxedEmOccEtaPhi10->Fill(iem->regionId().iphi(),
                                        iem->regionId().ieta(),
                                        DynamicScale(iem->regionId().ieta()));
      }
    }
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(RCTMonitor);
