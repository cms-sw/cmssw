/*
 * \L1TStage2OMTF.cc
 * \author Esmaeel Eskandari Tadavani
 * \November 2015
*/

#include "DQM/L1TMonitor/interface/L1TStage2OMTF.h"

L1TStage2OMTF::L1TStage2OMTF(const edm::ParameterSet& ps)
    : monitorDir(ps.getUntrackedParameter<std::string>("monitorDir", "")),
      omtfSource(ps.getParameter<edm::InputTag>("omtfSource")),
      verbose(ps.getUntrackedParameter<bool>("verbose", false)),
      global_phi(-1000) {
  omtfToken = consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("omtfSource"));
}

L1TStage2OMTF::~L1TStage2OMTF() {}

void L1TStage2OMTF::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run& iRun, const edm::EventSetup& eveSetup) {
  ibooker.setCurrentFolder(monitorDir);

  omtf_hwEta = ibooker.book1D("omtf_hwEta", "HW #eta", 447, -223.5, 223.5);
  omtf_hwLocalPhi = ibooker.book1D("omtf_hwLocalPhi", "HW Local #phi", 201, -100.5, 100.5);
  omtf_hwPt = ibooker.book1D("omtf_hwPt", "HW p_{T}", 511, -0.5, 510.5);
  omtf_hwQual = ibooker.book1D("omtf_hwQual", "HW Quality", 20, -0.5, 19.5);
  omtf_bx = ibooker.book1D("omtf_bx", "BX", 5, -2.5, 2.5);

  omtf_hwEta_hwLocalPhi =
      ibooker.book2D("omtf_hwEta_hwLocalPhi", "HW #eta vs HW Local #phi", 447, -223.5, 223.5, 201, -100.5, 100.5);
  omtf_hwEta_hwLocalPhi->setTitle(";HW #eta; HW Local #phi");

  omtf_hwPt_hwEta = ibooker.book2D("omtf_hwPt_hwEta", "HW p_{T} vs HW #eta", 511, -0.5, 510.5, 447, -223.5, 223.5);
  omtf_hwPt_hwEta->setTitle(";HW p_{T}; HW #eta");

  omtf_hwPt_hwLocalPhi =
      ibooker.book2D("omtf_hwPt_hwLocalPhi", "HW p_{T} vs HW Local #phi", 511, -0.5, 510.5, 201, -100.5, 100.5);
  omtf_hwPt_hwLocalPhi->setTitle(";HW p_{T}; HW Local #phi");

  omtf_hwEta_bx = ibooker.book2D("omtf_hwEta_bx", "HW #eta vs BX", 447, -223.5, 223.5, 5, -2.5, 2.5);
  omtf_hwEta_bx->setTitle(";HW #eta; BX");

  omtf_hwLocalPhi_bx = ibooker.book2D("omtf_hwLocalPhi_bx", "HW Local #phi vs BX", 201, -100.5, 100.5, 5, -2.5, 2.5);
  omtf_hwLocalPhi_bx->setTitle(";HW Local #phi; BX");

  omtf_hwPt_bx = ibooker.book2D("omtf_hwPt_bx", "HW p_{T} vs BX", 511, -0.5, 510.5, 5, -2.5, 2.5);
  omtf_hwPt_bx->setTitle(";HW p_{T}; BX");

  omtf_hwQual_bx = ibooker.book2D("omtf_hwQual_bx", "HW Quality vs BX", 20, -0.5, 19.5, 5, -2.5, 2.5);
  omtf_hwQual_bx->setTitle("; HW Quality; BX");
}

void L1TStage2OMTF::analyze(const edm::Event& eve, const edm::EventSetup& eveSetup) {
  if (verbose) {
    edm::LogInfo("L1TStage2OMTF") << "L1TStage2OMTF: analyze...." << std::endl;
  }

  edm::Handle<l1t::RegionalMuonCandBxCollection> omtfMuon;
  eve.getByToken(omtfToken, omtfMuon);

  for (int itBX = omtfMuon->getFirstBX(); itBX <= omtfMuon->getLastBX(); ++itBX) {
    for (l1t::RegionalMuonCandBxCollection::const_iterator itMuon = omtfMuon->begin(itBX);
         itMuon != omtfMuon->end(itBX);
         ++itMuon) {
      omtf_hwEta->Fill(itMuon->hwEta());
      omtf_hwLocalPhi->Fill(itMuon->hwPhi());
      omtf_hwPt->Fill(itMuon->hwPt());
      omtf_hwQual->Fill(itMuon->hwQual());

      omtf_bx->Fill(itBX);
      omtf_hwEta_bx->Fill(itMuon->hwEta(), itBX);
      omtf_hwLocalPhi_bx->Fill(itMuon->hwPhi(), itBX);
      omtf_hwPt_bx->Fill(itMuon->hwPt(), itBX);
      omtf_hwQual_bx->Fill(itMuon->hwQual(), itBX);

      omtf_hwEta_hwLocalPhi->Fill(itMuon->hwEta(), itMuon->hwPhi());
      omtf_hwPt_hwEta->Fill(itMuon->hwPt(), itMuon->hwEta());
      omtf_hwPt_hwLocalPhi->Fill(itMuon->hwPt(), itMuon->hwPhi());
    }
  }
}
