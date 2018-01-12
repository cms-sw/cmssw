/*
 * \L1TStage2OMTF.cc
 * \author Esmaeel Eskandari Tadavani
 * \November 2015
*/

#include "DQM/L1TMonitor/interface/L1TStage2OMTF.h"

L1TStage2OMTF::L1TStage2OMTF(const edm::ParameterSet & ps):
  monitorDir(ps.getUntrackedParameter<std::string>("monitorDir","")),
  omtfSource(ps.getParameter<edm::InputTag>("omtfSource")),
  verbose(ps.getUntrackedParameter<bool>("verbose", false)),
  global_phi(-1000)
{
  omtfToken=consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("omtfSource"));
}

L1TStage2OMTF::~L1TStage2OMTF()
{
}

void L1TStage2OMTF::dqmBeginRun(const edm::Run& iRrun, const edm::EventSetup& eveSetup, omtfdqm::Histograms& histograms) const
{
}

void L1TStage2OMTF::bookHistograms(DQMStore::ConcurrentBooker &booker, const edm::Run& iRun, const edm::EventSetup& eveSetup, omtfdqm::Histograms& histograms) const
{
  booker.setCurrentFolder(monitorDir);
  
  histograms.omtf_hwEta = booker.book1D("omtf_hwEta", "HW #eta", 447, -223.5, 223.5);
  histograms.omtf_hwLocalPhi = booker.book1D("omtf_hwLocalPhi", "HW Local #phi", 201, -100.5, 100.5);
  histograms.omtf_hwPt = booker.book1D("omtf_hwPt", "HW p_{T}", 511, -0.5, 510.5);
  histograms.omtf_hwQual= booker.book1D("omtf_hwQual", "HW Quality", 16, -0.5, 15.5);
  histograms.omtf_bx = booker.book1D("omtf_bx","BX", 5, -2.5, 2.5);

  histograms.omtf_hwEta_hwLocalPhi = booker.book2D("omtf_hwEta_hwLocalPhi", "HW #eta vs HW Local #phi", 447, -223.5, 223.5, 201, -100.5, 100.5);
  histograms.omtf_hwEta_hwLocalPhi.setTitle(";HW #eta; HW Local #phi");

  histograms.omtf_hwPt_hwEta = booker.book2D("omtf_hwPt_hwEta", "HW p_{T} vs HW #eta", 511, -0.5, 510.5, 447, -223.5, 223.5);
  histograms.omtf_hwPt_hwEta.setTitle(";HW p_{T}; HW #eta");

  histograms.omtf_hwPt_hwLocalPhi = booker.book2D("omtf_hwPt_hwLocalPhi", "HW p_{T} vs HW Local #phi", 511, -0.5, 510.5, 201, -100.5, 100.5);
  histograms.omtf_hwPt_hwLocalPhi.setTitle(";HW p_{T}; HW Local #phi");

  histograms.omtf_hwEta_bx = booker.book2D("omtf_hwEta_bx", "HW #eta vs BX", 447, -223.5, 223.5, 5, -2.5, 2.5);
  histograms.omtf_hwEta_bx.setTitle(";HW #eta; BX");

  histograms.omtf_hwLocalPhi_bx = booker.book2D("omtf_hwLocalPhi_bx", "HW Local #phi vs BX", 201, -100.5, 100.5, 5, -2.5, 2.5);
  histograms.omtf_hwLocalPhi_bx.setTitle(";HW Local #phi; BX");

  histograms.omtf_hwPt_bx = booker.book2D("omtf_hwPt_bx", "HW p_{T} vs BX", 511, -0.5, 510.5, 5, -2.5, 2.5);
  histograms.omtf_hwPt_bx.setTitle(";HW p_{T}; BX");

  histograms.omtf_hwQual_bx = booker.book2D("omtf_hwQual_bx", "HW Quality vs BX", 16, -0.5, 15.5, 5, -2.5, 2.5);
  histograms.omtf_hwQual_bx.setTitle("; HW Quality; BX");
}

void L1TStage2OMTF::dqmAnalyze(const edm::Event & eve, const edm::EventSetup & eveSetup, omtfdqm::Histograms const& histograms) const
{
  if (verbose) {
    edm::LogInfo("L1TStage2OMTF") << "L1TStage2OMTF: analyze...." << std::endl;
  }

  edm::Handle<l1t::RegionalMuonCandBxCollection> omtfMuon;
  eve.getByToken(omtfToken, omtfMuon);

  for(int itBX=omtfMuon->getFirstBX(); itBX<=omtfMuon->getLastBX(); ++itBX) {
    for(l1t::RegionalMuonCandBxCollection::const_iterator itMuon = omtfMuon->begin(itBX); itMuon != omtfMuon->end(itBX); ++itMuon) {
      histograms.omtf_hwEta.fill(itMuon->hwEta());
      histograms.omtf_hwLocalPhi.fill(itMuon->hwPhi());
      histograms.omtf_hwPt.fill(itMuon->hwPt());
      histograms.omtf_hwQual.fill(itMuon->hwQual());

      histograms.omtf_bx.fill(itBX);
      histograms.omtf_hwEta_bx.fill(itMuon->hwEta(), itBX);
      histograms.omtf_hwLocalPhi_bx.fill(itMuon->hwPhi(), itBX);
      histograms.omtf_hwPt_bx.fill(itMuon->hwPt(), itBX);
      histograms.omtf_hwQual_bx.fill(itMuon->hwQual(), itBX);

      histograms.omtf_hwEta_hwLocalPhi.fill(itMuon->hwEta(),itMuon->hwPhi());
      histograms.omtf_hwPt_hwEta.fill(itMuon->hwPt(), itMuon->hwEta());
      histograms.omtf_hwPt_hwLocalPhi.fill(itMuon->hwPt(), itMuon->hwPhi());
    }
  }
}

