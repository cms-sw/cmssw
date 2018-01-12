/*
 * \L1TStage2BMTF.cc
 * \author Esmaeel Eskandari Tadavani
 * \December 2015
/G.karathanasis     
*/

#include "DQM/L1TMonitor/interface/L1TStage2BMTF.h"

L1TStage2BMTF::L1TStage2BMTF(const edm::ParameterSet & ps):
  monitorDir(ps.getUntrackedParameter<std::string>("monitorDir", "")),
  bmtfSource(ps.getParameter<edm::InputTag>("bmtfSource")),
  //bmtfSourceTwinMux1(ps.getParameter<edm::InputTag>("bmtfSourceTwinMux1")),
  //bmtfSourceTwinMux2(ps.getParameter<edm::InputTag>("bmtfSourceTwinMux2")),
  verbose(ps.getUntrackedParameter<bool>("verbose", false))
{
  bmtfToken=consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("bmtfSource"));
  //bmtfTokenTwinMux1 = consumes<L1MuDTChambThContainer>(ps.getParameter<edm::InputTag>("bmtfSourceTwinMux1"));
  //bmtfTokenTwinMux2 = consumes<L1MuDTChambPhContainer>(ps.getParameter<edm::InputTag>("bmtfSourceTwinMux2"));

}

L1TStage2BMTF::~L1TStage2BMTF()
{
}

void L1TStage2BMTF::dqmBeginRun(const edm::Run& iRrun, const edm::EventSetup& eveSetup, bmtfdqm::Histograms& histograms) const
{
}

void L1TStage2BMTF::bookHistograms(DQMStore::ConcurrentBooker &booker, const edm::Run& iRun, const edm::EventSetup& eveSetup, bmtfdqm::Histograms& histograms) const
{
  booker.setCurrentFolder(monitorDir);
  
  histograms.bmtf_hwEta = booker.book1D("bmtf_hwEta", "HW #eta", 461, -230.5, 230.5);
  histograms.bmtf_hwLocalPhi = booker.book1D("bmtf_hwLocalPhi", "HW Local #phi", 201, -100.5, 100.5);
  histograms.bmtf_hwGlobalPhi = booker.book1D("bmtf_hwGlobalPhi", "HW Global #phi", 576, -0.5, 575.5);
  histograms.bmtf_hwPt = booker.book1D("bmtf_hwPt", "HW p_{T}", 512, -0.5, 511.5);
  histograms.bmtf_hwQual= booker.book1D("bmtf_hwQual", "HW Quality", 16, -0.5, 15.5);
  histograms.bmtf_proc = booker.book1D("bmtf_proc", "Processor", 12, -0.5, 11.5);

  histograms.bmtf_wedge_bx = booker.book2D("bmtf_wedge_bx", "Wedge vs BX", 12, -0.5, 11.5, 5, -2.5, 2.5);
  histograms.bmtf_wedge_bx.setTitle(";Wedge; BX");
  for (int bin = 1; bin < 13; ++bin) {
    histograms.bmtf_wedge_bx.setBinLabel(bin, std::to_string(bin), 1);
  }

  histograms.bmtf_hwEta_hwLocalPhi = booker.book2D("bmtf_hwEta_hwLocalPhi", "HW #eta vs HW Local #phi", 461, -230.5, 230.5, 201, -100.5, 100.5);
  histograms.bmtf_hwEta_hwLocalPhi.setTitle(";HW #eta; HW Local #phi");

  histograms.bmtf_hwEta_hwGlobalPhi = booker.book2D("bmtf_hwEta_hwGlobalPhi", "HW #eta vs HW Global #phi", 100, -230.5, 230.5, 120, -0.5, 575.5);
  histograms.bmtf_hwEta_hwGlobalPhi.setTitle(";HW #eta; HW Global #phi");

  histograms.bmtf_hwPt_hwEta = booker.book2D("bmtf_hwPt_hwEta", "HW p_{T} vs HW #eta", 511, -0.5, 510.5, 461, -230.5, 230.5);
  histograms.bmtf_hwPt_hwEta.setTitle(";HW p_{T}; HW #eta");

  histograms.bmtf_hwPt_hwLocalPhi = booker.book2D("bmtf_hwPt_hwLocalPhi", "HW p_{T} vs HW Local #phi", 511, -0.5, 510.5, 201, -100.5, 100.5);
  histograms.bmtf_hwPt_hwLocalPhi.setTitle(";HW p_{T}; HW Local #phi");

  histograms.bmtf_hwEta_bx = booker.book2D("bmtf_hwEta_bx", "HW #eta vs BX", 461, -230.5, 230.5, 5, -2.5, 2.5);
  histograms.bmtf_hwEta_bx.setTitle(";HW #eta; BX");

  histograms.bmtf_hwLocalPhi_bx = booker.book2D("bmtf_hwLocalPhi_bx", "HW Local #phi vs BX", 201, -100.5, 100.5, 5, -2.5, 2.5);
  histograms.bmtf_hwLocalPhi_bx.setTitle(";HW Local #phi; BX");

  histograms.bmtf_hwPt_bx = booker.book2D("bmtf_hwPt_bx", "HW p_{T} vs BX", 511, -0.5, 510.5, 5, -2.5, 2.5);
  histograms.bmtf_hwPt_bx.setTitle(";HW p_{T}; BX");

  histograms.bmtf_hwQual_bx = booker.book2D("bmtf_hwQual_bx", "HW Quality vs BX", 16, -0.5, 15.5, 5, -2.5, 2.5);
  histograms.bmtf_hwQual_bx.setTitle("; HW Quality; BX");

  // histograms.bmtf_twinmuxInput_PhiBX = booker.book1D("bmtf_twinmuxInput_PhiBX", "TwinMux Input Phi BX", 5, -2.5, 2.5);
  // histograms.bmtf_twinmuxInput_PhiPhi = booker.book1D("bmtf_twinmuxInput_PhiPhi", "TwinMux Input Phi HW Phi", 201, -100.5, 100.5);
  // histograms.bmtf_twinmuxInput_PhiPhiB = booker.book1D("bmtf_twinmuxInput_PhiPhiB", "TwinMux Input Phi HW PhiB", 201, -100.5, 100.5);
  // histograms.bmtf_twinmuxInput_PhiQual = booker.book1D("bmtf_twinmuxInput_PhiQual", "TwinMux Input Phi HW Quality", 20, -0.5, 19.5);
  // histograms.bmtf_twinmuxInput_PhiStation = booker.book1D("bmtf_twinmuxInput_PhiStation", "TwinMux Input Phi HW Station", 6, -1, 5);
  // histograms.bmtf_twinmuxInput_PhiSector = booker.book1D("bmtf_twinmuxInput_PhiSector", "TwinMux Input Phi HW Sector", 14, -1, 13);
  // histograms.bmtf_twinmuxInput_PhiWheel = booker.book1D("bmtf_twinmuxInput_PhiWheel", "TwinMux Input Phi HW Wheel", 16, -4, 4);
  // histograms.bmtf_twinmuxInput_PhiTrSeg = booker.book1D("bmtf_twinmuxInput_PhiTrSeg", "TwinMux Input Phi HW Track Segment", 6, -1, 5);
  // histograms.bmtf_twinmuxInput_PhiWheel_PhiSector = booker.book2D("bmtf_twinmuxInput_PhiWheel_PhiSector", "TwinMux Input Phi HW Wheel vs HW Sector", 16, -4, 4, 14, -1, 13);

  // histograms.bmtf_twinmuxInput_PhiWheel_PhiSector.setTitle("; TwinMux Input Phi HW Wheel; TwinMux Input Phi HW Sector");
  // for (int bin = 1; bin < 5; ++bin) {
  //   histograms.bmtf_twinmuxInput_PhiWheel_PhiSector.setBinLabel(bin, "station"+std::to_string(bin), 1);
  //   histograms.bmtf_twinmuxInput_PhiTrSeg.setBinLabel(bin, "station"+std::to_string(bin), 1);
  // }

  // histograms.bmtf_twinmuxInput_TheBX = booker.book1D("bmtf_twinmuxInput_TheBX", "TwinMux Input The BX", 5, -2.5, 2.5);
  // histograms.bmtf_twinmuxInput_ThePhi= booker.book1D("bmtf_twinmuxInput_ThePhi", "TwinMux Input The HW Phi", 201, -100.5, 100.5);
  // histograms.bmtf_twinmuxInput_ThePhiB = booker.book1D("bmtf_twinmuxInput_ThePhiB", "TwinMux Input The HW PhiB", 201, -100.5, 100.5);
  // histograms.bmtf_twinmuxInput_TheQual = booker.book1D("bmtf_twinmuxInput_TheQual", "TwinMux Input The HW Quality", 20, -0.5, 19.5);
  // histograms.bmtf_twinmuxInput_TheStation = booker.book1D("bmtf_twinmuxInput_TheStation", "TwinMux Input The HW Station", 6, -1, 5);
  // histograms.bmtf_twinmuxInput_TheSector = booker.book1D("bmtf_twinmuxInput_TheSector", "TwinMux Input The HW Sector", 14, -1, 13);
  // histograms.bmtf_twinmuxInput_TheWheel = booker.book1D("bmtf_twinmuxInput_TheWheel", "TwinMux Input The HW Wheel", 16, -4, 4);
  // histograms.bmtf_twinmuxInput_TheTrSeg = booker.book1D("bmtf_twinmuxInput_TheTrSeg", "TwinMux Input The HW Track Segment", 6, -1, 5);
  // histograms.bmtf_twinmuxInput_TheWheel_TheSector = booker.book2D("bmtf_twinmuxInput_TheWheel_TheSector", "TwinMux Input The HW Wheel vs HW Sector", 16, -4, 4, 14, -1, 13);

  // histograms.bmtf_twinmuxInput_TheWheel_TheSector.setTitle("; TwinMux Input The HW Wheel; TwinMux Input The HW Sector");
  // for (int bin = 1; bin < 5; ++bin) {
  //   histograms.bmtf_twinmuxInput_TheWheel_TheSector.setBinLabel(bin,  "station"+std::to_string(bin), 1);
  //   histograms.bmtf_twinmuxInput_TheTrSeg.setBinLabel(bin,  "station"+std::to_string(bin), 1);
  // }

}

void L1TStage2BMTF::dqmAnalyze(const edm::Event & eve, const edm::EventSetup & eveSetup, const bmtfdqm::Histograms & histograms) const
{
  if (verbose) {
    edm::LogInfo("L1TStage2BMTF") << "L1TStage2BMTF: analyze...." << std::endl;
  }

  edm::Handle<l1t::RegionalMuonCandBxCollection> bmtfMuon;
  eve.getByToken(bmtfToken, bmtfMuon);

  //  edm::Handle<L1MuDTChambThContainer> bmtfMuonTwinMux1;
  //  eve.getByToken(bmtfTokenTwinMux1, bmtfMuonTwinMux1);

  //  edm::Handle<L1MuDTChambPhContainer> bmtfMuonTwinMux2;
  //  eve.getByToken(bmtfTokenTwinMux2, bmtfMuonTwinMux2);

  for(int itBX=bmtfMuon->getFirstBX(); itBX<=bmtfMuon->getLastBX(); ++itBX) {
    for(l1t::RegionalMuonCandBxCollection::const_iterator itMuon = bmtfMuon->begin(itBX); itMuon != bmtfMuon->end(itBX); ++itMuon) {
      histograms.bmtf_hwEta.fill(itMuon->hwEta());
      histograms.bmtf_hwLocalPhi.fill(itMuon->hwPhi());
      histograms.bmtf_hwPt.fill(itMuon->hwPt());
      histograms.bmtf_hwQual.fill(itMuon->hwQual());
      histograms.bmtf_proc.fill(itMuon->processor());
      if (std::abs(bmtfMuon->getLastBX()-bmtfMuon->getFirstBX())>3) {
        histograms.bmtf_wedge_bx.fill(itMuon->processor(), itBX);
        histograms.bmtf_hwEta_bx.fill(itMuon->hwEta(), itBX);
        histograms.bmtf_hwLocalPhi_bx.fill(itMuon->hwPhi(), itBX);
        histograms.bmtf_hwPt_bx.fill(itMuon->hwPt(), itBX);
        histograms.bmtf_hwQual_bx.fill(itMuon->hwQual(), itBX);
      }
 
      histograms.bmtf_hwEta_hwLocalPhi.fill(itMuon->hwEta(),itMuon->hwPhi());
      histograms.bmtf_hwPt_hwEta.fill(itMuon->hwPt(), itMuon->hwEta());
      histograms.bmtf_hwPt_hwLocalPhi.fill(itMuon->hwPt(), itMuon->hwPhi());
 
      //if(itMuon->hwPhi()*0.010902>=0 && itMuon->hwPhi()*0.010902<=30)
      //  global_phi = itMuon->hwPhi() + itMuon->processor()*30.;
      //if(itMuon->hwPhi()*0.010902<0)
      //  global_phi = 30-itMuon->hwPhi() + (itMuon->processor()-1)*30.;
      //if(itMuon->hwPhi()*0.010902>30)
      //  global_phi = itMuon->hwPhi()-30 + (itMuon->processor()+1)*30.;
      float global_phi = itMuon->hwPhi() + itMuon->processor()*48. - 15;
      if (global_phi<0) global_phi = 576 + global_phi;

      histograms.bmtf_hwGlobalPhi.fill(global_phi);
      histograms.bmtf_hwEta_hwGlobalPhi.fill(itMuon->hwEta(), global_phi);
    }
  }

  //for(L1MuDTChambThContainer::The_Container::const_iterator itMuon = bmtfMuonTwinMux1->getContainer()->begin(); itMuon != bmtfMuonTwinMux1->getContainer()->end(); ++itMuon) {
  //  histograms.bmtf_twinmuxInput_TheBX.fill(itMuon->bxNum());
  //  //histograms.bmtf_twinmuxInput_ThePhi.fill(itMuon->phi());
  //  //histograms.bmtf_twinmuxInput_ThePhiB.fill(itMuon->phiB());
  //  //histograms.bmtf_twinmuxInput_TheQual.fill(itMuon->code());
  //  histograms.bmtf_twinmuxInput_TheStation.fill(itMuon->stNum());
  //  histograms.bmtf_twinmuxInput_TheSector.fill(itMuon->scNum());
  //  histograms.bmtf_twinmuxInput_TheWheel.fill(itMuon->whNum());

  //  for(int i = 1; i<=itMuon->stNum(); ++i) {
  //    //histograms.bmtf_twinmuxInput_TheTrSeg.fill(itMuon->Ts2Tag());
  //    histograms.bmtf_twinmuxInput_TheWheel_TheSector.fill(itMuon->whNum(), itMuon->scNum());
  //  }
  //}

  //for(L1MuDTChambPhContainer::Phi_Container::const_iterator itMuon = bmtfMuonTwinMux2->getContainer()->begin(); itMuon != bmtfMuonTwinMux2->getContainer()->end(); ++itMuon) {
  //  histograms.bmtf_twinmuxInput_PhiBX.fill(itMuon->bxNum());
  //  histograms.bmtf_twinmuxInput_PhiPhi.fill(itMuon->phi());
  //  histograms.bmtf_twinmuxInput_PhiPhiB.fill(itMuon->phiB());
  //  histograms.bmtf_twinmuxInput_PhiQual.fill(itMuon->code());
  //  histograms.bmtf_twinmuxInput_PhiStation.fill(itMuon->stNum());
  //  histograms.bmtf_twinmuxInput_PhiSector.fill(itMuon->scNum());
  //  histograms.bmtf_twinmuxInput_PhiWheel.fill(itMuon->whNum());

  //  for(int i = 1; i<= itMuon->stNum() ; ++i) {
  //    histograms.bmtf_twinmuxInput_PhiTrSeg.fill(itMuon->Ts2Tag());
  //    histograms.bmtf_twinmuxInput_PhiWheel_PhiSector.fill(itMuon->whNum(), itMuon->scNum());
  //  }
  //}
}

   






