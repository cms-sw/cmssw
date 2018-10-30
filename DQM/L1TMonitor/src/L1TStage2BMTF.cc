/*
 * \L1TStage2BMTF.cc
 * \author Esmaeel Eskandari Tadavani
 * \December 2015
/G.karathanasis     
*/

#include "DQM/L1TMonitor/interface/L1TStage2BMTF.h"

L1TStage2BMTF::L1TStage2BMTF(const edm::ParameterSet & ps):
  monitorDir(ps.getUntrackedParameter<std::string>("monitorDir","")),
  bmtfSource(ps.getParameter<edm::InputTag>("bmtfSource")),
  //  bmtfSourceTwinMux1(ps.getParameter<edm::InputTag>("bmtfSourceTwinMux1")),
  //  bmtfSourceTwinMux2(ps.getParameter<edm::InputTag>("bmtfSourceTwinMux2")),
  verbose(ps.getUntrackedParameter<bool>("verbose", false)),
  kalman(ps.getUntrackedParameter<bool>("kalman", false)),
  global_phi(-1000)
{
  bmtfToken=consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("bmtfSource"));
  //  bmtfTokenTwinMux1 = consumes<L1MuDTChambThContainer>(ps.getParameter<edm::InputTag>("bmtfSourceTwinMux1"));
  //  bmtfTokenTwinMux2 = consumes<L1MuDTChambPhContainer>(ps.getParameter<edm::InputTag>("bmtfSourceTwinMux2"));

}

L1TStage2BMTF::~L1TStage2BMTF()
{
}

void L1TStage2BMTF::dqmBeginRun(const edm::Run& iRrun, const edm::EventSetup& eveSetup)
{
}


void L1TStage2BMTF::bookHistograms(DQMStore::IBooker &ibooker, const edm::Run& iRun, const edm::EventSetup& eveSetup)
{
  std::string histoPrefix = "bmtf";
  if (kalman) {
    histoPrefix = "kbmtf";
  }

  int ptbins = 512;
  int hwQual_bxbins = 20;
  if (kalman) {
    ptbins = 522;
    hwQual_bxbins = 15;
  }

  ibooker.setCurrentFolder(monitorDir);
  bmtf_hwEta = ibooker.book1D(histoPrefix+"_hwEta", "HW #eta", 461, -230.5, 230.5);
  bmtf_hwLocalPhi = ibooker.book1D(histoPrefix+"_hwLocalPhi", "HW Local #phi", 201, -100.5, 100.5);
  bmtf_hwGlobalPhi = ibooker.book1D(histoPrefix+"_hwGlobalPhi", "HW Global #phi", 576, -0.5, 575.5);
  bmtf_hwPt  = ibooker.book1D(histoPrefix+"_hwPt", "HW p_{T}", ptbins, -0.5, ptbins-0.5);
  bmtf_hwQual= ibooker.book1D(histoPrefix+"_hwQual" , "HW Quality", 20, -0.5, 19.5);
  bmtf_proc  = ibooker.book1D(histoPrefix+"_proc" , "Processor", 12, -0.5, 11.5);

  bmtf_wedge_bx  = ibooker.book2D(histoPrefix+"_wedge_bx", "Wedge vs BX", 12, -0.5, 11.5, 5, -2.5, 2.5);
  bmtf_wedge_bx->setTitle(";Wedge; BX");
  for (int bin = 1; bin < 13; ++bin) {
    bmtf_wedge_bx->setBinLabel(bin, std::to_string(bin), 1);
  }

  bmtf_hwEta_hwLocalPhi = ibooker.book2D(histoPrefix+"_hwEta_hwLocalPhi", "HW #eta vs HW Local #phi" , 461, -230.5, 230.5, 201, -100.5, 100.5);
  bmtf_hwEta_hwLocalPhi->setTitle(";HW #eta; HW Local #phi");

  bmtf_hwEta_hwGlobalPhi = ibooker.book2D(histoPrefix+"_hwEta_hwGlobalPhi", "HW #eta vs HW Global #phi" , 100, -230.5, 230.5, 120, -0.5, 575.5);
  bmtf_hwEta_hwGlobalPhi->setTitle(";HW #eta; HW Global #phi");

  bmtf_hwPt_hwEta  = ibooker.book2D(histoPrefix+"_hwPt_hwEta" , "HW p_{T} vs HW #eta", 511, -0.5, 510.5, 461, -230.5, 230.5);
  bmtf_hwPt_hwEta->setTitle(";HW p_{T}; HW #eta");

  bmtf_hwPt_hwLocalPhi  = ibooker.book2D(histoPrefix+"_hwPt_hwLocalPhi" , "HW p_{T} vs HW Local #phi", 511, -0.5, 510.5, 201, -100.5, 100.5);
  bmtf_hwPt_hwLocalPhi->setTitle(";HW p_{T}; HW Local #phi");

  bmtf_hwEta_bx    = ibooker.book2D(histoPrefix+"_hwEta_bx"   , "HW #eta vs BX"      , 461, -230.5, 230.5,  5, -2.5, 2.5);
  bmtf_hwEta_bx->setTitle(";HW #eta; BX");

  bmtf_hwLocalPhi_bx    = ibooker.book2D(histoPrefix+"_hwLocalPhi_bx"   , "HW Local #phi vs BX"      , 201, -100.5, 100.5,  5, -2.5, 2.5);
  bmtf_hwLocalPhi_bx->setTitle(";HW Local #phi; BX");

  bmtf_hwPt_bx     = ibooker.book2D(histoPrefix+"_hwPt_bx"    , "HW p_{T} vs BX"     , 511,   -0.5, 510.5,  5, -2.5, 2.5);
  bmtf_hwPt_bx->setTitle(";HW p_{T}; BX");

  bmtf_hwQual_bx   = ibooker.book2D(histoPrefix+"_hwQual_bx"  , "HW Quality vs BX"      ,  hwQual_bxbins,   -0.5,  hwQual_bxbins-0.5,  5, -2.5, 2.5);
  bmtf_hwQual_bx->setTitle("; HW Quality; BX");

  bmtf_hwDXY = ibooker.book1D(histoPrefix+"_hwDXY", "HW DXY", 4, 0, 4);
  bmtf_hwPt2 = ibooker.book1D(histoPrefix+"_hwPt2", "HW p_{T}2", 512, -0.5, 511.5);

  // bmtf_twinmuxInput_PhiBX = ibooker.book1D(histoPrefix+"_twinmuxInput_PhiBX"  , "TwinMux Input Phi BX"      ,  5, -2.5, 2.5);
  // bmtf_twinmuxInput_PhiPhi = ibooker.book1D(histoPrefix+"_twinmuxInput_PhiPhi"  , "TwinMux Input Phi HW Phi"      , 201, -100.5, 100.5);
  // bmtf_twinmuxInput_PhiPhiB = ibooker.book1D(histoPrefix+"_twinmuxInput_PhiPhiB"  , "TwinMux Input Phi HW PhiB"   , 201, -100.5, 100.5);
  // bmtf_twinmuxInput_PhiQual = ibooker.book1D(histoPrefix+"_twinmuxInput_PhiQual"  , "TwinMux Input Phi HW Quality" , 20,   -0.5,  19.5);
  // bmtf_twinmuxInput_PhiStation = ibooker.book1D(histoPrefix+"_twinmuxInput_PhiStation"  , "TwinMux Input Phi HW Station", 6, -1, 5);
  // bmtf_twinmuxInput_PhiSector = ibooker.book1D(histoPrefix+"_twinmuxInput_PhiSector" , "TwinMux Input Phi HW Sector"    , 14, -1,  13 );
  // bmtf_twinmuxInput_PhiWheel = ibooker.book1D(histoPrefix+"_twinmuxInput_PhiWheel"  , "TwinMux Input Phi HW Wheel"      , 16 , -4, 4);
  // bmtf_twinmuxInput_PhiTrSeg = ibooker.book1D(histoPrefix+"_twinmuxInput_PhiTrSeg"  , "TwinMux Input Phi HW Track Segment"      , 6, -1, 5 );
  // bmtf_twinmuxInput_PhiWheel_PhiSector = ibooker.book2D(histoPrefix+"_twinmuxInput_PhiWheel_PhiSector"  , "TwinMux Input Phi HW Wheel vs HW Sector", 16 , -4, 4, 14, -1,  13 );

  // bmtf_twinmuxInput_PhiWheel_PhiSector->setTitle("; TwinMux Input Phi HW Wheel; TwinMux Input Phi HW Sector");
  // for (int bin = 1; bin < 5; ++bin) {
  //   bmtf_twinmuxInput_PhiWheel_PhiSector->setBinLabel(bin, "station"+std::to_string(bin), 1);
  //   bmtf_twinmuxInput_PhiTrSeg->setBinLabel(bin, "station"+std::to_string(bin), 1);
  // }

  // bmtf_twinmuxInput_TheBX = ibooker.book1D(histoPrefix+"_twinmuxInput_TheBX"  , "TwinMux Input The BX"      ,   5, -2.5, 2.5);
  // bmtf_twinmuxInput_ThePhi= ibooker.book1D(histoPrefix+"_twinmuxInput_ThePhi"  , "TwinMux Input The HW Phi"      ,  201, -100.5, 100.5);
  // bmtf_twinmuxInput_ThePhiB = ibooker.book1D(histoPrefix+"_twinmuxInput_ThePhiB"  , "TwinMux Input The HW PhiB"   ,  201, -100.5, 100.5);
  // bmtf_twinmuxInput_TheQual = ibooker.book1D(histoPrefix+"_twinmuxInput_TheQual"  , "TwinMux Input The HW Quality" ,  20,   -0.5,  19.5);
  // bmtf_twinmuxInput_TheStation = ibooker.book1D(histoPrefix+"_twinmuxInput_TheStation"  , "TwinMux Input The HW Station"      , 6, -1, 5);
  // bmtf_twinmuxInput_TheSector = ibooker.book1D(histoPrefix+"_twinmuxInput_TheSector" , "TwinMux Input The HW Sector"      ,  14, -1,  13 );
  // bmtf_twinmuxInput_TheWheel = ibooker.book1D(histoPrefix+"_twinmuxInput_TheWheel"  , "TwinMux Input The HW Wheel"      ,   16 , -4, 4);
  // bmtf_twinmuxInput_TheTrSeg = ibooker.book1D(histoPrefix+"_twinmuxInput_TheTrSeg"  , "TwinMux Input The HW Track Segment"      , 6, -1, 5 );
  // bmtf_twinmuxInput_TheWheel_TheSector = ibooker.book2D(histoPrefix+"_twinmuxInput_TheWheel_TheSector"  , "TwinMux Input The HW Wheel vs HW Sector", 16 , -4, 4, 14, -1,  13);

  // bmtf_twinmuxInput_TheWheel_TheSector->setTitle("; TwinMux Input The HW Wheel; TwinMux Input The HW Sector");
  // for (int bin = 1; bin < 5; ++bin) {
  //   bmtf_twinmuxInput_TheWheel_TheSector->setBinLabel(bin,  "station"+std::to_string(bin), 1);
  //   bmtf_twinmuxInput_TheTrSeg->setBinLabel(bin,  "station"+std::to_string(bin), 1);
  // }

}

void L1TStage2BMTF::analyze(const edm::Event & eve, const edm::EventSetup & eveSetup)
{

  edm::Handle<l1t::RegionalMuonCandBxCollection> bmtfMuon;
  eve.getByToken(bmtfToken, bmtfMuon);

  //  edm::Handle<L1MuDTChambThContainer> bmtfMuonTwinMux1;
  //  eve.getByToken(bmtfTokenTwinMux1, bmtfMuonTwinMux1);

  //  edm::Handle<L1MuDTChambPhContainer> bmtfMuonTwinMux2;
  //  eve.getByToken(bmtfTokenTwinMux2, bmtfMuonTwinMux2);

  for(int itBX=bmtfMuon->getFirstBX(); itBX<=bmtfMuon->getLastBX(); ++itBX)
    {
      for(l1t::RegionalMuonCandBxCollection::const_iterator itMuon = bmtfMuon->begin(itBX); itMuon != bmtfMuon->end(itBX); ++itMuon)
        {  

          bmtf_hwEta->Fill(itMuon->hwEta());
          bmtf_hwLocalPhi->Fill(itMuon->hwPhi());
          bmtf_hwPt->Fill(itMuon->hwPt());
          bmtf_hwQual->Fill(itMuon->hwQual());
          bmtf_proc->Fill(itMuon->processor());

	  bmtf_hwDXY->Fill(itMuon->hwDXY());
	  bmtf_hwPt2->Fill(itMuon->hwPt2());

          if (fabs(bmtfMuon->getLastBX()-bmtfMuon->getFirstBX())>3){
            bmtf_wedge_bx->Fill(itMuon->processor(), itBX);  
            bmtf_hwEta_bx->Fill(itMuon->hwEta(), itBX);  
            bmtf_hwLocalPhi_bx->Fill(itMuon->hwPhi(), itBX);  
            bmtf_hwPt_bx->Fill(itMuon->hwPt(), itBX);   
            bmtf_hwQual_bx->Fill(itMuon->hwQual(), itBX);}
 
          bmtf_hwEta_hwLocalPhi->Fill(itMuon->hwEta(),itMuon->hwPhi());
          bmtf_hwPt_hwEta->Fill(itMuon->hwPt(), itMuon->hwEta());
          bmtf_hwPt_hwLocalPhi->Fill(itMuon->hwPt(), itMuon->hwPhi());
 
        /*if(itMuon->hwPhi()*0.010902>=0 && itMuon->hwPhi()*0.010902<=30)
            global_phi = itMuon->hwPhi() + itMuon->processor()*30.;
          if(itMuon->hwPhi()*0.010902<0)
            global_phi = 30-itMuon->hwPhi() + (itMuon->processor()-1)*30.;
          if(itMuon->hwPhi()*0.010902>30)
            global_phi = itMuon->hwPhi()-30 + (itMuon->processor()+1)*30.;*/
          global_phi= itMuon->hwPhi() + itMuon->processor()*48.-15;
          if (global_phi<0) global_phi=576+global_phi;

          bmtf_hwGlobalPhi->Fill(global_phi);
          bmtf_hwEta_hwGlobalPhi->Fill(itMuon->hwEta(), global_phi);
        }
    }

      // for(L1MuDTChambThContainer::The_Container::const_iterator itMuon = bmtfMuonTwinMux1->getContainer()->begin(); itMuon != bmtfMuonTwinMux1->getContainer()->end(); ++itMuon)
      // 	{ 

      // 	  bmtf_twinmuxInput_TheBX->Fill(itMuon->bxNum());
      // 	  //	  bmtf_twinmuxInput_ThePhi->Fill(itMuon->phi());
      // 	  //	  bmtf_twinmuxInput_ThePhiB->Fill(itMuon->phiB());
      // 	  //	  bmtf_twinmuxInput_TheQual->Fill(itMuon->code());
      // 	  bmtf_twinmuxInput_TheStation->Fill(itMuon->stNum());
      // 	  bmtf_twinmuxInput_TheSector->Fill(itMuon->scNum());
      // 	  bmtf_twinmuxInput_TheWheel->Fill(itMuon->whNum());

      //     for(int i = 1; i<=itMuon->stNum(); ++i)
      //       {
      // 	      //	      bmtf_twinmuxInput_TheTrSeg->Fill(itMuon->Ts2Tag());
      // 	      bmtf_twinmuxInput_TheWheel_TheSector->Fill(itMuon->whNum(), itMuon->scNum());
      // 	    }
    
      // 	}

  // for(L1MuDTChambPhContainer::Phi_Container::const_iterator itMuon = bmtfMuonTwinMux2->getContainer()->begin(); itMuon != bmtfMuonTwinMux2->getContainer()->end(); ++itMuon)
  // 	{  

  // 	  bmtf_twinmuxInput_PhiBX->Fill(itMuon->bxNum());
  // 	  bmtf_twinmuxInput_PhiPhi->Fill(itMuon->phi());
  // 	  bmtf_twinmuxInput_PhiPhiB->Fill(itMuon->phiB());
  // 	  bmtf_twinmuxInput_PhiQual->Fill(itMuon->code());
  // 	  bmtf_twinmuxInput_PhiStation->Fill(itMuon->stNum());
  // 	  bmtf_twinmuxInput_PhiSector->Fill(itMuon->scNum());
  // 	  bmtf_twinmuxInput_PhiWheel->Fill(itMuon->whNum());

  // 	  for(int i = 1; i<= itMuon->stNum() ; ++i)
  // 	    {  
  // 	      bmtf_twinmuxInput_PhiTrSeg->Fill(itMuon->Ts2Tag());
  // 	      bmtf_twinmuxInput_PhiWheel_PhiSector->Fill(itMuon->whNum(), itMuon->scNum());
  // 	    }


  // 	}
  
}

   






