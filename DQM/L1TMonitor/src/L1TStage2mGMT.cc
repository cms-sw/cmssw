/*
 * \L1TStage2mGMT.cc
 * \author Esmaeel Eskandari Tadavani
*/

#include "DQM/L1TMonitor/interface/L1TStage2mGMT.h"

L1TStage2mGMT::L1TStage2mGMT(const edm::ParameterSet & ps):
  monitorDir(ps.getUntrackedParameter<std::string>("monitorDir","")),
  stage2mgmtSource(ps.getParameter<edm::InputTag>("stage2mgmtSource")),
  verbose(ps.getUntrackedParameter<bool>("verbose", false))
{
  stage2mgmtToken=consumes<l1t::MuonBxCollection>(ps.getParameter<edm::InputTag>("stage2mgmtSource"));
}

L1TStage2mGMT::~L1TStage2mGMT()
{
}

void L1TStage2mGMT::dqmBeginRun(const edm::Run& iRrun, const edm::EventSetup& eveSetup)
{
}

void L1TStage2mGMT::beginLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup& eveSetup)
{
}


void L1TStage2mGMT::bookHistograms(DQMStore::IBooker &ibooker, const edm::Run& iRun, const edm::EventSetup& eveSetup)
{

  ibooker.setCurrentFolder(monitorDir);

  hw_eta_ugmt = ibooker.book1D("hw_eta_ugmt", "HW #eta of uGMT", 99, -49.5, 49.5);
  hw_phi_ugmt = ibooker.book1D("hw_phi_ugmt", "HW #phi of uGMT", 126, -0.5, 125.5);
  hw_pt_ugmt  = ibooker.book1D("hw_pt_ugmt" , "HW p_{T} of  uGMT", 128, -0.5, 127.5);

  ph_eta_ugmt = ibooker.book1D("ph_eta_ugmt", "PH #eta of uGMT", 99, -2.475, 2.475);
  ph_phi_ugmt = ibooker.book1D("ph_phi_ugmt", "PH #phi of uGMT",126 , -0.5, 6.28);
  ph_pt_ugmt  = ibooker.book1D("ph_pt_ugmt" , "PH p_{T} of uGMT", 128, -1, 255);

  charge_ugmt = ibooker.book1D("charge_ugmt", "HW charge of uGMT", 3, -1, 2);
  chargeVal_ugmt = ibooker.book1D("chargeVal_ugmt", "chargeValid  of uGMT", 2, 0, 2);
  qual_ugmt = ibooker.book1D("qual_ugmt", "quality of uGMT", 20, 0, 20);
  iso_ugmt = ibooker.book1D("iso_ugmt", "iso of uGMT", 4, 0, 4);

  bx_ugmt = ibooker.book1D("bx", "BX", 5, -2.5, 2.5);

  hw_etaVSphi_ugmt      = ibooker.book2D("hw_etaVSphi_ugmt"       , "HW #eta VS HW #phi of uGMT"      , 99, -49.5, 49.5, 126, -0.5, 125.5);
  hw_phiVSpt_ugmt       = ibooker.book2D("hw_phiVSpt_ugmt"        , "HW #phi VS HW p_{T} of uGMT"      , 126, -0.5, 125.5, 128, -0.5, 127.5);
  hw_etaVSpt_ugmt       = ibooker.book2D("hw_etaVSpt_ugmt"        , "HW #eta VS HW p_{T} of uGMT"     , 99, -49.5, 49.5, 128, -0.5, 127.5);

  ph_etaVSphi_ugmt      = ibooker.book2D("ph_etaVSphi_ugmt"       , "PH #eta VS PH #phi of uGMT"      ,  99, -2.475, 2.475, 126, -0.05, 6.28);
  ph_phiVSpt_ugmt       = ibooker.book2D("ph_phiVSpt_ugmt"        , "PH #phi VS PH p_{T} of uGMT"     , 126,   -0.05, 6.28 , 128,   -1,  255);
  ph_etaVSpt_ugmt       = ibooker.book2D("ph_etaVSpt_ugmt"        , "PH #eta VS PH p_{T} of uGMT"     ,  99, -2.475, 2.475, 128,   -1,  255);

  hw_etaVSbx_ugmt       = ibooker.book2D("hw_etaVSbx_ugmt"        , "HW #eta VS bx of uGMT"        , 99, -49.5, 49.5,  5, -2.5, 2.5);
  hw_phiVSbx_ugmt       = ibooker.book2D("hw_phiVSbx_ugmt"        , "HW #phi VS bx of uGMT"        , 126, -0.5, 125.5,  5, -2.5, 2.5);
  hw_ptVSbx_ugmt        = ibooker.book2D("hw_ptVSbx_ugmt"         , "HW p_{T} VS bx of uGMT"       , 128, -0.5, 127.5,  5, -2.5, 2.5);

  ph_etaVSbx_ugmt       = ibooker.book2D("ph_etaVSbx_ugmt"        , "PH #eta VS bx of uGMT"        , 99, -2.475, 2.475,  5, -2.5, 2.5);
  ph_phiVSbx_ugmt       = ibooker.book2D("ph_phiVSbx_ugmt"        , "PH #phi VS bx of uGMT"        ,126,   -0.05, 6.28,  5, -2.5, 2.5);
  ph_ptVSbx_ugmt        = ibooker.book2D("ph_ptVSbx_ugmt"         , "PH p_{T} VS bx of uGMT"       , 128,   -1,  255,  5, -2.5, 2.5);

  chargeVSbx_ugmt       = ibooker.book2D("chargeVSbx_ugmt"     , "charge VS bx of uGMT"      , 3 ,   -1,    2,  5, -2.5, 2.5);
  chargeValVSbx_ugmt    = ibooker.book2D("chargeValVSbx_ugmt"  , "chargeValid VS bx of uGMT" , 2 ,   0,    2,  5, -2.5, 2.5);
  qualVSbx_ugmt         = ibooker.book2D("qualVSbx_ugmt"       , "quality VS bx of uGMT"     , 20,    0,   20,  5, -2.5, 2.5);
  isoVSbx_ugmt          = ibooker.book2D("isoVSbx_ugmt"        , "iso VS bx of uGMT"         , 4 ,    0,    4,  5, -2.5, 2.5);

}

void L1TStage2mGMT::analyze(const edm::Event & eve, const edm::EventSetup & eveSetup)
{
  if (verbose) {
    edm::LogInfo("L1TStage2mGMT") << "L1TStage2mGMT: analyze...." << std::endl;
  }
  // analyze Jet
  edm::Handle<l1t::MuonBxCollection> Muon;
  eve.getByToken(stage2mgmtToken,Muon);


  for(int itBX=Muon->getFirstBX(); itBX<=Muon->getLastBX(); ++itBX)
    {
      for(l1t::MuonBxCollection::const_iterator itMuon = Muon->begin(itBX); itMuon != Muon->end(itBX); ++itMuon)
	{  

	  hw_eta_ugmt->Fill(itMuon->hwEta());
	  hw_phi_ugmt->Fill(itMuon->hwPhi());
	  hw_pt_ugmt->Fill(itMuon->hwPt());

	  ph_eta_ugmt->Fill(itMuon->hwEta()*0.05);
	  ph_phi_ugmt->Fill(itMuon->hwPhi()*0.05);
	  ph_pt_ugmt->Fill(itMuon->hwPt()*2);

	  charge_ugmt->Fill(itMuon->hwCharge());
	  chargeVal_ugmt->Fill(itMuon->hwChargeValid());
	  qual_ugmt->Fill(itMuon->hwQual());
	  iso_ugmt->Fill(itMuon->hwIso());
	  bx_ugmt->Fill(itBX);

	  hw_etaVSbx_ugmt->Fill(itMuon->hwEta(),itBX); 
	  hw_phiVSbx_ugmt->Fill(itMuon->hwPhi(),itBX); 
	  hw_ptVSbx_ugmt->Fill(itMuon->hwPt()  ,itBX);  

	  ph_etaVSbx_ugmt->Fill(itMuon->hwEta()* 0.05, itBX); 
	  ph_phiVSbx_ugmt->Fill(itMuon->hwPhi()* 0.05, itBX); 
	  ph_ptVSbx_ugmt->Fill(itMuon->hwPt()*2, itBX);  

	  hw_etaVSphi_ugmt->Fill(itMuon->hwEta(),itMuon->hwPhi()); 
	  hw_phiVSpt_ugmt->Fill(itMuon->hwPhi(),itMuon->hwPt()); 
	  hw_etaVSpt_ugmt->Fill(itMuon->hwEta(),itMuon->hwPt());

	  ph_etaVSphi_ugmt->Fill(itMuon->hwEta()*0.05, itMuon->hwPhi()*0.05); 
	  ph_phiVSpt_ugmt->Fill(itMuon->hwPhi()*0.05,itMuon->hwPt()*2); 
	  ph_etaVSpt_ugmt->Fill(itMuon->hwEta()*0.05,itMuon->hwPt()*2);

	  chargeVSbx_ugmt->Fill(itMuon->hwCharge(), itBX);
	  chargeValVSbx_ugmt->Fill(itMuon->hwChargeValid(), itBX);
	  qualVSbx_ugmt->Fill(itMuon->hwQual(), itBX);
	  isoVSbx_ugmt->Fill(itMuon->hwIso(), itBX);	  

	}
    }
}



