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

  eta_mgmt = ibooker.book1D("eta_ugmt", "#eta of uGMT", 447, -223.5, 223.5);
  phi_mgmt = ibooker.book1D("phi_ugmt", "#phi of uGMT", 576, -0.5, 575.5);
  pt_mgmt = ibooker.book1D("pt_ugmt", "p_{T} of  uGMT", 511, -0.5, 510.5);
  charge_mgmt = ibooker.book1D("charge_ugmt", "charge of uGMT", 4, -2, 2);
  chargeVal_mgmt = ibooker.book1D("chargeVal_ugmt", "chargeValid  of uGMT", 4, -2, 2);
  qual_mgmt = ibooker.book1D("qual_ugmt", "quality of uGMT", 20, 0, 20);
  iso_mgmt = ibooker.book1D("iso_ugmt", "iso of uGMT", 4, 0, 4);

  bx_mgmt = ibooker.book1D("bx", "BX", 5, -2.5, 2.5);

  etaVSphi_mgmt      = ibooker.book2D("etaVSphi_ugmt"       , "#eta VS #phi of uGMT"      , 447, -223.5, 223.5, 576, -0.5, 575.5);
  phiVSpt_mgmt       = ibooker.book2D("phiVSpt_ugmt"        , "#phi VS p_{T} of uGMT"      , 576, -0.5, 575.5, 511, -0.5, 510.5);
  etaVSpt_mgmt       = ibooker.book2D("etaVSpt_ugmt"        , "#eta VS p_{T} of uGMT"     , 447, -223.5, 223.5, 511, -0.5, 510.5);

  etaVSbx_mgmt       = ibooker.book2D("etaVSbx_ugmt"        , "#eta VS bx of uGMT"        , 447, -223.5, 223.5,  5, -2.5, 2.5);
  phiVSbx_mgmt       = ibooker.book2D("phiVSbx_ugmt"        , "#phi VS bx of uGMT"        , 576, -0.5, 575.5,  5, -2.5, 2.5);
  ptVSbx_mgmt        = ibooker.book2D("ptVSbx_ugmt"         , "p_{T} VS bx of uGMT"       , 511, -0.5, 510.5,  5, -2.5, 2.5);
  chargeVSbx_mgmt    = ibooker.book2D("chargeVSbx_ugmt"     , "charge VS bx of uGMT"      , 4 ,   -2,    2,  5, -2.5, 2.5);
  chargeValVSbx_mgmt = ibooker.book2D("chargeValVSbx_ugmt"  , "chargeValid VS bx of uGMT" , 4 ,   -2,    2,  5, -2.5, 2.5);
  qualVSbx_mgmt      = ibooker.book2D("qualVSbx_ugmt"       , "quality VS bx of uGMT"     , 20,    0,   20,  5, -2.5, 2.5);
  isoVSbx_mgmt       = ibooker.book2D("isoVSbx_ugmt"        , "iso VS bx of uGMT"         , 4 ,    0,    4,  5, -2.5, 2.5);
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
	  eta_mgmt->Fill(itMuon->hwEta());
	  phi_mgmt->Fill(itMuon->hwPhi());
	  pt_mgmt->Fill(itMuon->hwPt());
	  charge_mgmt->Fill(itMuon->hwCharge());
	  chargeVal_mgmt->Fill(itMuon->hwChargeValid());
	  qual_mgmt->Fill(itMuon->hwQual());
	  iso_mgmt->Fill(itMuon->hwIso());
	  bx_mgmt->Fill(itBX);

	  etaVSbx_mgmt->Fill(itMuon->hwEta(),itBX); 
	  phiVSbx_mgmt->Fill(itMuon->hwPhi(),itBX); 
	  ptVSbx_mgmt->Fill(itMuon->hwPt()  ,itBX);  
	  chargeVSbx_mgmt->Fill(itMuon->hwCharge(), itBX);
	  chargeValVSbx_mgmt->Fill(itMuon->hwChargeValid(), itBX);
	  qualVSbx_mgmt->Fill(itMuon->hwQual(), itBX);
	  isoVSbx_mgmt->Fill(itMuon->hwIso(), itBX);

	  etaVSphi_mgmt->Fill(itMuon->hwEta(),itMuon->hwPhi()); 
	  phiVSpt_mgmt->Fill(itMuon->hwPhi(),itMuon->hwPt()); 
	  etaVSpt_mgmt->Fill(itMuon->hwEta(),itMuon->hwPt());
	  

	}
    }
}



