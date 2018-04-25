#include "DQM/L1TMonitor/interface/L1TObjectsTiming.h"


L1TObjectsTiming::L1TObjectsTiming(const edm::ParameterSet& ps)
    : ugmtMuonToken_(consumes<l1t::MuonBxCollection>(ps.getParameter<edm::InputTag>("muonProducer"))),
      stage2CaloLayer2JetToken_(consumes<l1t::JetBxCollection>(ps.getParameter<edm::InputTag>("stage2CaloLayer2JetProducer"))),
      stage2CaloLayer2EGammaToken_(consumes<l1t::EGammaBxCollection>(ps.getParameter<edm::InputTag>("stage2CaloLayer2EGammaProducer"))),
      stage2CaloLayer2TauToken_(consumes<l1t::TauBxCollection>(ps.getParameter<edm::InputTag>("stage2CaloLayer2TauProducer"))),
      stage2CaloLayer2EtSumToken_(consumes<l1t::EtSumBxCollection>(ps.getParameter<edm::InputTag>("stage2CaloLayer2EtSumProducer"))),
      l1tStage2uGtProducer_(consumes<GlobalAlgBlkBxCollection>(ps.getParameter<edm::InputTag>("ugtProducer"))),
      monitorDir_(ps.getUntrackedParameter<std::string>("monitorDir")),
      verbose_(ps.getUntrackedParameter<bool>("verbose")),
      gtUtil_(new l1t::L1TGlobalUtil(ps, consumesCollector(), *this, ps.getParameter<edm::InputTag>("ugtProducer"), ps.getParameter<edm::InputTag>("ugtProducer"))),
      algoBitFirstBxInTrain_(-1),
      algoBitLastBxInTrain_(-1),
      algoBitIsoBx_(-1),
      algoNameFirstBxInTrain_(ps.getUntrackedParameter<std::string>("firstBXInTrainAlgo","")),
      algoNameLastBxInTrain_(ps.getUntrackedParameter<std::string>("lastBXInTrainAlgo","")),
      algoNameIsoBx_(ps.getUntrackedParameter<std::string>("isoBXAlgo","")),
      bxrange_(5),
      egammaPtCuts_({10, 20, 30}),
      egammaPtCut_(20.),
      jetPtCut_(20.),
      tauPtCut_(20.),
      etsumPtCut_(20.),
      muonPtCut_(8.),
      muonQualCut_(12)
{
  if (ps.getUntrackedParameter<std::string>("useAlgoDecision").find("final") == 0) {
    useAlgoDecision_ = 2;
  } else if (ps.getUntrackedParameter<std::string>("useAlgoDecision").find("intermediate") == 0) {
    useAlgoDecision_ = 1;
  } else {
    useAlgoDecision_ = 0;
  }
}

L1TObjectsTiming::~L1TObjectsTiming() {}

void L1TObjectsTiming::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("muonProducer")->setComment("L1T muons");;
  desc.add<edm::InputTag>("stage2CaloLayer2JetProducer")->setComment("L1T jets");
  desc.add<edm::InputTag>("stage2CaloLayer2EGammaProducer")->setComment("L1T egamma");
  desc.add<edm::InputTag>("stage2CaloLayer2TauProducer")->setComment("L1T taus");
  desc.add<edm::InputTag>("stage2CaloLayer2EtSumProducer")->setComment("L1T etsums");
  desc.add<edm::InputTag>("ugtProducer")->setComment("uGT output");
  desc.addUntracked<std::string>("monitorDir", "")->setComment("Target directory in the DQM file. Will be created if not existing.");
  desc.addUntracked<bool>("verbose", false);
  desc.addUntracked<std::string>("firstBXInTrainAlgo", "")->setComment("Pick the right algo name for L1 First Collision In Train");
  desc.addUntracked<std::string>("lastBXInTrainAlgo", "")->setComment("Pick the right algo name for L1 Last Collision In Train");
  desc.addUntracked<std::string>("isoBXAlgo", "")->setComment("Pick the right algo name for L1 Isolated Bunch");
  desc.addUntracked<std::string>("useAlgoDecision", "initial")->setComment("Which algo decision should be checked [initial, intermediate, final].");
  descriptions.add("l1tObjectsTiming", desc);
}

void L1TObjectsTiming::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) {
  // Get the trigger menu information
  gtUtil_->retrieveL1Setup(c);
  // Get the algo bits needed for the timing histograms
  if (!gtUtil_->getAlgBitFromName(algoNameFirstBxInTrain_, algoBitFirstBxInTrain_)) {
    edm::LogWarning("L1TObjectsTiming") << "Algo \"" << algoNameFirstBxInTrain_ << "\" not found in the trigger menu " << gtUtil_->gtTriggerMenuName() << ". Could not retrieve algo bit number.";
  }

  if (!gtUtil_->getAlgBitFromName(algoNameLastBxInTrain_, algoBitLastBxInTrain_)) {
    edm::LogWarning("L1TObjectsTiming") << "Algo \"" << algoNameLastBxInTrain_ << "\" not found in the trigger menu " << gtUtil_->gtTriggerMenuName() << ". Could not retrieve algo bit number.";
  }

  if (!gtUtil_->getAlgBitFromName(algoNameIsoBx_, algoBitIsoBx_)) {
    edm::LogWarning("L1TObjectsTiming") << "Algo \"" << algoNameIsoBx_ << "\" not found in the trigger menu " << gtUtil_->gtTriggerMenuName() << ". Could not retrieve algo bit number.";
  }
}

void L1TObjectsTiming::beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) {}

void L1TObjectsTiming::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) {

  std::string bx_obj[5]={"minus2","minus1","0","plus1","plus2"};

  // all bunches
  ibooker.setCurrentFolder(monitorDir_+"/L1TMuon/timing"); 
  for(unsigned int i=0; i<bxrange_; ++i) {
    muons_eta_phi.push_back(ibooker.book2D("muons_eta_phi_bx_"+bx_obj[i],"L1T Muon #eta vs #phi BX="+bx_obj[i],25, -2.5, 2.5, 25, -3.2, 3.2));
    muons_eta_phi.at(i)->setAxisTitle("#eta", 1);
    muons_eta_phi.at(i)->setAxisTitle("#phi", 2);
  }
  denominator_muons = ibooker.book2D("denominator_muons","Denominator for L1TMuon",25, -2.5, 2.5, 25, -3.2, 3.2);
  denominator_muons->setAxisTitle("#eta", 1);
  denominator_muons->setAxisTitle("#phi", 2);
 
  ibooker.setCurrentFolder(monitorDir_+"/L1TJet/timing"); 
  for(unsigned int i=0; i<bxrange_; ++i) { 
    jet_eta_phi.push_back(ibooker.book2D("jet_eta_phi_bx_"+bx_obj[i],"L1T Jet #eta vs #phi BX="+bx_obj[i], 50, -5., 5., 25, -3.2, 3.2));
    jet_eta_phi.at(i)->setAxisTitle("#eta", 1);
    jet_eta_phi.at(i)->setAxisTitle("#phi", 2);
  }
  denominator_jet = ibooker.book2D("denominator_jet","Denominator for L1TJet", 50, -5., 5., 25, -3.2, 3.2);
  denominator_jet->setAxisTitle("#eta", 1);
  denominator_jet->setAxisTitle("#phi", 2);

  ibooker.setCurrentFolder(monitorDir_+"/L1TEGamma/timing"); 
  for(unsigned int i=0; i<bxrange_; ++i) {
    egamma_eta_phi.push_back(ibooker.book2D("egamma_eta_phi_bx_"+bx_obj[i],"L1T EGamma #eta vs #phi BX="+bx_obj[i], 30, -3., 3., 25, -3.2, 3.2));
    egamma_eta_phi.at(i)->setAxisTitle("#eta", 1);
    egamma_eta_phi.at(i)->setAxisTitle("#phi", 2);
  }
  denominator_egamma = ibooker.book2D("denominator_egamma","Denominator for L1TEGamma", 30, -3., 3., 25, -3.2, 3.2);
  denominator_egamma->setAxisTitle("#eta", 1);
  denominator_egamma->setAxisTitle("#phi", 2);

  ibooker.setCurrentFolder(monitorDir_+"/L1TTau/timing");
  for(unsigned int i=0; i<bxrange_; ++i) {
    tau_eta_phi.push_back(ibooker.book2D("tau_eta_phi_bx_"+bx_obj[i],"L1T Tau #eta vs #phi BX="+bx_obj[i], 30, -3., 3., 25, -3.2, 3.2));
    tau_eta_phi.at(i)->setAxisTitle("#eta", 1);
    tau_eta_phi.at(i)->setAxisTitle("#phi", 2);
  }
  denominator_tau = ibooker.book2D("denominator_tau","Denominator for L1TTau", 30, -3., 3., 25, -3.2, 3.2);
  denominator_tau->setAxisTitle("#eta", 1);
  denominator_tau->setAxisTitle("#phi", 2);

  ibooker.setCurrentFolder(monitorDir_+"/L1TEtSum/timing");
  for(unsigned int i=0; i<bxrange_; ++i) {
    etsum_eta_phi_MET.push_back(ibooker.book1D("etsum_phi_bx_MET_"+bx_obj[i],"L1T MET #phi BX="+bx_obj[i],25, -3.2, 3.2));
    etsum_eta_phi_MET.at(i)->setAxisTitle("#phi", 1);
    etsum_eta_phi_METHF.push_back(ibooker.book1D("etsum_phi_bx_METHF_"+bx_obj[i],"L1T METHF #phi BX="+bx_obj[i],25, -3.2, 3.2));
    etsum_eta_phi_METHF.at(i)->setAxisTitle("#phi", 1);
    etsum_eta_phi_MHT.push_back(ibooker.book1D("etsum_phi_bx_MHT_"+bx_obj[i],"L1T MHT #phi BX="+bx_obj[i],25, -3.2, 3.2));
    etsum_eta_phi_MHT.at(i)->setAxisTitle("#phi", 1);
    etsum_eta_phi_MHTHF.push_back(ibooker.book1D("etsum_phi_bx_MHTHF_"+bx_obj[i],"L1T MHTHF #phi BX="+bx_obj[i],25, -3.2, 3.2));
    etsum_eta_phi_MHTHF.at(i)->setAxisTitle("#phi", 1);
  }
  denominator_etsum_MET = ibooker.book1D("denominator_etsum_MET","Denominator for L1TEtSum MET", 25, -3.2, 3.2);
  denominator_etsum_MET->setAxisTitle("#phi", 1);
  denominator_etsum_METHF = ibooker.book1D("denominator_etsum_METHF","Denominator for L1TEtSum METHF", 25, -3.2, 3.2);
  denominator_etsum_METHF->setAxisTitle("#phi", 1);
  denominator_etsum_MHT = ibooker.book1D("denominator_etsum_MHT","Denominator for L1TEtSum MHT", 25, -3.2, 3.2);
  denominator_etsum_MHT->setAxisTitle("#phi", 1);
  denominator_etsum_MHTHF = ibooker.book1D("denominator_etsum_MHTHF","Denominator for L1TEtSum MHTHF", 25, -3.2, 3.2);
  denominator_etsum_MHTHF->setAxisTitle("#phi", 1);

  // isolated bunch
  if(algoBitIsoBx_ > -1) {
    ibooker.setCurrentFolder(monitorDir_+"/L1TMuon/timing/Isolated_bunch");  
    for(unsigned int i=0; i<bxrange_; ++i) {
      muons_eta_phi_isolated.push_back(ibooker.book2D("muons_eta_phi_bx_isolated_"+bx_obj[i],"L1T Muon #eta vs #phi for isolated bunch BX="+bx_obj[i],25, -2.5, 2.5, 25, -3.2, 3.2));
      muons_eta_phi_isolated.at(i)->setAxisTitle("#eta", 1);
      muons_eta_phi_isolated.at(i)->setAxisTitle("#phi", 2);
    }
    denominator_muons_isolated = ibooker.book2D("denominator_muons_isolated","Denominator for Isolated Bunch for L1TMuon",25, -2.5, 2.5, 25, -3.2, 3.2);
    denominator_muons_isolated->setAxisTitle("#eta", 1);
    denominator_muons_isolated->setAxisTitle("#phi", 2);
 
    ibooker.setCurrentFolder(monitorDir_+"/L1TJet/timing/Isolated_bunch");
    for(unsigned int i=0; i<bxrange_; ++i) {
      jet_eta_phi_isolated.push_back(ibooker.book2D("jet_eta_phi_bx_isolated_"+bx_obj[i],"L1T Jet #eta vs #phi for isolated bunch BX="+bx_obj[i], 50, -5., 5., 25, -3.2, 3.2));
      jet_eta_phi_isolated.at(i)->setAxisTitle("#eta", 1);
      jet_eta_phi_isolated.at(i)->setAxisTitle("#phi", 2);
    }
    denominator_jet_isolated = ibooker.book2D("denominator_jet_isolated","Denominator for Isolated Bunch for L1TJet", 50, -5., 5., 25, -3.2, 3.2);
    denominator_jet_isolated->setAxisTitle("#eta", 1);
    denominator_jet_isolated->setAxisTitle("#phi", 2);
 
    for (const auto egammaPtCut : egammaPtCuts_) {
      ibooker.setCurrentFolder(monitorDir_+"/L1TEGamma/timing/Isolated_bunch/ptmin_"+std::to_string(egammaPtCut)+"_gev");
      std::vector<MonitorElement*> vHelper;
      for(unsigned int i=0; i<bxrange_; ++i) {
        vHelper.push_back(ibooker.book2D("egamma_eta_phi_bx_isolated_"+bx_obj[i], "L1T EGamma p_{T}#geq"+std::to_string(egammaPtCut)+"_GeV #eta vs #phi for first bunch BX="+bx_obj[i], 30, -3., 3., 25, -3.2, 3.2));
        vHelper.back()->setAxisTitle("#eta", 1);
        vHelper.back()->setAxisTitle("#phi", 2);
      }
      egamma_eta_phi_isolated.push_back(vHelper);

      denominator_egamma_isolated.push_back(ibooker.book2D("denominator_egamma_isolated", "Denominator for Isolated Bunch for L1TEGamma p_{T}#geq"+std::to_string(egammaPtCut)+"_GeV", 30, -3., 3., 25, -3.2, 3.2));
      denominator_egamma_isolated.back()->setAxisTitle("#eta", 1);
      denominator_egamma_isolated.back()->setAxisTitle("#phi", 2);

      egamma_iso_bx_ieta_isolated.push_back(ibooker.book2D("egamma_iso_bx_ieta_isolated_ptmin"+std::to_string(egammaPtCut), "L1T EGamma iso with pT#geq"+std::to_string(egammaPtCut)+" GeV BX vs. i#eta for first bunch in train", 5, -2.5, 2.5, 70, -70, 70));
      egamma_iso_bx_ieta_isolated.back()->setAxisTitle("BX in event", 1);
      egamma_iso_bx_ieta_isolated.back()->setAxisTitle("i#eta", 2);

      egamma_noniso_bx_ieta_isolated.push_back(ibooker.book2D("egamma_noniso_bx_ieta_isolated_ptmin"+std::to_string(egammaPtCut), "L1T EGamma non iso with pT#geq"+std::to_string(egammaPtCut)+" GeV BX vs. i#eta for first bunch in train", 5, -2.5, 2.5, 70, -70, 70));
      egamma_noniso_bx_ieta_isolated.back()->setAxisTitle("BX in event", 1);
      egamma_noniso_bx_ieta_isolated.back()->setAxisTitle("i#eta", 2);
    }

    ibooker.setCurrentFolder(monitorDir_+"/L1TTau/timing/Isolated_bunch");
    for(unsigned int i=0; i<bxrange_; ++i) {
      tau_eta_phi_isolated.push_back(ibooker.book2D("tau_eta_phi_bx_isolated_"+bx_obj[i],"L1T Tau #eta vs #phi for isolated bunch BX="+bx_obj[i], 30, -3., 3., 25, -3.2, 3.2));
      tau_eta_phi_isolated.at(i)->setAxisTitle("#eta", 1);
      tau_eta_phi_isolated.at(i)->setAxisTitle("#phi", 2);
    }
    denominator_tau_isolated = ibooker.book2D("denominator_tau_isolated","Denominator for Isolated Bunch for L1TTau", 30, -3., 3., 25, -3.2, 3.2);
    denominator_tau_isolated->setAxisTitle("#eta", 1);
    denominator_tau_isolated->setAxisTitle("#phi", 2);
 
    ibooker.setCurrentFolder(monitorDir_+"/L1TEtSum/timing/Isolated_bunch");
    for(unsigned int i=0; i<bxrange_; ++i) {
      etsum_eta_phi_MET_isolated.push_back(ibooker.book1D("etsum_phi_bx_MET_isolated_"+bx_obj[i],"L1T MET #phi for isolated bunch BX="+bx_obj[i],25, -3.2, 3.2));
      etsum_eta_phi_MET_isolated.at(i)->setAxisTitle("#phi", 1);
      etsum_eta_phi_METHF_isolated.push_back(ibooker.book1D("etsum_phi_bx_METHF_isolated_"+bx_obj[i],"L1T METHF #phi for isolated bunch BX="+bx_obj[i],25, -3.2, 3.2));
      etsum_eta_phi_METHF_isolated.at(i)->setAxisTitle("#phi", 1);
      etsum_eta_phi_MHT_isolated.push_back(ibooker.book1D("etsum_phi_bx_MHT_isolated_"+bx_obj[i],"L1T MHT #phi for isolated bunch BX="+bx_obj[i],25, -3.2, 3.2));
      etsum_eta_phi_MHT_isolated.at(i)->setAxisTitle("#phi", 1);
      etsum_eta_phi_MHTHF_isolated.push_back(ibooker.book1D("etsum_phi_bx_MHTHF_isolated_"+bx_obj[i],"L1T MHTHF #phi for isolated bunch BX="+bx_obj[i],25, -3.2, 3.2));
      etsum_eta_phi_MHTHF_isolated.at(i)->setAxisTitle("#phi", 1);
    }
    denominator_etsum_isolated_MET = ibooker.book1D("denominator_etsum_isolated_MET","Denominator for Isolated Bunch for L1TEtSum MET", 25, -3.2, 3.2);
    denominator_etsum_isolated_MET->setAxisTitle("#phi", 1);
    denominator_etsum_isolated_METHF = ibooker.book1D("denominator_etsum_isolated_METHF","Denominator for Isolated Bunch for L1TEtSum METHF", 25, -3.2, 3.2); 
    denominator_etsum_isolated_METHF->setAxisTitle("#phi", 1);
    denominator_etsum_isolated_MHT = ibooker.book1D("denominator_etsum_isolated_MHT","Denominator for Isolated Bunch for L1TEtSum MHT", 25, -3.2, 3.2);
    denominator_etsum_isolated_MHT->setAxisTitle("#phi", 1);
    denominator_etsum_isolated_MHTHF = ibooker.book1D("denominator_etsum_isolated_MHTHF","Denominator for Isolated Bunch for L1TEtSum MHTHF", 25, -3.2, 3.2);
    denominator_etsum_isolated_MHTHF->setAxisTitle("#phi", 1);
   }

  if(algoBitFirstBxInTrain_ > -1) {
    // first bunch minus 2
    ibooker.setCurrentFolder(monitorDir_+"/L1TMuon"+"/timing"+"/First_bunch_minus_2");
    denominator_muons_firstbunchminus2 = ibooker.book2D("denominator_muons_firstbunchminus2","Denominator for first bunch minus 2 for L1TMuon",25, -2.5, 2.5, 25, -3.2, 3.2);
    denominator_muons_firstbunchminus2->setAxisTitle("#eta", 1);
    denominator_muons_firstbunchminus2->setAxisTitle("#phi", 2);
    ibooker.setCurrentFolder(monitorDir_+"/L1TJet"+"/timing"+"/First_bunch_minus_2");
    denominator_jet_firstbunchminus2 = ibooker.book2D("denominator_jet_firstbunchminus2","Denominator for first bunch minus 2 for L1TJet", 50, -5., 5., 25, -3.2, 3.2);
    denominator_jet_firstbunchminus2->setAxisTitle("#eta", 1);
    denominator_jet_firstbunchminus2->setAxisTitle("#phi", 2);
    ibooker.setCurrentFolder(monitorDir_+"/L1TEGamma"+"/timing"+"/First_bunch_minus_2");
    denominator_egamma_firstbunchminus2 = ibooker.book2D("denominator_egamma_firstbunchminus2","Denominator for first bunch minus 2 for L1TEGamma", 30, -3., 3., 25, -3.2, 3.2);
    denominator_egamma_firstbunchminus2->setAxisTitle("#eta", 1);
    denominator_egamma_firstbunchminus2->setAxisTitle("#phi", 2);
    ibooker.setCurrentFolder(monitorDir_+"/L1TTau"+"/timing"+"/First_bunch_minus_2");
    denominator_tau_firstbunchminus2 = ibooker.book2D("denominator_tau_firstbunchminus2","Denominator for first bunch minus 2 for L1TTau", 30, -3., 3., 25, -3.2, 3.2);
    denominator_tau_firstbunchminus2->setAxisTitle("#eta", 1);
    denominator_tau_firstbunchminus2->setAxisTitle("#phi", 2);
    ibooker.setCurrentFolder(monitorDir_+"/L1TEtSum"+"/timing"+"/First_bunch_minus_2");
    denominator_etsum_firstbunchminus2_MET = ibooker.book1D("denominator_etsum_firstbunchminus2_MET","Denominator for first bunch minus 2 for L1TEtSum MET", 25, -3.2, 3.2);
    denominator_etsum_firstbunchminus2_MET->setAxisTitle("#phi", 1);
    denominator_etsum_firstbunchminus2_METHF = ibooker.book1D("denominator_etsum_firstbunchminus2_METHF","Denominator for first bunch minus 2 for L1TEtSum METHF", 25, -3.2, 3.2);
    denominator_etsum_firstbunchminus2_METHF->setAxisTitle("#phi", 1);
    denominator_etsum_firstbunchminus2_MHT = ibooker.book1D("denominator_etsum_firstbunchminus2_MHT","Denominator for first bunch minus 2 for L1TEtSum MHT", 25, -3.2, 3.2);
    denominator_etsum_firstbunchminus2_MHT->setAxisTitle("#phi", 1);
    denominator_etsum_firstbunchminus2_MHTHF = ibooker.book1D("denominator_etsum_firstbunchminus2_MHTHF","Denominator for first bunch minus 2 for L1TEtSum MHTHF", 25, -3.2, 3.2);
    denominator_etsum_firstbunchminus2_MHTHF->setAxisTitle("#phi", 1);
 
    ibooker.setCurrentFolder(monitorDir_+"/L1TMuon"+"/timing"+"/First_bunch_minus_2");
    for(unsigned int i=0; i<bxrange_; ++i) {
      muons_eta_phi_firstbunchminus2.push_back(ibooker.book2D("muons_eta_phi_bx_firstbunchminus2_"+bx_obj[i],"L1T Muon #eta vs #phi for first bunch minus 2 BX="+bx_obj[i],25, -2.5, 2.5, 25, -3.2, 3.2));
      muons_eta_phi_firstbunchminus2.at(i)->setAxisTitle("#eta", 1);
      muons_eta_phi_firstbunchminus2.at(i)->setAxisTitle("#phi", 2);
    }
 
    ibooker.setCurrentFolder(monitorDir_+"/L1TJet"+"/timing"+"/First_bunch_minus_2");
    for(unsigned int i=0; i<bxrange_; ++i) {
      jet_eta_phi_firstbunchminus2.push_back(ibooker.book2D("jet_eta_phi_bx_firstbunchminus2_"+bx_obj[i],"L1T Jet #eta vs #phi for first bunch minus 2 BX="+bx_obj[i], 50, -5., 5., 25, -3.2, 3.2));
      jet_eta_phi_firstbunchminus2.at(i)->setAxisTitle("#eta", 1);
      jet_eta_phi_firstbunchminus2.at(i)->setAxisTitle("#phi", 2);
    }
 
    ibooker.setCurrentFolder(monitorDir_+"/L1TEGamma"+"/timing"+"/First_bunch_minus_2");
    for(unsigned int i=0; i<bxrange_; ++i) {
      egamma_eta_phi_firstbunchminus2.push_back(ibooker.book2D("egamma_eta_phi_bx_firstbunchminus2_"+bx_obj[i],"L1T EGamma #eta vs #phi for first bunch minus 2 BX="+bx_obj[i], 30, -3., 3., 25, -3.2, 3.2));
      egamma_eta_phi_firstbunchminus2.at(i)->setAxisTitle("#eta", 1);
      egamma_eta_phi_firstbunchminus2.at(i)->setAxisTitle("#phi", 2);
    }

    ibooker.setCurrentFolder(monitorDir_+"/L1TTau"+"/timing"+"/First_bunch_minus_2");
    for(unsigned int i=0; i<bxrange_; ++i) {
      tau_eta_phi_firstbunchminus2.push_back(ibooker.book2D("tau_eta_phi_bx_firstbunchminus2_"+bx_obj[i],"L1T Tau #eta vs #phi for first bunch minus 2 BX="+bx_obj[i], 30, -3., 3., 25, -3.2, 3.2));
      tau_eta_phi_firstbunchminus2.at(i)->setAxisTitle("#eta", 1);
      tau_eta_phi_firstbunchminus2.at(i)->setAxisTitle("#phi", 2);
    }

    ibooker.setCurrentFolder(monitorDir_+"/L1TEtSum"+"/timing"+"/First_bunch_minus_2");
    for(unsigned int i=0; i<bxrange_; ++i) {
      etsum_eta_phi_MET_firstbunchminus2.push_back(ibooker.book1D("etsum_phi_bx_MET_firstbunchminus2_"+bx_obj[i],"L1T MET #phi for first bunch minus 2 BX="+bx_obj[i],25, -3.2, 3.2));
      etsum_eta_phi_MET_firstbunchminus2.at(i)->setAxisTitle("#phi", 1);
      etsum_eta_phi_METHF_firstbunchminus2.push_back(ibooker.book1D("etsum_phi_bx_METHF_firstbunchminus2_"+bx_obj[i],"L1T METHF #phi for first bunch minus 2 BX="+bx_obj[i],25, -3.2, 3.2));
      etsum_eta_phi_METHF_firstbunchminus2.at(i)->setAxisTitle("#phi", 1);
      etsum_eta_phi_MHT_firstbunchminus2.push_back(ibooker.book1D("etsum_phi_bx_MHT_firstbunchminus2_"+bx_obj[i],"L1T MHT #phi for first bunch minus 2 BX="+bx_obj[i],25, -3.2, 3.2));
      etsum_eta_phi_MHT_firstbunchminus2.at(i)->setAxisTitle("#phi", 1);
      etsum_eta_phi_MHTHF_firstbunchminus2.push_back(ibooker.book1D("etsum_phi_bx_MHTHF_firstbunchminus2_"+bx_obj[i],"L1T MHTHF #phi for first bunch minus 2 BX="+bx_obj[i],25, -3.2, 3.2));
      etsum_eta_phi_MHTHF_firstbunchminus2.at(i)->setAxisTitle("#phi", 1);
    }

    // first bunch minus 1
    ibooker.setCurrentFolder(monitorDir_+"/L1TMuon"+"/timing"+"/First_bunch_minus_1");
    denominator_muons_firstbunchminus1 = ibooker.book2D("denominator_muons_firstbunchminus1","Denominator for first bunch minus 1 for L1TMuon",25, -2.5, 2.5, 25, -3.2, 3.2);
    denominator_muons_firstbunchminus1->setAxisTitle("#eta", 1);
    denominator_muons_firstbunchminus1->setAxisTitle("#phi", 2);
    ibooker.setCurrentFolder(monitorDir_+"/L1TJet"+"/timing"+"/First_bunch_minus_1");
    denominator_jet_firstbunchminus1 = ibooker.book2D("denominator_jet_firstbunchminus1","Denominator for first bunch minus 1 for L1TJet", 50, -5., 5., 25, -3.2, 3.2);
    denominator_jet_firstbunchminus1->setAxisTitle("#eta", 1);
    denominator_jet_firstbunchminus1->setAxisTitle("#phi", 2);
    ibooker.setCurrentFolder(monitorDir_+"/L1TEGamma"+"/timing"+"/First_bunch_minus_1");
    denominator_egamma_firstbunchminus1 = ibooker.book2D("denominator_egamma_firstbunchminus1","Denominator for first bunch minus 1 for L1TEGamma", 30, -3., 3., 25, -3.2, 3.2);
    denominator_egamma_firstbunchminus1->setAxisTitle("#eta", 1);
    denominator_egamma_firstbunchminus1->setAxisTitle("#phi", 2);
    ibooker.setCurrentFolder(monitorDir_+"/L1TTau"+"/timing"+"/First_bunch_minus_1");
    denominator_tau_firstbunchminus1 = ibooker.book2D("denominator_tau_firstbunchminus1","Denominator for first bunch minus 1 for L1TTau", 30, -3., 3., 25, -3.2, 3.2);
    denominator_tau_firstbunchminus1->setAxisTitle("#eta", 1);
    denominator_tau_firstbunchminus1->setAxisTitle("#phi", 2);
    ibooker.setCurrentFolder(monitorDir_+"/L1TEtSum"+"/timing"+"/First_bunch_minus_1");
    denominator_etsum_firstbunchminus1_MET = ibooker.book1D("denominator_etsum_firstbunchminus1_MET","Denominator for first bunch minus 1 for L1TEtSum MET", 25, -3.2, 3.2);
    denominator_etsum_firstbunchminus1_MET->setAxisTitle("#phi", 1);
    denominator_etsum_firstbunchminus1_METHF = ibooker.book1D("denominator_etsum_firstbunchminus1_METHF","Denominator for first bunch minus 1 for L1TEtSum METHF", 25, -3.2, 3.2);
    denominator_etsum_firstbunchminus1_METHF->setAxisTitle("#phi", 1);
    denominator_etsum_firstbunchminus1_MHT = ibooker.book1D("denominator_etsum_firstbunchminus1_MHT","Denominator for first bunch minus 1 for L1TEtSum MHT", 25, -3.2, 3.2);
    denominator_etsum_firstbunchminus1_MHT->setAxisTitle("#phi", 1);
    denominator_etsum_firstbunchminus1_MHTHF = ibooker.book1D("denominator_etsum_firstbunchminus1_MHTHF","Denominator for first bunch minus 1 for L1TEtSum MHTHF", 25, -3.2, 3.2);
    denominator_etsum_firstbunchminus1_MHTHF->setAxisTitle("#phi", 1);
 
    ibooker.setCurrentFolder(monitorDir_+"/L1TMuon"+"/timing"+"/First_bunch_minus_1");
    for(unsigned int i=0; i<bxrange_; ++i) {
      muons_eta_phi_firstbunchminus1.push_back(ibooker.book2D("muons_eta_phi_bx_firstbunchminus1_"+bx_obj[i],"L1T Muon #eta vs #phi for first bunch minus 1 BX="+bx_obj[i],25, -2.5, 2.5, 25, -3.2, 3.2));
      muons_eta_phi_firstbunchminus1.at(i)->setAxisTitle("#eta", 1);
      muons_eta_phi_firstbunchminus1.at(i)->setAxisTitle("#phi", 2);
    }
 
    ibooker.setCurrentFolder(monitorDir_+"/L1TJet"+"/timing"+"/First_bunch_minus_1");
    for(unsigned int i=0; i<bxrange_; ++i) {
      jet_eta_phi_firstbunchminus1.push_back(ibooker.book2D("jet_eta_phi_bx_firstbunchminus1_"+bx_obj[i],"L1T Jet #eta vs #phi for first bunch minus 1 BX="+bx_obj[i], 50, -5., 5., 25, -3.2, 3.2));
      jet_eta_phi_firstbunchminus1.at(i)->setAxisTitle("#eta", 1);
      jet_eta_phi_firstbunchminus1.at(i)->setAxisTitle("#phi", 2);
    }
 
    ibooker.setCurrentFolder(monitorDir_+"/L1TEGamma"+"/timing"+"/First_bunch_minus_1");
    for(unsigned int i=0; i<bxrange_; ++i) {
      egamma_eta_phi_firstbunchminus1.push_back(ibooker.book2D("egamma_eta_phi_bx_firstbunchminus1_"+bx_obj[i],"L1T EGamma #eta vs #phi for first bunch minus 1 BX="+bx_obj[i], 30, -3., 3., 25, -3.2, 3.2));
      egamma_eta_phi_firstbunchminus1.at(i)->setAxisTitle("#eta", 1);
      egamma_eta_phi_firstbunchminus1.at(i)->setAxisTitle("#phi", 2);
    }

    ibooker.setCurrentFolder(monitorDir_+"/L1TTau"+"/timing"+"/First_bunch_minus_1");
    for(unsigned int i=0; i<bxrange_; ++i) {
      tau_eta_phi_firstbunchminus1.push_back(ibooker.book2D("tau_eta_phi_bx_firstbunchminus1_"+bx_obj[i],"L1T Tau #eta vs #phi for first bunch minus 1 BX="+bx_obj[i], 30, -3., 3., 25, -3.2, 3.2));
      tau_eta_phi_firstbunchminus1.at(i)->setAxisTitle("#eta", 1);
      tau_eta_phi_firstbunchminus1.at(i)->setAxisTitle("#phi", 2);
    }

    ibooker.setCurrentFolder(monitorDir_+"/L1TEtSum"+"/timing"+"/First_bunch_minus_1");
    for(unsigned int i=0; i<bxrange_; ++i) {
      etsum_eta_phi_MET_firstbunchminus1.push_back(ibooker.book1D("etsum_phi_bx_MET_firstbunchminus1_"+bx_obj[i],"L1T MET #phi for first bunch minus 1 BX="+bx_obj[i],25, -3.2, 3.2));
      etsum_eta_phi_MET_firstbunchminus1.at(i)->setAxisTitle("#phi", 1);
      etsum_eta_phi_METHF_firstbunchminus1.push_back(ibooker.book1D("etsum_phi_bx_METHF_firstbunchminus1_"+bx_obj[i],"L1T METHF #phi for first bunch minus 1 BX="+bx_obj[i],25, -3.2, 3.2));
      etsum_eta_phi_METHF_firstbunchminus1.at(i)->setAxisTitle("#phi", 1);
      etsum_eta_phi_MHT_firstbunchminus1.push_back(ibooker.book1D("etsum_phi_bx_MHT_firstbunchminus1_"+bx_obj[i],"L1T MHT #phi for first bunch minus 1 BX="+bx_obj[i],25, -3.2, 3.2));
      etsum_eta_phi_MHT_firstbunchminus1.at(i)->setAxisTitle("#phi", 1);
      etsum_eta_phi_MHTHF_firstbunchminus1.push_back(ibooker.book1D("etsum_phi_bx_MHTHF_firstbunchminus1_"+bx_obj[i],"L1T MHTHF #phi for first bunch minus 1 BX="+bx_obj[i],25, -3.2, 3.2));
      etsum_eta_phi_MHTHF_firstbunchminus1.at(i)->setAxisTitle("#phi", 1);
    }

    // first bunch in train
    ibooker.setCurrentFolder(monitorDir_+"/L1TMuon/timing/First_bunch");
    for(unsigned int i=0; i<bxrange_-2; ++i) {
      muons_eta_phi_firstbunch.push_back(ibooker.book2D("muons_eta_phi_bx_firstbunch_"+bx_obj[i],"L1T Muon #eta vs #phi for first bunch BX="+bx_obj[i],25, -2.5, 2.5, 25, -3.2, 3.2));
      muons_eta_phi_firstbunch.at(i)->setAxisTitle("#eta", 1);
      muons_eta_phi_firstbunch.at(i)->setAxisTitle("#phi", 2);
    }
    denominator_muons_firstbunch = ibooker.book2D("denominator_muons_firstbunch","Denominator for First Bunch for L1TMuon",25, -2.5, 2.5, 25, -3.2, 3.2);
    denominator_muons_firstbunch->setAxisTitle("#eta", 1);
    denominator_muons_firstbunch->setAxisTitle("#phi", 2);
 
    ibooker.setCurrentFolder(monitorDir_+"/L1TJet/timing/First_bunch");
    for(unsigned int i=0; i<bxrange_-2; ++i) {
      jet_eta_phi_firstbunch.push_back(ibooker.book2D("jet_eta_phi_bx_firstbunch_"+bx_obj[i],"L1T Jet #eta vs #phi for first bunch BX="+bx_obj[i], 50, -5., 5., 25, -3.2, 3.2));
      jet_eta_phi_firstbunch.at(i)->setAxisTitle("#eta", 1);
      jet_eta_phi_firstbunch.at(i)->setAxisTitle("#phi", 2);
    }
    denominator_jet_firstbunch = ibooker.book2D("denominator_jet_firstbunch","Denominator for First Bunch for L1TJet", 50, -5., 5., 25, -3.2, 3.2);
    denominator_jet_firstbunch->setAxisTitle("#eta", 1);
    denominator_jet_firstbunch->setAxisTitle("#phi", 2);
 
    for (const auto egammaPtCut : egammaPtCuts_) {
      ibooker.setCurrentFolder(monitorDir_+"/L1TEGamma/timing/First_bunch/ptmin_"+std::to_string(egammaPtCut)+"_gev");
      std::vector<MonitorElement*> vHelper;
      for(unsigned int i=0; i<bxrange_-2; ++i) {
        vHelper.push_back(ibooker.book2D("egamma_eta_phi_bx_firstbunch_"+bx_obj[i], "L1T EGamma p_{T}#geq"+std::to_string(egammaPtCut)+"_GeV #eta vs #phi for first bunch BX="+bx_obj[i], 30, -3., 3., 25, -3.2, 3.2));
        vHelper.back()->setAxisTitle("#eta", 1);
        vHelper.back()->setAxisTitle("#phi", 2);
      }
      egamma_eta_phi_firstbunch.push_back(vHelper);

      denominator_egamma_firstbunch.push_back(ibooker.book2D("denominator_egamma_firstbunch", "Denominator for First Bunch for L1TEGamma p_{T}#geq"+std::to_string(egammaPtCut)+"_GeV", 30, -3., 3., 25, -3.2, 3.2));
      denominator_egamma_firstbunch.back()->setAxisTitle("#eta", 1);
      denominator_egamma_firstbunch.back()->setAxisTitle("#phi", 2);

      egamma_iso_bx_ieta_firstbunch.push_back(ibooker.book2D("egamma_iso_bx_ieta_firstbunch_ptmin"+std::to_string(egammaPtCut), "L1T EGamma iso with pT#geq"+std::to_string(egammaPtCut)+" GeV BX vs. i#eta for first bunch in train", 5, -2.5, 2.5, 70, -70, 70));
      egamma_iso_bx_ieta_firstbunch.back()->setAxisTitle("BX in event", 1);
      egamma_iso_bx_ieta_firstbunch.back()->setAxisTitle("i#eta", 2);

      egamma_noniso_bx_ieta_firstbunch.push_back(ibooker.book2D("egamma_noniso_bx_ieta_firstbunch_ptmin"+std::to_string(egammaPtCut), "L1T EGamma non iso with pT#geq"+std::to_string(egammaPtCut)+" GeV BX vs. i#eta for first bunch in train", 5, -2.5, 2.5, 70, -70, 70));
      egamma_noniso_bx_ieta_firstbunch.back()->setAxisTitle("BX in event", 1);
      egamma_noniso_bx_ieta_firstbunch.back()->setAxisTitle("i#eta", 2);
    }

    ibooker.setCurrentFolder(monitorDir_+"/L1TTau/timing/First_bunch");
    for(unsigned int i=0; i<bxrange_-2; ++i) {
      tau_eta_phi_firstbunch.push_back(ibooker.book2D("tau_eta_phi_bx_firstbunch_"+bx_obj[i],"L1T Tau #eta vs #phi for first bunch BX="+bx_obj[i], 30, -3., 3., 25, -3.2, 3.2));
      tau_eta_phi_firstbunch.at(i)->setAxisTitle("#eta", 1);
      tau_eta_phi_firstbunch.at(i)->setAxisTitle("#phi", 2);
    }
    denominator_tau_firstbunch = ibooker.book2D("denominator_tau_firstbunch","Denominator for First Bunch for L1TTau", 30, -3., 3., 25, -3.2, 3.2);
    denominator_tau_firstbunch->setAxisTitle("#eta", 1);
    denominator_tau_firstbunch->setAxisTitle("#phi", 2);

    ibooker.setCurrentFolder(monitorDir_+"/L1TEtSum/timing/First_bunch");
    for(unsigned int i=0; i<bxrange_-2; ++i) {
      etsum_eta_phi_MET_firstbunch.push_back(ibooker.book1D("etsum_phi_bx_MET_firstbunch_"+bx_obj[i],"L1T MET #phi for firstbunch bunch BX="+bx_obj[i],25, -3.2, 3.2));
      etsum_eta_phi_MET_firstbunch.at(i)->setAxisTitle("#phi", 1);
      etsum_eta_phi_METHF_firstbunch.push_back(ibooker.book1D("etsum_phi_bx_METHF_firstbunch_"+bx_obj[i],"L1T METHF #phi for firstbunch bunch BX="+bx_obj[i],25, -3.2, 3.2));
      etsum_eta_phi_METHF_firstbunch.at(i)->setAxisTitle("#phi", 1);
      etsum_eta_phi_MHT_firstbunch.push_back(ibooker.book1D("etsum_phi_bx_MHT_firstbunch_"+bx_obj[i],"L1T MHT #phi for firstbunch bunch BX="+bx_obj[i],25, -3.2, 3.2));
      etsum_eta_phi_MHT_firstbunch.at(i)->setAxisTitle("#phi", 1);
      etsum_eta_phi_MHTHF_firstbunch.push_back(ibooker.book1D("etsum_phi_bx_MHTHF_firstbunch_"+bx_obj[i],"L1T MHTHF #phi for firstbunch bunch BX="+bx_obj[i],25, -3.2, 3.2));
      etsum_eta_phi_MHTHF_firstbunch.at(i)->setAxisTitle("#phi", 1);
    }
    denominator_etsum_firstbunch_MET = ibooker.book1D("denominator_etsum_firstbunch_MET","Denominator for First Bunch for L1TEtSum MET", 25, -3.2, 3.2);
    denominator_etsum_firstbunch_MET->setAxisTitle("#phi", 1);
    denominator_etsum_firstbunch_METHF = ibooker.book1D("denominator_etsum_firstbunch_METHF","Denominator for First Bunch for L1TEtSum METHF", 25, -3.2, 3.2);
    denominator_etsum_firstbunch_METHF->setAxisTitle("#phi", 1);
    denominator_etsum_firstbunch_MHT = ibooker.book1D("denominator_etsum_firstbunch_MHT","Denominator for First Bunch for L1TEtSum MHT", 25, -3.2, 3.2);
    denominator_etsum_firstbunch_MHT->setAxisTitle("#phi", 1);
    denominator_etsum_firstbunch_MHTHF = ibooker.book1D("denominator_etsum_firstbunch_MHTHF","Denominator for First Bunch for L1TEtSum MHTHF", 25, -3.2, 3.2);
    denominator_etsum_firstbunch_MHTHF->setAxisTitle("#phi", 1);
   } 
 
  // last bunch in train
  if(algoBitLastBxInTrain_ > -1) {
    ibooker.setCurrentFolder(monitorDir_+"/L1TMuon/timing/Last_bunch");
    for(unsigned int i=0; i<bxrange_-2; ++i) {
      muons_eta_phi_lastbunch.push_back(ibooker.book2D("muons_eta_phi_bx_lastbunch_"+bx_obj[i+2],"L1T Muon #eta vs #phi for last bunch BX="+bx_obj[i+2],25, -2.5, 2.5, 25, -3.2, 3.2));
      muons_eta_phi_lastbunch.at(i)->setAxisTitle("#eta", 1);
      muons_eta_phi_lastbunch.at(i)->setAxisTitle("#phi", 2);
    }
    denominator_muons_lastbunch = ibooker.book2D("denominator_muons_lastbunch","Denominator for Last Bunch for L1TMuon",25, -2.5, 2.5, 25, -3.2, 3.2);
    denominator_muons_lastbunch->setAxisTitle("#eta", 1);
    denominator_muons_lastbunch->setAxisTitle("#phi", 2);

    ibooker.setCurrentFolder(monitorDir_+"/L1TJet/timing/Last_bunch");
    for(unsigned int i=0; i<bxrange_-2; ++i) {
      jet_eta_phi_lastbunch.push_back(ibooker.book2D("jet_eta_phi_bx_lastbunch_"+bx_obj[i+2],"L1T Jet #eta vs #phi for last bunch BX="+bx_obj[i+2], 50, -5., 5., 25, -3.2, 3.2));
      jet_eta_phi_lastbunch.at(i)->setAxisTitle("#eta", 1);
      jet_eta_phi_lastbunch.at(i)->setAxisTitle("#phi", 2);
    }
    denominator_jet_lastbunch = ibooker.book2D("denominator_jet_lastbunch","Denominator for Last Bunch for L1TJet", 50, -5., 5., 25, -3.2, 3.2);
    denominator_jet_lastbunch->setAxisTitle("#eta", 1);
    denominator_jet_lastbunch->setAxisTitle("#phi", 2);

    for (const auto egammaPtCut : egammaPtCuts_) {
      ibooker.setCurrentFolder(monitorDir_+"/L1TEGamma/timing/Last_bunch/ptmin_"+std::to_string(egammaPtCut)+"_gev");
      std::vector<MonitorElement*> vHelper;
      for(unsigned int i=0; i<bxrange_-2; ++i) {
        vHelper.push_back(ibooker.book2D("egamma_eta_phi_bx_lastbunch_"+bx_obj[i+2], "L1T EGamma p_{T}#geq"+std::to_string(egammaPtCut)+"_GeV #eta vs #phi for first bunch BX="+bx_obj[i+2], 30, -3., 3., 25, -3.2, 3.2));
        vHelper.back()->setAxisTitle("#eta", 1);
        vHelper.back()->setAxisTitle("#phi", 2);
      }
      egamma_eta_phi_lastbunch.push_back(vHelper);

      denominator_egamma_lastbunch.push_back(ibooker.book2D("denominator_egamma_lastbunch", "Denominator for Last Bunch for L1TEGamma p_{T}#geq"+std::to_string(egammaPtCut)+"_GeV", 30, -3., 3., 25, -3.2, 3.2));
      denominator_egamma_lastbunch.back()->setAxisTitle("#eta", 1);
      denominator_egamma_lastbunch.back()->setAxisTitle("#phi", 2);

      egamma_iso_bx_ieta_lastbunch.push_back(ibooker.book2D("egamma_iso_bx_ieta_lastbunch_ptmin"+std::to_string(egammaPtCut), "L1T EGamma iso with pT#geq"+std::to_string(egammaPtCut)+" GeV BX vs. i#eta for first bunch in train", 5, -2.5, 2.5, 70, -70, 70));
      egamma_iso_bx_ieta_lastbunch.back()->setAxisTitle("BX in event", 1);
      egamma_iso_bx_ieta_lastbunch.back()->setAxisTitle("i#eta", 2);

      egamma_noniso_bx_ieta_lastbunch.push_back(ibooker.book2D("egamma_noniso_bx_ieta_lastbunch_ptmin"+std::to_string(egammaPtCut), "L1T EGamma non iso with pT#geq"+std::to_string(egammaPtCut)+" GeV BX vs. i#eta for first bunch in train", 5, -2.5, 2.5, 70, -70, 70));
      egamma_noniso_bx_ieta_lastbunch.back()->setAxisTitle("BX in event", 1);
      egamma_noniso_bx_ieta_lastbunch.back()->setAxisTitle("i#eta", 2);
    }

    ibooker.setCurrentFolder(monitorDir_+"/L1TTau/timing/Last_bunch");
    for(unsigned int i=0; i<bxrange_-2; ++i) {
      tau_eta_phi_lastbunch.push_back(ibooker.book2D("tau_eta_phi_bx_lastbunch_"+bx_obj[i+2],"L1T Tau #eta vs #phi for last bunch BX="+bx_obj[i+2], 30, -3., 3., 25, -3.2, 3.2));
      tau_eta_phi_lastbunch.at(i)->setAxisTitle("#eta", 1);
      tau_eta_phi_lastbunch.at(i)->setAxisTitle("#phi", 2);
    }
    denominator_tau_lastbunch = ibooker.book2D("denominator_tau_lastbunch","Denominator for Last Bunch for L1TTau", 30, -3., 3., 25, -3.2, 3.2);
    denominator_tau_lastbunch->setAxisTitle("#eta", 1);
    denominator_tau_lastbunch->setAxisTitle("#phi", 2);
 
    ibooker.setCurrentFolder(monitorDir_+"/L1TEtSum/timing/Last_bunch");
    for(unsigned int i=0; i<bxrange_-2; ++i) {
      etsum_eta_phi_MET_lastbunch.push_back(ibooker.book1D("etsum_phi_bx_MET_lastbunch_"+bx_obj[i+2],"L1T MET #phi for lastbunch bunch BX="+bx_obj[i+2],25, -3.2, 3.2));
      etsum_eta_phi_MET_lastbunch.at(i)->setAxisTitle("#phi", 1);
      etsum_eta_phi_METHF_lastbunch.push_back(ibooker.book1D("etsum_phi_bx_METHF_lastbunch_"+bx_obj[i+2],"L1T METHF #phi for lastbunch bunch BX="+bx_obj[i+2],25, -3.2, 3.2));
      etsum_eta_phi_METHF_lastbunch.at(i)->setAxisTitle("#phi", 1);
      etsum_eta_phi_MHT_lastbunch.push_back(ibooker.book1D("etsum_phi_bx_MHT_lastbunch_"+bx_obj[i+2],"L1T MHT #phi for lastbunch bunch BX="+bx_obj[i+2],25, -3.2, 3.2));
      etsum_eta_phi_MHT_lastbunch.at(i)->setAxisTitle("#phi", 1);
      etsum_eta_phi_MHTHF_lastbunch.push_back(ibooker.book1D("etsum_phi_bx_MHTHF_lastbunch_"+bx_obj[i+2],"L1T MHTHF #phi for lastbunch bunch BX="+bx_obj[i+2],25, -3.2, 3.2));
      etsum_eta_phi_MHTHF_lastbunch.at(i)->setAxisTitle("#phi", 1);
    }
    denominator_etsum_lastbunch_MET = ibooker.book1D("denominator_etsum_lastbunch_MET","Denominator for Last Bunch for L1TEtSum MET", 25, -3.2, 3.2);
    denominator_etsum_lastbunch_MET->setAxisTitle("#phi", 1);
    denominator_etsum_lastbunch_METHF = ibooker.book1D("denominator_etsum_lastbunch_METHF","Denominator for Last Bunch for L1TEtSum METHF", 25, -3.2, 3.2);
    denominator_etsum_lastbunch_METHF->setAxisTitle("#phi", 1);
    denominator_etsum_lastbunch_MHT = ibooker.book1D("denominator_etsum_lastbunch_MHT","Denominator for Last Bunch for L1TEtSum MHT", 25, -3.2, 3.2);
    denominator_etsum_lastbunch_MHT->setAxisTitle("#phi", 1);
    denominator_etsum_lastbunch_MHTHF = ibooker.book1D("denominator_etsum_lastbunch_MHTHF","Denominator for Last Bunch for L1TEtSum MHTHF", 25, -3.2, 3.2);
    denominator_etsum_lastbunch_MHTHF->setAxisTitle("#phi", 1);
  } 
}

void L1TObjectsTiming::analyze(const edm::Event& e, const edm::EventSetup& c) {

  if (verbose_) edm::LogInfo("L1TObjectsTiming") << "L1TObjectsTiming: analyze..." << std::endl;

  // Muon Collection
  edm::Handle<l1t::MuonBxCollection> MuonBxCollection;
  e.getByToken(ugmtMuonToken_, MuonBxCollection);
  // Jet Collection
  edm::Handle<l1t::JetBxCollection> JetBxCollection;
  e.getByToken(stage2CaloLayer2JetToken_, JetBxCollection);
  // EGamma Collection
  edm::Handle<l1t::EGammaBxCollection> EGammaBxCollection;
  e.getByToken(stage2CaloLayer2EGammaToken_, EGammaBxCollection);
  // Tau Collection
  edm::Handle<l1t::TauBxCollection> TauBxCollection;
  e.getByToken(stage2CaloLayer2TauToken_, TauBxCollection);
  // EtSum Collection
  edm::Handle<l1t::EtSumBxCollection> EtSumBxCollection;
  e.getByToken(stage2CaloLayer2EtSumToken_, EtSumBxCollection);

  // Open uGT readout record
  edm::Handle<GlobalAlgBlkBxCollection> uGtAlgs;
  e.getByToken(l1tStage2uGtProducer_, uGtAlgs);
          
  // Filling eta-phi map for muons for BX=-2,-1,0,+1,+2
  for (int itBX = MuonBxCollection->getFirstBX(); itBX <= MuonBxCollection->getLastBX(); ++itBX) {
    for (l1t::MuonBxCollection::const_iterator Muon = MuonBxCollection->begin(itBX); Muon != MuonBxCollection->end(itBX); ++Muon) {
      if (Muon->pt() >= muonPtCut_ and Muon->hwQual() >= muonQualCut_) {
        denominator_muons->Fill(Muon->eta(), Muon->phi());
        int index = (int)itBX - std::min(0, 1 - (int)bxrange_%2 - (int)std::floor(bxrange_/2.)); // the correlation from itBX to respective index of the vector
        muons_eta_phi.at(index)->Fill(Muon->eta(), Muon->phi());
      }
    }
  }
    
  // Filling eta-phi map for jets for BX=-2,-1,0,+1,+2
  for (int itBX = JetBxCollection->getFirstBX(); itBX <= JetBxCollection->getLastBX(); ++itBX) {
    for (l1t::JetBxCollection::const_iterator jet = JetBxCollection->begin(itBX); jet != JetBxCollection->end(itBX); ++jet) {
      if (jet->pt() >= jetPtCut_) {
        denominator_jet->Fill(jet->eta(), jet->phi());
        int index = itBX - std::min(0, 1 - (int)bxrange_%2 - (int)std::floor(bxrange_/2.)); // the correlation from itBX to respective index of the vector
        jet_eta_phi.at(index)->Fill(jet->eta(), jet->phi());
      }
    }
  }

  // Filling eta-phi map for egamma for BX=-2,-1,0,+1,+2
  for (int itBX = EGammaBxCollection->getFirstBX(); itBX <= EGammaBxCollection->getLastBX(); ++itBX) {
    for (l1t::EGammaBxCollection::const_iterator egamma = EGammaBxCollection->begin(itBX); egamma != EGammaBxCollection->end(itBX); ++egamma) {
      if (egamma->pt() >= egammaPtCut_) {
        denominator_egamma->Fill(egamma->eta(), egamma->phi());
        int index = itBX - std::min(0, 1 - (int)bxrange_%2 - (int)std::floor(bxrange_/2.)); // the correlation from itBX to respective index of the vector
        egamma_eta_phi.at(index)->Fill(egamma->eta(), egamma->phi());
      }
    }
  }

  // Filling eta-phi map for tau for BX=-2,-1,0,+1,+2
  for (int itBX = TauBxCollection->getFirstBX(); itBX <= TauBxCollection->getLastBX(); ++itBX) {
    for (l1t::TauBxCollection::const_iterator tau = TauBxCollection->begin(itBX); tau != TauBxCollection->end(itBX); ++tau) {
      if (tau->pt() >= tauPtCut_) {
        denominator_tau->Fill(tau->eta(), tau->phi());
        int index = itBX - std::min(0, 1 - (int)bxrange_%2 - (int)std::floor(bxrange_/2.)); // the correlation from itBX to respective index of the vector
        tau_eta_phi.at(index)->Fill(tau->eta(), tau->phi());
      }
    }
  }

  // Filling eta-phi map for etsum for BX=-2,-1,0,+1,+2
  for (int itBX = EtSumBxCollection->getFirstBX(); itBX <= EtSumBxCollection->getLastBX(); ++itBX) {
    for (l1t::EtSumBxCollection::const_iterator EtSum = EtSumBxCollection->begin(itBX); EtSum != EtSumBxCollection->end(itBX); ++EtSum) {
      if (EtSum->pt() >= etsumPtCut_) {
        int index = itBX - std::min(0, 1 - (int)bxrange_%2 - (int)std::floor(bxrange_/2.)); // the correlation from itBX to respective index of the vector
        if (l1t::EtSum::EtSumType::kMissingEt == EtSum->getType()) {
          etsum_eta_phi_MET.at(index)->Fill(EtSum->phi());
          denominator_etsum_MET->Fill(EtSum->phi());
        }
        else if (l1t::EtSum::EtSumType::kMissingEtHF == EtSum->getType()) {
          etsum_eta_phi_METHF.at(index)->Fill(EtSum->phi());
          denominator_etsum_METHF->Fill(EtSum->phi());
          
        }
        else if(l1t::EtSum::EtSumType::kMissingHt == EtSum->getType()) {
          etsum_eta_phi_MHT.at(index)->Fill(EtSum->phi());
          denominator_etsum_MHT->Fill(EtSum->phi());
          
        }
        else if(l1t::EtSum::EtSumType::kMissingHtHF == EtSum->getType()) {
          etsum_eta_phi_MHTHF.at(index)->Fill(EtSum->phi());
          denominator_etsum_MHTHF->Fill(EtSum->phi());
        }
      }
    }
  }

  // Find out in which BX the first collision in train, isolated bunch, and last collision in train have fired.
  // In case of pre firing it will be in BX 1 or BX 2 and this will determine the BX shift that
  // will be applied to the timing histogram later.
  int bxShiftFirst = -999;
  int bxShiftIso = -999;
  int bxShiftLast = -999;
  for (int bx = uGtAlgs->getFirstBX(); bx <= uGtAlgs->getLastBX(); ++bx) {
    for (GlobalAlgBlkBxCollection::const_iterator itr = uGtAlgs->begin(bx); itr != uGtAlgs->end(bx); ++itr) {
      // first bunch in train
      if (algoBitFirstBxInTrain_ != -1) {
        bool bit = false;
        switch (useAlgoDecision_) {
          case 0:
            bit = itr->getAlgoDecisionInitial(algoBitFirstBxInTrain_);
            break;
          case 1:
            bit = itr->getAlgoDecisionInterm(algoBitFirstBxInTrain_);
            break;
          case 2:
            bit = itr->getAlgoDecisionFinal(algoBitFirstBxInTrain_);
            break;
        }
        if (bit) {
          bxShiftFirst = bx;
        }
      }
      // last bunch in train
      if(algoBitLastBxInTrain_ != -1) {
        bool bit = false;
        switch (useAlgoDecision_) {
          case 0:
            bit = itr->getAlgoDecisionInitial(algoBitLastBxInTrain_);
            break;
          case 1:
            bit = itr->getAlgoDecisionInterm(algoBitLastBxInTrain_);
            break;
          case 2:
            bit = itr->getAlgoDecisionFinal(algoBitLastBxInTrain_);
            break;
        }
        if (bit) {
          bxShiftLast = bx;
        }
      }
      // isolated bunch
      if (algoBitIsoBx_ != -1) {
        bool bit = false;
        switch (useAlgoDecision_) {
          case 0:
            bit = itr->getAlgoDecisionInitial(algoBitIsoBx_);
            break;
          case 1:
            bit = itr->getAlgoDecisionInterm(algoBitIsoBx_);
            break;
          case 2:
            bit = itr->getAlgoDecisionFinal(algoBitIsoBx_);
            break;
        }
        if (bit) {
          bxShiftIso = bx;
        }
      }
    }
  }

  // fill the first bunch in train maps
  if (bxShiftFirst > -999) {
    // muons
    for (int itBX = std::max(MuonBxCollection->getFirstBX(), MuonBxCollection->getFirstBX() + bxShiftFirst); itBX <= std::min(MuonBxCollection->getLastBX(), MuonBxCollection->getLastBX() + bxShiftFirst); ++itBX) {
      int index = itBX - bxShiftFirst - uGtAlgs->getFirstBX();
      if (index >= 0 and index < (int)muons_eta_phi_firstbunch.size()) {
        for (l1t::MuonBxCollection::const_iterator muon = MuonBxCollection->begin(itBX); muon != MuonBxCollection->end(itBX); ++muon) { // Starting with Muons
          if (muon->pt() >= muonPtCut_ and muon->hwQual() >= muonQualCut_) {
            denominator_muons_firstbunch->Fill(muon->eta(), muon->phi());
            muons_eta_phi_firstbunch.at(index)->Fill(muon->eta(), muon->phi());
          }
        }
      }
    }
    // jets
    for (int itBX = std::max(JetBxCollection->getFirstBX(), JetBxCollection->getFirstBX() + bxShiftFirst); itBX <= std::min(JetBxCollection->getLastBX(), JetBxCollection->getLastBX() + bxShiftFirst); ++itBX) {
      int index = itBX - bxShiftFirst - uGtAlgs->getFirstBX();
      if (index >= 0 and index < (int)jet_eta_phi_firstbunch.size()) {
        for (l1t::JetBxCollection::const_iterator jet = JetBxCollection->begin(itBX); jet != JetBxCollection->end(itBX); ++jet) {
          if (jet->pt() >= jetPtCut_) {
            denominator_jet_firstbunch->Fill(jet->eta(), jet->phi());
            jet_eta_phi_firstbunch.at(index)->Fill(jet->eta(), jet->phi());
          }
        }
      }
    }
    // egammas
    for (int itBX = std::max(EGammaBxCollection->getFirstBX(), EGammaBxCollection->getFirstBX() + bxShiftFirst); itBX <= std::min(EGammaBxCollection->getLastBX(), EGammaBxCollection->getLastBX() + bxShiftFirst); ++itBX) {
      int index = itBX - bxShiftFirst - uGtAlgs->getFirstBX();
      for (l1t::EGammaBxCollection::const_iterator egamma = EGammaBxCollection->begin(itBX); egamma != EGammaBxCollection->end(itBX); ++egamma) {
        for (size_t i = 0; i < egammaPtCuts_.size(); ++i) {
          if (egamma->pt() >= egammaPtCuts_.at(i)) {
            if (index >= 0 and index < (int)egamma_eta_phi_firstbunch.size()) {
              denominator_egamma_firstbunch.at(i)->Fill(egamma->eta(), egamma->phi());
              egamma_eta_phi_firstbunch.at(i).at(index)->Fill(egamma->eta(), egamma->phi());
            }
            if ((bool)egamma->hwIso()) {
              egamma_iso_bx_ieta_firstbunch.at(i)->Fill(itBX - bxShiftFirst, egamma->hwEta());
            }
            egamma_noniso_bx_ieta_firstbunch.at(i)->Fill(itBX - bxShiftFirst, egamma->hwEta());
          }
        }
      }
    }
    // taus
    for (int itBX = std::max(TauBxCollection->getFirstBX(), TauBxCollection->getFirstBX() + bxShiftFirst); itBX <= std::min(TauBxCollection->getLastBX(), TauBxCollection->getLastBX() + bxShiftFirst); ++itBX) {
      int index = itBX - bxShiftFirst - uGtAlgs->getFirstBX();
      if (index >= 0 and index < (int)tau_eta_phi_firstbunch.size()) {
        for (l1t::TauBxCollection::const_iterator tau = TauBxCollection->begin(itBX); tau != TauBxCollection->end(itBX); ++tau) {
          if (tau->pt() >= tauPtCut_) {
            denominator_tau_firstbunch->Fill(tau->eta(), tau->phi());
            tau_eta_phi_firstbunch.at(index)->Fill(tau->eta(), tau->phi());
          }
        }
      }
    }
    // etsums
    for (int itBX = std::max(EtSumBxCollection->getFirstBX(), EtSumBxCollection->getFirstBX() + bxShiftFirst); itBX <= std::min(EtSumBxCollection->getLastBX(), EtSumBxCollection->getLastBX() + bxShiftFirst); ++itBX) {
      int index = itBX - bxShiftFirst - uGtAlgs->getFirstBX();
      if (index >= 0) {
        for (l1t::EtSumBxCollection::const_iterator EtSum = EtSumBxCollection->begin(itBX); EtSum != EtSumBxCollection->end(itBX); ++EtSum) {
          if (EtSum->pt() >= etsumPtCut_) {
            if (l1t::EtSum::EtSumType::kMissingEt == EtSum->getType()) {
              if (index < (int)etsum_eta_phi_MET_firstbunch.size()) {
                denominator_etsum_firstbunch_MET->Fill(EtSum->phi());
                etsum_eta_phi_MET_firstbunch.at(index)->Fill(EtSum->phi());
              }
            }
            else if (l1t::EtSum::EtSumType::kMissingEtHF == EtSum->getType()) {
              if (index < (int)etsum_eta_phi_METHF_firstbunch.size()) {
                denominator_etsum_firstbunch_METHF->Fill(EtSum->phi());
                etsum_eta_phi_METHF_firstbunch.at(index)->Fill(EtSum->phi());
              }
            }
            else if(l1t::EtSum::EtSumType::kMissingHt == EtSum->getType()) {
              if (index < (int)etsum_eta_phi_MHT_firstbunch.size()) {
                denominator_etsum_firstbunch_MHT->Fill(EtSum->phi());
                etsum_eta_phi_MHT_firstbunch.at(index)->Fill(EtSum->phi());
              }
            }
            else if(l1t::EtSum::EtSumType::kMissingHtHF == EtSum->getType()) {
              if (index < (int)etsum_eta_phi_MHTHF_firstbunch.size()) {
                denominator_etsum_firstbunch_MHTHF->Fill(EtSum->phi());
                etsum_eta_phi_MHTHF_firstbunch.at(index)->Fill(EtSum->phi());
              }
            }
          }
        }
      }
    }
  }

  // fill the last bunch in train maps
  if (bxShiftLast > -999) {
    // muons
    for (int itBX = std::max(MuonBxCollection->getFirstBX(), MuonBxCollection->getFirstBX() + bxShiftLast); itBX <= std::min(MuonBxCollection->getLastBX(), MuonBxCollection->getLastBX() + bxShiftLast); ++itBX) {
      int index = itBX - bxShiftLast - uGtAlgs->getFirstBX();
      if (index >= 0 and index < (int)muons_eta_phi_lastbunch.size()) {
        for (l1t::MuonBxCollection::const_iterator muon = MuonBxCollection->begin(itBX); muon != MuonBxCollection->end(itBX); ++muon) { // Starting with Muons
          if (muon->pt() >= muonPtCut_ and muon->hwQual() >= muonQualCut_) {
            denominator_muons_lastbunch->Fill(muon->eta(), muon->phi());
            muons_eta_phi_lastbunch.at(index)->Fill(muon->eta(), muon->phi());
          }
        }
      }
    }
    // jets
    for (int itBX = std::max(JetBxCollection->getFirstBX(), JetBxCollection->getFirstBX() + bxShiftLast); itBX <= std::min(JetBxCollection->getLastBX(), JetBxCollection->getLastBX() + bxShiftLast); ++itBX) {
      int index = itBX - bxShiftLast - uGtAlgs->getFirstBX();
      if (index >= 0 and index < (int)jet_eta_phi_lastbunch.size()) {
        for (l1t::JetBxCollection::const_iterator jet = JetBxCollection->begin(itBX); jet != JetBxCollection->end(itBX); ++jet) {
          if (jet->pt() >= jetPtCut_) {
            denominator_jet_lastbunch->Fill(jet->eta(), jet->phi());
            jet_eta_phi_lastbunch.at(index)->Fill(jet->eta(), jet->phi());
          }
        }
      }
    }
    // egammas
    for (int itBX = std::max(EGammaBxCollection->getFirstBX(), EGammaBxCollection->getFirstBX() + bxShiftLast); itBX <= std::min(EGammaBxCollection->getLastBX(), EGammaBxCollection->getLastBX() + bxShiftLast); ++itBX) {
      int index = itBX - bxShiftLast - uGtAlgs->getFirstBX();
      for (l1t::EGammaBxCollection::const_iterator egamma = EGammaBxCollection->begin(itBX); egamma != EGammaBxCollection->end(itBX); ++egamma) {
        for (size_t i = 0; i < egammaPtCuts_.size(); ++i) {
          if (egamma->pt() >= egammaPtCuts_.at(i)) {
            if (index >= 0 and index < (int)egamma_eta_phi_lastbunch.size()) {
              denominator_egamma_lastbunch.at(i)->Fill(egamma->eta(), egamma->phi());
              egamma_eta_phi_lastbunch.at(i).at(index)->Fill(egamma->eta(), egamma->phi());
            }
            if ((bool)egamma->hwIso()) {
              egamma_iso_bx_ieta_lastbunch.at(i)->Fill(itBX - bxShiftLast, egamma->hwEta());
            }
            egamma_noniso_bx_ieta_lastbunch.at(i)->Fill(itBX - bxShiftLast, egamma->hwEta());
          }
        }
      }
    }
    // taus
    for (int itBX = std::max(TauBxCollection->getFirstBX(), TauBxCollection->getFirstBX() + bxShiftLast); itBX <= std::min(TauBxCollection->getLastBX(), TauBxCollection->getLastBX() + bxShiftLast); ++itBX) {
      int index = itBX - bxShiftLast - uGtAlgs->getFirstBX();
      if (index >= 0 and index < (int)tau_eta_phi_lastbunch.size()) {
        for (l1t::TauBxCollection::const_iterator tau = TauBxCollection->begin(itBX); tau != TauBxCollection->end(itBX); ++tau) {
          if (tau->pt() >= tauPtCut_) {
            denominator_tau_lastbunch->Fill(tau->eta(), tau->phi());
            tau_eta_phi_lastbunch.at(index)->Fill(tau->eta(), tau->phi());
          }
        }
      }
    }
    // etsums
    for (int itBX = std::max(EtSumBxCollection->getFirstBX(), EtSumBxCollection->getFirstBX() + bxShiftLast); itBX <= std::min(EtSumBxCollection->getLastBX(), EtSumBxCollection->getLastBX() + bxShiftLast); ++itBX) {
      int index = itBX - bxShiftLast - uGtAlgs->getFirstBX();
      if (index >= 0) {
        for (l1t::EtSumBxCollection::const_iterator EtSum = EtSumBxCollection->begin(itBX); EtSum != EtSumBxCollection->end(itBX); ++EtSum) {
          if (EtSum->pt() >= etsumPtCut_) {
            if (l1t::EtSum::EtSumType::kMissingEt == EtSum->getType()) {
              if (index < (int)etsum_eta_phi_MET_lastbunch.size()) {
                denominator_etsum_lastbunch_MET->Fill(EtSum->phi());
                etsum_eta_phi_MET_lastbunch.at(index)->Fill(EtSum->phi());
              }
            }
            else if (l1t::EtSum::EtSumType::kMissingEtHF == EtSum->getType()) {
              if (index < (int)etsum_eta_phi_METHF_lastbunch.size()) {
                denominator_etsum_lastbunch_METHF->Fill(EtSum->phi());
                etsum_eta_phi_METHF_lastbunch.at(index)->Fill(EtSum->phi());
              }
            }
            else if(l1t::EtSum::EtSumType::kMissingHt == EtSum->getType()) {
              if (index < (int)etsum_eta_phi_MHT_lastbunch.size()) {
                denominator_etsum_lastbunch_MHT->Fill(EtSum->phi());
                etsum_eta_phi_MHT_lastbunch.at(index)->Fill(EtSum->phi());
              }
            }
            else if(l1t::EtSum::EtSumType::kMissingHtHF == EtSum->getType()) {
              if (index < (int)etsum_eta_phi_MHTHF_lastbunch.size()) {
                denominator_etsum_lastbunch_MHTHF->Fill(EtSum->phi());
                etsum_eta_phi_MHTHF_lastbunch.at(index)->Fill(EtSum->phi());
              }
            }
          }
        }
      }
    }
  }

  // fill the isolated bunch
  if (bxShiftIso > -999) {
    // muons
    for (int itBX = std::max(MuonBxCollection->getFirstBX(), MuonBxCollection->getFirstBX() + bxShiftIso); itBX <= std::min(MuonBxCollection->getLastBX(), MuonBxCollection->getLastBX() + bxShiftIso); ++itBX) {
      int index = itBX - bxShiftIso - uGtAlgs->getFirstBX();
      if (index >= 0 and index < (int)muons_eta_phi_isolated.size()) {
        for (l1t::MuonBxCollection::const_iterator muon = MuonBxCollection->begin(itBX); muon != MuonBxCollection->end(itBX); ++muon) { // Starting with Muons
          if (muon->pt() >= muonPtCut_ and muon->hwQual() >= muonQualCut_) {
            denominator_muons_isolated->Fill(muon->eta(), muon->phi());
            muons_eta_phi_isolated.at(index)->Fill(muon->eta(), muon->phi());
          }
        }
      }
    }
    // jets
    for (int itBX = std::max(JetBxCollection->getFirstBX(), JetBxCollection->getFirstBX() + bxShiftIso); itBX <= std::min(JetBxCollection->getLastBX(), JetBxCollection->getLastBX() + bxShiftIso); ++itBX) {
      int index = itBX - bxShiftIso - uGtAlgs->getFirstBX();
      if (index >= 0 and index < (int)jet_eta_phi_isolated.size()) {
        for (l1t::JetBxCollection::const_iterator jet = JetBxCollection->begin(itBX); jet != JetBxCollection->end(itBX); ++jet) {
          if (jet->pt() >= jetPtCut_) {
            denominator_jet_isolated->Fill(jet->eta(), jet->phi());
            jet_eta_phi_isolated.at(index)->Fill(jet->eta(), jet->phi());
          }
        }
      }
    }
    // egammas
    for (int itBX = std::max(EGammaBxCollection->getFirstBX(), EGammaBxCollection->getFirstBX() + bxShiftIso); itBX <= std::min(EGammaBxCollection->getLastBX(), EGammaBxCollection->getLastBX() + bxShiftIso); ++itBX) {
      int index = itBX - bxShiftIso - uGtAlgs->getFirstBX();
      for (l1t::EGammaBxCollection::const_iterator egamma = EGammaBxCollection->begin(itBX); egamma != EGammaBxCollection->end(itBX); ++egamma) {
        for (size_t i = 0; i < egammaPtCuts_.size(); ++i) {
          if (egamma->pt() >= egammaPtCuts_.at(i)) {
            if (index >= 0 and index < (int)egamma_eta_phi_isolated.size()) {
              denominator_egamma_isolated.at(i)->Fill(egamma->eta(), egamma->phi());
              egamma_eta_phi_isolated.at(i).at(index)->Fill(egamma->eta(), egamma->phi());
            }
            if ((bool)egamma->hwIso()) {
              egamma_iso_bx_ieta_isolated.at(i)->Fill(itBX - bxShiftIso, egamma->hwEta());
            }
            egamma_noniso_bx_ieta_isolated.at(i)->Fill(itBX - bxShiftIso, egamma->hwEta());
          }
        }
      }
    }
    // taus
    for (int itBX = std::max(TauBxCollection->getFirstBX(), TauBxCollection->getFirstBX() + bxShiftIso); itBX <= std::min(TauBxCollection->getLastBX(), TauBxCollection->getLastBX() + bxShiftIso); ++itBX) {
      int index = itBX - bxShiftIso - uGtAlgs->getFirstBX();
      if (index >= 0 and index < (int)tau_eta_phi_isolated.size()) {
        for (l1t::TauBxCollection::const_iterator tau = TauBxCollection->begin(itBX); tau != TauBxCollection->end(itBX); ++tau) {
          if (tau->pt() >= tauPtCut_) {
            denominator_tau_isolated->Fill(tau->eta(), tau->phi());
            tau_eta_phi_isolated.at(index)->Fill(tau->eta(), tau->phi());
          }
        }
      }
    }
    // etsums
    for (int itBX = std::max(EtSumBxCollection->getFirstBX(), EtSumBxCollection->getFirstBX() + bxShiftIso); itBX <= std::min(EtSumBxCollection->getLastBX(), EtSumBxCollection->getLastBX() + bxShiftIso); ++itBX) {
      int index = itBX - bxShiftIso - uGtAlgs->getFirstBX();
      if (index >= 0) {
        for (l1t::EtSumBxCollection::const_iterator EtSum = EtSumBxCollection->begin(itBX); EtSum != EtSumBxCollection->end(itBX); ++EtSum) {
          if (EtSum->pt() >= etsumPtCut_) {
            if (l1t::EtSum::EtSumType::kMissingEt == EtSum->getType()) {
              if (index < (int)etsum_eta_phi_MET_isolated.size()) {
                denominator_etsum_isolated_MET->Fill(EtSum->phi());
                etsum_eta_phi_MET_isolated.at(index)->Fill(EtSum->phi());
              }
            }
            else if (l1t::EtSum::EtSumType::kMissingEtHF == EtSum->getType()) {
              if (index < (int)etsum_eta_phi_METHF_isolated.size()) {
                denominator_etsum_isolated_METHF->Fill(EtSum->phi());
                etsum_eta_phi_METHF_isolated.at(index)->Fill(EtSum->phi());
              }
            }
            else if(l1t::EtSum::EtSumType::kMissingHt == EtSum->getType()) {
              if (index < (int)etsum_eta_phi_MHT_isolated.size()) {
                denominator_etsum_isolated_MHT->Fill(EtSum->phi());
                etsum_eta_phi_MHT_isolated.at(index)->Fill(EtSum->phi());
              }
            }
            else if(l1t::EtSum::EtSumType::kMissingHtHF == EtSum->getType()) {
              if (index < (int)etsum_eta_phi_MHTHF_isolated.size()) {
                denominator_etsum_isolated_MHTHF->Fill(EtSum->phi());
                etsum_eta_phi_MHTHF_isolated.at(index)->Fill(EtSum->phi());
              }
            }
          }
        }
      }
    }
  }

  if (uGtAlgs->getLastBX() >= 2) {
    for(GlobalAlgBlkBxCollection::const_iterator itr = uGtAlgs->begin(2); itr != uGtAlgs->end(2); ++itr) { // If algoBitFirstBxInTrain_ fired in L1A BX 2 something else must have prefired in the actual BX -2 before the bunch crossing
      if(algoBitFirstBxInTrain_ != -1 && itr->getAlgoDecisionInitial(algoBitFirstBxInTrain_)) { // Filling eta-phi map for all objects for first bunch each BX
        for (int itBX = MuonBxCollection->getFirstBX(); itBX <= MuonBxCollection->getLastBX() ; ++itBX) {
          for (l1t::MuonBxCollection::const_iterator muon = MuonBxCollection->begin(itBX); muon != MuonBxCollection->end(itBX); ++muon) { // Starting with Muons
            if (muon->pt() >= muonPtCut_ and muon->hwQual() >= muonQualCut_) {
              denominator_muons_firstbunchminus2->Fill(muon->eta(),muon->phi());
              int index = itBX - std::min(0, 1 - (int)bxrange_%2 - (int)std::floor(bxrange_/2.)); // the correlation from itBX to respective index of the vector 
              muons_eta_phi_firstbunchminus2.at(index)->Fill(muon->eta(),muon->phi());
            }
          }
        }
        for (int itBX = JetBxCollection->getFirstBX(); itBX <= JetBxCollection->getLastBX(); ++itBX) {
          for (l1t::JetBxCollection::const_iterator jet = JetBxCollection->begin(itBX); jet != JetBxCollection->end(itBX); ++jet) {
            if (jet->pt() >= jetPtCut_) {
              denominator_jet_firstbunchminus2->Fill(jet->eta(), jet->phi());
              int index = itBX - std::min(0, 1 - (int)bxrange_%2 - (int)std::floor(bxrange_/2.)); // the correlation from itBX to respective index of the vector
              jet_eta_phi_firstbunchminus2.at(index)->Fill(jet->eta(), jet->phi());
            }
          }
        }
        for (int itBX = EGammaBxCollection->getFirstBX(); itBX <= EGammaBxCollection->getLastBX(); ++itBX) {
          for (l1t::EGammaBxCollection::const_iterator egamma = EGammaBxCollection->begin(itBX); egamma != EGammaBxCollection->end(itBX); ++egamma) {
            if (egamma->pt() >= egammaPtCut_) {
              denominator_egamma_firstbunchminus2->Fill(egamma->eta(), egamma->phi());
              int index = itBX - std::min(0, 1 - (int)bxrange_%2 - (int)std::floor(bxrange_/2.)); // the correlation from itBX to respective index of the vector
              egamma_eta_phi_firstbunchminus2.at(index)->Fill(egamma->eta(), egamma->phi());
            }
          }
        }
        for (int itBX = TauBxCollection->getFirstBX(); itBX <= TauBxCollection->getLastBX(); ++itBX) {
          for (l1t::TauBxCollection::const_iterator tau = TauBxCollection->begin(itBX); tau != TauBxCollection->end(itBX); ++tau) {
            if (tau->pt() >= tauPtCut_) {
              denominator_tau_firstbunchminus2->Fill(tau->eta(), tau->phi());
              int index = itBX - std::min(0, 1 - (int)bxrange_%2 - (int)std::floor(bxrange_/2.)); // the correlation from itBX to respective index of the vector
              tau_eta_phi_firstbunchminus2.at(index)->Fill(tau->eta(), tau->phi());
            }
          }
        }
        for (int itBX = EtSumBxCollection->getFirstBX(); itBX <= EtSumBxCollection->getLastBX(); ++itBX) {
          for (l1t::EtSumBxCollection::const_iterator EtSum = EtSumBxCollection->begin(itBX); EtSum != EtSumBxCollection->end(itBX); ++EtSum) {
            if (EtSum->pt() >= etsumPtCut_) {
              int index = itBX - std::min(0, 1 - (int)bxrange_%2 - (int)std::floor(bxrange_/2.)); // the correlation from itBX to respective index of the vector
              if (l1t::EtSum::EtSumType::kMissingEt == EtSum->getType()) {
                etsum_eta_phi_MET_firstbunchminus2.at(index)->Fill(EtSum->phi());
                denominator_etsum_firstbunchminus2_MET->Fill(EtSum->phi());
              }
              else if (l1t::EtSum::EtSumType::kMissingEtHF == EtSum->getType()) {
                etsum_eta_phi_METHF_firstbunchminus2.at(index)->Fill(EtSum->phi());
                denominator_etsum_firstbunchminus2_METHF->Fill(EtSum->phi());
              }
              else if(l1t::EtSum::EtSumType::kMissingHt == EtSum->getType()) {
                etsum_eta_phi_MHT_firstbunchminus2.at(index)->Fill(EtSum->phi());
                denominator_etsum_firstbunchminus2_MHT->Fill(EtSum->phi());
              }
              else if(l1t::EtSum::EtSumType::kMissingHtHF == EtSum->getType()) {
                etsum_eta_phi_MHTHF_firstbunchminus2.at(index)->Fill(EtSum->phi());
                denominator_etsum_firstbunchminus2_MHTHF->Fill(EtSum->phi());
              }
            }
          }
        } 
      }
    }
  }

  if (uGtAlgs->getLastBX() >= 1) {
    for(GlobalAlgBlkBxCollection::const_iterator itr = uGtAlgs->begin(1); itr != uGtAlgs->end(1); ++itr) { // If algoBitFirstBxInTrain_ fired in L1A BX 1 something else must have prefired in the actual BX -1 before the bunch crossing
      if(algoBitFirstBxInTrain_ != -1 && itr->getAlgoDecisionInitial(algoBitFirstBxInTrain_)) { // Filling eta-phi map for all objects for first bunch each BX
        for (int itBX = MuonBxCollection->getFirstBX(); itBX <= MuonBxCollection->getLastBX() ; ++itBX) {
          for (l1t::MuonBxCollection::const_iterator muon = MuonBxCollection->begin(itBX); muon != MuonBxCollection->end(itBX); ++muon) { // Starting with Muons
            if (muon->pt() >= muonPtCut_ and muon->hwQual() >= muonQualCut_) {
              denominator_muons_firstbunchminus1->Fill(muon->eta(),muon->phi());
              int index = itBX - std::min(0, 1 - (int)bxrange_%2 - (int)std::floor(bxrange_/2.)); // the correlation from itBX to respective index of the vector 
              muons_eta_phi_firstbunchminus1.at(index)->Fill(muon->eta(),muon->phi());
            }
          }
        }
        for (int itBX = JetBxCollection->getFirstBX(); itBX <= JetBxCollection->getLastBX(); ++itBX) {
          for (l1t::JetBxCollection::const_iterator jet = JetBxCollection->begin(itBX); jet != JetBxCollection->end(itBX); ++jet) {
            if (jet->pt() >= jetPtCut_) {
              denominator_jet_firstbunchminus1->Fill(jet->eta(), jet->phi());
              int index = itBX - std::min(0, 1 - (int)bxrange_%2 - (int)std::floor(bxrange_/2.)); // the correlation from itBX to respective index of the vector
              jet_eta_phi_firstbunchminus1.at(index)->Fill(jet->eta(), jet->phi());
            }
          }
        }
        for (int itBX = EGammaBxCollection->getFirstBX(); itBX <= EGammaBxCollection->getLastBX(); ++itBX) {
          for (l1t::EGammaBxCollection::const_iterator egamma = EGammaBxCollection->begin(itBX); egamma != EGammaBxCollection->end(itBX); ++egamma) {
            if (egamma->pt() >= egammaPtCut_) {
              denominator_egamma_firstbunchminus1->Fill(egamma->eta(), egamma->phi());
              int index = itBX - std::min(0, 1 - (int)bxrange_%2 - (int)std::floor(bxrange_/2.)); // the correlation from itBX to respective index of the vector
              egamma_eta_phi_firstbunchminus1.at(index)->Fill(egamma->eta(), egamma->phi());
            }
          }
        }
        for (int itBX = TauBxCollection->getFirstBX(); itBX <= TauBxCollection->getLastBX(); ++itBX) {
          for (l1t::TauBxCollection::const_iterator tau = TauBxCollection->begin(itBX); tau != TauBxCollection->end(itBX); ++tau) {
            if (tau->pt() >= tauPtCut_) {
              denominator_tau_firstbunchminus1->Fill(tau->eta(), tau->phi());
              int index = itBX - std::min(0, 1 - (int)bxrange_%2 - (int)std::floor(bxrange_/2.)); // the correlation from itBX to respective index of the vector
              tau_eta_phi_firstbunchminus1.at(index)->Fill(tau->eta(), tau->phi());
            }
          }
        }
        for (int itBX = EtSumBxCollection->getFirstBX(); itBX <= EtSumBxCollection->getLastBX(); ++itBX) {
          for (l1t::EtSumBxCollection::const_iterator EtSum = EtSumBxCollection->begin(itBX); EtSum != EtSumBxCollection->end(itBX); ++EtSum) {
            if (EtSum->pt() >= etsumPtCut_) {
              int index = itBX - std::min(0, 1 - (int)bxrange_%2 - (int)std::floor(bxrange_/2.)); // the correlation from itBX to respective index of the vector
              if (l1t::EtSum::EtSumType::kMissingEt == EtSum->getType()) {
                etsum_eta_phi_MET_firstbunchminus1.at(index)->Fill(EtSum->phi());
                denominator_etsum_firstbunchminus1_MET->Fill(EtSum->phi());
              }
              else if (l1t::EtSum::EtSumType::kMissingEtHF == EtSum->getType()) {
                etsum_eta_phi_METHF_firstbunchminus1.at(index)->Fill(EtSum->phi());
                denominator_etsum_firstbunchminus1_METHF->Fill(EtSum->phi());
              }
              else if(l1t::EtSum::EtSumType::kMissingHt == EtSum->getType()) {
                etsum_eta_phi_MHT_firstbunchminus1.at(index)->Fill(EtSum->phi());
                denominator_etsum_firstbunchminus1_MHT->Fill(EtSum->phi());
              }
              else if(l1t::EtSum::EtSumType::kMissingHtHF == EtSum->getType()) {
                etsum_eta_phi_MHTHF_firstbunchminus1.at(index)->Fill(EtSum->phi());
                denominator_etsum_firstbunchminus1_MHTHF->Fill(EtSum->phi());
              }
            }
          }
        }
      }
    }
  }
}

