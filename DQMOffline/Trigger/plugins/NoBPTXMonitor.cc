#include "DQMOffline/Trigger/plugins/NoBPTXMonitor.h"

// -----------------------------
//  constructors and destructor
// -----------------------------

NoBPTXMonitor::NoBPTXMonitor( const edm::ParameterSet& iConfig ) : 
  folderName_             ( iConfig.getParameter<std::string>("FolderName") )
  , jetToken_             ( mayConsume<reco::CaloJetCollection>      (iConfig.getParameter<edm::InputTag>("jets")      ) )   
  , muonToken_             ( mayConsume<reco::TrackCollection>       (iConfig.getParameter<edm::InputTag>("muons")     ) )   
  , jetE_variable_binning_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("jetEBinning") )
  , jetE_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("jetEPSet")    ) )
  , jetEta_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("jetEtaPSet")    ) )
  , jetPhi_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("jetPhiPSet")    ) )
  , muonPt_variable_binning_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("muonPtBinning") )
  , muonPt_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("muonPtPSet")    ) )
  , muonEta_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("muonEtaPSet")    ) )
  , muonPhi_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("muonPhiPSet")    ) )
  , ls_binning_           ( getHistoLSPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("lsPSet")     ) )
  , bx_binning_           ( getHistoLSPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("bxPSet")     ) )
  , num_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"),consumesCollector(), *this))
  , den_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"),consumesCollector(), *this))
  , jetSelection_ ( iConfig.getParameter<std::string>("jetSelection") )
  , muonSelection_ ( iConfig.getParameter<std::string>("muonSelection") )
  , njets_      ( iConfig.getParameter<int>("njets" )      )
  , nmuons_     ( iConfig.getParameter<int>("nmuons" )     )
{

  jetENoBPTX_.numerator   = nullptr;
  jetENoBPTX_.denominator = nullptr;
  jetENoBPTX_variableBinning_.numerator   = nullptr;
  jetENoBPTX_variableBinning_.denominator = nullptr;
  jetEVsLS_.numerator   = nullptr;
  jetEVsLS_.denominator = nullptr;
  jetEVsBX_.numerator   = nullptr;
  jetEVsBX_.denominator = nullptr;
  jetEtaNoBPTX_.numerator   = nullptr;
  jetEtaNoBPTX_.denominator = nullptr;
  jetEtaVsLS_.numerator   = nullptr;
  jetEtaVsLS_.denominator = nullptr;
  jetEtaVsBX_.numerator   = nullptr;
  jetEtaVsBX_.denominator = nullptr;
  jetPhiNoBPTX_.numerator   = nullptr;
  jetPhiNoBPTX_.denominator = nullptr;
  jetPhiVsLS_.numerator   = nullptr;
  jetPhiVsLS_.denominator = nullptr;
  jetPhiVsBX_.numerator   = nullptr;
  jetPhiVsBX_.denominator = nullptr;

  muonPtNoBPTX_.numerator   = nullptr;
  muonPtNoBPTX_.denominator = nullptr;
  muonPtNoBPTX_variableBinning_.numerator   = nullptr;
  muonPtNoBPTX_variableBinning_.denominator = nullptr;
  muonPtVsLS_.numerator   = nullptr;
  muonPtVsLS_.denominator = nullptr;
  muonPtVsBX_.numerator   = nullptr;
  muonPtVsBX_.denominator = nullptr;
  muonEtaNoBPTX_.numerator   = nullptr;
  muonEtaNoBPTX_.denominator = nullptr;
  muonEtaVsLS_.numerator   = nullptr;
  muonEtaVsLS_.denominator = nullptr;
  muonEtaVsBX_.numerator   = nullptr;
  muonEtaVsBX_.denominator = nullptr;
  muonPhiNoBPTX_.numerator   = nullptr;
  muonPhiNoBPTX_.denominator = nullptr;
  muonPhiVsLS_.numerator   = nullptr;
  muonPhiVsLS_.denominator = nullptr;
  muonPhiVsBX_.numerator   = nullptr;
  muonPhiVsBX_.denominator = nullptr;
  
}

NoBPTXMonitor::~NoBPTXMonitor()
{
  if (num_genTriggerEventFlag_) delete num_genTriggerEventFlag_;
  if (den_genTriggerEventFlag_) delete den_genTriggerEventFlag_;
}

NoBPTXbinning NoBPTXMonitor::getHistoPSet(edm::ParameterSet pset)
{
  return NoBPTXbinning{
    pset.getParameter<int32_t>("nbins"),
      pset.getParameter<double>("xmin"),
      pset.getParameter<double>("xmax"),
      };
}

NoBPTXbinning NoBPTXMonitor::getHistoLSPSet(edm::ParameterSet pset)
{
  return NoBPTXbinning{
    pset.getParameter<int32_t>("nbins"),
      0.,
      double(pset.getParameter<int32_t>("nbins"))
      };
}

void NoBPTXMonitor::setNoBPTXTitle(NoBPTXME& me, std::string titleX, std::string titleY)
{
  me.numerator->setAxisTitle(titleX,1);
  me.numerator->setAxisTitle(titleY,2);
  me.denominator->setAxisTitle(titleX,1);
  me.denominator->setAxisTitle(titleY,2);

}

void NoBPTXMonitor::bookNoBPTX(DQMStore::IBooker &ibooker, NoBPTXME& me, const std::string& histname, const std::string& histtitle, int nbins, double min, double max)
{
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, min, max);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, min, max);
}
void NoBPTXMonitor::bookNoBPTX(DQMStore::IBooker &ibooker, NoBPTXME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binning)
{
  int nbins = binning.size()-1;
  std::vector<float> fbinning(binning.begin(),binning.end());
  float* arr = &fbinning[0];
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, arr);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, arr);
}
void NoBPTXMonitor::bookNoBPTX(DQMStore::IBooker &ibooker, NoBPTXME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, double ymin, double ymax)
{
  me.numerator   = ibooker.bookProfile(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, ymin, ymax);
  me.denominator = ibooker.bookProfile(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, ymin, ymax);
}
void NoBPTXMonitor::bookNoBPTX(DQMStore::IBooker &ibooker, NoBPTXME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, int nbinsY, double ymin, double ymax)
{
  me.numerator   = ibooker.book2D(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, nbinsY, ymin, ymax);
  me.denominator = ibooker.book2D(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, nbinsY, ymin, ymax);
}
void NoBPTXMonitor::bookNoBPTX(DQMStore::IBooker &ibooker, NoBPTXME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX, const std::vector<double>& binningY)
{
  int nbinsX = binningX.size()-1;
  std::vector<float> fbinningX(binningX.begin(),binningX.end());
  float* arrX = &fbinningX[0];
  int nbinsY = binningY.size()-1;
  std::vector<float> fbinningY(binningY.begin(),binningY.end());
  float* arrY = &fbinningY[0];

  me.numerator   = ibooker.book2D(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, arrX, nbinsY, arrY);
  me.denominator = ibooker.book2D(histname+"_denominator", histtitle+" (denominator)", nbinsX, arrX, nbinsY, arrY);
}

void NoBPTXMonitor::bookHistograms(DQMStore::IBooker     & ibooker,
				 edm::Run const        & iRun,
				 edm::EventSetup const & iSetup) 
{  
  
  std::string histname, histtitle;

  std::string currentFolder = folderName_ ;
  ibooker.setCurrentFolder(currentFolder.c_str());

  histname = "jetE"; histtitle = "jetE";
  bookNoBPTX(ibooker,jetENoBPTX_,histname,histtitle,jetE_binning_.nbins,jetE_binning_.xmin, jetE_binning_.xmax);
  setNoBPTXTitle(jetENoBPTX_,"Jet E [GeV]","Events / [GeV]");

  histname = "jetE_variable"; histtitle = "jetE";
  bookNoBPTX(ibooker,jetENoBPTX_variableBinning_,histname,histtitle,jetE_variable_binning_);
  setNoBPTXTitle(jetENoBPTX_variableBinning_,"Jet E [GeV]","Events / [GeV]");

  histname = "jetEVsLS"; histtitle = "jetE vs LS";
  bookNoBPTX(ibooker,jetEVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,jetE_binning_.xmin, jetE_binning_.xmax);
  setNoBPTXTitle(jetEVsLS_,"LS","Jet E [GeV]");

  histname = "jetEVsBX"; histtitle = "jetE vs BX";
  bookNoBPTX(ibooker,jetEVsBX_,histname,histtitle,bx_binning_.nbins, bx_binning_.xmin, bx_binning_.xmax,jetE_binning_.xmin, jetE_binning_.xmax);
  setNoBPTXTitle(jetEVsBX_,"BX","Jet E [GeV]");

  histname = "jetEta"; histtitle = "jetEta";
  bookNoBPTX(ibooker,jetEtaNoBPTX_,histname,histtitle,jetEta_binning_.nbins,jetEta_binning_.xmin, jetEta_binning_.xmax);
  setNoBPTXTitle(jetEtaNoBPTX_,"Jet #eta","Events");

  histname = "jetEtaVsLS"; histtitle = "jetEta vs LS";
  bookNoBPTX(ibooker,jetEtaVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,jetEta_binning_.xmin, jetEta_binning_.xmax);
  setNoBPTXTitle(jetEtaVsLS_,"LS","Jet #eta");

  histname = "jetEtaVsBX"; histtitle = "jetEta vs BX";
  bookNoBPTX(ibooker,jetEtaVsBX_,histname,histtitle,bx_binning_.nbins, bx_binning_.xmin, bx_binning_.xmax,jetEta_binning_.xmin, jetEta_binning_.xmax);
  setNoBPTXTitle(jetEtaVsBX_,"BX","Jet #eta");

  histname = "jetPhi"; histtitle = "jetPhi";
  bookNoBPTX(ibooker,jetPhiNoBPTX_,histname,histtitle,jetPhi_binning_.nbins,jetPhi_binning_.xmin, jetPhi_binning_.xmax);
  setNoBPTXTitle(jetPhiNoBPTX_,"Jet #phi","Events");

  histname = "jetPhiVsLS"; histtitle = "jetPhi vs LS";
  bookNoBPTX(ibooker,jetPhiVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,jetPhi_binning_.xmin, jetPhi_binning_.xmax);
  setNoBPTXTitle(jetPhiVsLS_,"LS","Jet #phi");

  histname = "jetPhiVsBX"; histtitle = "jetPhi vs BX";
  bookNoBPTX(ibooker,jetPhiVsBX_,histname,histtitle,bx_binning_.nbins, bx_binning_.xmin, bx_binning_.xmax,jetPhi_binning_.xmin, jetPhi_binning_.xmax);
  setNoBPTXTitle(jetPhiVsBX_,"BX","Jet #phi");

  histname = "muonPt"; histtitle = "muonPt";
  bookNoBPTX(ibooker,muonPtNoBPTX_,histname,histtitle,muonPt_binning_.nbins,muonPt_binning_.xmin, muonPt_binning_.xmax);
  setNoBPTXTitle(muonPtNoBPTX_,"DisplacedStandAlone Muon p_{T} [GeV]","Events / [GeV]");

  histname = "muonPt_variable"; histtitle = "muonPt";
  bookNoBPTX(ibooker,muonPtNoBPTX_variableBinning_,histname,histtitle,muonPt_variable_binning_);
  setNoBPTXTitle(muonPtNoBPTX_variableBinning_,"DisplacedStandAlone Muon p_{T} [GeV]","Events / [GeV]");

  histname = "muonPtVsLS"; histtitle = "muonPt vs LS";
  bookNoBPTX(ibooker,muonPtVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,muonPt_binning_.xmin, muonPt_binning_.xmax);
  setNoBPTXTitle(muonPtVsLS_,"LS","DisplacedStandAlone Muon p_{T} [GeV]");

  histname = "muonPtVsBX"; histtitle = "muonPt vs BX";
  bookNoBPTX(ibooker,muonPtVsBX_,histname,histtitle,bx_binning_.nbins, bx_binning_.xmin, bx_binning_.xmax,muonPt_binning_.xmin, muonPt_binning_.xmax);
  setNoBPTXTitle(muonPtVsBX_,"BX","DisplacedStandAlone Muon p_{T} [GeV]");

  histname = "muonEta"; histtitle = "muonEta";
  bookNoBPTX(ibooker,muonEtaNoBPTX_,histname,histtitle,muonEta_binning_.nbins,muonEta_binning_.xmin, muonEta_binning_.xmax);
  setNoBPTXTitle(muonEtaNoBPTX_,"DisplacedStandAlone Muon #eta","Events");

  histname = "muonEtaVsLS"; histtitle = "muonEta vs LS";
  bookNoBPTX(ibooker,muonEtaVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,muonEta_binning_.xmin, muonEta_binning_.xmax);
  setNoBPTXTitle(muonEtaVsLS_,"LS","DisplacedStandAlone Muon #eta");

  histname = "muonEtaVsBX"; histtitle = "muonEta vs BX";
  bookNoBPTX(ibooker,muonEtaVsBX_,histname,histtitle,bx_binning_.nbins, bx_binning_.xmin, bx_binning_.xmax,muonEta_binning_.xmin, muonEta_binning_.xmax);
  setNoBPTXTitle(muonEtaVsBX_,"BX","DisplacedStandAlone Muon #eta");

  histname = "muonPhi"; histtitle = "muonPhi";
  bookNoBPTX(ibooker,muonPhiNoBPTX_,histname,histtitle,muonPhi_binning_.nbins,muonPhi_binning_.xmin, muonPhi_binning_.xmax);
  setNoBPTXTitle(muonPhiNoBPTX_,"DisplacedStandAlone Muon #phi","Events");

  histname = "muonPhiVsLS"; histtitle = "muonPhi vs LS";
  bookNoBPTX(ibooker,muonPhiVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,muonPhi_binning_.xmin, muonPhi_binning_.xmax);
  setNoBPTXTitle(muonPhiVsLS_,"LS","DisplacedStandAlone Muon #phi");

  histname = "muonPhiVsBX"; histtitle = "muonPhi vs BX";
  bookNoBPTX(ibooker,muonPhiVsBX_,histname,histtitle,bx_binning_.nbins, bx_binning_.xmin, bx_binning_.xmax,muonPhi_binning_.xmin, muonPhi_binning_.xmax);
  setNoBPTXTitle(muonPhiVsBX_,"BX","DisplacedStandAlone Muon #phi");

  // Initialize the GenericTriggerEventFlag
  if ( num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on() ) num_genTriggerEventFlag_->initRun( iRun, iSetup );
  if ( den_genTriggerEventFlag_ && den_genTriggerEventFlag_->on() ) den_genTriggerEventFlag_->initRun( iRun, iSetup );

}

void NoBPTXMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)  {

  // Filter out events if Trigger Filtering is requested
  if (den_genTriggerEventFlag_->on() && ! den_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  int ls = iEvent.id().luminosityBlock();
  int bx = iEvent.bunchCrossing();

  edm::Handle<reco::CaloJetCollection> jetHandle;
  iEvent.getByToken( jetToken_, jetHandle );
  std::vector<reco::CaloJet> jets;
  if ( int(jetHandle->size()) < njets_ ) return;
  for ( auto const & j : *jetHandle ) {
    if ( jetSelection_( j ) ) jets.push_back(j);
  }
  if ( int(jets.size()) < njets_ ) return;
  double jetE = -999;
  double jetEta = -999;
  double jetPhi = -999;
  if(jets.size()>0){
    jetE = jets[0].energy();
    jetEta = jets[0].eta();
    jetPhi = jets[0].phi();
  }

  edm::Handle<reco::TrackCollection> DSAHandle;
  iEvent.getByToken( muonToken_, DSAHandle );
  if ( int(DSAHandle->size()) < nmuons_ ) return;
  std::vector<reco::Track> muons;
  for ( auto const & m : *DSAHandle ) {
    if ( muonSelection_( m ) ) muons.push_back(m);
  }
  if ( int(muons.size()) < nmuons_ ) return;
  double muonPt = -999;
  double muonEta = -999;
  double muonPhi = -999;
  if(muons.size()>0){
    muonPt = muons[0].pt();
    muonPt = muons[0].eta();
    muonPt = muons[0].phi();
  }

  // filling histograms (denominator)  
  jetENoBPTX_.denominator -> Fill(jetE);
  jetENoBPTX_variableBinning_.denominator -> Fill(jetE);
  jetEVsLS_.denominator -> Fill(ls, jetE);
  jetEVsBX_.denominator -> Fill(bx, jetE);
  jetEtaNoBPTX_.denominator -> Fill(jetEta);
  jetEtaVsLS_.denominator -> Fill(ls, jetEta);
  jetEtaVsBX_.denominator -> Fill(bx, jetEta);
  jetPhiNoBPTX_.denominator -> Fill(jetPhi);
  jetPhiVsLS_.denominator -> Fill(ls, jetPhi);
  jetPhiVsBX_.denominator -> Fill(bx, jetPhi);
  muonPtNoBPTX_.denominator -> Fill(muonPt);
  muonPtNoBPTX_variableBinning_.denominator -> Fill(muonPt);
  muonPtVsLS_.denominator -> Fill(ls, muonPt);
  muonPtVsBX_.denominator -> Fill(bx, muonPt);
  muonEtaNoBPTX_.denominator -> Fill(muonEta);
  muonEtaVsLS_.denominator -> Fill(ls, muonEta);
  muonEtaVsBX_.denominator -> Fill(bx, muonEta);
  muonPhiNoBPTX_.denominator -> Fill(muonPhi);
  muonPhiVsLS_.denominator -> Fill(ls, muonPhi);
  muonPhiVsBX_.denominator -> Fill(bx, muonPhi);

  // filling histograms (numerator)  
  if (num_genTriggerEventFlag_->on() && ! num_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;
  jetENoBPTX_.numerator -> Fill(jetE);
  jetENoBPTX_variableBinning_.numerator -> Fill(jetE);
  jetEVsLS_.numerator -> Fill(ls, jetE);
  jetEVsBX_.numerator -> Fill(bx, jetE);
  jetEtaNoBPTX_.numerator -> Fill(jetEta);
  jetEtaVsLS_.numerator -> Fill(ls, jetEta);
  jetEtaVsBX_.numerator -> Fill(bx, jetEta);
  jetPhiNoBPTX_.numerator -> Fill(jetPhi);
  jetPhiVsLS_.numerator -> Fill(ls, jetPhi);
  jetPhiVsBX_.numerator -> Fill(bx, jetPhi);
  muonPtNoBPTX_.numerator -> Fill(muonPt);
  muonPtNoBPTX_variableBinning_.numerator -> Fill(muonPt);
  muonPtVsLS_.numerator -> Fill(ls, muonPt);
  muonPtVsBX_.numerator -> Fill(bx, muonPt);
  muonEtaNoBPTX_.numerator -> Fill(muonEta);
  muonEtaVsLS_.numerator -> Fill(ls, muonEta);
  muonEtaVsBX_.numerator -> Fill(bx, muonEta);
  muonPhiNoBPTX_.numerator -> Fill(muonPhi);
  muonPhiVsLS_.numerator -> Fill(ls, muonPhi);
  muonPhiVsBX_.numerator -> Fill(bx, muonPhi);

}

void NoBPTXMonitor::fillHistoPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<int>   ( "nbins", 200);
  pset.add<double>( "xmin", -0.5 );
  pset.add<double>( "xmax", 19999.5 );
}

void NoBPTXMonitor::fillHistoLSPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<int>   ( "nbins", 2500);
}

void NoBPTXMonitor::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<std::string>  ( "FolderName", "HLT/NoBPTX" );

  desc.add<edm::InputTag>( "jets",     edm::InputTag("ak4CaloJets") );
  desc.add<edm::InputTag>( "muons",    edm::InputTag("displacedStandAloneMuons") );
  desc.add<std::string>("jetSelection", "pt > 0");
  desc.add<std::string>("muonSelection", "pt > 0");
  desc.add<int>("njets",      0);
  desc.add<int>("nmuons",     0);

  edm::ParameterSetDescription genericTriggerEventPSet;
  genericTriggerEventPSet.add<bool>("andOr");
  genericTriggerEventPSet.add<edm::InputTag>("dcsInputTag", edm::InputTag("scalersRawToDigi") );
  genericTriggerEventPSet.add<std::vector<int> >("dcsPartitions",{});
  genericTriggerEventPSet.add<bool>("andOrDcs", false);
  genericTriggerEventPSet.add<bool>("errorReplyDcs", true);
  genericTriggerEventPSet.add<std::string>("dbLabel","");
  genericTriggerEventPSet.add<bool>("andOrHlt", true);
  genericTriggerEventPSet.add<edm::InputTag>("hltInputTag", edm::InputTag("TriggerResults::HLT") );
  genericTriggerEventPSet.add<std::vector<std::string> >("hltPaths",{});
  genericTriggerEventPSet.add<std::string>("hltDBKey","");
  genericTriggerEventPSet.add<bool>("errorReplyHlt",false);
  genericTriggerEventPSet.add<unsigned int>("verbosityLevel",1);

  desc.add<edm::ParameterSetDescription>("numGenericTriggerEventPSet", genericTriggerEventPSet);
  desc.add<edm::ParameterSetDescription>("denGenericTriggerEventPSet", genericTriggerEventPSet);

  edm::ParameterSetDescription histoPSet;
  edm::ParameterSetDescription jetEPSet;
  edm::ParameterSetDescription jetEtaPSet;
  edm::ParameterSetDescription jetPhiPSet;
  edm::ParameterSetDescription muonPtPSet;
  edm::ParameterSetDescription muonEtaPSet;
  edm::ParameterSetDescription muonPhiPSet;
  edm::ParameterSetDescription lsPSet;
  edm::ParameterSetDescription bxPSet;
  fillHistoPSetDescription(jetEPSet);
  fillHistoPSetDescription(jetEtaPSet);
  fillHistoPSetDescription(jetPhiPSet);
  fillHistoPSetDescription(muonPtPSet);
  fillHistoPSetDescription(muonEtaPSet);
  fillHistoPSetDescription(muonPhiPSet);
  fillHistoLSPSetDescription(lsPSet);
  fillHistoLSPSetDescription(bxPSet);
  histoPSet.add<edm::ParameterSetDescription>("jetEPSet", jetEPSet);
  histoPSet.add<edm::ParameterSetDescription>("jetEtaPSet", jetEtaPSet);
  histoPSet.add<edm::ParameterSetDescription>("jetPhiPSet", jetPhiPSet);
  histoPSet.add<edm::ParameterSetDescription>("muonPtPSet", muonPtPSet);
  histoPSet.add<edm::ParameterSetDescription>("muonEtaPSet", muonEtaPSet);
  histoPSet.add<edm::ParameterSetDescription>("muonPhiPSet", muonPhiPSet);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("bxPSet", bxPSet);
  std::vector<double> bins = {0.,20.,40.,60.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.,190.,200.,220.,240.,260.,280.,300.,350.,400.,450.,1000.};
  histoPSet.add<std::vector<double> >("jetEBinning", bins);
  histoPSet.add<std::vector<double> >("muonPtBinning", bins);

  desc.add<edm::ParameterSetDescription>("histoPSet",histoPSet);

  descriptions.add("NoBPTXMonitoring", desc);
}

// Define this as a plug-in
DEFINE_FWK_MODULE(NoBPTXMonitor);
