#include "DQMOffline/Trigger/plugins/NoBPTXMonitor.h"

// -----------------------------
//  constructors and destructor
// -----------------------------

NoBPTXMonitor::NoBPTXMonitor( const edm::ParameterSet& iConfig ) :
  folderName_             ( iConfig.getParameter<std::string>("FolderName") )
  , jetToken_             ( consumes<reco::CaloJetCollection>      (iConfig.getParameter<edm::InputTag>("jets")      ) )
  , muonToken_             ( consumes<reco::TrackCollection>       (iConfig.getParameter<edm::InputTag>("muons")     ) )
  , jetE_variable_binning_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("jetEBinning") )
  , jetE_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("jetEPSet")    ) )
  , jetEta_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("jetEtaPSet")    ) )
  , jetPhi_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("jetPhiPSet")    ) )
  , muonPt_variable_binning_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("muonPtBinning") )
  , muonPt_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("muonPtPSet")    ) )
  , muonEta_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("muonEtaPSet")    ) )
  , muonPhi_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("muonPhiPSet")    ) )
  , ls_binning_           ( getHistoPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("lsPSet")     ) )
  , bx_binning_           ( getHistoLSPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("bxPSet")     ) )
  , num_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"),consumesCollector(), *this))
  , den_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"),consumesCollector(), *this))
  , jetSelection_ ( iConfig.getParameter<std::string>("jetSelection") )
  , muonSelection_ ( iConfig.getParameter<std::string>("muonSelection") )
  , njets_      ( iConfig.getParameter<unsigned int>("njets" )      )
  , nmuons_     ( iConfig.getParameter<unsigned int>("nmuons" )     )
{

  jetENoBPTX_.numerator   = nullptr;
  jetENoBPTX_.denominator = nullptr;
  jetENoBPTX_variableBinning_.numerator   = nullptr;
  jetENoBPTX_variableBinning_.denominator = nullptr;
  jetEVsLS_.numerator   = nullptr;
  jetEVsLS_.denominator = nullptr;
  jetEVsBX_.numerator   = nullptr;
  jetEtaNoBPTX_.numerator   = nullptr;
  jetEtaNoBPTX_.denominator = nullptr;
  jetEtaVsLS_.numerator   = nullptr;
  jetEtaVsBX_.numerator   = nullptr;
  jetPhiNoBPTX_.numerator   = nullptr;
  jetPhiNoBPTX_.denominator = nullptr;
  jetPhiVsLS_.numerator   = nullptr;
  jetPhiVsBX_.numerator   = nullptr;

  muonPtNoBPTX_.numerator   = nullptr;
  muonPtNoBPTX_.denominator = nullptr;
  muonPtNoBPTX_variableBinning_.numerator   = nullptr;
  muonPtNoBPTX_variableBinning_.denominator = nullptr;
  muonPtVsLS_.numerator   = nullptr;
  muonPtVsBX_.numerator   = nullptr;
  muonEtaNoBPTX_.numerator   = nullptr;
  muonEtaNoBPTX_.denominator = nullptr;
  muonEtaVsLS_.numerator   = nullptr;
  muonEtaVsBX_.numerator   = nullptr;
  muonPhiNoBPTX_.numerator   = nullptr;
  muonPhiNoBPTX_.denominator = nullptr;
  muonPhiVsLS_.numerator   = nullptr;
  muonPhiVsBX_.numerator   = nullptr;

}

NoBPTXMonitor::~NoBPTXMonitor() = default;

NoBPTXMonitor::NoBPTXbinning NoBPTXMonitor::getHistoPSet(const edm::ParameterSet & pset)
{
  return NoBPTXbinning{
    pset.getParameter<unsigned int>("nbins"),
      pset.getParameter<double>("xmin"),
      pset.getParameter<double>("xmax"),
      };
}

NoBPTXMonitor::NoBPTXbinning NoBPTXMonitor::getHistoLSPSet(const edm::ParameterSet & pset)
{
  return NoBPTXbinning{
    pset.getParameter<unsigned int>("nbins"),
      0.,
      double(pset.getParameter<unsigned int>("nbins"))
      };
}

void NoBPTXMonitor::setNoBPTXTitle(NoBPTXME& me, const std::string& titleX, const std::string& titleY, bool bookDen)
{
  me.numerator->setAxisTitle(titleX,1);
  me.numerator->setAxisTitle(titleY,2);
  if(bookDen) {
    me.denominator->setAxisTitle(titleX,1);
    me.denominator->setAxisTitle(titleY,2);
  }

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
void NoBPTXMonitor::bookNoBPTX(DQMStore::IBooker &ibooker, NoBPTXME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, double ymin, double ymax, bool bookDen)
{
  me.numerator   = ibooker.bookProfile(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, ymin, ymax);
  if(bookDen) me.denominator = ibooker.bookProfile(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, ymin, ymax);
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
  bool bookDen;

  std::string currentFolder = folderName_ ;
  ibooker.setCurrentFolder(currentFolder);

  histname = "jetE"; histtitle = "jetE";
  bookDen = true;
  bookNoBPTX(ibooker,jetENoBPTX_,histname,histtitle,jetE_binning_.nbins,jetE_binning_.xmin, jetE_binning_.xmax);
  setNoBPTXTitle(jetENoBPTX_,"Jet E [GeV]","Events / [GeV]", bookDen);

  histname = "jetE_variable"; histtitle = "jetE";
  bookDen = true;
  bookNoBPTX(ibooker,jetENoBPTX_variableBinning_,histname,histtitle,jetE_variable_binning_);
  setNoBPTXTitle(jetENoBPTX_variableBinning_,"Jet E [GeV]","Events / [GeV]", bookDen);

  histname = "jetEVsLS"; histtitle = "jetE vs LS";
  bookDen = true;
  bookNoBPTX(ibooker,jetEVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,jetE_binning_.xmin, jetE_binning_.xmax, bookDen);
  setNoBPTXTitle(jetEVsLS_,"LS","Jet E [GeV]", bookDen);

  histname = "jetEVsBX"; histtitle = "jetE vs BX";
  bookDen = false;
  bookNoBPTX(ibooker,jetEVsBX_,histname,histtitle,bx_binning_.nbins, bx_binning_.xmin, bx_binning_.xmax,jetE_binning_.xmin, jetE_binning_.xmax, bookDen);
  setNoBPTXTitle(jetEVsBX_,"BX","Jet E [GeV]", bookDen);

  histname = "jetEta"; histtitle = "jetEta";
  bookDen = true;
  bookNoBPTX(ibooker,jetEtaNoBPTX_,histname,histtitle,jetEta_binning_.nbins,jetEta_binning_.xmin, jetEta_binning_.xmax);
  setNoBPTXTitle(jetEtaNoBPTX_,"Jet #eta","Events", bookDen);

  histname = "jetEtaVsLS"; histtitle = "jetEta vs LS";
  bookDen = false;
  bookNoBPTX(ibooker,jetEtaVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,jetEta_binning_.xmin, jetEta_binning_.xmax, bookDen);
  setNoBPTXTitle(jetEtaVsLS_,"LS","Jet #eta", bookDen);

  histname = "jetEtaVsBX"; histtitle = "jetEta vs BX";
  bookDen = false;
  bookNoBPTX(ibooker,jetEtaVsBX_,histname,histtitle,bx_binning_.nbins, bx_binning_.xmin, bx_binning_.xmax,jetEta_binning_.xmin, jetEta_binning_.xmax, bookDen);
  setNoBPTXTitle(jetEtaVsBX_,"BX","Jet #eta", bookDen);

  histname = "jetPhi"; histtitle = "jetPhi";
  bookDen = true;
  bookNoBPTX(ibooker,jetPhiNoBPTX_,histname,histtitle,jetPhi_binning_.nbins,jetPhi_binning_.xmin, jetPhi_binning_.xmax);
  setNoBPTXTitle(jetPhiNoBPTX_,"Jet #phi","Events", bookDen);

  histname = "jetPhiVsLS"; histtitle = "jetPhi vs LS";
  bookDen = false;
  bookNoBPTX(ibooker,jetPhiVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,jetPhi_binning_.xmin, jetPhi_binning_.xmax, bookDen);
  setNoBPTXTitle(jetPhiVsLS_,"LS","Jet #phi", bookDen);

  histname = "jetPhiVsBX"; histtitle = "jetPhi vs BX";
  bookDen = false;
  bookNoBPTX(ibooker,jetPhiVsBX_,histname,histtitle,bx_binning_.nbins, bx_binning_.xmin, bx_binning_.xmax,jetPhi_binning_.xmin, jetPhi_binning_.xmax, bookDen);
  setNoBPTXTitle(jetPhiVsBX_,"BX","Jet #phi", bookDen);

  histname = "muonPt"; histtitle = "muonPt";
  bookDen = true;
  bookNoBPTX(ibooker,muonPtNoBPTX_,histname,histtitle,muonPt_binning_.nbins,muonPt_binning_.xmin, muonPt_binning_.xmax);
  setNoBPTXTitle(muonPtNoBPTX_,"DisplacedStandAlone Muon p_{T} [GeV]","Events / [GeV]", bookDen);

  histname = "muonPt_variable"; histtitle = "muonPt";
  bookDen = true;
  bookNoBPTX(ibooker,muonPtNoBPTX_variableBinning_,histname,histtitle,muonPt_variable_binning_);
  setNoBPTXTitle(muonPtNoBPTX_variableBinning_,"DisplacedStandAlone Muon p_{T} [GeV]","Events / [GeV]", bookDen);

  histname = "muonPtVsLS"; histtitle = "muonPt vs LS";
  bookDen = false;
  bookNoBPTX(ibooker,muonPtVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,muonPt_binning_.xmin, muonPt_binning_.xmax, bookDen);
  setNoBPTXTitle(muonPtVsLS_,"LS","DisplacedStandAlone Muon p_{T} [GeV]", bookDen);

  histname = "muonPtVsBX"; histtitle = "muonPt vs BX";
  bookDen = false;
  bookNoBPTX(ibooker,muonPtVsBX_,histname,histtitle,bx_binning_.nbins, bx_binning_.xmin, bx_binning_.xmax,muonPt_binning_.xmin, muonPt_binning_.xmax, bookDen);
  setNoBPTXTitle(muonPtVsBX_,"BX","DisplacedStandAlone Muon p_{T} [GeV]", bookDen);

  histname = "muonEta"; histtitle = "muonEta";
  bookDen = true;
  bookNoBPTX(ibooker,muonEtaNoBPTX_,histname,histtitle,muonEta_binning_.nbins,muonEta_binning_.xmin, muonEta_binning_.xmax);
  setNoBPTXTitle(muonEtaNoBPTX_,"DisplacedStandAlone Muon #eta","Events", bookDen);

  histname = "muonEtaVsLS"; histtitle = "muonEta vs LS";
  bookDen = false;
  bookNoBPTX(ibooker,muonEtaVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,muonEta_binning_.xmin, muonEta_binning_.xmax, bookDen);
  setNoBPTXTitle(muonEtaVsLS_,"LS","DisplacedStandAlone Muon #eta", bookDen);

  histname = "muonEtaVsBX"; histtitle = "muonEta vs BX";
  bookDen = false;
  bookNoBPTX(ibooker,muonEtaVsBX_,histname,histtitle,bx_binning_.nbins, bx_binning_.xmin, bx_binning_.xmax,muonEta_binning_.xmin, muonEta_binning_.xmax, bookDen);
  setNoBPTXTitle(muonEtaVsBX_,"BX","DisplacedStandAlone Muon #eta", bookDen);

  histname = "muonPhi"; histtitle = "muonPhi";
  bookDen = true;
  bookNoBPTX(ibooker,muonPhiNoBPTX_,histname,histtitle,muonPhi_binning_.nbins,muonPhi_binning_.xmin, muonPhi_binning_.xmax);
  setNoBPTXTitle(muonPhiNoBPTX_,"DisplacedStandAlone Muon #phi","Events", bookDen);

  histname = "muonPhiVsLS"; histtitle = "muonPhi vs LS";
  bookDen = false;
  bookNoBPTX(ibooker,muonPhiVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,muonPhi_binning_.xmin, muonPhi_binning_.xmax, bookDen);
  setNoBPTXTitle(muonPhiVsLS_,"LS","DisplacedStandAlone Muon #phi", bookDen);

  histname = "muonPhiVsBX"; histtitle = "muonPhi vs BX";
  bookDen = false;
  bookNoBPTX(ibooker,muonPhiVsBX_,histname,histtitle,bx_binning_.nbins, bx_binning_.xmin, bx_binning_.xmax,muonPhi_binning_.xmin, muonPhi_binning_.xmax, bookDen);
  setNoBPTXTitle(muonPhiVsBX_,"BX","DisplacedStandAlone Muon #phi", bookDen);

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
  if ((unsigned int)(jetHandle->size()) < njets_ ) return;
  for ( auto const & j : *jetHandle ) {
    if ( jetSelection_( j ) ) jets.push_back(j);
  }
  if ((unsigned int)(jets.size()) < njets_ ) return;
  double jetE = -999;
  double jetEta = -999;
  double jetPhi = -999;
  if(!jets.empty()){
    jetE = jets[0].energy();
    jetEta = jets[0].eta();
    jetPhi = jets[0].phi();
  }

  edm::Handle<reco::TrackCollection> DSAHandle;
  iEvent.getByToken( muonToken_, DSAHandle );
  if ((unsigned int)(DSAHandle->size()) < nmuons_ ) return;
  std::vector<reco::Track> muons;
  for ( auto const & m : *DSAHandle ) {
    if ( muonSelection_( m ) ) muons.push_back(m);
  }
  if ((unsigned int)(muons.size()) < nmuons_ ) return;
  double muonPt = -999;
  double muonEta = -999;
  double muonPhi = -999;
  if(!muons.empty()){
    muonPt = muons[0].pt();
    muonEta = muons[0].eta();
    muonPhi = muons[0].phi();
  }

  // filling histograms (denominator)
  jetENoBPTX_.denominator -> Fill(jetE);
  jetENoBPTX_variableBinning_.denominator -> Fill(jetE);
  jetEVsLS_.denominator -> Fill(ls, jetE);
  jetEtaNoBPTX_.denominator -> Fill(jetEta);
  jetPhiNoBPTX_.denominator -> Fill(jetPhi);
  muonPtNoBPTX_.denominator -> Fill(muonPt);
  muonPtNoBPTX_variableBinning_.denominator -> Fill(muonPt);
  muonEtaNoBPTX_.denominator -> Fill(muonEta);
  muonPhiNoBPTX_.denominator -> Fill(muonPhi);

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
  pset.add<unsigned int>   ( "nbins", 200);
  pset.add<double>( "xmin", -0.5 );
  pset.add<double>( "xmax", 19999.5 );
}

void NoBPTXMonitor::fillHistoLSPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<unsigned int>   ( "nbins", 2000);
}

void NoBPTXMonitor::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<std::string>  ( "FolderName", "HLT/NoBPTX" );

  desc.add<edm::InputTag>( "jets",     edm::InputTag("ak4CaloJets") );
  desc.add<edm::InputTag>( "muons",    edm::InputTag("displacedStandAloneMuons") );
  desc.add<std::string>("jetSelection", "pt > 0");
  desc.add<std::string>("muonSelection", "pt > 0");
  desc.add<unsigned int>("njets",      0);
  desc.add<unsigned int>("nmuons",     0);

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
  fillHistoPSetDescription(lsPSet);
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
