#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMOffline/Trigger/plugins/HTMonitor.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

#include "DataFormats/Math/interface/deltaPhi.h"

// -----------------------------
//  constructors and destructor
// -----------------------------

HTMonitor::HTMonitor( const edm::ParameterSet& iConfig ) : 
  folderName_             ( iConfig.getParameter<std::string>("FolderName") )
  , metInputTag_          ( iConfig.getParameter<edm::InputTag>    ("met")          )
  , jetInputTag_          ( iConfig.getParameter<edm::InputTag>    ("jets")         )
  , eleInputTag_          ( iConfig.getParameter<edm::InputTag>    ("electrons")    ) 
  , muoInputTag_          ( iConfig.getParameter<edm::InputTag>    ("muons")        ) 
  , vtxInputTag_          ( iConfig.getParameter<edm::InputTag>    ("vertices")     ) 
  , metToken_             ( consumes<reco::PFMETCollection>        ( metInputTag_ ) )
  , jetToken_             ( mayConsume<reco::JetView>              ( jetInputTag_ ) )
  , eleToken_             ( mayConsume<reco::GsfElectronCollection>( eleInputTag_ ) )
  , muoToken_             ( mayConsume<reco::MuonCollection>       ( muoInputTag_ ) )
  , vtxToken_             ( mayConsume<reco::VertexCollection>     ( vtxInputTag_ ) )
  , ht_variable_binning_  ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("htBinning") )
  , ht_binning_           ( getHistoPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("htPSet")    ) )
  , ls_binning_           ( getHistoPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("lsPSet")     ) )
  , num_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"),consumesCollector(), *this))
  , den_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"),consumesCollector(), *this))
  , metSelection_ ( iConfig.getParameter<std::string>("metSelection") )
  , jetSelection_ ( iConfig.getParameter<std::string>("jetSelection") )
  , eleSelection_ ( iConfig.getParameter<std::string>("eleSelection") )
  , muoSelection_ ( iConfig.getParameter<std::string>("muoSelection") )
  , jetSelection_HT_ ( iConfig.getParameter<std::string>("jetSelection_HT") )
  , njets_      ( iConfig.getParameter<unsigned>("njets" )      )
  , nelectrons_ ( iConfig.getParameter<unsigned>("nelectrons" ) )
  , nmuons_     ( iConfig.getParameter<unsigned>("nmuons" )     )
  , dEtaCut_    ( iConfig.getParameter<double>("dEtaCut")       )
{
  /* mia: THIS CODE SHOULD BE DELETED !!!! */
    string quantity = iConfig.getParameter<std::string>("quantity");
    if(quantity == "HT")
    {
        quantity_ = HT;
    }
    else if(quantity == "Mjj")
    {
        quantity_ = MJJ;
    }
    else if(quantity == "softdrop")
    {
        quantity_ = SOFTDROP;
    }
    else
    {
        throw cms::Exception("quantity not defined") << "the quantity '" << quantity << "' is undefined. Please check your config!" << std::endl;
    }

    // this vector has to be alligned to the the number of Tokens accessed by this module
    warningPrinted4token_.push_back(false); // PFMETCollection
    warningPrinted4token_.push_back(false); // JetCollection
    warningPrinted4token_.push_back(false); // GsfElectronCollection
    warningPrinted4token_.push_back(false); // MuonCollection
    warningPrinted4token_.push_back(false); // VertexCollection
}

HTMonitor::~HTMonitor() = default;

HTMonitor::MEHTbinning HTMonitor::getHistoPSet(const edm::ParameterSet& pset)
{
  return HTMonitor::MEHTbinning{
    pset.getParameter<unsigned>("nbins"),
      pset.getParameter<double>("xmin"),
      pset.getParameter<double>("xmax"),
      };
}

HTMonitor::MEHTbinning HTMonitor::getHistoLSPSet(const edm::ParameterSet& pset)
{
  return HTMonitor::MEHTbinning{
    pset.getParameter<unsigned>("nbins"),
      0.,
      double(pset.getParameter<unsigned>("nbins"))
      };
}

void HTMonitor::setHTitle(HTME& me, const std::string& titleX, const std::string& titleY)
{
  me.numerator->setAxisTitle(titleX,1);
  me.numerator->setAxisTitle(titleY,2);
  me.denominator->setAxisTitle(titleX,1);
  me.denominator->setAxisTitle(titleY,2);

}

void HTMonitor::bookME(DQMStore::IBooker &ibooker, HTME& me, const std::string& histname, const std::string& histtitle, int nbins, double min, double max)
{
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, min, max);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, min, max);
}
void HTMonitor::bookME(DQMStore::IBooker &ibooker, HTME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binning)
{
  int nbins = binning.size()-1;
  std::vector<float> fbinning(binning.begin(),binning.end());
  float* arr = &fbinning[0];
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, arr);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, arr);
}
void HTMonitor::bookME(DQMStore::IBooker &ibooker, HTME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, double ymin, double ymax)
{
  me.numerator   = ibooker.bookProfile(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, ymin, ymax);
  me.denominator = ibooker.bookProfile(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, ymin, ymax);
}
void HTMonitor::bookME(DQMStore::IBooker &ibooker, HTME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, int nbinsY, double ymin, double ymax)
{
  me.numerator   = ibooker.book2D(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, nbinsY, ymin, ymax);
  me.denominator = ibooker.book2D(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, nbinsY, ymin, ymax);
}
void HTMonitor::bookME(DQMStore::IBooker &ibooker, HTME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX, const std::vector<double>& binningY)
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

void HTMonitor::bookHistograms(DQMStore::IBooker     & ibooker,
				 edm::Run const        & iRun,
				 edm::EventSetup const & iSetup) 
{
  std::string histname, histtitle;

  std::string currentFolder = folderName_ ;
  ibooker.setCurrentFolder(currentFolder);

  switch(quantity_)
  {
    case HT:
    {
        histname = "ht_variable"; histtitle = "HT";
        bookME(ibooker,qME_variableBinning_,histname,histtitle,ht_variable_binning_);
        setHTitle(qME_variableBinning_,"HT [GeV]","events / [GeV]");

        histname = "htVsLS"; histtitle = "HT vs LS";
        bookME(ibooker,htVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,ht_binning_.xmin, ht_binning_.xmax);
        setHTitle(htVsLS_,"LS","HT [GeV]");

        histname = "deltaphi_metjet1"; histtitle = "DPHI_METJ1";
        bookME(ibooker,deltaphimetj1ME_,histname,histtitle,phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
        setHTitle(deltaphimetj1ME_,"delta phi (met, j1)","events / 0.1 rad");

        histname = "deltaphi_jet1jet2"; histtitle = "DPHI_J1J2";
        bookME(ibooker,deltaphij1j2ME_,histname,histtitle,phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
        setHTitle(deltaphij1j2ME_,"delta phi (j1, j2)","events / 0.1 rad");
        break;
    }


    case MJJ:
    {
        histname = "mjj_variable"; histtitle = "Mjj";
        bookME(ibooker,qME_variableBinning_,histname,histtitle, ht_variable_binning_);
        setHTitle(qME_variableBinning_,"Mjj [GeV]","events / [GeV]");
        break;
    }


    case SOFTDROP:
    {
        histname = "softdrop_variable"; histtitle = "softdropmass";
        bookME(ibooker,qME_variableBinning_,histname,histtitle, ht_variable_binning_);
        setHTitle(qME_variableBinning_,"leading jet softdropmass [GeV]","events / [GeV]");
        break;
    }
  }

  // Initialize the GenericTriggerEventFlag
  if ( num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on() ) num_genTriggerEventFlag_->initRun( iRun, iSetup );
  if ( den_genTriggerEventFlag_ && den_genTriggerEventFlag_->on() ) den_genTriggerEventFlag_->initRun( iRun, iSetup );
}

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
void HTMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)  {

  // Filter out events if Trigger Filtering is requested
  if (den_genTriggerEventFlag_->on() && ! den_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  edm::Handle<reco::PFMETCollection> metHandle;
  iEvent.getByToken( metToken_, metHandle );
  if ( !metHandle.isValid() ) {
    if (!warningPrinted4token_[0]) {
      edm::LogWarning("HTMonitor") << "skipping events because the collection " << metInputTag_.label().c_str() << " is not available";
      warningPrinted4token_[0] = true;
    }
    return;
  }
  reco::PFMET pfmet = metHandle->front();
  if ( ! metSelection_( pfmet ) ) return;

  edm::Handle<reco::JetView> jetHandle; //add a configurable jet collection & jet pt selection
  iEvent.getByToken( jetToken_, jetHandle );
  if ( !jetHandle.isValid() ) {
    if (!warningPrinted4token_[1]) {
      edm::LogWarning("HTMonitor") << "skipping events because the collection " << jetInputTag_.label().c_str() << " is not available";
      warningPrinted4token_[1] = true;
    }
    return;
  }
  std::vector<reco::Jet> jets;
  if ( jetHandle->size() < njets_ ) return;
  for ( auto const & j : *jetHandle ) {
    if ( jetSelection_( j ) ) {
      jets.push_back(j);
    }
  }

  if ( jets.size() < njets_ ) return;

  float deltaPhi_met_j1 = 10.0;
  float deltaPhi_j1_j2 = 10.0;

  if (!jets.empty()) deltaPhi_met_j1 = fabs( deltaPhi( pfmet.phi(),  jets[0].phi() ));
  if (jets.size() >= 2) deltaPhi_j1_j2 = fabs( deltaPhi( jets[0].phi(),  jets[1].phi() ));

  std::vector<reco::GsfElectron> electrons;
  edm::Handle<reco::GsfElectronCollection> eleHandle;
  iEvent.getByToken( eleToken_, eleHandle );
  if ( eleHandle.isValid() ) {
    if ( eleHandle->size() < nelectrons_ ) return;
    for ( auto const & e : *eleHandle ) {
      if ( eleSelection_( e ) ) electrons.push_back(e);
    }
    if ( electrons.size() < nelectrons_ ) return;
  } else {
    if (!warningPrinted4token_[2]) {
      warningPrinted4token_[2] = true;
      if ( eleInputTag_.label().empty() ) 
	edm::LogWarning("HTMonitor") << "GsfElectronCollection not set";
      else
	edm::LogWarning("HTMonitor") << "skipping events because the collection " << eleInputTag_.label().c_str() << " is not available";
    }
    if ( !eleInputTag_.label().empty() ) return;
  }
  

  reco::Vertex vtx;
  edm::Handle<reco::VertexCollection> vtxHandle;
  iEvent.getByToken(vtxToken_, vtxHandle);
  if ( vtxHandle.isValid() ) {
    for (auto const & v : *vtxHandle) {
      bool isFake =  v.isFake() ;
      
      if (!isFake) {
	vtx = v;
	break;
      }
    }
  } else {
    if (!warningPrinted4token_[3]) {
      warningPrinted4token_[3] = true;
      if ( vtxInputTag_.label().empty() )
	edm::LogWarning("HTMonitor") << "VertexCollection not set";
      else
	edm::LogWarning("HTMonitor") << "skipping events because the collection " << vtxInputTag_.label().c_str() << " is not available";
    }
    if ( !vtxInputTag_.label().empty() ) return;
  }

  std::vector<reco::Muon> muons;
  edm::Handle<reco::MuonCollection> muoHandle;
  iEvent.getByToken( muoToken_, muoHandle );
  if ( muoHandle.isValid() ) {
    if ( muoHandle->size() < nmuons_ ) return;
    for ( auto const & m : *muoHandle ) {
      if ( muoSelection_( m ) && m.isGlobalMuon() && m.isPFMuon() && m.globalTrack()->normalizedChi2() < 10. && m.globalTrack()->hitPattern().numberOfValidMuonHits() > 0 && m.numberOfMatchedStations() > 1 && fabs(m.muonBestTrack()->dxy(vtx.position())) < 0.2 && fabs(m.muonBestTrack()->dz(vtx.position())) < 0.5 && m.innerTrack()->hitPattern().numberOfValidPixelHits() > 0 && m.innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5 )  muons.push_back(m);
    }
    if ( muons.size() < nmuons_ ) return;
  } else {
    if (!warningPrinted4token_[4]) {
      warningPrinted4token_[4] = true;
      if ( muoInputTag_.label().empty() )
	edm::LogWarning("HTMonitor") << "MuonCollection not set";
      else
	edm::LogWarning("HTMonitor") << "skipping events because the collection " << muoInputTag_.label().c_str() << " is not available";
    }
    if ( !muoInputTag_.label().empty() ) return;
  }

  // fill histograms
  switch(quantity_)
  {
    case HT:
    {
        float ht = 0.0;
        for ( auto const & j : *jetHandle )
        {
            if ( jetSelection_HT_(j)) ht += j.pt();
        }

        // filling histograms (denominator)  
        qME_variableBinning_.denominator -> Fill(ht);

        deltaphimetj1ME_.denominator -> Fill(deltaPhi_met_j1);
        deltaphij1j2ME_.denominator -> Fill(deltaPhi_j1_j2);

        int ls = iEvent.id().luminosityBlock();
        htVsLS_.denominator -> Fill(ls, ht);

        // applying selection for numerator
        if (num_genTriggerEventFlag_->on() && ! num_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

        // filling histograms (num_genTriggerEventFlag_)  
        qME_variableBinning_.numerator -> Fill(ht);

        htVsLS_.numerator -> Fill(ls, ht);
        deltaphimetj1ME_.numerator  -> Fill(deltaPhi_met_j1); 
        deltaphij1j2ME_.numerator  -> Fill(deltaPhi_j1_j2);
        break;
    }

    case MJJ:
    {
        if (jets.size() < 2) return;

        // deltaEta cut
        if(fabs(jets[0].p4().Eta() - jets[1].p4().Eta()) >= dEtaCut_) return;
        float mjj = (jets[0].p4() + jets[1].p4()).M();

        qME_variableBinning_.denominator -> Fill(mjj);

        // applying selection for numerator
        if (num_genTriggerEventFlag_->on() && ! num_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

        qME_variableBinning_.numerator -> Fill(mjj);
        break;
    }

    case SOFTDROP:
    {
        if (jets.size() < 2) return;

        // deltaEta cut
        if(fabs(jets[0].p4().Eta() - jets[1].p4().Eta()) >= dEtaCut_) return;

        float softdrop = jets[0].p4().M();

        qME_variableBinning_.denominator -> Fill(softdrop);

        // applying selection for numerator
        if (num_genTriggerEventFlag_->on() && ! num_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

        qME_variableBinning_.numerator -> Fill(softdrop);
        break;
    }
  }
}

void HTMonitor::fillHistoPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<unsigned int>   ( "nbins");
  pset.add<double>( "xmin" );
  pset.add<double>( "xmax" );
}

void HTMonitor::fillHistoLSPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<unsigned int>   ( "nbins", 2500);
  pset.add<double>         ( "xmin",     0.);
  pset.add<double>         ( "xmax",  2500.);
}

void HTMonitor::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<std::string>  ( "FolderName", "HLT/HT" );
  desc.add<std::string>  ( "quantity", "HT" );


  desc.add<edm::InputTag>( "met",      edm::InputTag("pfMet") );
  desc.add<edm::InputTag>( "jets",     edm::InputTag("ak4PFJetsCHS") );
  desc.add<edm::InputTag>( "electrons",edm::InputTag("gedGsfElectrons") );
  desc.add<edm::InputTag>( "muons",    edm::InputTag("muons") );
  desc.add<edm::InputTag>( "vertices",edm::InputTag("offlinePrimaryVertices") );
  desc.add<std::string>("metSelection", "pt > 0");
  desc.add<std::string>("jetSelection", "pt > 0");
  desc.add<std::string>("eleSelection", "pt > 0");
  desc.add<std::string>("muoSelection", "pt > 0");
  desc.add<std::string>("jetSelection_HT", "pt > 30 && eta < 2.5");
  desc.add<unsigned>("njets",      0);
  desc.add<unsigned>("nelectrons", 0);
  desc.add<unsigned>("nmuons",     0);
  desc.add<double>("dEtaCut",      1.3);

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
  edm::ParameterSetDescription htPSet;
  fillHistoPSetDescription(htPSet);
  histoPSet.add<edm::ParameterSetDescription>("htPSet", htPSet);
  std::vector<double> bins = {0.,20.,40.,60.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.,190.,200.,220.,240.,260.,280.,300.,350.,400.,450.,500.,550.,600.,650.,700.,750.,800.,850.,900.,950.,1000.,1050.,1100.,1200.,1300.,1400.,1500.,2000.,2500.};
  histoPSet.add<std::vector<double> >("htBinning", bins);

  edm::ParameterSetDescription lsPSet;
  fillHistoLSPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);

  desc.add<edm::ParameterSetDescription>("histoPSet",histoPSet);

  descriptions.add("htMonitoring", desc);
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HTMonitor);
