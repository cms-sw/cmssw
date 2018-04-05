#include "DQM/TrackingMonitor/interface/V0Monitor.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DQM/TrackingMonitor/interface/GetLumi.h"


// -----------------------------
//  constructors and destructor
// -----------------------------

V0Monitor::V0Monitor( const edm::ParameterSet& iConfig ) : 
  folderName_( iConfig.getParameter<std::string>("FolderName") )
  , v0Token_         ( consumes<reco::VertexCompositeCandidateCollection>(iConfig.getParameter<edm::InputTag>("v0") )            )
  , bsToken_         ( consumes<reco::BeamSpot>                          (iConfig.getParameter<edm::InputTag>("beamSpot") )      )
  , pvToken_         ( consumes<reco::VertexCollection>                  (iConfig.getParameter<edm::InputTag>("primaryVertex") ) )
  , lumiscalersToken_( consumes<LumiScalersCollection>                   (iConfig.getParameter<edm::InputTag>("lumiScalers") )   )
  , pvNDOF_ ( iConfig.getParameter<int> ("pvNDOF") )
  , genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("genericTriggerEventPSet"),consumesCollector(), *this))
{

  v0_N_                = nullptr;
  v0_mass_             = nullptr;
  v0_pt_               = nullptr;
  v0_eta_              = nullptr;
  v0_phi_              = nullptr;
  v0_Lxy_              = nullptr;
  v0_Lxy_wrtBS_        = nullptr;
  v0_chi2oNDF_         = nullptr;
  v0_mass_vs_p_        = nullptr;
  v0_mass_vs_pt_       = nullptr;
  v0_mass_vs_eta_      = nullptr;
  v0_deltaMass_        = nullptr;
  v0_deltaMass_vs_pt_  = nullptr;
  v0_deltaMass_vs_eta_ = nullptr;

  v0_Lxy_vs_deltaMass_ = nullptr;
  v0_Lxy_vs_pt_        = nullptr;
  v0_Lxy_vs_eta_       = nullptr;
  
  n_vs_BX_            = nullptr;
  v0_N_vs_BX_         = nullptr;
  v0_mass_vs_BX_      = nullptr;
  v0_Lxy_vs_BX_       = nullptr;
  v0_deltaMass_vs_BX_ = nullptr;
  
  n_vs_lumi_            = nullptr;
  v0_N_vs_lumi_         = nullptr;
  v0_mass_vs_lumi_      = nullptr;
  v0_Lxy_vs_lumi_       = nullptr;
  v0_deltaMass_vs_lumi_ = nullptr;
  
  n_vs_PU_            = nullptr;
  v0_N_vs_PU_         = nullptr;
  v0_mass_vs_PU_      = nullptr;
  v0_Lxy_vs_PU_       = nullptr;
  v0_deltaMass_vs_PU_ = nullptr;

  n_vs_LS_    = nullptr;
  v0_N_vs_LS_ = nullptr;

  edm::ParameterSet histoPSet    = iConfig.getParameter<edm::ParameterSet>("histoPSet");
  getHistoPSet(histoPSet.getParameter<edm::ParameterSet>("massPSet"),     mass_binning_    );
  getHistoPSet(histoPSet.getParameter<edm::ParameterSet>("ptPSet"),       pt_binning_      );
  getHistoPSet(histoPSet.getParameter<edm::ParameterSet>("etaPSet"),      eta_binning_     );
  getHistoPSet(histoPSet.getParameter<edm::ParameterSet>("LxyPSet"),      Lxy_binning_     );
  getHistoPSet(histoPSet.getParameter<edm::ParameterSet>("chi2oNDFPSet"), chi2oNDF_binning_);
  getHistoPSet(histoPSet.getParameter<edm::ParameterSet>("lumiPSet"),     lumi_binning_    );
  getHistoPSet(histoPSet.getParameter<edm::ParameterSet>("puPSet"),       pu_binning_      );
  getHistoPSet(histoPSet.getParameter<edm::ParameterSet>("lsPSet"),       ls_binning_      );

}

V0Monitor::~V0Monitor()
{
  if (genTriggerEventFlag_) delete genTriggerEventFlag_;
}

void V0Monitor::getHistoPSet(edm::ParameterSet pset, MEbinning& mebinning)
{
  mebinning.nbins = pset.getParameter<int32_t>("nbins");
  mebinning.xmin  = pset.getParameter<double>("xmin");
  mebinning.xmax  = pset.getParameter<double>("xmax");
}

MonitorElement* V0Monitor::bookHisto1D(DQMStore::IBooker & ibooker,std::string name, std::string title, std::string xaxis, std::string yaxis, MEbinning binning)
{
  std::string title_w_axes = title+";"+xaxis+";"+yaxis;
  return ibooker.book1D(name, title_w_axes, 
			binning.nbins, binning.xmin, binning.xmax);  
  
}
MonitorElement* V0Monitor::bookHisto2D(DQMStore::IBooker & ibooker,std::string name, std::string title, std::string xaxis, std::string yaxis, MEbinning xbinning, MEbinning ybinning)
{
  std::string title_w_axes = title+";"+xaxis+";"+yaxis;
  return ibooker.book2D(name, title_w_axes, 
			xbinning.nbins, xbinning.xmin, xbinning.xmax,
			ybinning.nbins, ybinning.xmin, ybinning.xmax
			);  
}
MonitorElement* V0Monitor::bookProfile(DQMStore::IBooker & ibooker,std::string name, std::string title, std::string xaxis, std::string yaxis, MEbinning xbinning, MEbinning ybinning)
{
  std::string title_w_axes = title+";"+xaxis+";"+yaxis;
  return ibooker.bookProfile(name, title_w_axes, 
			     xbinning.nbins, xbinning.xmin, xbinning.xmax,
			     ybinning.xmin, ybinning.xmax
			     );  
}

void V0Monitor::bookHistograms(DQMStore::IBooker     & ibooker,
			       edm::Run const        & iRun,
			       edm::EventSetup const & iSetup) 
{  
  
  std::string histname, histtitle;

  std::string currentFolder = folderName_ ;
  ibooker.setCurrentFolder(currentFolder);

  MEbinning N_binning; N_binning.nbins = 15; N_binning.xmin = -0.5; N_binning.xmax = 14.5;
  v0_N_        = bookHisto1D(ibooker,"v0_N",        "# v0",     "# v0",                      "events",N_binning);
  v0_mass_     = bookHisto1D(ibooker,"v0_mass",     "mass",     "mass [GeV]",                "events",mass_binning_);
  v0_pt_       = bookHisto1D(ibooker,"v0_pt",       "pt",       "p_{T} [GeV]",               "events",pt_binning_  );
  v0_eta_      = bookHisto1D(ibooker,"v0_eta",      "eta",      "#eta",                      "events",eta_binning_ );
  MEbinning phi_binning; phi_binning.nbins = 34; phi_binning.xmin = -3.2; phi_binning.xmax = 3.2;
  v0_phi_       = bookHisto1D(ibooker,"v0_phi",      "phi",      "#phi [rad]",                "events",phi_binning  );
  v0_Lxy_       = bookHisto1D(ibooker,"v0_Lxy",      "Lxy",      "L_{xy} w.r.t. PV [cm]",     "events",Lxy_binning_ );
  v0_Lxy_wrtBS_ = bookHisto1D(ibooker,"v0_Lxy_wrtBS","Lxy",      "L_{xy} w.r.t. BS [cm]",     "events",Lxy_binning_ );
  v0_chi2oNDF_  = bookHisto1D(ibooker,"v0_chi2oNDF", "chi2oNDF", "vertex normalized #chi^{2}","events",chi2oNDF_binning_ ); 

  v0_mass_vs_p_   = bookProfile(ibooker,"v0_mass_vs_p",  "mass vs p",  "p [GeV]","mass [GeV]",    pt_binning_, mass_binning_);
  v0_mass_vs_pt_  = bookProfile(ibooker,"v0_mass_vs_pt", "mass vs pt", "p_{T} [GeV]","mass [GeV]",pt_binning_, mass_binning_);
  v0_mass_vs_eta_ = bookProfile(ibooker,"v0_mass_vs_eta","mass vs eta","#eta",       "mass [GeV]",eta_binning_,mass_binning_); 

  MEbinning delta_binning; delta_binning.nbins = 150; delta_binning.xmin = -0.15; delta_binning.xmax = 0.15;
  v0_deltaMass_ = bookHisto1D(ibooker,"v0_deltaMass", "deltaMass", "m-m_{PDG}/m_{DPG}", "events",delta_binning ); 
  v0_deltaMass_vs_pt_  = bookProfile(ibooker,"v0_deltaMass_vs_pt",  "deltaMass vs pt",  "p_{T} [GeV]", "m-m_{PDG}/m_{DPG}", pt_binning_,  delta_binning);
  v0_deltaMass_vs_eta_ = bookProfile(ibooker,"v0_deltaMass_vs_eta", "deltaMass vs eta", "#eta",        "m-m_{PDG}/m_{DPG}", eta_binning_, delta_binning);

  v0_Lxy_vs_deltaMass_ = bookProfile(ibooker,"v0_Lxy_vs_deltaMass","L_{xy} vs deltaMass","m-m_{PDG}/m_{DPG}","L_{xy} [cm]",delta_binning,Lxy_binning_);
  v0_Lxy_vs_pt_        = bookProfile(ibooker,"v0_Lxy_vs_pt",       "L_{xy} vs p_{T}",    "p_{T} [GeV]",      "L_{xy} [cm]",pt_binning_,  Lxy_binning_);
  v0_Lxy_vs_eta_       = bookProfile(ibooker,"v0_Lxy_vs_eta",      "L_{xy} vs #eta",     "#eta",             "L_{xy} [cm]",eta_binning_, Lxy_binning_);

  MEbinning bx_binning; bx_binning.nbins = 3564; bx_binning.xmin = 0.5; bx_binning.xmax = 3564.5;
  n_vs_BX_          = bookHisto1D(ibooker,"n_vs_BX","# events vs BX","BX", "# events",bx_binning);
  v0_N_vs_BX_         = bookProfile(ibooker,"v0_N_vs_BX",        "# v0 vs BX",     "BX", "# v0",             bx_binning, N_binning   );
  v0_mass_vs_BX_      = bookProfile(ibooker,"v0_mass_vs_BX",     "mass vs BX",     "BX", "mass [GeV]",       bx_binning, mass_binning_);
  v0_Lxy_vs_BX_       = bookProfile(ibooker,"v0_Lxy_vs_BX",      "L_{xy} vs BX",   "BX", "L_{xy} [cm]",      bx_binning, Lxy_binning_ );
  v0_deltaMass_vs_BX_ = bookProfile(ibooker,"v0_deltaMass_vs_BX","deltaMass vs BX","BX", "m-m_{PDG}/m_{DPG}",bx_binning, delta_binning);

  n_vs_lumi_            = bookHisto1D(ibooker,"n_vs_lumi","# events vs lumi","inst. lumi x10^{30} [Hz cm^{-2}]", "# events",lumi_binning_);
  v0_N_vs_lumi_         = bookProfile(ibooker,"v0_N_vs_lumi",        "# v0 vs lumi",     "inst. lumi x10^{30} [Hz cm^{-2}]", "# v0",             lumi_binning_, N_binning   );
  v0_mass_vs_lumi_      = bookProfile(ibooker,"v0_mass_vs_lumi",     "mass vs lumi",     "inst. lumi x10^{30} [Hz cm^{-2}]", "mass [GeV]",       lumi_binning_, mass_binning_);
  v0_Lxy_vs_lumi_       = bookProfile(ibooker,"v0_Lxy_vs_lumi",      "L_{xy} vs lumi",   "inst. lumi x10^{30} [Hz cm^{-2}]", "L_{xy} [cm]",      lumi_binning_, Lxy_binning_ );
  v0_deltaMass_vs_lumi_ = bookProfile(ibooker,"v0_deltaMass_vs_lumi","deltaMass vs lumi","inst. lumi x10^{30} [Hz cm^{-2}]", "m-m_{PDG}/m_{DPG}",lumi_binning_, delta_binning);

  n_vs_PU_            = bookHisto1D(ibooker,"n_vs_PU","# events vs PU","# good PV", "# events",pu_binning_);
  v0_N_vs_PU_         = bookProfile(ibooker,"v0_N_vs_PU",        "# v0 vs PU",     "# good PV", "# v0",             pu_binning_, N_binning   );
  v0_mass_vs_PU_      = bookProfile(ibooker,"v0_mass_vs_PU",     "mass vs PU",     "# good PV", "mass [GeV]",       pu_binning_, mass_binning_);
  v0_Lxy_vs_PU_       = bookProfile(ibooker,"v0_Lxy_vs_PU",      "L_{xy} vs PU",   "# good PV", "L_{xy} [cm]",      pu_binning_, Lxy_binning_ );
  v0_deltaMass_vs_PU_ = bookProfile(ibooker,"v0_deltaMass_vs_PU","deltaMass vs PU","# good PV", "m-m_{PDG}/m_{DPG}",pu_binning_, delta_binning);


  n_vs_LS_    = bookHisto1D(ibooker,"n_vs_LS",   "# events vs LS","LS", "# events",ls_binning_);
  v0_N_vs_LS_ = bookProfile(ibooker,"v0_N_vs_LS","# v0 vs LS",    "LS", "# v0",    ls_binning_, N_binning );
  v0_N_vs_LS_->getTH1()->SetCanExtend(TH1::kAllAxes);

  // Initialize the GenericTriggerEventFlag
  if ( genTriggerEventFlag_->on() ) genTriggerEventFlag_->initRun( iRun, iSetup );  

}

void V0Monitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)  {

  // Filter out events if Trigger Filtering is requested
  if (genTriggerEventFlag_->on()&& ! genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  //  int ls = iEvent.id().luminosityBlock();
  
  size_t bx = iEvent.bunchCrossing();
  n_vs_BX_->Fill(bx);

  float lumi = -1.;
  edm::Handle<LumiScalersCollection> lumiScalers;
  iEvent.getByToken(lumiscalersToken_, lumiScalers);
  if ( lumiScalers.isValid() && !lumiScalers->empty() ) {
    LumiScalersCollection::const_iterator scalit = lumiScalers->begin();
    lumi = scalit->instantLumi();
  } else 
    lumi = -1.;
  n_vs_lumi_->Fill(lumi);

  edm::Handle<reco::BeamSpot> beamspotHandle;
  iEvent.getByToken(bsToken_,beamspotHandle);
  reco::BeamSpot const * bs = nullptr;
  if (beamspotHandle.isValid())
    bs = &(*beamspotHandle);


  edm::Handle< reco::VertexCollection > pvHandle;
  iEvent.getByToken(pvToken_, pvHandle );
  reco::Vertex const * pv = nullptr;  
  size_t nPV = 0;
  if (pvHandle.isValid()) {
    pv = &pvHandle->front();
    //--- pv fake (the pv collection should have size==1 and the pv==beam spot)
    if (   pv->isFake() || pv->tracksSize()==0
	   // definition of goodOfflinePrimaryVertex
	   || pv->ndof() < pvNDOF_ || pv->z() > 24.)  pv = nullptr;

    for (auto v : *pvHandle) {
      if (v.isFake()        ) continue;
      if (v.ndof() < pvNDOF_) continue;
      if (v.z() > 24.       ) continue; 
      ++nPV;
    }
  }
  n_vs_PU_->Fill(nPV);

  float nLS = static_cast<float>(iEvent.id().luminosityBlock());
  n_vs_LS_->Fill(nLS);

  edm::Handle<reco::VertexCompositeCandidateCollection> v0Handle;
  iEvent.getByToken(v0Token_, v0Handle);
  int n = ( v0Handle.isValid() ? v0Handle->size() : -1 );
  v0_N_         -> Fill(n);
  v0_N_vs_BX_   -> Fill(bx,  n);
  v0_N_vs_lumi_ -> Fill(lumi,n);
  v0_N_vs_PU_   -> Fill(nPV, n);
  v0_N_vs_LS_   -> Fill(nLS, n);

  if ( !v0Handle.isValid() or n==0)
    return;
  
  reco::VertexCompositeCandidateCollection v0s = *v0Handle.product();    
  for ( auto v0 : v0s ) {
    float mass     = v0.mass();
    float pt       = v0.pt();
    float p        = v0.p();
    float eta      = v0.eta();
    float phi      = v0.phi();
    int pdgID      = v0.pdgId();
    float chi2oNDF = v0.vertexNormalizedChi2();
    GlobalPoint displacementFromPV = ( pv==nullptr ? GlobalPoint(-9.,-9.,0) : GlobalPoint( (pv->x() - v0.vx()), 
											   (pv->y() - v0.vy()), 
											   0. ) );
    GlobalPoint displacementFromBS = ( bs==nullptr ? GlobalPoint(-9.-9.,0.) : GlobalPoint( -1*((bs->position().x() - v0.vx()) + (v0.vz() - bs->position().z()) * bs->dxdz()),
											   -1*((bs->position().y() - v0.vy()) + (v0.vz() - bs->position().z()) * bs->dydz()), 
											   0 ) );
    float lxy      = ( pv==nullptr ? -9. : displacementFromPV.perp() );
    float lxyWRTbs = ( bs==nullptr ? -9. : displacementFromBS.perp() );

    v0_mass_     -> Fill(mass);
    v0_pt_       -> Fill(pt);
    v0_eta_      -> Fill(eta);
    v0_phi_      -> Fill(phi);
    v0_Lxy_      -> Fill(lxy);
    v0_Lxy_wrtBS_-> Fill(lxyWRTbs);
    v0_chi2oNDF_ -> Fill(chi2oNDF);
    
    v0_mass_vs_p_    -> Fill(p,   mass);
    v0_mass_vs_pt_   -> Fill(pt,  mass);
    v0_mass_vs_eta_  -> Fill(eta, mass);
    v0_mass_vs_BX_   -> Fill(bx,  mass);
    v0_mass_vs_lumi_ -> Fill(lumi,mass);
    v0_mass_vs_PU_   -> Fill(nPV, mass);

    v0_Lxy_vs_BX_   -> Fill(bx,  lxy);
    v0_Lxy_vs_lumi_ -> Fill(lumi,lxy);
    v0_Lxy_vs_PU_   -> Fill(nPV, lxy);

    float PDGmass = -9999.;
    switch(pdgID) {
    case 130: // K_s
    case 310: // K_L
      PDGmass = 0.497614; // GeV
      break;
    case 3122: // Lambda
    case -3122: // Lambda
      PDGmass = 1.115683; // GeV
      break;
    case 4122: // Lambda_c
    case -4122: // Lambda_c
    case 5122: // Lambda_b
    case -5122: // Lambda_b
    default:
      break;
    }
    float delta = (PDGmass > 0. ? (mass-PDGmass)/PDGmass : -9.);
    v0_deltaMass_         -> Fill (delta);
    v0_deltaMass_vs_pt_   -> Fill (pt,  delta);
    v0_deltaMass_vs_eta_  -> Fill (eta, delta);
    v0_deltaMass_vs_BX_   -> Fill (bx,  delta);
    v0_deltaMass_vs_lumi_ -> Fill (lumi,delta);
    v0_deltaMass_vs_PU_   -> Fill (nPV, delta);

    v0_Lxy_vs_deltaMass_ -> Fill (delta,lxy);
    v0_Lxy_vs_pt_        -> Fill (pt,   lxy);
    v0_Lxy_vs_eta_       -> Fill (eta,  lxy);
  }

}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(V0Monitor);
