#include "DQMOffline/Trigger/plugins/TopMonitor.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/TrackingMonitor/interface/GetLumi.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"


// -----------------------------
//  constructors and destructor
// -----------------------------

TopMonitor::TopMonitor( const edm::ParameterSet& iConfig ) : 
  folderName_             ( iConfig.getParameter<std::string>("FolderName") )
  , metToken_             ( consumes<reco::PFMETCollection>      (iConfig.getParameter<edm::InputTag>("met")       ) )   
  , jetToken_             ( mayConsume<reco::PFJetCollection>      (iConfig.getParameter<edm::InputTag>("jets")      ) )   
  , eleToken_             ( mayConsume<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("electrons") ) )   
  , muoToken_             ( mayConsume<reco::MuonCollection>       (iConfig.getParameter<edm::InputTag>("muons")     ) )   
  , met_variable_binning_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("metBinning") )
  , met_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("metPSet")    ) )
  , ls_binning_           ( getHistoLSPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("lsPSet")     ) )
  , phi_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("phiPSet")    ) )
  , pt_binning_           ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("ptPSet")    ) )
  , eta_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("etaPSet")    ) )
  , HT_binning_           ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("htPSet")    ) )
  , num_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"),consumesCollector(), *this))
  , den_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"),consumesCollector(), *this))
  , metSelection_ ( iConfig.getParameter<std::string>("metSelection") )
  , jetSelection_ ( iConfig.getParameter<std::string>("jetSelection") )
  , eleSelection_ ( iConfig.getParameter<std::string>("eleSelection") )
  , muoSelection_ ( iConfig.getParameter<std::string>("muoSelection") )
  , HTdefinition_ ( iConfig.getParameter<std::string>("HTdefinition") )
  , njets_      ( iConfig.getParameter<unsigned int>("njets" )      )
  , nelectrons_ ( iConfig.getParameter<unsigned int>("nelectrons" ) )
  , nmuons_     ( iConfig.getParameter<unsigned int>("nmuons" )     )
  , leptJetDeltaRmin_     ( iConfig.getParameter<double>("leptJetDeltaRmin" )     )
  , HTcut_     ( iConfig.getParameter<double>("HTcut" )     )
{

    METME empty;
    empty.numerator = nullptr;
    empty.denominator = nullptr;

    metME_ = empty ;   
    metME_variableBinning_ = empty ;   
    metVsLS_ = empty ;   
    metPhiME_ = empty ;   
    eventHT_ = empty ;   
    jetVsLS_ = empty ; 
    muVsLS_ = empty ; 
    eleVsLS_ = empty ; 
    htVsLS_ = empty ; 
    jetEtaPhi_ = empty; // for HEP17 monitoring

    for (unsigned int iMu=0; iMu<nmuons_; ++iMu){
        muPhi_.push_back(empty);
        muEta_.push_back(empty);
        muPt_.push_back(empty);
    }
    for (unsigned int iEle=0; iEle<nelectrons_; ++iEle){
        elePhi_.push_back(empty);
        eleEta_.push_back(empty);
        elePt_.push_back(empty);
    }
    for (unsigned int iJet=0; iJet<njets_; ++iJet){
        jetPhi_.push_back(empty);
        jetEta_.push_back(empty);
        jetPt_.push_back(empty);
    }

}

TopMonitor::~TopMonitor()
{
  if (num_genTriggerEventFlag_) delete num_genTriggerEventFlag_;
  if (den_genTriggerEventFlag_) delete den_genTriggerEventFlag_;
}

MEbinning TopMonitor::getHistoPSet(edm::ParameterSet pset)
{
  return MEbinning{
    pset.getParameter<int32_t>("nbins"),
      pset.getParameter<double>("xmin"),
      pset.getParameter<double>("xmax"),
      };
}

MEbinning TopMonitor::getHistoLSPSet(edm::ParameterSet pset)
{
  return MEbinning{
    pset.getParameter<int32_t>("nbins"),
      0.,
      double(pset.getParameter<int32_t>("nbins"))
      };
}

void TopMonitor::setMETitle(METME& me, std::string titleX, std::string titleY)
{
  me.numerator->setAxisTitle(titleX,1);
  me.numerator->setAxisTitle(titleY,2);
  me.denominator->setAxisTitle(titleX,1);
  me.denominator->setAxisTitle(titleY,2);

}

void TopMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, const std::string& histname, const std::string& histtitle, int nbins, double min, double max)
{
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, min, max);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, min, max);
}
void TopMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binning)
{
  int nbins = binning.size()-1;
  std::vector<float> fbinning(binning.begin(),binning.end());
  float* arr = &fbinning[0];
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, arr);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, arr);
}
void TopMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, double ymin, double ymax)
{
  me.numerator   = ibooker.bookProfile(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, ymin, ymax);
  me.denominator = ibooker.bookProfile(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, ymin, ymax);
}
void TopMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, int nbinsY, double ymin, double ymax)
{
  me.numerator   = ibooker.book2D(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, nbinsY, ymin, ymax);
  me.denominator = ibooker.book2D(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, nbinsY, ymin, ymax);
}
void TopMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX, const std::vector<double>& binningY)
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

void TopMonitor::bookHistograms(DQMStore::IBooker     & ibooker,
				 edm::Run const        & iRun,
				 edm::EventSetup const & iSetup) 
{  
  std::string histname, histtitle;

  std::string currentFolder = folderName_ ;
  ibooker.setCurrentFolder(currentFolder.c_str());

  histname = "met"; histtitle = "PFMET";
  bookME(ibooker,metME_,histname,histtitle,met_binning_.nbins,met_binning_.xmin, met_binning_.xmax);
  setMETitle(metME_,"PF MET [GeV]","events / [GeV]");

  histname = "met_variable"; histtitle = "PFMET";
  bookME(ibooker,metME_variableBinning_,histname,histtitle,met_variable_binning_);
  setMETitle(metME_variableBinning_,"PF MET [GeV]","events / [GeV]");

  histname = "metVsLS"; histtitle = "PFMET vs LS";
  bookME(ibooker,metVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,met_binning_.xmin, met_binning_.xmax);
  setMETitle(metVsLS_,"LS","PF MET [GeV]");

  if (njets_ > 0){
      histname = "jetVsLS"; histtitle = "jet pt vs LS";
      bookME(ibooker,jetVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,pt_binning_.xmin, pt_binning_.xmax);
      setMETitle(jetVsLS_,"LS","jet pt [GeV]");
  }
  if (nmuons_ > 0){
      histname = "muVsLS"; histtitle = "muon pt vs LS";
      bookME(ibooker,muVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,pt_binning_.xmin, pt_binning_.xmax);
      setMETitle(muVsLS_,"LS","muon pt [GeV]");
  }
  if (nelectrons_ > 0){
      histname = "eleVsLS"; histtitle = "electron pt vs LS";
      bookME(ibooker,eleVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,pt_binning_.xmin, pt_binning_.xmax);
      setMETitle(eleVsLS_,"LS","electron pt [GeV]");
  }

  histname = "htVsLS"; histtitle = "event HT vs LS";
  bookME(ibooker,htVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,pt_binning_.xmin, pt_binning_.xmax);
  setMETitle(htVsLS_,"LS","event HT [GeV]");

  histname = "metPhi"; histtitle = "PFMET phi";
  bookME(ibooker,metPhiME_,histname,histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
  setMETitle(metPhiME_,"PF MET #phi","events / 0.1 rad");

  if ((muPt_.size()!=nmuons_) || (muEta_.size()!=nmuons_) || (muPhi_.size()!=nmuons_)){
      edm::LogWarning("TopMonitor") << "Number of histograms does not match with number of required muons \n";
      return;
  } 

  for (unsigned int iMu=0; iMu<nmuons_; ++iMu){
      std::string index = std::to_string(iMu+1);

      histname = "muPt_"; histtitle = "muon p_{T} - ";
      histname.append(index); histtitle.append(index);
      bookME(ibooker,muPt_.at(iMu),histname,histtitle, pt_binning_.nbins, pt_binning_.xmin, pt_binning_.xmax);
      setMETitle(muPt_.at(iMu),"muon p_{T} [GeV]","events");

      histname = "muPhi_"; histtitle = "muon #phi - ";
      histname.append(index); histtitle.append(index);
      bookME(ibooker,muPhi_.at(iMu),histname,histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
      setMETitle(muPhi_.at(iMu)," muon #phi","events");

      histname = "muEta_"; histtitle = "muon #eta - ";
      histname.append(index); histtitle.append(index);
      bookME(ibooker,muEta_.at(iMu),histname,histtitle, eta_binning_.nbins,eta_binning_.xmin, eta_binning_.xmax);
      setMETitle(muEta_.at(iMu)," muon #eta","events");

  }

  if ((elePt_.size()!=nelectrons_) || (eleEta_.size()!=nelectrons_) || (elePhi_.size()!=nelectrons_)){
      edm::LogWarning("TopMonitor") << "Number of histograms does not match with number of required electrons \n";
      return;
  } 

  for (unsigned int iEle=0; iEle<nelectrons_; ++iEle){
      std::string index = std::to_string(iEle+1);

      histname = "elePt_"; histtitle = "electron p_{T} - ";
      histname.append(index); histtitle.append(index);
      bookME(ibooker,elePt_.at(iEle),histname,histtitle, pt_binning_.nbins, pt_binning_.xmin, pt_binning_.xmax);
      setMETitle(elePt_.at(iEle),"electron p_{T} [GeV]","events");

      histname = "elePhi_"; histtitle = "electron #phi - ";
      histname.append(index); histtitle.append(index);
      bookME(ibooker,elePhi_.at(iEle),histname,histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
      setMETitle(elePhi_.at(iEle)," electron #phi","events");

      histname = "eleEta_"; histtitle = "electron #eta - ";
      histname.append(index); histtitle.append(index);
      bookME(ibooker,eleEta_.at(iEle),histname,histtitle, eta_binning_.nbins,eta_binning_.xmin, eta_binning_.xmax);
      setMETitle(eleEta_.at(iEle)," electron #eta","events");

  }

  if ((jetPt_.size()!=njets_) || (jetEta_.size()!=njets_) || (jetPhi_.size()!=njets_)){
      edm::LogWarning("TopMonitor") << "Number of histograms does not match with number of required jets \n";
      return;
  } 

  for (unsigned int iJet=0; iJet<njets_; ++iJet){
      std::string index = std::to_string(iJet+1);

      histname = "jetPt_"; histtitle = "jet p_{T} - ";
      histname.append(index); histtitle.append(index);
      bookME(ibooker,jetPt_.at(iJet),histname,histtitle, pt_binning_.nbins, pt_binning_.xmin, pt_binning_.xmax);
      setMETitle(jetPt_.at(iJet),"jet p_{T} [GeV]","events");

      histname = "jetPhi_"; histtitle = "jet #phi - ";
      histname.append(index); histtitle.append(index);
      bookME(ibooker,jetPhi_.at(iJet),histname,histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
      setMETitle(jetPhi_.at(iJet)," jet #phi","events");

      histname = "jetEta_"; histtitle = "jet #eta - ";
      histname.append(index); histtitle.append(index);
      bookME(ibooker,jetEta_.at(iJet),histname,histtitle, eta_binning_.nbins,eta_binning_.xmin, eta_binning_.xmax);
      setMETitle(jetEta_.at(iJet)," jet #eta","events");

  }

  histname = "eventHT"; histtitle = "event HT";
  bookME(ibooker,eventHT_,histname,histtitle, HT_binning_.nbins,HT_binning_.xmin, HT_binning_.xmax);
  setMETitle(eventHT_," event HT [GeV]","events");

  histname = "jetEtaPhi"; histtitle = "jet #eta-#phi";
  bookME(ibooker,jetEtaPhi_,histname,histtitle,10,-2.5,2.5,18,-3.1415,3.1415); // for HEP17 monitoring
  setMETitle(eventHT_,"jet #eta","jet #phi");


  // Initialize the GenericTriggerEventFlag
  if ( num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on() ) num_genTriggerEventFlag_->initRun( iRun, iSetup );
  if ( den_genTriggerEventFlag_ && den_genTriggerEventFlag_->on() ) den_genTriggerEventFlag_->initRun( iRun, iSetup );

}

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
void TopMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)  {

  // Filter out events if Trigger Filtering is requested
  if (den_genTriggerEventFlag_->on() && ! den_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  edm::Handle<reco::PFMETCollection> metHandle;
  iEvent.getByToken( metToken_, metHandle );
  if (!metHandle.isValid()){
      edm::LogWarning("TopMonitor") << "MET handle not valid \n";
      return;
  }
  reco::PFMET pfmet = metHandle->front();
  if ( ! metSelection_( pfmet ) ) return;
  
  float met = pfmet.pt();
  float phi = pfmet.phi();

  edm::Handle<reco::GsfElectronCollection> eleHandle;
  iEvent.getByToken( eleToken_, eleHandle );
  if (!eleHandle.isValid()){
      edm::LogWarning("TopMonitor") << "Electron handle not valid \n";
      return;
  }
  std::vector<reco::GsfElectron> electrons;
  if ( eleHandle->size() < nelectrons_ ) return;
  for ( auto const & e : *eleHandle ) {
    if ( eleSelection_( e ) ) electrons.push_back(e);
  }
  if ( electrons.size() < nelectrons_ ) return;
  
  edm::Handle<reco::MuonCollection> muoHandle;
  iEvent.getByToken( muoToken_, muoHandle );
  if (!muoHandle.isValid()){
      edm::LogWarning("TopMonitor") << "Muon handle not valid \n";
      return;
  }
  if ( muoHandle->size() < nmuons_ ) return;
  std::vector<reco::Muon> muons;
  for ( auto const & m : *muoHandle ) {
    if ( muoSelection_( m ) ) muons.push_back(m);
  }
  if ( muons.size() < nmuons_ ) return;

  double eventHT = 0.;

  edm::Handle<reco::PFJetCollection> jetHandle;
  iEvent.getByToken( jetToken_, jetHandle );
  if (!jetHandle.isValid()){
      edm::LogWarning("TopMonitor") << "Jet handle not valid \n";
      return;
  }
  std::vector<reco::PFJet> jets;
  if ( jetHandle->size() < njets_ ) return;
  for ( auto const & j : *jetHandle ) {
      if ( HTdefinition_ ( j ) ){
          eventHT += j.pt();
      }
      if ( jetSelection_( j ) ){
          bool isJetOverlappedWithLepton = false;
          TLorentzVector jp4; jp4.SetPtEtaPhiE(j.pt(),j.eta(),j.phi(),j.energy());
          if(nmuons_>0){
              for (auto const m : muons){
                  TLorentzVector mp4; mp4.SetPtEtaPhiE(m.pt(),m.eta(),m.phi(),m.energy());
                  if (mp4.DeltaR(jp4)<leptJetDeltaRmin_){
                      isJetOverlappedWithLepton=true;
                      break;
                  }
              }
          }
          if (isJetOverlappedWithLepton) continue;
          if(nelectrons_>0){
              for (auto const e : electrons){
                  TLorentzVector ep4; ep4.SetPtEtaPhiE(e.pt(),e.eta(),e.phi(),e.energy());
                  if (ep4.DeltaR(jp4)<leptJetDeltaRmin_){
                      isJetOverlappedWithLepton=true;
                      break;
                  }
              }
          }
          if (isJetOverlappedWithLepton) continue;
          jets.push_back(j);
      }

  }
  if ( jets.size() < njets_ ) return;

  if (eventHT < HTcut_) return;

  // filling histograms (denominator)  
  metME_.denominator -> Fill(met);
  metME_variableBinning_.denominator -> Fill(met);
  metPhiME_.denominator -> Fill(phi);
  eventHT_.denominator -> Fill(eventHT);

  int ls = iEvent.id().luminosityBlock();
  metVsLS_.denominator -> Fill(ls, met);
  htVsLS_.denominator -> Fill(ls, eventHT);

  if (nmuons_ > 0)     muVsLS_.denominator -> Fill(ls, muons.at(0).pt());
  if (nelectrons_ > 0) eleVsLS_.denominator -> Fill(ls, electrons.at(0).pt());
  if (njets_ > 0)      jetVsLS_.denominator -> Fill(ls, jets.at(0).pt());

  if ((muPt_.size()!=nmuons_) || (muEta_.size()!=nmuons_) || (muPhi_.size()!=nmuons_)){
      edm::LogWarning("TopMonitor") << "Number of histograms does not match with number of required muons \n";
      return;
  } 
  if ((elePt_.size()!=nelectrons_) || (eleEta_.size()!=nelectrons_) || (elePhi_.size()!=nelectrons_)){
      edm::LogWarning("TopMonitor") << "Number of histograms does not match with number of required electrons \n";
      return;
  } 
  if ((jetPt_.size()!=njets_) || (jetEta_.size()!=njets_) || (jetPhi_.size()!=njets_)){
      edm::LogWarning("TopMonitor") << "Number of histograms does not match with number of required jets \n";
      return;
  } 


  for (unsigned int iMu=0; iMu<muons.size(); ++iMu){
      if (iMu>=nmuons_) break;
      muPhi_.at(iMu).denominator  -> Fill(muons.at(iMu).phi());
      muEta_.at(iMu).denominator  -> Fill(muons.at(iMu).eta());
      muPt_.at(iMu).denominator   -> Fill(muons.at(iMu).pt() );
  }
  for (unsigned int iEle=0; iEle<electrons.size(); ++iEle){
      if (iEle>=nelectrons_) break;
      elePhi_.at(iEle).denominator  -> Fill(electrons.at(iEle).phi());
      eleEta_.at(iEle).denominator  -> Fill(electrons.at(iEle).eta());
      elePt_.at(iEle).denominator   -> Fill(electrons.at(iEle).pt() );
  }
  for (unsigned int iJet=0; iJet<jets.size(); ++iJet){
      if (iJet>=njets_) break;
      jetPhi_.at(iJet).denominator  -> Fill(jets.at(iJet).phi());
      jetEta_.at(iJet).denominator  -> Fill(jets.at(iJet).eta());
      jetPt_.at(iJet).denominator   -> Fill(jets.at(iJet).pt() );
  }

  if (jets.size() > 0){
      jetEtaPhi_.denominator -> Fill (jets.at(0).eta(), jets.at(0).phi()); // for HEP17 monitorning
  }

  // applying selection for numerator
  if (num_genTriggerEventFlag_->on() && ! num_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  // filling histograms (num_genTriggerEventFlag_)  
  metME_.numerator -> Fill(met);
  metME_variableBinning_.numerator -> Fill(met);
  metPhiME_.numerator -> Fill(phi);
  metVsLS_.numerator -> Fill(ls, met);
  htVsLS_.numerator -> Fill(ls, eventHT);
  eventHT_.numerator -> Fill(eventHT);

  if (nmuons_ > 0)     muVsLS_.numerator -> Fill(ls, muons.at(0).pt());
  if (nelectrons_ > 0) eleVsLS_.numerator -> Fill(ls, electrons.at(0).pt());
  if (njets_ > 0)      jetVsLS_.numerator -> Fill(ls, jets.at(0).pt());

  for (unsigned int iMu=0; iMu<muons.size(); ++iMu){
      if (iMu>=nmuons_) break;
      muPhi_.at(iMu).numerator  -> Fill(muons.at(iMu).phi());
      muEta_.at(iMu).numerator  -> Fill(muons.at(iMu).eta());
      muPt_.at(iMu).numerator   -> Fill(muons.at(iMu).pt() );
  }
  for (unsigned int iEle=0; iEle<electrons.size(); ++iEle){
      if (iEle>=nelectrons_) break;
      elePhi_.at(iEle).numerator  -> Fill(electrons.at(iEle).phi());
      eleEta_.at(iEle).numerator  -> Fill(electrons.at(iEle).eta());
      elePt_.at(iEle).numerator   -> Fill(electrons.at(iEle).pt() );
  }
  for (unsigned int iJet=0; iJet<jets.size(); ++iJet){
      if (iJet>=njets_) break;
      jetPhi_.at(iJet).numerator  -> Fill(jets.at(iJet).phi());
      jetEta_.at(iJet).numerator  -> Fill(jets.at(iJet).eta());
      jetPt_.at(iJet).numerator   -> Fill(jets.at(iJet).pt() );
  }

  if (jets.size() > 0){
      jetEtaPhi_.numerator -> Fill (jets.at(0).eta(), jets.at(0).phi()); // for HEP17 monitorning
  }


}

void TopMonitor::fillHistoPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<int>   ( "nbins");
  pset.add<double>( "xmin" );
  pset.add<double>( "xmax" );
}

void TopMonitor::fillHistoLSPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<int>   ( "nbins", 2500);
}

void TopMonitor::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<std::string>  ( "FolderName", "HLT/TOP" );

  desc.add<edm::InputTag>( "met",      edm::InputTag("pfMet") );
  desc.add<edm::InputTag>( "jets",     edm::InputTag("ak4PFJetsCHS") );
  desc.add<edm::InputTag>( "electrons",edm::InputTag("gedGsfElectrons") );
  desc.add<edm::InputTag>( "muons",    edm::InputTag("muons") );
  desc.add<std::string>("metSelection", "pt > 0");
  desc.add<std::string>("jetSelection", "pt > 0");
  desc.add<std::string>("eleSelection", "pt > 0");
  desc.add<std::string>("muoSelection", "pt > 0");
  desc.add<std::string>("HTdefinition", "pt > 0");
  desc.add<unsigned int>("njets",      0);
  desc.add<unsigned int>("nelectrons", 0);
  desc.add<unsigned int>("nmuons",     0);
  desc.add<double>("leptJetDeltaRmin", 0);
  desc.add<double>("HTcut", 0);

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
  edm::ParameterSetDescription metPSet;
  edm::ParameterSetDescription phiPSet;
  edm::ParameterSetDescription etaPSet;
  edm::ParameterSetDescription ptPSet;
  edm::ParameterSetDescription htPSet;
  fillHistoPSetDescription(metPSet);
  fillHistoPSetDescription(phiPSet);
  fillHistoPSetDescription(ptPSet);
  fillHistoPSetDescription(etaPSet);
  fillHistoPSetDescription(htPSet);
  histoPSet.add<edm::ParameterSetDescription>("metPSet", metPSet);
  histoPSet.add<edm::ParameterSetDescription>("etaPSet", etaPSet);
  histoPSet.add<edm::ParameterSetDescription>("phiPSet", phiPSet);
  histoPSet.add<edm::ParameterSetDescription>("ptPSet", ptPSet);
  histoPSet.add<edm::ParameterSetDescription>("htPSet", htPSet);
  std::vector<double> bins = {0.,20.,40.,60.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.,190.,200.,220.,240.,260.,280.,300.,350.,400.,450.,1000.};
  histoPSet.add<std::vector<double> >("metBinning", bins);

  edm::ParameterSetDescription lsPSet;
  fillHistoLSPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);

  desc.add<edm::ParameterSetDescription>("histoPSet",histoPSet);

  descriptions.add("topMonitoring", desc);
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TopMonitor);
