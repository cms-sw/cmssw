/*   A macro for making a histogram of Jet Pt with cuts
This is a basic way to cut out jets of a certain Pt and Eta using an if statement
This example creates a histogram of Jet Pt, using Jets with Pt above 30 and ETA above -2.1 and below 2.1
*/

#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TSystem.h"
#include "TLorentzVector.h"

#include "PhysicsTools/SelectorUtils/interface/strbitset.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "PhysicsTools/SelectorUtils/interface/Selector.h"
#include "PhysicsTools/SelectorUtils/interface/JetIDSelectionFunctor.h"
#include "PhysicsTools/SelectorUtils/interface/PFJetIDSelectionFunctor.h"
#include "PhysicsTools/SelectorUtils/interface/RunLumiSelector.h"
#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "DataFormats/FWLite/interface/ChainEvent.h"
#include "FWCore/PythonParameterSet/interface/PythonProcessDesc.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"


#include <iostream>
#include <memory>
#include <cmath>      //necessary for absolute function fabs()

using namespace std;

///-------------------------
/// DRIVER FUNCTION
///-------------------------

// -*- C++ -*-

// CMS includes
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "PhysicsTools/SelectorUtils/interface/EventSelector.h"


// Root includes
#include "TROOT.h"

using namespace std;


class JetIDStudiesSelector : public EventSelector {
public:
  JetIDStudiesSelector( edm::ParameterSet const & caloJetIdParams,
			edm::ParameterSet const & pfJetIdParams,
			edm::ParameterSet const & params ) :
    jetSel_    (new JetIDSelectionFunctor(caloJetIdParams)), 
    pfJetSel_  (new PFJetIDSelectionFunctor(pfJetIdParams)),
    jetSrc_    (params.getParameter<edm::InputTag>("jetSrc")), 
    pfJetSrc_  (params.getParameter<edm::InputTag>("pfJetSrc")) 
  {
    bool useCalo = params.getParameter<bool>("useCalo");
	       
    push_back("Calo Cuts");
    push_back("Calo Kin Cuts");
    push_back("Calo Delta Phi");
    push_back("Calo Jet ID");
    push_back("PF Cuts");
    push_back("PF Kin Cuts");
    push_back("PF Delta Phi");
    push_back("PF Jet ID");

    set("Calo Cuts", useCalo);
    set("Calo Kin Cuts", useCalo);
    set("Calo Delta Phi", useCalo);
    set("Calo Jet ID", useCalo);
    set("PF Cuts", !useCalo);
    set("PF Kin Cuts", !useCalo);
    set("PF Delta Phi", !useCalo);    
    set("PF Jet ID", !useCalo);

    // Indices for fast caching
    caloCuts_      = index_type(&bits_, std::string("Calo Cuts") );
    caloKin_       = index_type(&bits_, std::string("Calo Kin Cuts"));
    caloDeltaPhi_  = index_type(&bits_, std::string("Calo Delta Phi"));
    caloJetID_     = index_type(&bits_, std::string("Calo Jet ID"));
    pfCuts_        = index_type(&bits_, std::string("PF Cuts"));
    pfKin_         = index_type(&bits_, std::string("PF Kin Cuts"));
    pfDeltaPhi_    = index_type(&bits_, std::string("PF Delta Phi"));
    pfJetID_       = index_type(&bits_, std::string("PF Jet ID"));
    
  }

  virtual ~JetIDStudiesSelector() {}

  virtual bool operator()( edm::EventBase const & event, pat::strbitset & ret) override{

    pat::strbitset retCaloJet = jetSel_->getBitTemplate();
    pat::strbitset retPFJet = pfJetSel_->getBitTemplate();


    if ( considerCut(caloCuts_) ) {
      passCut(ret, caloCuts_);
      event.getByLabel( jetSrc_, h_jets_ );
      // Calo Cuts
      if ( h_jets_->size() >= 2 || ignoreCut(caloKin_) ) {
	passCut(ret, caloKin_);
	pat::Jet const & jet0 = h_jets_->at(0);
	pat::Jet const & jet1 = h_jets_->at(1);
	double dphi = fabs(deltaPhi<double>( jet0.phi(),
					     jet1.phi() ) );
	  
	if ( fabs(dphi - TMath::Pi()) < 1.0 || ignoreCut(caloDeltaPhi_) ) {
	  passCut(ret, caloDeltaPhi_);

	    
	  retCaloJet.set(false);
	  bool pass0 = (*jetSel_)( jet0, retCaloJet );
	  retCaloJet.set(false);
	  bool pass1 = (*jetSel_)( jet1, retCaloJet );
	  if ( (pass0 && pass1) || ignoreCut(caloJetID_) ) {
	    passCut(ret, caloJetID_);
	    caloJet0_ = edm::Ptr<pat::Jet>( h_jets_, 0);
	    caloJet1_ = edm::Ptr<pat::Jet>( h_jets_, 1);

	    return (bool)ret;
	  }// end if found 2 "loose" jet ID jets
	}// end if delta phi
      }// end calo kin cuts
    }// end if calo cuts


    if ( considerCut(pfCuts_) ) {

      passCut(ret, pfCuts_);
      event.getByLabel( pfJetSrc_, h_pfjets_ );
      // PF Cuts
      if ( h_pfjets_->size() >= 2 || ignoreCut(pfKin_) ) {
	passCut( ret, pfKin_);
	pat::Jet const & jet0 = h_pfjets_->at(0);
	pat::Jet const & jet1 = h_pfjets_->at(1);
	double dphi = fabs(deltaPhi<double>( jet0.phi(),
					     jet1.phi() ) );
	  
	if ( fabs(dphi - TMath::Pi()) < 1.0 || ignoreCut(pfDeltaPhi_) ) {
	  passCut(ret, pfDeltaPhi_);

	    
	  retPFJet.set(false);
	  bool pass0 = (*pfJetSel_)( jet0, retPFJet );
	  retPFJet.set(false);
	  bool pass1 = (*pfJetSel_)( jet1, retPFJet );
	  if ( (pass0 && pass1) || ignoreCut(pfJetID_) ) {
	    passCut(ret, pfJetID_);
	    pfJet0_ = edm::Ptr<pat::Jet>( h_pfjets_, 0);
	    pfJet1_ = edm::Ptr<pat::Jet>( h_pfjets_, 1);

	    return (bool)ret;
	  }// end if found 2 "loose" jet ID jets
	}// end if delta phi
      }// end pf kin cuts
    }// end if pf cuts


    setIgnored(ret);

    return false;
  }// end of method

  std::shared_ptr<JetIDSelectionFunctor> const &   jetSel()     const { return jetSel_;}
  std::shared_ptr<PFJetIDSelectionFunctor> const & pfJetSel()   const { return pfJetSel_;}

  vector<pat::Jet>            const &   allCaloJets () const { return *h_jets_; }
  vector<pat::Jet>            const &   allPFJets   () const { return *h_pfjets_; }

  pat::Jet                    const &   caloJet0() const { return *caloJet0_; }
  pat::Jet                    const &   caloJet1() const { return *caloJet1_; }

  pat::Jet                    const &   pfJet0() const { return *pfJet0_; }
  pat::Jet                    const &   pfJet1() const { return *pfJet1_; }


  // Fast caching indices
  index_type const &   caloCuts()        const {return caloCuts_;}      
  index_type const &   caloKin()         const {return caloKin_;}       
  index_type const &   caloDeltaPhi()    const {return caloDeltaPhi_;}  
  index_type const &   caloJetID()       const {return caloJetID_;}     
  index_type const &   pfCuts()          const {return pfCuts_;}        
  index_type const &   pfKin()           const {return pfKin_;}         
  index_type const &   pfDeltaPhi()      const {return pfDeltaPhi_;}    
  index_type const &   pfJetID()         const {return pfJetID_;}       


protected:
  std::shared_ptr<JetIDSelectionFunctor>   jetSel_;
  std::shared_ptr<PFJetIDSelectionFunctor> pfJetSel_;
  edm::InputTag                              jetSrc_;
  edm::InputTag                              pfJetSrc_;
  
  edm::Handle<vector<pat::Jet> >             h_jets_;
  edm::Handle<vector<pat::Jet> >             h_pfjets_;

  edm::Ptr<pat::Jet>                         caloJet0_;
  edm::Ptr<pat::Jet>                         caloJet1_;

  edm::Ptr<pat::Jet>                         pfJet0_;
  edm::Ptr<pat::Jet>                         pfJet1_;

  // Fast caching indices
  index_type   caloCuts_;      
  index_type   caloKin_;       
  index_type   caloDeltaPhi_;  
  index_type   caloJetID_;     
  index_type   pfCuts_;        
  index_type   pfKin_;         
  index_type   pfDeltaPhi_;    
  index_type   pfJetID_;       

};

///////////////////////////
// ///////////////////// //
// // Main Subroutine // //
// ///////////////////// //
///////////////////////////

int main (int argc, char* argv[]) 
{


  if ( argc < 2 ) {
    std::cout << "Usage : " << argv[0] << " [parameters.py]" << std::endl;
    return 0;
  }

  cout << "Hello from " << argv[0] << "!" << endl;


  // load framework libraries
  gSystem->Load( "libFWCoreFWLite" );
  FWLiteEnabler::enable();


  cout << "Getting parameters" << endl;
  // Get the python configuration
  PythonProcessDesc builder(argv[1]);
  std::shared_ptr<edm::ProcessDesc> b = builder.processDesc();
  std::shared_ptr<edm::ParameterSet> parameters = b->getProcessPSet();
  parameters->registerIt(); 

  edm::ParameterSet const& jetStudiesParams    = parameters->getParameter<edm::ParameterSet>("jetStudies");
  edm::ParameterSet const& pfJetStudiesParams  = parameters->getParameter<edm::ParameterSet>("pfJetStudies");
  edm::ParameterSet const& caloJetIDParameters = parameters->getParameter<edm::ParameterSet>("jetIDSelector");
  edm::ParameterSet const& pfJetIDParameters   = parameters->getParameter<edm::ParameterSet>("pfJetIDSelector");
  edm::ParameterSet const& plotParameters      = parameters->getParameter<edm::ParameterSet>("plotParameters");
  edm::ParameterSet const& inputs              = parameters->getParameter<edm::ParameterSet>("inputs");
  edm::ParameterSet const& outputs             = parameters->getParameter<edm::ParameterSet>("outputs");

  cout << "Making RunLumiSelector" << endl;
  RunLumiSelector runLumiSel( inputs );
  
  cout << "setting up TFileService" << endl;
  // book a set of histograms
  fwlite::TFileService fs = fwlite::TFileService( outputs.getParameter<std::string>("outputName") );
  TFileDirectory theDir = fs.mkdir( "histos" ); 
    
  cout << "Setting up chain event" << endl;
  // This object 'event' is used both to get all information from the
  // event as well as to store histograms, etc.
  fwlite::ChainEvent ev ( inputs.getParameter<std::vector<std::string> > ("fileNames") );


  cout << "Booking histograms" << endl;
   // Book histograms

  std::map<std::string, TH1*> hists;

   hists["hist_nJet"                     ] = theDir.make<TH1D>( "hist_nJet"                    ,"Number of Calo Jets", 10, 0, 10 ) ;
   hists["hist_nPFJet"                   ] = theDir.make<TH1D>( "hist_nPFJet"                  ,"Number of PF Jets", 10, 0, 10 ) ;

   hists["hist_jetPt"                    ] = theDir.make<TH1D>( "hist_jetPt"                   , "Jet p_{T}", 400, 0, 400 ) ;
   hists["hist_jetEtaVsPhi"              ] = theDir.make<TH2D>( "hist_jetEtaVsPhi"             , "Jet #phi versus #eta;#eta;#phi", 50, -5.0, 5.0, 50, -TMath::Pi(), TMath::Pi() ) ;
   hists["hist_jetNTracks"               ] = theDir.make<TH1D>( "hist_jetNTracks"              , "Jet N_{TRACKS}", 20, 0, 20 ) ;
   hists["hist_jetNTracksVsPt"           ] = theDir.make<TH2D>( "hist_jetNTracksVsPt"          , "Number of Tracks versus Jet p_{T};Jet p_{T}(GeV/c) ;N_{Tracks}",20, 0, 200, 20, 0, 20 ) ;
   hists["hist_jetEMF"                   ] = theDir.make<TH1D>( "hist_jetEMF"                  , "Jet EMF", 200, 0, 1) ;
   hists["hist_jetPdgID"                 ] = theDir.make<TH1D>( "hist_jetPdgID"                , "PDG Id of Jet Constituents", 10000, 0, 10000 ) ;
   hists["hist_jetGenEmE"                ] = theDir.make<TH1D>( "hist_jetGenEmE"               , "Gen Jet EM Energy", 200, 0, 200 ) ;
   hists["hist_jetGenHadE"               ] = theDir.make<TH1D>( "hist_jetGenHadE"              , "Gen Jet HAD Energy", 200, 0, 200 ) ;
   hists["hist_jetGenEMF"                ] = theDir.make<TH1D>( "hist_jetGenEMF"               , "Gen Jet EMF", 200, 0, 1) ;
   hists["hist_jetEoverGenE"             ] = theDir.make<TH1D>( "hist_jetEoverGenE"            , "Energy of reco Jet / Energy of gen Jet", 200, 0, 2.0) ;
   hists["hist_jetCorr"                  ] = theDir.make<TH1D>( "hist_jetCorr"                 , "Jet Correction Factor", 200, 0, 1.0 ) ;
   hists["hist_n90Hits"                  ] = theDir.make<TH1D>( "hist_n90Hits"                 , "Jet n90Hits", 20, 0, 20) ;
   hists["hist_fHPD"                     ] = theDir.make<TH1D>( "hist_fHPD"                    , "Jet fHPD", 200, 0, 1) ;
   hists["hist_nConstituents"            ] = theDir.make<TH1D>( "hist_nConstituents"           , "Jet nConstituents", 20, 0, 20 ) ;
   hists["hist_jetCHF"                   ] = theDir.make<TH1D>( "hist_jetCHF"                  , "Jet Charged Hadron Fraction", 200, 0, 1.0) ;
	                                      
   hists["hist_good_jetPt"             ] = theDir.make<TH1D>( "hist_good_jetPt"            , "Jet p_{T}", 400, 0, 400 ) ;
   hists["hist_good_jetEtaVsPhi"       ] = theDir.make<TH2D>( "hist_good_jetEtaVsPhi"      , "Jet #phi versus #eta;#eta;#phi", 50, -5.0, 5.0, 50, -TMath::Pi(), TMath::Pi() ) ;
   hists["hist_good_jetNTracks"        ] = theDir.make<TH1D>( "hist_good_jetNTracks"       , "Jet N_{TRACKS}", 20, 0, 20 ) ;
   hists["hist_good_jetNTracksVsPt"    ] = theDir.make<TH2D>( "hist_good_jetNTracksVsPt"   , "Number of Tracks versus Jet p_{T};Jet p_{T}(GeV/c) ;N_{Tracks}",20, 0, 200, 20, 0, 20 ) ;
   hists["hist_good_jetEMF"            ] = theDir.make<TH1D>( "hist_good_jetEMF"           , "Jet EMF", 200, 0, 1) ;
   hists["hist_good_jetPdgID"          ] = theDir.make<TH1D>( "hist_good_jetPdgID"         , "PDG Id of Jet Constituents", 10000, 0, 10000 ) ;
   hists["hist_good_jetGenEmE"         ] = theDir.make<TH1D>( "hist_good_jetGenEmE"        , "Gen Jet EM Energy", 200, 0, 200 ) ;
   hists["hist_good_jetGenHadE"        ] = theDir.make<TH1D>( "hist_good_jetGenHadE"       , "Gen Jet HAD Energy", 200, 0, 200 ) ;
   hists["hist_good_jetGenEMF"         ] = theDir.make<TH1D>( "hist_good_jetGenEMF"        , "Gen Jet EMF", 200, 0, 1) ;
   hists["hist_good_jetEoverGenE"      ] = theDir.make<TH1D>( "hist_good_jetEoverGenE"     , "Energy of reco Jet / Energy of gen Jet", 200, 0, 2.0) ;
   hists["hist_good_jetCorr"           ] = theDir.make<TH1D>( "hist_good_jetCorr"          , "Jet Correction Factor", 200, 0, 1.0 ) ;
   hists["hist_good_n90Hits"           ] = theDir.make<TH1D>( "hist_good_n90Hits"          , "Jet n90Hits", 20, 0, 20) ;
   hists["hist_good_fHPD"              ] = theDir.make<TH1D>( "hist_good_fHPD"             , "Jet fHPD", 200, 0, 1) ;
   hists["hist_good_nConstituents"     ] = theDir.make<TH1D>( "hist_good_nConstituents"    , "Jet nConstituents", 20, 0, 20 ) ;
   hists["hist_good_jetCHF"            ] = theDir.make<TH1D>( "hist_good_jetCHF"           , "Jet Charged Hadron Fraction", 200, 0, 1.0) ;
   	                                                  
	                                      
   hists["hist_pf_jetPt"                 ] = theDir.make<TH1D>( "hist_pf_jetPt"                , "PFJet p_{T}", 400, 0, 400 ) ;
   hists["hist_pf_jetEtaVsPhi"           ] = theDir.make<TH2D>( "hist_pf_jetEtaVsPhi"          , "PFJet #phi versus #eta;#eta;#phi", 50, -5.0, 5.0, 50, -TMath::Pi(), TMath::Pi() ) ;
   hists["hist_pf_jetNTracks"            ] = theDir.make<TH1D>( "hist_pf_jetNTracks"           , "PFJet N_{TRACKS}", 20, 0, 20 ) ;
   hists["hist_pf_jetNTracksVsPt"        ] = theDir.make<TH2D>( "hist_pf_jetNTracksVsPt"       , "Number of Tracks versus Jet p_{T};Jet p_{T}(GeV/c) ;N_{Tracks}",20, 0, 200, 20, 0, 20 ) ;
   hists["hist_pf_jetEMF"                ] = theDir.make<TH1D>( "hist_pf_jetEMF"               , "PFJet EMF", 200, 0, 1) ;
   hists["hist_pf_jetCHF"                ] = theDir.make<TH1D>( "hist_pf_jetCHF"               , "PFJet CHF", 200, 0, 1) ;
   hists["hist_pf_jetNHF"                ] = theDir.make<TH1D>( "hist_pf_jetNHF"               , "PFJet NHF", 200, 0, 1) ;
   hists["hist_pf_jetCEF"                ] = theDir.make<TH1D>( "hist_pf_jetCEF"               , "PFJet CEF", 200, 0, 1) ;
   hists["hist_pf_jetNEF"                ] = theDir.make<TH1D>( "hist_pf_jetNEF"               , "PFJet NEF", 200, 0, 1) ;
   hists["hist_pf_jetPdgID"              ] = theDir.make<TH1D>( "hist_pf_jetPdgID"             , "PDG Id of Jet Constituents", 10000, 0, 10000 ) ;
   hists["hist_pf_jetGenEmE"             ] = theDir.make<TH1D>( "hist_pf_jetGenEmE"            , "Gen Jet EM Energy", 200, 0, 200 ) ;
   hists["hist_pf_jetGenHadE"            ] = theDir.make<TH1D>( "hist_pf_jetGenHadE"           , "Gen Jet HAD Energy", 200, 0, 200 ) ;
   hists["hist_pf_jetGenEMF"             ] = theDir.make<TH1D>( "hist_pf_jetGenEMF"            , "Gen Jet EMF", 200, 0, 1) ;
   hists["hist_pf_jetEoverGenE"          ] = theDir.make<TH1D>( "hist_pf_jetEoverGenE"         , "Energy of reco Jet / Energy of gen Jet", 200, 0, 2.0) ;
   hists["hist_pf_jetCorr"               ] = theDir.make<TH1D>( "hist_pf_jetCorr"              , "PFJet Correction Factor", 200, 0, 1.0 ) ;
   hists["hist_pf_nConstituents"         ] = theDir.make<TH1D>( "hist_pf_nConstituents"        , "PFJet nConstituents", 20, 0, 20 ) ;
	                                      
   hists["hist_pf_good_jetPt"          ] = theDir.make<TH1D>( "hist_pf_good_jetPt"         , "PFJet p_{T}", 400, 0, 400 ) ;
   hists["hist_pf_good_jetEtaVsPhi"    ] = theDir.make<TH2D>( "hist_pf_good_jetEtaVsPhi"   , "PFJet #phi versus #eta;#eta;#phi", 50, -5.0, 5.0, 50, -TMath::Pi(), TMath::Pi() ) ;
   hists["hist_pf_good_jetNTracks"     ] = theDir.make<TH1D>( "hist_pf_good_jetNTracks"    , "PFJet N_{TRACKS}", 20, 0, 20 ) ;
   hists["hist_pf_good_jetNTracksVsPt" ] = theDir.make<TH2D>( "hist_pf_good_jetNTracksVsPt", "Number of Tracks versus Jet p_{T};Jet p_{T}(GeV/c) ;N_{Tracks}",20, 0, 200, 20, 0, 20 ) ;
   hists["hist_pf_good_jetEMF"         ] = theDir.make<TH1D>( "hist_pf_good_jetEMF"        , "PFJet EMF", 200, 0, 1) ;
   hists["hist_pf_good_jetCHF"         ] = theDir.make<TH1D>( "hist_pf_good_jetCHF"        , "PFJet CHF", 200, 0, 1) ;
   hists["hist_pf_good_jetNHF"         ] = theDir.make<TH1D>( "hist_pf_good_jetNHF"        , "PFJet NHF", 200, 0, 1) ;
   hists["hist_pf_good_jetCEF"         ] = theDir.make<TH1D>( "hist_pf_good_jetCEF"        , "PFJet CEF", 200, 0, 1) ;
   hists["hist_pf_good_jetNEF"         ] = theDir.make<TH1D>( "hist_pf_good_jetNEF"        , "PFJet NEF", 200, 0, 1) ;
   hists["hist_pf_good_jetPdgID"       ] = theDir.make<TH1D>( "hist_pf_good_jetPdgID"      , "PDG Id of Jet Constituents", 10000, 0, 10000 ) ;
   hists["hist_pf_good_jetGenEmE"      ] = theDir.make<TH1D>( "hist_pf_good_jetGenEmE"     , "Gen Jet EM Energy", 200, 0, 200 ) ;
   hists["hist_pf_good_jetGenHadE"     ] = theDir.make<TH1D>( "hist_pf_good_jetGenHadE"    , "Gen Jet HAD Energy", 200, 0, 200 ) ;
   hists["hist_pf_good_jetGenEMF"      ] = theDir.make<TH1D>( "hist_pf_good_jetGenEMF"     , "Gen Jet EMF", 200, 0, 1) ;
   hists["hist_pf_good_jetEoverGenE"   ] = theDir.make<TH1D>( "hist_pf_good_jetEoverGenE"  , "Energy of reco Jet / Energy of gen Jet", 200, 0, 2.0) ;
   hists["hist_pf_good_jetCorr"        ] = theDir.make<TH1D>( "hist_pf_good_jetCorr"       , "PFJet Correction Factor", 200, 0, 1.0 ) ;
   hists["hist_pf_good_nConstituents"  ] = theDir.make<TH1D>( "hist_pf_good_nConstituents" , "PFJet nConstituents", 20, 0, 20 ) ;

   hists["hist_mjj"                           ] = theDir.make<TH1D>( "hist_mjj"                          , "Dijet mass", 300, 0, 300 ) ;
   hists["hist_dR_jj"                         ] = theDir.make<TH1D>( "hist_dR_jj"                        , "#Delta R_{JJ}", 200, 0, TMath::TwoPi() ) ;
   hists["hist_imbalance_jj"                  ] = theDir.make<TH1D>( "hist_imbalance_jj"                 , "Dijet imbalance", 200, -1.0, 1.0 )  ;
	                                      
   hists["hist_pf_mjj"                        ] = theDir.make<TH1D>( "hist_pf_mjj"                       , "Dijet mass", 300, 0, 300 ) ;
   hists["hist_pf_dR_jj"                      ] = theDir.make<TH1D>( "hist_pf_dR_jj"                     , "#Delta R_{JJ}", 200, 0, TMath::TwoPi() ) ;
   hists["hist_pf_imbalance_jj"               ] = theDir.make<TH1D>( "hist_pf_imbalance_jj"              , "Dijet imbalance", 200, -1.0, 1.0 )  ;


   
   cout << "Making functors" << endl;
   JetIDStudiesSelector caloSelector( caloJetIDParameters,
				      pfJetIDParameters,
				      jetStudiesParams );

   JetIDStudiesSelector pfSelector( caloJetIDParameters,
				    pfJetIDParameters,
				    pfJetStudiesParams );

   bool doTracks = plotParameters.getParameter<bool>("doTracks");
   bool useMC    = plotParameters.getParameter<bool>("useMC");

   cout << "About to begin looping" << endl;

  int nev = 0;
  //loop through each event
  for (ev.toBegin(); ! ev.atEnd(); ++ev, ++nev) {


    edm::EventBase const & event = ev;

    if ( runLumiSel(ev) == false ) continue;

    if ( nev % 100 == 0 ) cout << "Processing run " << event.id().run() << ", lumi " << event.id().luminosityBlock() << ", event " << event.id().event() << endl;




    pat::strbitset retCalo = caloSelector.getBitTemplate();
    caloSelector( event, retCalo );


    pat::strbitset retPF = pfSelector.getBitTemplate();
    pfSelector( event, retPF );

    ///------------------
    /// CALO JETS
    ///------------------
    if ( retCalo.test( caloSelector.caloKin() ) ) {
      if ( retCalo.test( caloSelector.caloDeltaPhi() ) ) {
	vector<pat::Jet>  const & allCaloJets = caloSelector.allCaloJets();

	for ( std::vector<pat::Jet>::const_iterator jetBegin = allCaloJets.begin(),
		jetEnd = jetBegin + 2, ijet = jetBegin;
	      ijet != jetEnd; ++ijet ) {
	
	  const pat::Jet & jet = *ijet;

	  double pt = jet.pt();

	  const reco::TrackRefVector & jetTracks = jet.associatedTracks();

	  hists["hist_jetPt"]->Fill( pt );
	  hists["hist_jetEtaVsPhi"]->Fill( jet.eta(), jet.phi() );
	  hists["hist_jetNTracks"]->Fill( jetTracks.size() );
	  hists["hist_jetNTracksVsPt"]->Fill( pt, jetTracks.size() );
	  hists["hist_jetEMF"]->Fill( jet.emEnergyFraction() );	
	  hists["hist_jetCorr"]->Fill( jet.jecFactor("Uncorrected") );
	  hists["hist_n90Hits"]->Fill( static_cast<int>(jet.jetID().n90Hits) );
	  hists["hist_fHPD"]->Fill( jet.jetID().fHPD );
	  hists["hist_nConstituents"]->Fill( jet.nConstituents() );

	  if ( useMC && jet.genJet() != 0 ) {
	    hists["hist_jetGenEmE"]->Fill( jet.genJet()->emEnergy() );
	    hists["hist_jetGenHadE"]->Fill( jet.genJet()->hadEnergy() );
	    hists["hist_jetEoverGenE"]->Fill( jet.energy() / jet.genJet()->energy() );
	    hists["hist_jetGenEMF"]->Fill( jet.genJet()->emEnergy() / jet.genJet()->energy() );
	  }
	  if ( doTracks ) {
	    TLorentzVector p4_tracks(0,0,0,0);
	    for ( reco::TrackRefVector::const_iterator itrk = jetTracks.begin(),
		    itrkEnd = jetTracks.end();
		  itrk != itrkEnd; ++itrk ) {
	      TLorentzVector p4_trk;
	      double M_PION = 0.140;
	      p4_trk.SetPtEtaPhiM( (*itrk)->pt(), (*itrk)->eta(), (*itrk)->phi(), M_PION );
	      p4_tracks += p4_trk;
	    }
	    hists["hist_jetCHF"]->Fill( p4_tracks.Energy() / jet.energy() );
	  }

      
	} // end loop over jets

    

	if ( retCalo.test( caloSelector.caloJetID() ) ) {
	  pat::Jet const & jet0 = caloSelector.caloJet0();
	  pat::Jet const & jet1 = caloSelector.caloJet1();

	  TLorentzVector p4_j0( jet0.px(), jet0.py(), jet0.pz(), jet0.energy() );
	  TLorentzVector p4_j1( jet1.px(), jet1.py(), jet1.pz(), jet1.energy() );

	  TLorentzVector p4_jj = p4_j0 + p4_j1;

	  hists["hist_mjj"]->Fill( p4_jj.M() );
	  hists["hist_dR_jj"]->Fill( p4_j0.DeltaR( p4_j1 ) );
	  hists["hist_imbalance_jj"]->Fill( (p4_j0.Perp() - p4_j1.Perp() ) /
						     (p4_j0.Perp() + p4_j1.Perp() ) );

	  hists["hist_good_jetPt"]->Fill( jet0.pt() );
	  hists["hist_good_jetEtaVsPhi"]->Fill( jet0.eta(), jet0.phi() );
	  hists["hist_good_jetNTracks"]->Fill( jet0.associatedTracks().size() );
	  hists["hist_good_jetNTracksVsPt"]->Fill( jet0.pt(), jet0.associatedTracks().size() );
	  hists["hist_good_jetEMF"]->Fill( jet0.emEnergyFraction() );	
	  hists["hist_good_jetCorr"]->Fill( jet0.jecFactor("Uncorrected") );
	  hists["hist_good_n90Hits"]->Fill( static_cast<int>(jet0.jetID().n90Hits) );
	  hists["hist_good_fHPD"]->Fill( jet0.jetID().fHPD );
	  hists["hist_good_nConstituents"]->Fill( jet0.nConstituents() );


	  hists["hist_good_jetPt"]->Fill( jet1.pt() );
	  hists["hist_good_jetEtaVsPhi"]->Fill( jet1.eta(), jet1.phi() );
	  hists["hist_good_jetNTracks"]->Fill( jet1.associatedTracks().size() );
	  hists["hist_good_jetNTracksVsPt"]->Fill( jet1.pt(), jet1.associatedTracks().size() );
	  hists["hist_good_jetEMF"]->Fill( jet1.emEnergyFraction() );	
	  hists["hist_good_jetCorr"]->Fill( jet1.jecFactor("Uncorrected") );
	  hists["hist_good_n90Hits"]->Fill( static_cast<int>(jet1.jetID().n90Hits) );
	  hists["hist_good_fHPD"]->Fill( jet1.jetID().fHPD );
	  hists["hist_good_nConstituents"]->Fill( jet1.nConstituents() );

	}// end if passed calo jet id
      }// end if passed dphi cuts
    }// end if passed kin cuts


    ///------------------
    /// PF JETS
    ///------------------
    if ( retPF.test( pfSelector.pfDeltaPhi() ) ) {

      vector<pat::Jet> const & allPFJets = pfSelector.allPFJets();

      for ( std::vector<pat::Jet>::const_iterator jetBegin = allPFJets.begin(),
	      jetEnd = jetBegin + 2, ijet = jetBegin;
	    ijet != jetEnd; ++ijet ) {
	
	const pat::Jet & jet = *ijet;
	
	double pt = jet.pt();
      
	hists["hist_pf_jetPt"]->Fill( pt );
	hists["hist_pf_jetEtaVsPhi"]->Fill( jet.eta(), jet.phi() );
	hists["hist_pf_nConstituents"]->Fill( jet.nConstituents() );
	hists["hist_pf_jetCEF"]->Fill( jet.chargedEmEnergyFraction()  );
	hists["hist_pf_jetNEF"]->Fill( jet.neutralEmEnergyFraction()  );
	hists["hist_pf_jetCHF"]->Fill( jet.chargedHadronEnergyFraction()  );
	hists["hist_pf_jetNHF"]->Fill( jet.neutralHadronEnergyFraction()  );


	if ( useMC && jet.genJet() != 0 ) {
	  hists["hist_pf_jetGenEmE"]->Fill( jet.genJet()->emEnergy() );
	  hists["hist_pf_jetGenHadE"]->Fill( jet.genJet()->hadEnergy() );
	  hists["hist_pf_jetEoverGenE"]->Fill( jet.energy() / jet.genJet()->energy() );

	  hists["hist_pf_jetGenEMF"]->Fill( jet.genJet()->emEnergy() / jet.genJet()->energy() );
	}
 
      } // end loop over jets

    

      if ( retPF.test( pfSelector.pfJetID() ) ) {
	pat::Jet const & jet0 = pfSelector.pfJet0();
	pat::Jet const & jet1 = pfSelector.pfJet1();

	TLorentzVector p4_j0( jet0.px(), jet0.py(), jet0.pz(), jet0.energy() );
	TLorentzVector p4_j1( jet1.px(), jet1.py(), jet1.pz(), jet1.energy() );

	TLorentzVector p4_jj = p4_j0 + p4_j1;

	hists["hist_pf_mjj"]->Fill( p4_jj.M() );
	hists["hist_pf_dR_jj"]->Fill( p4_j0.DeltaR( p4_j1 ) );
	hists["hist_pf_imbalance_jj"]->Fill( (p4_j0.Perp() - p4_j1.Perp() ) /
						      (p4_j0.Perp() + p4_j1.Perp() ) );

	hists["hist_pf_good_jetPt"]->Fill( jet0.pt() );
	hists["hist_pf_good_jetEtaVsPhi"]->Fill( jet0.eta(), jet0.phi() );
	hists["hist_pf_good_nConstituents"]->Fill( jet0.nConstituents() );
	hists["hist_pf_good_jetCEF"]->Fill( jet0.chargedEmEnergyFraction()  );
	hists["hist_pf_good_jetNEF"]->Fill( jet0.neutralEmEnergyFraction()  );
	hists["hist_pf_good_jetCHF"]->Fill( jet0.chargedHadronEnergyFraction()  );
	hists["hist_pf_good_jetNHF"]->Fill( jet0.neutralHadronEnergyFraction()  );


	hists["hist_pf_good_jetPt"]->Fill( jet1.pt() );
	hists["hist_pf_good_jetEtaVsPhi"]->Fill( jet1.eta(), jet1.phi() );
	hists["hist_pf_good_nConstituents"]->Fill( jet1.nConstituents() );
	hists["hist_pf_good_jetCEF"]->Fill( jet1.chargedEmEnergyFraction()  );
	hists["hist_pf_good_jetNEF"]->Fill( jet1.neutralEmEnergyFraction()  );
	hists["hist_pf_good_jetCHF"]->Fill( jet1.chargedHadronEnergyFraction()  );
	hists["hist_pf_good_jetNHF"]->Fill( jet1.neutralHadronEnergyFraction()  );

      } // end if 2 good PF jets
    
    }// end if delta phi pf cuts
    
  } // end loop over events
    
  cout << "Calo jet selection" << endl;
  caloSelector.print(std::cout);
  cout << "PF jet selection" << endl;
  pfSelector.print(std::cout);




  return 0;
}

