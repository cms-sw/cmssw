// system include files
#include <memory>
#include <sys/time.h>
#include <cstdlib>

// user include files
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

//for collections
#include "HLTrigger/JetMET/interface/AlphaT.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "TLorentzVector.h"

#include <unordered_map>


struct hltPlot {
  
  std::pair<MonitorElement*,bool> nME;
  std::pair<MonitorElement*,bool> etaME;
  std::pair<MonitorElement*,bool> phiME;
  std::pair<MonitorElement*,bool> ptME;
  std::pair<MonitorElement*,bool> massME;
  std::pair<MonitorElement*,bool> energyME;
  std::pair<MonitorElement*,bool> csvME;
  std::pair<MonitorElement*,bool> etaVSphiME;
  std::pair<MonitorElement*,bool> ptMEhep17;
  std::pair<MonitorElement*,bool> ptMEhem17; // in harvesting step ratio
  std::pair<MonitorElement*,bool> mrME;
  std::pair<MonitorElement*,bool> rsqME;
  std::pair<MonitorElement*,bool> dxyME;
  std::pair<MonitorElement*,bool> dzME;
  std::pair<MonitorElement*,bool> dimassME;
  std::pair<MonitorElement*,bool> dRME;
  std::pair<MonitorElement*,bool> dRetaVSphiME;
  std::pair<MonitorElement*,bool> q1q2ME;
  
  std::string label;
  std::string pathNAME;
  int         pathIDX;
  std::string moduleNAME;
  
  std::string xTITLE;
  std::vector<double> etaBINNING;
  std::vector<double> ptBINNING;
  std::vector<double> phiBINNING;
  std::vector<double> massBINNING;
  std::vector<double> dxyBINNING;
  std::vector<double> dzBINNING;
  std::vector<double> dimassBINNING;
  
  bool doPlot2D;
  bool doPlotETA;
  bool doPlotMASS;
  bool doPlotENERGY;
  bool doPlotHEP17;
  bool doPlotCSV;
  bool doCALO;
  bool doPF;
  bool doPlotRazor;
  bool doPlotDXY;
  bool doPlotDZ;
  bool doPlotDiMass;
};
//
// class declaration
//

//using namespace edm;
using std::unordered_map;

class HLTObjectsMonitor : public DQMEDAnalyzer {

   public:
      explicit HLTObjectsMonitor(const edm::ParameterSet&);
      ~HLTObjectsMonitor() override = default;

  //      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      void analyze(const edm::Event&, const edm::EventSetup&) override;
      void bookHistograms(DQMStore::IBooker &i, edm::Run const&, edm::EventSetup const&) override;
      void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;

      static hltPlot getPlotPSet(edm::ParameterSet pset);
      void getPSet();
      bool isHEP17(double eta, double phi);
      bool isHEM17(double eta, double phi);

      double dxyFinder(double, double, edm::Handle<reco::RecoChargedCandidateCollection>, edm::Handle<reco::BeamSpot>, double);
      double dzFinder(double, double, double, double, edm::Handle<reco::RecoChargedCandidateCollection>, double);
      // ----------member data ---------------------------

  std::string TopFolder_;
  std::string label_;
  std::string processName_;
  std::vector<edm::ParameterSet> plotPSETS_;

  HLTConfigProvider hltConfig_;

  MonitorElement* eventsPlot_;
  std::vector<hltPlot> hltPlots_;

  bool debug_;

  std::string mainFolder_;
  std::string backupFolder_;


  edm::EDGetTokenT<edm::TriggerResults>   triggerResultsToken_;
  edm::EDGetTokenT<trigger::TriggerEvent> triggerEventToken_;

  edm::InputTag beamSpot_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;

  edm::EDGetTokenT<reco::JetTagCollection> caloJetBTagsToken_;
  edm::EDGetTokenT<reco::JetTagCollection> pfJetBTagsToken_;
  
  edm::InputTag muCandidates_;
  edm::EDGetTokenT<std::vector<reco::RecoChargedCandidate>> muCandidatesToken_;
  edm::InputTag eleCandidates_;
  edm::EDGetTokenT<std::vector<reco::RecoChargedCandidate>> eleCandidatesToken_;

  const double MASS_MU = .105658;


struct MEbinning {
  int nbins;
  double xmin;
  double xmax;
};


double MAX_PHI = 3.2;
int N_PHI = 64;
const MEbinning phi_binning_{
  N_PHI, -MAX_PHI, MAX_PHI
};

double MAX_CSV = 1.;
int N_CSV = 20;
const MEbinning csv_binning_{
  N_CSV, -MAX_CSV, MAX_CSV
};

std::vector<double> phi_variable_binning_;

/*
  HEP17 covers 
  - phi between 310° and 330° (-50° to -30°, or -0.87 t.52 rad)
  - eta between +1.3 and +3.0 (positive side only)
*/
double MAX_PHI_HEP17 = -0.52;
double MIN_PHI_HEP17 = -0.87;
int N_PHI_HEP17 = 7;
const MEbinning phi_binning_hep17_{
  N_PHI_HEP17, MIN_PHI_HEP17, MAX_PHI_HEP17
};
double MAX_ETA_HEP17 = 3.0;
double MIN_ETA_HEP17 = 1.3;
int N_ETA_HEP17 = 6;
const MEbinning eta_binning_hep17_{
  N_ETA_HEP17, MIN_ETA_HEP17, MAX_ETA_HEP17
};

const MEbinning eta_binning_hem17_{
  N_ETA_HEP17, -MAX_ETA_HEP17, MIN_ETA_HEP17
};

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//
hltPlot
HLTObjectsMonitor::getPlotPSet(edm::ParameterSet pset) {

  return hltPlot{
    std::make_pair<MonitorElement*,bool>(nullptr,false), 
      std::make_pair<MonitorElement*,bool>(nullptr,pset.getParameter<bool>("displayInPrimary_eta")      ),
      std::make_pair<MonitorElement*,bool>(nullptr,pset.getParameter<bool>("displayInPrimary_phi")      ),
      std::make_pair<MonitorElement*,bool>(nullptr,pset.getParameter<bool>("displayInPrimary_pt")       ),
      std::make_pair<MonitorElement*,bool>(nullptr,pset.getParameter<bool>("displayInPrimary_mass")     ),
      std::make_pair<MonitorElement*,bool>(nullptr,pset.getParameter<bool>("displayInPrimary_energy")   ),
      std::make_pair<MonitorElement*,bool>(nullptr,pset.getParameter<bool>("displayInPrimary_csv")      ),
      std::make_pair<MonitorElement*,bool>(nullptr,pset.getParameter<bool>("displayInPrimary_etaVSphi") ),
      std::make_pair<MonitorElement*,bool>(nullptr,pset.getParameter<bool>("displayInPrimary_pt_HEP17")  ),      
      std::make_pair<MonitorElement*,bool>(nullptr,pset.getParameter<bool>("displayInPrimary_pt_HEM17")  ),      
      std::make_pair<MonitorElement*,bool>(nullptr,pset.getParameter<bool>("displayInPrimary_MR")        ),      
      std::make_pair<MonitorElement*,bool>(nullptr,pset.getParameter<bool>("displayInPrimary_RSQ")       ),      
      std::make_pair<MonitorElement*,bool>(nullptr,pset.getParameter<bool>("displayInPrimary_dxy")       ),      
      std::make_pair<MonitorElement*,bool>(nullptr,pset.getParameter<bool>("displayInPrimary_dz")        ),      
      std::make_pair<MonitorElement*,bool>(nullptr,pset.getParameter<bool>("displayInPrimary_dimass")    ),      
      std::make_pair<MonitorElement*,bool>(nullptr,false),
      std::make_pair<MonitorElement*,bool>(nullptr,false),
      std::make_pair<MonitorElement*,bool>(nullptr,false),
      pset.getParameter<std::string>("label"     ),
      pset.getParameter<std::string>("pathNAME"  ),
      -1,
      pset.getParameter<std::string>("moduleNAME"),      
      pset.getParameter<std::string>("xTITLE"    ),
      pset.getParameter<std::vector<double> >("etaBINNING"    ),
      pset.getParameter<std::vector<double> >("ptBINNING"     ),      
      pset.getParameter<std::vector<double> >("phiBINNING"    ),      
      pset.getParameter<std::vector<double> >("massBINNING"   ),      
      pset.getParameter<std::vector<double> >("dxyBINNING"    ),      
      pset.getParameter<std::vector<double> >("dzBINNING"     ),      
      pset.getParameter<std::vector<double> >("dimassBINNING" ),      
      pset.getUntrackedParameter<bool>("doPlot2D",     false ),
      pset.getUntrackedParameter<bool>("doPlotETA",    true  ),
      pset.getUntrackedParameter<bool>("doPlotMASS",   false ),
      pset.getUntrackedParameter<bool>("doPlotENERGY", false ),
      pset.getUntrackedParameter<bool>("doPlotHEP17",  true  ),
      pset.getUntrackedParameter<bool>("doPlotCSV",    false ),
      pset.getUntrackedParameter<bool>("doCALO",       false ),
      pset.getUntrackedParameter<bool>("doPF",         false ),
      pset.getUntrackedParameter<bool>("doPlotRazor",  false ),
      pset.getUntrackedParameter<bool>("doPlotDXY",    false ),
      pset.getUntrackedParameter<bool>("doPlotDZ",     false ),
      pset.getUntrackedParameter<bool>("doPlotDiMass", false )
      };

}

void
HLTObjectsMonitor::getPSet() {
  
  for ( const auto & pset : plotPSETS_ )
    hltPlots_.push_back( getPlotPSet(pset) );
}

bool HLTObjectsMonitor::isHEP17(double eta, double phi) {
  if ( (eta >= MIN_ETA_HEP17 && eta <= MAX_ETA_HEP17) &&
       (phi >= MIN_PHI_HEP17 && phi <= MAX_PHI_HEP17) ) return true;
  else
    return false;
}
bool HLTObjectsMonitor::isHEM17(double eta, double phi) {
  if ( (eta >= -MAX_ETA_HEP17 && eta <= -MIN_ETA_HEP17) &&
       (phi >= MIN_PHI_HEP17 && phi <= MAX_PHI_HEP17) ) return true;
  else
    return false;
}
//
// constructors and destructor
//
HLTObjectsMonitor::HLTObjectsMonitor(const edm::ParameterSet& iConfig)
  : TopFolder_   ( iConfig.getParameter<std::string>("TopFolder")                 )
  , label_       ( iConfig.getParameter<std::string>("label")                     )
  , processName_ ( iConfig.getParameter<std::string>("processName")               )
  , plotPSETS_   ( iConfig.getParameter<std::vector<edm::ParameterSet> >("plots") )
  , debug_       ( iConfig.getUntrackedParameter<bool>("debug",false)             )
  , triggerResultsToken_ ( consumes<edm::TriggerResults>  (iConfig.getParameter<edm::InputTag>("TriggerResults") )                  )
  , triggerEventToken_   ( consumes<trigger::TriggerEvent>(iConfig.getParameter<edm::InputTag>("TriggerSummary") )                  )
  , beamSpot_      ( iConfig.getParameter<edm::InputTag>("beamspot") )
  , beamSpotToken_ ( consumes<reco::BeamSpot>(beamSpot_)             )
  , caloJetBTagsToken_   ( consumes<reco::JetTagCollection>(iConfig.getParameter<edm::InputTag>("caloJetBTags") )                   )
  , pfJetBTagsToken_     ( consumes<reco::JetTagCollection>(iConfig.getParameter<edm::InputTag>("pfJetBTags") )                     )
  , muCandidates_        ( iConfig.getParameter<edm::InputTag>("muCandidates")                )
  , muCandidatesToken_   ( consumes<std::vector<reco::RecoChargedCandidate> >(muCandidates_)  )
  , eleCandidates_       ( iConfig.getParameter<edm::InputTag>("eleCandidates")               )
  , eleCandidatesToken_  ( consumes<std::vector<reco::RecoChargedCandidate> >(eleCandidates_) )
{
  getPSet();

   //now do what ever initialization is needed
  mainFolder_   = TopFolder_+"/MainShifter";
  backupFolder_ = TopFolder_+"/Backup";
  
  //set Token(s)


  double step = 2*MAX_PHI/double(N_PHI);
  for ( int i=0; i<=N_PHI; i++)
    phi_variable_binning_.push_back(-MAX_PHI + step*i);

}


//
// member functions
//

// ------------ method called for each event  ------------
void
HLTObjectsMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //  if ( debug_ ) 
  //    std::cout << "[HLTObjectsMonitor::analyze]" << std::endl;

  // access trigger results
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(triggerResultsToken_, triggerResults);
  if (!triggerResults.isValid()) return;

  edm::Handle<trigger::TriggerEvent> triggerEvent;
  iEvent.getByToken(triggerEventToken_, triggerEvent);
  if (!triggerEvent.isValid()) return;

  edm::Handle<reco::JetTagCollection> caloJetBTags;
  iEvent.getByToken(caloJetBTagsToken_, caloJetBTags);
	
  edm::Handle<reco::JetTagCollection> pfJetBTags;
  iEvent.getByToken(pfJetBTagsToken_, pfJetBTags);
	
  edm::Handle<std::vector<reco::RecoChargedCandidate>> muCandidates;
  iEvent.getByToken(muCandidatesToken_, muCandidates);

  edm::Handle<std::vector<reco::RecoChargedCandidate>> eleCandidates;
  iEvent.getByToken(eleCandidatesToken_, eleCandidates);

  edm::Handle<reco::BeamSpot> beamspot;
  iEvent.getByToken(beamSpotToken_, beamspot);

  // loop over path
  int ibin = -1;
  for (auto & plot : hltPlots_) {
    ibin++;
    if ( plot.pathIDX <= 0 ) continue;

    if ( triggerResults->accept(plot.pathIDX) ) {
      if ( debug_ )
	std::cout << plot.pathNAME << " --> bin: " << ibin << std::endl;
      eventsPlot_->Fill(ibin);

      const trigger::TriggerObjectCollection objects = triggerEvent->getObjects();
      if ( hltConfig_.saveTags(plot.moduleNAME) ) {
	if ( debug_ )
	  std::cout << "objects: " << objects.size() << std::endl;
	
	bool moduleFOUND = false;
	std::vector<std::string> moduleNames = hltConfig_.moduleLabels(plot.pathIDX);
	for ( const auto & module : moduleNames ) {
	  if ( module == plot.moduleNAME) moduleFOUND = true;
	}
	if ( debug_ ) 
	  std::cout << plot.moduleNAME << (moduleFOUND ? "" : "NOT" ) << " found in the list of modules" << std::endl;

	if (debug_)
	  for ( const auto & module : moduleNames ) {
	    unsigned int idx = triggerEvent->filterIndex(edm::InputTag(module,"",processName_));	      
	    std::cout << "module: " << module;
	    if ( idx < triggerEvent->sizeFilters() )
	      std::cout << " --> " << idx;
	    std::cout << std::endl;
	  }
	//
	// trigger accepted and collection w/ objects is available
	edm::InputTag moduleName = edm::InputTag(plot.moduleNAME,"",processName_);
	unsigned int moduleIDX = triggerEvent->filterIndex(moduleName);
	if (debug_)
	  std::cout << "moduleNAME: " << plot.moduleNAME << " --> " << moduleIDX << std::endl;
			
	if ( moduleIDX >= triggerEvent->sizeFilters() ) {
	  LogDebug ("HLTObjectsMonitor") << plot.pathNAME << " " << plot.moduleNAME << " is not available ! please, fix update DQM/HLTEvF/python/HLTObjectsMonitor_cfi.py";
	  return;
	}

	const trigger::Keys &keys = triggerEvent->filterKeys( moduleIDX );
	if ( debug_ )
	  std::cout << "keys: " << keys.size() << std::endl;

	plot.nME.first->Fill(keys.size());

	double MR  = 0.;
	double RSQ = 0.;
	for ( const auto & key : keys ) {
	  
	  double pt     = objects[key].pt();
	  double eta    = objects[key].eta();
	  double phi    = objects[key].phi();
	  double mass   = objects[key].mass(); 
	  double energy = objects[key].energy(); 
	  int    id     = objects[key].id();
	  if ( debug_ )
	    std::cout << "object ID " << id << " mass: " << mass << std::endl;
	  
	  // single-object plots
	  plot.ptME.first->Fill(pt);
	  if ( plot.doPlotETA ) plot.etaME.first->Fill(eta);
	  plot.phiME.first->Fill(phi);
	  
	  if ( plot.doPlotCSV ) {
	    
	    if ( plot.doCALO ) {
	      if ( !caloJetBTags.isValid() ) plot.csvME.first->Fill(-10.);
	      else {
		for ( auto it = caloJetBTags->begin();
		      it != caloJetBTags->end(); ++it ) {
		  double dR = deltaR(eta,phi,it->first->eta(),it->first->phi());
		  if ( debug_ )
		    std::cout << "[HLTObjectsMonitor::analyze] deltaR: " << dR << " matched ? " << ( dR <= 0.4 ? "YEAP" : "NOPE" ) << std::endl;
		  plot.csvME.first->Fill(it->second);
		}
	      }
	      
	    } else if ( plot.doPF ) {
	      if ( !pfJetBTags.isValid() ) plot.csvME.first->Fill(-10.);
	      else {
		for ( auto it = pfJetBTags->begin();
		      it != pfJetBTags->end(); ++it ) {
		  double dR = deltaR(eta,phi,it->first->eta(),it->first->phi());
		  if ( debug_ )
		    std::cout << "[HLTObjectsMonitor::analyze] deltaR: " << dR << " matched ? " << ( dR <= 0.4 ? "YEAP" : "NOPE" ) << std::endl;
		  plot.csvME.first->Fill(it->second);
		}
	      }
	    }
	  }
	  if ( plot.doPlotMASS   ) plot.massME.first->Fill(mass);
	  if ( plot.doPlotENERGY ) plot.energyME.first->Fill(energy);
	  if ( plot.doPlot2D ) plot.etaVSphiME.first->Fill(eta,phi);
	  if ( plot.doPlotHEP17 ) {
	    if ( isHEP17(eta,phi) ) plot.ptMEhep17.first->Fill(pt);
	    if ( isHEM17(eta,phi) ) plot.ptMEhem17.first->Fill(pt);
	  }
	  
	  if(id == 0){ //the MET object containing MR and Rsq will show up with ID = 0
	    MR  = objects[key].px(); //razor variables stored in dummy reco::MET objects
	    RSQ = objects[key].py();
	  }
	  
	  if ( plot.doPlotDXY ) {
	    double dxy = -99.;
	    if ( abs(id) == 13 )
	      dxy = dxyFinder(eta,phi,muCandidates, beamspot, 0.1); // dRcut = 0.1
	    else 
	      dxy = dxyFinder(eta,phi,eleCandidates, beamspot, 0.1); // dRcut = 0.1
	    plot.dxyME.first->Fill(dxy);
	  }
	} // end loop on keys
	if ( plot.doPlotRazor ) {
	  plot.mrME.first->Fill(MR);
	  plot.rsqME.first->Fill(RSQ);
	}
	
	if ( keys.size() < 2 ) {
	  if ( plot.doPlotDiMass || plot.doPlotDZ )
	    LogDebug ("HLTObjectsMonitor") << plot.pathNAME << " " << plot.moduleNAME << " # objects is (" << keys.size() << ") less than 2 ! you probably want to either change the moduleNAME or switch off di-object system plots (doPlotDZ: " << plot.doPlotDZ << " doPlotDiMass: " << plot.doPlotDiMass << ") in DQM/HLTEvF/python/HLTObjectsMonitor_cfi.py)";
	} else {
	  for ( const auto & key : keys ) {
	    double pt   = objects[key].pt();
	    double eta  = objects[key].eta();
	    double phi  = objects[key].phi();
	    int    id   = objects[key].id();
	    
	    unsigned int kCnt0 = 0;
	    
	    TLorentzVector v1;
	    if ( abs(id) == 13 ) // check if it is a muon
	      v1.SetPtEtaPhiM(pt,eta,phi,MASS_MU);
	    else
	      v1.SetPtEtaPhiM(pt,eta,phi,0);
	    
	    unsigned int kCnt1 = 0;
	    for ( const auto & key1: keys ) {
	      
	      if (key != key1 && kCnt1 > kCnt0) { // avoid filling hists with same objs && avoid double counting separate objs
		
		double pt2  = objects[key1].phi();		  
		double eta2 = objects[key1].eta();
		double phi2 = objects[key1].phi();
		int id2 = objects[key1].id();

		double dR = deltaR( eta, phi, eta2, phi2);
		plot.dRME.first->Fill(dR);
		plot.dRetaVSphiME.first->Fill(eta,phi,dR);

		int q1 = ( id==0 ? 0 : id/abs(id) );
		int q2 = ( id2==0 ? 0 : id2/abs(id2) );
		int q1q2 = q1*q2;
		plot.q1q2ME.first->Fill(q1q2);

		if ( abs(id) != abs(id2) )
		  edm::LogInfo ("HLTObjectsMonitor") << plot.pathNAME << " " << plot.moduleNAME << " objects have different ID !?!" << abs(id) << " and " << abs(id2);

		if( (id+id2 ) == 0 ) {   // check di-object system charge and flavor
		  
		  TLorentzVector v2;
		  if ( abs( id2 ) == 13 ) // check if it is a muon
		    v2.SetPtEtaPhiM(pt2,eta2,phi2, MASS_MU);
		  else
		    v2.SetPtEtaPhiM(pt2,eta2,phi2, 0);
		  
		  if ( plot.doPlotDiMass ) {		
		    TLorentzVector v = v1+v2;
		    plot.dimassME.first->Fill(v.M());
		  }
		  
		  if ( plot.doPlotDZ ) {
		    double dz = -99.;
		    if ( abs(id) == 13 )
		      dz = dzFinder(eta,phi,eta2,phi2,muCandidates, 0.1); // dRcut = 0.1
		    else
		      dz = dzFinder(eta,phi,eta2,phi2,eleCandidates, 0.1); // dRcut = 0.1
		    plot.dzME.first->Fill(dz);
		  }
		}
		
	      }
	      kCnt1++;
	    }
	    kCnt0++;
	  }
	  
	}
      }
    }
  }
  
}

// ------------ method called when starting to processes a run  ------------
void
HLTObjectsMonitor::dqmBeginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{
  bool changed = true;
  if (hltConfig_.init(iRun, iSetup, processName_, changed))
    if (debug_) std::cout << "[HLTObjectsMonitor::dqmBeginRun] extracting HLTconfig" << std::endl;

  //get path indicies from menu 
  std::string pathName_noVersion;
  std::vector<std::string> triggerPaths = hltConfig_.triggerNames();

  if ( debug_ )
      std::cout << "[HLTObjectsMonitor::dqmBeginRun] triggerPaths: " << triggerPaths.size() << " <--> " << hltPlots_.size() << std::endl;

  for (const auto & pathName : triggerPaths) {

    if ( debug_ )
      std::cout << "[HLTObjectsMonitor::dqmBeginRun] " << pathName << std::endl;

    pathName_noVersion = hltConfig_.removeVersion(pathName);
    //    std::cout << "pathName: " << pathName << " --> " << pathName_noVersion << std::endl;
    for (auto & plot : hltPlots_) {
      if (plot.pathNAME == pathName_noVersion) {
	plot.pathIDX = hltConfig_.triggerIndex(pathName);
	// check that the index makes sense, otherwise force pathIDX = -1
	if ( plot.pathIDX <= 0 || plot.pathIDX == int(triggerPaths.size()) )
	  plot.pathIDX = -1;
      }
    }
  }

  if ( debug_ ) {
    for (const auto & plot : hltPlots_)
      std::cout << "plot: " << plot.pathNAME << " --> pathIDX: " << plot.pathIDX << std::endl;
    std::cout << "[HLTObjectsMonitor::dqmBeginRun] DONE" << std::endl;
  }

}

void HLTObjectsMonitor::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup)
{
  
  if ( debug_ ) 
    std::cout << "[HLTObjectsMonitor::bookHistograms]" << std::endl;

  ibooker.setCurrentFolder(TopFolder_);

  std::string name  = "eventsPerPath_"+label_;
  std::string title = " events per path";
  int nbins = hltPlots_.size();
  eventsPlot_ = ibooker.book1D(name,title,nbins,-0.5,double(nbins)-0.5);
  eventsPlot_->setAxisTitle("HLT path");
  for ( int i=0; i<nbins; i++ ) {
    eventsPlot_->setBinLabel(i+1,hltPlots_[i].pathNAME);
    if ( debug_ )
      std::cout << hltPlots_[i].pathNAME << " --> bin: " << i+1 << std::endl;
  }

  for (auto & plot : hltPlots_) {
    
    if ( debug_ )
      std::cout << "booking plots for " << plot.label << std::endl;

    if ( plot.pathIDX <= 0 ) {
      LogDebug ("HLTObjectsMonitor") << plot.pathNAME << " is not available in the HLT menu ! no plots are going to be booked for it (update DQM/HLTEvF/python/HLTObjectsMonitor_cfi.py)";
      continue;    
    }
    if ( debug_ )
      std::cout << "booking histograms for " << plot.pathNAME << std::endl;

    {
      if ( plot.nME.second )
	ibooker.setCurrentFolder(mainFolder_);
      else
	ibooker.setCurrentFolder(backupFolder_);
      
      name  = plot.label+"_nobjects";
      title = plot.pathNAME+" # objects";
      plot.nME.first = ibooker.book1D(name,title,20,-0.5,19.5);
      plot.nME.first->setAxisTitle(plot.xTITLE+" # objects");
    }

    if ( plot.ptME.second )
	ibooker.setCurrentFolder(mainFolder_);
    else
      ibooker.setCurrentFolder(backupFolder_);
      
    name  = plot.label+"_pt";
    title = plot.pathNAME+" p_T";
    int nbins = (plot.ptBINNING).size()-1;
    std::vector<float> fbinning((plot.ptBINNING).begin(),(plot.ptBINNING).end());
    float* arr = &fbinning[0];
    plot.ptME.first = ibooker.book1D(name,title,nbins,arr);
    plot.ptME.first->setAxisTitle(plot.xTITLE+" p_{T} [GeV]");


    {
    if ( plot.phiME.second )
	ibooker.setCurrentFolder(mainFolder_);
    else
      ibooker.setCurrentFolder(backupFolder_);
      
    name  = plot.label+"_phi";
    title = plot.pathNAME+" #phi";
    int nbins = (plot.phiBINNING).size()-1;
    std::vector<float> fbinning((plot.phiBINNING).begin(),(plot.phiBINNING).end());
    float* arr = &fbinning[0];
    plot.phiME.first = ibooker.book1D(name,title,nbins,arr);
    plot.phiME.first->setAxisTitle(plot.xTITLE+" #phi [rad]");
    }

    if ( plot.doPlotETA ) {
      if ( plot.etaME.second )
	ibooker.setCurrentFolder(mainFolder_);
      else
	ibooker.setCurrentFolder(backupFolder_);
      
      name  = plot.label+"_eta";
      title = plot.pathNAME+" #eta";
      int nbins = (plot.etaBINNING).size()-1;
      std::vector<float> fbinning((plot.etaBINNING).begin(),(plot.etaBINNING).end());
      float* arr = &fbinning[0];

      plot.etaME.first = ibooker.book1D(name,title,nbins,arr);
      plot.etaME.first->setAxisTitle(plot.xTITLE+" #eta");
    }

    if ( plot.doPlotMASS ) {
      if ( plot.massME.second )
	ibooker.setCurrentFolder(mainFolder_);
      else
	ibooker.setCurrentFolder(backupFolder_);
      
      name  = plot.label+"_mass";
      title = plot.pathNAME+" mass";
      int nbins = (plot.massBINNING).size()-1;
      std::vector<float> fbinning((plot.massBINNING).begin(),(plot.massBINNING).end());
      float* arr = &fbinning[0];

      plot.massME.first = ibooker.book1D(name,title,nbins,arr);
      plot.massME.first->setAxisTitle(plot.xTITLE+" mass [GeV]");
    }

    if ( plot.doPlotENERGY ) {
      if ( plot.energyME.second )
	ibooker.setCurrentFolder(mainFolder_);
      else
	ibooker.setCurrentFolder(backupFolder_);
      
      name  = plot.label+"_energy";
      title = plot.pathNAME+" energy";
      int nbins = (plot.ptBINNING).size()-1;
      std::vector<float> fbinning((plot.ptBINNING).begin(),(plot.ptBINNING).end());
      float* arr = &fbinning[0];

      plot.energyME.first = ibooker.book1D(name,title,nbins,arr);
      plot.energyME.first->setAxisTitle(plot.xTITLE+" energy [GeV]");
    }

    if ( plot.doPlotCSV ) {
      if ( plot.csvME.second )
	ibooker.setCurrentFolder(mainFolder_);
      else
	ibooker.setCurrentFolder(backupFolder_);
      
      name  = plot.label+"_csv";
      title = plot.pathNAME+" CSV";

      plot.csvME.first = ibooker.book1D(name,title,csv_binning_.nbins,csv_binning_.xmin,csv_binning_.xmax);
      plot.csvME.first->setAxisTitle(plot.xTITLE+" CSV discriminator");
    }

    if ( plot.doPlot2D ) {
      if ( plot.etaVSphiME.second )
	ibooker.setCurrentFolder(mainFolder_);
      else
	ibooker.setCurrentFolder(backupFolder_);
      
      name  = plot.label+"_etaVSphi";
      title = plot.pathNAME+" #eta vs #phi";
      int nbinsX = (plot.etaBINNING).size()-1;
      std::vector<float> fbinningX((plot.etaBINNING).begin(),(plot.etaBINNING).end());
      float* arrX = &fbinningX[0];
      int nbinsY = (plot.phiBINNING).size()-1;;
      std::vector<float> fbinningY((plot.phiBINNING).begin(),(plot.phiBINNING).end());
      float* arrY = &fbinningY[0];
      plot.etaVSphiME.first = ibooker.book2D(name,title,nbinsX,arrX,nbinsY,arrY);
      plot.etaVSphiME.first->setAxisTitle(plot.xTITLE+" #eta",1);
      plot.etaVSphiME.first->setAxisTitle(plot.xTITLE+" #phi",2);
    }

    if ( plot.doPlotHEP17 ) {

      if ( plot.ptMEhep17.second )
	ibooker.setCurrentFolder(mainFolder_);
      else
	ibooker.setCurrentFolder(backupFolder_);
      
      int nbins = (plot.ptBINNING).size()-1;
      std::vector<float> fbinning((plot.ptBINNING).begin(),(plot.ptBINNING).end());
      float* arr = &fbinning[0];

      name  = plot.label+"_pt_HEP17";
      title = plot.pathNAME+" p_{T} HEP17";
      plot.ptMEhep17.first = ibooker.book1D(name,title,nbins,arr);
      plot.ptMEhep17.first->setAxisTitle(plot.xTITLE+" p_{T} [GeV]",1);

      if ( plot.ptMEhem17.second )
	ibooker.setCurrentFolder(mainFolder_);
      else
	ibooker.setCurrentFolder(backupFolder_);
      
      name  = plot.label+"_pt_HEM17";
      title = plot.pathNAME+" p_{T} HEM17";
      plot.ptMEhem17.first = ibooker.book1D(name,title,nbins,arr);
      plot.ptMEhem17.first->setAxisTitle(plot.xTITLE+" p_{T} [GeV]",1);
    }

    if ( plot.doPlotRazor ) {
      if ( plot.mrME.second )
	ibooker.setCurrentFolder(mainFolder_);
      else
	ibooker.setCurrentFolder(backupFolder_);
      
      name  = plot.label+"_mr";
      title = plot.pathNAME+" M_{R}";
      plot.mrME.first = ibooker.book1D(name,title,100,0.,100.);
      plot.mrME.first->setAxisTitle(plot.xTITLE+" M_{R} [GeV]",1);

      if ( plot.rsqME.second )
	ibooker.setCurrentFolder(mainFolder_);
      else
	ibooker.setCurrentFolder(backupFolder_);
      
      name  = plot.label+"_rsq";
      title = plot.pathNAME+" RSQ";
      plot.rsqME.first = ibooker.book1D(name,title,100,0.,100.);
      plot.rsqME.first->setAxisTitle(plot.xTITLE+" RSQ [GeV]",1);

    }

    if ( plot.doPlotDXY ) {

      if ( plot.dxyME.second )
	ibooker.setCurrentFolder(mainFolder_);
      else
	ibooker.setCurrentFolder(backupFolder_);
      
      name  = plot.label+"_dxy";
      title = plot.pathNAME+" d_{xy}";
      int nbins = (plot.dxyBINNING).size()-1;
      std::vector<float> fbinning((plot.dxyBINNING).begin(),(plot.dxyBINNING).end());
      float* arr = &fbinning[0];
      plot.dxyME.first = ibooker.book1D(name,title,nbins,arr);
      plot.dxyME.first->setAxisTitle(plot.xTITLE+" d_{xy} [cm]");
      
    }

    if ( plot.doPlotDZ ) {

      if ( plot.dzME.second )
	ibooker.setCurrentFolder(mainFolder_);
      else
	ibooker.setCurrentFolder(backupFolder_);
      
      name  = plot.label+"_dz";
      title = plot.pathNAME+" d_{z}";
      int nbins = (plot.dzBINNING).size()-1;
      std::vector<float> fbinning((plot.dzBINNING).begin(),(plot.dzBINNING).end());
      float* arr = &fbinning[0];
      plot.dzME.first = ibooker.book1D(name,title,nbins,arr);
      plot.dzME.first->setAxisTitle(plot.xTITLE+" d_{z} [cm]");
      
    }

    if ( plot.dRME.second )
      ibooker.setCurrentFolder(mainFolder_);
    else
      ibooker.setCurrentFolder(backupFolder_);
    
    name  = plot.label+"_dR";
    title = plot.pathNAME+" di-object dR";
    plot.dRME.first = ibooker.book1D(name,title,50,0.,5.);
    plot.dRME.first->setAxisTitle(plot.xTITLE+" dR_{obj,obj}");
    
    if ( plot.dRetaVSphiME.second )
      ibooker.setCurrentFolder(mainFolder_);
    else
      ibooker.setCurrentFolder(backupFolder_);
    
    name  = plot.label+"_dR_etaVSphi";
    title = plot.pathNAME+" di-object dR in the #eta-#phi plane (of 1st obj)";
    plot.dRetaVSphiME.first = ibooker.bookProfile2D(name,title,60,-3.,3.,64,-3.2,3.2,0.,5.);
    plot.dRetaVSphiME.first->setAxisTitle(plot.xTITLE+" #eta",1);
    plot.dRetaVSphiME.first->setAxisTitle(plot.xTITLE+" #phi",2);
    plot.dRetaVSphiME.first->setAxisTitle(plot.xTITLE+" dR_{obj,obj}",3);

    if ( plot.q1q2ME.second )
      ibooker.setCurrentFolder(mainFolder_);
    else
      ibooker.setCurrentFolder(backupFolder_);
    
    name  = plot.label+"_q1q2";
    title = plot.pathNAME+" di-object q1xq2";
    plot.q1q2ME.first = ibooker.book1D(name,title,3,-1.,1.);
    plot.q1q2ME.first->setAxisTitle(plot.xTITLE+" q_{obj1} x q_{obj2}");
    
    if ( plot.doPlotDiMass ) {

      if ( plot.dimassME.second )
	ibooker.setCurrentFolder(mainFolder_);
      else
	ibooker.setCurrentFolder(backupFolder_);
      
      name  = plot.label+"_dimass";
      title = plot.pathNAME+" di-object mass";
      int nbins = (plot.dimassBINNING).size()-1;
      std::vector<float> fbinning((plot.dimassBINNING).begin(),(plot.dimassBINNING).end());
      float* arr = &fbinning[0];
      plot.dimassME.first = ibooker.book1D(name,title,nbins,arr);
      plot.dimassME.first->setAxisTitle(plot.xTITLE+" m_{obj,obj} [GeV]");
      
    }
  }

  if ( debug_ ) 
    std::cout << "[HLTObjectsMonitor::bookHistograms] DONE" << std::endl;

}

double 
HLTObjectsMonitor::dxyFinder(double eta, double phi, edm::Handle<reco::RecoChargedCandidateCollection> candidates, edm::Handle<reco::BeamSpot> beamspot, double dRcut = 0.1)
{
  double dxy = -99.;
  if ( !candidates.isValid() ) {
    LogDebug ("HLTObjectsMonitor") << "either " << muCandidates_ << " or " << eleCandidates_ << " is not valid ! please, update DQM/HLTEvF/python/HLTObjectsMonitor_cfi.py" 
					 << " by switching OFF doPlotDXY or updating the InputTag collection";
    return dxy;
  }
  if ( !beamspot.isValid() ) {
    LogDebug ("HLTObjectsMonitor") << beamSpot_ << " is not valid ! please, update DQM/HLTEvF/python/HLTObjectsMonitor_cfi.py"
					 << " by switching OFF doPlotDXY or updating the InputTag collection";
    return dxy;
  }

  bool matched = false;
  for (reco::RecoChargedCandidateCollection::const_iterator candidate = candidates->begin();
       candidate != candidates->end(); ++candidate) {
    
    if ( deltaR( eta,phi,candidate->eta(),candidate->phi() ) < dRcut ) {
      matched = true;
      dxy = (-(candidate->vx()-beamspot->x0()) * candidate->py() + (candidate->vy()-beamspot->y0()) * candidate->px())/candidate->pt();
      break;
    }
  }
  if (!matched)
    edm::LogWarning ("HLTObjectsMonitor") << "trigger object does not match ( dR > " << dRcut << ") to any of the candidates in either " 
					 << muCandidates_ << " or " << eleCandidates_;

  return dxy;
}

double 
HLTObjectsMonitor::dzFinder(double eta1, double phi1, double eta2, double phi2, edm::Handle<reco::RecoChargedCandidateCollection> candidates, double dRcut = 0.1)
{
  double dz = -99.;
  if ( !candidates.isValid() ) {
    LogDebug ("HLTObjectsMonitor") << "either " << muCandidates_ << " or " << eleCandidates_ << " is not valid ! please, update DQM/HLTEvF/python/HLTObjectsMonitor_cfi.py" 
					 << " by switching OFF doPlotDZ or updating the InputTag collection";
    return dz;
  }

  const reco::RecoChargedCandidate* cand1;
  const reco::RecoChargedCandidate* cand2;
  bool matched1 = false;
  bool matched2 = false;
  for (reco::RecoChargedCandidateCollection::const_iterator candidate = candidates->begin();
       candidate != candidates->end(); ++candidate) {
    
    if ( deltaR( eta1,phi1,candidate->eta(),candidate->phi() ) < dRcut ) {
      matched1 = true;
      cand1 = &*candidate;
      if ( debug_ )
	std::cout << "cand1: " << cand1->pt() << " " << cand1->eta() << " " << cand1->phi() << std::endl;
      break;
    }
  }
  if (!matched1) {
    LogDebug  ("HLTObjectsMonitor") << "trigger object1 does not match ( dR > " << dRcut << ") to any of the candidates in either " 
					 << muCandidates_ << " or " << eleCandidates_;
    return dz;
  }

  for (reco::RecoChargedCandidateCollection::const_iterator candidate = candidates->begin();
       candidate != candidates->end(); ++candidate) {
    if ( debug_ ) {
      std::cout << "candidate: " << candidate->pt() << " cand1: " << cand1->pt() << std::endl;
      std::cout << "candidate: " << candidate->eta() << " cand1: " << cand1->eta() << std::endl;
      std::cout << "candidate: " << candidate->phi() << " cand1: " << cand1->phi() << std::endl;
    }
    if (&*candidate == cand1) continue;

    if ( deltaR( eta2,phi2,candidate->eta(),candidate->phi() ) < dRcut ) {
      matched2 = true;
      cand2 = &*candidate;
      if ( debug_ )
	std::cout << "cand2: " << cand2->pt() << " " << cand2->eta() << " " << cand2->phi() << std::endl;
      break;
    }
  }
  if (!matched2) {
    LogDebug  ("HLTObjectsMonitor") << "trigger object2 does not match ( dR > " << dRcut << ") to any of the candidates in either " 
					 << muCandidates_ << " or " << eleCandidates_;
    return dz;
  }

  dz = cand1->vz() - cand2->vz();
  return dz;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HLTObjectsMonitor);
