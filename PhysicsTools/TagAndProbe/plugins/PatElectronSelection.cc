#include "PhysicsTools/TagAndProbe/interface/PatElectronSelection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <string>
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "PhysicsTools/Utilities/interface/deltaR.h"


PatElectronSelection::PatElectronSelection(const edm::ParameterSet &params)
{

  histogramFile_    = params.getParameter<std::string>("histogramFile");
  _inputProducer = 
    params.getUntrackedParameter<std::string>("src", "selectedLayer1Electrons");
  _BarrelMaxEta  = params.getUntrackedParameter<double>("BarrelMaxEta", 1.4442);
  _EndcapMinEta  = params.getUntrackedParameter<double>("EndcapMinEta", 1.560);
  _EndcapMaxEta  = params.getUntrackedParameter<double>("EndcapMaxEta", 2.5);
  _etMin         = params.getUntrackedParameter<double>("etMin", 20.0);
  _etMax         = params.getUntrackedParameter<double>("etMax", 1000.0);
  _requireID      = params.getUntrackedParameter<bool>("requireID", false);
  _requireTkIso   = params.getUntrackedParameter<bool>("requireTkIso", false);
  _requireEcalIso = params.getUntrackedParameter<bool>("requireEcalIso", false);
  _requireHcalIso = params.getUntrackedParameter<bool>("requireHcalIso", false);
  _requireTrigMatch = params.getUntrackedParameter<bool>("requireTrigMatch", false);
  _verbose     = params.getUntrackedParameter<bool>("verbose", false);

  deltaEtaCutBarrel_ 
    = params.getUntrackedParameter<double>("deltaEtaCutBarrel",      0.5);
  deltaEtaCutEndcaps_     
    = params.getUntrackedParameter<double>("deltaEtaCutEndcaps",     0.5);
  deltaPhiCutBarrel_      
    = params.getUntrackedParameter<double>("deltaPhiCutBarrel",      0.5);
  deltaPhiCutEndcaps_     
    = params.getUntrackedParameter<double>("deltaPhiCutEndcaps",     0.5);
  sigmaEtaEtaCutBarrel_   
    = params.getUntrackedParameter<double>("sigmaEtaEtaCutBarrel",   0.5);
  sigmaEtaEtaCutEndcaps_  
    = params.getUntrackedParameter<double>("sigmaEtaEtaCutEndcaps",  0.5);
  tkIsoCutBarrel_         
    = params.getUntrackedParameter<double>( "tkIsoCutBarrel",        10000.0);
  tkIsoCutEndcaps_        
    = params.getUntrackedParameter<double>( "tkIsoCutEndcaps",       10000.0);
  ecalIsoCutBarrel_       
    = params.getUntrackedParameter<double>( "ecalIsoCutBarrel",      10000.0);
  ecalIsoCutEndcaps_      
    = params.getUntrackedParameter<double>( "ecalIsoCutEndcaps",     10000.0);
  hcalIsoCutBarrel_       
    = params.getUntrackedParameter<double>( "hcalIsoCutBarrel",      10000.0);
  hcalIsoCutEndcaps_      
    = params.getUntrackedParameter<double>( "hcalIsoCutEndcaps",     10000.0);

  const edm::InputTag dSummaryObj( "hltTriggerSummaryAOD","","HLT" );
  triggerSummaryLabel_ = 
    params.getUntrackedParameter<edm::InputTag>("triggerSummaryLabel",  dSummaryObj );

   const edm::InputTag 
     dHLT("hltL1NonIsoHLTNonIsoSingleElectronLWEt15TrackIsolFilter","","HLT");
    hltFilter_ = params.getUntrackedParameter<edm::InputTag>("hltFilter",dHLT);

  produces<std::vector<pat::Electron> >();
}




PatElectronSelection::~PatElectronSelection() { }


//
// member functions
//


// ------------ method called to produce the data  ------------

void PatElectronSelection::produce(edm::Event &event, const edm::EventSetup &eventSetup)
{

  using namespace edm;
  using namespace trigger;
  using namespace std;

   // --------- Create the output collection ---------
   std::auto_ptr<std::vector<pat::Electron> > 
     outCol(new std::vector<pat::Electron> );


   // --------- Read the electron collection in the event
   edm::Handle<edm::View<pat::Electron> > electrons;
   try {
      event.getByLabel(_inputProducer, electrons);
   }
   catch(cms::Exception &ex) {
      edm::LogError("pat::Electron ") << "Error! Can't get collection " << 
	_inputProducer;
      throw ex;
   }



   // ------------ trigger objects 
  edm::Handle<TriggerEvent> triggerObj;
  event.getByLabel(triggerSummaryLabel_,triggerObj); 
  if(!triggerObj.isValid()) { 
    edm::LogInfo("TriggerEvent") << " objects not found"; 
  }
  


  // --------------------------------------------


   int index =0, eClass = -1;
   double elecEta=10.0, elecPhi=10.0, elecE=-1.0, elecEt=-1.0, 
     deltaEta=10.0, deltaPhi=10.0, sigmaEtaEta=10.0;
   double tkIsolation = 100.0, ecalIsolation = 100.0, hcalIsolation = 100.0;
   float electronId = 0.0;
   bool cutresult = false, trigresult = false, idresult = false;


   for(edm::View<pat::Electron>::const_iterator  
	 elec = electrons->begin(); elec != electrons->end();++elec) {
     

     ////// -------- ensure that the electron has correct charge ------ 
     int q = (elec->charge() > 0 ? 1 : -1);
     if( abs(_charge)==1 && !(q==_charge) ) continue;

     
     /////// ------- Get basic kinematic variables -----------         
     reco::SuperClusterRef sc = elec->superCluster();
     elecEta = sc.get()->eta();
     elecPhi = sc.get()->phi ();
     elecE   = sc.get()->energy();
     elecEt  = elecE / cosh(elecEta);
     
     
     ///// --- make sure that the electron is within eta, et acceptance    
     if( !CheckAcceptance( elecEta, elecEt) )  continue;
     
     
     // ---------- dPhi, dEta, sigmaEtaEta -------------
     deltaEta  = elec->deltaEtaSuperClusterTrackAtVtx();
     deltaPhi  = elec->deltaPhiSuperClusterTrackAtVtx();
     sigmaEtaEta  = elec->scSigmaIEtaIEta();


     // ------------ isolation, id variables -----------------  
     tkIsolation = elec->trackIso();
     ecalIsolation = elec->ecalIso();
     hcalIsolation = elec->hcalIso();
     if(_requireID) electronId = elec->electronID("eidRobustLoose");
     else electronId = 1.0;
     idresult = false;
     if( electronId == 1.0 ) idresult = true;


     // ------ analysis cuts -----------------
     eClass = elec->classification();
     cutresult = cutDecision ( eClass, deltaEta, deltaPhi, sigmaEtaEta, 
			       tkIsolation, ecalIsolation, hcalIsolation);

     trigresult = CheckTriggerMatch( triggerObj, elecEta, elecPhi);

     if( cutresult && idresult && trigresult ) outCol->push_back(*elec);

     if(_verbose) {
       if( cutresult && idresult && trigresult ) 
	 std::cout << "%%%%%%%%%%%%%%  passing all cuts" << std::endl;
       else std::cout << "OOOOOOOOOOOOOO caution: failing all cuts" << std::endl;

       std::cout << "------------ deltaEta = " <<  deltaEta   << std::endl;
       std::cout << "------------ deltaPhi = " <<  deltaPhi   << std::endl;
       std::cout << "------------ sigmaEtaEta = " <<  sigmaEtaEta << std::endl;
       
       std::cout << "------------ track Iso = " <<  tkIsolation   << std::endl;
       std::cout << "------------ ecal Iso  = " <<  ecalIsolation << std::endl;
       std::cout << "------------ hcal Iso  = " <<  hcalIsolation << std::endl;
       std::cout << "------------ electron id  = " << electronId  << std::endl;
       
       std::cout << "trigresult = " << trigresult << std::endl;
     }

     // Fill all the histograms
     if( cutresult && idresult && trigresult ) {       
       TString hname;
       hname = "deltaEta";
       FillHist(hname,m_HistNames1D,deltaEta);
       hname = "deltaPhi";
       FillHist(hname,m_HistNames1D,deltaPhi);
       hname = "sigmaIetaIeta";
       FillHist(hname,m_HistNames1D,sigmaEtaEta);
       hname = "trackIso";
       FillHist(hname,m_HistNames1D,tkIsolation);
       hname = "ecalIso";
       FillHist(hname,m_HistNames1D,ecalIsolation);
       hname = "hcalIso";
       FillHist(hname,m_HistNames1D,hcalIsolation);
       hname = "elecId";
       FillHist(hname,m_HistNames1D,electronId);
       hname = "elecEta";
       FillHist(hname,m_HistNames1D,elecEta);
       hname = "elecPhi";
       FillHist(hname,m_HistNames1D,elecPhi);
       hname = "elecE";
       FillHist(hname,m_HistNames1D,elecE);
       hname = "elecEt";
       FillHist(hname,m_HistNames1D,elecEt);
     }

     ++index;
   }
   //
   event.put(outCol);
}







//////////////////////////////////////////////////////////////////////////////////////////
void PatElectronSelection::FillHist(const TString& histName, std::map<TString, TH1*> 
					HistNames, const double& x) 
{
  std::map<TString, TH1*>::iterator hid = HistNames.find(histName);
  if (hid==HistNames.end())
    std::cout << "%fillHist -- Could not find histogram with name: " << histName << std::endl;
  else
    hid->second->Fill(x);
}




// --------------- apply analysis cuts here --------------------

bool PatElectronSelection::cutDecision ( int classification, double deta, 
				  double dphi, double sietaeta, double tkiso,
				  double ecaliso, double hcaliso) {

  double deltaEtaCut_, deltaPhiCut_, sigmaEtaEtaCut_, 
    tkIsoCut_, ecalIsoCut_, hcalIsoCut_;

  if( classification < 100 ) {  // barrel
    deltaEtaCut_      = deltaEtaCutBarrel_;
    deltaPhiCut_      = deltaPhiCutBarrel_;
    sigmaEtaEtaCut_   = sigmaEtaEtaCutBarrel_;
    tkIsoCut_         = tkIsoCutBarrel_;
    ecalIsoCut_       = ecalIsoCutBarrel_;
    hcalIsoCut_       = hcalIsoCutBarrel_;
  }
  else {
    deltaEtaCut_     = deltaEtaCutEndcaps_;
    deltaPhiCut_     = deltaPhiCutEndcaps_;
    sigmaEtaEtaCut_  = sigmaEtaEtaCutEndcaps_;
    tkIsoCut_        = tkIsoCutEndcaps_;
    ecalIsoCut_      = ecalIsoCutEndcaps_;
    hcalIsoCut_      = hcalIsoCutEndcaps_;
  }


  // if we do not need to apply isolation

  if( !(_requireTkIso) )   tkIsoCut_   = 100000.0;
  if( !(_requireEcalIso) ) ecalIsoCut_ = 100000.0;
  if( !(_requireHcalIso) ) hcalIsoCut_ = 100000.0;


  bool decision = (fabs(deta) < deltaEtaCut_) && 
    (fabs(dphi) < deltaPhiCut_) &&
    (sietaeta < sigmaEtaEtaCut_) && (tkiso < tkIsoCut_) && 
    (ecaliso < ecalIsoCut_) && (hcaliso < hcalIsoCut_);

  return decision;
}






// ------------- perform trigger matching ----------------------

bool PatElectronSelection::CheckTriggerMatch( edm::Handle<trigger::TriggerEvent> triggerObj, 
				       double eta, double phi) {

  if ( !(_requireTrigMatch) ) return true;

  bool result = false;
  const trigger::TriggerObjectCollection & toc(triggerObj->getObjects());
  const int trigindex = triggerObj->filterIndex( hltFilter_ );
  if ( trigindex >= triggerObj->sizeFilters() ) return false; 

  const trigger::Keys & l1k = triggerObj->filterKeys( trigindex );
  if ( l1k.size() <= 0 ) return false; 


  for (trigger::Keys::const_iterator ki = l1k.begin(); ki !=l1k.end(); ++ki ) {
    
    if (reco::deltaR( eta, phi, toc[*ki].eta(),toc[*ki].phi()) < 0.3) 
      result = true;      /////// ---------- trigger match                 
  }
  
  return result;
}






// ----------- make sure that pt and eta are within the fiducial volume -----

bool PatElectronSelection::CheckAcceptance( double eta, double et) {

  bool withinFiducialEta = false;
  bool withinFiducialEt  = false;
  
  if ( fabs(eta) < _BarrelMaxEta ||  
       ( fabs(eta) > _EndcapMinEta && fabs(eta) < _EndcapMaxEta) )
    withinFiducialEta = true;
  
  if( et > _etMin && et < _etMax ) withinFiducialEt = true; 
  
  return withinFiducialEta && withinFiducialEt;
}








// --------- method called once each job just before starting event loop  ---

void PatElectronSelection::beginJob(const edm::EventSetup &eventSetup) { 

  m_file_ = new TFile(histogramFile_.c_str(),"RECREATE");
  TString hname;
  hname = "deltaEta";
  m_HistNames1D[hname] = new TH1F(hname,hname,1000, -0.1, 0.1); 
  hname = "deltaPhi";
  m_HistNames1D[hname] = new TH1F(hname,hname,1000, -0.1, 0.1); 
  hname = "sigmaIetaIeta";
  m_HistNames1D[hname] = new TH1F(hname,hname,1000, 0.0, 0.2); 
  hname = "trackIso";
  m_HistNames1D[hname] = new TH1F(hname,hname,10000, 0, 100); 
  hname = "ecalIso";
  m_HistNames1D[hname] = new TH1F(hname,hname,10000, 0, 100); 
  hname = "hcalIso";
  m_HistNames1D[hname] = new TH1F(hname,hname,10000, 0, 100); 
  hname = "elecId";
  m_HistNames1D[hname] = new TH1F(hname,hname,10, -5, 5); 
  hname = "elecEta";
  m_HistNames1D[hname] = new TH1F(hname,hname,1000, -3, 3); 
  hname = "elecPhi";
  m_HistNames1D[hname] = new TH1F(hname,hname,1000, -3.5, 3.5); 
  hname = "elecE";
  m_HistNames1D[hname] = new TH1F(hname,hname,5000, 0, 500); 
  hname = "elecEt";
  m_HistNames1D[hname] = new TH1F(hname,hname,5000, 0, 500); 
}



void PatElectronSelection::endJob() { 

  if (m_file_ !=0) 
    {
      m_file_->cd();
      for (std::map<TString, TH1*>::iterator hid = m_HistNames1D.begin(); 
	   hid != m_HistNames1D.end(); hid++)
        hid->second->Write();
      delete m_file_;
      m_file_ = 0;      
    }
}



//define this as a plug-in
DEFINE_FWK_MODULE( PatElectronSelection );
