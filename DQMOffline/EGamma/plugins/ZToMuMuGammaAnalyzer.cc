#include <iostream>
#include <iomanip>
//

#include "DQMOffline/EGamma/plugins/ZToMuMuGammaAnalyzer.h"
#include "CommonTools/UtilAlgos/interface/DeltaR.h"


/** \class ZToMuMuGammaAnalyzer
 **
 **
 **  $Id: ZToMuMuGammaAnalyzer
 **  $Date: 2012/01/26 13:48:55 $
 **  authors:
 **   Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

using namespace std;


ZToMuMuGammaAnalyzer::ZToMuMuGammaAnalyzer( const edm::ParameterSet& pset )
{

    fName_                  = pset.getUntrackedParameter<string>("Name");
    verbosity_              = pset.getUntrackedParameter<int>("Verbosity");
    prescaleFactor_         = pset.getUntrackedParameter<int>("prescaleFactor",1);
    standAlone_             = pset.getParameter<bool>("standAlone");
    outputFileName_         = pset.getParameter<string>("OutputFileName");
    isHeavyIon_             = pset.getUntrackedParameter<bool>("isHeavyIon",false);
    triggerEvent_           = pset.getParameter<edm::InputTag>("triggerEvent");
    useTriggerFiltering_    = pset.getParameter<bool>("useTriggerFiltering");
    //
    photonProducer_         = pset.getParameter<string>("phoProducer");
    photonCollection_       = pset.getParameter<string>("photonCollection");

    barrelRecHitProducer_   = pset.getParameter<string>("barrelRecHitProducer");
    barrelRecHitCollection_ = pset.getParameter<string>("barrelRecHitCollection");

    endcapRecHitProducer_   = pset.getParameter<string>("endcapRecHitProducer");
    endcapRecHitCollection_ = pset.getParameter<string>("endcapRecHitCollection");

    muonProducer_         = pset.getParameter<string>("muonProducer");
    muonCollection_       = pset.getParameter<string>("muonCollection");
    // Muon selection
    muonMinPt_             = pset.getParameter<double>("muonMinPt");
    minPixStripHits_       = pset.getParameter<int>("minPixStripHits");
    muonMaxChi2_           = pset.getParameter<double>("muonMaxChi2");
    muonMaxDxy_            = pset.getParameter<double>("muonMaxDxy");
    muonMatches_           = pset.getParameter<int>("muonMatches");
    validPixHits_          = pset.getParameter<int>("validPixHits");
    validMuonHits_         = pset.getParameter<int>("validMuonHits");
    muonTrackIso_          = pset.getParameter<double>("muonTrackIso");
    muonTightEta_          = pset.getParameter<double>("muonTightEta");
    // Dimuon selection
    minMumuInvMass_       = pset.getParameter<double>("minMumuInvMass");
    maxMumuInvMass_       = pset.getParameter<double>("maxMumuInvMass");
    // Photon selection
    photonMinEt_             = pset.getParameter<double>("photonMinEt");
    photonMaxEta_            = pset.getParameter<double>("photonMaxEta");
    photonTrackIso_          = pset.getParameter<double>("photonTrackIso");
    // mumuGamma selection
    nearMuonDr_               = pset.getParameter<double>("nearMuonDr");
    nearMuonHcalIso_          = pset.getParameter<double>("nearMuonHcalIso");
    farMuonEcalIso_           = pset.getParameter<double>("farMuonEcalIso");
    farMuonTrackIso_          = pset.getParameter<double>("farMuonTrackIso");
    farMuonMinPt_             = pset.getParameter<double>("farMuonMinPt");
    minMumuGammaInvMass_  = pset.getParameter<double>("minMumuGammaInvMass");
    maxMumuGammaInvMass_  = pset.getParameter<double>("maxMumuGammaInvMass");
    //
    parameters_ = pset;

}



ZToMuMuGammaAnalyzer::~ZToMuMuGammaAnalyzer() {}


void ZToMuMuGammaAnalyzer::beginJob()
{

  nEvt_=0;
  nEntry_=0;

  dbe_ = 0;
  dbe_ = edm::Service<DQMStore>().operator->();



  double eMin = parameters_.getParameter<double>("eMin");
  double eMax = parameters_.getParameter<double>("eMax");
  int    eBin = parameters_.getParameter<int>("eBin");

  double etMin = parameters_.getParameter<double>("etMin");
  double etMax = parameters_.getParameter<double>("etMax");
  int    etBin = parameters_.getParameter<int>("etBin");

  double sumMin = parameters_.getParameter<double>("sumMin");
  double sumMax = parameters_.getParameter<double>("sumMax");
  int    sumBin = parameters_.getParameter<int>("sumBin");

  double etaMin = parameters_.getParameter<double>("etaMin");
  double etaMax = parameters_.getParameter<double>("etaMax");
  int    etaBin = parameters_.getParameter<int>("etaBin");

  double phiMin = parameters_.getParameter<double>("phiMin");
  double phiMax = parameters_.getParameter<double>("phiMax");
  int    phiBin = parameters_.getParameter<int>("phiBin");

  double r9Min = parameters_.getParameter<double>("r9Min");
  double r9Max = parameters_.getParameter<double>("r9Max");
  int    r9Bin = parameters_.getParameter<int>("r9Bin");

  double hOverEMin = parameters_.getParameter<double>("hOverEMin");
  double hOverEMax = parameters_.getParameter<double>("hOverEMax");
  int    hOverEBin = parameters_.getParameter<int>("hOverEBin");

  double xMin = parameters_.getParameter<double>("xMin");
  double xMax = parameters_.getParameter<double>("xMax");
  int    xBin = parameters_.getParameter<int>("xBin");

  double yMin = parameters_.getParameter<double>("yMin");
  double yMax = parameters_.getParameter<double>("yMax");
  int    yBin = parameters_.getParameter<int>("yBin");

  double numberMin = parameters_.getParameter<double>("numberMin");
  double numberMax = parameters_.getParameter<double>("numberMax");
  int    numberBin = parameters_.getParameter<int>("numberBin");

  double zMin = parameters_.getParameter<double>("zMin");
  double zMax = parameters_.getParameter<double>("zMax");
  int    zBin = parameters_.getParameter<int>("zBin");

  double rMin = parameters_.getParameter<double>("rMin");
  double rMax = parameters_.getParameter<double>("rMax");
  int    rBin = parameters_.getParameter<int>("rBin");

  double dPhiTracksMin = parameters_.getParameter<double>("dPhiTracksMin");
  double dPhiTracksMax = parameters_.getParameter<double>("dPhiTracksMax");
  int    dPhiTracksBin = parameters_.getParameter<int>("dPhiTracksBin");

  double dEtaTracksMin = parameters_.getParameter<double>("dEtaTracksMin");
  double dEtaTracksMax = parameters_.getParameter<double>("dEtaTracksMax");
  int    dEtaTracksBin = parameters_.getParameter<int>("dEtaTracksBin");

  double sigmaIetaMin = parameters_.getParameter<double>("sigmaIetaMin");
  double sigmaIetaMax = parameters_.getParameter<double>("sigmaIetaMax");
  int    sigmaIetaBin = parameters_.getParameter<int>("sigmaIetaBin");

  double eOverPMin = parameters_.getParameter<double>("eOverPMin");
  double eOverPMax = parameters_.getParameter<double>("eOverPMax");
  int    eOverPBin = parameters_.getParameter<int>("eOverPBin");

  double chi2Min = parameters_.getParameter<double>("chi2Min");
  double chi2Max = parameters_.getParameter<double>("chi2Max");
  int    chi2Bin = parameters_.getParameter<int>("chi2Bin");


  int reducedEtBin  = etBin/4;
  int reducedEtaBin = etaBin/4;
  int reducedSumBin = sumBin/4;
  int reducedR9Bin  = r9Bin/4;


  ////////////////START OF BOOKING FOR ALL HISTOGRAMS////////////////

  if (dbe_) {


    dbe_->setCurrentFolder("Egamma/PhotonAnalyzer/ZToMuMuGamma");
    h1_mumuInvMass_ = dbe_->book1D("mumuInvMass","Two muon invariant mass: M (GeV)",etBin/2,etMin,etMax/2);
    h1_mumuGammaInvMass_ = dbe_->book1D("mumuGammaInvMass","Two-muon plus gamma invariant mass: M (GeV)",etBin/2,etMin,etMax/2);
 


  }//end if(dbe_)


}//end BeginJob



void ZToMuMuGammaAnalyzer::analyze( const edm::Event& e, const edm::EventSetup& esup )
{
  using namespace edm;

  if (nEvt_% prescaleFactor_ ) return;
  nEvt_++;
  LogInfo("ZToMuMuGammaAnalyzer") << "ZToMuMuGammaAnalyzer Analyzing event number: " << e.id() << " Global Counter " << nEvt_ <<"\n";


  // Get the trigger results
  bool validTriggerEvent=true;
  edm::Handle<trigger::TriggerEvent> triggerEventHandle;
  trigger::TriggerEvent triggerEvent;
  e.getByLabel(triggerEvent_,triggerEventHandle);
  if(!triggerEventHandle.isValid()) {
    edm::LogInfo("PhotonAnalyzer") << "Error! Can't get the product "<< triggerEvent_.label() << endl;
    validTriggerEvent=false;
  }
  if(validTriggerEvent) triggerEvent = *(triggerEventHandle.product());


  // Get the reconstructed photons
  bool validPhotons=true;
  Handle<reco::PhotonCollection> photonHandle;
  reco::PhotonCollection photonCollection;
  e.getByLabel(photonProducer_, photonCollection_ , photonHandle);
  if ( !photonHandle.isValid()) {
    edm::LogInfo("ZToMuMuGammaAnalyzer") << "Error! Can't get the product "<< photonCollection_ << endl;
    validPhotons=false;
  }
  if(validPhotons) photonCollection = *(photonHandle.product());

  // Get the PhotonId objects
  bool validloosePhotonID=true;
  Handle<edm::ValueMap<bool> > loosePhotonFlag;
  edm::ValueMap<bool> loosePhotonID;
  e.getByLabel("PhotonIDProd", "PhotonCutBasedIDLoose", loosePhotonFlag);
  if ( !loosePhotonFlag.isValid()) {
    edm::LogInfo("ZToMuMuGammaAnalyzer") << "Error! Can't get the product "<< "PhotonCutBasedIDLoose" << endl;
    validloosePhotonID=false;
  }
  if (validloosePhotonID) loosePhotonID = *(loosePhotonFlag.product());

  bool validtightPhotonID=true;
  Handle<edm::ValueMap<bool> > tightPhotonFlag;
  edm::ValueMap<bool> tightPhotonID;
  e.getByLabel("PhotonIDProd", "PhotonCutBasedIDTight", tightPhotonFlag);
  if ( !tightPhotonFlag.isValid()) {
    edm::LogInfo("ZToMuMuGammaAnalyzer") << "Error! Can't get the product "<< "PhotonCutBasedIDTight" << endl;
    validtightPhotonID=false;
  }
  if (validtightPhotonID) tightPhotonID = *(tightPhotonFlag.product());

  // Get the reconstructed muons
  bool validMuons=true;
  Handle<reco::MuonCollection> muonHandle;
  reco::MuonCollection muonCollection;
  e.getByLabel(muonProducer_, muonCollection_ , muonHandle);
  if ( !muonHandle.isValid()) {
    edm::LogInfo("ZToMuMuGammaAnalyzer") << "Error! Can't get the product "<< muonCollection_ << endl;
    validMuons=false;
  }
  if(validMuons) muonCollection = *(muonHandle.product());

  // Get the beam spot
  edm::Handle<reco::BeamSpot> bsHandle;
  e.getByLabel("offlineBeamSpot", bsHandle);
  if (!bsHandle.isValid()) {
      edm::LogError("TrackerOnlyConversionProducer") << "Error! Can't get the product primary Vertex Collection "<< "\n";
      return;
  }
  const reco::BeamSpot &thebs = *bsHandle.product();




  //Prepare list of photon-related HLT filter names
  vector<int> Keys;
  for(uint filterIndex=0;filterIndex<triggerEvent.sizeFilters();++filterIndex){  //loop over all trigger filters in event (i.e. filters passed)
    string label = triggerEvent.filterTag(filterIndex).label();
    if(label.find( "Photon" ) != string::npos ) {  //get photon-related filters
      for(uint filterKeyIndex=0;filterKeyIndex<triggerEvent.filterKeys(filterIndex).size();++filterKeyIndex){  //loop over keys to objects passing this filter
	Keys.push_back(triggerEvent.filterKeys(filterIndex)[filterKeyIndex]);  //add keys to a vector for later reference
      }
    }
  }

  // sort Keys vector in ascending order
  // and erases duplicate entries from the vector
  sort(Keys.begin(),Keys.end());
  for ( uint i=0 ; i<Keys.size() ; )
   {
    if (i!=(Keys.size()-1))
     {
      if (Keys[i]==Keys[i+1]) Keys.erase(Keys.begin()+i+1) ;
      else ++i ;
     }
    else ++i ;
   }



  ////////////// event selection
  if ( muonCollection.size() < 2 ) return;

  for( reco::MuonCollection::const_iterator  iMu = muonCollection.begin(); iMu != muonCollection.end(); iMu++) {
    if ( !basicMuonSelection (*iMu) ) continue;
 
    for( reco::MuonCollection::const_iterator  iMu2 = iMu+1; iMu2 != muonCollection.end(); iMu2++) {
      if ( !basicMuonSelection (*iMu2) ) continue;
      if ( iMu->charge()*iMu2->charge() > 0) continue;

      if ( !muonSelection(*iMu,thebs) && !muonSelection(*iMu2,thebs) ) continue;

 
     
      float mumuMass = mumuInvMass(*iMu,*iMu2) ;
      if ( mumuMass <  minMumuInvMass_  ||  mumuMass >  maxMumuInvMass_ ) continue;

      h1_mumuInvMass_ -> Fill (mumuMass);      

      if (  photonCollection.size() < 1 ) continue;

      reco::Muon nearMuon;
      reco::Muon farMuon;
      for( reco::PhotonCollection::const_iterator  iPho = photonCollection.begin(); iPho != photonCollection.end(); iPho++) {
	if ( !photonSelection (*iPho) ) continue;


	DeltaR<reco::Muon, reco::Photon> deltaR;
	double dr1 = deltaR(*iMu, *iPho);
	double dr2 = deltaR(*iMu2,*iPho);
	double drNear = dr1;
	if (dr1 < dr2) {
	  nearMuon =*iMu ; farMuon  = *iMu2; drNear = dr1;
	} else {
	  nearMuon = *iMu2; farMuon  = *iMu; drNear = dr2;
	}

	if ( nearMuon.isolationR03().hadEt > nearMuonHcalIso_ )  continue;
        if ( farMuon.isolationR03().sumPt > farMuonTrackIso_ )  continue;
        if ( farMuon.isolationR03().emEt  > farMuonEcalIso_ )  continue;
	if ( farMuon.pt() < farMuonMinPt_ )       continue;
	if ( drNear > nearMuonDr_)               continue;


	float mumuGammaMass = mumuGammaInvMass(*iMu,*iMu2,*iPho) ;
        if ( mumuGammaMass < minMumuGammaInvMass_ || mumuGammaMass > maxMumuGammaInvMass_ ) continue;
	    
	h1_mumuGammaInvMass_ ->Fill (mumuGammaMass);


      } 
      
    }

  }


}//End of Analyze method

void ZToMuMuGammaAnalyzer::endRun(const edm::Run& run, const edm::EventSetup& setup)
{
  if(!standAlone_){

    dbe_->setCurrentFolder("Egamma/PhotonAnalyzer/ZToMuMuGamma");


  }

}


void ZToMuMuGammaAnalyzer::endJob()
{
  //dbe_->showDirStructure();
  if(standAlone_){
    dbe_->setCurrentFolder("Egamma/PhotonAnalyzer/ZToMuMuGamma");

    dbe_->save(outputFileName_);
  }


}

bool ZToMuMuGammaAnalyzer::basicMuonSelection ( const reco::Muon & mu) {
  bool result=true;
  if (!mu.innerTrack().isNonnull())    result=false;
  if (!mu.globalTrack().isNonnull())   result=false;
  if ( !mu.isGlobalMuon() )            result=false; 
  if ( mu.pt() < muonMinPt_ )                  result=false;
  if ( fabs(mu.eta())>2.4 )            result=false;

  int pixHits=0;
  int tkHits=0;
  if ( mu.innerTrack().isNonnull() ) {
    pixHits=mu.innerTrack()->hitPattern().numberOfValidPixelHits();
    tkHits=mu.innerTrack()->hitPattern().numberOfValidStripHits();
  }

  if ( pixHits+tkHits < minPixStripHits_ ) result=false;
  

  return result;  
}

bool ZToMuMuGammaAnalyzer::muonSelection ( const reco::Muon & mu,  const reco::BeamSpot& beamSpot) {
  bool result=true;
  if ( mu.globalTrack()->normalizedChi2() > muonMaxChi2_ )          result=false;
  if ( fabs( mu.globalTrack()->dxy(beamSpot)) > muonMaxDxy_ )       result=false;
  if ( mu.numberOfMatches() < muonMatches_ )                                   result=false;

  if ( mu.track()-> hitPattern().numberOfValidPixelHits() <  validPixHits_ )     result=false;
  if ( mu.globalTrack()->hitPattern().numberOfValidMuonHits() < validMuonHits_ ) result=false;
  if ( !mu.isTrackerMuon() )                                        result=false;
  // track isolation 
  if ( mu.isolationR03().sumPt > muonTrackIso_ )                                result=false;
  if ( fabs(mu.eta())>  muonTightEta_ )                                         result=false;
 

  return result;  
}


bool ZToMuMuGammaAnalyzer::photonSelection ( const reco::Photon & pho) {
  bool result=true;
  if ( pho.pt() < photonMinEt_ )          result=false;
  if ( fabs(pho.eta())> photonMaxEta_ )   result=false;
  if ( pho.isEBEEGap() )       result=false;
  if ( pho.trkSumPtHollowConeDR04() >   photonTrackIso_ )   result=false; // check how to exclude the muon track (which muon track).


  return result;  
}



float ZToMuMuGammaAnalyzer::mumuInvMass(const reco::Muon & mu1,const reco::Muon & mu2 )
 {
  math::XYZTLorentzVector p12 = mu1.p4()+mu2.p4() ;
  float mumuMass2 = p12.Dot(p12) ;
  float invMass = sqrt(mumuMass2) ;
  return invMass ;
 }

float ZToMuMuGammaAnalyzer::mumuGammaInvMass(const reco::Muon & mu1,const reco::Muon & mu2, const reco::Photon& pho )
 {
   math::XYZTLorentzVector p12 = mu1.p4()+mu2.p4()+pho.p4() ;
   float Mass2 = p12.Dot(p12) ;
   float invMass = sqrt(Mass2) ;
   return invMass ;
 }

