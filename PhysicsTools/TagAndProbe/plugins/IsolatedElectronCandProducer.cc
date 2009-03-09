// Framework
#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/TagAndProbe/interface/IsolatedElectronCandProducer.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/ElectronTkIsolation.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

IsolatedElectronCandProducer::IsolatedElectronCandProducer(const edm::ParameterSet &params)
{
  // use configuration file to setup input/output collection names
  electronProducer_  = params.getParameter<edm::InputTag>("electronProducer");
  trackProducer_     = params.getParameter<edm::InputTag>("trackProducer");
  ptMin_             = params.getParameter<double>("ptMin");
  intRadius_         = params.getParameter<double>("intRadius");
  extRadius_         = params.getParameter<double>("extRadius");
  maxVtxDist_        = params.getParameter<double>("maxVtxDist");
  absolut_           = params.getParameter<bool>("absolut");
  isoCut_            = params.getParameter<double>( "isoCut" );
  beamspotProducer_  = params.getParameter<edm::InputTag>("BeamspotProducer");
  drb_               = params.getParameter<double>("maxVtxDistXY");

  //register your products
  produces < reco::GsfElectronCollection>();

}

IsolatedElectronCandProducer::~IsolatedElectronCandProducer(){}


								       
void IsolatedElectronCandProducer::produce(edm::Event& iEvent, 
					   const edm::EventSetup& iSetup)
{

   std::auto_ptr<reco::GsfElectronCollection> 
     outCol(new reco::GsfElectronCollection);

  // Get the  filtered objects
  edm::Handle< reco::GsfElectronCollection> electronHandle;
  try
    {
      iEvent.getByLabel(electronProducer_,electronHandle);
    }
  catch(cms::Exception &ex)
    {
      edm::LogError("GsfElectron ") << "Error! Can't get collection " << 
	electronProducer_;
      throw ex;
    }
      
  
  //get the tracks
  edm::Handle<reco::TrackCollection> tracks;
  try
    {
      iEvent.getByLabel(trackProducer_,tracks);
    }
  catch(cms::Exception &ex)
    {
      edm::LogError("Tracks ") << "Error! Can't get collection " << 
	trackProducer_;
      throw ex;
    }
  
  const reco::TrackCollection* trackCollection = tracks.product();

  edm::Handle<reco::BeamSpot> beamSpotH;
  iEvent.getByLabel(beamspotProducer_,beamSpotH);
  reco::TrackBase::Point beamspot = beamSpotH->position();

  ElectronTkIsolation myTkIsolation (extRadius_,intRadius_, 
				     ptMin_,maxVtxDist_,drb_,trackCollection,beamspot) ;
  
  for(unsigned int i = 0 ; i < electronHandle->size(); ++i ){

    edm::Ref<reco::GsfElectronCollection> electronRef(electronHandle, i);

    double isoValue = myTkIsolation.getPtTracks(&(electronHandle->at(i)));

    if(absolut_==false){
      reco::SuperClusterRef sc = (electronHandle->at(i)).superCluster();
      double et = sc.get()->energy()*sin(2*atan(exp(-sc.get()->eta())));
      isoValue /= et;
    }

    if(isoValue<isoCut_) outCol->push_back(*electronRef);
  }
  
  iEvent.put(outCol);
}


// ------- method called once each job just before starting event loop --


void IsolatedElectronCandProducer::beginJob(const edm::EventSetup &iSetup){}


void IsolatedElectronCandProducer::endJob() {}



//define this as a plug-in
DEFINE_FWK_MODULE( IsolatedElectronCandProducer );
