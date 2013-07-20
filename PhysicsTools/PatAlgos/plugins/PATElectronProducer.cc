// $Id: PATElectronProducer.cc,v 1.75 2013/04/12 09:11:18 beaudett Exp $
//
#include "PhysicsTools/PatAlgos/plugins/PATElectronProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "PhysicsTools/PatUtils/interface/TrackerIsolationPt.h"
#include "PhysicsTools/PatUtils/interface/CaloIsolationEnergy.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"


#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/IPTools/interface/IPTools.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include <vector>
#include <memory>


using namespace pat;
using namespace std;


PATElectronProducer::PATElectronProducer(const edm::ParameterSet & iConfig) :
  isolator_(iConfig.exists("userIsolation") ? iConfig.getParameter<edm::ParameterSet>("userIsolation") : edm::ParameterSet(), false) ,
  useUserData_(iConfig.exists("userData"))
{
  // general configurables
  electronSrc_ = iConfig.getParameter<edm::InputTag>( "electronSource" );
  embedGsfElectronCore_ = iConfig.getParameter<bool>( "embedGsfElectronCore" );
  embedGsfTrack_ = iConfig.getParameter<bool>( "embedGsfTrack" );
  embedSuperCluster_ = iConfig.getParameter<bool>         ( "embedSuperCluster"    );
  embedPflowSuperCluster_ = iConfig.getParameter<bool>    ( "embedPflowSuperCluster"    );
  embedSeedCluster_ = iConfig.getParameter<bool>( "embedSeedCluster" );
  embedBasicClusters_ = iConfig.getParameter<bool>( "embedBasicClusters" );
  embedPreshowerClusters_ = iConfig.getParameter<bool>( "embedPreshowerClusters" );
  embedPflowBasicClusters_ = iConfig.getParameter<bool>( "embedPflowBasicClusters" );
  embedPflowPreshowerClusters_ = iConfig.getParameter<bool>( "embedPflowPreshowerClusters" );
  embedTrack_ = iConfig.getParameter<bool>( "embedTrack" );
  embedRecHits_ = iConfig.getParameter<bool>( "embedRecHits" );
  // pflow configurables
  pfElecSrc_ = iConfig.getParameter<edm::InputTag>( "pfElectronSource" );
  pfCandidateMap_ = iConfig.getParameter<edm::InputTag>( "pfCandidateMap" );
  useParticleFlow_ = iConfig.getParameter<bool>( "useParticleFlow" );
  embedPFCandidate_ = iConfig.getParameter<bool>( "embedPFCandidate" );
  // mva input variables
  reducedBarrelRecHitCollection_ = iConfig.getParameter<edm::InputTag>("reducedBarrelRecHitCollection");
  reducedEndcapRecHitCollection_ = iConfig.getParameter<edm::InputTag>("reducedEndcapRecHitCollection");
  // MC matching configurables (scheduled mode)
  addGenMatch_ = iConfig.getParameter<bool>( "addGenMatch" );
  if (addGenMatch_) {
    embedGenMatch_ = iConfig.getParameter<bool>( "embedGenMatch" );
    if (iConfig.existsAs<edm::InputTag>("genParticleMatch")) {
      genMatchSrc_.push_back(iConfig.getParameter<edm::InputTag>( "genParticleMatch" ));
    }
    else {
      genMatchSrc_ = iConfig.getParameter<std::vector<edm::InputTag> >( "genParticleMatch" );
    }
  }
  // resolution configurables
  addResolutions_ = iConfig.getParameter<bool>( "addResolutions" );
  if (addResolutions_) {
    resolutionLoader_ = pat::helper::KinResolutionsLoader(iConfig.getParameter<edm::ParameterSet>("resolutions"));
  }
  // electron ID configurables
  addElecID_ = iConfig.getParameter<bool>( "addElectronID" );
  if (addElecID_) {
    // it might be a single electron ID
    if (iConfig.existsAs<edm::InputTag>("electronIDSource")) {
      elecIDSrcs_.push_back(NameTag("", iConfig.getParameter<edm::InputTag>("electronIDSource")));
    }
    // or there might be many of them
    if (iConfig.existsAs<edm::ParameterSet>("electronIDSources")) {
      // please don't configure me twice
      if (!elecIDSrcs_.empty()){
	throw cms::Exception("Configuration") << "PATElectronProducer: you can't specify both 'electronIDSource' and 'electronIDSources'\n";
      }
      // read the different electron ID names
      edm::ParameterSet idps = iConfig.getParameter<edm::ParameterSet>("electronIDSources");
      std::vector<std::string> names = idps.getParameterNamesForType<edm::InputTag>();
      for (std::vector<std::string>::const_iterator it = names.begin(), ed = names.end(); it != ed; ++it) {
	elecIDSrcs_.push_back(NameTag(*it, idps.getParameter<edm::InputTag>(*it)));
      }
    }
    // but in any case at least once
    if (elecIDSrcs_.empty()){
      throw cms::Exception("Configuration") <<
	"PATElectronProducer: id addElectronID is true, you must specify either:\n" <<
	"\tInputTag electronIDSource = <someTag>\n" << "or\n" <<
	"\tPSet electronIDSources = { \n" <<
	"\t\tInputTag <someName> = <someTag>   // as many as you want \n " <<
	"\t}\n";
    }
  }
  // construct resolution calculator

  //   // IsoDeposit configurables
  //   if (iConfig.exists("isoDeposits")) {
  //      edm::ParameterSet depconf = iConfig.getParameter<edm::ParameterSet>("isoDeposits");
  //      if (depconf.exists("tracker")) isoDepositLabels_.push_back(std::make_pair(TrackerIso, depconf.getParameter<edm::InputTag>("tracker")));
  //      if (depconf.exists("ecal"))    isoDepositLabels_.push_back(std::make_pair(ECalIso, depconf.getParameter<edm::InputTag>("ecal")));
  //      if (depconf.exists("hcal"))    isoDepositLabels_.push_back(std::make_pair(HCalIso, depconf.getParameter<edm::InputTag>("hcal")));


  //      if (depconf.exists("user")) {
  //         std::vector<edm::InputTag> userdeps = depconf.getParameter<std::vector<edm::InputTag> >("user");
  //         std::vector<edm::InputTag>::const_iterator it = userdeps.begin(), ed = userdeps.end();
  //         int key = UserBaseIso;
  //         for ( ; it != ed; ++it, ++key) {
  //             isoDepositLabels_.push_back(std::make_pair(IsolationKeys(key), *it));
  //         }
  //      }
  //   }

  // read isoDeposit labels, for direct embedding
  readIsolationLabels(iConfig, "isoDeposits", isoDepositLabels_);
  // read isolation value labels, for direct embedding
  readIsolationLabels(iConfig, "isolationValues", isolationValueLabels_);
  // read isolation value labels for non PF identified electron, for direct embedding
  readIsolationLabels(iConfig, "isolationValuesNoPFId", isolationValueLabelsNoPFId_);
  // Efficiency configurables
  addEfficiencies_ = iConfig.getParameter<bool>("addEfficiencies");
  if (addEfficiencies_) {
    efficiencyLoader_ = pat::helper::EfficiencyLoader(iConfig.getParameter<edm::ParameterSet>("efficiencies"));
  }
  // Check to see if the user wants to add user data
  if ( useUserData_ ) {
    userDataHelper_ = PATUserDataHelper<Electron>(iConfig.getParameter<edm::ParameterSet>("userData"));
  }
  // embed high level selection variables?
  embedHighLevelSelection_ = iConfig.getParameter<bool>("embedHighLevelSelection");
  beamLineSrc_ = iConfig.getParameter<edm::InputTag>("beamLineSrc");
  if ( embedHighLevelSelection_ ) {
    usePV_ = iConfig.getParameter<bool>("usePV");
    pvSrc_ = iConfig.getParameter<edm::InputTag>("pvSrc");
  }
  // produces vector of muons
  produces<std::vector<Electron> >();
  }


  PATElectronProducer::~PATElectronProducer()
{
}


void PATElectronProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup)
{
  // switch off embedding (in unschedules mode)
  if (iEvent.isRealData()){
    addGenMatch_ = false;
    embedGenMatch_ = false;
  }

  edm::ESHandle<CaloTopology> theCaloTopology;
  iSetup.get<CaloTopologyRecord>().get(theCaloTopology);
  ecalTopology_ = & (*theCaloTopology);

  // Get the collection of electrons from the event
  edm::Handle<edm::View<reco::GsfElectron> > electrons;
  iEvent.getByLabel(electronSrc_, electrons);

  // for additional mva variables
  edm::InputTag  reducedEBRecHitCollection(string("reducedEcalRecHitsEB"));
  edm::InputTag  reducedEERecHitCollection(string("reducedEcalRecHitsEE"));
  //EcalClusterLazyTools lazyTools(iEvent, iSetup, reducedEBRecHitCollection, reducedEERecHitCollection);
  EcalClusterLazyTools lazyTools(iEvent, iSetup, reducedBarrelRecHitCollection_, reducedEndcapRecHitCollection_);

  // for conversion veto selection
  edm::Handle<reco::ConversionCollection> hConversions;
  iEvent.getByLabel("allConversions", hConversions);

  // Get the ESHandle for the transient track builder, if needed for
  // high level selection embedding
  edm::ESHandle<TransientTrackBuilder> trackBuilder;

  if (isolator_.enabled()) isolator_.beginEvent(iEvent,iSetup);

  if (efficiencyLoader_.enabled()) efficiencyLoader_.newEvent(iEvent);
  if (resolutionLoader_.enabled()) resolutionLoader_.newEvent(iEvent, iSetup);

  IsoDepositMaps deposits(isoDepositLabels_.size());
  for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
    iEvent.getByLabel(isoDepositLabels_[j].second, deposits[j]);
  }

  IsolationValueMaps isolationValues(isolationValueLabels_.size());
  for (size_t j = 0; j<isolationValueLabels_.size(); ++j) {
    iEvent.getByLabel(isolationValueLabels_[j].second, isolationValues[j]);
  }

  IsolationValueMaps isolationValuesNoPFId(isolationValueLabelsNoPFId_.size());
  for (size_t j = 0; j<isolationValueLabelsNoPFId_.size(); ++j) {
    iEvent.getByLabel(isolationValueLabelsNoPFId_[j].second, isolationValuesNoPFId[j]);
  }

  // prepare the MC matching
  GenAssociations  genMatches(genMatchSrc_.size());
  if (addGenMatch_) {
    for (size_t j = 0, nd = genMatchSrc_.size(); j < nd; ++j) {
      iEvent.getByLabel(genMatchSrc_[j], genMatches[j]);
    }
  }

  // prepare ID extraction
  std::vector<edm::Handle<edm::ValueMap<float> > > idhandles;
  std::vector<pat::Electron::IdPair>               ids;
  if (addElecID_) {
    idhandles.resize(elecIDSrcs_.size());
    ids.resize(elecIDSrcs_.size());
    for (size_t i = 0; i < elecIDSrcs_.size(); ++i) {
      iEvent.getByLabel(elecIDSrcs_[i].second, idhandles[i]);
      ids[i].first = elecIDSrcs_[i].first;
    }
  }


  // prepare the high level selection:
  // needs beamline
  reco::TrackBase::Point beamPoint(0,0,0);
  reco::Vertex primaryVertex;
  reco::BeamSpot beamSpot;
  bool beamSpotIsValid = false;
  bool primaryVertexIsValid = false;

  // Get the beamspot
  edm::Handle<reco::BeamSpot> beamSpotHandle;
  iEvent.getByLabel(beamLineSrc_, beamSpotHandle);

  if ( embedHighLevelSelection_ ) {
    // Get the primary vertex
    edm::Handle< std::vector<reco::Vertex> > pvHandle;
    iEvent.getByLabel( pvSrc_, pvHandle );

    // This is needed by the IPTools methods from the tracking group
    iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", trackBuilder);

    if ( ! usePV_ ) {

      if ( beamSpotHandle.isValid() ){
        beamSpot = *beamSpotHandle;
        beamSpotIsValid = true;
      } else{
        edm::LogError("DataNotAvailable")
          << "No beam spot available from EventSetup, not adding high level selection \n";
      }

      double x0 = beamSpot.x0();
      double y0 = beamSpot.y0();
      double z0 = beamSpot.z0();

      beamPoint = reco::TrackBase::Point ( x0, y0, z0 );
    } else {
      if ( pvHandle.isValid() && !pvHandle->empty() ) {
	primaryVertex = pvHandle->at(0);
	primaryVertexIsValid = true;
      } else {
	edm::LogError("DataNotAvailable")
	  << "No primary vertex available from EventSetup, not adding high level selection \n";
      }
    }
  }

  std::vector<Electron> * patElectrons = new std::vector<Electron>();

  if( useParticleFlow_ ) {
    edm::Handle< reco::PFCandidateCollection >  pfElectrons;
    iEvent.getByLabel(pfElecSrc_, pfElectrons);
    unsigned index=0;

    for( reco::PFCandidateConstIterator i = pfElectrons->begin();
	 i != pfElectrons->end(); ++i, ++index) {

      reco::PFCandidateRef pfRef(pfElectrons, index);
      reco::PFCandidatePtr ptrToPFElectron(pfElectrons,index);
//       reco::CandidateBaseRef pfBaseRef( pfRef );

      reco::GsfTrackRef PfTk= i->gsfTrackRef();

      bool Matched=false;
      bool MatchedToAmbiguousGsfTrack=false;
      for (edm::View<reco::GsfElectron>::const_iterator itElectron = electrons->begin(); itElectron != electrons->end(); ++itElectron) {
	unsigned int idx = itElectron - electrons->begin();
	if (Matched || MatchedToAmbiguousGsfTrack) continue;

	reco::GsfTrackRef EgTk= itElectron->gsfTrack();

	if (itElectron->gsfTrack()==i->gsfTrackRef()){
	  Matched=true;
	}
	else {
	  for( reco::GsfTrackRefVector::const_iterator it = itElectron->ambiguousGsfTracksBegin() ;
	       it!=itElectron->ambiguousGsfTracksEnd(); it++ ){
	    MatchedToAmbiguousGsfTrack |= (bool)(i->gsfTrackRef()==(*it));
	  }
	}

	if (Matched || MatchedToAmbiguousGsfTrack){

	  // ptr needed for finding the matched gen particle
	  reco::CandidatePtr ptrToGsfElectron(electrons,idx);

	  // ref to base needed for the construction of the pat object
	  const edm::RefToBase<reco::GsfElectron>& elecsRef = electrons->refAt(idx);
	  Electron anElectron(elecsRef);
	  anElectron.setPFCandidateRef( pfRef  );

          //it should be always true when particleFlow electrons are used.
          anElectron.setIsPF( true );

	  if( embedPFCandidate_ ) anElectron.embedPFCandidate();

	  if ( useUserData_ ) {
	    userDataHelper_.add( anElectron, iEvent, iSetup );
	  }

          double ip3d = -999; // for mva variable

	  // embed high level selection
	  if ( embedHighLevelSelection_ ) {
	    // get the global track
	    reco::GsfTrackRef track = PfTk;

	    // Make sure the collection it points to is there
	    if ( track.isNonnull() && track.isAvailable() ) {

	      reco::TransientTrack tt = trackBuilder->build(track);
	      embedHighLevel( anElectron,
			      track,
			      tt,
			      primaryVertex,
			      primaryVertexIsValid,
			      beamSpot,
			      beamSpotIsValid );

              std::pair<bool,Measurement1D> ip3dpv = IPTools::absoluteImpactParameter3D(tt, primaryVertex);
              ip3d = ip3dpv.second.value(); // for mva variable

	      if ( !usePV_ ) {
		double corr_d0 = track->dxy( beamPoint );
		anElectron.setDB( corr_d0, -1.0 );
	      } else {
                 std::pair<bool,Measurement1D> result = IPTools::absoluteTransverseImpactParameter(tt, primaryVertex);
                double d0_corr = result.second.value();
                double d0_err = result.second.error();
		anElectron.setDB( d0_corr, d0_err );
	      }
	    }
	  }

	  //Electron Id

	  if (addElecID_) {
	    //STANDARD EL ID
	    for (size_t i = 0; i < elecIDSrcs_.size(); ++i) {
	      ids[i].second = (*idhandles[i])[elecsRef];
	    }
	    //SPECIFIC PF ID
	    ids.push_back(std::make_pair("pf_evspi",pfRef->mva_e_pi()));
	    ids.push_back(std::make_pair("pf_evsmu",pfRef->mva_e_mu()));
	    anElectron.setElectronIDs(ids);
	  }

          // add missing mva variables
          double r9 = lazyTools.e3x3( *( itElectron->superCluster()->seed())) / itElectron->superCluster()->rawEnergy() ;
          double sigmaIphiIphi;
          double sigmaIetaIphi;
          std::vector<float> vCov = lazyTools.localCovariances(*( itElectron->superCluster()->seed()));
          if( !edm::isNotFinite(vCov[2])) sigmaIphiIphi = sqrt(vCov[2]);
          else sigmaIphiIphi = 0;
          sigmaIetaIphi = vCov[1];
          anElectron.setMvaVariables( r9, sigmaIphiIphi, sigmaIetaIphi, ip3d);

	  // get list of EcalDetId within 5x5 around the seed 
	  bool barrel = itElectron->isEB();
	  DetId seed = lazyTools.getMaximum(*(itElectron->superCluster()->seed())).first;
	  std::vector<DetId> selectedCells = (barrel) ? ecalTopology_->getSubdetectorTopology(DetId::Ecal,EcalBarrel)->getWindow(seed,5,5):
	    ecalTopology_->getSubdetectorTopology(DetId::Ecal,EcalEndcap)->getWindow(seed,5,5);          

	  // Do it for all basic clusters in 5x5
	  reco::CaloCluster_iterator itscl = itElectron->superCluster()->clustersBegin();
	  reco::CaloCluster_iterator itsclE = itElectron->superCluster()->clustersEnd();
	  std::vector<DetId> cellsIn5x5;
	  for ( ; itscl!= itsclE ; ++ itscl) {
	    DetId seed=lazyTools.getMaximum(*(*itscl)).first;
	    bool bcbarrel = seed.subdetId()==EcalBarrel; 
	    std::vector<DetId> cellsToAdd = (bcbarrel) ? ecalTopology_->getSubdetectorTopology(DetId::Ecal,EcalBarrel)->getWindow(seed,5,5):
	      ecalTopology_->getSubdetectorTopology(DetId::Ecal,EcalEndcap)->getWindow(seed,5,5);
	    cellsIn5x5.insert(cellsIn5x5.end(),cellsToAdd.begin(), cellsToAdd.end());

	  }

	  // Add to the list of selectedCells checking that there is no duplicate 
	  unsigned nCellsIn5x5 = cellsIn5x5.size() ;

	  for(unsigned i=0; i< nCellsIn5x5 ; ++i ) {
	    std::vector<DetId>::const_iterator itcheck = find(selectedCells.begin(), selectedCells.end(),cellsIn5x5[i]);
	    if (itcheck == selectedCells.end())
	      selectedCells.push_back(cellsIn5x5[i]);
	  }


	  // add the DetId of the SC
	  std::vector< std::pair<DetId, float> >::const_iterator it=itElectron->superCluster()->hitsAndFractions().begin();
	  std::vector< std::pair<DetId, float> >::const_iterator itend=itElectron->superCluster()->hitsAndFractions().end();
	  for( ; it!=itend ; ++it) {
	    DetId id=it->first;
	    // check if already saved
	    std::vector<DetId>::const_iterator itcheck = find(selectedCells.begin(),selectedCells.end(),id);
	    if ( itcheck == selectedCells.end()) {
	      selectedCells.push_back(id);
	    }
	  }
	  // Retrieve the corresponding RecHits

	  edm::Handle< EcalRecHitCollection > rechitsH ;
	  if(barrel) 
	    iEvent.getByLabel(reducedBarrelRecHitCollection_,rechitsH);
	  else
	    iEvent.getByLabel(reducedEndcapRecHitCollection_,rechitsH);

	  EcalRecHitCollection selectedRecHits;
	  const EcalRecHitCollection *recHits = rechitsH.product();

	  unsigned nSelectedCells = selectedCells.size();
	  for (unsigned icell = 0 ; icell < nSelectedCells ; ++icell) {
	   EcalRecHitCollection::const_iterator  it = recHits->find( selectedCells[icell] );
	    if ( it != recHits->end() ) {
	      selectedRecHits.push_back(*it);
	    }
	  }
	  selectedRecHits.sort();
	  if (embedRecHits_) anElectron.embedRecHits(& selectedRecHits);
         
	    // set conversion veto selection
          bool passconversionveto = false;
          if( hConversions.isValid()){
            // this is recommended method
            passconversionveto = !ConversionTools::hasMatchedConversion( *itElectron, hConversions, beamSpotHandle->position());
          }else{
            // use missing hits without vertex fit method
            passconversionveto = itElectron->gsfTrack()->trackerExpectedHitsInner().numberOfLostHits() < 1;
          }

          anElectron.setPassConversionVeto( passconversionveto );


// 	  fillElectron(anElectron,elecsRef,pfBaseRef,
// 		       genMatches, deposits, isolationValues);

	  //COLIN small warning !
	  // we are currently choosing to take the 4-momentum of the PFCandidate;
	  // the momentum of the GsfElectron is saved though
	  // we must therefore match the GsfElectron.
	  // because of this, we should not change the source of the electron matcher
	  // to the collection of PFElectrons in the python configuration
	  // I don't know what to do with the efficiencyLoader, since I don't know
	  // what this class is for.
	  fillElectron2( anElectron,
			 ptrToPFElectron,
			 ptrToGsfElectron,
			 ptrToGsfElectron,
			 genMatches, deposits, isolationValues );

	  //COLIN need to use fillElectron2 in the non-pflow case as well, and to test it.

	  patElectrons->push_back(anElectron);
	}
      }
      //if( !Matched && !MatchedToAmbiguousGsfTrack) std::cout << "!!!!A pf electron could not be matched to a gsf!!!!"  << std::endl;
    }
  }

  else{
    // Try to access PF electron collection
    edm::Handle<edm::ValueMap<reco::PFCandidatePtr> >ValMapH;
    bool valMapPresent = iEvent.getByLabel(pfCandidateMap_,ValMapH);
    // Try to access a PFCandidate collection, as supplied by the user
    edm::Handle< reco::PFCandidateCollection >  pfElectrons;
    bool pfCandsPresent = iEvent.getByLabel(pfElecSrc_, pfElectrons);

    for (edm::View<reco::GsfElectron>::const_iterator itElectron = electrons->begin(); itElectron != electrons->end(); ++itElectron) {
      // construct the Electron from the ref -> save ref to original object
      //FIXME: looks like a lot of instances could be turned into const refs
      unsigned int idx = itElectron - electrons->begin();
      edm::RefToBase<reco::GsfElectron> elecsRef = electrons->refAt(idx);
      reco::CandidateBaseRef elecBaseRef(elecsRef);
      Electron anElectron(elecsRef);

      // Is this GsfElectron also identified as an e- in the particle flow?
      bool pfId = false;

      if ( pfCandsPresent ) {
	// PF electron collection not available.
	const reco::GsfTrackRef& trkRef = itElectron->gsfTrack();
	int index = 0;
	for( reco::PFCandidateConstIterator ie = pfElectrons->begin();
	     ie != pfElectrons->end(); ++ie, ++index) {
	  if(ie->particleId()!=reco::PFCandidate::e) continue;
	  const reco::GsfTrackRef& pfTrkRef= ie->gsfTrackRef();
	  if( trkRef == pfTrkRef ) {
	    pfId = true;
	    reco::PFCandidateRef pfRef(pfElectrons, index);
	    anElectron.setPFCandidateRef( pfRef );
	    break;
	  }
	}
      }
      else if( valMapPresent ) {
        // use value map if PF collection not available
	const edm::ValueMap<reco::PFCandidatePtr> & myValMap(*ValMapH);
	// Get the PFCandidate
	const reco::PFCandidatePtr& pfElePtr(myValMap[elecsRef]);
	pfId= pfElePtr.isNonnull();
      }
      // set PFId function
      anElectron.setIsPF( pfId );

      // add resolution info

      // Isolation
      if (isolator_.enabled()) {
        isolator_.fill(*electrons, idx, isolatorTmpStorage_);
        typedef pat::helper::MultiIsolator::IsolationValuePairs IsolationValuePairs;
        // better to loop backwards, so the vector is resized less times
        for (IsolationValuePairs::const_reverse_iterator it = isolatorTmpStorage_.rbegin(), ed = isolatorTmpStorage_.rend(); it != ed; ++it) {
	  anElectron.setIsolation(it->first, it->second);
        }
      }

      for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
        anElectron.setIsoDeposit(isoDepositLabels_[j].first, (*deposits[j])[elecsRef]);
      }

      // add electron ID info
      if (addElecID_) {
        for (size_t i = 0; i < elecIDSrcs_.size(); ++i) {
	  ids[i].second = (*idhandles[i])[elecsRef];
        }
        anElectron.setElectronIDs(ids);
      }


      if ( useUserData_ ) {
	userDataHelper_.add( anElectron, iEvent, iSetup );
      }


      double ip3d = -999; //for mva variable

      // embed high level selection
      if ( embedHighLevelSelection_ ) {
	// get the global track
	reco::GsfTrackRef track = itElectron->gsfTrack();

	// Make sure the collection it points to is there
	if ( track.isNonnull() && track.isAvailable() ) {

	  reco::TransientTrack tt = trackBuilder->build(track);
	  embedHighLevel( anElectron,
			  track,
			  tt,
			  primaryVertex,
			  primaryVertexIsValid,
			  beamSpot,
			  beamSpotIsValid );

          std::pair<bool,Measurement1D> ip3dpv = IPTools::absoluteImpactParameter3D(tt, primaryVertex);
          ip3d = ip3dpv.second.value(); // for mva variable

	  if ( !usePV_ ) {
	    double corr_d0 = track->dxy( beamPoint );
	    anElectron.setDB( corr_d0, -1.0 );
	  } else {
            std::pair<bool,Measurement1D> result = IPTools::absoluteTransverseImpactParameter(tt, primaryVertex);
            double d0_corr = result.second.value();
            double d0_err = result.second.error();
	    anElectron.setDB( d0_corr, d0_err );
	  }
	}
      }

      // add mva variables
      double r9 = lazyTools.e3x3( *( itElectron->superCluster()->seed())) / itElectron->superCluster()->rawEnergy() ;
      double sigmaIphiIphi;
      double sigmaIetaIphi;
      std::vector<float> vCov = lazyTools.localCovariances(*( itElectron->superCluster()->seed()));
      if( !edm::isNotFinite(vCov[2])) sigmaIphiIphi = sqrt(vCov[2]);
      else sigmaIphiIphi = 0;
      sigmaIetaIphi = vCov[1];
      anElectron.setMvaVariables( r9, sigmaIphiIphi, sigmaIetaIphi, ip3d);

      // get list of EcalDetId within 5x5 around the seed 
      bool barrel= itElectron->isEB();
	
      DetId seed=lazyTools.getMaximum(*(itElectron->superCluster()->seed())).first;
      std::vector<DetId> selectedCells = (barrel) ? ecalTopology_->getSubdetectorTopology(DetId::Ecal,EcalBarrel)->getWindow(seed,5,5):
	ecalTopology_->getSubdetectorTopology(DetId::Ecal,EcalEndcap)->getWindow(seed,5,5);


      // Do it for all basic clusters in 5x5
      reco::CaloCluster_iterator itscl = itElectron->superCluster()->clustersBegin();
      reco::CaloCluster_iterator itsclE = itElectron->superCluster()->clustersEnd();
      std::vector<DetId> cellsIn5x5;
      for ( ; itscl!= itsclE ; ++ itscl) {
	DetId seed=lazyTools.getMaximum(*(*itscl)).first;
	bool bcbarrel = seed.subdetId()==EcalBarrel; 
	std::vector<DetId> cellsToAdd = (bcbarrel) ? ecalTopology_->getSubdetectorTopology(DetId::Ecal,EcalBarrel)->getWindow(seed,5,5):
	  ecalTopology_->getSubdetectorTopology(DetId::Ecal,EcalEndcap)->getWindow(seed,5,5);
	cellsIn5x5.insert(cellsIn5x5.end(),cellsToAdd.begin(), cellsToAdd.end());

      }
      // Add to the list of selectedCells checking that there is no duplicate 
      unsigned nCellsIn5x5 = cellsIn5x5.size() ;

      for(unsigned i=0; i< nCellsIn5x5 ; ++i ) {
	std::vector<DetId>::const_iterator itcheck = find(selectedCells.begin(), selectedCells.end(),cellsIn5x5[i]);
	if (itcheck == selectedCells.end())
	  selectedCells.push_back(cellsIn5x5[i]);
      }

      // Add all RecHits of the SC if not already present
      std::vector< std::pair<DetId, float> >::const_iterator it=itElectron->superCluster()->hitsAndFractions().begin();
      std::vector< std::pair<DetId, float> >::const_iterator itend=itElectron->superCluster()->hitsAndFractions().end();
      for( ; it!=itend ; ++it) {
	DetId id=it->first;
	// check if already saved
	std::vector<DetId>::const_iterator itcheck = find(selectedCells.begin(),selectedCells.end(),id);
	if ( itcheck == selectedCells.end()) {
	  selectedCells.push_back(id);
	}
      }
      // Retrieve the corresponding RecHits

      edm::Handle< EcalRecHitCollection > rechitsH ;
      if(barrel)
	iEvent.getByLabel(reducedBarrelRecHitCollection_,rechitsH);
      else
	iEvent.getByLabel(reducedEndcapRecHitCollection_,rechitsH);

      EcalRecHitCollection selectedRecHits;
      const EcalRecHitCollection *recHits = rechitsH.product();

      unsigned nSelectedCells = selectedCells.size();
      for (unsigned icell = 0 ; icell < nSelectedCells ; ++icell) {
        EcalRecHitCollection::const_iterator  it = recHits->find( selectedCells[icell] );
	if ( it != recHits->end() ) {
	  selectedRecHits.push_back(*it);
	}
      }
      selectedRecHits.sort();
      if (embedRecHits_) anElectron.embedRecHits(& selectedRecHits);

      // set conversion veto selection
      bool passconversionveto = false;
      if( hConversions.isValid()){
        // this is recommended method
        passconversionveto = !ConversionTools::hasMatchedConversion( *itElectron, hConversions, beamSpotHandle->position());
      }else{
        // use missing hits without vertex fit method
        passconversionveto = itElectron->gsfTrack()->trackerExpectedHitsInner().numberOfLostHits() < 1;
      }
      anElectron.setPassConversionVeto( passconversionveto );

      // add sel to selected
      fillElectron( anElectron, elecsRef,elecBaseRef,
		    genMatches, deposits, pfId, isolationValues, isolationValuesNoPFId);
      patElectrons->push_back(anElectron);
    }
  }

  // sort electrons in pt
  std::sort(patElectrons->begin(), patElectrons->end(), pTComparator_);

  // add the electrons to the event output
  std::auto_ptr<std::vector<Electron> > ptr(patElectrons);
  iEvent.put(ptr);

  // clean up
  if (isolator_.enabled()) isolator_.endEvent();

}

void PATElectronProducer::fillElectron(Electron& anElectron,
				       const edm::RefToBase<reco::GsfElectron>& elecRef,
				       const reco::CandidateBaseRef& baseRef,
				       const GenAssociations& genMatches,
				       const IsoDepositMaps& deposits,
                                       const bool pfId,
				       const IsolationValueMaps& isolationValues,
				       const IsolationValueMaps& isolationValuesNoPFId
				       ) const {

  //COLIN: might want to use the PFCandidate 4-mom. Which one is in use now?
  //   if (useParticleFlow_)
  //     aMuon.setP4( aMuon.pfCandidateRef()->p4() );

  //COLIN:
  //In the embedding case, the reference cannot be used to look into a value map.
  //therefore, one has to had the PFCandidateRef to this function, which becomes a bit
  //too much specific.

  // in fact, this function needs a baseref or ptr for genmatch
  // and a baseref or ptr for isodeposits and isolationvalues.
  // baseref is not needed
  // the ptrForIsolation and ptrForMatching should be defined upstream.

  // is the concrete elecRef needed for the efficiency loader? what is this loader?
  // how can we make it compatible with the particle flow electrons?

  if (embedGsfElectronCore_) anElectron.embedGsfElectronCore();
  if (embedGsfTrack_) anElectron.embedGsfTrack();
  if (embedSuperCluster_) anElectron.embedSuperCluster();
  if (embedPflowSuperCluster_) anElectron.embedPflowSuperCluster();
  if (embedSeedCluster_) anElectron.embedSeedCluster();
  if (embedBasicClusters_) anElectron.embedBasicClusters();
  if (embedPreshowerClusters_) anElectron.embedPreshowerClusters();
  if (embedPflowBasicClusters_ ) anElectron.embedPflowBasicClusters();
  if (embedPflowPreshowerClusters_ ) anElectron.embedPflowPreshowerClusters();
  if (embedTrack_) anElectron.embedTrack();

  // store the match to the generated final state muons
  if (addGenMatch_) {
    for(size_t i = 0, n = genMatches.size(); i < n; ++i) {
      if(useParticleFlow_) {
	reco::GenParticleRef genElectron = (*genMatches[i])[anElectron.pfCandidateRef()];
	anElectron.addGenParticleRef(genElectron);
      }
      else {
	reco::GenParticleRef genElectron = (*genMatches[i])[elecRef];
	anElectron.addGenParticleRef(genElectron);
      }
    }
    if (embedGenMatch_) anElectron.embedGenParticle();
  }

  if (efficiencyLoader_.enabled()) {
    efficiencyLoader_.setEfficiencies( anElectron, elecRef );
  }

  if (resolutionLoader_.enabled()) {
    resolutionLoader_.setResolutions(anElectron);
  }

  for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
    if(useParticleFlow_) {

      reco::PFCandidateRef pfcandref =  anElectron.pfCandidateRef();
      assert(!pfcandref.isNull());
      reco::CandidatePtr source = pfcandref->sourceCandidatePtr(0);
      anElectron.setIsoDeposit(isoDepositLabels_[j].first,
			  (*deposits[j])[source]);
    }
    else
      anElectron.setIsoDeposit(isoDepositLabels_[j].first,
                          (*deposits[j])[elecRef]);
  }

  for (size_t j = 0; j<isolationValues.size(); ++j) {
    if(useParticleFlow_) {
      reco::CandidatePtr source = anElectron.pfCandidateRef()->sourceCandidatePtr(0);
      anElectron.setIsolation(isolationValueLabels_[j].first,
			 (*isolationValues[j])[source]);
    }
    else
      if(pfId){
        anElectron.setIsolation(isolationValueLabels_[j].first,(*isolationValues[j])[elecRef]);
      }
  }

  //for electrons not identified as PF electrons
  for (size_t j = 0; j<isolationValuesNoPFId.size(); ++j) {
    if( !pfId) {
      anElectron.setIsolation(isolationValueLabelsNoPFId_[j].first,(*isolationValuesNoPFId[j])[elecRef]);
    }
  }

}

void PATElectronProducer::fillElectron2( Electron& anElectron,
					 const reco::CandidatePtr& candPtrForIsolation,
					 const reco::CandidatePtr& candPtrForGenMatch,
					 const reco::CandidatePtr& candPtrForLoader,
					 const GenAssociations& genMatches,
					 const IsoDepositMaps& deposits,
					 const IsolationValueMaps& isolationValues) const {

  //COLIN/Florian: use the PFCandidate 4-mom.
  anElectron.setEcalDrivenMomentum(anElectron.p4()) ;
  anElectron.setP4( anElectron.pfCandidateRef()->p4() );


  // is the concrete elecRef needed for the efficiency loader? what is this loader?
  // how can we make it compatible with the particle flow electrons?

  if (embedGsfElectronCore_) anElectron.embedGsfElectronCore();
  if (embedGsfTrack_) anElectron.embedGsfTrack();
  if (embedSuperCluster_) anElectron.embedSuperCluster();
  if (embedPflowSuperCluster_ ) anElectron.embedPflowSuperCluster();
  if (embedSeedCluster_) anElectron.embedSeedCluster();
  if (embedBasicClusters_) anElectron.embedBasicClusters();
  if (embedPreshowerClusters_) anElectron.embedPreshowerClusters();
  if (embedPflowBasicClusters_ ) anElectron.embedPflowBasicClusters();
  if (embedPflowPreshowerClusters_ ) anElectron.embedPflowPreshowerClusters();
  if (embedTrack_) anElectron.embedTrack();

  // store the match to the generated final state muons

  if (addGenMatch_) {
    for(size_t i = 0, n = genMatches.size(); i < n; ++i) {
      reco::GenParticleRef genElectron = (*genMatches[i])[candPtrForGenMatch];
      anElectron.addGenParticleRef(genElectron);
    }
    if (embedGenMatch_) anElectron.embedGenParticle();
  }

  //COLIN what's this? does it have to be GsfElectron specific?
  if (efficiencyLoader_.enabled()) {
    efficiencyLoader_.setEfficiencies( anElectron, candPtrForLoader );
  }

  if (resolutionLoader_.enabled()) {
    resolutionLoader_.setResolutions(anElectron);
  }

  for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
    if( isoDepositLabels_[j].first==pat::TrackIso ||
	isoDepositLabels_[j].first==pat::EcalIso ||
	isoDepositLabels_[j].first==pat::HcalIso ||
	deposits[j]->contains(candPtrForGenMatch.id())) {
      anElectron.setIsoDeposit(isoDepositLabels_[j].first,
 			       (*deposits[j])[candPtrForGenMatch]);
    }
    else if (deposits[j]->contains(candPtrForIsolation.id())) {
      anElectron.setIsoDeposit(isoDepositLabels_[j].first,
 			       (*deposits[j])[candPtrForIsolation]);
    }
    else {
      anElectron.setIsoDeposit(isoDepositLabels_[j].first,
			       (*deposits[j])[candPtrForIsolation->sourceCandidatePtr(0)]);
    }
  }

  for (size_t j = 0; j<isolationValues.size(); ++j) {
    if( isolationValueLabels_[j].first==pat::TrackIso ||
	isolationValueLabels_[j].first==pat::EcalIso ||
	isolationValueLabels_[j].first==pat::HcalIso ||
	isolationValues[j]->contains(candPtrForGenMatch.id())) {
      anElectron.setIsolation(isolationValueLabels_[j].first,
 			      (*isolationValues[j])[candPtrForGenMatch]);
    }
    else if (isolationValues[j]->contains(candPtrForIsolation.id())) {
      anElectron.setIsolation(isolationValueLabels_[j].first,
 			      (*isolationValues[j])[candPtrForIsolation]);
    }
    else {
      anElectron.setIsolation(isolationValueLabels_[j].first,
			      (*isolationValues[j])[candPtrForIsolation->sourceCandidatePtr(0)]);
    }
  }
}


// ParameterSet description for module
void PATElectronProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription iDesc;
  iDesc.setComment("PAT electron producer module");

  // input source
  iDesc.add<edm::InputTag>("pfCandidateMap", edm::InputTag("no default"))->setComment("input collection");
  iDesc.add<edm::InputTag>("electronSource", edm::InputTag("no default"))->setComment("input collection");

  // embedding
  iDesc.add<bool>("embedGsfElectronCore", true)->setComment("embed external gsf electron core");
  iDesc.add<bool>("embedGsfTrack", true)->setComment("embed external gsf track");
  iDesc.add<bool>("embedSuperCluster", true)->setComment("embed external super cluster");
  iDesc.add<bool>("embedPflowSuperCluster", true)->setComment("embed external super cluster");
  iDesc.add<bool>("embedSeedCluster", true)->setComment("embed external seed cluster");
  iDesc.add<bool>("embedBasicClusters", true)->setComment("embed external basic clusters");
  iDesc.add<bool>("embedPreshowerClusters", true)->setComment("embed external preshower clusters");
  iDesc.add<bool>("embedPflowBasicClusters", true)->setComment("embed external pflow basic clusters");
  iDesc.add<bool>("embedPflowPreshowerClusters", true)->setComment("embed external pflow preshower clusters");
  iDesc.add<bool>("embedTrack", false)->setComment("embed external track");
  iDesc.add<bool>("embedRecHits", true)->setComment("embed external RecHits");

  // pf specific parameters
  iDesc.add<edm::InputTag>("pfElectronSource", edm::InputTag("pfElectrons"))->setComment("particle flow input collection");
  iDesc.add<bool>("useParticleFlow", false)->setComment("whether to use particle flow or not");
  iDesc.add<bool>("embedPFCandidate", false)->setComment("embed external particle flow object");

  // MC matching configurables
  iDesc.add<bool>("addGenMatch", true)->setComment("add MC matching");
  iDesc.add<bool>("embedGenMatch", false)->setComment("embed MC matched MC information");
  std::vector<edm::InputTag> emptySourceVector;
  iDesc.addNode( edm::ParameterDescription<edm::InputTag>("genParticleMatch", edm::InputTag(), true) xor
                 edm::ParameterDescription<std::vector<edm::InputTag> >("genParticleMatch", emptySourceVector, true)
		 )->setComment("input with MC match information");

  // electron ID configurables
  iDesc.add<bool>("addElectronID",true)->setComment("add electron ID variables");
  edm::ParameterSetDescription electronIDSourcesPSet;
  electronIDSourcesPSet.setAllowAnything();
  iDesc.addNode( edm::ParameterDescription<edm::InputTag>("electronIDSource", edm::InputTag(), true) xor
                 edm::ParameterDescription<edm::ParameterSetDescription>("electronIDSources", electronIDSourcesPSet, true)
                 )->setComment("input with electron ID variables");


  // IsoDeposit configurables
  edm::ParameterSetDescription isoDepositsPSet;
  isoDepositsPSet.addOptional<edm::InputTag>("tracker");
  isoDepositsPSet.addOptional<edm::InputTag>("ecal");
  isoDepositsPSet.addOptional<edm::InputTag>("hcal");
  isoDepositsPSet.addOptional<edm::InputTag>("pfAllParticles");
  isoDepositsPSet.addOptional<edm::InputTag>("pfChargedHadrons");
  isoDepositsPSet.addOptional<edm::InputTag>("pfChargedAll");
  isoDepositsPSet.addOptional<edm::InputTag>("pfPUChargedHadrons");
  isoDepositsPSet.addOptional<edm::InputTag>("pfNeutralHadrons");
  isoDepositsPSet.addOptional<edm::InputTag>("pfPhotons");
  isoDepositsPSet.addOptional<std::vector<edm::InputTag> >("user");
  iDesc.addOptional("isoDeposits", isoDepositsPSet);

  // isolation values configurables
  edm::ParameterSetDescription isolationValuesPSet;
  isolationValuesPSet.addOptional<edm::InputTag>("tracker");
  isolationValuesPSet.addOptional<edm::InputTag>("ecal");
  isolationValuesPSet.addOptional<edm::InputTag>("hcal");
  isolationValuesPSet.addOptional<edm::InputTag>("pfAllParticles");
  isolationValuesPSet.addOptional<edm::InputTag>("pfChargedHadrons");
  isolationValuesPSet.addOptional<edm::InputTag>("pfChargedAll");
  isolationValuesPSet.addOptional<edm::InputTag>("pfPUChargedHadrons");
  isolationValuesPSet.addOptional<edm::InputTag>("pfNeutralHadrons");
  isolationValuesPSet.addOptional<edm::InputTag>("pfPhotons");
  isolationValuesPSet.addOptional<std::vector<edm::InputTag> >("user");
  iDesc.addOptional("isolationValues", isolationValuesPSet);

  // isolation values configurables
  edm::ParameterSetDescription isolationValuesNoPFIdPSet;
  isolationValuesNoPFIdPSet.addOptional<edm::InputTag>("tracker");
  isolationValuesNoPFIdPSet.addOptional<edm::InputTag>("ecal");
  isolationValuesNoPFIdPSet.addOptional<edm::InputTag>("hcal");
  isolationValuesNoPFIdPSet.addOptional<edm::InputTag>("pfAllParticles");
  isolationValuesNoPFIdPSet.addOptional<edm::InputTag>("pfChargedHadrons");
  isolationValuesNoPFIdPSet.addOptional<edm::InputTag>("pfChargedAll");
  isolationValuesNoPFIdPSet.addOptional<edm::InputTag>("pfPUChargedHadrons");
  isolationValuesNoPFIdPSet.addOptional<edm::InputTag>("pfNeutralHadrons");
  isolationValuesNoPFIdPSet.addOptional<edm::InputTag>("pfPhotons");
  isolationValuesNoPFIdPSet.addOptional<std::vector<edm::InputTag> >("user");
  iDesc.addOptional("isolationValuesNoPFId", isolationValuesNoPFIdPSet);

  // Efficiency configurables
  edm::ParameterSetDescription efficienciesPSet;
  efficienciesPSet.setAllowAnything(); // TODO: the pat helper needs to implement a description.
  iDesc.add("efficiencies", efficienciesPSet);
  iDesc.add<bool>("addEfficiencies", false);

  // Check to see if the user wants to add user data
  edm::ParameterSetDescription userDataPSet;
  PATUserDataHelper<Electron>::fillDescription(userDataPSet);
  iDesc.addOptional("userData", userDataPSet);

  // electron shapes
  iDesc.add<bool>("addElectronShapes", true);
  iDesc.add<edm::InputTag>("reducedBarrelRecHitCollection", edm::InputTag("reducedEcalRecHitsEB"));
  iDesc.add<edm::InputTag>("reducedEndcapRecHitCollection", edm::InputTag("reducedEcalRecHitsEE"));

  edm::ParameterSetDescription isolationPSet;
  isolationPSet.setAllowAnything(); // TODO: the pat helper needs to implement a description.
  iDesc.add("userIsolation", isolationPSet);

  // Resolution configurables
  pat::helper::KinResolutionsLoader::fillDescription(iDesc);

  iDesc.add<bool>("embedHighLevelSelection", true)->setComment("embed high level selection");
  edm::ParameterSetDescription highLevelPSet;
  highLevelPSet.setAllowAnything();
  iDesc.addNode( edm::ParameterDescription<edm::InputTag>("beamLineSrc", edm::InputTag(), true)
                 )->setComment("input with high level selection");
  iDesc.addNode( edm::ParameterDescription<edm::InputTag>("pvSrc", edm::InputTag(), true)
                 )->setComment("input with high level selection");
  iDesc.addNode( edm::ParameterDescription<bool>("usePV", bool(), true)
                 )->setComment("input with high level selection, use primary vertex (true) or beam line (false)");

  descriptions.add("PATElectronProducer", iDesc);

}



void PATElectronProducer::readIsolationLabels( const edm::ParameterSet & iConfig,
					       const char* psetName,
					       IsolationLabels& labels) {

  labels.clear();

  if (iConfig.exists( psetName )) {
    edm::ParameterSet depconf
      = iConfig.getParameter<edm::ParameterSet>(psetName);

    if (depconf.exists("tracker")) labels.push_back(std::make_pair(pat::TrackIso, depconf.getParameter<edm::InputTag>("tracker")));
    if (depconf.exists("ecal"))    labels.push_back(std::make_pair(pat::EcalIso, depconf.getParameter<edm::InputTag>("ecal")));
    if (depconf.exists("hcal"))    labels.push_back(std::make_pair(pat::HcalIso, depconf.getParameter<edm::InputTag>("hcal")));
    if (depconf.exists("pfAllParticles"))  {
      labels.push_back(std::make_pair(pat::PfAllParticleIso, depconf.getParameter<edm::InputTag>("pfAllParticles")));
    }
    if (depconf.exists("pfChargedHadrons"))  {
      labels.push_back(std::make_pair(pat::PfChargedHadronIso, depconf.getParameter<edm::InputTag>("pfChargedHadrons")));
    }
    if (depconf.exists("pfChargedAll"))  {
      labels.push_back(std::make_pair(pat::PfChargedAllIso, depconf.getParameter<edm::InputTag>("pfChargedAll")));
    }
    if (depconf.exists("pfPUChargedHadrons"))  {
      labels.push_back(std::make_pair(pat::PfPUChargedHadronIso, depconf.getParameter<edm::InputTag>("pfPUChargedHadrons")));
    }
    if (depconf.exists("pfNeutralHadrons"))  {
      labels.push_back(std::make_pair(pat::PfNeutralHadronIso, depconf.getParameter<edm::InputTag>("pfNeutralHadrons")));
    }
    if (depconf.exists("pfPhotons")) {
      labels.push_back(std::make_pair(pat::PfGammaIso, depconf.getParameter<edm::InputTag>("pfPhotons")));
    }
    if (depconf.exists("user")) {
      std::vector<edm::InputTag> userdeps = depconf.getParameter<std::vector<edm::InputTag> >("user");
      std::vector<edm::InputTag>::const_iterator it = userdeps.begin(), ed = userdeps.end();
      int key = UserBaseIso;
      for ( ; it != ed; ++it, ++key) {
	labels.push_back(std::make_pair(IsolationKeys(key), *it));
      }
    }
  }


}


// embed various impact parameters with errors
// embed high level selection
void PATElectronProducer::embedHighLevel( pat::Electron & anElectron,
					  reco::GsfTrackRef track,
					  reco::TransientTrack & tt,
					  reco::Vertex & primaryVertex,
					  bool primaryVertexIsValid,
					  reco::BeamSpot & beamspot,
					  bool beamspotIsValid
					  )
{
  // Correct to PV

  // PV2D
  std::pair<bool,Measurement1D> result =
    IPTools::signedTransverseImpactParameter(tt,
					     GlobalVector(track->px(),
							  track->py(),
							  track->pz()),
					     primaryVertex);
  double d0_corr = result.second.value();
  double d0_err = primaryVertexIsValid ? result.second.error() : -1.0;
  anElectron.setDB( d0_corr, d0_err, pat::Electron::PV2D);


  // PV3D
  result =
    IPTools::signedImpactParameter3D(tt,
				     GlobalVector(track->px(),
						  track->py(),
						  track->pz()),
				     primaryVertex);
  d0_corr = result.second.value();
  d0_err = primaryVertexIsValid ? result.second.error() : -1.0;
  anElectron.setDB( d0_corr, d0_err, pat::Electron::PV3D);


  // Correct to beam spot
  // make a fake vertex out of beam spot
  reco::Vertex vBeamspot(beamspot.position(), beamspot.covariance3D());

  // BS2D
  result =
    IPTools::signedTransverseImpactParameter(tt,
					     GlobalVector(track->px(),
							  track->py(),
							  track->pz()),
					     vBeamspot);
  d0_corr = result.second.value();
  d0_err = beamspotIsValid ? result.second.error() : -1.0;
  anElectron.setDB( d0_corr, d0_err, pat::Electron::BS2D);

  // BS3D
  result =
    IPTools::signedImpactParameter3D(tt,
				     GlobalVector(track->px(),
						  track->py(),
						  track->pz()),
				     vBeamspot);
  d0_corr = result.second.value();
  d0_err = beamspotIsValid ? result.second.error() : -1.0;
  anElectron.setDB( d0_corr, d0_err, pat::Electron::BS3D);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATElectronProducer);
