// -*- C++ -*-
//
// Package:    MuonIdentification
// Class:      MuonIdProducer
//
//
// Original Author:  Dmytro Kovalskyi
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
//#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/MuonCocktails.h"
#include "DataFormats/MuonReco/interface/MuonTime.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtraMap.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"

#include <boost/regex.hpp>
#include "RecoMuon/MuonIdentification/plugins/MuonIdProducer.h"

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractorFactory.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include <algorithm>

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include "RecoMuon/MuonIdentification/interface/MuonMesh.h"


#include "RecoMuon/MuonIdentification/interface/MuonKinkFinder.h"

MuonIdProducer::MuonIdProducer(const edm::ParameterSet& iConfig):
muIsoExtractorCalo_(0),muIsoExtractorTrack_(0),muIsoExtractorJet_(0)
{
   produces<reco::MuonCollection>();
   produces<reco::CaloMuonCollection>();
   produces<reco::MuonTimeExtraMap>("combined");
   produces<reco::MuonTimeExtraMap>("dt");
   produces<reco::MuonTimeExtraMap>("csc");

   minPt_                   = iConfig.getParameter<double>("minPt");
   minP_                    = iConfig.getParameter<double>("minP");
   minPCaloMuon_            = iConfig.getParameter<double>("minPCaloMuon");
   minNumberOfMatches_      = iConfig.getParameter<int>("minNumberOfMatches");
   addExtraSoftMuons_       = iConfig.getParameter<bool>("addExtraSoftMuons");
   maxAbsEta_               = iConfig.getParameter<double>("maxAbsEta");
   maxAbsDx_                = iConfig.getParameter<double>("maxAbsDx");
   maxAbsPullX_             = iConfig.getParameter<double>("maxAbsPullX");
   maxAbsDy_                = iConfig.getParameter<double>("maxAbsDy");
   maxAbsPullY_             = iConfig.getParameter<double>("maxAbsPullY");
   fillCaloCompatibility_   = iConfig.getParameter<bool>("fillCaloCompatibility");
   fillEnergy_              = iConfig.getParameter<bool>("fillEnergy");
   fillMatching_            = iConfig.getParameter<bool>("fillMatching");
   fillIsolation_           = iConfig.getParameter<bool>("fillIsolation");
   writeIsoDeposits_        = iConfig.getParameter<bool>("writeIsoDeposits");
   fillGlobalTrackQuality_  = iConfig.getParameter<bool>("fillGlobalTrackQuality");
   fillGlobalTrackRefits_   = iConfig.getParameter<bool>("fillGlobalTrackRefits");
   //SK: (maybe temporary) run it only if the global is also run
   fillTrackerKink_         = false;
   if (fillGlobalTrackQuality_)  fillTrackerKink_ =  iConfig.getParameter<bool>("fillTrackerKink");

   ptThresholdToFillCandidateP4WithGlobalFit_    = iConfig.getParameter<double>("ptThresholdToFillCandidateP4WithGlobalFit");
   sigmaThresholdToFillCandidateP4WithGlobalFit_ = iConfig.getParameter<double>("sigmaThresholdToFillCandidateP4WithGlobalFit");
   caloCut_ = iConfig.getParameter<double>("minCaloCompatibility"); //CaloMuons
   arbClean_ = iConfig.getParameter<bool>("runArbitrationCleaner"); // muon mesh

   // Load TrackDetectorAssociator parameters
   edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
   edm::ConsumesCollector iC = consumesCollector();
   parameters_.loadParameters( parameters, iC );

   // Load parameters for the TimingFiller
   edm::ParameterSet timingParameters = iConfig.getParameter<edm::ParameterSet>("TimingFillerParameters");
   theTimingFiller_ = new MuonTimingFiller(timingParameters,consumesCollector());
   

   if (fillCaloCompatibility_){
      // Load MuonCaloCompatibility parameters
      parameters = iConfig.getParameter<edm::ParameterSet>("MuonCaloCompatibility");
      muonCaloCompatibility_.configure( parameters );
   }

   if (fillIsolation_){
      // Load MuIsoExtractor parameters
      edm::ParameterSet caloExtractorPSet = iConfig.getParameter<edm::ParameterSet>("CaloExtractorPSet");
      std::string caloExtractorName = caloExtractorPSet.getParameter<std::string>("ComponentName");
      muIsoExtractorCalo_ = IsoDepositExtractorFactory::get()->create( caloExtractorName, caloExtractorPSet,consumesCollector());

      edm::ParameterSet trackExtractorPSet = iConfig.getParameter<edm::ParameterSet>("TrackExtractorPSet");
      std::string trackExtractorName = trackExtractorPSet.getParameter<std::string>("ComponentName");
      muIsoExtractorTrack_ = IsoDepositExtractorFactory::get()->create( trackExtractorName, trackExtractorPSet,consumesCollector());

      edm::ParameterSet jetExtractorPSet = iConfig.getParameter<edm::ParameterSet>("JetExtractorPSet");
      std::string jetExtractorName = jetExtractorPSet.getParameter<std::string>("ComponentName");
      muIsoExtractorJet_ = IsoDepositExtractorFactory::get()->create( jetExtractorName, jetExtractorPSet,consumesCollector());
   }
   if (fillIsolation_ && writeIsoDeposits_){
     trackDepositName_ = iConfig.getParameter<std::string>("trackDepositName");
     produces<reco::IsoDepositMap>(trackDepositName_);
     ecalDepositName_ = iConfig.getParameter<std::string>("ecalDepositName");
     produces<reco::IsoDepositMap>(ecalDepositName_);
     hcalDepositName_ = iConfig.getParameter<std::string>("hcalDepositName");
     produces<reco::IsoDepositMap>(hcalDepositName_);
     hoDepositName_ = iConfig.getParameter<std::string>("hoDepositName");
     produces<reco::IsoDepositMap>(hoDepositName_);
     jetDepositName_ = iConfig.getParameter<std::string>("jetDepositName");
     produces<reco::IsoDepositMap>(jetDepositName_);
   }

   inputCollectionLabels_ = iConfig.getParameter<std::vector<edm::InputTag> >("inputCollectionLabels");
   inputCollectionTypes_  = iConfig.getParameter<std::vector<std::string> >("inputCollectionTypes");
   if (inputCollectionLabels_.size() != inputCollectionTypes_.size())
     throw cms::Exception("ConfigurationError") << "Number of input collection labels is different from number of types. " <<
     "For each collection label there should be exactly one collection type specified.";
   if (inputCollectionLabels_.size()>7 ||inputCollectionLabels_.empty())
     throw cms::Exception("ConfigurationError") << "Number of input collections should be from 1 to 7.";

   debugWithTruthMatching_    = iConfig.getParameter<bool>("debugWithTruthMatching");
   if (debugWithTruthMatching_) edm::LogWarning("MuonIdentification")
     << "========================================================================\n"
     << "Debugging mode with truth matching is turned on!!! Make sure you understand what you are doing!\n"
     << "========================================================================\n";
   if (fillGlobalTrackQuality_){
     globalTrackQualityInputTag_ = iConfig.getParameter<edm::InputTag>("globalTrackQualityInputTag");
   }

   if (fillTrackerKink_) {
     trackerKinkFinder_.reset(new MuonKinkFinder(iConfig.getParameter<edm::ParameterSet>("TrackerKinkFinderParameters")));
   }

   //create mesh holder
   meshAlgo_ = new MuonMesh(iConfig.getParameter<edm::ParameterSet>("arbitrationCleanerOptions"));


   edm::InputTag rpcHitTag("rpcRecHits");
   rpcHitToken_ = consumes<RPCRecHitCollection>(rpcHitTag);
   glbQualToken_ = consumes<edm::ValueMap<reco::MuonQuality> >(globalTrackQualityInputTag_);
   

   //Consumes... UGH
   for ( unsigned int i = 0; i < inputCollectionLabels_.size(); ++i ) {
      if ( inputCollectionTypes_[i] == "inner tracks" ) {
	innerTrackCollectionToken_ = consumes<reco::TrackCollection>(inputCollectionLabels_.at(i));
	 continue;
      }
      if ( inputCollectionTypes_[i] == "outer tracks" ) {
	outerTrackCollectionToken_ = consumes<reco::TrackCollection>(inputCollectionLabels_.at(i));
	 continue;
      }
      if ( inputCollectionTypes_[i] == "links" ) {
	linkCollectionToken_ = consumes<reco::MuonTrackLinksCollection>(inputCollectionLabels_.at(i));
	 continue;
      }
      if ( inputCollectionTypes_[i] == "muons" ) {
	muonCollectionToken_ = consumes<reco::MuonCollection>(inputCollectionLabels_.at(i));
	 continue;
      }
      if ( fillGlobalTrackRefits_  && inputCollectionTypes_[i] == "tev firstHit" ) {
	tpfmsCollectionToken_ = consumes<reco::TrackToTrackMap>(inputCollectionLabels_.at(i));
	 continue;
      }

      if ( fillGlobalTrackRefits_  && inputCollectionTypes_[i] == "tev picky" ) {
	pickyCollectionToken_ = consumes<reco::TrackToTrackMap>(inputCollectionLabels_.at(i));
	 continue;
      }

      if ( fillGlobalTrackRefits_  && inputCollectionTypes_[i] == "tev dyt" ) {
	dytCollectionToken_ = consumes<reco::TrackToTrackMap>(inputCollectionLabels_.at(i));
	 continue;
      }
      throw cms::Exception("FatalError") << "Unknown input collection type: " << inputCollectionTypes_[i];
   }


}



MuonIdProducer::~MuonIdProducer()
{
   if (muIsoExtractorCalo_) delete muIsoExtractorCalo_;
   if (muIsoExtractorTrack_) delete muIsoExtractorTrack_;
   if (muIsoExtractorJet_) delete muIsoExtractorJet_;
   if (theTimingFiller_) delete theTimingFiller_;
   if (meshAlgo_) delete meshAlgo_;
   // TimingReport::current()->dump(std::cout);
}

void MuonIdProducer::init(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   innerTrackCollectionHandle_.clear();
   outerTrackCollectionHandle_.clear();
   linkCollectionHandle_.clear();
   muonCollectionHandle_.clear();

   tpfmsCollectionHandle_.clear();
   pickyCollectionHandle_.clear();
   dytCollectionHandle_.clear();


   edm::ESHandle<Propagator> propagator;
   iSetup.get<TrackingComponentsRecord>().get("SteppingHelixPropagatorAny", propagator);
   trackAssociator_.setPropagator(propagator.product());

   if (fillTrackerKink_) trackerKinkFinder_->init(iSetup);

   for ( unsigned int i = 0; i < inputCollectionLabels_.size(); ++i ) {
      if ( inputCollectionTypes_[i] == "inner tracks" ) {
	 iEvent.getByToken(innerTrackCollectionToken_, innerTrackCollectionHandle_);
	 if (! innerTrackCollectionHandle_.isValid()) 
	   throw cms::Exception("FatalError") << "Failed to get input track collection with label: " << inputCollectionLabels_[i];
	 LogTrace("MuonIdentification") << "Number of input inner tracks: " << innerTrackCollectionHandle_->size();
	 continue;
      }
      if ( inputCollectionTypes_[i] == "outer tracks" ) {
	 iEvent.getByToken(outerTrackCollectionToken_, outerTrackCollectionHandle_);
	 if (! outerTrackCollectionHandle_.isValid()) 
	   throw cms::Exception("FatalError") << "Failed to get input track collection with label: " << inputCollectionLabels_[i];
	 LogTrace("MuonIdentification") << "Number of input outer tracks: " << outerTrackCollectionHandle_->size();
	 continue;
      }
      if ( inputCollectionTypes_[i] == "links" ) {
	 iEvent.getByToken(linkCollectionToken_, linkCollectionHandle_);
	 if (! linkCollectionHandle_.isValid()) 
	   throw cms::Exception("FatalError") << "Failed to get input link collection with label: " << inputCollectionLabels_[i];
	 LogTrace("MuonIdentification") << "Number of input links: " << linkCollectionHandle_->size();
	 continue;
      }
      if ( inputCollectionTypes_[i] == "muons" ) {
	 iEvent.getByToken(muonCollectionToken_, muonCollectionHandle_);
	 if (! muonCollectionHandle_.isValid()) 
	   throw cms::Exception("FatalError") << "Failed to get input muon collection with label: " << inputCollectionLabels_[i];
	 LogTrace("MuonIdentification") << "Number of input muons: " << muonCollectionHandle_->size();
	 continue;
      }
      if ( fillGlobalTrackRefits_  && inputCollectionTypes_[i] == "tev firstHit" ) {
	 iEvent.getByToken(tpfmsCollectionToken_, tpfmsCollectionHandle_);
	 if (! tpfmsCollectionHandle_.isValid()) 
	   throw cms::Exception("FatalError") << "Failed to get input muon collection with label: " << inputCollectionLabels_[i];
	 LogTrace("MuonIdentification") << "Number of input muons: " << tpfmsCollectionHandle_->size();
	 continue;
      }

      if ( fillGlobalTrackRefits_  && inputCollectionTypes_[i] == "tev picky" ) {
	 iEvent.getByToken(pickyCollectionToken_, pickyCollectionHandle_);
	 if (! pickyCollectionHandle_.isValid()) 
	   throw cms::Exception("FatalError") << "Failed to get input muon collection with label: " << inputCollectionLabels_[i];
	 LogTrace("MuonIdentification") << "Number of input muons: " << pickyCollectionHandle_->size();
	 continue;
      }

      if ( fillGlobalTrackRefits_  && inputCollectionTypes_[i] == "tev dyt" ) {
	 iEvent.getByToken(dytCollectionToken_, dytCollectionHandle_);
	 if (! dytCollectionHandle_.isValid()) 
	   throw cms::Exception("FatalError") << "Failed to get input muon collection with label: " << inputCollectionLabels_[i];
	 LogTrace("MuonIdentification") << "Number of input muons: " << dytCollectionHandle_->size();
	 continue;
      }
      throw cms::Exception("FatalError") << "Unknown input collection type: " << inputCollectionTypes_[i];
   }
}

reco::Muon MuonIdProducer::makeMuon(edm::Event& iEvent, const edm::EventSetup& iSetup,
				     const reco::TrackRef& track, MuonIdProducer::TrackType type)
{
   LogTrace("MuonIdentification") << "Creating a muon from a track " << track.get()->pt() <<
     " Pt (GeV), eta: " << track.get()->eta();
   reco::Muon aMuon( makeMuon( *(track.get()) ) );

   aMuon.setMuonTrack(type,track);
   aMuon.setBestTrack(type);
   aMuon.setTunePBestTrack(type);

   return aMuon;
}

reco::CaloMuon MuonIdProducer::makeCaloMuon( const reco::Muon& muon )
{
   reco::CaloMuon aMuon;
   aMuon.setInnerTrack( muon.innerTrack() );

   if (muon.isEnergyValid()) aMuon.setCalEnergy( muon.calEnergy() );
   // get calo compatibility
   if (fillCaloCompatibility_) aMuon.setCaloCompatibility( muonCaloCompatibility_.evaluate(muon) );
   return aMuon;
}


reco::Muon MuonIdProducer::makeMuon( const reco::MuonTrackLinks& links )
{
   LogTrace("MuonIdentification") << "Creating a muon from a link to tracks object";

   reco::Muon aMuon;
   reco::Muon::MuonTrackTypePair chosenTrack;
   reco::TrackRef tpfmsRef;
   reco::TrackRef pickyRef;
   reco::TrackRef dytRef;
   bool useSigmaSwitch = false;

   if (tpfmsCollectionHandle_.isValid() && !tpfmsCollectionHandle_.failedToGet() &&
       pickyCollectionHandle_.isValid() && !pickyCollectionHandle_.failedToGet()) {

     tpfmsRef = muon::getTevRefitTrack(links.globalTrack(), *tpfmsCollectionHandle_);
     pickyRef = muon::getTevRefitTrack(links.globalTrack(), *pickyCollectionHandle_);
     dytRef = muon::getTevRefitTrack(links.globalTrack(), *dytCollectionHandle_);

     if (tpfmsRef.isNull() && pickyRef.isNull() && dytRef.isNull()){
       edm::LogWarning("MakeMuonWithTEV")<<"Failed to get  TEV refits, fall back to sigma switch.";
       useSigmaSwitch = true;
     }
   } else {
     useSigmaSwitch = true;
   }

   if (useSigmaSwitch){
     chosenTrack = muon::sigmaSwitch( links.globalTrack(), links.trackerTrack(),
				      sigmaThresholdToFillCandidateP4WithGlobalFit_,
				      ptThresholdToFillCandidateP4WithGlobalFit_);
   } else {
     chosenTrack = muon::tevOptimized( links.globalTrack(), links.trackerTrack(),
				       tpfmsRef, pickyRef, dytRef,
				       ptThresholdToFillCandidateP4WithGlobalFit_);
   }
   aMuon = makeMuon(*chosenTrack.first);
   aMuon.setInnerTrack( links.trackerTrack() );
   aMuon.setOuterTrack( links.standAloneTrack() );
   aMuon.setGlobalTrack( links.globalTrack() );
   aMuon.setBestTrack(chosenTrack.second);
   aMuon.setTunePBestTrack(chosenTrack.second);

   if(fillGlobalTrackRefits_){
     if (tpfmsCollectionHandle_.isValid() && !tpfmsCollectionHandle_.failedToGet()) {
       reco::TrackToTrackMap::const_iterator it = tpfmsCollectionHandle_->find(links.globalTrack());
       if (it != tpfmsCollectionHandle_->end()) aMuon.setMuonTrack(reco::Muon::TPFMS, (it->val));
     }
     if (pickyCollectionHandle_.isValid() && !pickyCollectionHandle_.failedToGet()) {
       reco::TrackToTrackMap::const_iterator it = pickyCollectionHandle_->find(links.globalTrack());
       if (it != pickyCollectionHandle_->end()) aMuon.setMuonTrack(reco::Muon::Picky, (it->val));
     }
     if (dytCollectionHandle_.isValid() && !dytCollectionHandle_.failedToGet()) {
       reco::TrackToTrackMap::const_iterator it = dytCollectionHandle_->find(links.globalTrack());
       if (it != dytCollectionHandle_->end()) aMuon.setMuonTrack(reco::Muon::DYT, (it->val));
     }
   }
   return aMuon;
}


bool MuonIdProducer::isGoodTrack( const reco::Track& track )
{
   // Pt and absolute momentum requirement
   if (track.pt() < minPt_ || (track.p() < minP_ && track.p() < minPCaloMuon_)){
      LogTrace("MuonIdentification") << "Skipped low momentum track (Pt,P): " << track.pt() <<
	", " << track.p() << " GeV";
      return false;
   }

   // Eta requirement
   if ( fabs(track.eta()) > maxAbsEta_ ){
      LogTrace("MuonIdentification") << "Skipped track with large pseudo rapidity (Eta: " << track.eta() << " )";
      return false;
   }
   return true;
}

unsigned int MuonIdProducer::chamberId( const DetId& id )
{
   switch ( id.det() ) {
    case DetId::Muon:
      switch ( id.subdetId() ) {
       case MuonSubdetId::DT:
	   {
	      DTChamberId detId(id.rawId());
	      return detId.rawId();
	   }
	 break;
       case MuonSubdetId::CSC:
	   {
	      CSCDetId detId(id.rawId());
	      return detId.chamberId().rawId();
	   }
	 break;
       default:
	 return 0;
      }
    default:
      return 0;
   }
   return 0;
}


int MuonIdProducer::overlap(const reco::Muon& muon, const reco::Track& track)
{
   int numberOfCommonDetIds = 0;
   if ( ! muon.isMatchesValid() ||
	track.extra().isNull() ||
	track.extra()->recHitsSize()==0 ) return numberOfCommonDetIds;
   const std::vector<reco::MuonChamberMatch>& matches( muon.matches() );
   for ( std::vector<reco::MuonChamberMatch>::const_iterator match = matches.begin();
	 match != matches.end(); ++match )
     {
	if ( match->segmentMatches.empty() ) continue;
	bool foundCommonDetId = false;

	for ( auto hit = track.extra()->recHitsBegin();
	      hit != track.extra()->recHitsEnd(); ++hit )
	  {
	     // LogTrace("MuonIdentification") << "hit DetId: " << std::hex << hit->get()->geographicalId().rawId() <<
	     //  "\t hit chamber DetId: " << getChamberId(hit->get()->geographicalId()) <<
	     //  "\t segment DetId: " << match->id.rawId() << std::dec;

	     if ( chamberId((*hit)->geographicalId()) == match->id.rawId() ) {
		foundCommonDetId = true;
		break;
	     }
	  }
	if ( foundCommonDetId ) {
	   numberOfCommonDetIds++;
	   break;
	}
     }
   return numberOfCommonDetIds;
}

void MuonIdProducer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  edm::ESHandle<CSCGeometry> geomHandle;
  iSetup.get<MuonGeometryRecord>().get(geomHandle);

  meshAlgo_->setCSCGeometry(geomHandle.product());

}

bool validateGlobalMuonPair( const reco::MuonTrackLinks& goodMuon,
			     const reco::MuonTrackLinks& badMuon )
{
  if ( std::min(goodMuon.globalTrack()->hitPattern().numberOfValidMuonHits(),
		 badMuon.globalTrack()->hitPattern().numberOfValidMuonHits()) > 10 ){
    if ( goodMuon.globalTrack()->normalizedChi2() >
	  badMuon.globalTrack()->normalizedChi2() )
      return false;
    else
      return true;
  }
  if ( goodMuon.globalTrack()->hitPattern().numberOfValidMuonHits() <
       badMuon.globalTrack()->hitPattern().numberOfValidMuonHits() ) return false;
  return true;
}

void MuonIdProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   std::auto_ptr<reco::MuonCollection> outputMuons(new reco::MuonCollection);
   std::auto_ptr<reco::CaloMuonCollection> caloMuons( new reco::CaloMuonCollection );

   init(iEvent, iSetup);

   std::auto_ptr<reco::MuonTimeExtraMap> muonTimeMap(new reco::MuonTimeExtraMap());
   reco::MuonTimeExtraMap::Filler filler(*muonTimeMap);
   std::auto_ptr<reco::MuonTimeExtraMap> muonTimeMapDT(new reco::MuonTimeExtraMap());
   reco::MuonTimeExtraMap::Filler fillerDT(*muonTimeMapDT);
   std::auto_ptr<reco::MuonTimeExtraMap> muonTimeMapCSC(new reco::MuonTimeExtraMap());
   reco::MuonTimeExtraMap::Filler fillerCSC(*muonTimeMapCSC);

   std::auto_ptr<reco::IsoDepositMap> trackDepMap(new reco::IsoDepositMap());
   reco::IsoDepositMap::Filler trackDepFiller(*trackDepMap);
   std::auto_ptr<reco::IsoDepositMap> ecalDepMap(new reco::IsoDepositMap());
   reco::IsoDepositMap::Filler ecalDepFiller(*ecalDepMap);
   std::auto_ptr<reco::IsoDepositMap> hcalDepMap(new reco::IsoDepositMap());
   reco::IsoDepositMap::Filler hcalDepFiller(*hcalDepMap);
   std::auto_ptr<reco::IsoDepositMap> hoDepMap(new reco::IsoDepositMap());
   reco::IsoDepositMap::Filler hoDepFiller(*hoDepMap);
   std::auto_ptr<reco::IsoDepositMap> jetDepMap(new reco::IsoDepositMap());
   reco::IsoDepositMap::Filler jetDepFiller(*jetDepMap);

   // loop over input collections

   // muons first - no cleaning, take as is.
   if ( muonCollectionHandle_.isValid() )
     for ( reco::MuonCollection::const_iterator muon = muonCollectionHandle_->begin();
	   muon !=  muonCollectionHandle_->end(); ++muon )
       outputMuons->push_back(*muon);

   // links second ( assume global muon type )
   if ( linkCollectionHandle_.isValid() ){
     std::vector<bool> goodmuons(linkCollectionHandle_->size(),true);
     if ( goodmuons.size()>1 ){
       // check for shared tracker tracks
       for ( unsigned int i=0; i<linkCollectionHandle_->size()-1; ++i ){
	 if (!checkLinks(&linkCollectionHandle_->at(i))) continue;
	 for ( unsigned int j=i+1; j<linkCollectionHandle_->size(); ++j ){
	   if (!checkLinks(&linkCollectionHandle_->at(j))) continue;
	   if ( linkCollectionHandle_->at(i).trackerTrack().isNonnull() &&
		linkCollectionHandle_->at(i).trackerTrack() ==
		linkCollectionHandle_->at(j).trackerTrack() )
	     {
	       // Tracker track is the essential part that dominates muon resolution
	       // so taking either muon is fine. All that is important is to preserve
	       // the muon identification information. If number of hits is small,
	       // keep the one with large number of hits, otherwise take the smalest chi2/ndof
	       if ( validateGlobalMuonPair(linkCollectionHandle_->at(i),linkCollectionHandle_->at(j)) )
		 goodmuons[j] = false;
	       else
		 goodmuons[i] = false;
	     }
	 }
       }
       // check for shared stand-alone muons.
       for ( unsigned int i=0; i<linkCollectionHandle_->size()-1; ++i ){
	 if ( !goodmuons[i] ) continue;
	 if (!checkLinks(&linkCollectionHandle_->at(i))) continue;
	 for ( unsigned int j=i+1; j<linkCollectionHandle_->size(); ++j ){
	   if ( !goodmuons[j] ) continue;
	   if (!checkLinks(&linkCollectionHandle_->at(j))) continue;
	   if ( linkCollectionHandle_->at(i).standAloneTrack().isNonnull() &&
		linkCollectionHandle_->at(i).standAloneTrack() ==
		linkCollectionHandle_->at(j).standAloneTrack() )
	     {
	       if ( validateGlobalMuonPair(linkCollectionHandle_->at(i),linkCollectionHandle_->at(j)) )
		 goodmuons[j] = false;
	       else
		 goodmuons[i] = false;
	     }
	 }
       }
     }
     for ( unsigned int i=0; i<linkCollectionHandle_->size(); ++i ){
       if ( !goodmuons[i] ) continue;
       const reco::MuonTrackLinks* links = &linkCollectionHandle_->at(i);
       if ( ! checkLinks(links))   continue;
       // check if this muon is already in the list
       bool newMuon = true;
       for ( reco::MuonCollection::const_iterator muon = outputMuons->begin();
	     muon !=  outputMuons->end(); ++muon )
	 if ( muon->track() == links->trackerTrack() &&
	      muon->standAloneMuon() == links->standAloneTrack() &&
	      muon->combinedMuon() == links->globalTrack() )
	   newMuon = false;
       if ( newMuon ) {
	 outputMuons->push_back( makeMuon( *links ) );
	 outputMuons->back().setType(reco::Muon::GlobalMuon | reco::Muon::StandAloneMuon);
       }
     }
   }

   // tracker and calo muons are next
   if ( innerTrackCollectionHandle_.isValid() ) {
      LogTrace("MuonIdentification") << "Creating tracker muons";
      for ( unsigned int i = 0; i < innerTrackCollectionHandle_->size(); ++i )
	{
	   const reco::Track& track = innerTrackCollectionHandle_->at(i);
	   if ( ! isGoodTrack( track ) ) continue;
	   bool splitTrack = false;
	   if ( track.extra().isAvailable() &&
		TrackDetectorAssociator::crossedIP( track ) ) splitTrack = true;
	   std::vector<TrackDetectorAssociator::Direction> directions;
	   if ( splitTrack ) {
	      directions.push_back(TrackDetectorAssociator::InsideOut);
	      directions.push_back(TrackDetectorAssociator::OutsideIn);
	   } else {
	      directions.push_back(TrackDetectorAssociator::Any);
	   }
	   for ( std::vector<TrackDetectorAssociator::Direction>::const_iterator direction = directions.begin();
		 direction != directions.end(); ++direction )
	     {
		// make muon
	       reco::Muon trackerMuon( makeMuon(iEvent, iSetup, reco::TrackRef( innerTrackCollectionHandle_, i ), reco::Muon::InnerTrack ) );
		fillMuonId(iEvent, iSetup, trackerMuon, *direction);

		if ( debugWithTruthMatching_ ) {
		   // add MC hits to a list of matched segments.
		   // Since it's debugging mode - code is slow
		   MuonIdTruthInfo::truthMatchMuon(iEvent, iSetup, trackerMuon);
		}

		// check if this muon is already in the list
		// have to check where muon hits are really located
		// to match properly
		bool newMuon = true;
		bool goodTrackerMuon = isGoodTrackerMuon( trackerMuon );
		bool goodRPCMuon = isGoodRPCMuon( trackerMuon );
		if ( goodTrackerMuon ) trackerMuon.setType( trackerMuon.type() | reco::Muon::TrackerMuon );
		if ( goodRPCMuon ) trackerMuon.setType( trackerMuon.type() | reco::Muon::RPCMuon );
		for ( reco::MuonCollection::iterator muon = outputMuons->begin();
		      muon !=  outputMuons->end(); ++muon )
		  {
		     if ( muon->innerTrack().get() == trackerMuon.innerTrack().get() &&
			  cos(phiOfMuonIneteractionRegion(*muon) -
			      phiOfMuonIneteractionRegion(trackerMuon)) > 0 )
		       {
			  newMuon = false;
			  muon->setMatches( trackerMuon.matches() );
			  if (trackerMuon.isTimeValid()) muon->setTime( trackerMuon.time() );
			  if (trackerMuon.isEnergyValid()) muon->setCalEnergy( trackerMuon.calEnergy() );
			  if (goodTrackerMuon) muon->setType( muon->type() | reco::Muon::TrackerMuon );
			  if (goodRPCMuon) muon->setType( muon->type() | reco::Muon::RPCMuon );
			  LogTrace("MuonIdentification") << "Found a corresponding global muon. Set energy, matches and move on";
			  break;
		       }
		  }
		if ( newMuon ) {
		   if ( goodTrackerMuon || goodRPCMuon ){
		      outputMuons->push_back( trackerMuon );
		   } else {
		      LogTrace("MuonIdentification") << "track failed minimal number of muon matches requirement";
		      const reco::CaloMuon& caloMuon = makeCaloMuon(trackerMuon);
		      if ( ! caloMuon.isCaloCompatibilityValid() || caloMuon.caloCompatibility() < caloCut_ || caloMuon.p() < minPCaloMuon_) continue;
		      caloMuons->push_back( caloMuon );
		   }
		}
	     }
	}
   }

   // and at last the stand alone muons
   if ( outerTrackCollectionHandle_.isValid() ) {
      LogTrace("MuonIdentification") << "Looking for new muons among stand alone muon tracks";
      for ( unsigned int i = 0; i < outerTrackCollectionHandle_->size(); ++i )
	{
	   // check if this muon is already in the list of global muons
	   bool newMuon = true;
	   for ( reco::MuonCollection::iterator muon = outputMuons->begin();
		 muon !=  outputMuons->end(); ++muon )
	     {
		if ( ! muon->standAloneMuon().isNull() ) {
		   // global muon
		   if ( muon->standAloneMuon().get() ==  &(outerTrackCollectionHandle_->at(i)) ) {
		      newMuon = false;
		      break;
		   }
		} else {
		   // tracker muon - no direct links to the standalone muon
		   // since we have only a few real muons in an event, matching
		   // the stand alone muon to the tracker muon by DetIds should
		   // be good enough for association. At the end it's up to a
		   // user to redefine the association and what it means. Here
		   // we would like to avoid obvious double counting and we
		   // tolerate a potential miss association
		   if ( overlap(*muon,outerTrackCollectionHandle_->at(i))>0 ) {
		      LogTrace("MuonIdentification") << "Found associated tracker muon. Set a reference and move on";
		      newMuon = false;
		      muon->setOuterTrack( reco::TrackRef( outerTrackCollectionHandle_, i ) );
		      muon->setType( muon->type() | reco::Muon::StandAloneMuon );
		      break;
		   }
		}
	     }
	   if ( newMuon ) {
	      LogTrace("MuonIdentification") << "No associated stand alone track is found. Making a muon";
	      outputMuons->push_back( makeMuon(iEvent, iSetup,
					       reco::TrackRef( outerTrackCollectionHandle_, i ), reco::Muon::OuterTrack ) );
	      outputMuons->back().setType( reco::Muon::StandAloneMuon );
	   }
	}
   }

   LogTrace("MuonIdentification") << "Dress up muons if it's necessary";

   int nMuons=outputMuons->size();

   std::vector<reco::MuonTimeExtra> dtTimeColl(nMuons);
   std::vector<reco::MuonTimeExtra> cscTimeColl(nMuons);
   std::vector<reco::MuonTimeExtra> combinedTimeColl(nMuons);
   std::vector<reco::IsoDeposit> trackDepColl(nMuons);
   std::vector<reco::IsoDeposit> ecalDepColl(nMuons);
   std::vector<reco::IsoDeposit> hcalDepColl(nMuons);
   std::vector<reco::IsoDeposit> hoDepColl(nMuons);
   std::vector<reco::IsoDeposit> jetDepColl(nMuons);

   // Fill various information
   unsigned int i=0;
   for ( reco::MuonCollection::iterator muon = outputMuons->begin(); muon != outputMuons->end(); ++muon )
     {
	// Fill muonID
	if ( ( fillMatching_ && ! muon->isMatchesValid() ) ||
	     ( fillEnergy_ && !muon->isEnergyValid() ) )
	  {
	     // predict direction based on the muon interaction region location
	     // if it's available
	     if ( muon->isStandAloneMuon() ) {
		if ( cos(phiOfMuonIneteractionRegion(*muon) - muon->phi()) > 0 )
		  fillMuonId(iEvent, iSetup, *muon, TrackDetectorAssociator::InsideOut);
		else
		  fillMuonId(iEvent, iSetup, *muon, TrackDetectorAssociator::OutsideIn);
	     } else {
		LogTrace("MuonIdentification") << "THIS SHOULD NEVER HAPPEN";
		fillMuonId(iEvent, iSetup, *muon);
	     }
	  }

	if (fillGlobalTrackQuality_){
	  // Fill global quality information
	  fillGlbQuality(iEvent, iSetup, *muon);
	}
	LogDebug("MuonIdentification");

        if (fillTrackerKink_) {
            fillTrackerKink(*muon);
        }

	if ( fillCaloCompatibility_ ) muon->setCaloCompatibility( muonCaloCompatibility_.evaluate(*muon) );

	if ( fillIsolation_ ) fillMuonIsolation(iEvent, iSetup, *muon,
						trackDepColl[i], ecalDepColl[i], hcalDepColl[i], hoDepColl[i], jetDepColl[i]);

        // fill timing information
        reco::MuonTime muonTime;
        reco::MuonTimeExtra dtTime;
        reco::MuonTimeExtra cscTime;
        reco::MuonTimeExtra combinedTime;

        theTimingFiller_->fillTiming(*muon, dtTime, cscTime, combinedTime, iEvent, iSetup);

        muonTime.nDof=combinedTime.nDof();
        muonTime.timeAtIpInOut=combinedTime.timeAtIpInOut();
        muonTime.timeAtIpInOutErr=combinedTime.timeAtIpInOutErr();
        muonTime.timeAtIpOutIn=combinedTime.timeAtIpOutIn();
        muonTime.timeAtIpOutInErr=combinedTime.timeAtIpOutInErr();

        muon->setTime(	muonTime);
        dtTimeColl[i] = dtTime;
        cscTimeColl[i] = cscTime;
        combinedTimeColl[i] = combinedTime;

        i++;

     }

   LogTrace("MuonIdentification") << "number of muons produced: " << outputMuons->size();
   if ( fillMatching_ ) fillArbitrationInfo( outputMuons.get() );
   edm::OrphanHandle<reco::MuonCollection> muonHandle = iEvent.put(outputMuons);

   filler.insert(muonHandle, combinedTimeColl.begin(), combinedTimeColl.end());
   filler.fill();
   fillerDT.insert(muonHandle, dtTimeColl.begin(), dtTimeColl.end());
   fillerDT.fill();
   fillerCSC.insert(muonHandle, cscTimeColl.begin(), cscTimeColl.end());
   fillerCSC.fill();

   iEvent.put(muonTimeMap,"combined");
   iEvent.put(muonTimeMapDT,"dt");
   iEvent.put(muonTimeMapCSC,"csc");

   if (writeIsoDeposits_ && fillIsolation_){
     trackDepFiller.insert(muonHandle, trackDepColl.begin(), trackDepColl.end());
     trackDepFiller.fill();
     iEvent.put(trackDepMap, trackDepositName_);
     ecalDepFiller.insert(muonHandle, ecalDepColl.begin(), ecalDepColl.end());
     ecalDepFiller.fill();
     iEvent.put(ecalDepMap,  ecalDepositName_);
     hcalDepFiller.insert(muonHandle, hcalDepColl.begin(), hcalDepColl.end());
     hcalDepFiller.fill();
     iEvent.put(hcalDepMap,  hcalDepositName_);
     hoDepFiller.insert(muonHandle, hoDepColl.begin(), hoDepColl.end());
     hoDepFiller.fill();
     iEvent.put(hoDepMap,    hoDepositName_);
     jetDepFiller.insert(muonHandle, jetDepColl.begin(), jetDepColl.end());
     jetDepFiller.fill();
     iEvent.put(jetDepMap,  jetDepositName_);
   }

   iEvent.put(caloMuons);
}


bool MuonIdProducer::isGoodTrackerMuon( const reco::Muon& muon )
{
  if(muon.track()->pt() < minPt_ || muon.track()->p() < minP_) return false;
   if ( addExtraSoftMuons_ &&
	muon.pt()<5 && fabs(muon.eta())<1.5 &&
	muon.numberOfMatches( reco::Muon::NoArbitration ) >= 1 ) return true;
   return ( muon.numberOfMatches( reco::Muon::NoArbitration ) >= minNumberOfMatches_ );
}

bool MuonIdProducer::isGoodRPCMuon( const reco::Muon& muon )
{
  if(muon.track()->pt() < minPt_ || muon.track()->p() < minP_) return false;
   if ( addExtraSoftMuons_ &&
	muon.pt()<5 && fabs(muon.eta())<1.5 &&
	muon.numberOfMatchedRPCLayers( reco::Muon::RPCHitAndTrackArbitration ) > 1 ) return true;
   return ( muon.numberOfMatchedRPCLayers( reco::Muon::RPCHitAndTrackArbitration ) > minNumberOfMatches_ );
}

void MuonIdProducer::fillMuonId(edm::Event& iEvent, const edm::EventSetup& iSetup,
				reco::Muon& aMuon,
				TrackDetectorAssociator::Direction direction)
{
   // perform track - detector association
   const reco::Track* track = 0;
   if ( ! aMuon.track().isNull() )
     track = aMuon.track().get();
   else
     {
	if ( ! aMuon.standAloneMuon().isNull() )
	  track = aMuon.standAloneMuon().get();
	else
	  throw cms::Exception("FatalError") << "Failed to fill muon id information for a muon with undefined references to tracks";
     }

   TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup, *track, parameters_, direction);

   if ( fillEnergy_ ) {
      reco::MuonEnergy muonEnergy;
      muonEnergy.em      = info.crossedEnergy(TrackDetMatchInfo::EcalRecHits);
      muonEnergy.had     = info.crossedEnergy(TrackDetMatchInfo::HcalRecHits);
      muonEnergy.ho      = info.crossedEnergy(TrackDetMatchInfo::HORecHits);
      muonEnergy.tower   = info.crossedEnergy(TrackDetMatchInfo::TowerTotal);
      muonEnergy.emS9    = info.nXnEnergy(TrackDetMatchInfo::EcalRecHits,1); // 3x3 energy
      muonEnergy.emS25   = info.nXnEnergy(TrackDetMatchInfo::EcalRecHits,2); // 5x5 energy
      muonEnergy.hadS9   = info.nXnEnergy(TrackDetMatchInfo::HcalRecHits,1); // 3x3 energy
      muonEnergy.hoS9    = info.nXnEnergy(TrackDetMatchInfo::HORecHits,1);   // 3x3 energy
      muonEnergy.towerS9 = info.nXnEnergy(TrackDetMatchInfo::TowerTotal,1);  // 3x3 energy
      muonEnergy.ecal_position = info.trkGlobPosAtEcal;
      muonEnergy.hcal_position = info.trkGlobPosAtHcal;
      if (! info.crossedEcalIds.empty() ) muonEnergy.ecal_id = info.crossedEcalIds.front();
      if (! info.crossedHcalIds.empty() ) muonEnergy.hcal_id = info.crossedHcalIds.front();
      // find maximal energy depositions and their time
      DetId emMaxId      = info.findMaxDeposition(TrackDetMatchInfo::EcalRecHits,2); // max energy deposit in 5x5 shape
      for(std::vector<const EcalRecHit*>::const_iterator hit=info.ecalRecHits.begin();
	  hit!=info.ecalRecHits.end(); ++hit) {
	 if ((*hit)->id() != emMaxId) continue;
	 muonEnergy.emMax   = (*hit)->energy();
	 muonEnergy.ecal_time = (*hit)->time();
      }
      DetId hadMaxId     = info.findMaxDeposition(TrackDetMatchInfo::HcalRecHits,1); // max energy deposit in 3x3 shape
      for(std::vector<const HBHERecHit*>::const_iterator hit=info.hcalRecHits.begin();
	  hit!=info.hcalRecHits.end(); ++hit) {
	 if ((*hit)->id() != hadMaxId) continue;
	 muonEnergy.hadMax   = (*hit)->energy();
	 muonEnergy.hcal_time = (*hit)->time();
      }
      aMuon.setCalEnergy( muonEnergy );
   }
   if ( ! fillMatching_ && ! aMuon.isTrackerMuon() && ! aMuon.isRPCMuon() ) return;

  edm::Handle<RPCRecHitCollection> rpcRecHits;
  iEvent.getByToken(rpcHitToken_, rpcRecHits);

   // fill muon match info
   std::vector<reco::MuonChamberMatch> muonChamberMatches;
   unsigned int nubmerOfMatchesAccordingToTrackAssociator = 0;
   for( std::vector<TAMuonChamberMatch>::const_iterator chamber=info.chambers.begin();
	chamber!=info.chambers.end(); chamber++ )
     {
       if  (chamber->id.subdetId() == 3 && rpcRecHits.isValid()  ) continue; // Skip RPC chambers, they are taken care of below)
	reco::MuonChamberMatch matchedChamber;

	LocalError localError = chamber->tState.localError().positionError();
	matchedChamber.x = chamber->tState.localPosition().x();
	matchedChamber.y = chamber->tState.localPosition().y();
	matchedChamber.xErr = sqrt( localError.xx() );
	matchedChamber.yErr = sqrt( localError.yy() );

	matchedChamber.dXdZ = chamber->tState.localDirection().z()!=0?chamber->tState.localDirection().x()/chamber->tState.localDirection().z():9999;
	matchedChamber.dYdZ = chamber->tState.localDirection().z()!=0?chamber->tState.localDirection().y()/chamber->tState.localDirection().z():9999;
	// DANGEROUS - compiler cannot guaranty parameters ordering
	AlgebraicSymMatrix55 trajectoryCovMatrix = chamber->tState.localError().matrix();
	matchedChamber.dXdZErr = trajectoryCovMatrix(1,1)>0?sqrt(trajectoryCovMatrix(1,1)):0;
	matchedChamber.dYdZErr = trajectoryCovMatrix(2,2)>0?sqrt(trajectoryCovMatrix(2,2)):0;

	matchedChamber.edgeX = chamber->localDistanceX;
	matchedChamber.edgeY = chamber->localDistanceY;

	matchedChamber.id = chamber->id;
	if ( ! chamber->segments.empty() ) ++nubmerOfMatchesAccordingToTrackAssociator;

	// fill segments
	for( std::vector<TAMuonSegmentMatch>::const_iterator segment = chamber->segments.begin();
     segment != chamber->segments.end(); segment++ )
	  {
	     reco::MuonSegmentMatch matchedSegment;
	     matchedSegment.x = segment->segmentLocalPosition.x();
	     matchedSegment.y = segment->segmentLocalPosition.y();
	     matchedSegment.dXdZ = segment->segmentLocalDirection.z()?segment->segmentLocalDirection.x()/segment->segmentLocalDirection.z():0;
	     matchedSegment.dYdZ = segment->segmentLocalDirection.z()?segment->segmentLocalDirection.y()/segment->segmentLocalDirection.z():0;
	     matchedSegment.xErr = segment->segmentLocalErrorXX>0?sqrt(segment->segmentLocalErrorXX):0;
	     matchedSegment.yErr = segment->segmentLocalErrorYY>0?sqrt(segment->segmentLocalErrorYY):0;
	     matchedSegment.dXdZErr = segment->segmentLocalErrorDxDz>0?sqrt(segment->segmentLocalErrorDxDz):0;
	     matchedSegment.dYdZErr = segment->segmentLocalErrorDyDz>0?sqrt(segment->segmentLocalErrorDyDz):0;
	     matchedSegment.t0 = segment->t0;
	     matchedSegment.mask = 0;
             matchedSegment.dtSegmentRef  = segment->dtSegmentRef;
             matchedSegment.cscSegmentRef = segment->cscSegmentRef;
        matchedSegment.hasZed_ = segment->hasZed;
        matchedSegment.hasPhi_ = segment->hasPhi;
	     // test segment
	     bool matchedX = false;
	     bool matchedY = false;
	     LogTrace("MuonIdentification") << " matching local x, segment x: " << matchedSegment.x <<
	       ", chamber x: " << matchedChamber.x << ", max: " << maxAbsDx_;
	     LogTrace("MuonIdentification") << " matching local y, segment y: " << matchedSegment.y <<
	       ", chamber y: " << matchedChamber.y << ", max: " << maxAbsDy_;
	     if (matchedSegment.xErr>0 && matchedChamber.xErr>0 )
	       LogTrace("MuonIdentification") << " xpull: " <<
	       fabs(matchedSegment.x - matchedChamber.x)/sqrt(pow(matchedSegment.xErr,2) + pow(matchedChamber.xErr,2));
	     if (matchedSegment.yErr>0 && matchedChamber.yErr>0 )
	       LogTrace("MuonIdentification") << " ypull: " <<
	       fabs(matchedSegment.y - matchedChamber.y)/sqrt(pow(matchedSegment.yErr,2) + pow(matchedChamber.yErr,2));

	     if (fabs(matchedSegment.x - matchedChamber.x) < maxAbsDx_) matchedX = true;
	     if (fabs(matchedSegment.y - matchedChamber.y) < maxAbsDy_) matchedY = true;
	     if (matchedSegment.xErr>0 && matchedChamber.xErr>0 &&
		 fabs(matchedSegment.x - matchedChamber.x)/sqrt(pow(matchedSegment.xErr,2) + pow(matchedChamber.xErr,2)) < maxAbsPullX_) matchedX = true;
	     if (matchedSegment.yErr>0 && matchedChamber.yErr>0 &&
		 fabs(matchedSegment.y - matchedChamber.y)/sqrt(pow(matchedSegment.yErr,2) + pow(matchedChamber.yErr,2)) < maxAbsPullY_) matchedY = true;
	     if (matchedX && matchedY) matchedChamber.segmentMatches.push_back(matchedSegment);
	  }
	muonChamberMatches.push_back(matchedChamber);
     }

  // Fill RPC info
  if ( rpcRecHits.isValid() )
  {

   for( std::vector<TAMuonChamberMatch>::const_iterator chamber=info.chambers.begin();
	chamber!=info.chambers.end(); chamber++ )
     {

      if ( chamber->id.subdetId() != 3 ) continue; // Consider RPC chambers only

      reco::MuonChamberMatch matchedChamber;

      LocalError localError = chamber->tState.localError().positionError();
      matchedChamber.x = chamber->tState.localPosition().x();
      matchedChamber.y = chamber->tState.localPosition().y();
      matchedChamber.xErr = sqrt( localError.xx() );
      matchedChamber.yErr = sqrt( localError.yy() );

      matchedChamber.dXdZ = chamber->tState.localDirection().z()!=0?chamber->tState.localDirection().x()/chamber->tState.localDirection().z():9999;
      matchedChamber.dYdZ = chamber->tState.localDirection().z()!=0?chamber->tState.localDirection().y()/chamber->tState.localDirection().z():9999;
      // DANGEROUS - compiler cannot guaranty parameters ordering
      AlgebraicSymMatrix55 trajectoryCovMatrix = chamber->tState.localError().matrix();
      matchedChamber.dXdZErr = trajectoryCovMatrix(1,1)>0?sqrt(trajectoryCovMatrix(1,1)):0;
      matchedChamber.dYdZErr = trajectoryCovMatrix(2,2)>0?sqrt(trajectoryCovMatrix(2,2)):0;

      matchedChamber.edgeX = chamber->localDistanceX;
      matchedChamber.edgeY = chamber->localDistanceY;

      matchedChamber.id = chamber->id;

      for ( RPCRecHitCollection::const_iterator rpcRecHit = rpcRecHits->begin();
            rpcRecHit != rpcRecHits->end(); ++rpcRecHit )
      {
        reco::MuonRPCHitMatch rpcHitMatch;

        if ( rpcRecHit->rawId() != chamber->id.rawId() ) continue;

        rpcHitMatch.x = rpcRecHit->localPosition().x();
        rpcHitMatch.mask = 0;
        rpcHitMatch.bx = rpcRecHit->BunchX();

        const double AbsDx = fabs(rpcRecHit->localPosition().x()-chamber->tState.localPosition().x());
        if( AbsDx <= 20 or AbsDx/sqrt(localError.xx()) <= 4 ) matchedChamber.rpcMatches.push_back(rpcHitMatch);
      }

      muonChamberMatches.push_back(matchedChamber);
    }
  }

   aMuon.setMatches(muonChamberMatches);

   LogTrace("MuonIdentification") << "number of muon chambers: " << aMuon.matches().size() << "\n"
     << "number of chambers with segments according to the associator requirements: " <<
     nubmerOfMatchesAccordingToTrackAssociator;
   LogTrace("MuonIdentification") << "number of segment matches with the producer requirements: " <<
     aMuon.numberOfMatches( reco::Muon::NoArbitration );

   // fillTime( iEvent, iSetup, aMuon );
}

void MuonIdProducer::fillArbitrationInfo( reco::MuonCollection* pOutputMuons )
{
   //
   // apply segment flags
   //
   std::vector<std::pair<reco::MuonChamberMatch*,reco::MuonSegmentMatch*> > chamberPairs;     // for chamber segment sorting
   std::vector<std::pair<reco::MuonChamberMatch*,reco::MuonSegmentMatch*> > stationPairs;     // for station segment sorting
   std::vector<std::pair<reco::MuonChamberMatch*,reco::MuonSegmentMatch*> > arbitrationPairs; // for muon segment arbitration

   // muonIndex1
   for( unsigned int muonIndex1 = 0; muonIndex1 < pOutputMuons->size(); ++muonIndex1 )
   {
      // chamberIter1
      for( std::vector<reco::MuonChamberMatch>::iterator chamberIter1 = pOutputMuons->at(muonIndex1).matches().begin();
            chamberIter1 != pOutputMuons->at(muonIndex1).matches().end(); ++chamberIter1 )
      {
         if(chamberIter1->segmentMatches.empty()) continue;
         chamberPairs.clear();

         // segmentIter1
         for( std::vector<reco::MuonSegmentMatch>::iterator segmentIter1 = chamberIter1->segmentMatches.begin();
               segmentIter1 != chamberIter1->segmentMatches.end(); ++segmentIter1 )
         {
            chamberPairs.push_back(std::make_pair(&(*chamberIter1), &(*segmentIter1)));
            if(!segmentIter1->isMask()) // has not yet been arbitrated
            {
               arbitrationPairs.clear();
               arbitrationPairs.push_back(std::make_pair(&(*chamberIter1), &(*segmentIter1)));



               // find identical segments with which to arbitrate
               // tracker muons only
               if (pOutputMuons->at(muonIndex1).isTrackerMuon()) {
                  // muonIndex2
                  for( unsigned int muonIndex2 = muonIndex1+1; muonIndex2 < pOutputMuons->size(); ++muonIndex2 )
                  {
                     // tracker muons only
                     if (! pOutputMuons->at(muonIndex2).isTrackerMuon()) continue;
                     // chamberIter2
                     for( std::vector<reco::MuonChamberMatch>::iterator chamberIter2 = pOutputMuons->at(muonIndex2).matches().begin();
                           chamberIter2 != pOutputMuons->at(muonIndex2).matches().end(); ++chamberIter2 )
                     {
                        // segmentIter2
                        for( std::vector<reco::MuonSegmentMatch>::iterator segmentIter2 = chamberIter2->segmentMatches.begin();
                              segmentIter2 != chamberIter2->segmentMatches.end(); ++segmentIter2 )
                        {
                           if(segmentIter2->isMask()) continue; // has already been arbitrated
                           if(fabs(segmentIter2->x       - segmentIter1->x      ) < 1E-3 &&
                                 fabs(segmentIter2->y       - segmentIter1->y      ) < 1E-3 &&
                                 fabs(segmentIter2->dXdZ    - segmentIter1->dXdZ   ) < 1E-3 &&
                                 fabs(segmentIter2->dYdZ    - segmentIter1->dYdZ   ) < 1E-3 &&
                                 fabs(segmentIter2->xErr    - segmentIter1->xErr   ) < 1E-3 &&
                                 fabs(segmentIter2->yErr    - segmentIter1->yErr   ) < 1E-3 &&
                                 fabs(segmentIter2->dXdZErr - segmentIter1->dXdZErr) < 1E-3 &&
                                 fabs(segmentIter2->dYdZErr - segmentIter1->dYdZErr) < 1E-3)
                              arbitrationPairs.push_back(std::make_pair(&(*chamberIter2), &(*segmentIter2)));
                        } // segmentIter2
                     } // chamberIter2
                  } // muonIndex2
               }

               // arbitration segment sort
               if(arbitrationPairs.empty()) continue; // this should never happen
               if(arbitrationPairs.size()==1) {
                  arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::BelongsToTrackByDRSlope);
                  arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::BelongsToTrackByDXSlope);
                  arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::BelongsToTrackByDR);
                  arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::BelongsToTrackByDX);
                  arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::Arbitrated);
               } else {
                  sort(arbitrationPairs.begin(), arbitrationPairs.end(), SortMuonSegmentMatches(reco::MuonSegmentMatch::BelongsToTrackByDRSlope));
                  arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::BelongsToTrackByDRSlope);
                  sort(arbitrationPairs.begin(), arbitrationPairs.end(), SortMuonSegmentMatches(reco::MuonSegmentMatch::BelongsToTrackByDXSlope));
                  arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::BelongsToTrackByDXSlope);
                  sort(arbitrationPairs.begin(), arbitrationPairs.end(), SortMuonSegmentMatches(reco::MuonSegmentMatch::BelongsToTrackByDR));
                  arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::BelongsToTrackByDR);
                  sort(arbitrationPairs.begin(), arbitrationPairs.end(), SortMuonSegmentMatches(reco::MuonSegmentMatch::BelongsToTrackByDX));
                  arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::BelongsToTrackByDX);
                  for( unsigned int it = 0; it < arbitrationPairs.size(); ++it )
                     arbitrationPairs.at(it).second->setMask(reco::MuonSegmentMatch::Arbitrated);
               }
            }

	    // setup me1a cleaning for later
	    if( chamberIter1->id.subdetId() == MuonSubdetId::CSC && arbClean_ ) {
	      CSCDetId cscid(chamberIter1->id);
	      if(cscid.ring() == 4)
		for( std::vector<reco::MuonSegmentMatch>::iterator segmentIter2 = chamberIter1->segmentMatches.begin();
		     segmentIter2 != chamberIter1->segmentMatches.end(); ++segmentIter2 ) {
		  if( segmentIter1->cscSegmentRef.isNonnull() && segmentIter2->cscSegmentRef.isNonnull() )
		    if( meshAlgo_->isDuplicateOf(segmentIter1->cscSegmentRef,segmentIter2->cscSegmentRef) &&
			(segmentIter2->mask & 0x1e0000) &&
			(segmentIter1->mask & 0x1e0000) )
		      segmentIter2->setMask(reco::MuonSegmentMatch::BelongsToTrackByME1aClean);
		  //if the track has lost the segment already through normal arbitration no need to do it again.
		}
	    }// mark all ME1/a duplicates that this track owns

         } // segmentIter1

         // chamber segment sort
         if(chamberPairs.empty()) continue; // this should never happen
         if(chamberPairs.size()==1) {
            chamberPairs.front().second->setMask(reco::MuonSegmentMatch::BestInChamberByDRSlope);
            chamberPairs.front().second->setMask(reco::MuonSegmentMatch::BestInChamberByDXSlope);
            chamberPairs.front().second->setMask(reco::MuonSegmentMatch::BestInChamberByDR);
            chamberPairs.front().second->setMask(reco::MuonSegmentMatch::BestInChamberByDX);
         } else {
            sort(chamberPairs.begin(), chamberPairs.end(), SortMuonSegmentMatches(reco::MuonSegmentMatch::BestInChamberByDRSlope));
            chamberPairs.front().second->setMask(reco::MuonSegmentMatch::BestInChamberByDRSlope);
            sort(chamberPairs.begin(), chamberPairs.end(), SortMuonSegmentMatches(reco::MuonSegmentMatch::BestInChamberByDXSlope));
            chamberPairs.front().second->setMask(reco::MuonSegmentMatch::BestInChamberByDXSlope);
            sort(chamberPairs.begin(), chamberPairs.end(), SortMuonSegmentMatches(reco::MuonSegmentMatch::BestInChamberByDR));
            chamberPairs.front().second->setMask(reco::MuonSegmentMatch::BestInChamberByDR);
            sort(chamberPairs.begin(), chamberPairs.end(), SortMuonSegmentMatches(reco::MuonSegmentMatch::BestInChamberByDX));
            chamberPairs.front().second->setMask(reco::MuonSegmentMatch::BestInChamberByDX);
         }
      } // chamberIter1

      // station segment sort
      for( int stationIndex = 1; stationIndex < 5; ++stationIndex )
         for( int detectorIndex = 1; detectorIndex < 4; ++detectorIndex )
         {
            stationPairs.clear();

            // chamberIter
            for( std::vector<reco::MuonChamberMatch>::iterator chamberIter = pOutputMuons->at(muonIndex1).matches().begin();
                  chamberIter != pOutputMuons->at(muonIndex1).matches().end(); ++chamberIter )
            {
               if(!(chamberIter->station()==stationIndex && chamberIter->detector()==detectorIndex)) continue;
               if(chamberIter->segmentMatches.empty()) continue;

               for( std::vector<reco::MuonSegmentMatch>::iterator segmentIter = chamberIter->segmentMatches.begin();
                     segmentIter != chamberIter->segmentMatches.end(); ++segmentIter )
                  stationPairs.push_back(std::make_pair(&(*chamberIter), &(*segmentIter)));
            } // chamberIter

            if(stationPairs.empty()) continue; // this may very well happen
            if(stationPairs.size()==1) {
               stationPairs.front().second->setMask(reco::MuonSegmentMatch::BestInStationByDRSlope);
               stationPairs.front().second->setMask(reco::MuonSegmentMatch::BestInStationByDXSlope);
               stationPairs.front().second->setMask(reco::MuonSegmentMatch::BestInStationByDR);
               stationPairs.front().second->setMask(reco::MuonSegmentMatch::BestInStationByDX);
            } else {
               sort(stationPairs.begin(), stationPairs.end(), SortMuonSegmentMatches(reco::MuonSegmentMatch::BestInStationByDRSlope));
               stationPairs.front().second->setMask(reco::MuonSegmentMatch::BestInStationByDRSlope);
               sort(stationPairs.begin(), stationPairs.end(), SortMuonSegmentMatches(reco::MuonSegmentMatch::BestInStationByDXSlope));
               stationPairs.front().second->setMask(reco::MuonSegmentMatch::BestInStationByDXSlope);
               sort(stationPairs.begin(), stationPairs.end(), SortMuonSegmentMatches(reco::MuonSegmentMatch::BestInStationByDR));
               stationPairs.front().second->setMask(reco::MuonSegmentMatch::BestInStationByDR);
               sort(stationPairs.begin(), stationPairs.end(), SortMuonSegmentMatches(reco::MuonSegmentMatch::BestInStationByDX));
               stationPairs.front().second->setMask(reco::MuonSegmentMatch::BestInStationByDX);
            }
         }

   } // muonIndex1

   if(arbClean_) {
     // clear old mesh, create and prune new mesh!
     meshAlgo_->clearMesh();
     meshAlgo_->runMesh(pOutputMuons);
   }
}

void MuonIdProducer::fillMuonIsolation(edm::Event& iEvent, const edm::EventSetup& iSetup, reco::Muon& aMuon,
				       reco::IsoDeposit& trackDep, reco::IsoDeposit& ecalDep, reco::IsoDeposit& hcalDep, reco::IsoDeposit& hoDep,
				       reco::IsoDeposit& jetDep)
{
   reco::MuonIsolation isoR03, isoR05;
   const reco::Track* track = 0;
   if ( ! aMuon.track().isNull() )
     track = aMuon.track().get();
   else
     {
	if ( ! aMuon.standAloneMuon().isNull() )
	  track = aMuon.standAloneMuon().get();
	else
	  throw cms::Exception("FatalError") << "Failed to compute muon isolation information for a muon with undefined references to tracks";
     }

   // get deposits
   reco::IsoDeposit depTrk = muIsoExtractorTrack_->deposit(iEvent, iSetup, *track );
   std::vector<reco::IsoDeposit> caloDeps = muIsoExtractorCalo_->deposits(iEvent, iSetup, *track);
   reco::IsoDeposit depJet = muIsoExtractorJet_->deposit(iEvent, iSetup, *track );

   if(caloDeps.size()!=3) {
      LogTrace("MuonIdentification") << "Failed to fill vector of calorimeter isolation deposits!";
      return;
   }

   reco::IsoDeposit depEcal = caloDeps.at(0);
   reco::IsoDeposit depHcal = caloDeps.at(1);
   reco::IsoDeposit depHo   = caloDeps.at(2);

   trackDep = depTrk;
   ecalDep = depEcal;
   hcalDep = depHcal;
   hoDep = depHo;
   jetDep = depJet;

   isoR03.sumPt     = depTrk.depositWithin(0.3);
   isoR03.emEt      = depEcal.depositWithin(0.3);
   isoR03.hadEt     = depHcal.depositWithin(0.3);
   isoR03.hoEt      = depHo.depositWithin(0.3);
   isoR03.nTracks   = depTrk.depositAndCountWithin(0.3).second;
   isoR03.nJets     = depJet.depositAndCountWithin(0.3).second;
   isoR03.trackerVetoPt  = depTrk.candEnergy();
   isoR03.emVetoEt       = depEcal.candEnergy();
   isoR03.hadVetoEt      = depHcal.candEnergy();
   isoR03.hoVetoEt       = depHo.candEnergy();

   isoR05.sumPt     = depTrk.depositWithin(0.5);
   isoR05.emEt      = depEcal.depositWithin(0.5);
   isoR05.hadEt     = depHcal.depositWithin(0.5);
   isoR05.hoEt      = depHo.depositWithin(0.5);
   isoR05.nTracks   = depTrk.depositAndCountWithin(0.5).second;
   isoR05.nJets     = depJet.depositAndCountWithin(0.5).second;
   isoR05.trackerVetoPt  = depTrk.candEnergy();
   isoR05.emVetoEt       = depEcal.candEnergy();
   isoR05.hadVetoEt      = depHcal.candEnergy();
   isoR05.hoVetoEt       = depHo.candEnergy();


   aMuon.setIsolation(isoR03, isoR05);

}

reco::Muon MuonIdProducer::makeMuon( const reco::Track& track )
{
   //FIXME: E = sqrt(p^2 + m^2), where m == 0.105658369(9)GeV
   double energy = sqrt(track.p() * track.p() + 0.011163691);
   math::XYZTLorentzVector p4(track.px(),
			      track.py(),
			      track.pz(),
			      energy);
   return reco::Muon( track.charge(), p4, track.vertex() );
}

double MuonIdProducer::sectorPhi( const DetId& id )
{
   double phi = 0;
   if( id.subdetId() ==  MuonSubdetId::DT ) {    // DT
      DTChamberId muonId(id.rawId());
      if ( muonId.sector() <= 12 )
	phi = (muonId.sector()-1)/6.*M_PI;
      if ( muonId.sector() == 13 ) phi = 3/6.*M_PI;
      if ( muonId.sector() == 14 ) phi = 9/6.*M_PI;
   }
   if( id.subdetId() == MuonSubdetId::CSC ) {    // CSC
      CSCDetId muonId(id.rawId());
      phi = M_PI/4+(muonId.triggerSector()-1)/3.*M_PI;
   }
   if ( phi > M_PI ) phi -= 2*M_PI;
   return phi;
}

double MuonIdProducer::phiOfMuonIneteractionRegion( const reco::Muon& muon ) const
{
   if ( muon.isStandAloneMuon() ) return muon.standAloneMuon()->innerPosition().phi();
   // the rest is tracker muon only
   if ( muon.matches().empty() ){
      if ( muon.innerTrack().isAvailable() &&
	   muon.innerTrack()->extra().isAvailable() )
	return muon.innerTrack()->outerPosition().phi();
      else
	return muon.phi(); // makes little sense, but what else can I use
   }
   return sectorPhi(muon.matches().at(0).id);
}

void MuonIdProducer::fillGlbQuality(edm::Event& iEvent, const edm::EventSetup& iSetup, reco::Muon& aMuon)
{
  edm::Handle<edm::ValueMap<reco::MuonQuality> > glbQualH;
  iEvent.getByToken(glbQualToken_, glbQualH);

  if(aMuon.isGlobalMuon() && glbQualH.isValid() && !glbQualH.failedToGet()) {
    aMuon.setCombinedQuality((*glbQualH)[aMuon.combinedMuon()]);
  }

  LogDebug("MuonIdentification") << "tkChiVal " << aMuon.combinedQuality().trkRelChi2;

}

void MuonIdProducer::fillTrackerKink( reco::Muon& aMuon ) {
    // skip muons with no tracks
    if (aMuon.innerTrack().isNull()) return;
    // get quality from muon if already there, otherwise make empty one
    reco::MuonQuality quality = (aMuon.isQualityValid() ? aMuon.combinedQuality() : reco::MuonQuality());
    // fill it
    bool filled = trackerKinkFinder_->fillTrkKink(quality, *aMuon.innerTrack());
    // if quality was there, or if we filled it, commit to the muon
    if (filled || aMuon.isQualityValid()) aMuon.setCombinedQuality(quality);
}

bool MuonIdProducer::checkLinks(const reco::MuonTrackLinks* links) const {
  bool trackBAD = links->trackerTrack().isNull();
  bool staBAD = links->standAloneTrack().isNull();
  bool glbBAD = links->globalTrack().isNull();
  if (trackBAD || staBAD || glbBAD )
    {
      edm::LogWarning("muonIDbadLinks") << "Global muon links to constituent tracks are invalid: trkBad "
					<<trackBAD <<" standaloneBad "<<staBAD<<" globalBad "<<glbBAD
					<<". There should be no such object. Muon is skipped.";
      return false;
    }
  return true;
}
