//

#ifndef PhysicsTools_PatAlgos_PATElectronProducer_h
#define PhysicsTools_PatAlgos_PATElectronProducer_h

/**
  \class    pat::PATElectronProducer PATElectronProducer.h "PhysicsTools/PatAlgos/interface/PATElectronProducer.h"
  \brief    Produces pat::Electron's

   The PATElectronProducer produces analysis-level pat::Electron's starting from
   a collection of objects of reco::GsfElectron.

  \author   Steven Lowette, James Lamb\
  \version  $Id: PATElectronProducer.h,v 1.31 2013/02/27 23:26:56 wmtan Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"

#include "CommonTools/Utils/interface/PtComparator.h"

#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"
#include "PhysicsTools/PatAlgos/interface/KinResolutionsLoader.h"

#include "DataFormats/PatCandidates/interface/Electron.h"

#include "DataFormats/PatCandidates/interface/UserData.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

#include <string>


namespace pat {


  class TrackerIsolationPt;
  class CaloIsolationEnergy;
  class LeptonLRCalc;


  class PATElectronProducer : public edm::EDProducer {

    public:

      explicit PATElectronProducer(const edm::ParameterSet & iConfig);
      ~PATElectronProducer();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup& iSetup) override;

      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

    private:

      // configurables
      edm::EDGetTokenT<edm::View<reco::GsfElectron> > electronToken_;
      edm::EDGetTokenT<reco::ConversionCollection> hConversionsToken_;
      bool          embedGsfElectronCore_;
      bool          embedGsfTrack_;
      bool          embedSuperCluster_;
      bool          embedPflowSuperCluster_;
      bool          embedSeedCluster_;
      bool          embedBasicClusters_;
      bool          embedPreshowerClusters_;
      bool          embedPflowBasicClusters_;
      bool          embedPflowPreshowerClusters_;
      bool          embedTrack_;
      bool          addGenMatch_;
      bool          embedGenMatch_;
      bool          embedRecHits_;

      typedef std::vector<edm::Handle<edm::Association<reco::GenParticleCollection> > > GenAssociations;

      std::vector<edm::EDGetTokenT<edm::Association<reco::GenParticleCollection> > > genMatchTokens_;

      /// pflow specific
      bool          useParticleFlow_;
      edm::EDGetTokenT<reco::PFCandidateCollection> pfElecToken_;
      edm::EDGetTokenT<edm::ValueMap<reco::PFCandidatePtr> > pfCandidateMapToken_;
      bool          embedPFCandidate_;

      /// mva input variables
      edm::InputTag reducedBarrelRecHitCollection_;
      edm::EDGetTokenT<EcalRecHitCollection> reducedBarrelRecHitCollectionToken_;
      edm::InputTag reducedEndcapRecHitCollection_;
      edm::EDGetTokenT<EcalRecHitCollection> reducedEndcapRecHitCollectionToken_;

      /// embed high level selection variables?
      bool          embedHighLevelSelection_;
      edm::EDGetTokenT<reco::BeamSpot> beamLineToken_;
      bool          usePV_;
      edm::EDGetTokenT<std::vector<reco::Vertex> > pvToken_;

      typedef edm::RefToBase<reco::GsfElectron> ElectronBaseRef;
      typedef std::vector< edm::Handle< edm::ValueMap<IsoDeposit> > > IsoDepositMaps;
      typedef std::vector< edm::Handle< edm::ValueMap<double> > > IsolationValueMaps;


      /// common electron filling, for both the standard and PF2PAT case
      void fillElectron( Electron& aElectron,
			 const ElectronBaseRef& electronRef,
			 const reco::CandidateBaseRef& baseRef,
			 const GenAssociations& genMatches,
			 const IsoDepositMaps& deposits,
                         const bool pfId,
			 const IsolationValueMaps& isolationValues,
                         const IsolationValueMaps& isolationValuesNoPFId) const;

      void fillElectron2( Electron& anElectron,
			  const reco::CandidatePtr& candPtrForIsolation,
			  const reco::CandidatePtr& candPtrForGenMatch,
			  const reco::CandidatePtr& candPtrForLoader,
			  const GenAssociations& genMatches,
			  const IsoDepositMaps& deposits,
			  const IsolationValueMaps& isolationValues ) const;

    // embed various impact parameters with errors
    // embed high level selection
    void embedHighLevel( pat::Electron & anElectron,
			 reco::GsfTrackRef track,
			 reco::TransientTrack & tt,
			 reco::Vertex & primaryVertex,
			 bool primaryVertexIsValid,
			 reco::BeamSpot & beamspot,
			 bool beamspotIsValid );

      typedef std::pair<pat::IsolationKeys,edm::InputTag> IsolationLabel;
      typedef std::vector<IsolationLabel> IsolationLabels;

      /// fill the labels vector from the contents of the parameter set,
      /// for the isodeposit or isolation values embedding
      template<typename T> void readIsolationLabels( const edm::ParameterSet & iConfig,
				                     const char* psetName,
				                     IsolationLabels& labels,
					             std::vector<edm::EDGetTokenT<edm::ValueMap<T> > > & tokens);

      bool          addElecID_;
      typedef std::pair<std::string, edm::InputTag> NameTag;
      std::vector<NameTag> elecIDSrcs_;
      std::vector<edm::EDGetTokenT<edm::ValueMap<float> > > elecIDTokens_;

      // tools
      GreaterByPt<Electron>       pTComparator_;

      pat::helper::MultiIsolator isolator_;
      pat::helper::MultiIsolator::IsolationValuePairs isolatorTmpStorage_; // better here than recreate at each event
      IsolationLabels isoDepositLabels_;
      std::vector<edm::EDGetTokenT<edm::ValueMap<IsoDeposit> > > isoDepositTokens_;
      IsolationLabels isolationValueLabels_;
      std::vector<edm::EDGetTokenT<edm::ValueMap<double> > > isolationValueTokens_;
      IsolationLabels isolationValueLabelsNoPFId_;
      std::vector<edm::EDGetTokenT<edm::ValueMap<double> > > isolationValueNoPFIdTokens_;

      bool addEfficiencies_;
      pat::helper::EfficiencyLoader efficiencyLoader_;

      bool addResolutions_;
      pat::helper::KinResolutionsLoader resolutionLoader_;

      bool useUserData_;
      pat::PATUserDataHelper<pat::Electron>      userDataHelper_;

      const CaloTopology * ecalTopology_;

  };


}


using namespace pat;


template<typename T>
void PATElectronProducer::readIsolationLabels( const edm::ParameterSet & iConfig,
					       const char* psetName,
					       IsolationLabels& labels,
					       std::vector<edm::EDGetTokenT<edm::ValueMap<T> > > & tokens) {

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
  tokens = edm::vector_transform(labels, [this](IsolationLabel const & label){return consumes<edm::ValueMap<T> >(label.second);});


}

#endif
