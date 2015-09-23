//
//

#ifndef PhysicsTools_PatAlgos_PATPhotonProducer_h
#define PhysicsTools_PatAlgos_PATPhotonProducer_h

/**
  \class    pat::PATPhotonProducer PATPhotonProducer.h "PhysicsTools/PatAlgos/interface/PATPhotonProducer.h"
  \brief    Produces the pat::Photon

   The PATPhotonProducer produces the analysis-level pat::Photon starting from
   a collection of objects of PhotonType.

  \author   Steven Lowette
  \version  $Id: PATPhotonProducer.h,v 1.19 2009/06/25 23:49:35 gpetrucc Exp $
*/


#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "CommonTools/Utils/interface/EtComparator.h"

#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"
#include "PhysicsTools/PatAlgos/interface/KinResolutionsLoader.h"

#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"


#include "DataFormats/PatCandidates/interface/UserData.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"

#include "RecoEgamma/EgammaTools/interface/EcalClusterLocal.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

namespace pat {

  class PATPhotonProducer : public edm::stream::EDProducer<> {

    public:

      explicit PATPhotonProducer(const edm::ParameterSet & iConfig);
      ~PATPhotonProducer();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup& iSetup) override;

      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

    private:

      // configurables
      edm::EDGetTokenT<edm::View<reco::Photon> > photonToken_;
      edm::EDGetTokenT<reco::GsfElectronCollection> electronToken_;
      edm::EDGetTokenT<reco::ConversionCollection> hConversionsToken_;
      edm::EDGetTokenT<reco::BeamSpot> beamLineToken_;

      bool embedSuperCluster_;
      bool          embedSeedCluster_;
      bool          embedBasicClusters_;
      bool          embedPreshowerClusters_;
      bool          embedRecHits_;

      edm::InputTag reducedBarrelRecHitCollection_;
      edm::EDGetTokenT<EcalRecHitCollection> reducedBarrelRecHitCollectionToken_;
      edm::InputTag reducedEndcapRecHitCollection_;
      edm::EDGetTokenT<EcalRecHitCollection> reducedEndcapRecHitCollectionToken_;      
      
      bool addPFClusterIso_;
      edm::EDGetTokenT<edm::ValueMap<float> > ecalPFClusterIsoT_;
      edm::EDGetTokenT<edm::ValueMap<float> > hcalPFClusterIsoT_;

      bool addGenMatch_;
      bool embedGenMatch_;
      std::vector<edm::EDGetTokenT<edm::Association<reco::GenParticleCollection> > > genMatchTokens_;

      // tools
      GreaterByEt<Photon> eTComparator_;

      typedef std::vector< edm::Handle< edm::ValueMap<IsoDeposit> > > IsoDepositMaps;
      typedef std::vector< edm::Handle< edm::ValueMap<double> > > IsolationValueMaps;
      typedef std::pair<pat::IsolationKeys,edm::InputTag> IsolationLabel;
      typedef std::vector<IsolationLabel> IsolationLabels;

      pat::helper::MultiIsolator isolator_;
      pat::helper::MultiIsolator::IsolationValuePairs isolatorTmpStorage_; // better here than recreate at each event
      std::vector<edm::EDGetTokenT<edm::ValueMap<IsoDeposit> > > isoDepositTokens_;
      std::vector<edm::EDGetTokenT<edm::ValueMap<double> > > isolationValueTokens_;
 
      IsolationLabels isoDepositLabels_;
      IsolationLabels isolationValueLabels_;

      /// fill the labels vector from the contents of the parameter set,
      /// for the isodeposit or isolation values embedding
      template<typename T> void readIsolationLabels( const edm::ParameterSet & iConfig,
                                                     const char* psetName,
                                                     IsolationLabels& labels,
                                                     std::vector<edm::EDGetTokenT<edm::ValueMap<T> > >  & tokens);

      bool addEfficiencies_;
      pat::helper::EfficiencyLoader efficiencyLoader_;

      bool addResolutions_;
      pat::helper::KinResolutionsLoader resolutionLoader_;

      bool          addPhotonID_;
      typedef std::pair<std::string, edm::InputTag> NameTag;
      std::vector<NameTag> photIDSrcs_;
      std::vector<edm::EDGetTokenT<edm::ValueMap<Bool_t> > > photIDTokens_;

      bool useUserData_;
      pat::PATUserDataHelper<pat::Photon>      userDataHelper_;
      
      const CaloTopology * ecalTopology_;
      const CaloGeometry * ecalGeometry_;
      EcalClusterLocal ecl_;


  };

}



template<typename T>
void pat::PATPhotonProducer::readIsolationLabels( const edm::ParameterSet & iConfig,
                                               const char* psetName,
                                               pat::PATPhotonProducer::IsolationLabels& labels,
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
      int key = pat::IsolationKeys::UserBaseIso;
      for ( ; it != ed; ++it, ++key) {
       labels.push_back(std::make_pair(pat::IsolationKeys(key), *it));
      }
    }
    
    tokens = edm::vector_transform(labels, [this](IsolationLabel const & label){return consumes<edm::ValueMap<T> >(label.second);});
    
  }
	tokens = edm::vector_transform(labels, [this](IsolationLabel const & label){return consumes<edm::ValueMap<T> >(label.second);});
}


#endif
