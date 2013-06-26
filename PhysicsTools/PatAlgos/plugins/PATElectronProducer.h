// $Id: PATElectronProducer.h,v 1.32 2013/04/09 18:39:29 tjkim Exp $
//

#ifndef PhysicsTools_PatAlgos_PATElectronProducer_h
#define PhysicsTools_PatAlgos_PATElectronProducer_h

/**
  \class    pat::PATElectronProducer PATElectronProducer.h "PhysicsTools/PatAlgos/interface/PATElectronProducer.h"
  \brief    Produces pat::Electron's

   The PATElectronProducer produces analysis-level pat::Electron's starting from
   a collection of objects of reco::GsfElectron.

  \author   Steven Lowette, James Lamb\
  \version  $Id: PATElectronProducer.h,v 1.32 2013/04/09 18:39:29 tjkim Exp $
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
      edm::InputTag electronSrc_;
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
      
      std::vector<edm::InputTag> genMatchSrc_;

      /// pflow specific
      bool          useParticleFlow_;
      edm::InputTag pfElecSrc_;
      edm::InputTag pfCandidateMap_;
      bool          embedPFCandidate_;

      /// mva input variables
      edm::InputTag reducedBarrelRecHitCollection_;
      edm::InputTag reducedEndcapRecHitCollection_;
 
      /// embed high level selection variables?
      bool          embedHighLevelSelection_;
      edm::InputTag beamLineSrc_;
      bool          usePV_;
      edm::InputTag pvSrc_;

      typedef std::vector<edm::Handle<edm::Association<reco::GenParticleCollection> > > GenAssociations;
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
      void readIsolationLabels( const edm::ParameterSet & iConfig,
				const char* psetName,
				IsolationLabels& labels);

      bool          addElecID_;
      typedef std::pair<std::string, edm::InputTag> NameTag;
      std::vector<NameTag> elecIDSrcs_;

      // tools
      GreaterByPt<Electron>       pTComparator_;

      pat::helper::MultiIsolator isolator_;
      pat::helper::MultiIsolator::IsolationValuePairs isolatorTmpStorage_; // better here than recreate at each event
      IsolationLabels isoDepositLabels_;
      IsolationLabels isolationValueLabels_;
      IsolationLabels isolationValueLabelsNoPFId_;

      bool addEfficiencies_;
      pat::helper::EfficiencyLoader efficiencyLoader_;

      bool addResolutions_;
      pat::helper::KinResolutionsLoader resolutionLoader_;

      bool useUserData_;
      pat::PATUserDataHelper<pat::Electron>      userDataHelper_;

      const CaloTopology * ecalTopology_;

  };


}

#endif
