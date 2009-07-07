//
// $Id: PATMuonProducer.h,v 1.21 2009/06/30 22:00:54 cbern Exp $
//

#ifndef PhysicsTools_PatAlgos_PATMuonProducer_h
#define PhysicsTools_PatAlgos_PATMuonProducer_h

/**
  \class    pat::PATMuonProducer PATMuonProducer.h "PhysicsTools/PatAlgos/interface/PATMuonProducer.h"
  \brief    Produces pat::Muon's

   The PATMuonProducer produces analysis-level pat::Muon's starting from
   a collection of objects of reco::Muon.

  \author   Steven Lowette, Roger Wolf
  \version  $Id: PATMuonProducer.h,v 1.21 2009/06/30 22:00:54 cbern Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"

#include "CommonTools/Utils/interface/PtComparator.h"

#include "DataFormats/PatCandidates/interface/Muon.h"

#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"
#include "PhysicsTools/PatAlgos/interface/KinResolutionsLoader.h"

#include "DataFormats/PatCandidates/interface/UserData.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"

#include <string>


namespace pat {

  class LeptonLRCalc;
  class TrackerIsolationPt;
  class CaloIsolationEnergy;


  class PATMuonProducer : public edm::EDProducer {

    public:

      explicit PATMuonProducer(const edm::ParameterSet & iConfig);
      ~PATMuonProducer();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

    private:

      // configurables
      edm::InputTag muonSrc_;

      bool          embedTrack_;
      bool          embedStandAloneMuon_;
      bool          embedCombinedMuon_;


      bool          embedPickyMuon_;
      bool          embedTpfmsMuon_;
      
      bool          addTeVRefits_;
      edm::InputTag pickySrc_;
      edm::InputTag tpfmsSrc_;

  
      bool          addGenMatch_;
      bool          embedGenMatch_;
      std::vector<edm::InputTag> genMatchSrc_;
      bool          addResolutions_;
      pat::helper::KinResolutionsLoader resolutionLoader_;
      bool          addLRValues_;


      /// Use PF2PAT?
      bool          useParticleFlow_;

      /// for the input collection of PFCandidates, to be transformed into pat::Muons
      edm::InputTag pfMuonSrc_;

      bool          embedPFCandidate_;


      typedef std::vector<edm::Handle<edm::Association<reco::GenParticleCollection> > > GenAssociations;
      typedef edm::RefToBase<reco::Muon> MuonBaseRef;
      typedef std::vector< edm::Handle< edm::ValueMap<IsoDeposit> > > IsoDepositMaps;
      typedef std::vector< edm::Handle< edm::ValueMap<double> > > IsolationValueMaps;
      
      /// common muon filling, for both the standard and PF2PAT case
      void fillMuon( Muon& aMuon, 
		     const MuonBaseRef& muonRef,
		     const reco::CandidateBaseRef& baseRef,
		     const GenAssociations& genMatches, 
		     const IsoDepositMaps& deposits, 
		     const IsolationValueMaps& isolationValues) const;

      typedef std::pair<pat::IsolationKeys,edm::InputTag> IsolationLabel;
      typedef std::vector<IsolationLabel> IsolationLabels;

      /// fill the labels vector from the contents of the parameter set, 
      /// for the isodeposit or isolation values embedding
      void readIsolationLabels( const edm::ParameterSet & iConfig,
				const char* psetName, 
				IsolationLabels& labels); 
	
      // tools
      GreaterByPt<Muon>      pTComparator_;

      pat::helper::MultiIsolator isolator_; 
      pat::helper::MultiIsolator::IsolationValuePairs isolatorTmpStorage_; // better here than recreate at each event

      IsolationLabels isoDepositLabels_;
      IsolationLabels isolationValueLabels_;

      bool addEfficiencies_;
      pat::helper::EfficiencyLoader efficiencyLoader_;

      bool useUserData_;
      pat::PATUserDataHelper<pat::Muon>      userDataHelper_;

  };


}

#endif
