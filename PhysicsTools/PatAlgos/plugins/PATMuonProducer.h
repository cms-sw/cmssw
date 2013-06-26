//
// $Id: PATMuonProducer.h,v 1.30 2013/02/27 23:26:56 wmtan Exp $
//

#ifndef PhysicsTools_PatAlgos_PATMuonProducer_h
#define PhysicsTools_PatAlgos_PATMuonProducer_h

/**
  \class    pat::PATMuonProducer PATMuonProducer.h "PhysicsTools/PatAlgos/interface/PATMuonProducer.h"
  \brief    Produces pat::Muon's

   The PATMuonProducer produces analysis-level pat::Muon's starting from
   a collection of objects of reco::Muon.

  \author   Steven Lowette, Roger Wolf
  \version  $Id: PATMuonProducer.h,v 1.30 2013/02/27 23:26:56 wmtan Exp $
*/

#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "CommonTools/Utils/interface/PtComparator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/IPTools/interface/IPTools.h"
#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"
#include "PhysicsTools/PatAlgos/interface/KinResolutionsLoader.h"

#include "DataFormats/PatCandidates/interface/UserData.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"


namespace pat {
  /// foward declarations
  class TrackerIsolationPt;
  class CaloIsolationEnergy;

  /// class definition
  class PATMuonProducer : public edm::EDProducer {
    
  public:
    /// default constructir
    explicit PATMuonProducer(const edm::ParameterSet & iConfig);
    /// default destructur
    ~PATMuonProducer();
    /// everything that needs to be done during the event loop
    virtual void produce(edm::Event & iEvent, const edm::EventSetup& iSetup) override;
    /// description of config file parameters
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  private:
    /// typedefs for convenience
    typedef edm::RefToBase<reco::Muon> MuonBaseRef;
    typedef std::vector<edm::Handle<edm::Association<reco::GenParticleCollection> > > GenAssociations;
    typedef std::vector< edm::Handle< edm::ValueMap<IsoDeposit> > > IsoDepositMaps;
    typedef std::vector< edm::Handle< edm::ValueMap<double> > > IsolationValueMaps;
    typedef std::pair<pat::IsolationKeys,edm::InputTag> IsolationLabel;
    typedef std::vector<IsolationLabel> IsolationLabels;


    /// common muon filling, for both the standard and PF2PAT case
    void fillMuon( Muon& aMuon, const MuonBaseRef& muonRef, const reco::CandidateBaseRef& baseRef, const GenAssociations& genMatches, const IsoDepositMaps& deposits, const IsolationValueMaps& isolationValues) const;
    /// fill label vector from the contents of the parameter set, 
    /// for the embedding of isoDeposits or userIsolation values
    void readIsolationLabels( const edm::ParameterSet & iConfig, const char* psetName, IsolationLabels& labels); 


    // embed various impact parameters with errors
    // embed high level selection
    void embedHighLevel( pat::Muon & aMuon,
			 reco::TrackRef track,
			 reco::TransientTrack & tt,
			 reco::Vertex & primaryVertex,
			 bool primaryVertexIsValid,
			 reco::BeamSpot & beamspot,
			 bool beamspotIsValid );

    
  private:
    /// input source
    edm::InputTag muonSrc_;

    /// embed the track from best muon measurement
    bool embedBestTrack_;
    /// embed the track from inner tracker into the muon
    bool embedTrack_;
    /// embed track from muon system into the muon
    bool embedStandAloneMuon_;
    /// embed track of the combined fit into the muon
    bool embedCombinedMuon_;
    /// embed muon MET correction info for caloMET into the muon
    bool embedCaloMETMuonCorrs_;
    /// source of caloMET muon corrections
    edm::InputTag caloMETMuonCorrs_;
    /// embed muon MET correction info for tcMET into the muon
    bool embedTcMETMuonCorrs_;
    /// source of tcMET muon corrections
    edm::InputTag tcMETMuonCorrs_;
    /// embed track from picky muon fit into the muon
    bool embedPickyMuon_;
    /// embed track from tpfms muon fit into the muon
    bool embedTpfmsMuon_;
    /// embed track from DYT muon fit into the muon
    bool embedDytMuon_;
    /// add generator match information    
    bool addGenMatch_;
    /// input tags for generator match information
    std::vector<edm::InputTag> genMatchSrc_;
    /// embed the gen match information into the muon
    bool embedGenMatch_;
    /// add resolutions to the muon (this will be data members of th muon even w/o embedding)
    bool addResolutions_;
    /// helper class to add resolutions to the muon
    pat::helper::KinResolutionsLoader resolutionLoader_;    
    /// switch to use particle flow (PF2PAT) or not
    bool useParticleFlow_;    
    /// input source pfCandidates that will be to be transformed into pat::Muons, when using PF2PAT
    edm::InputTag pfMuonSrc_;
    /// embed pfCandidates into the muon
    bool embedPFCandidate_;
    /// embed high level selection variables
    bool embedHighLevelSelection_;
    /// input source of the primary vertex/beamspot
    edm::InputTag beamLineSrc_;
    /// use the primary vertex or the beamspot
    bool usePV_;
    /// input source of the primary vertex
    edm::InputTag pvSrc_;
    /// input source for isoDeposits
    IsolationLabels isoDepositLabels_;
    /// input source isolation value maps
    IsolationLabels isolationValueLabels_;
    /// add efficiencies to the muon (this will be data members of th muon even w/o embedding)
    bool addEfficiencies_;    
    /// add user data to the muon (this will be data members of th muon even w/o embedding)
    bool useUserData_;

    /// --- tools ---
    /// comparator for pt ordering
    GreaterByPt<Muon> pTComparator_;
    /// helper class to add userdefined isolation values to the muon
    pat::helper::MultiIsolator isolator_; 
    /// isolation value pair for temporary storage before being folded into the muon 
    pat::helper::MultiIsolator::IsolationValuePairs isolatorTmpStorage_;
    /// helper class to add efficiencies to the muon
    pat::helper::EfficiencyLoader efficiencyLoader_;
    /// helper class to add userData to the muon
    pat::PATUserDataHelper<pat::Muon> userDataHelper_;    
  };

}

#endif
