//
//

#ifndef PhysicsTools_PatAlgos_PATMuonProducer_h
#define PhysicsTools_PatAlgos_PATMuonProducer_h

/**
  \class    pat::PATMuonProducer PATMuonProducer.h "PhysicsTools/PatAlgos/interface/PATMuonProducer.h"
  \brief    Produces pat::Muon's

   The PATMuonProducer produces analysis-level pat::Muon's starting from
   a collection of objects of reco::Muon.

  \author   Steven Lowette, Roger Wolf
  \version  $Id: PATMuonProducer.h,v 1.29 2012/08/22 15:02:52 bellan Exp $
*/

#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "CommonTools/Utils/interface/PtComparator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/IPTools/interface/IPTools.h"
#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"
#include "PhysicsTools/PatAlgos/interface/KinResolutionsLoader.h"

#include "DataFormats/PatCandidates/interface/UserData.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "PhysicsTools/PatAlgos/interface/MuonMvaEstimator.h"

namespace pat {
  /// foward declarations
  class TrackerIsolationPt;
  class CaloIsolationEnergy;

  /// class definition
  class PATMuonProducer : public edm::stream::EDProducer<> {

  public:
    /// default constructir
    explicit PATMuonProducer(const edm::ParameterSet & iConfig);
    /// default destructur
    ~PATMuonProducer() override;
    /// everything that needs to be done during the event loop
    void produce(edm::Event & iEvent, const edm::EventSetup& iSetup) override;
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
    template<typename T> void readIsolationLabels( const edm::ParameterSet & iConfig, const char* psetName, IsolationLabels& labels, std::vector<edm::EDGetTokenT<edm::ValueMap<T> > > & tokens);

    void setMuonMiniIso(pat::Muon& aMuon, const pat::PackedCandidateCollection *pc);
    double getRelMiniIsoPUCorrected(const pat::Muon& muon, float rho);

    // embed various impact parameters with errors
    // embed high level selection
    void embedHighLevel( pat::Muon & aMuon,
			 reco::TrackRef track,
			 reco::TransientTrack & tt,
			 reco::Vertex & primaryVertex,
			 bool primaryVertexIsValid,
			 reco::BeamSpot & beamspot,
			 bool beamspotIsValid );
    double relMiniIsoPUCorrected( const pat::Muon& aMuon,
				  double rho);

  private:
    /// input source
    edm::EDGetTokenT<edm::View<reco::Muon> > muonToken_;
    
    // for mini-iso calculation
    edm::EDGetTokenT<pat::PackedCandidateCollection > pcToken_;
    bool computeMiniIso_;
    std::vector<double> miniIsoParams_;
    double relMiniIsoPUCorrected_;

    /// embed the track from best muon measurement (global pflow)
    bool embedBestTrack_;
    /// embed the track from best muon measurement (muon only)
    bool embedTunePBestTrack_;
    /// force separate embed of the best track even if already embedded 
    bool forceEmbedBestTrack_;
    /// embed the track from inner tracker into the muon
    bool embedTrack_;
    /// embed track from muon system into the muon
    bool embedStandAloneMuon_;
    /// embed track of the combined fit into the muon
    bool embedCombinedMuon_;
    /// embed muon MET correction info for caloMET into the muon
    bool embedCaloMETMuonCorrs_;
    /// source of caloMET muon corrections
    edm::EDGetTokenT<edm::ValueMap<reco::MuonMETCorrectionData> > caloMETMuonCorrsToken_;
    /// embed muon MET correction info for tcMET into the muon
    bool embedTcMETMuonCorrs_;
    /// source of tcMET muon corrections
    edm::EDGetTokenT<edm::ValueMap<reco::MuonMETCorrectionData> > tcMETMuonCorrsToken_;
    /// embed track from picky muon fit into the muon
    bool embedPickyMuon_;
    /// embed track from tpfms muon fit into the muon
    bool embedTpfmsMuon_;
    /// embed track from DYT muon fit into the muon
    bool embedDytMuon_;
    /// add generator match information
    bool addGenMatch_;
    /// input tags for generator match information
    std::vector<edm::EDGetTokenT<edm::Association<reco::GenParticleCollection> > > genMatchTokens_;
    /// embed the gen match information into the muon
    bool embedGenMatch_;
    /// add resolutions to the muon (this will be data members of th muon even w/o embedding)
    bool addResolutions_;
    /// helper class to add resolutions to the muon
    pat::helper::KinResolutionsLoader resolutionLoader_;
    /// switch to use particle flow (PF2PAT) or not
    bool useParticleFlow_;
    /// input source pfCandidates that will be to be transformed into pat::Muons, when using PF2PAT
    edm::EDGetTokenT<reco::PFCandidateCollection> pfMuonToken_;
    /// embed pfCandidates into the muon
    bool embedPFCandidate_;
    /// embed high level selection variables
    bool embedHighLevelSelection_;
    /// input source of the primary vertex/beamspot
    edm::EDGetTokenT<reco::BeamSpot> beamLineToken_;
    /// input source of the primary vertex
    edm::EDGetTokenT<std::vector<reco::Vertex> > pvToken_;
    /// input source for isoDeposits
    IsolationLabels isoDepositLabels_;
    std::vector<edm::EDGetTokenT<edm::ValueMap<IsoDeposit> > > isoDepositTokens_;
    /// input source isolation value maps
    IsolationLabels isolationValueLabels_;
    std::vector<edm::EDGetTokenT<edm::ValueMap<double> > > isolationValueTokens_;
    /// add efficiencies to the muon (this will be data members of th muon even w/o embedding)
    bool addEfficiencies_;
    /// add user data to the muon (this will be data members of th muon even w/o embedding)
    bool useUserData_;
    /// add ecal PF energy
    bool embedPfEcalEnergy_;
    /// add puppi isolation
    bool addPuppiIsolation_;
    //PUPPI isolation tokens
    edm::EDGetTokenT<edm::ValueMap<float> > PUPPIIsolation_charged_hadrons_;
    edm::EDGetTokenT<edm::ValueMap<float> > PUPPIIsolation_neutral_hadrons_;
    edm::EDGetTokenT<edm::ValueMap<float> > PUPPIIsolation_photons_;
    //PUPPINoLeptons isolation tokens
    edm::EDGetTokenT<edm::ValueMap<float> > PUPPINoLeptonsIsolation_charged_hadrons_;
    edm::EDGetTokenT<edm::ValueMap<float> > PUPPINoLeptonsIsolation_neutral_hadrons_;
    edm::EDGetTokenT<edm::ValueMap<float> > PUPPINoLeptonsIsolation_photons_;
    /// standard muon selectors
    bool computeMuonMVA_;
    bool recomputeBasicSelectors_;
    double mvaDrMax_;
    bool mvaUseJec_;
    edm::EDGetTokenT<reco::JetTagCollection> mvaBTagCollectionTag_;
    edm::EDGetTokenT<reco::JetCorrector> mvaL1Corrector_;
    edm::EDGetTokenT<reco::JetCorrector> mvaL1L2L3ResCorrector_;
    edm::EDGetTokenT<double> rho_;
    pat::MuonMvaEstimator mvaEstimator_;
    std::string mvaTrainingFile_;
    
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

    /// MC info
    edm::EDGetTokenT<edm::ValueMap<reco::MuonSimInfo> > simInfo_;
  };

}




template<typename T>
void pat::PATMuonProducer::readIsolationLabels( const edm::ParameterSet & iConfig, const char* psetName, pat::PATMuonProducer::IsolationLabels& labels, std::vector<edm::EDGetTokenT<edm::ValueMap<T> > > & tokens)
{
  labels.clear();

  if (iConfig.exists( psetName )) {
    edm::ParameterSet depconf = iConfig.getParameter<edm::ParameterSet>(psetName);

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
      tokens = edm::vector_transform(labels, [this](IsolationLabel const & label){return consumes<edm::ValueMap<T> >(label.second);});
    }
  }
  tokens = edm::vector_transform(labels, [this](pat::PATMuonProducer::IsolationLabel const & label){return consumes<edm::ValueMap<T> >(label.second);});
}

#endif
