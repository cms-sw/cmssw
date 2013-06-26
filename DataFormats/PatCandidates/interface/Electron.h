//
// $Id: Electron.h,v 1.46 2013/04/11 22:42:32 beaudett Exp $
//

#ifndef DataFormats_PatCandidates_Electron_h
#define DataFormats_PatCandidates_Electron_h

/**
  \class    pat::Electron Electron.h "DataFormats/PatCandidates/interface/Electron.h"
  \brief    Analysis-level electron class

   pat::Electron implements the analysis-level electron class within the
   'pat' namespace.

   Please post comments and questions to the Physics Tools hypernews:
   https://hypernews.cern.ch/HyperNews/CMS/get/physTools.html

  \author   Steven Lowette, Giovanni Petrucciani, Frederic Ronga
  \version  $Id: Electron.h,v 1.46 2013/04/11 22:42:32 beaudett Exp $
*/


#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCoreFwd.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/PatCandidates/interface/Lepton.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

// Define typedefs for convenience
namespace pat {
  class Electron;
  typedef std::vector<Electron>              ElectronCollection;
  typedef edm::Ref<ElectronCollection>       ElectronRef;
  typedef edm::RefVector<ElectronCollection> ElectronRefVector;
}

namespace reco {
  /// pipe operator (introduced to use pat::Electron with PFTopProjectors)
  std::ostream& operator<<(std::ostream& out, const pat::Electron& obj);
}

// Class definition
namespace pat {


  class Electron : public Lepton<reco::GsfElectron> {

    public:

      typedef std::pair<std::string,float> IdPair;

      /// default constructor
      Electron();
      /// constructor from reco::GsfElectron
      Electron(const reco::GsfElectron & anElectron);
      /// constructor from a RefToBase to a reco::GsfElectron (to be superseded by Ptr counterpart)
      Electron(const edm::RefToBase<reco::GsfElectron> & anElectronRef);
      /// constructor from a Ptr to a reco::GsfElectron
      Electron(const edm::Ptr<reco::GsfElectron> & anElectronRef);
      /// destructor
      virtual ~Electron();

      /// required reimplementation of the Candidate's clone method
      virtual Electron * clone() const { return new Electron(*this); }

      // ---- methods for content embedding ----
      /// override the virtual reco::GsfElectron::core method, so that the embedded core can be used by GsfElectron client methods
      virtual reco::GsfElectronCoreRef core() const;
      /// override the reco::GsfElectron::gsfTrack method, to access the internal storage of the supercluster
      reco::GsfTrackRef gsfTrack() const;
      /// override the reco::GsfElectron::superCluster method, to access the internal storage of the supercluster
      reco::SuperClusterRef superCluster() const;
      /// override the reco::GsfElectron::pflowSuperCluster method, to access the internal storage of the pflowSuperCluster
      reco::SuperClusterRef pflowSuperCluster() const;
      /// returns nothing. Use either gsfTrack or closestCtfTrack
      reco::TrackRef track() const;
      /// override the reco::GsfElectron::closestCtfTrackRef method, to access the internal storage of the track
      reco::TrackRef closestCtfTrackRef() const;
      /// direct access to the seed cluster
      reco::CaloClusterPtr seed() const; 

      //method to access the basic clusters
      const std::vector<reco::CaloCluster>& basicClusters() const { return basicClusters_ ; }
      //method to access the preshower clusters
      const std::vector<reco::CaloCluster>& preshowerClusters() const { return preshowerClusters_ ; }
      //method to access the pflow basic clusters
      const std::vector<reco::CaloCluster>& pflowBasicClusters() const { return pflowBasicClusters_ ; }
      //method to access the pflow preshower clusters
      const std::vector<reco::CaloCluster>& pflowPreshowerClusters() const { return pflowPreshowerClusters_ ; }

      using reco::RecoCandidate::track; // avoid hiding the base implementation
      /// method to store the electron's core internally
      void embedGsfElectronCore();
      /// method to store the electron's GsfTrack internally
      void embedGsfTrack();
      /// method to store the electron's SuperCluster internally
      void embedSuperCluster();
      /// method to store the electron's PflowSuperCluster internally
      void embedPflowSuperCluster();
      /// method to store the electron's seedcluster internally
      void embedSeedCluster();
      /// method to store the electron's basic clusters
      void embedBasicClusters();
      /// method to store the electron's preshower clusters
      void embedPreshowerClusters();
      /// method to store the electron's pflow basic clusters
      void embedPflowBasicClusters();
      /// method to store the electron's pflow preshower clusters
      void embedPflowPreshowerClusters();
      /// method to store the electron's Track internally
      void embedTrack();
      /// method to store the RecHits internally - can be called from the PATElectronProducer
      void embedRecHits(const EcalRecHitCollection * rechits); 

      // ---- methods for electron ID ----
      /// Returns a specific electron ID associated to the pat::Electron given its name
      // For cut-based IDs, the value map has the following meaning:
      // 0: fails,
      // 1: passes electron ID only,
      // 2: passes electron Isolation only,
      // 3: passes electron ID and Isolation only,
      // 4: passes conversion rejection,
      // 5: passes conversion rejection and ID,
      // 6: passes conversion rejection and Isolation,
      // 7: passes the whole selection.
      // For more details have a look at:
      // https://twiki.cern.ch/twiki/bin/view/CMS/SimpleCutBasedEleID
      // https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideCategoryBasedElectronID
      // Note: an exception is thrown if the specified ID is not available
      float electronID(const std::string& name) const;
      float electronID(const char* name) const { return electronID( std::string(name) );}
      /// Returns true if a specific ID is available in this pat::Electron
      bool isElectronIDAvailable(const std::string& name) const;
      bool isElectronIDAvailable(const char* name) const {
	return isElectronIDAvailable(std::string(name));
      }
      /// Returns all the electron IDs in the form of <name,value> pairs. The 'default' ID is the first in the list
      const std::vector<IdPair> &  electronIDs() const { return electronIDs_; }
      /// Store multiple electron ID values, discarding existing ones. The first one in the list becomes the 'default' electron id
      void setElectronIDs(const std::vector<IdPair> & ids) { electronIDs_ = ids; }

      // ---- overload of isolation functions ----
      /// Overload of pat::Lepton::trackIso(); returns the value of the summed track pt in a cone of deltaR<0.4
      float trackIso() const { return dr04TkSumPt(); }
      /// Overload of pat::Lepton::ecalIso(); returns the value of the summed Et of all recHits in the ecal in a cone of deltaR<0.4
      float ecalIso()  const { return dr04EcalRecHitSumEt(); }
      /// Overload of pat::Lepton::hcalIso(); returns the value of the summed Et of all caloTowers in the hcal in a cone of deltaR<0.4
      float hcalIso()  const { return dr04HcalTowerSumEt(); }
      /// Overload of pat::Lepton::caloIso(); returns the sum of ecalIso() and hcalIso
      float caloIso()  const { return ecalIso()+hcalIso(); }

      // ---- PF specific methods ----
      bool isPF() const{ return isPF_; }
      void setIsPF(bool hasPFCandidate) { isPF_ = hasPFCandidate ; }

      /// reference to the source PFCandidates; null if this has been built from a standard electron
      reco::PFCandidateRef pfCandidateRef() const;
      /// add a reference to the source IsolatedPFCandidate
      void setPFCandidateRef(const reco::PFCandidateRef& ref) {
        pfCandidateRef_ = ref;
      }
      /// embed the PFCandidate pointed to by pfCandidateRef_
      void embedPFCandidate();
      /// get the number of non-null PFCandidates
      size_t numberOfSourceCandidatePtrs() const {
        return pfCandidateRef_.isNonnull() ? 1 : 0;
      }
      /// get the source candidate pointer with index i
      reco::CandidatePtr sourceCandidatePtr( size_type i ) const;

      // ---- embed various impact parameters with errors ----
      typedef enum IPTYPE { None = 0, PV2D = 1, PV3D = 2, BS2D = 3, BS3D = 4 } IpType;
      /// Impact parameter wrt primary vertex or beamspot
      double dB(IpType type = None) const;
      /// Uncertainty on the corresponding impact parameter
      double edB(IpType type = None) const;
      /// Set impact parameter of a certain type and its uncertainty
      void setDB(double dB, double edB, IpType type = None);

      // ---- Momentum estimate specific methods ----
      const LorentzVector & ecalDrivenMomentum() const {return ecalDrivenMomentum_;}
      void setEcalDrivenMomentum(const Candidate::LorentzVector& mom) {ecalDrivenMomentum_=mom;}

      /// pipe operator (introduced to use pat::Electron with PFTopProjectors)
      friend std::ostream& reco::operator<<(std::ostream& out, const pat::Electron& obj);

      /// additional mva input variables
      /// R9 variable
      double r9() const { return r9_; };
      /// sigmaIPhiPhi
      double sigmaIphiIphi() const { return sigmaIphiIphi_; };
      /// sigmaIEtaIPhi
      double sigmaIetaIphi() const { return sigmaIetaIphi_; };
      /// ip3d
      double ip3d() const { return ip3d_; }
      /// set missing mva input variables
      void setMvaVariables( double r9, double sigmaIphiIphi, double sigmaIetaIphi, double ip3d );

      const EcalRecHitCollection * recHits() const { return &recHits_;}

      /// additional regression variables
      /// regression1
      double ecalRegressionEnergy() const { return ecalRegressionEnergy_;}
      double ecalRegressionError() const { return ecalRegressionError_;}
      /// regression2
      double ecalTrackRegressionEnergy() const { return ecalTrackRegressionEnergy_; }
      double ecalTrackRegressionError() const { return ecalTrackRegressionError_; }
      /// set regression1
      void setEcalRegressionEnergy(double val, double err) { ecalRegressionEnergy_ = val; ecalRegressionError_ = err; }
      /// set regression2
      void setEcalTrackRegressionEnergy(double val, double err) { ecalTrackRegressionEnergy_ = val; ecalTrackRegressionError_ = err; }

      /// set scale corrections / smearings
      void setEcalScale(double val) { ecalScale_= val ;}               
      void setEcalSmear(double val) { ecalSmear_= val;}                
      void setEcalRegressionScale(double val) {	ecalRegressionScale_ = val ;}     
      void setEcalRegressionSmear(double val) {	ecalRegressionSmear_ = val;}     
      void setEcalTrackRegressionScale(double val) { ecalTrackRegressionScale_ = val;}
      void setEcalTrackRegressionSmear(double val) { ecalTrackRegressionSmear_ = val;}

      /// get scale corrections /smearings
      double ecalScale() const  { return ecalScale_ ;}               
      double ecalSmear() const  { return ecalSmear_;}                
      double ecalRegressionScale() const  { return ecalRegressionScale_  ;}     
      double ecalRegressionSmear() const  {	return ecalRegressionSmear_ ;}     
      double ecalTrackRegressionScale() const  { return ecalTrackRegressionScale_ ;}
      double ecalTrackRegressionSmear() const  { return ecalTrackRegressionSmear_ ;}
      /// vertex fit combined with missing number of hits method
      bool passConversionVeto() const { return passConversionVeto_; }
      void setPassConversionVeto( bool flag ) { passConversionVeto_ = flag; }

    protected:
      /// init impact parameter defaults (for use in a constructor)
      void initImpactParameters();

      // ---- for content embedding ----
      /// True if electron's gsfElectronCore is stored internally
      bool embeddedGsfElectronCore_;
      /// Place to store electron's gsfElectronCore internally
      std::vector<reco::GsfElectronCore> gsfElectronCore_;
      /// True if electron's gsfTrack is stored internally
      bool embeddedGsfTrack_;
      /// Place to store electron's gsfTrack internally
      std::vector<reco::GsfTrack> gsfTrack_;
      /// True if electron's supercluster is stored internally
      bool embeddedSuperCluster_;
      /// True if electron's pflowsupercluster is stored internally
      bool embeddedPflowSuperCluster_;
      /// Place to store electron's supercluster internally
      std::vector<reco::SuperCluster> superCluster_;
      /// Place to store electron's basic clusters internally 
      std::vector<reco::CaloCluster> basicClusters_;
      /// Place to store electron's preshower clusters internally      
      std::vector<reco::CaloCluster> preshowerClusters_;
      /// Place to store electron's pflow basic clusters internally
      std::vector<reco::CaloCluster> pflowBasicClusters_;
      /// Place to store electron's pflow preshower clusters internally
      std::vector<reco::CaloCluster> pflowPreshowerClusters_;
      /// Place to store electron's pflow supercluster internally
      std::vector<reco::SuperCluster> pflowSuperCluster_;
      /// True if electron's track is stored internally
      bool embeddedTrack_;
      /// Place to store electron's track internally
      std::vector<reco::Track> track_;
      /// True if seed cluster is stored internally
      bool embeddedSeedCluster_;
      /// Place to store electron's seed cluster internally
      std::vector<reco::CaloCluster> seedCluster_;
      /// True if RecHits stored internally
      bool embeddedRecHits_;	
      /// Place to store electron's RecHits internally (5x5 around seed+ all RecHits)
      EcalRecHitCollection recHits_;

      // ---- electron ID's holder ----
      /// Electron IDs
      std::vector<IdPair> electronIDs_;

      // ---- PF specific members ----
      bool isPF_;
      /// true if the IsolatedPFCandidate is embedded
      bool embeddedPFCandidate_;
      /// A copy of the source IsolatedPFCandidate is stored in this vector if embeddedPFCandidate_ if True
      reco::PFCandidateCollection pfCandidate_;
      /// reference to the IsolatedPFCandidate this has been built from; null if this has been built from a standard electron
      reco::PFCandidateRef pfCandidateRef_;

      // ---- specific members : Momentum estimates ----
      /// ECAL-driven momentum
      LorentzVector ecalDrivenMomentum_;

      // V+Jets group selection variables.
      /// True if impact parameter has been cached
      bool    cachedDB_;
      /// Impact parameter at the primary vertex
      double  dB_;
      /// Impact paramater uncertainty at the primary vertex
      double  edB_;

      /// additional missing mva variables : 14/04/2012
      double r9_;
      double sigmaIphiIphi_;
      double sigmaIetaIphi_;
      double ip3d_;

      /// output of regression
      double ecalRegressionEnergy_;
      double ecalTrackRegressionEnergy_;
      double ecalRegressionError_;
      double ecalTrackRegressionError_;

      /// scale corrections and smearing applied or to be be applied. Initialized to -99999.
      double ecalScale_;
      double ecalSmear_;

      double ecalRegressionScale_;
      double ecalRegressionSmear_;

      double ecalTrackRegressionScale_;
      double ecalTrackRegressionSmear_;
      
      
      /// conversion veto
      bool passConversionVeto_;

      // ---- cached impact parameters ----
      /// True if the IP (former dB) has been cached
      std::vector<bool>    cachedIP_;
      /// Impact parameter at the primary vertex,
      std::vector<double>  ip_;
      /// Impact parameter uncertainty as recommended by the tracking group
      std::vector<double>  eip_;
  };
}

#endif
