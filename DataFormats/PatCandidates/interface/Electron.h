//
// $Id: Electron.h,v 1.34 2011/06/08 20:40:18 rwolf Exp $
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
  \version  $Id: Electron.h,v 1.34 2011/06/08 20:40:18 rwolf Exp $
*/


#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCoreFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/PatCandidates/interface/Lepton.h"

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
      /// constructor from a reco electron
      Electron(const reco::GsfElectron & anElectron);
      /// constructor from a RefToBase to a reco electron (to be superseded by Ptr counterpart)
      Electron(const edm::RefToBase<reco::GsfElectron> & anElectronRef);
      /// constructor from a Ptr to a reco electron
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
      /// override the reco::GsfElectron::track method, to access the internal storage of the track
      reco::TrackRef track() const;
      using reco::RecoCandidate::track; // avoid hiding the base implementation
      /// method to store the electron's core internally
      void embedGsfElectronCore();
      /// method to store the electron's GsfTrack internally
      void embedGsfTrack();
      /// method to store the electron's SuperCluster internally
      void embedSuperCluster();
      /// method to store the electron's Track internally
      void embedTrack();

      // ---- methods for electron ID ----
      /// Returns a specific electron ID associated to the pat::Electron given its name
      /// For cut-based IDs, the value map has the following meaning:
      /// 0: fails
      /// 1: passes electron ID only
      /// 2: passes electron Isolation only
      /// 3: passes electron ID and Isolation only
      /// 4: passes conversion rejection
      /// 5: passes conversion rejection and ID
      /// 6: passes conversion rejection and Isolation
      /// 7: passes the whole selection
      /// For more details have a look at:
      /// https://twiki.cern.ch/twiki/bin/view/CMS/SimpleCutBasedEleID
      /// https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideCategoryBasedElectronID
      /// Note: an exception is thrown if the specified ID is not available
      float electronID(const std::string & name) const;
      /// Returns true if a specific ID is available in this pat::Electron
      bool isElectronIDAvailable(const std::string & name) const;
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
      /// get the candidate pointer with index i
      reco::CandidatePtr sourceCandidatePtr( size_type i ) const;

      // ---- embed various impact parameters with errors ----
      //
      // example:
      //
      //    // this will return the muon inner track
      //    // transverse impact parameter
      //    // relative to the primary vertex
      //    muon->dB(pat::Muon::PV2D);
      //
      //    // this will return the uncertainty
      //    // on the muon inner track
      //    // transverse impact parameter
      //    // relative to the primary vertex
      //    // or -1.0 if there is no valid PV in the event
      //    muon->edB(pat::Muon::PV2D);
      //
      // IpType defines the type of the impact parameter
      // None is default and reverts to old behavior controlled by 
      // patMuons.usePV = True/False
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

    protected:
      void initImpactParameters(); // init IP defaults in a constructor

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
      /// Place to store electron's supercluster internally
      std::vector<reco::SuperCluster> superCluster_;
      /// True if electron's track is stored internally
      bool embeddedTrack_;
      /// Place to store electron's track internally
      std::vector<reco::Track> track_;

      // ---- electron ID's holder ----
      /// Electron IDs
      std::vector<IdPair> electronIDs_;

      // ---- PF specific members ----
      /// true if the IsolatedPFCandidate is embedded
      bool embeddedPFCandidate_;
      /// if embeddedPFCandidate_, a copy of the source IsolatedPFCandidate
      /// is stored in this vector
      reco::PFCandidateCollection pfCandidate_;
      /// reference to the IsolatedPFCandidate this has been built from
      /// null if this has been built from a standard electron
      reco::PFCandidateRef pfCandidateRef_;

      // ---- specific members : Momentum estimates ----
      /// ECAL-driven momentum
      LorentzVector ecalDrivenMomentum_;

      // V+Jets group selection variables.
      // True if impact parameter has been cached
      bool    cachedDB_;
      /// Impact parameter at the primary vertex
      double  dB_;
      /// Impact paramater uncertainty at the primary vertex
      double  edB_;

      // ---- cached impact parameters ----
      std::vector<bool>    cachedIP_;  // has the IP (former dB) been cached?
      std::vector<double>  ip_;        // dB and edB are the impact parameter at the primary vertex,
      std::vector<double>  eip_;       // and its uncertainty as recommended by the tracking group
      
  };
}

#endif
