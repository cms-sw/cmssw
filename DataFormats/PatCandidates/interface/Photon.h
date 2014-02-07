//
//

#ifndef DataFormats_PatCandidates_Photon_h
#define DataFormats_PatCandidates_Photon_h

/**
  \class    pat::Photon Photon.h "DataFormats/PatCandidates/interface/Photon.h"
  \brief    Analysis-level Photon class

   pat::Photon implements the analysis-level photon class within the 'pat'
   namespace.

   Please post comments and questions to the Physics Tools hypernews:
   https://hypernews.cern.ch/HyperNews/CMS/get/physTools.html

  \author   Steven Lowette, Giovanni Petrucciani, Frederic Ronga
*/

#include "DataFormats/PatCandidates/interface/PATObject.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/PatCandidates/interface/Isolation.h"


// Define typedefs for convenience
namespace pat {
  class Photon;
  typedef std::vector<Photon>              PhotonCollection; 
  typedef edm::Ref<PhotonCollection>       PhotonRef; 
  typedef edm::RefVector<PhotonCollection> PhotonRefVector; 
}

namespace reco {
  /// pipe operator (introduced to use pat::Photon with PFTopProjectors)
  std::ostream& operator<<(std::ostream& out, const pat::Photon& obj);
}

// Class definition
namespace pat {


  class Photon : public PATObject<reco::Photon> {

    public:

      typedef std::pair<std::string,Bool_t> IdPair;

      /// default constructor
      Photon();
      /// constructor from a reco photon
      Photon(const reco::Photon & aPhoton);
      /// constructor from a RefToBase to a reco photon (to be superseded by Ptr counterpart)
      Photon(const edm::RefToBase<reco::Photon> & aPhotonRef);
      /// constructor from a Ptr to a reco photon
      Photon(const edm::Ptr<reco::Photon> & aPhotonRef);
      /// destructor
      virtual ~Photon();

      /// required reimplementation of the Candidate's clone method
      virtual Photon * clone() const { return new Photon(*this); }

      // ---- methods for content embedding ----
      /// override the superCluster method from CaloJet, to access the internal storage of the supercluster
      reco::SuperClusterRef superCluster() const;
      /// method to store the photon's supercluster internally
      void embedSuperCluster();

      // ---- methods for access the generated photon ----
      /// return the match to the generated photon
      const reco::Candidate * genPhoton() const { return genParticle(); }
      /// method to set the generated photon
      void setGenPhoton(const reco::GenParticleRef & gp, bool embed=false) { setGenParticleRef(gp, embed); }

      // ---- methods for photon ID ----
      /// Returns a specific photon ID associated to the pat::Photon given its name.
      /// Note: an exception is thrown if the specified ID is not available
      Bool_t photonID(const std::string & name) const;
      /// Returns true if a specific ID is available in this pat::Photon
      bool isPhotonIDAvailable(const std::string & name) const;
      /// Returns all the Photon IDs in the form of <name,value> pairs
      /// The 'default' ID is the first in the list
      const std::vector<IdPair> &  photonIDs() const { return photonIDs_; }
      /// Store multiple photon ID values, discarding existing ones
      /// The first one in the list becomes the 'default' photon id 
      void setPhotonIDs(const std::vector<IdPair> & ids) { photonIDs_ = ids; }


      // ---- methods for photon isolation ----
      /// Returns the summed track pt in a cone of deltaR<0.4 
      /// including the region of the reconstructed photon 
      float trackIso() const { return trkSumPtSolidConeDR04(); }
      /// Returns the summed Et in a cone of deltaR<0.4 
      /// calculated from recHits
      float ecalIso()  const { return ecalRecHitSumEtConeDR04(); }
      /// Returns summed Et in a cone of deltaR<0.4 calculated 
      /// from caloTowers
      float hcalIso()  const { return hcalTowerSumEtConeDR04(); }
      /// Returns the calorimeter isolation combined from ecal 
      /// and hcal 
      float caloIso()  const { return ecalIso()+hcalIso(); }

      /// PARTICLE FLOW ISOLATION
      /// Returns the isolation calculated with all the PFCandidates
      float particleIso() const { return userIsolation(pat::PfAllParticleIso); }
      /// Returns the isolation calculated with only the charged hadron
      /// PFCandidates
      float chargedHadronIso() const { return userIsolation(pat::PfChargedHadronIso); }
      /// Returns the isolation calculated with only the neutral hadron
      /// PFCandidates
      float neutralHadronIso() const { return userIsolation(pat::PfNeutralHadronIso); }        
      /// Returns the isolation calculated with only the gamma
      /// PFCandidates
      float photonIso() const { return userIsolation(pat::PfGammaIso); }
      /// Returns the isolation calculated with only the pile-up charged hadron
      /// PFCandidates
      float puChargedHadronIso() const { return userIsolation(pat::PfPUChargedHadronIso); }        

      /// Returns a user defined isolation value
      float userIso(uint8_t index=0)  const { return userIsolation(IsolationKeys(UserBaseIso + index)); }
      /// Returns the isolation variable for a specifc key (or 
      /// pseudo-key like CaloIso), or -1.0 if not available
      float userIsolation(IsolationKeys key) const { 
          if (key >= 0) {
	    //if (key >= isolations_.size()) throw cms::Excepton("Missing Data") 
	    //<< "Isolation corresponding to key " << key 
	    //<< " was not stored for this particle.";
              if (size_t(key) >= isolations_.size()) return -1.0;
              return isolations_[key];
          } else switch (key) {
	  case pat::CaloIso:  
		//if (isolations_.size() <= pat::HcalIso) throw cms::Excepton("Missing Data") 
		//<< "CalIsoo Isolation was not stored for this particle.";
		if (isolations_.size() <= pat::HcalIso) return -1.0; 
		return isolations_[pat::EcalIso] + isolations_[pat::HcalIso];
	  default:
	    return -1.0;
	    //throw cms::Excepton("Missing Data") << "Isolation corresponding to key " 
	    //<< key << " was not stored for this particle.";
          }
      }
      /// Sets the isolation variable for a specifc key.
      /// Note that you can't set isolation for a pseudo-key like CaloIso
      void setIsolation(IsolationKeys key, float value) {
          if (key >= 0) {
              if (size_t(key) >= isolations_.size()) isolations_.resize(key+1, -1.0);
              isolations_[key] = value;
          } else {
              throw cms::Exception("Illegal Argument") << 
                  "The key for which you're setting isolation does not correspond " <<
                  "to an individual isolation but to the sum of more independent isolations " <<
                  "(e.g. Calo = Ecal + Hcal), so you can't SET the value, just GET it.\n" <<
                  "Please set up each component independly.\n";
          }
      }
      /// Sets tracker isolation variable
      void setTrackIso(float trackIso) { setIsolation(TrackIso, trackIso); }
      /// Sets ecal isolation variable
      void setEcalIso(float caloIso)   { setIsolation(EcalIso,  caloIso);  } 
      /// Sets hcal isolation variable
      void setHcalIso(float caloIso)   { setIsolation(HcalIso,  caloIso);  }
      /// Sets user isolation variable #index
      void setUserIso(float value, uint8_t index=0)  { setIsolation(IsolationKeys(UserBaseIso + index), value); }

      // ---- methods for photon isolation deposits ----
      /// Returns the IsoDeposit associated with some key, or a null pointer if it is not available
      const IsoDeposit * isoDeposit(IsolationKeys key) const {
          for (IsoDepositPairs::const_iterator it = isoDeposits_.begin(), ed = isoDeposits_.end(); 
                  it != ed; ++it) 
          {
              if (it->first == key) return & it->second;
          }
          return 0;
      } 
      /// Return the tracker IsoDeposit
      const IsoDeposit * trackIsoDeposit() const { return isoDeposit(pat::TrackIso); }
      /// Return the ecal IsoDeposit
      const IsoDeposit * ecalIsoDeposit()  const { return isoDeposit(pat::EcalIso ); }
      /// Return the hcal IsoDeposit
      const IsoDeposit * hcalIsoDeposit()  const { return isoDeposit(pat::HcalIso ); }
      /// Return a specified user-level IsoDeposit
      const IsoDeposit * userIsoDeposit(uint8_t index=0) const { return isoDeposit(IsolationKeys(UserBaseIso + index)); }
      /// Sets the IsoDeposit associated with some key; if it is already existent, it is overwritten.
      void setIsoDeposit(IsolationKeys key, const IsoDeposit &dep) {
          IsoDepositPairs::iterator it = isoDeposits_.begin(), ed = isoDeposits_.end();
          for (; it != ed; ++it) {
              if (it->first == key) { it->second = dep; return; }
          }
          isoDeposits_.push_back(std::make_pair(key,dep));
      } 
      /// Sets tracker IsoDeposit
      void trackIsoDeposit(const IsoDeposit &dep) { setIsoDeposit(pat::TrackIso, dep); }
      /// Sets ecal IsoDeposit
      void ecalIsoDeposit(const IsoDeposit &dep)  { setIsoDeposit(pat::EcalIso,  dep); }
      /// Sets hcal IsoDeposit
      void hcalIsoDeposit(const IsoDeposit &dep)  { setIsoDeposit(pat::HcalIso,  dep); }
      /// Sets user-level IsoDeposit
      void userIsoDeposit(const IsoDeposit &dep, uint8_t index=0) { setIsoDeposit(IsolationKeys(UserBaseIso + index), dep); }

      /// pipe operator (introduced to use pat::Photon with PFTopProjectors)
      friend std::ostream& reco::operator<<(std::ostream& out, const pat::Photon& obj);

    protected:

      // ---- for content embedding ----
      bool embeddedSuperCluster_;
      std::vector<reco::SuperCluster> superCluster_;
      // ---- photon ID's holder ----
      std::vector<IdPair> photonIDs_;
      // ---- Isolation and IsoDeposit related datamebers ----
      typedef std::vector<std::pair<IsolationKeys, pat::IsoDeposit> > IsoDepositPairs;
      IsoDepositPairs    isoDeposits_;
      std::vector<float> isolations_;

  };


}

#endif
