//
// $Id: Photon.h,v 1.16 2008/10/08 11:44:31 fronga Exp $
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
  \version  $Id: Photon.h,v 1.16 2008/10/08 11:44:31 fronga Exp $
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


// Class definition
namespace pat {


  typedef reco::Photon PhotonType;


  class Photon : public PATObject<PhotonType> {

    public:

      /// default constructor
      Photon();
      /// constructor from a reco photon
      Photon(const PhotonType & aPhoton);
      /// constructor from a RefToBase to a reco photon (to be superseded by Ptr counterpart)
      Photon(const edm::RefToBase<PhotonType> & aPhotonRef);
      /// constructor from a Ptr to a reco photon
      Photon(const edm::Ptr<PhotonType> & aPhotonRef);
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
      const reco::Particle * genPhoton() const { return genParticle(); }
      /// method to set the generated photon
      void setGenPhoton(const reco::GenParticleRef & gp, bool embed=false) { setGenParticleRef(gp, embed); }


/*       /// Method from reco::PhotonID, throws exception if there is no photon ID in this pat::Photon */
/*       bool isLoosePhoton() const { return photonIDOrThrow().isLoosePhoton(); } */
/*       /// Method from reco::PhotonID, throws exception if there is no photon ID in this pat::Photon */
/*       bool isTightPhoton() const { return photonIDOrThrow().isTightPhoton(); } */
/*       /// Returns computed EcalRecHit isolation */
      /// Method from reco::PhotonID, throws exception if there is no photon ID in this pat::Photon



      // ---- methods for photon isolation ----
      /// Returns the isolation variable for a specifc key (or pseudo-key like CaloIso), or -1.0 if not available
      float isolation(IsolationKeys key) const { 
          if (key >= 0) {
              //if (key >= isolations_.size()) throw cms::Excepton("Missing Data") << "Isolation corresponding to key " << key << " was not stored for this particle.";
              if (size_t(key) >= isolations_.size()) return -1.0;
              return isolations_[key];
          } else switch (key) {
              case CaloIso:  
                  //if (isolations_.size() <= HCalIso) throw cms::Excepton("Missing Data") << "CalIsoo Isolation was not stored for this particle.";
                  if (isolations_.size() <= HCalIso) return -1.0; 
                  return isolations_[ECalIso] + isolations_[HCalIso];
              default:
                  return -1.0;
                  //throw cms::Excepton("Missing Data") << "Isolation corresponding to key " << key << " was not stored for this particle.";
          }
      }
      /// Return the tracker isolation variable that was stored in this object when produced, or -1.0 if there is none
      float trackIso() const { return isolation(TrackerIso); }
      /// Return the sum of ecal and hcal isolation variable that were stored in this object when produced, or -1.0 if at least one is missing
      float caloIso()  const { return isolation(CaloIso); }
      /// Return the ecal isolation variable that was stored in this object when produced, or -1.0 if there is none
      float ecalIso()  const { return isolation(ECalIso); }
      /// Return the hcal isolation variable that was stored in this object when produced, or -1.0 if there is none
      float hcalIso()  const { return isolation(HCalIso); }
      /// Return the user defined isolation variable #index that was stored in this object when produced, or -1.0 if there is none
      float userIso(uint8_t index=0)  const { return isolation(IsolationKeys(UserBaseIso + index)); }
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
                  "(e.g. Calo = ECal + HCal), so you can't SET the value, just GET it.\n" <<
                  "Please set up each component independly.\n";
          }
      }
      /// Sets tracker isolation variable
      void setTrackIso(float trackIso) { setIsolation(TrackerIso, trackIso); }
      /// Sets ecal isolation variable
      void setECalIso(float caloIso)   { setIsolation(ECalIso, caloIso);  } 
      /// Sets hcal isolation variable
      void setHCalIso(float caloIso)   { setIsolation(HCalIso, caloIso);  }
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
      const IsoDeposit * trackerIsoDeposit() const { return isoDeposit(TrackerIso); }
      /// Return the ecal IsoDeposit
      const IsoDeposit * ecalIsoDeposit()    const { return isoDeposit(ECalIso); }
      /// Return the hcal IsoDeposit
      const IsoDeposit * hcalIsoDeposit()    const { return isoDeposit(HCalIso); }
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
      void trackerIsoDeposit(const IsoDeposit &dep) { setIsoDeposit(TrackerIso, dep); }
      /// Sets ecal IsoDeposit
      void ecalIsoDeposit(const IsoDeposit &dep)    { setIsoDeposit(ECalIso, dep); }
      /// Sets hcal IsoDeposit
      void hcalIsoDeposit(const IsoDeposit &dep)    { setIsoDeposit(HCalIso, dep); }
      /// Sets user-level IsoDeposit
      void userIsoDeposit(const IsoDeposit &dep, uint8_t index=0) { setIsoDeposit(IsolationKeys(UserBaseIso + index), dep); }


    protected:

      // ---- for content embedding ----
      bool embeddedSuperCluster_;
      std::vector<reco::SuperCluster> superCluster_;

      // ---- Isolation and IsoDeposit related datamebers ----
      typedef std::vector<std::pair<IsolationKeys, pat::IsoDeposit> > IsoDepositPairs;
      IsoDepositPairs    isoDeposits_;
      std::vector<float> isolations_;

  };


}

#endif
