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
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Common/interface/AtomicPtrCache.h"


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
  class PATPhotonSlimmer;

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
      /// direct access to the seed cluster
      reco::CaloClusterPtr seed() const; 

      //method to access the basic clusters
      const std::vector<reco::CaloCluster>& basicClusters() const { return basicClusters_ ; }
      //method to access the preshower clusters
      const std::vector<reco::CaloCluster>& preshowerClusters() const { return preshowerClusters_ ; }      
      
      //method to access embedded ecal RecHits
      const EcalRecHitCollection * recHits() const { return &recHits_;}      
      
      /// method to store the photon's supercluster internally
      void embedSuperCluster();
      /// method to store the electron's seedcluster internally
      void embedSeedCluster();
      /// method to store the electron's basic clusters
      void embedBasicClusters();
      /// method to store the electron's preshower clusters
      void embedPreshowerClusters();
      /// method to store the RecHits internally - can be called from the PATElectronProducer
      void embedRecHits(const EcalRecHitCollection * rechits); 
      
      
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

      /// get and set PFCluster isolation
      float ecalPFClusterIso() const { return ecalPFClusIso_;}
      float hcalPFClusterIso() const { return hcalPFClusIso_;}
      void setEcalPFClusterIso(float ecalPFClus) { ecalPFClusIso_=ecalPFClus;}
      void setHcalPFClusterIso(float hcalPFClus) { hcalPFClusIso_=hcalPFClus;}

      /// PARTICLE FLOW ISOLATION
      /// Returns the isolation calculated with all the PFCandidates
      float patParticleIso() const { return userIsolation(pat::PfAllParticleIso); }
      /// Returns the isolation calculated with only the charged hadron
      /// PFCandidates
      float chargedHadronIso() const { return reco::Photon::chargedHadronIso(); }
      /// Returns the isolation calculated with only the neutral hadron
      /// PFCandidates
      float neutralHadronIso() const { return reco::Photon::neutralHadronIso(); }
      /// Returns the isolation calculated with only the gamma
      /// PFCandidates
      float photonIso() const { return reco::Photon::photonIso(); }
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
      /// vertex fit method 
      bool passElectronVeto() const { return passElectronVeto_; }
      void setPassElectronVeto( bool flag ) { passElectronVeto_ = flag; }
      //pixel seed to veto electron (not recommended by EGM POG but it seems very efficient)
      bool hasPixelSeed() const { return hasPixelSeed_; }
      void setHasPixelSeed( bool flag ) { hasPixelSeed_ = flag; }

       /// input variables for regression energy corrections
      float seedEnergy() const { return seedEnergy_;}
      void setSeedEnergy( float e ){ seedEnergy_ = e; }

      float eMax() const { return eMax_;}
      void setEMax( float e ){ eMax_ = e;}
      float e2nd() const { return e2nd_;}
      void setE2nd( float e ){ e2nd_ = e;}
      float e3x3() const { return e3x3_;}
      void setE3x3( float e ){ e3x3_ = e;}
      float eTop() const { return eTop_;}
      void setETop( float e ){ eTop_ = e;}
      float eBottom() const { return eBottom_;}
      void setEBottom( float e ){ eBottom_ = e;}
      float eLeft() const { return eLeft_;}
      void setELeft( float e ){ eLeft_ = e;}
      float eRight() const { return eRight_;}
      void setERight( float e ){ eRight_ = e;}
  
      float see() const { return see_;}
      void setSee( float s ){ see_ = s;}
      float spp() const { return spp_;}
      void setSpp( float s ){ spp_ = s;}
      float sep() const { return sep_;}
      void setSep( float s ){ sep_ = s;}

      float maxDR() const { return maxDR_;}
      void setMaxDR( float m ){ maxDR_ = m;}
      float maxDRDPhi() const { return maxDRDPhi_;}
      void setMaxDRDPhi( float m ){ maxDRDPhi_ = m;}
      float maxDRDEta() const { return maxDRDEta_;}
      void setMaxDRDEta( float m ){ maxDRDEta_ = m;}
      float maxDRRawEnergy() const { return maxDRRawEnergy_;}
      void setMaxDRRawEnergy( float m ){ maxDRRawEnergy_ = m;}

      float subClusRawE1() const { return subClusRawE1_;}
      void setSubClusRawE1( float s ){ subClusRawE1_ = s;}
      float subClusRawE2() const { return subClusRawE2_;}
      void setSubClusRawE2( float s ){ subClusRawE2_ = s;}
      float subClusRawE3() const { return subClusRawE3_;}
      void setSubClusRawE3( float s ){ subClusRawE3_ = s;}

      float subClusDPhi1() const { return subClusDPhi1_;}
      void setSubClusDPhi1( float s ){ subClusDPhi1_ = s;}
      float subClusDPhi2() const { return subClusDPhi2_;}
      void setSubClusDPhi2( float s ){ subClusDPhi2_ = s;}
      float subClusDPhi3() const { return subClusDPhi3_;}
      void setSubClusDPhi3( float s ){ subClusDPhi3_ = s;}

      float subClusDEta1() const { return subClusDEta1_;}
      void setSubClusDEta1( float s ){ subClusDEta1_ = s;}
      float subClusDEta2() const { return subClusDEta2_;}
      void setSubClusDEta2( float s ){ subClusDEta2_ = s;}
      float subClusDEta3() const { return subClusDEta3_;}
      void setSubClusDEta3( float s ){ subClusDEta3_ = s;}

      float cryPhi() const { return cryPhi_;}
      void setCryPhi( float c ){ cryPhi_ = c;}
      float cryEta() const { return cryEta_;}
      void setCryEta( float c ){ cryEta_ = c;}

      float iPhi() const { return iPhi_;}
      void setIPhi( float i ){ iPhi_ = i;}
      float iEta() const { return iEta_;}
      void setIEta( float i ){ iEta_ = i;}

      /// pipe operator (introduced to use pat::Photon with PFTopProjectors)
      friend std::ostream& reco::operator<<(std::ostream& out, const pat::Photon& obj);

      /// References to PFCandidates linked to this object (e.g. for isolation vetos or masking before jet reclustering)
      edm::RefVector<pat::PackedCandidateCollection> associatedPackedPFCandidates() const ;
      /// References to PFCandidates linked to this object (e.g. for isolation vetos or masking before jet reclustering)
      template<typename T>
      void setAssociatedPackedPFCandidates(const edm::RefProd<pat::PackedCandidateCollection> & refprod,
                                           T beginIndexItr,
                                           T endIndexItr) {
        packedPFCandidates_ = refprod;
        associatedPackedFCandidateIndices_.clear();
        associatedPackedFCandidateIndices_.insert(associatedPackedFCandidateIndices_.begin(),
                                                  beginIndexItr,
                                                  endIndexItr);
      }
 
      /// get the number of non-null PFCandidates
      size_t numberOfSourceCandidatePtrs() const { return associatedPackedFCandidateIndices_.size(); }
      /// get the source candidate pointer with index i
      reco::CandidatePtr sourceCandidatePtr( size_type i ) const;

      friend class PATPhotonSlimmer;

    protected:

      // ---- for content embedding ----
      bool embeddedSuperCluster_;
      std::vector<reco::SuperCluster> superCluster_;
      /// Place to temporarily store the electron's supercluster after relinking the seed to it
      edm::AtomicPtrCache<std::vector<reco::SuperCluster> > superClusterRelinked_;
      /// Place to store electron's basic clusters internally 
      std::vector<reco::CaloCluster> basicClusters_;
      /// Place to store electron's preshower clusters internally      
      std::vector<reco::CaloCluster> preshowerClusters_;      
      /// True if seed cluster is stored internally
      bool embeddedSeedCluster_;
      /// Place to store electron's seed cluster internally
      std::vector<reco::CaloCluster> seedCluster_;
      /// True if RecHits stored internally
      bool embeddedRecHits_;    
      /// Place to store electron's RecHits internally (5x5 around seed+ all RecHits)
      EcalRecHitCollection recHits_;      
      // ---- photon ID's holder ----
      std::vector<IdPair> photonIDs_;
      // ---- Isolation and IsoDeposit related datamebers ----
      typedef std::vector<std::pair<IsolationKeys, pat::IsoDeposit> > IsoDepositPairs;
      IsoDepositPairs    isoDeposits_;
      std::vector<float> isolations_;

      /// ---- conversion veto ----
      bool passElectronVeto_;
      bool hasPixelSeed_;
      
      /// ---- input variables for regression energy corrections ----
      float seedEnergy_;
      float eMax_;
      float e2nd_;
      float e3x3_;
      float eTop_;
      float eBottom_;
      float eLeft_;
      float eRight_;

      float see_;
      float spp_;
      float sep_;

      float maxDR_;
      float maxDRDPhi_;
      float maxDRDEta_;
      float maxDRRawEnergy_;
   
      float subClusRawE1_;
      float subClusRawE2_;
      float subClusRawE3_;

      float subClusDPhi1_;
      float subClusDPhi2_;
      float subClusDPhi3_;

      float subClusDEta1_;
      float subClusDEta2_;
      float subClusDEta3_;

      float cryEta_;
      float cryPhi_;
      float iEta_;
      float iPhi_;

      float ecalPFClusIso_;
      float hcalPFClusIso_;

      // ---- link to PackedPFCandidates
      edm::RefProd<pat::PackedCandidateCollection> packedPFCandidates_;
      std::vector<uint16_t> associatedPackedFCandidateIndices_;
  };


}

#endif
