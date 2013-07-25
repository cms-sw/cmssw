//
// $Id: Lepton.h,v 1.24 2011/09/29 05:19:08 cbern Exp $
//

#ifndef DataFormats_PatCandidates_Lepton_h
#define DataFormats_PatCandidates_Lepton_h

/**
  \class    pat::Lepton Lepton.h "DataFormats/PatCandidates/interface/Lepton.h"
  \brief    Analysis-level lepton class

   Lepton implements the analysis-level charged lepton class within the 'pat'
   namespace. It currently provides the link to the generated lepton and
   the isolation information.

   Please post comments and questions to the Physics Tools hypernews:
   https://hypernews.cern.ch/HyperNews/CMS/get/physTools.html

  \author   Steven Lowette, Giovanni Petrucciani, Frederic Ronga
  \version  $Id: Lepton.h,v 1.24 2011/09/29 05:19:08 cbern Exp $
*/

#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"
#include "DataFormats/PatCandidates/interface/Isolation.h"


namespace pat {


  template <class LeptonType>
  class Lepton : public PATObject<LeptonType> {

    public:

      Lepton();
      Lepton(const LeptonType & aLepton);
      Lepton(const edm::RefToBase<LeptonType> & aLeptonRef);
      Lepton(const edm::Ptr<LeptonType> & aLeptonRef);
      virtual ~Lepton();

      virtual Lepton<LeptonType> * clone() const { return new Lepton<LeptonType>(*this); }

      const reco::GenParticle * genLepton() const { return PATObject<LeptonType>::genParticle(); }

      void setGenLepton(const reco::GenParticleRef & gl, bool embed=false) { PATObject<LeptonType>::setGenParticleRef(gl, embed); }

      //============ BEGIN ISOLATION BLOCK =====      
      /// Returns the isolation variable for a specific key (or 
      /// pseudo-key like CaloIso), or -1.0 if not available
      float userIsolation(IsolationKeys key) const { 
          if (key >= 0) {
	    //if (key >= isolations_.size()) throw cms::Excepton("Missing Data") 
	    //<< "Isolation corresponding to key " 
	    //<< key << " was not stored for this particle.";
              if (size_t(key) >= isolations_.size()) return -1.0;
              return isolations_[key];
          } else switch (key) {
	  case pat::CaloIso:  
	    //if (isolations_.size() <= pat::HcalIso) throw cms::Excepton("Missing Data") 
	    //<< "CaloIso Isolation was not stored for this particle.";
	    if (isolations_.size() <= pat::HcalIso) return -1.0; 
	    return isolations_[pat::EcalIso] + isolations_[pat::HcalIso];
	  default:
	    return -1.0;
	    //throw cms::Excepton("Missing Data") << "Isolation corresponding to key " 
	    //<< key << " was not stored for this particle.";
          }
      }
      /// Returns the isolation variable for string type function arguments
      /// (to be used with the cut-string parser);
      /// the possible values of the strings are the enums defined in
      /// DataFormats/PatCandidates/interface/Isolation.h
      float userIsolation(const std::string& key) const {
	// remove leading namespace specifier
	std::string prunedKey = ( key.find("pat::") == 0 ) ? std::string(key, 5) : key;
	if ( prunedKey == "TrackIso" ) return userIsolation(pat::TrackIso);
	if ( prunedKey == "EcalIso" ) return userIsolation(pat::EcalIso);
	if ( prunedKey == "HcalIso" ) return userIsolation(pat::HcalIso);
	if ( prunedKey == "PfAllParticleIso" ) return userIsolation(pat::PfAllParticleIso);
	if ( prunedKey == "PfChargedHadronIso" ) return userIsolation(pat::PfChargedHadronIso);
	if ( prunedKey == "PfNeutralHadronIso" ) return userIsolation(pat::PfNeutralHadronIso);
	if ( prunedKey == "PfGammaIso" ) return userIsolation(pat::PfGammaIso);
	if ( prunedKey == "User1Iso" ) return userIsolation(pat::User1Iso);
	if ( prunedKey == "User2Iso" ) return userIsolation(pat::User2Iso);
	if ( prunedKey == "User3Iso" ) return userIsolation(pat::User3Iso);
	if ( prunedKey == "User4Iso" ) return userIsolation(pat::User4Iso);
	if ( prunedKey == "User5Iso" ) return userIsolation(pat::User5Iso);
	if ( prunedKey == "UserBaseIso" ) return userIsolation(pat::UserBaseIso);
	if ( prunedKey == "CaloIso" ) return userIsolation(pat::CaloIso);
	if ( prunedKey == "PfPUChargedHadronIso" ) return userIsolation(pat::PfPUChargedHadronIso);
	//throw cms::Excepton("Missing Data")
	//<< "Isolation corresponding to key " 
	//<< key << " was not stored for this particle.";
	return -1.0;
      }
      /// Sets the userIsolation variable for a specific key.
      /// Note that you can't set isolation for a pseudo-key 
      /// like CaloIso
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

      // ---- specific getters ----
      /// Returns the tracker isolation variable that was stored in this 
      /// object when produced, or -1.0 if there is none (overloaded if
      /// specific isolation functions are available from the derived 
      /// objects)
      float trackIso() const { return userIsolation(pat::TrackIso); }
      /// Returns the sum of ecal and hcal isolation variable that were 
      /// stored in this object when produced, or -1.0 if at least one 
      /// is missing (overloaded if specific isolation functions are 
      /// available from the derived objects) 
      float caloIso()  const { return userIsolation(pat::CaloIso); }
      /// Returns the ecal isolation variable that was stored in this 
      /// object when produced, or -1.0 if there is none (overloaded 
      /// if specific isolation functions are available from the 
      /// derived objects)
      float ecalIso()  const { return userIsolation(pat::EcalIso); }
      /// Returns the hcal isolation variable that was stored in this 
      /// object when produced, or -1.0 if there is none (overloaded 
      /// if specific isolation functions are available from the 
      /// derived objects)
      float hcalIso()  const { return userIsolation(pat::HcalIso); }

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
      /// Returns the user defined isolation variable #index that was 
      /// stored in this object when produced, or -1.0 if there is none
      float userIso(uint8_t index=0)  const { return userIsolation(IsolationKeys(UserBaseIso + index)); }

      // ---- specific setters ----
      /// Sets tracker isolation variable
      void setTrackIso(float trackIso) { setIsolation(pat::TrackIso, trackIso); }
      /// Sets ecal isolation variable
      void setEcalIso(float caloIso)   { setIsolation(pat::EcalIso, caloIso);  } 
      /// Sets hcal isolation variable
      void setHcalIso(float caloIso)   { setIsolation(pat::HcalIso, caloIso);  }
      /// Sets user isolation variable #index
      void setUserIso(float value, uint8_t index=0)  { setIsolation(IsolationKeys(UserBaseIso + index), value); }


      //============ BEGIN ISODEPOSIT BLOCK =====
      /// Returns the IsoDeposit associated with some key, or a null pointer if it is not available
      const IsoDeposit * isoDeposit(IsolationKeys key) const {
          for (IsoDepositPairs::const_iterator it = isoDeposits_.begin(), ed = isoDeposits_.end(); 
                  it != ed; ++it) 
          {
              if (it->first == key) return & it->second;
          }
          return 0;
      } 

      /// Sets the IsoDeposit associated with some key; if it is already existent, it is overwritten.
      void setIsoDeposit(IsolationKeys key, const IsoDeposit &dep) {
          IsoDepositPairs::iterator it = isoDeposits_.begin(), ed = isoDeposits_.end();
          for (; it != ed; ++it) {
              if (it->first == key) { it->second = dep; return; }
          }
          isoDeposits_.push_back(std::make_pair(key,dep));
      } 

      // ---- specific getters ----
      const IsoDeposit * trackIsoDeposit() const { return isoDeposit(pat::TrackIso); }
      const IsoDeposit * ecalIsoDeposit()  const { return isoDeposit(pat::EcalIso ); }
      const IsoDeposit * hcalIsoDeposit()  const { return isoDeposit(pat::HcalIso ); }
      const IsoDeposit * userIsoDeposit(uint8_t index=0) const { return isoDeposit(IsolationKeys(UserBaseIso + index)); }

      // ---- specific setters ----
      void trackIsoDeposit(const IsoDeposit &dep) { setIsoDeposit(pat::TrackIso,dep); }
      void ecalIsoDeposit(const IsoDeposit &dep)  { setIsoDeposit(pat::EcalIso, dep); }
      void hcalIsoDeposit(const IsoDeposit &dep)  { setIsoDeposit(pat::HcalIso, dep); }
      void userIsoDeposit(const IsoDeposit &dep, uint8_t index=0) { setIsoDeposit(IsolationKeys(UserBaseIso + index), dep); }


    protected:
      // --- Isolation and IsoDeposit related datamebers ---
      typedef std::vector<std::pair<IsolationKeys, pat::IsoDeposit> > IsoDepositPairs;
      IsoDepositPairs    isoDeposits_;
      std::vector<float> isolations_;
  };


  /// default constructor
  template <class LeptonType>
  Lepton<LeptonType>::Lepton() :
    PATObject<LeptonType>(LeptonType()) {
    // no common constructor, so initialize the candidate manually
    this->setCharge(0);
    this->setP4(reco::Particle::LorentzVector(0, 0, 0, 0));
    this->setVertex(reco::Particle::Point(0, 0, 0));
  }


  /// constructor from LeptonType
  template <class LeptonType>
  Lepton<LeptonType>::Lepton(const LeptonType & aLepton) :
    PATObject<LeptonType>(aLepton) {
  }


  /// constructor from ref to LeptonType
  template <class LeptonType>
  Lepton<LeptonType>::Lepton(const edm::RefToBase<LeptonType> & aLeptonRef) :
    PATObject<LeptonType>(aLeptonRef) {
  }


  /// constructor from ref to LeptonType
  template <class LeptonType>
  Lepton<LeptonType>::Lepton(const edm::Ptr<LeptonType> & aLeptonRef) :
    PATObject<LeptonType>(aLeptonRef) {
  }


  /// destructor
  template <class LeptonType>
  Lepton<LeptonType>::~Lepton() {
  }
}

#endif
