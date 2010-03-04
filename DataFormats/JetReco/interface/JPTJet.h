#ifndef JetReco_JPTJet_h
#define JetReco_JPTJet_h

/** \class reco::JPTJet
 *
 * \short Jets made from CaloJets corrected for ZSP and tracks
 *
 * JPTJet represents Jets made from CaloTowers
 * and corrected for tracks
 * in addition to generic Jet parameters it gives
 * reference to the original jet, ZSP scale, associated tracks  
 *
 * \author Olga Kodolova
 *
 ************************************************************/


#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
class JPTJet : public Jet {
 public:
 
  struct Specific {
    Specific () :
         mZSPCor(0),
	 mChargedHadronEnergy (0),	 
	 mNeutralHadronEnergy (0),
	 mChargedEmEnergy (0),
	 mNeutralEmEnergy (0),
         mResponseOfChargedWithEff (0),
         mResponseOfChargedWithoutEff (0),
         mSumPtOfChargedWithEff (0),
         mSumPtOfChargedWithoutEff (0),
         mSumEnergyOfChargedWithEff (0),
         mSumEnergyOfChargedWithoutEff (0),
         R2mom_tr (0),
         Eta2mom_tr (0),
         Phi2mom_tr (0),
         P_out (0),
         Z_ch (0)
    {}
    float mZSPCor;
    edm::RefToBase<reco::Jet> theCaloJetRef;    
    reco::TrackRefVector pions_inVertexInCalo;
    reco::TrackRefVector pions_inVertexOutCalo;
    reco::TrackRefVector pions_OutVertexInCalo;
    reco::TrackRefVector muons_inVertexInCalo;
    reco::TrackRefVector muons_inVertexOutCalo;
    reco::TrackRefVector muons_OutVertexInCalo;
    reco::TrackRefVector elecs_inVertexInCalo;
    reco::TrackRefVector elecs_inVertexOutCalo;
    reco::TrackRefVector elecs_OutVertexInCalo;
    float mChargedHadronEnergy;
    float mNeutralHadronEnergy;
    float mChargedEmEnergy;
    float mNeutralEmEnergy;
    float  mResponseOfChargedWithEff;
    float  mResponseOfChargedWithoutEff;
    float  mSumPtOfChargedWithEff;
    float  mSumPtOfChargedWithoutEff;
    float  mSumEnergyOfChargedWithEff;
    float  mSumEnergyOfChargedWithoutEff;
    float R2mom_tr;
    float Eta2mom_tr;
    float Phi2mom_tr;
    float P_out;
    float Z_ch;
 };
   
  /** Default constructor*/
  JPTJet() {}
  
  /** Constructor from values*/
  JPTJet(const LorentzVector& fP4, const Point& fVertex, const Specific& fSpecific, const Jet::Constituents& fConstituents); 
  
  /** backward compatible, vertex=(0,0,0) */
  JPTJet(const LorentzVector& fP4, const Specific& fSpecific, const Jet::Constituents& fConstituents);
  
  virtual ~JPTJet() {};


  /// chargedHadronEnergy 
  float chargedHadronEnergy () const {return m_specific.mChargedHadronEnergy;}
  ///  chargedHadronEnergyFraction
  float  chargedHadronEnergyFraction () const {return chargedHadronEnergy () / energy ();}
  /// neutralHadronEnergy
  float neutralHadronEnergy () const {return m_specific.mNeutralHadronEnergy;}
  /// neutralHadronEnergyFraction
  float neutralHadronEnergyFraction () const {return neutralHadronEnergy () / energy ();}
  /// chargedEmEnergy
  float chargedEmEnergy () const {return m_specific.mChargedEmEnergy;}
  /// chargedEmEnergyFraction
  float chargedEmEnergyFraction () const {return chargedEmEnergy () / energy ();}
  /// neutralEmEnergy
  float neutralEmEnergy () const {return m_specific.mNeutralEmEnergy;}
  /// neutralEmEnergyFraction
  float neutralEmEnergyFraction () const {return neutralEmEnergy () / energy ();}
  /// chargedMultiplicity
  int chargedMultiplicity () const {return m_specific.muons_inVertexInCalo.size()+m_specific.muons_inVertexOutCalo.size()+
                                           m_specific.pions_inVertexInCalo.size()+m_specific.pions_inVertexOutCalo.size()+
					   m_specific.elecs_inVertexInCalo.size()+m_specific.elecs_inVertexOutCalo.size();}
  /// muonMultiplicity
  int muonMultiplicity () const {return m_specific.muons_inVertexInCalo.size()+m_specific.muons_inVertexOutCalo.size();}
  /// elecMultiplicity
  int elecMultiplicity () const {return m_specific.elecs_inVertexInCalo.size()+m_specific.elecs_inVertexOutCalo.size();}
  /// Tracks   
  const reco::TrackRefVector& getPions_inVertexInCalo() const{return m_specific.pions_inVertexInCalo;}
  const reco::TrackRefVector& getPions_inVertexOutCalo() const{return m_specific.pions_inVertexOutCalo;}
  const reco::TrackRefVector& getPions_OutVertexInCalo() const{return m_specific.pions_OutVertexInCalo;}
  const reco::TrackRefVector& getMuons_inVertexInCalo() const{return m_specific.muons_inVertexInCalo;}
  const reco::TrackRefVector& getMuons_inVertexOutCalo() const{return m_specific.muons_inVertexOutCalo;}
  const reco::TrackRefVector& getMuons_OutVertexInCalo() const{return m_specific.muons_OutVertexInCalo;}
  const reco::TrackRefVector& getElecs_inVertexInCalo() const{return m_specific.elecs_inVertexInCalo;}
  const reco::TrackRefVector& getElecs_inVertexOutCalo() const{return m_specific.elecs_inVertexOutCalo;}
  const reco::TrackRefVector& getElecs_OutVertexInCalo() const{return m_specific.elecs_OutVertexInCalo;}
  

  
  const float& getZSPCor() const {return m_specific.mZSPCor;} 

//  const edm::RefToBase<reco::CaloJet>& getCaloJetRef() {return m_specific.theCaloJetRef;}
  const edm::RefToBase<reco::Jet>& getCaloJetRef() const {return m_specific.theCaloJetRef;}
  /// block accessors
  
  const Specific& getSpecific () const {return m_specific;}
  
  /// Polymorphic clone
  virtual JPTJet* clone () const;
  
  /// Print object in details
  virtual void printJet () const;
  
  virtual std::string print () const;
  
 private:
  /// Polymorphic overlap
  virtual bool overlap( const Candidate & ) const;
  
  //Variables specific to to the JPTJet class
 
  Specific m_specific;
  //reco::CaloJetRef theCaloJetRef;
};

// streamer
 //std::ostream& operator<<(std::ostream& out, const reco::JPTJet& jet);
}
// temporary fix before include_checcker runs globally
#include "DataFormats/JetReco/interface/JPTJetCollection.h" //INCLUDECHECKER:SKIP
#endif
