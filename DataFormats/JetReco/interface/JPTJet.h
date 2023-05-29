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
      Specific()
          : mChargedHadronEnergy(0),
            mChargedEmEnergy(0),
            mResponseOfChargedWithEff(0),
            mResponseOfChargedWithoutEff(0),
            mSumPtOfChargedWithEff(0),
            mSumPtOfChargedWithoutEff(0),
            mSumEnergyOfChargedWithEff(0),
            mSumEnergyOfChargedWithoutEff(0),
            R2momtr(0),
            Eta2momtr(0),
            Phi2momtr(0),
            Pout(0),
            Zch(0),
            JPTSeed(0) {}
      edm::RefToBase<reco::Jet> theCaloJetRef;
      reco::TrackRefVector pionsInVertexInCalo;
      reco::TrackRefVector pionsInVertexOutCalo;
      reco::TrackRefVector pionsOutVertexInCalo;
      reco::TrackRefVector muonsInVertexInCalo;
      reco::TrackRefVector muonsInVertexOutCalo;
      reco::TrackRefVector muonsOutVertexInCalo;
      reco::TrackRefVector elecsInVertexInCalo;
      reco::TrackRefVector elecsInVertexOutCalo;
      reco::TrackRefVector elecsOutVertexInCalo;
      float mChargedHadronEnergy;
      float mChargedEmEnergy;
      float mResponseOfChargedWithEff;
      float mResponseOfChargedWithoutEff;
      float mSumPtOfChargedWithEff;
      float mSumPtOfChargedWithoutEff;
      float mSumEnergyOfChargedWithEff;
      float mSumEnergyOfChargedWithoutEff;
      float R2momtr;
      float Eta2momtr;
      float Phi2momtr;
      float Pout;
      float Zch;
      int JPTSeed;
    };

    /** Default constructor*/
    JPTJet() {}

    /** Constructor from values*/
    JPTJet(const LorentzVector& fP4,
           const Point& fVertex,
           const Specific& fSpecific,
           const Jet::Constituents& fConstituents);

    /** backward compatible, vertex=(0,0,0) */
    JPTJet(const LorentzVector& fP4, const Specific& fSpecific, const Jet::Constituents& fConstituents);

    ~JPTJet() override{};
    /// chargedHadronEnergy
    float chargedHadronEnergy() const { return mspecific.mChargedHadronEnergy; }
    ///  chargedHadronEnergyFraction
    float chargedHadronEnergyFraction() const { return chargedHadronEnergy() / energy(); }
    /// chargedEmEnergy
    float chargedEmEnergy() const { return mspecific.mChargedEmEnergy; }
    /// chargedEmEnergyFraction
    float chargedEmEnergyFraction() const { return chargedEmEnergy() / energy(); }
    /// chargedMultiplicity
    int chargedMultiplicity() const {
      return mspecific.muonsInVertexInCalo.size() + mspecific.muonsInVertexOutCalo.size() +
             mspecific.pionsInVertexInCalo.size() + mspecific.pionsInVertexOutCalo.size() +
             mspecific.elecsInVertexInCalo.size() + mspecific.elecsInVertexOutCalo.size();
    }
    /// muonMultiplicity
    int muonMultiplicity() const {
      return mspecific.muonsInVertexInCalo.size() + mspecific.muonsInVertexOutCalo.size();
    }
    /// elecMultiplicity
    int elecMultiplicity() const {
      return mspecific.elecsInVertexInCalo.size() + mspecific.elecsInVertexOutCalo.size();
    }
    /// Tracks
    const reco::TrackRefVector& getPionsInVertexInCalo() const { return mspecific.pionsInVertexInCalo; }
    const reco::TrackRefVector& getPionsInVertexOutCalo() const { return mspecific.pionsInVertexOutCalo; }
    const reco::TrackRefVector& getPionsOutVertexInCalo() const { return mspecific.pionsOutVertexInCalo; }
    const reco::TrackRefVector& getMuonsInVertexInCalo() const { return mspecific.muonsInVertexInCalo; }
    const reco::TrackRefVector& getMuonsInVertexOutCalo() const { return mspecific.muonsInVertexOutCalo; }
    const reco::TrackRefVector& getMuonsOutVertexInCalo() const { return mspecific.muonsOutVertexInCalo; }
    const reco::TrackRefVector& getElecsInVertexInCalo() const { return mspecific.elecsInVertexInCalo; }
    const reco::TrackRefVector& getElecsInVertexOutCalo() const { return mspecific.elecsInVertexOutCalo; }
    const reco::TrackRefVector& getElecsOutVertexInCalo() const { return mspecific.elecsOutVertexInCalo; }

    const edm::RefToBase<reco::Jet>& getCaloJetRef() const { return mspecific.theCaloJetRef; }
    /// block accessors

    const Specific& getSpecific() const { return mspecific; }

    /// Polymorphic clone
    JPTJet* clone() const override;

    /// Print object in details
    virtual void printJet() const;

    std::string print() const override;

  private:
    /// Polymorphic overlap
    bool overlap(const Candidate&) const override;

    //Variables specific to to the JPTJet class

    Specific mspecific;
  };

  // streamer
  //std::ostream& operator<<(std::ostream& out, const reco::JPTJet& jet);
}  // namespace reco
// temporary fix before include_checcker runs globally
#include "DataFormats/JetReco/interface/JPTJetCollection.h"  //INCLUDECHECKER:SKIP
#endif
