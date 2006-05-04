#ifndef HLTReco_HLTFilterObject_h
#define HLTReco_HLTFilterObject_h

/** \class HLTFilterObject
 *
 *
 *  If HLT cuts of intermediate or final HLT filters are satisfied,
 *  instances of this class hold the combination of reconstructed
 *  physics objects (e/gamma/mu/jet/MMet...) satisfying the cuts.
 *
 *  This implementation is not completely space-efficient as some
 *  physics object containers may stay empty. However, the big
 *  advantage is that the solution is generic, i.e., works for all
 *  possible HLT filters. Hence we accept the reasonably small
 *  overhead of empty containers.
 *
 *  $Date: 2006/04/26 09:27:44 $
 *  $Revision: 1.1 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/Common/interface/HLTenums.h"
#include "DataFormats/HLTReco/interface/HLTParticle.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include <map>

namespace reco
{
  using namespace std;

  class HLTFilterObject {

    typedef edm::hlt::HLTScalar HLTScalar;

    typedef HLTParticle                           HLTCaloJet;
    typedef HLTParticleWithRef<CaloJetCollection> HLTCaloJetWithRef;
    typedef           edm::Ref<CaloJetCollection>    CaloJetRef;

  private:
    bool accept_;
    map<HLTScalar,float> scalars_;
    vector<HLTCaloJet> jets_;
    // similar for electron/muon/gamma ...

  public:

    HLTFilterObject(): accept_(), scalars_(), jets_() { }

    void setAccept(const bool accept) {accept_=accept;}
    bool getAccept() const { return accept_;}

    void putScalar(const HLTScalar scalar, const float value) {
      scalars_[scalar] = value;
    }

    bool getScalar(const HLTScalar scalar, float& value) const {
      if (scalars_.find(scalar)==scalars_.end()) {
        return false;
      } else {
        value = scalars_.find(scalar)->second;
        return true;
      }
    }

    void putJet(const CaloJetRef& jetref) {
      // Construct our jet from jetref and save it!
      Particle::LorentzVector p4(jetref->px(),jetref->py(),jetref->pz(),jetref->energy());
      HLTParticle             particle(0,p4);
      HLTCaloJet              jet(particle);
      jets_.push_back(jet);
    }

    void putJet(const HLTCaloJetWithRef& jetwithref) {
      HLTCaloJet jet=jetwithref;
      jets_.push_back(jet);
    }

    const vector<HLTCaloJet>& getJets() const {return jets_;}

  };


  class HLTFilterObjectWithRefs {

    typedef edm::hlt::HLTScalar HLTScalar;

    typedef HLTParticleWithRef<CaloJetCollection> HLTCaloJetWithRef;
    typedef           edm::Ref<CaloJetCollection>    CaloJetRef;

  private:
    bool accept_;
    map<HLTScalar,float> scalars_;
    vector<HLTCaloJetWithRef> jets_;
    // similar for electron/muon/gamma ...

  public:

    HLTFilterObjectWithRefs(): accept_(), scalars_(), jets_() { }

    void setAccept(const bool accept) {accept_=accept;}
    bool getAccept() const { return accept_;}

    void putScalar(const HLTScalar scalar, const float value) {
      scalars_[scalar] = value;
    }

    bool getScalar(const HLTScalar scalar, float& value) const {
      if (scalars_.find(scalar)==scalars_.end()) {
        return false;
      } else {
        value = scalars_.find(scalar)->second;
        return true;
      }
    }

    void putJet(const CaloJetRef& jetref) {
      // Construct our jet from jetref and save it!
      Particle::LorentzVector p4(jetref->px(),jetref->py(),jetref->pz(),jetref->energy());
      HLTParticle             particle(0,p4);
      HLTCaloJetWithRef       jet(particle,jetref);
      jets_.push_back(jet);
    }

    const vector<HLTCaloJetWithRef>& getJets() const {return jets_;}

  };
}

#endif
