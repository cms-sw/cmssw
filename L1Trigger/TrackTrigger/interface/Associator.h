#ifndef L1Trigger_TrackTrigger_Associator_h
#define L1Trigger_TrackTrigger_Associator_h

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "SimDataFormats/Associations/interface/TTTypes.h"
#include "SimDataFormats/Associations/interface/StubAssociation.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"

#include <vector>
#include <map>
#include <utility>
#include <set>

namespace tt {

  /*! \class  tt::Associator
   *  \brief  Class to associate TrackingParticles with TTStubs and vice versa.
   *  \author Thomas Schuh
   *  \date   2025, Aug
   */
  class Associator {
  public:
    // configuration
    struct Config {
      // required number of associated stub layers to a TP to consider it reconstruct-able
      int minLayers_;
      // required number of associated ps stub layers to a TP to consider it reconstruct-able
      int minLayersPS_;
      // required number of layers a found track has to have in common with a TP to consider it matched
      int minLayersGood_;
      // required number of ps layers a found track has to have in common with a TP to consider it matched
      int minLayersGoodPS_;
      // max number of unassociated 2S stubs allowed to still associate TTTrack with TP
      int maxLayersBad_;
      // max number of unassociated PS stubs allowed to still associate TTTrack with TP
      int maxLayersBadPS_;
    };
    // default constructor
    Associator() {}
    // proper constructor
    Associator(const Config& config, const Setup* setup) : config_(config), setup_(setup) {}
    // destructor
    ~Associator() = default;
    // stores Association maps
    void consume(const StubAssociation& sa) { sa_ = &sa; }
    // checks if stub collection is considered forming a reconstructable track
    bool reconstructable(const std::vector<TTStubRef>& ttStubRefs) const;
    // Get all TPs that are matched to these stubs in at least 'tpMinLayers' layers and 'tpMinLayersPS' ps layers
    std::vector<TPPtr> associate(const std::vector<TTStubRef>& ttStubRefs) const;
    // Get all TPs that are matched to these stubs in at least 'tpMinLayers' layers and 'tpMinLayersPS' ps layers with not more then 'tpMaxBadStubs2S' not associated 2S stubs and not more then 'tpMaxBadStubsPS' associated PS stubs
    std::vector<TPPtr> associateFinal(const std::vector<TTStubRef>& ttStubRefs) const;
    // returns map containing TTStubRef and their associated collection of TPPtrs
    const std::map<TTStubRef, std::vector<TPPtr>>& getTTStubToTrackingParticlesMap() const {
      return sa_->getTTStubToTrackingParticlesMap();
    }
    // returns map containing TPPtr and their associated collection of TTStubRefs
    const std::map<TPPtr, std::vector<TTStubRef>>& getTrackingParticleToTTStubsMap() const {
      return sa_->getTrackingParticleToTTStubsMap();
    }
    // returns collection of TPPtrs associated to given TTstubRef
    const std::vector<TPPtr>& findTrackingParticlePtrs(const TTStubRef& ttStubRef) const {
      return sa_->findTrackingParticlePtrs(ttStubRef);
    }
    // returns collection of TTStubRefs associated to given TPPtr
    const std::vector<TTStubRef>& findTTStubRefs(const TPPtr& tpPtr) const { return sa_->findTTStubRefs(tpPtr); }
    // total number of stubs associated with TPs
    int numStubs() const { return sa_->numStubs(); }
    // total number of TPs associated with stubs
    int numTPs() const { return sa_->numTPs(); }
    // returns primary TP
    TPPtr getPrimaryTP(const TPPtr&) const;

  private:
    // configuration
    Config config_;
    // stores, calculates and provides run-time constants
    const Setup* setup_ = nullptr;
    // stores association maps
    const StubAssociation* sa_ = nullptr;
  };

}  // namespace tt

#endif
