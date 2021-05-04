/*! \class TTStubBuilder
* \brief Plugin to load the Stub finding algorithm and produce the
* collection of Stubs that goes in the event content.
* \details After moving from SimDataFormats to DataFormats,
* the template structure of the class was maintained
* in order to accomodate any types other than PixelDigis
* in case there is such a need in the future.
*
* \author Andrew W. Rose
* \author Nicola Pozzobon
* \author Ivan Reid
* \author Ian Tomalin
* \date 2013 - 2020
*
*/

#ifndef L1_TRACK_TRIGGER_STUB_BUILDER_H
#define L1_TRACK_TRIGGER_STUB_BUILDER_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm.h"
#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithmRecord.h"

#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include <memory>
#include <map>
#include <vector>

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

template <typename T>
class TTStubBuilder : public edm::one::EDProducer<edm::one::WatchRuns> {
public:
  /// Constructor
  explicit TTStubBuilder(const edm::ParameterSet& iConfig);

  /// Destructor;
  ~TTStubBuilder() override;

  // TTStub bendOffset has this added to it, if stub truncated by FE, to indicate reason.
  enum FEreject { CBCFailOffset = 500, CICFailOffset = 1000 };

private:
  /// Data members
  edm::ESHandle<TTStubAlgorithm<T>> theStubFindingAlgoHandle;
  edm::EDGetTokenT<edmNew::DetSetVector<TTCluster<T>>> clustersToken;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tGeomToken;
  edm::ESGetToken<TTStubAlgorithm<T>, TTStubAlgorithmRecord> ttStubToken;
  bool ForbidMultipleStubs;

  /// Mandatory methods
  void beginRun(const edm::Run& run, const edm::EventSetup& iSetup) override;
  void endRun(const edm::Run& run, const edm::EventSetup& iSetup) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  /// Sorting method for stubs
  /// NOTE: this must be static!
  static bool SortStubBendPairs(const std::pair<unsigned int, double>& left,
                                const std::pair<unsigned int, double>& right);
  static bool SortStubsBend(const TTStub<T>& left, const TTStub<T>& right);

  /// Fill output cluster & stub collections.
  template <typename TT>
  void fill(edmNew::DetSetVector<TT>& outputEP, const DetId& detId, const std::vector<TT>& inputVec) const {
    /// Create the FastFiller
    typename edmNew::DetSetVector<TT>::FastFiller outputFiller(outputEP, detId);
    outputFiller.resize(inputVec.size());
    std::copy(inputVec.begin(), inputVec.end(), outputFiller.begin());
  }

  /// Update output stubs with Refs to cluster collection that is associated to stubs.
  void updateStubs(const edm::OrphanHandle<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>>& clusterHandle,
                   const edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>& inputEDstubs,
                   edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>& outputEDstubs) const;

  /// FE truncation

  bool applyFE;  // Turn ON (true) or OFF (false) the dynamic FE stub truncation.

  // Tuncation cut-offs
  unsigned int maxStubs_2S;         // CBC chip limit (in stubs/chip/BX)
  unsigned int maxStubs_PS;         // MPA chip limit (in stubs/chip/2BX)
  unsigned int maxStubs_2S_CIC_5;   // 2S 5G chip limit (in stubs/CIC/8BX)
  unsigned int maxStubs_PS_CIC_5;   // PS 5G chip limit (in stubs/CIC/8BX)
  unsigned int maxStubs_PS_CIC_10;  // PS 10G chip limit (in stubs/CIC/8BX)

  // Which modules read by 10Gb/s links instead of 5Gb/s
  // (Unlike TkLayout, CMSSW starts ring count at 1 for the innermost physically present ring in each disk)
  // sviret comment (221217): this info should be made available in conddb at some point
  // (not in TrackerTopology, as modules may switch between 10G & 5G transmission schems during running?)
  unsigned int high_rate_max_ring[5];  //Outermost ring with 10Gb/s link vs disk.
  unsigned int high_rate_max_layer;    // Outermost barrel layer with 10Gb/s link.

  /// Temporary storage for stubs over several events for truncation use.
  int ievt;
  std::unordered_map<int, std::vector<TTStub<Ref_Phase2TrackerDigi_>>> moduleStubs_CIC;
  std::unordered_map<int, int> moduleStubs_MPA;
  std::unordered_map<int, int> moduleStubs_CBC;

};  /// Close class

/*! \brief Implementation of methods
* \details Here, in the header file, the methods which do not depend
* on the specific type <T> that can fit the template.
* Other methods, with type-specific features, are implemented
* in the source file.
*/

/// Constructors
template <typename T>
TTStubBuilder<T>::TTStubBuilder(const edm::ParameterSet& iConfig) {
  clustersToken = consumes<edmNew::DetSetVector<TTCluster<T>>>(iConfig.getParameter<edm::InputTag>("TTClusters"));
  tTopoToken = esConsumes<TrackerTopology, TrackerTopologyRcd>();
  tGeomToken = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>();
  ttStubToken = esConsumes<TTStubAlgorithm<T>, TTStubAlgorithmRecord, edm::Transition::BeginRun>();
  ForbidMultipleStubs = iConfig.getParameter<bool>("OnlyOnePerInputCluster");
  applyFE = iConfig.getParameter<bool>("FEineffs");
  maxStubs_2S = iConfig.getParameter<uint32_t>("CBClimit");
  maxStubs_PS = iConfig.getParameter<uint32_t>("MPAlimit");
  maxStubs_2S_CIC_5 = iConfig.getParameter<uint32_t>("SS5GCIClimit");
  maxStubs_PS_CIC_5 = iConfig.getParameter<uint32_t>("PS5GCIClimit");
  maxStubs_PS_CIC_10 = iConfig.getParameter<uint32_t>("PS10GCIClimit");
  unsigned int tedd1_max10Gring = iConfig.getParameter<uint32_t>("TEDD1Max10GRing");
  unsigned int tedd2_max10Gring = iConfig.getParameter<uint32_t>("TEDD2Max10GRing");
  high_rate_max_layer = iConfig.getParameter<uint32_t>("BarrelMax10GLay");
  // Stubs passing & failing FE chip cuts, plus associated clusters.
  produces<edmNew::DetSetVector<TTCluster<T>>>("ClusterAccepted");
  produces<edmNew::DetSetVector<TTCluster<T>>>("ClusterRejected");
  produces<edmNew::DetSetVector<TTStub<T>>>("StubAccepted");
  produces<edmNew::DetSetVector<TTStub<T>>>("StubRejected");

  high_rate_max_ring[0] = tedd1_max10Gring;
  high_rate_max_ring[1] = tedd1_max10Gring;
  high_rate_max_ring[2] = tedd2_max10Gring;
  high_rate_max_ring[3] = tedd2_max10Gring;
  high_rate_max_ring[4] = tedd2_max10Gring;
}

/// Destructor
template <typename T>
TTStubBuilder<T>::~TTStubBuilder() {}

/// Begin run
template <typename T>
void TTStubBuilder<T>::beginRun(const edm::Run& run, const edm::EventSetup& iSetup) {
  /// Get the stub finding algorithm
  theStubFindingAlgoHandle = iSetup.getHandle(ttStubToken);
  ievt = 0;
  moduleStubs_CIC.clear();
  moduleStubs_MPA.clear();
  moduleStubs_CBC.clear();
}

/// End run
template <typename T>
void TTStubBuilder<T>::endRun(const edm::Run& run, const edm::EventSetup& iSetup) {}

/// Sort routine for stub ordering
template <typename T>
bool TTStubBuilder<T>::SortStubBendPairs(const std::pair<unsigned int, double>& left,
                                         const std::pair<unsigned int, double>& right) {
  return std::abs(left.second) < std::abs(right.second);
}

/// Analogous sorting routine directly from stubs
template <typename T>
bool TTStubBuilder<T>::SortStubsBend(const TTStub<T>& left, const TTStub<T>& right) {
  return std::abs(left.bendFE()) < std::abs(right.bendFE());
}

/// Implement the producer
template <>
void TTStubBuilder<Ref_Phase2TrackerDigi_>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup);

#endif
