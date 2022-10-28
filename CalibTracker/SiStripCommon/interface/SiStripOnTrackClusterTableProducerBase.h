#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

class SiStripOnTrackClusterTableProducerBase : public edm::stream::EDProducer<> {
public:
  explicit SiStripOnTrackClusterTableProducerBase(const edm::ParameterSet& params)
      : m_name(params.getParameter<std::string>("name")),
        m_doc(params.existsAs<std::string>("doc") ? params.getParameter<std::string>("doc") : ""),
        m_extension(params.existsAs<bool>("extension") ? params.getParameter<bool>("extension") : true),
        m_tracks_token(consumes<edm::View<reco::Track>>(params.getParameter<edm::InputTag>("Tracks"))),
        m_association_token(consumes<TrajTrackAssociationCollection>(params.getParameter<edm::InputTag>("Tracks"))) {
    produces<nanoaod::FlatTable>();
  }
  ~SiStripOnTrackClusterTableProducerBase() override;

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) final;

  struct OnTrackCluster {
    uint32_t det;
    const SiStripCluster* cluster;
    const Trajectory* traj;
    const reco::Track* track;
    const TrajectoryMeasurement& measurement;
    OnTrackCluster(uint32_t detId,
                   const SiStripCluster* stripCluster,
                   const Trajectory* trajectory,
                   const reco::Track* track_,
                   const TrajectoryMeasurement& measurement_)
        : det{detId}, cluster{stripCluster}, traj{trajectory}, track{track_}, measurement{measurement_} {}
  };

  virtual void fillTable(const std::vector<OnTrackCluster>& clusters,
                         const edm::View<reco::Track>& tracks,
                         nanoaod::FlatTable* table,
                         const edm::EventSetup& iSetup) = 0;

  template <typename VALUES>
  static void addColumn(nanoaod::FlatTable* table, const std::string& name, VALUES&& values, const std::string& doc) {
    using value_type = typename std::remove_reference<VALUES>::type::value_type;
    table->template addColumn<value_type>(name, values, doc);
  }

private:
  const std::string m_name;
  const std::string m_doc;
  bool m_extension;

  const edm::EDGetTokenT<edm::View<reco::Track>> m_tracks_token;
  const edm::EDGetTokenT<TrajTrackAssociationCollection> m_association_token;
};
