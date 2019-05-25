#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"

#include <random>

namespace {
  template <typename T, size_t N>
  std::array<T, N+1> makeLogBins(const double min, const double max) {
    const double minLog10 = std::log10(min);
    const double maxLog10 = std::log10(max);
    const double width = (maxLog10-minLog10)/N;
    std::array<T, N+1> ret;
    ret[0] = std::pow(10, minLog10);
    const double mult = std::pow(10, width);
    for ( size_t i=1; i<=N; ++i) {
      ret[i] = ret[i-1]*mult;
    }
    return ret;
  }

  template <typename T>
  T sqr(T val) {
    return val*val;
  }
}

class PrimaryVertexResolution: public DQMEDAnalyzer{
 public:
  PrimaryVertexResolution(const edm::ParameterSet& iConfig);
  ~PrimaryVertexResolution() override;

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup& ) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  std::vector<reco::TransientTrack> sortTracksByPt(const reco::Vertex& thePV,
                                                   const TransientTrackBuilder& ttBuilder,
                                                   const reco::BeamSpot& beamspot);

  edm::EDGetTokenT<reco::VertexCollection> vertexSrc_;
  edm::EDGetTokenT<reco::BeamSpot> beamspotSrc_;
  edm::EDGetTokenT<LumiScalersCollection> lumiScalersSrc_;
  std::string rootFolder_;
  std::string transientTrackBuilder_;

  AdaptiveVertexFitter fitter_;

  struct BinningY {
    explicit BinningY(const edm::ParameterSet& iConfig):
      maxResol_(iConfig.getUntrackedParameter<double>("maxResol")),
      binsResol_(iConfig.getUntrackedParameter<int>("binsResol")),
      maxPull_(iConfig.getUntrackedParameter<double>("maxPull")),
      binsPull_(iConfig.getUntrackedParameter<int>("binsPull"))
    {}

    const double maxResol_;
    const int binsResol_;
    const double maxPull_;
    const int binsPull_;
  };

  struct BinningX {
    explicit BinningX(const edm::ParameterSet& iConfig):
      minNtracks_(iConfig.getUntrackedParameter<double>("minNtracks")),
      maxNtracks_(iConfig.getUntrackedParameter<double>("maxNtracks")),
      binsNtracks_(iConfig.getUntrackedParameter<int>("binsNtracks")),
      minNvertices_(iConfig.getUntrackedParameter<double>("minNvertices")),
      maxNvertices_(iConfig.getUntrackedParameter<double>("maxNvertices")),
      binsNvertices_(iConfig.getUntrackedParameter<int>("binsNvertices")),
      maxXY_(iConfig.getUntrackedParameter<double>("maxXY")),
      binsXY_(iConfig.getUntrackedParameter<int>("binsXY")),
      maxZ_(iConfig.getUntrackedParameter<double>("maxZ")),
      binsZ_(iConfig.getUntrackedParameter<int>("binsZ")),
      minPt_(iConfig.getUntrackedParameter<double>("minPt")),
      maxPt_(iConfig.getUntrackedParameter<double>("maxPt")),
      minLumi_(iConfig.getUntrackedParameter<double>("minLumi")),
      maxLumi_(iConfig.getUntrackedParameter<double>("maxLumi"))
    {}

    const int minNtracks_;
    const int maxNtracks_;
    const int binsNtracks_;
    const int minNvertices_;
    const int maxNvertices_;
    const int binsNvertices_;
    const double maxXY_;
    const int binsXY_;
    const double maxZ_;
    const int binsZ_;
    const double minPt_;
    const double maxPt_;
    const double minLumi_;
    const double maxLumi_;
  };

  class Resolution {
  public:
    Resolution(const reco::Vertex& vertex1, const reco::Vertex& vertex2) {
      const double diffx = vertex1.x() - vertex2.x();
      const double diffy = vertex1.y() - vertex2.y();
      const double diffz = vertex1.z() - vertex2.z();

      // Take into account the need to divide by sqrt(2) already in
      // the filling so that we can use DQMGenericClient for the
      // gaussian fits
      const double invSqrt2 = 1./std::sqrt(2.);
      resx_ = diffx * invSqrt2;
      resy_ = diffy * invSqrt2;
      resz_ = diffz * invSqrt2;

      pullx_ = diffx / std::sqrt(sqr(vertex1.xError()) + sqr(vertex2.xError()));
      pully_ = diffy / std::sqrt(sqr(vertex1.yError()) + sqr(vertex2.yError()));
      pullz_ = diffz / std::sqrt(sqr(vertex1.zError()) + sqr(vertex2.zError()));

      avgx_ = (vertex1.x()+vertex2.x())/2.;
      avgy_ = (vertex1.y()+vertex2.y())/2.;
      avgz_ = (vertex1.z()+vertex2.z())/2.;

    }

    double resx() const { return resx_; }
    double resy() const { return resy_; }
    double resz() const { return resz_; }

    double pullx() const { return pullx_; }
    double pully() const { return pully_; }
    double pullz() const { return pullz_; }

    double avgx() const { return avgx_; }
    double avgy() const { return avgy_; }
    double avgz() const { return avgz_; }

  private:
    double resx_;
    double resy_;
    double resz_;
    double pullx_;
    double pully_;
    double pullz_;
    double avgx_; 
    double avgy_; 
    double avgz_; 
  };

  class DiffPlots {
  public:
    explicit DiffPlots(const std::string& postfix, const BinningY& binY):
      postfix_(postfix),
      binningY_(binY)
    {}

    template <typename T>
    void bookLogX(DQMStore::IBooker& iBooker, const T& binArray) {
      book(iBooker, binArray.size()-1, binArray.front(), binArray.back());
      setLogX(binArray.size()-1, binArray.data());
    }

    template <typename ...Args>
    void book(DQMStore::IBooker& iBooker, Args&&... args) {
      const auto binsResol = binningY_.binsResol_;
      const auto maxResol = binningY_.maxResol_;
      hDiffX_ = iBooker.book2D("res_x_vs_"+postfix_, "Resolution of X vs. "+postfix_, std::forward<Args>(args)..., binsResol,-maxResol,maxResol);
      hDiffY_ = iBooker.book2D("res_y_vs_"+postfix_, "Resolution of Y vs. "+postfix_, std::forward<Args>(args)..., binsResol,-maxResol,maxResol);
      hDiffZ_ = iBooker.book2D("res_z_vs_"+postfix_, "Resolution of Z vs. "+postfix_, std::forward<Args>(args)..., binsResol,-maxResol,maxResol);

      const auto binsPull = binningY_.binsPull_;
      const auto maxPull = binningY_.maxPull_;
      hPullX_ = iBooker.book2D("pull_x_vs_"+postfix_, "Pull of X vs. "+postfix_, std::forward<Args>(args)..., binsPull,-maxPull,maxPull);
      hPullY_ = iBooker.book2D("pull_y_vs_"+postfix_, "Pull of Y vs. "+postfix_, std::forward<Args>(args)..., binsPull,-maxPull,maxPull);
      hPullZ_ = iBooker.book2D("pull_z_vs_"+postfix_, "Pull of Z vs. "+postfix_, std::forward<Args>(args)..., binsPull,-maxPull,maxPull);
    }
    template <typename ...Args>
    void setLogX(Args&&... args) {
      hDiffX_->getTH2F()->GetXaxis()->Set(std::forward<Args>(args)...);
      hDiffY_->getTH2F()->GetXaxis()->Set(std::forward<Args>(args)...);
      hDiffZ_->getTH2F()->GetXaxis()->Set(std::forward<Args>(args)...);

      hPullX_->getTH2F()->GetXaxis()->Set(std::forward<Args>(args)...);
      hPullY_->getTH2F()->GetXaxis()->Set(std::forward<Args>(args)...);
      hPullZ_->getTH2F()->GetXaxis()->Set(std::forward<Args>(args)...);
    }

    template <typename T>
    void fill(const Resolution& res, const T ref) {
      hDiffX_->Fill(ref, res.resx());
      hDiffY_->Fill(ref, res.resy());
      hDiffZ_->Fill(ref, res.resz());
      hPullX_->Fill(ref, res.pullx());
      hPullY_->Fill(ref, res.pully());
      hPullZ_->Fill(ref, res.pullz());
    }

  private:
    std::string postfix_;
    const BinningY& binningY_;
    MonitorElement *hDiffX_ = nullptr;
    MonitorElement *hDiffY_ = nullptr;
    MonitorElement *hDiffZ_ = nullptr;
    MonitorElement *hPullX_ = nullptr;
    MonitorElement *hPullY_ = nullptr;
    MonitorElement *hPullZ_ = nullptr;
  };

  class Plots {
  public:
    Plots(const BinningX& binX, const BinningY& binY):
      binningX_(binX),
      binningY_(binY),
      hDiff_Ntracks_("ntracks", binY),
      hDiff_sumPt_("sumpt", binY),
      hDiff_Nvertices_("nvertices", binY),
      hDiff_X_("X",binY),
      hDiff_Y_("Y",binY),
      hDiff_Z_("Z",binY),
      hDiff_instLumiScal_("instLumiScal", binY)
    {}

    void book(DQMStore::IBooker& iBooker) {
      const auto binsResol = binningY_.binsResol_;
      const auto maxResol = binningY_.maxResol_;
      hDiffX_ = iBooker.book1D("res_x", "Resolution of X", binsResol, -maxResol, maxResol);
      hDiffY_ = iBooker.book1D("res_y", "Resolution of Y", binsResol, -maxResol, maxResol);
      hDiffZ_ = iBooker.book1D("res_z", "Resolution of Z", binsResol, -maxResol, maxResol);

      const auto binsPull = binningY_.binsPull_;
      const auto maxPull = binningY_.maxPull_;
      hPullX_ = iBooker.book1D(+"pull_x", "Pull of X", binsPull, -maxPull, maxPull);
      hPullY_ = iBooker.book1D(+"pull_y", "Pull of Y", binsPull, -maxPull, maxPull);
      hPullZ_ = iBooker.book1D(+"pull_z", "Pull of Z", binsPull, -maxPull, maxPull);

      hDiff_Ntracks_.book(iBooker, binningX_.binsNtracks_, binningX_.minNtracks_, binningX_.maxNtracks_);
      hDiff_Nvertices_.book(iBooker, binningX_.binsNvertices_, binningX_.minNvertices_, binningX_.maxNvertices_);
      hDiff_X_.book(iBooker, binningX_.binsXY_,-binningX_.maxXY_,binningX_.maxXY_);
      hDiff_Y_.book(iBooker, binningX_.binsXY_,-binningX_.maxXY_,binningX_.maxXY_);
      hDiff_Z_.book(iBooker, binningX_.binsZ_,-binningX_.maxZ_,binningX_.maxZ_);

      constexpr int binsPt = 30;
      hDiff_sumPt_.bookLogX(iBooker, makeLogBins<float, binsPt>(binningX_.minPt_, binningX_.maxPt_));

      constexpr int binsLumi = 100;
      hDiff_instLumiScal_.bookLogX(iBooker, makeLogBins<float, binsLumi>(binningX_.minLumi_, binningX_.maxLumi_));
    }

    void calculateAndFillResolution(const std::vector<reco::TransientTrack>& tracks,
                                    size_t nvertices,
                                    const LumiScalersCollection& lumiScalers,
                                    std::mt19937& engine,
                                    AdaptiveVertexFitter& fitter);

  private:
    const BinningX& binningX_;
    const BinningY& binningY_;

    MonitorElement *hDiffX_ = nullptr;
    MonitorElement *hDiffY_ = nullptr;
    MonitorElement *hDiffZ_ = nullptr;
    MonitorElement *hPullX_ = nullptr;
    MonitorElement *hPullY_ = nullptr;
    MonitorElement *hPullZ_ = nullptr;

    DiffPlots hDiff_Ntracks_;
    DiffPlots hDiff_sumPt_;
    DiffPlots hDiff_Nvertices_;
    DiffPlots hDiff_X_;
    DiffPlots hDiff_Y_;
    DiffPlots hDiff_Z_;
    DiffPlots hDiff_instLumiScal_;
  };

  // Binning
  BinningX binningX_;
  BinningY binningY_;

  // Histograms
  Plots hPV_;
  Plots hOtherV_;

  std::mt19937 engine_;
};

PrimaryVertexResolution::PrimaryVertexResolution(const edm::ParameterSet& iConfig):
  vertexSrc_(consumes<reco::VertexCollection>(iConfig.getUntrackedParameter<edm::InputTag>("vertexSrc"))),
  beamspotSrc_(consumes<reco::BeamSpot>(iConfig.getUntrackedParameter<edm::InputTag>("beamspotSrc"))),
  lumiScalersSrc_(consumes<LumiScalersCollection>(iConfig.getUntrackedParameter<edm::InputTag>("lumiScalersSrc"))),
  rootFolder_(iConfig.getUntrackedParameter<std::string>("rootFolder")),
  transientTrackBuilder_(iConfig.getUntrackedParameter<std::string>("transientTrackBuilder")),
  binningX_(iConfig),
  binningY_(iConfig),
  hPV_(binningX_, binningY_),
  hOtherV_(binningX_, binningY_)
{}

PrimaryVertexResolution::~PrimaryVertexResolution() {}

void PrimaryVertexResolution::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<edm::InputTag>("vertexSrc", edm::InputTag("trackingDQMgoodOfflinePrimaryVertices"));
  desc.addUntracked<edm::InputTag>("beamspotSrc", edm::InputTag("offlineBeamSpot"));
  desc.addUntracked<edm::InputTag>("lumiScalersSrc", edm::InputTag("scalersRawToDigi"));
  desc.addUntracked<std::string>("rootFolder", "OfflinePV/Resolution");
  desc.addUntracked<std::string>("transientTrackBuilder", "TransientTrackBuilder");

  // Y axes
  desc.addUntracked<double>("maxResol", 0.02);
  desc.addUntracked<int>("binsResol", 100);

  desc.addUntracked<double>("maxPull", 5);
  desc.addUntracked<int>("binsPull", 100);

  // X axes
  desc.addUntracked<double>("minNtracks", -0.5);
  desc.addUntracked<double>("maxNtracks", 119.5);
  desc.addUntracked<int>("binsNtracks", 60);

  desc.addUntracked<double>("minNvertices", -0.5);
  desc.addUntracked<double>("maxNvertices", 199.5);
  desc.addUntracked<int>("binsNvertices", 100);

  desc.addUntracked<double>("maxXY",  0.15);
  desc.addUntracked<int>("binsXY", 100);

  desc.addUntracked<double>("maxZ", 30.);
  desc.addUntracked<int>("binsZ", 100);

  desc.addUntracked<double>("minPt", 1);
  desc.addUntracked<double>("maxPt", 1e3);

  desc.addUntracked<double>("minLumi", 200.);
  desc.addUntracked<double>("maxLumi", 20000.);

  descriptions.add("primaryVertexResolution", desc);
}

void PrimaryVertexResolution::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const&, edm::EventSetup const&) {
  iBooker.setCurrentFolder(rootFolder_+"/PV");
  hPV_.book(iBooker);

  iBooker.setCurrentFolder(rootFolder_+"/OtherV");
  hOtherV_.book(iBooker);
}

void PrimaryVertexResolution::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<reco::VertexCollection> hvertices;
  iEvent.getByToken(vertexSrc_, hvertices);
  const reco::VertexCollection& vertices = *hvertices;
  if(vertices.empty())
    return;

  edm::Handle<reco::BeamSpot> hbeamspot;
  iEvent.getByToken(beamspotSrc_, hbeamspot);
  const reco::BeamSpot& beamspot = *hbeamspot;

  edm::Handle<LumiScalersCollection> hscalers;
  iEvent.getByToken(lumiScalersSrc_, hscalers);
  const LumiScalersCollection& lumiScalers = *hscalers;

  edm::ESHandle<TransientTrackBuilder> ttBuilderHandle;
  iSetup.get<TransientTrackRecord>().get(transientTrackBuilder_, ttBuilderHandle);
  const TransientTrackBuilder& ttBuilder = *ttBuilderHandle;

  // deterministic seed from the event number
  // should not bias the result as the event number is already
  // assigned randomly-enough
  engine_.seed( iEvent.id().event() + (iEvent.id().luminosityBlock()<<10) + (iEvent.id().run()<<20) );

  // The PV
  auto iPV = cbegin(vertices);
  const reco::Vertex& thePV = *iPV;
  const auto nvertices = vertices.size();
  if(thePV.tracksSize() >= 4) {
    auto sortedTracks = sortTracksByPt(thePV, ttBuilder, beamspot);
    hPV_.calculateAndFillResolution(sortedTracks, nvertices, lumiScalers, engine_, fitter_);
  }
  ++iPV;

  // Other vertices
  for(auto endPV = cend(vertices); iPV != endPV; ++iPV) {
    if(iPV->tracksSize() >= 4) {
      auto sortedTracks = sortTracksByPt(*iPV, ttBuilder, beamspot);
      hOtherV_.calculateAndFillResolution(sortedTracks, nvertices, lumiScalers, engine_, fitter_);
    }
  }
}

std::vector<reco::TransientTrack> PrimaryVertexResolution::sortTracksByPt(const reco::Vertex& thePV,
                                                                          const TransientTrackBuilder& ttBuilder,
                                                                          const reco::BeamSpot& beamspot) {
  std::vector<const reco::Track *> sortedTracks;
  sortedTracks.reserve(thePV.tracksSize());
  std::transform(thePV.tracks_begin(), thePV.tracks_end(), std::back_inserter(sortedTracks), [](const reco::TrackBaseRef& ref) {
      return &(*ref);
    });
  std::sort(sortedTracks.begin(), sortedTracks.end(), [](const reco::Track *a, const reco::Track *b) {
      return a->pt() > b->pt();
    });

  std::vector<reco::TransientTrack> ttracks;
  ttracks.reserve(sortedTracks.size());
  std::transform(sortedTracks.begin(), sortedTracks.end(), std::back_inserter(ttracks), [&](const reco::Track *track) {
      auto tt =  ttBuilder.build(track);
      tt.setBeamSpot(beamspot);
      return tt;
    });
  return ttracks;
}

void PrimaryVertexResolution::Plots::calculateAndFillResolution(const std::vector<reco::TransientTrack>& tracks,
                                                                size_t nvertices,
                                                                const LumiScalersCollection& lumiScalers,
                                                                std::mt19937& engine,
                                                                AdaptiveVertexFitter& fitter) {
  const size_t end = tracks.size()%2 == 0 ? tracks.size() : tracks.size()-1;

  std::vector<reco::TransientTrack> set1, set2;
  set1.reserve(end/2); set2.reserve(end/2);

  auto dis = std::uniform_int_distribution<>(0, 1); // [0, 1]

  double sumpt1=0, sumpt2=0;
  for(size_t i=0; i<end; i += 2) {
    const size_t set1_i = dis(engine);
    const size_t set2_i = 1-set1_i;

    set1.push_back(tracks[i+set1_i]);
    set2.push_back(tracks[i+set2_i]);

    sumpt1 += set1.back().track().pt();
    sumpt2 += set2.back().track().pt();
  }

  // For resolution we only fit
  TransientVertex vertex1 = fitter.vertex(set1);
  TransientVertex vertex2 = fitter.vertex(set2);

  Resolution res(vertex1, vertex2);
  hDiffX_->Fill(res.resx());
  hDiffY_->Fill(res.resy());
  hDiffZ_->Fill(res.resz());
  hPullX_->Fill(res.pullx());
  hPullY_->Fill(res.pully());
  hPullZ_->Fill(res.pullz());

  hDiff_Ntracks_.fill(res, set1.size());
  hDiff_sumPt_.fill(res, (sumpt1+sumpt2)/2.0); // taking average is probably the best we can do, anyway they should be close to each other
  hDiff_Nvertices_.fill(res, nvertices);

  if(vertex1.isValid() && vertex2.isValid()){
    
    hDiff_X_.fill(res,res.avgx());
    hDiff_Y_.fill(res,res.avgy());
    hDiff_Z_.fill(res,res.avgz());

  }

    

  if(!lumiScalers.empty()) {
    hDiff_instLumiScal_.fill(res, lumiScalers.front().instantLumi());
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PrimaryVertexResolution);
