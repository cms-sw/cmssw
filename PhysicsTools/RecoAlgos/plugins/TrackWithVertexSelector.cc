#include "PhysicsTools/RecoAlgos/plugins/TrackWithVertexSelector.h"
//
// constructors and destructor
//
using reco::modules::TrackWithVertexSelector;

TrackWithVertexSelector::TrackWithVertexSelector(const edm::ParameterSet& iConfig) :
        numberOfValidHits_(iConfig.getParameter<uint32_t>("numberOfValidHits")),
        normalizedChi2_(iConfig.getParameter<double>("normalizedChi2")),
        ptMin_(iConfig.getParameter<double>("ptMin")),
        ptMax_(iConfig.getParameter<double>("ptMax")),
        etaMin_(iConfig.getParameter<double>("etaMin")),
        etaMax_(iConfig.getParameter<double>("etaMax")),
        dzMax_(iConfig.getParameter<double>("dzMax")),
        d0Max_(iConfig.getParameter<double>("d0Max")),
        nVertices_(iConfig.getParameter<bool>("useVtx") ? iConfig.getParameter<uint32_t>("nVertices") : 0),
        vertexTag_(iConfig.getParameter<edm::InputTag>("vertexTag")),
        vtxFallback_(iConfig.getParameter<bool>("vtxFallback")),
        zetaVtx_(iConfig.getParameter<double>("zetaVtx")),
        rhoVtx_(iConfig.getParameter<double>("rhoVtx"))
   { } 

TrackWithVertexSelector::~TrackWithVertexSelector() {  }

bool TrackWithVertexSelector::operator()(const reco::Track &t, const edm::Event &evt) {
    using std::abs;
    if (    (t.numberOfValidHits() >= numberOfValidHits_) &&
            (t.normalizedChi2()    <= normalizedChi2_) &&
            (t.pt()         >= ptMin_)      &&
            (t.pt()         <= ptMax_)      &&
            (abs(t.eta())   <= etaMax_)     &&
            (abs(t.eta())   >= etaMin_)     &&
            (abs(t.dz())    <= dzMax_)      &&
            (abs(t.d0())    <= d0Max_)  ) {

        if ((nVertices_ == 0)) return true;

        bool ok = true;
        edm::Handle<reco::VertexCollection> hVtx;
        evt.getByLabel(vertexTag_, hVtx);

        if (nVertices_ > 0) {
            ok = false;
            const Point &pca = t.vertex();
            if (hVtx->size() > 0) {
                unsigned int tested = 1;
                for (reco::VertexCollection::const_iterator it = hVtx->begin(), ed = hVtx->end();
                        it != ed; ++it) {
                    if (testPoint(pca, it->position())) { ok = true; break; }
                    if (tested++ >= nVertices_) break;
                }
            } else if (vtxFallback_) {
                if (testPoint(pca, Point())) ok = true;
            }
        }
        return ok;
    } 
    return false;
}

bool TrackWithVertexSelector::testPoint(const Point &point, const Point &vtx) {
    using std::abs;
    math::XYZVector d = point - vtx;
    return ((abs(d.z()) < zetaVtx_) && (abs(d.Rho()) < rhoVtx_));
}
