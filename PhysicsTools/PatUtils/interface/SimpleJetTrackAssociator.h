#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Common/interface/View.h"

namespace helper {
class SimpleJetTrackAssociator {
        public:
                SimpleJetTrackAssociator() :
                    deltaR2_(0), nHits_(0), chi2nMax_(0) { }
                SimpleJetTrackAssociator(double deltaR, int32_t nHits, double chi2nMax) :
                    deltaR2_(deltaR*deltaR), nHits_(nHits), chi2nMax_(chi2nMax) {}
                
                // 100% FWLite compatible (but will make up transient refs)
                void associateTransient(const math::XYZVector &dir, const reco::TrackCollection  &in, reco::TrackRefVector &out) ;

                // more versatile, persistent refs, for when we're in full framework
                void associate(const math::XYZVector &dir, const edm::View<reco::Track> &in, reco::TrackRefVector &out) ;
        private:
                double  deltaR2_;
                int32_t nHits_;
                double  chi2nMax_;
};
}

