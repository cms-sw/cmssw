#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"

#include "DataFormats/Candidate/interface/Candidate.h"
typedef SimpleFlatTableProducer<reco::Candidate> SimpleCandidateFlatTableProducer;

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
typedef EventSingletonSimpleFlatTableProducer<GenEventInfoProduct> SimpleGenEventFlatTableProducer;

#include "SimDataFormats/HTXS/interface/HiggsTemplateCrossSections.h"
typedef EventSingletonSimpleFlatTableProducer<HTXS::HiggsClassification> SimpleHTXSFlatTableProducer;

#include "DataFormats/ProtonReco/interface/ForwardProton.h"
typedef SimpleFlatTableProducer<reco::ForwardProton> SimpleProtonTrackFlatTableProducer;

#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
typedef SimpleFlatTableProducer<CTPPSLocalTrackLite> SimpleLocalTrackFlatTableProducer;

#include "DataFormats/Math/interface/Point3D.h"
typedef EventSingletonSimpleFlatTableProducer<math::XYZPointF> SimpleXYZPointFlatTableProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SimpleCandidateFlatTableProducer);
DEFINE_FWK_MODULE(SimpleGenEventFlatTableProducer);
DEFINE_FWK_MODULE(SimpleHTXSFlatTableProducer);
DEFINE_FWK_MODULE(SimpleProtonTrackFlatTableProducer);
DEFINE_FWK_MODULE(SimpleLocalTrackFlatTableProducer);
DEFINE_FWK_MODULE(SimpleXYZPointFlatTableProducer);