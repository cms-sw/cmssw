#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversion.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

class PFConversionProducer : public edm::stream::EDProducer<> {
public:
  ///Constructor
  explicit PFConversionProducer(const edm::ParameterSet&);

  ///Destructor
  ~PFConversionProducer() override;

private:
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;

  ///Produce the PFRecTrack collection
  void produce(edm::Event&, const edm::EventSetup&) override;

  ///PFTrackTransformer
  PFTrackTransformer* pfTransformer_;
  edm::EDGetTokenT<reco::ConversionCollection> pfConversionContainer_;
  edm::EDGetTokenT<reco::VertexCollection> vtx_h;

  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> transientTrackToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFConversionProducer);

typedef std::multimap<unsigned, std::vector<unsigned> > BlockMap;
using namespace std;
using namespace edm;

PFConversionProducer::PFConversionProducer(const ParameterSet& iConfig)
    : pfTransformer_(nullptr),
      transientTrackToken_(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
      magneticFieldToken_(esConsumes<edm::Transition::BeginRun>()) {
  produces<reco::PFRecTrackCollection>();
  produces<reco::PFConversionCollection>();

  pfConversionContainer_ = consumes<reco::ConversionCollection>(iConfig.getParameter<InputTag>("conversionCollection"));

  vtx_h = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("PrimaryVertexLabel"));
}

PFConversionProducer::~PFConversionProducer() { delete pfTransformer_; }

void PFConversionProducer::produce(Event& iEvent, const EventSetup& iSetup) {
  //create the empty collections
  auto pfConversionColl = std::make_unique<reco::PFConversionCollection>();
  auto pfRecTrackColl = std::make_unique<reco::PFRecTrackCollection>();

  TransientTrackBuilder const& thebuilder = iSetup.getData(transientTrackToken_);

  reco::PFRecTrackRefProd pfTrackRefProd = iEvent.getRefBeforePut<reco::PFRecTrackCollection>();
  Handle<reco::ConversionCollection> convCollH;
  iEvent.getByToken(pfConversionContainer_, convCollH);

  const reco::ConversionCollection& convColl = *(convCollH.product());

  Handle<reco::VertexCollection> vertex;
  iEvent.getByToken(vtx_h, vertex);
  //Find PV for IP calculation, if there is no PV in collection than use dummy
  reco::Vertex dummy;
  const reco::Vertex* pv = &dummy;
  if (vertex.isValid()) {
    pv = &*vertex->begin();
  } else {  // create a dummy PV
    reco::Vertex::Error e;
    e(0, 0) = 0.0015 * 0.0015;
    e(1, 1) = 0.0015 * 0.0015;
    e(2, 2) = 15. * 15.;
    reco::Vertex::Point p(0, 0, 0);
    dummy = reco::Vertex(p, e, 0, 0, 0);
  }

  int idx = 0;                                    //index of track in PFRecTrack collection
  multimap<unsigned int, unsigned int> trackmap;  //Map of Collections and tracks
  std::vector<unsigned int> conv_coll(0);

  // CLEAN CONVERSION COLLECTION FOR DUPLICATES
  for (unsigned int icoll1 = 0; icoll1 < convColl.size(); icoll1++) {
    if ((!convColl[icoll1].quality(reco::Conversion::arbitratedMergedEcalGeneral)) ||
        (!convColl[icoll1].quality(reco::Conversion::highPurity)))
      continue;

    bool greater_prob = false;
    std::vector<edm::RefToBase<reco::Track> > tracksRefColl1 = convColl[icoll1].tracks();
    for (unsigned it1 = 0; it1 < tracksRefColl1.size(); it1++) {
      reco::TrackRef trackRef1 = (tracksRefColl1[it1]).castTo<reco::TrackRef>();

      for (unsigned int icoll2 = 0; icoll2 < convColl.size(); icoll2++) {
        if (icoll1 == icoll2)
          continue;
        if ((!convColl[icoll2].quality(reco::Conversion::arbitratedMergedEcalGeneral)) ||
            (!convColl[icoll2].quality(reco::Conversion::highPurity)))
          continue;
        std::vector<edm::RefToBase<reco::Track> > tracksRefColl2 = convColl[icoll2].tracks();
        for (unsigned it2 = 0; it2 < tracksRefColl2.size(); it2++) {
          reco::TrackRef trackRef2 = (tracksRefColl2[it2]).castTo<reco::TrackRef>();
          double like1 = -999;
          double like2 = -999;
          //number of shared hits
          int shared = 0;
          for (auto const& hit1 : trackRef1->recHits())
            if (hit1->isValid()) {
              //count number of shared hits
              for (auto const& hit2 : trackRef2->recHits()) {
                if (hit2->isValid() && hit1->sharesInput(hit2, TrackingRecHit::some))
                  shared++;
              }
            }
          float frac = 0;
          //number of valid hits in tracks that are duplicates
          float size1 = trackRef1->found();
          float size2 = trackRef2->found();
          //divide number of shared hits by the total number of hits for the track with less hits
          if (size1 > size2)
            frac = (double)shared / size2;
          else
            frac = (double)shared / size1;
          if (frac > 0.9) {
            like1 = ChiSquaredProbability(convColl[icoll1].conversionVertex().chi2(),
                                          convColl[icoll1].conversionVertex().ndof());
            like2 = ChiSquaredProbability(convColl[icoll2].conversionVertex().chi2(),
                                          convColl[icoll2].conversionVertex().ndof());
          }
          if (like2 > like1) {
            greater_prob = true;
            break;
          }
        }  //end loop over tracks in collection 2

        if (greater_prob)
          break;  //if a duplicate track is found in a collection with greater Chi^2 probability for Vertex fit then break out of comparison loop
      }           //end loop over collection 2 checking
      if (greater_prob)
        break;  //if a duplicate track is found in a collection with greater Chi^2 probability for Vertex fit then one does not need to check the other track the collection will not be stored
    }           //end loop over tracks in collection 1
    if (!greater_prob)
      conv_coll.push_back(icoll1);
  }  //end loop over collection 1

  //Finally fill empty collections
  for (unsigned iColl = 0; iColl < conv_coll.size(); iColl++) {
    unsigned int collindex = conv_coll[iColl];
    //std::cout<<"Filling this collection"<<collindex<<endl;
    std::vector<reco::PFRecTrackRef> pfRecTkcoll;

    std::vector<edm::RefToBase<reco::Track> > tracksRefColl = convColl[collindex].tracks();
    // convert the secondary tracks
    for (unsigned it = 0; it < tracksRefColl.size(); it++) {
      reco::TrackRef trackRef = (tracksRefColl[it]).castTo<reco::TrackRef>();
      reco::PFRecTrack pfRecTrack(trackRef->charge(), reco::PFRecTrack::KF, trackRef.key(), trackRef);
      //std::cout<<"Track Pt "<<trackRef->pt()<<std::endl;
      Trajectory FakeTraj;
      bool valid = pfTransformer_->addPoints(pfRecTrack, *trackRef, FakeTraj);
      if (valid) {
        double stip = -999;
        const reco::PFTrajectoryPoint& atECAL = pfRecTrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance);
        //if extrapolation to ECAL is valid then calculate STIP
        if (atECAL.isValid()) {
          GlobalVector direction(pfRecTrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance).position().x(),
                                 pfRecTrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance).position().y(),
                                 pfRecTrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance).position().z());
          stip = IPTools::signedTransverseImpactParameter(thebuilder.build(*trackRef), direction, *pv)
                     .second.significance();
        }
        pfRecTrack.setSTIP(stip);
        pfRecTkcoll.push_back(reco::PFRecTrackRef(pfTrackRefProd, idx++));
        pfRecTrackColl->push_back(pfRecTrack);
      }
    }  //end loop over tracks
    //store reference to the Conversion collection
    reco::ConversionRef niRef(convCollH, collindex);
    pfConversionColl->push_back(reco::PFConversion(niRef, pfRecTkcoll));
  }  //end loop over collections
  iEvent.put(std::move(pfRecTrackColl));
  iEvent.put(std::move(pfConversionColl));
}

// ------------ method called once each job just before starting event loop  ------------
void PFConversionProducer::beginRun(const edm::Run& run, const EventSetup& iSetup) {
  auto const& magneticField = &iSetup.getData(magneticFieldToken_);
  pfTransformer_ = new PFTrackTransformer(math::XYZVector(magneticField->inTesla(GlobalPoint(0, 0, 0))));
  pfTransformer_->OnlyProp();
}

// ------------ method called once each job just after ending the event loop  ------------
void PFConversionProducer::endRun(const edm::Run& run, const EventSetup& iSetup) {
  delete pfTransformer_;
  pfTransformer_ = nullptr;
}
