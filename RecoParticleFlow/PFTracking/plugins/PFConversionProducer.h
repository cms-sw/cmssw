#ifndef PFConversionProducer_H
#define PFConversionProducer_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversion.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversionFwd.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

class PFTrackTransformer;
class PFConversionProducer : public edm::stream::EDProducer<> {
public:
  ///Constructor
  explicit PFConversionProducer(const edm::ParameterSet &);

  ///Destructor
  ~PFConversionProducer() override;

private:
  void beginRun(const edm::Run &, const edm::EventSetup &) override;
  void endRun(const edm::Run &, const edm::EventSetup &) override;

  ///Produce the PFRecTrack collection
  void produce(edm::Event &, const edm::EventSetup &) override;

  ///PFTrackTransformer
  PFTrackTransformer *pfTransformer_;
  edm::EDGetTokenT<reco::ConversionCollection> pfConversionContainer_;
  edm::EDGetTokenT<reco::VertexCollection> vtx_h;

  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> transientTrackToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;
};
#endif
