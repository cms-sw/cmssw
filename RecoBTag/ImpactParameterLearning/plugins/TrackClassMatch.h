#ifndef TrackClassMatch_H
#define TrackClassMatch_H

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "CondFormats/BTauObjects/interface/TrackProbabilityCategoryData.h"
#include "CondFormats/BTauObjects/interface/TrackProbabilityCalibration.h"

  /**  filter to define the belonging of a track to a TrackClass
   */ 
class TrackClassMatch 
{
 public:

 TrackClassMatch() {}

 class Input
 {
 public:
  Input(const reco::Track & t,const reco::Jet &j, const reco::Vertex & v) :
                       track(t), jet(j), vertex(v) {}

  const reco::Track & track;
  const reco::Jet & jet;
  const reco::Vertex & vertex;
 };

 typedef Input first_argument_type;
 typedef TrackProbabilityCalibration::Entry second_argument_type;
 typedef bool result_type;

 bool operator()(const first_argument_type & input , const second_argument_type & category ) const
{
 const reco::Track & track = input.track;
// const reco::Jet & jet = input.jet;
// const reco::Vertex & pv = input.vertex;
 const TrackProbabilityCategoryData & d = category.category;
  //Track Data
  double p=track.p();
  double eta=track.eta();
  double nhit=track.numberOfValidHits();
  double npix=track.hitPattern().numberOfValidPixelHits();
  bool   firstPixel=track.hitPattern().hasValidHitInFirstPixelBarrel();
  double chi=track.normalizedChi2();
  //Chi^2 cut  if used
  bool chicut=(chi >= d.chiMin        &&       chi < d.chiMax );
  if(d.chiMin<=0.01 && d.chiMax<=0.01) chicut=true;

  //First Pixel Hit cut 1=there should be an hit in first layer, -1=there should not be an hit, 0 = I do not care
  bool  fisrtPixelCut = ( (firstPixel && d.withFirstPixel == 1) || (!firstPixel && d.withFirstPixel == -1) || d.withFirstPixel == 0 );

  //the AND of everything
  bool result=(       p >  d.pMin       &&         p <  d.pMax       &&
           fabs(eta) >  d.etaMin     &&  fabs(eta) <  d.etaMax     &&
               nhit >= d.nHitsMin      &&      nhit <= d.nHitsMax      &&
               npix >= d.nPixelHitsMin &&      npix <= d.nPixelHitsMax &&
                chicut && fisrtPixelCut );
  return result;
 }



};


#endif








