#ifndef TrackerAlignment_AlignmentPrescaler_H
#define TrackerAlignment_AlignmentPrescaler_H

#include <Riostream.h>
#include <string>
#include "TFile.h"
#include "TTree.h"
#include "TRandom3.h"
#include "TH1F.h"


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventPrincipal.h" 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Utilities/General/interface/ClassName.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "DataFormats/Alignment/interface/AlignmentClusterFlag.h"
#include "DataFormats/Alignment/interface/AliClusterValueMap.h"


class AlignmentPrescaler : public edm::EDProducer{

 public:
  AlignmentPrescaler(const edm::ParameterSet &iConfig);
  ~AlignmentPrescaler();
  void beginJob( const edm::EventSetup & );
  void endJob();
  virtual void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) ;

 private:
  edm::InputTag src_;//tracks in input
  edm::InputTag AM_;//Hit-quality association map

  std::string prescfilename_;//name of the file containing the TTree with the prescaling factors
  std::string presctreename_;//name of the  TTree with the prescaling factors

  TFile *fpresc_;
  TTree *tpresc_;
  TRandom3 *myrand;
  /*
  //temp, just for debug
  TFile *fh;
  TH1F *hrr;
  TH1F *hnhitssubdet;
  TH1I *hnhitsTIBL3;
  */


  int layerFromId (const DetId& id) const;

  unsigned int detid_;
  float preschit_, prescoverlap_;
  int totnhitspxl_;
};
#endif
