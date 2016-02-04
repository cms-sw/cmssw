#ifndef RecoBTag_SoftLepton_SoftElectronProducer_h
#define RecoBTag_SoftLepton_SoftElectronProducer_h
/** \class SoftElectronProducer
 *
 *
 *  $Id: SoftElectronProducer.h,v 1.3 2009/05/04 19:02:13 fwyzard Exp $
 *  $Date: 2009/05/04 19:02:13 $
 *  $Revision: 1.3 $
 *
 *  \author P. Demin - UCL, Louvain-la-Neuve - Belgium
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TrackDetectorAssociator;
class TrackAssociatorParameters;
class ElectronIdMLP;

// SoftElectronProducer inherits from EDProducer, so it can be a module:
class SoftElectronProducer : public edm::EDProducer
{

 public:

  SoftElectronProducer (const edm::ParameterSet &iConf);
  ~SoftElectronProducer();

 private:

  virtual void produce(edm::Event &iEvent, const edm::EventSetup &iSetup);

  edm::ParameterSet theConf;

  edm::InputTag theTrackTag;
  edm::InputTag theHBHERecHitTag;
  edm::InputTag theBasicClusterTag; //, theBasicClusterShapeTag;
  edm::InputTag thePrimaryVertexTag;
  edm::InputTag barrelRecHitCollection_;
  edm::InputTag endcapRecHitCollection_;

  double theHOverEConeSize;
  double theDiscriminatorCut;

  TrackDetectorAssociator   *theTrackAssociator;
  TrackAssociatorParameters *theTrackAssociatorParameters;

  ElectronIdMLP *theElecNN;

};

#endif

