#ifndef RecoBTag_SoftLepton_SoftElectronProducer_h
#define RecoBTag_SoftLepton_SoftElectronProducer_h
/** \class SoftElectronProducer
 *
 *
 *  $Id: SoftElectronProducer.h,v 1.1 2007/08/17 23:10:10 fwyzard Exp $
 *  $Date: 2007/08/17 23:10:10 $
 *  $Revision: 1.1 $
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

  virtual void beginJob(edm::EventSetup const &iSetup);
  virtual void produce(edm::Event &iEvent, const edm::EventSetup &iSetup);

  edm::ParameterSet theConf;

  edm::InputTag theTrackTag;
  edm::InputTag theHBHERecHitTag;
  edm::InputTag theBasicClusterTag, theBasicClusterShapeTag;
  edm::InputTag thePrimaryVertexTag;

  double theHOverEConeSize;
  double theDiscriminatorCut;

  TrackDetectorAssociator   *theTrackAssociator;
  TrackAssociatorParameters *theTrackAssociatorParameters;

  ElectronIdMLP *theElecNN;

};

#endif

