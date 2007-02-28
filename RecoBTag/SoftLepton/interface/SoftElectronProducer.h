#ifndef RecoBTag_SoftLepton_SoftElectronProducer_h
#define RecoBTag_SoftLepton_SoftElectronProducer_h
/** \class SoftElectronProducer
 *
 *
 *  $Id: SoftElectronProducer.h,v 1.3 2007/02/15 20:09:13 fwyzard Exp $
 *  $Date: 2007/02/15 20:09:13 $
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

  TrackDetectorAssociator *theTrackAssociator;

  ElectronIdMLP *theElecNN;

};

#endif

