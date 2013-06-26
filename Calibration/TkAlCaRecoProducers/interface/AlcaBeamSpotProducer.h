#ifndef TkAlCaRecoProducer_AlcaBeamSpotProducer_h
#define TkAlCaRecoProducer_AlcaBeamSpotProducer_h

/**_________________________________________________________________
   class:   AlcaBeamSpotProducer.h
   package: Calibration/TkAlCaRecoProducers
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: AlcaBeamSpotProducer.h,v 1.2 2013/05/17 20:25:10 chrjones Exp $

________________________________________________________________**/


// C++ standard
#include <string>
// CMS
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoVertex/BeamSpotProducer/interface/BeamFitter.h"


class AlcaBeamSpotProducer : public edm::one::EDProducer<edm::EndLuminosityBlockProducer,
                                                         edm::one::WatchLuminosityBlocks> {
 public:
  explicit AlcaBeamSpotProducer(const edm::ParameterSet&);
  ~AlcaBeamSpotProducer();

 private:
  virtual void beginLuminosityBlock     (edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) override final;
  virtual void endLuminosityBlock       (edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) override final;
  virtual void endLuminosityBlockProduce(edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup) override final;
  virtual void produce                  (edm::Event& iEvent, const edm::EventSetup& iSetup) override final;
  
  int ftotalevents;
  int fitNLumi_;
  int resetFitNLumi_;
  int countEvt_;       //counter
  int countLumi_;      //counter
  int ftmprun0, ftmprun;
  int beginLumiOfBSFit_;
  int endLumiOfBSFit_;
  std::time_t refBStime[2];

  bool write2DB_;
  bool runbeamwidthfit_;
  bool runallfitters_;
  double inputBeamWidth_;

  BeamFitter * theBeamFitter;
};

#endif
