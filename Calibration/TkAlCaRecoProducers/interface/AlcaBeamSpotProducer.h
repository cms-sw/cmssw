#ifndef TkAlCaRecoProducer_AlcaBeamSpotProducer_h
#define TkAlCaRecoProducer_AlcaBeamSpotProducer_h

/**_________________________________________________________________
   class:   AlcaBeamSpotProducer.h
   package: Calibration/TkAlCaRecoProducers



 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)


________________________________________________________________**/

// C++ standard
#include <string>
// CMS
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoVertex/BeamSpotProducer/interface/BeamFitter.h"

class AlcaBeamSpotProducer
    : public edm::one::EDProducer<edm::EndLuminosityBlockProducer, edm::one::WatchLuminosityBlocks> {
public:
  explicit AlcaBeamSpotProducer(const edm::ParameterSet &);
  ~AlcaBeamSpotProducer() override;

private:
  void beginLuminosityBlock(edm::LuminosityBlock const &lumiSeg, const edm::EventSetup &iSetup) final;
  void endLuminosityBlock(edm::LuminosityBlock const &lumiSeg, const edm::EventSetup &iSetup) final;
  void endLuminosityBlockProduce(edm::LuminosityBlock &lumiSeg, const edm::EventSetup &iSetup) final;
  void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) final;

  int ftotalevents;
  int fitNLumi_;
  int resetFitNLumi_;
  int countEvt_;   // counter
  int countLumi_;  // counter
  int ftmprun0, ftmprun;
  int beginLumiOfBSFit_;
  int endLumiOfBSFit_;
  std::time_t refBStime[2];

  bool write2DB_;
  bool runbeamwidthfit_;
  bool runallfitters_;
  double inputBeamWidth_;

  BeamFitter *theBeamFitter;
};

#endif
