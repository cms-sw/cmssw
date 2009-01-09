#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "MuonAnalysis/MomentumScaleCalibration/interface/MomentumScaleCorrector.h"

#include <TFile.h>
#include <string>


class ScaleDBWriter : public edm::EDAnalyzer {
public:
  explicit ScaleDBWriter(const edm::ParameterSet&);
  ~ScaleDBWriter();
  
  
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  auto_ptr<MomentumScaleCorrector> corrector_;
};

