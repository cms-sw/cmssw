#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "MuonAnalysis/MomentumScaleCalibration/interface/MomentumScaleCorrector.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/ResolutionFunction.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/BackgroundFunction.h"

#include <TFile.h>
#include <string>


class DBWriter : public edm::EDAnalyzer {
public:
  explicit DBWriter(const edm::ParameterSet&);
  ~DBWriter();
  
private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() {};

  std::auto_ptr<BaseFunction> corrector_;
};
