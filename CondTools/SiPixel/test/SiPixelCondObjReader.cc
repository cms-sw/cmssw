#include <memory>

#include "CondTools/SiPixel/test/SiPixelCondObjReader.h"

SiPixelCondObjReader::SiPixelCondObjReader(const edm::ParameterSet& iConfig)
{
}

void
SiPixelCondObjReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  edm::ESHandle<SiPixelGainCalibration> SiPixelGainCalibration_;
  iSetup.get<SiPixelGainCalibrationRcd>().get(SiPixelGainCalibration_);

  edm::LogInfo("SiPixelCondObjReader") << "[SiPixelCondObjReader::analyze] End Reading CondObjects" << std::endl;

}

// ------------ method called once each job just before starting event loop  ------------
void 
SiPixelCondObjReader::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiPixelCondObjReader::endJob() {
}
