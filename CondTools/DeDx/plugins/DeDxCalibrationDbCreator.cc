// system include files
#include <iostream>
#include <fstream>
#include <vector>

// user include files
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CondFormats/DataRecord/interface/DeDxCalibrationRcd.h"
#include "CondFormats/PhysicsToolsObjects/interface/DeDxCalibration.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

class DeDxCalibrationDbCreator : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  typedef std::pair<uint32_t, unsigned char> ChipId;
  enum AllDetector { PXB = 0, PXF = 1, TIB = 2, TID = 3, TOB = 4, TECThin = 5, TECThick = 6, nDets };

  explicit DeDxCalibrationDbCreator(const edm::ParameterSet&);
  ~DeDxCalibrationDbCreator() override{};

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override{};

  void readStripProps(std::vector<double>&, std::vector<double>&, std::vector<double>&);
  void readGainCorrection(std::map<ChipId, float>&);

  const std::string propFile_, gainFile_;
};

DeDxCalibrationDbCreator::DeDxCalibrationDbCreator(const edm::ParameterSet& iConfig)
    : propFile_(iConfig.getParameter<std::string>("propFile")),
      gainFile_(iConfig.getParameter<std::string>("gainFile")) {}

void DeDxCalibrationDbCreator::beginJob() {
  std::map<ChipId, float> gain;
  std::vector<double> thr, alpha, sigma;
  readStripProps(thr, alpha, sigma);
  readGainCorrection(gain);
  DeDxCalibration TD(thr, alpha, sigma, gain);

  edm::Service<cond::service::PoolDBOutputService> pool;
  if (pool.isAvailable())
    pool->writeOneIOV(TD, pool->beginOfTime(), "DeDxCalibrationRcd");
}

/*****************************************************************************/
void DeDxCalibrationDbCreator::readStripProps(std::vector<double>& thr,
                                              std::vector<double>& alpha,
                                              std::vector<double>& sigma) {
  std::cout << " reading strip properties from " << propFile_;
  std::ifstream file(edm::FileInPath(propFile_).fullPath());

  int det;
  for (det = PXB; det <= PXF; det++) {
    thr.emplace_back(0.);
    alpha.emplace_back(0.);
    sigma.emplace_back(0.);
  }

  while (!file.eof()) {
    std::string detName;
    float f;

    file >> detName;
    file >> f;
    thr.emplace_back(f);
    file >> f;
    alpha.emplace_back(f);
    file >> f;
    sigma.emplace_back(f);

    det++;
  }

  file.close();
  std::cout << " [done]" << std::endl;
}

/*****************************************************************************/
void DeDxCalibrationDbCreator::readGainCorrection(std::map<ChipId, float>& gain) {
  std::cout << " reading gain from " << gainFile_;
  std::ifstream fileGain(edm::FileInPath(gainFile_).fullPath());

  int i = 0;
  while (!fileGain.eof()) {
    uint32_t det;
    int chip;

    int d;
    float g, f;
    std::string s;

    fileGain >> std::hex >> det;
    fileGain >> std::dec >> chip;

    ChipId detId(det, (unsigned char)chip);

    fileGain >> std::dec >> d;
    fileGain >> g;
    fileGain >> f;
    fileGain >> s;

    if (!fileGain.eof()) {
      if (g > 0.5 && g < 2.0)
        gain[detId] = g;
      else
        gain[detId] = -1.;
    }

    if (i++ % 5000 == 0)
      std::cout << ".";
  }

  fileGain.close();
  std::cout << " [done]" << std::endl;
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DeDxCalibrationDbCreator);
