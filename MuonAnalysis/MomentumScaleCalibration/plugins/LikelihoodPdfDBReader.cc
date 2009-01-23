// #include "CondFormats/PhysicsToolsObjects/interface/Histogram2D.h"
// #include "MuonAnalysis/MomentumScaleCalibrationObjects/interface/MuScleFitLikelihoodPdf.h"
#include "CondFormats/RecoMuonObjects/interface/MuScleFitLikelihoodPdf.h"
// #include "MuonAnalysis/MomentumScaleCalibrationObjects/interface/MuScleFitLikelihoodPdfRcd.h"
#include "CondFormats/DataRecord/interface/MuScleFitLikelihoodPdfRcd.h"

#include "LikelihoodPdfDBReader.h"

#include <iostream>
#include <stdio.h>
#include <sys/time.h>
#include <string>

using namespace std;
using namespace cms;

LikelihoodPdfDBReader::LikelihoodPdfDBReader( const edm::ParameterSet& iConfig ){}
//:  printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug",1)){}

LikelihoodPdfDBReader::~LikelihoodPdfDBReader(){}

void LikelihoodPdfDBReader::analyze( const edm::Event& e, const edm::EventSetup& iSetup){

  edm::ESHandle<MuScleFitLikelihoodPdf> likelihoodPdf;
  iSetup.get<MuScleFitLikelihoodPdfRcd>().get(likelihoodPdf);
  edm::LogInfo("LikelihoodPdfDBReader") << "[LikelihoodPdfDBReader::analyze] End Reading MuScleFitLikelihoodPdfRcd" << endl;
  vector<PhysicsTools::Calibration::HistogramD2D>::const_iterator histo = likelihoodPdf->histograms.begin();
  vector<string>::const_iterator name = likelihoodPdf->names.begin();
  vector<int>::const_iterator xBins = likelihoodPdf->xBins.begin();
  vector<int>::const_iterator yBins = likelihoodPdf->yBins.begin();
  for( ; histo != likelihoodPdf->histograms.end(); ++histo, ++name, ++xBins, ++yBins ) {
    int nBinsX = *xBins;
    int nBinsY = *yBins;
    cout << *name << ": nBinsX = " << nBinsX << ", nBinsY = " << nBinsY << endl;
    for(int ix=0; ix<=nBinsX+1; ix++){
      for(int iy=0; iy<=nBinsY+1; iy++){

        // cout << "("<<ix<<", "<<iy<<" ) = " << histo->binContent(ix, iy ) << endl;

      }
    }
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(LikelihoodPdfDBReader);
