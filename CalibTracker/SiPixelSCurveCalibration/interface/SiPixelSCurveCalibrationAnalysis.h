#ifndef SiPixelSCurveCalibration_SiPixelSCurveCalibrationAnalysis_h
#define SiPixelSCurveCalibration_SiPixelSCurveCalibrationAnalysis_h

/** \class SiPixelSCurveCalibrationAnalysis
 *
 * A class which will perform an SCurve Calibration
 * analysis, given an SCurve file.
 *
 * \authors Jason Keller (University of Nebraska)
 *
 * \version 1.3 July 5, 2007

 *
 ***********************************************************/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibFormats/SiPixelObjects/interface/PixelCalib.h"  
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "CalibTracker/SiPixelSCurveCalibration/interface/SCurveContainer.h" 
#include "DataFormats/DetId/interface/DetId.h" 
#include "FWCore/Framework/interface/ESHandle.h" 
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"

#include <TF1.h>
#include <TH1F.h> 
#include <TObjArray.h>

#include <memory>
#include <map>
#include <string>

class SiPixelSCurveCalibrationAnalysis : public edm::EDAnalyzer
{
  public:
    explicit SiPixelSCurveCalibrationAnalysis( const edm::ParameterSet& conf);
    ~SiPixelSCurveCalibrationAnalysis();

    virtual void beginJob(const edm::EventSetup&);
    virtual void analyze(const edm::Event&, const edm::EventSetup&);
    virtual void endJob();
    std::string makeName(const int&, const int&, const DetId&);
    std::string makeTitle(const int&, const int&, const DetId&); 
    std::string makeRocName(const int&, const int&, const DetId&, const int&);
    std::string makeRocTitle(const int&, const int&, const DetId&, const int&);
    bool makeHistogram(const SCurveContainer&, const int&, const int&, const int&);
    bool isPeculiar(const TH1F*);

  private:
    edm::ParameterSet conf_; 
    edm::ESHandle<SiPixelFedCablingMap> map_;
    std::string pixsrc_; 
    unsigned int evtnum_;
    std::string inputcalibfile_;
    std::string outputtxtfile_;
    unsigned int fedid_; 
    unsigned int histoNum_;
    bool printHistos_; 
    PixelCalib* calib_;
    unsigned int vcalmin_;
    unsigned int vcalmax_;
    unsigned int vcalstep_;
    unsigned int ntriggers_; 
    edm::Service<TFileService> fs_;
    std::map<unsigned int, SCurveContainer> detIdMap_;
    TF1* fitfunc_;
    TObjArray* meanhistos_;
    TObjArray* sigmahistos_;
};

#endif
