//A class that will perform SCurve Calibration
//when given the digis.
#include "CalibTracker/SiPixelSCurveCalibration/interface/SiPixelSCurveCalibrationAnalysis.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h" 
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelTopologyBuilder.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"

#include <TFile.h>
#include <TH1F.h>
#include <TMath.h>

#include <iostream>
#include <sstream>

SiPixelSCurveCalibrationAnalysis::SiPixelSCurveCalibrationAnalysis(const edm::ParameterSet& conf) :
  conf_(conf),
  pixsrc_(conf.getUntrackedParameter<std::string>("src", "source")),
  evtnum_(0),
  inputcalibfile_(conf.getParameter<std::string>("inputCalibFile")),
  histoNum_(0)
  {
    calib_ = new PixelCalib(inputcalibfile_);
    vcalmin_ = calib_->vcal_first();
    vcalmax_ = calib_->vcal_last();
    vcalstep_ = calib_->vcal_step();
    ntriggers_ = calib_->nTriggersPerPattern();
    fitfunc_ = new TF1("fit", "0.5*[0]*(1+TMath::Erf((x-[1])/([2]*sqrt(x))))", vcalmin_, vcalmax_);
    fitfunc_->SetParameters(1.0, 15.0, 4.0);
  }

SiPixelSCurveCalibrationAnalysis::~SiPixelSCurveCalibrationAnalysis()
{
  delete calib_;
}

void SiPixelSCurveCalibrationAnalysis::beginJob(const edm::EventSetup& iSetup)
{
  std::cout << "The starting Vcal value is " << vcalmin_ << std::endl;
  std::cout << "The ending Vcal value is " << vcalmax_ << std::endl;
  std::cout << "Vcal will be incremented in steps of " << vcalstep_ << std::endl;
  std::cout << "The number of triggers is " << ntriggers_ << std::endl;
}

void SiPixelSCurveCalibrationAnalysis::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  //std::cout <<"Entering the analyze function" << std::endl;
  using namespace edm;
  //std::cout <<"Getting Vcal" <<std::endl;
  unsigned int vcal = calib_->vcal_fromeventno(evtnum_);
  ++evtnum_;

  Handle<DetSetVector<PixelDigi> > pixelDigis;
  e.getByLabel(pixsrc_, pixelDigis );
  DetSetVector<PixelDigi>::const_iterator digiiter;
  //std::cout << "Entering Big For Loop" << std::endl;
  for(digiiter = pixelDigis->begin() ; digiiter != pixelDigis->end(); ++digiiter)
  {
    unsigned int detid = digiiter->id;
    DetId detector(detid);
    // std::cout << "Checking Map for DetId" << std::endl;
    std::map<unsigned int, SCurveContainer>::const_iterator test = detIdMap_.find(detid);
    if(test == detIdMap_.end())
    {
      //std::cout << "DetId not Found.  Creating Entry" << std::endl;
      ESHandle<TrackerGeometry> geom;
      es.get<TrackerDigiGeometryRecord>().get( geom );
      const TrackerGeometry& theTracker(*geom);
      const PixelGeomDetUnit *theGeomDet = dynamic_cast<const PixelGeomDetUnit*>                                             (theTracker.idToDet(detector));
      unsigned int rows = theGeomDet->specificTopology().nrows();
      unsigned int cols = theGeomDet->specificTopology().ncolumns();
      histoNum_ += rows * cols;
      SCurveContainer temp(vcalmin_, vcalmax_, vcalstep_,
                           ntriggers_, rows, cols, detid);
      //std::cout << "Adding new SCurveContainer to map" << std::endl;
      detIdMap_.insert(std::make_pair(detid, temp));
    }

    DetSet<PixelDigi>::const_iterator pixiter;
    //std::cout << "entering little for loop" << std::endl;
    for(pixiter = digiiter->data.begin(); pixiter != digiiter->data.end(); ++pixiter)
    {
      int adc = pixiter->adc();
      int row = pixiter->row();
      int col = pixiter->column();
  //    std::cout << "Setting efficiency" << std::endl;
      detIdMap_[detid].setEff(adc, vcal, row, col);
    }  
  }
//  std::cout << "Leaving the analyze function" << std::endl;
}

void SiPixelSCurveCalibrationAnalysis::endJob()
{
  std::cout << "Entering the endJob function" << std::endl;
  std::cout << "Making approximately " << histoNum_ << " histograms." << std::endl;
  std::map<unsigned int, SCurveContainer>::iterator siter;
  int i = 0;
  for(siter = detIdMap_.begin(); siter != detIdMap_.end(); ++siter)
  {
    int rows = siter->second.getRowMax();
    int cols = siter->second.getColMax();
    for(int j = 0 ; j != rows; ++j)
    {
      for(int k = 0; k != cols; ++k)
      {
        makeHistogram(siter->second, j, k);
        if(i % 250 == 0)
          std::cout << "Making histogram " << i << " out of " << histoNum_ << std::endl;
        ++i;
      }
    }
  }
  std::cout << "Leaving the endJob function" << std::endl;
}

std::string SiPixelSCurveCalibrationAnalysis::makeName(const int& row, const int& col, const int& rawid)
{
  std::stringstream name("");
  name << "d" << rawid << "r" << row << "c" << col;
  return name.str();
}

std::string SiPixelSCurveCalibrationAnalysis::makeTitle(const int& row, const int& col, const int& rawid)
{
  std::stringstream title("");
  title << "SCurve for DetId " << rawid << " Row " << row << " Col " << col;
  return title.str();
}

void SiPixelSCurveCalibrationAnalysis::makeHistogram(const SCurveContainer& sc, const int& row, const int& col)
{
  unsigned int rawid = sc.getRawId();
  std::string name = makeName(row, col, rawid);
  std::string title = makeTitle(row, col, rawid); 
  TH1F* histo = fs_->make<TH1F>(name.c_str(), title.c_str(), calib_->nVcal(), vcalmin_, vcalmax_);
  for(int l = vcalmin_; l != vcalmax_ + vcalstep_; l += vcalstep_)
  {
    double temp = sc.getEff(l, row, col);
    histo->Fill(l, temp);
  }
  histo->Fit("fit", "RQ0");
  histo->Write();
  delete histo;
}

