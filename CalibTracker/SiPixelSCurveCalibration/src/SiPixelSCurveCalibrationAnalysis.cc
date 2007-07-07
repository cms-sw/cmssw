// File: SiPixelSCurveCalibrationAnalysis.cc
// Description:  see SiPixelSCurveCalibrationAnalysis.h
// Author: Jason Keller (University of Nebraska)
//--------------------------------------------


#include "CalibTracker/SiPixelSCurveCalibration/interface/SiPixelSCurveCalibrationAnalysis.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h" 
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelTopologyBuilder.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <TFile.h>
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
    fitfunc_ = new TF1("fit", "0.5*[0]*(1+TMath::Erf((x-[1])/([2]*sqrt(2))))", vcalmin_, vcalmax_);
    fitfunc_->SetParameters(1.0, 15.0, 0.2);
    mean_ = fs_->make<TH1F>("mean", "mean", 500, vcalmin_, vcalmax_);
    sigma_ = fs_->make<TH1F>("sigma", "sigma", 100, 0, 1);
  }

SiPixelSCurveCalibrationAnalysis::~SiPixelSCurveCalibrationAnalysis()
{
  delete calib_;
}

void SiPixelSCurveCalibrationAnalysis::beginJob(const edm::EventSetup& iSetup)
{
  using namespace edm;
  LogInfo("SCurve Calibration") << "The starting Vcal value is " << vcalmin_;
  LogInfo("SCurve Calibration") << "The ending Vcal value is " << vcalmax_;
  LogInfo("SCurve Calibration") << "Vcal will be incremented in steps of " << vcalstep_;
  LogInfo("SCurve Calibration") << "The number of triggers is " << ntriggers_;
}

void SiPixelSCurveCalibrationAnalysis::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  using namespace edm;
  unsigned int vcal = calib_->vcal_fromeventno(evtnum_);
  ++evtnum_;

  Handle<DetSetVector<PixelDigi> > pixelDigis;
  e.getByLabel(pixsrc_, pixelDigis );
  DetSetVector<PixelDigi>::const_iterator digiiter;
  for(digiiter = pixelDigis->begin() ; digiiter != pixelDigis->end(); ++digiiter)
  {
    unsigned int detid = digiiter->id;
    DetId detector(detid);
    std::map<unsigned int, SCurveContainer>::const_iterator test = detIdMap_.find(detid);
    if(test == detIdMap_.end())
    {
      ESHandle<TrackerGeometry> geom;
      es.get<TrackerDigiGeometryRecord>().get( geom );
      const TrackerGeometry& theTracker(*geom);
      const PixelGeomDetUnit *theGeomDet = dynamic_cast<const PixelGeomDetUnit*>                                             (theTracker.idToDet(detector));
      unsigned int rows = theGeomDet->specificTopology().nrows();
      unsigned int cols = theGeomDet->specificTopology().ncolumns();
      histoNum_ += rows * cols;
      SCurveContainer temp(vcalmin_, vcalmax_, vcalstep_,
                           ntriggers_, rows, cols, detid);
      detIdMap_.insert(std::make_pair(detid, temp));
    }

    DetSet<PixelDigi>::const_iterator pixiter;
    for(pixiter = digiiter->data.begin(); pixiter != digiiter->data.end(); ++pixiter)
    {
      int adc = pixiter->adc();
      int row = pixiter->row();
      int col = pixiter->column();
      detIdMap_[detid].setEff(adc, vcal, row, col);
    }  
  }
}

void SiPixelSCurveCalibrationAnalysis::endJob()
{
  using namespace edm;
  LogInfo("SCurve Calibration") << "Making approximately " << histoNum_ << " histograms.";
  std::map<unsigned int, SCurveContainer>::iterator siter;
  int i = 1;
  for(siter = detIdMap_.begin(); siter != detIdMap_.end(); ++siter)
  {
    int rows = siter->second.getRowMax();
    int cols = siter->second.getColMax();
    for(int j = 0 ; j != rows; ++j)
    {
      for(int k = 0; k != cols; ++k)
      {
        makeHistogram(siter->second, j, k);
        if(i % 1000 == 0)
          LogInfo("SCurve Calibration") << "Making histogram " << i << " out of " << histoNum_;
        ++i;
      }
    }
  }
}

std::string SiPixelSCurveCalibrationAnalysis::makeName(const int& row, const int& col, const DetId& pixdet)
{
  std::stringstream name("");
  if(pixdet.subdetId() == 1)
  {
    PixelBarrelName barrel(pixdet);
    name  << barrel.name() << "r" << row << "c" << col;
  }

  else
  {
    PixelEndcapName endcap(pixdet);
    name << endcap.name() << "r" << row << "c" << col;
  }

  return name.str();
}

std::string SiPixelSCurveCalibrationAnalysis::makeTitle(const int& row, const int& col, const DetId& pixdet)
{
  std::stringstream title("");

  if(pixdet.subdetId() == 1)
  {
    PixelBarrelName barrel(pixdet);
    title << "SCurve for" << barrel.name() << " Row " << row << " Col " << col;
  }

  else
  {
    PixelEndcapName endcap(pixdet);
    title << "SCurve for" << endcap.name() << " Row " << row << " Col " << col;
  }
  return title.str();
}

void SiPixelSCurveCalibrationAnalysis::makeHistogram(const SCurveContainer& sc, const int& row, const int& col)
{
  unsigned int rawid = sc.getRawId();
  std::string name = makeName(row, col, rawid);
  std::string title = makeTitle(row, col, rawid); 
  TH1F* histo = fs_->make<TH1F>(name.c_str(), title.c_str(), calib_->nVcal(), vcalmin_, vcalmax_);
  histo->GetXaxis()->SetTitle("Vcal");
  histo->GetYaxis()->SetTitle("Efficiency");
  for(int l = vcalmin_; l != vcalmax_ + vcalstep_; l += vcalstep_)
  {
    double temp = sc.getEff(l, row, col);
    histo->Fill(l, temp);
  } 
  fitfunc_->SetParameters(1.0, 15.0, 0.2);
  histo->Fit("fit", "RQ0");
  //histo->Write();
  mean_->Fill(fitfunc_->GetParameter(1));
  sigma_->Fill(fitfunc_->GetParameter(2));
  delete histo;
}

