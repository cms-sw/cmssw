#ifndef CalibTracker_SiPixelESProducers_SiPixelDetInfoFileWriter_h
#define CalibTracker_SiPixelESProducers_SiPixelDetInfoFileWriter_h
// -*- C++ -*-
//
// Package:    SiPixelDetInfoFileWriter
// Class:      SiPixelDetInfoFileWriter
// 
/**\class SiPixelDetInfoFileWriter SiPixelDetInfoFileWriter.cc CalibTracker/SiPixelCommon/src/SiPixelDetInfoFileWriter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo Chiochia
//         Created:  Mon Nov 20 10:04:31 CET 2006
// $Id: SiPixelDetInfoFileWriter.h,v 1.2 2010/01/13 16:25:38 ursl Exp $
//
//

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <string>
#include <iostream>
#include <fstream>

class SiPixelDetInfoFileWriter : public edm::EDAnalyzer {

public:

  explicit SiPixelDetInfoFileWriter(const edm::ParameterSet&);
  ~SiPixelDetInfoFileWriter();

private:

  void beginJob();
  void beginRun(const edm::Run &, const edm::EventSetup &);
  void analyze(const edm::Event &, const edm::EventSetup &);

private:


  std::ofstream outputFile_; 
  std::string filePath_;


};
#endif
