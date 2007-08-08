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
// $Id: SiPixelDetInfoFileWriter.h,v 1.1 2007/07/09 11:24:03 gbruno Exp $
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

  void beginJob(const edm::EventSetup& iSetup);

  void analyze(const edm::Event &, const edm::EventSetup &){};

private:


  std::ofstream outputFile_; 
  std::string filePath_;


};
#endif
