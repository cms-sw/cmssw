#ifndef CalibTracker_SiStripChannelGain_SiStripDetInfoFileWriter_h
#define CalibTracker_SiStripChannelGain_SiStripDetInfoFileWriter_h
// -*- C++ -*-
//
// Package:    SiStripDetInfoFileWriter
// Class:      SiStripDetInfoFileWriter
// 
/**\class SiStripDetInfoFileWriter SiStripDetInfoFileWriter.cc CalibTracker/SiStripCommon/src/SiStripDetInfoFileWriter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  G. Bruno
//         Created:  Mon Nov 20 10:04:31 CET 2006
// $Id: SiStripDetInfoFileWriter.h,v 1.1 2007/06/13 14:03:35 gbruno Exp $
//
//

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <string>
#include <iostream>
#include <fstream>

class SiStripDetInfoFileWriter : public edm::EDAnalyzer {

public:

  explicit SiStripDetInfoFileWriter(const edm::ParameterSet&);
  ~SiStripDetInfoFileWriter();

private:

  void beginJob(const edm::EventSetup& iSetup);

  void analyze(const edm::Event &, const edm::EventSetup &){};

private:


  std::ofstream outputFile_; 
  std::string filePath_;


};
#endif
