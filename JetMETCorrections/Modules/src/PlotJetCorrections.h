#ifndef PLOTJETCORRECTIONS_H
#define PLOTJETCORRECTIONS_H
// Template to demonstrate accessing to correction service from user analyzer
// plot correction as a function of et:eta
// author: F.Ratnikov UMd Mar. 16, 2007
// 
#include <vector>
#include <string>
#include "FWCore/Framework/interface/EDAnalyzer.h"

 class PlotJetCorrections : public edm::EDAnalyzer {
 public:
   PlotJetCorrections (const edm::ParameterSet&);
   virtual ~PlotJetCorrections() {}
   
   virtual void analyze(const edm::Event&, const edm::EventSetup&);
 private:
   std::vector <std::string> mCorrectorNames;
   std::string mFileName;
   bool mAllDone;
 };

#endif
