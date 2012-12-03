#include <memory>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CalibMuon/CSCCalibration/interface/CSCIndexerBase.h"
#include "CalibMuon/CSCCalibration/interface/CSCIndexerRecord.h"
#include <iostream>

class CSCIndexerAnalyzer : public edm::EDAnalyzer {
public:
  explicit CSCIndexerAnalyzer(const edm::ParameterSet&);
  ~CSCIndexerAnalyzer();

private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) ;
  virtual void endJob() ;

  std::string algoName;
};

CSCIndexerAnalyzer::CSCIndexerAnalyzer(const edm::ParameterSet& pset) {}

CSCIndexerAnalyzer::~CSCIndexerAnalyzer() {}

void CSCIndexerAnalyzer::analyze(const edm::Event& ev, const edm::EventSetup& esu)
{
  const int evalues[10] = {  1,  2,  1,  2,  1,  2,  1,  2,  1,  2 }; // endcap 1=+z, 2=-z
  const int svalues[10] = {  1,  1,  1,  1,  4,  4,  4,  4,  4,  4 }; // station 1-4
  const int rvalues[10] = {  1,  1,  4,  4,  2,  2,  2,  2,  2,  2 }; // ring 1-4
  const int cvalues[10] = {  1,  1,  1,  1,  1,  1, 36, 36,  1,  1 }; // chamber 1-18/36
  const int lvalues[10] = {  1,  1,  1,  1,  1,  1,  1,  1,  6,  6 }; // layer 1-6
  const int tvalues[10] = {  1,  1,  1,  1,  1,  1,  1,  1, 80, 80 }; // strip 1-80 (16, 48 64)

  const CSCIndexerRecord& irec = esu.get<CSCIndexerRecord>();
  edm::ESHandle<CSCIndexerBase> indexer_;
  irec.get(indexer_);

  algoName = indexer_->name();

  std::cout << "CSCIndexerAnalyzer: analyze sees algorithm " << algoName  << " in Event Setup" << std::endl;

  for (int i=0; i<10; ++i ) {
    int ie=evalues[i];
    int is=svalues[i];
    int ir=rvalues[i];
    int ic=cvalues[i];
    int il=lvalues[i];
    int istrip=tvalues[i];
  
    std::cout << "CSCIndexerAnalyzer: calling " << algoName << "::stripChannelIndex(" << ie << "," << is << "," << ir << ","
	    << ic << "," << il << "," << istrip << ") = " 
   << indexer_->stripChannelIndex(ie,is,ir,ic,il,istrip) << std::endl;
  }
}


void CSCIndexerAnalyzer::beginJob() {}

void CSCIndexerAnalyzer::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(CSCIndexerAnalyzer);
