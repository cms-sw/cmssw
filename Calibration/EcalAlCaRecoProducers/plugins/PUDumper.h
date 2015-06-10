#ifndef PUDumper_h
#define PUDumper_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

#include "TTree.h"


class PUDumper : public edm::EDAnalyzer
{
 public:
  
  //! ctor
  explicit PUDumper(const edm::ParameterSet&);
  
  //! dtor 
  ~PUDumper();
  
  
  
 private:
  
  //! the actual analyze method 
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  
  
  
 private:

  edm::EDGetTokenT< std::vector<PileupSummaryInfo> > pileupSummaryToken_;

  //edm::InputTag MCPileupTag_;
  
  TTree* PUTree_;

  Int_t     	runNumber;   ///< 
  Long64_t      eventNumber; ///<
  Int_t         lumiBlock;   ///< lumi section
  //UInt_t 	runTime;     ///< unix time
  
  Int_t nBX;
  Int_t BX_[100];
  Int_t nPUtrue_;
  Int_t nPUobs_[100];
};

#endif
