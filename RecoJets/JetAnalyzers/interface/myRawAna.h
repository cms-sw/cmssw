#ifndef RecoExamples_myRawAna_h
#define RecoExamples_myRawAna_h
#include <TH1.h>
#include <TH2.h>
#include <TProfile.h>
#include <TFile.h>

/* \class myRawAna
 *
 * \author Jim Hirschauer
 *
 * \version 1
 *
 */
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

// class TFile;

class myRawAna : public edm::EDAnalyzer {

public:
  myRawAna( const edm::ParameterSet & );

private:
  void beginJob(  ) override;
  void analyze ( const edm::Event& , const edm::EventSetup& ) override;
  void endJob() override;

  TH2F *fedSize;
  TH1F *totFedSize;
  
};

#endif
