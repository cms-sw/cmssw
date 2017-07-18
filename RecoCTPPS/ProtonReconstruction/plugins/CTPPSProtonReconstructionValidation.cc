/****************************************************************************
*
* Authors:
*  Jan Kašpar (jan.kaspar@gmail.com) 
*    
****************************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// TODO: needed?
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/ProtonReco/interface/ProtonTrack.h"

#include "TFile.h"
#include "TGraph.h"
#include "TH1D.h"
#include "TH2D.h"

//----------------------------------------------------------------------------------------------------

class CTPPSProtonReconstructionValidation : public edm::one::EDAnalyzer<>
{
  public:
    explicit CTPPSProtonReconstructionValidation(const edm::ParameterSet&);
    ~CTPPSProtonReconstructionValidation() {}

  private: 
    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

    virtual void endJob() override;

    edm::EDGetTokenT<std::vector<reco::ProtonTrack>> tokenRecoProtons;

    std::string outputFile;
};

//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

CTPPSProtonReconstructionValidation::CTPPSProtonReconstructionValidation(const edm::ParameterSet &ps) :
  tokenRecoProtons(consumes<std::vector<reco::ProtonTrack>>(ps.getParameter<InputTag>("tagRecoProtons"))),
  outputFile(ps.getParameter<string>("outputFile"))
{
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonReconstructionValidation::analyze(const edm::Event &event, const edm::EventSetup&)
{
  // get input
  Handle<vector<reco::ProtonTrack>> recoProtons;
  event.getByToken(tokenRecoProtons, recoProtons);

  printf("reco protons: %lu\n", recoProtons->size());

  // TODO
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonReconstructionValidation::endJob()
{
  TFile *f_out = TFile::Open(outputFile.c_str(), "recreate");
  
  // TODO

  delete f_out;
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSProtonReconstructionValidation);
