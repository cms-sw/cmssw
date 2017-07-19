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

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

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

    struct SingleRPPlots
    {
      TH1D *h_xi = NULL;

      void Init()
      {
        h_xi = new TH1D("", ";#xi", 100, 0., 0.2);
      }

      void Fill(const reco::ProtonTrack &p)
      {
        if (!h_xi)
          Init();

        if (p.valid())
        {
          h_xi->Fill(p.xi());
        }
      }

      void Write() const
      {
        h_xi->Write("h_xi");
      }
    };

    std::map<unsigned int, SingleRPPlots> singleRPPlots;

    struct MultiRPPlots
    {
      TH1D *h_xi=NULL, *h_th_x=NULL, *h_th_y=NULL, *h_vtx_y=NULL, *h_chi_sq=NULL, *h_chi_sq_norm=NULL;

      void Init()
      {
        h_xi = new TH1D("", ";#xi", 100, 0., 0.2);
        h_th_x = new TH1D("", ";#theta_{x}", 100, -500E-6, +500E-6);
        h_th_y = new TH1D("", ";#theta_{y}", 100, -500E-6, +500E-6);
        h_vtx_y = new TH1D("", ";vtx_{y}", 100, -0.002, +0.002);
        h_chi_sq = new TH1D("", ";#chi^{2}", 100, 0., 0.);
        h_chi_sq_norm = new TH1D("", ";#chi^{2}/ndf", 100, 0., 5.);
      }

      void Fill(const reco::ProtonTrack &p)
      {
        if (!h_xi)
          Init();

        if (p.valid())
        {
          h_xi->Fill(p.xi());
          h_th_x->Fill(p.direction().x());
          h_th_y->Fill(p.direction().y());
          h_vtx_y->Fill(p.vertex().y());
          h_chi_sq->Fill(p.fitChiSq);
          if (p.fitNDF > 0)
            h_chi_sq_norm->Fill(p.fitChiSq / p.fitNDF);
        }
      }

      void Write() const
      {
        h_xi->Write("h_xi");
        h_th_x->Write("h_th_x");
        h_th_y->Write("h_th_y");
        h_vtx_y->Write("h_vtx_y");
        h_chi_sq->Write("h_chi_sq");
        h_chi_sq_norm->Write("h_chi_sq_norm");
      }
    };

    std::map<unsigned int, MultiRPPlots> multiRPPlots;
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

  // make single-RP-reco plots
  for (const auto & proton : *recoProtons)
  {
    if (proton.method == reco::ProtonTrack::rmSingleRP)
    {
      CTPPSDetId rpId(* proton.contributingRPIds.begin());
      unsigned int decRPId = rpId.arm()*100 + rpId.station()*10 + rpId.rp();
      singleRPPlots[decRPId].Fill(proton);
    }
  }

  // make multi-RP-reco plots
  for (const auto & proton : *recoProtons)
  {
    if (proton.method == reco::ProtonTrack::rmMultiRP)
    {
      CTPPSDetId rpId(* proton.contributingRPIds.begin());
      unsigned int armId = rpId.arm();
      multiRPPlots[armId].Fill(proton);
    }
  }
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonReconstructionValidation::endJob()
{
  TFile *f_out = TFile::Open(outputFile.c_str(), "recreate");

  TDirectory *d_singleRPPlots = f_out->mkdir("singleRPPlots");
  for (const auto it : singleRPPlots)
  {
    char buf[100];
    sprintf(buf, "rp%u", it.first);
    gDirectory = d_singleRPPlots->mkdir(buf); 
    it.second.Write();
  }

  TDirectory *d_multiRPPlots = f_out->mkdir("multiRPPlots");
  for (const auto it : multiRPPlots)
  {
    char buf[100];
    sprintf(buf, "arm%u", it.first);
    gDirectory = d_multiRPPlots->mkdir(buf); 
    it.second.Write();
  }

  delete f_out;
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSProtonReconstructionValidation);
