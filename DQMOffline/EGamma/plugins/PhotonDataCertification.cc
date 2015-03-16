#include <iostream>

#include "DQMOffline/EGamma/plugins/PhotonDataCertification.h"
#include "FWCore/Framework/interface/Run.h"
#include "RooGlobalFunc.h"
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooBreitWigner.h"
#include "RooDataHist.h"
#include "RooFitResult.h"


/**\class PhotonDataCertification
*/
//
// Original Author:  Louis James Antonelli
//         Created:  Thu Jan 22 13:42:28CET 2009
//

using namespace std;

PhotonDataCertification::PhotonDataCertification(const edm::ParameterSet& pset)

{


  parameters_ = pset;
  verbose_ = parameters_.getParameter<bool>("verbose");

  if(verbose_) cout << ">>> Constructor (PhotonDataCertification) <<<" << endl;

}


PhotonDataCertification::~PhotonDataCertification() {}



// ------------ method called right after a run ends ------------
void
PhotonDataCertification::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {

  using namespace RooFit;
  if (verbose_) std::cout << ">>> endJob (PhotonDataCertification) <<<" << std::endl;

  //booking histograms according to naming conventions
  ibooker.setCurrentFolder("Egamma/EventInfo/");
  reportSummary_ = ibooker.bookFloat("reportSummary");
  reportSummaryMap_ = ibooker.book2D("reportSummaryMap","reportSummaryMap",3,0,3,1,0,1);

  TH2F * reportSummaryMapTH2 = reportSummaryMap_->getTH2F();
  reportSummaryMapTH2->GetXaxis()->SetBinLabel(1, "EB");
  reportSummaryMapTH2->GetXaxis()->SetBinLabel(2, "EE");
  reportSummaryMapTH2->GetXaxis()->SetBinLabel(3, "Total");
  reportSummaryMapTH2->GetYaxis()->SetBinLabel(1, "InvMassTest");

  float EBResult = invMassZtest("Egamma/PhotonAnalyzer/InvMass/h_02_invMassIsoPhotonsEBarrel",
      "invMassIsolatedPhotonsEB", igetter);

  float EEResult = invMassZtest("Egamma/PhotonAnalyzer/InvMass/h_03_invMassIsoPhotonsEEndcap",
      "invMassIsolatedPhotonsEE", igetter);

  float AllResult = invMassZtest("Egamma/PhotonAnalyzer/InvMass/h_01_invMassAllIsolatedPhotons",
      "invMassAllIsolatedPhotons", igetter);

  if (verbose_) {
    std::cout << "EBResult: " << EBResult << std::endl;
    std::cout << "EEResult: " << EEResult << std::endl;
    std::cout << "AllResult: " << AllResult << std::endl;
  }

  reportSummaryMapTH2->SetBinContent(1, 1, EBResult);
  reportSummaryMapTH2->SetBinContent(2, 1, EEResult);
  reportSummaryMapTH2->SetBinContent(3, 1, AllResult);
  reportSummary_->Fill(AllResult);
}

float PhotonDataCertification::invMassZtest(string path, TString name,
    DQMStore::IGetter & igetter) {

  float ZMass = 91.2;
  float ZWidth = 2.5;
  MonitorElement *TestElem = 0;
  TestElem = igetter.get(path);
  if (TestElem == 0) return 0;
  TH1F *TestHist = TestElem->getTH1F();
  if (TestHist == 0) return 0;
  RooRealVar mass("mass","Mass_{2#gamma}", 0, 200,"GeV");
  RooRealVar mRes("M_{Z}", "Z Mass", ZMass, 70, 110);
  RooRealVar gamma("#Gamma", "#Gamma", ZWidth, 0, 10.0);
  RooBreitWigner BreitWigner("BreitWigner", "Breit-Wigner", mass, mRes, gamma);
  RooDataHist test(name, name, mass, TestHist);

  BreitWigner.fitTo(test, RooFit::Range(80, 100));

  if (abs(mRes.getValV() - ZMass) < ZWidth) {
    return 1.0;
  } else if (abs(mRes.getValV() - ZMass) < gamma.getValV()) {
    return 0.9;
  } else {
    return 0.0;
  }
}
