// -*- C++ -*-
//
// Package:    EcalLaserAnalyzerYousi
// Class:      EcalLaserAnalyzerYousi
//
/**\class EcalLaserAnalyzerYousi EcalLaserAnalyzerYousi.cc CalibCalorimetry/EcalLaserAnalyzerYousi/src/EcalLaserAnalyzerYousi.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Yousi Ma
//         Created:  Tue Jun 19 23:06:36 CEST 2007
//
//

// system include files
#include <memory>
#include <iostream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TROOT.h"
#include "TF1.h"
#include "TNtuple.h"
#include "TDirectory.h"

//
// class decleration
//

class EcalLaserAnalyzerYousi : public edm::one::EDAnalyzer<> {
public:
  explicit EcalLaserAnalyzerYousi(const edm::ParameterSet &);
  ~EcalLaserAnalyzerYousi() override = default;

private:
  void beginJob() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override;

  // ----------member data ---------------------------
  // Declare histograms and ROOT trees, etc.
  TH1F *C_APD[1700];
  TH1F *C_APDPN[1700];
  TH1F *C_PN[1700];
  TH1F *C_J[1700];
  TH2F *C_APDPN_J[1700];

  TH1F *peakAPD[2];
  TH1F *peakAPDPN[2];
  TH1F *APD_LM[9];
  TH1F *APDPN_LM[9];
  TH2F *APDPN_J_LM[9];
  TH2F *APDPN_J_H[2];

  //fixme make declare and init separate
  TH2F *APD;
  TH2F *APD_RMS;
  TH2F *APDPN;
  TH2F *APDPN_RMS;
  TH2F *PN;
  TH2F *APDPN_J;
  TH2F *APDPN_C;

  TH1F *FitHist;
  TH2F *Count;

  TFile *fPN;
  TFile *fAPD;
  TFile *fROOT;

  TNtuple *C_Tree[1700];

  //parameters

  const std::string hitCollection_;
  const std::string hitProducer_;
  //  std::string PNFileName_ ;
  //  std::string ABFileName_ ;
  const std::string outFileName_;
  const std::string SM_;
  const std::string Run_;
  const std::string digiProducer_;
  const std::string PNdigiCollection_;
  const edm::EDGetTokenT<EcalRawDataCollection> rawDataToken_;
  const edm::EDGetTokenT<EBUncalibratedRecHitCollection> hitToken_;
  const edm::EDGetTokenT<EcalPnDiodeDigiCollection> pnDigiToken_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EcalLaserAnalyzerYousi::EcalLaserAnalyzerYousi(const edm::ParameterSet &iConfig)
    : hitCollection_(iConfig.getUntrackedParameter<std::string>("hitCollection")),
      hitProducer_(iConfig.getUntrackedParameter<std::string>("hitProducer")),
      outFileName_(iConfig.getUntrackedParameter<std::string>("outFileName")),
      SM_(iConfig.getUntrackedParameter<std::string>("SM")),
      Run_(iConfig.getUntrackedParameter<std::string>("Run")),
      digiProducer_(iConfig.getUntrackedParameter<std::string>("digiProducer")),
      PNdigiCollection_(iConfig.getUntrackedParameter<std::string>("PNdigiCollection")),
      rawDataToken_(consumes<EcalRawDataCollection>(edm::InputTag(digiProducer_))),
      hitToken_(consumes<EBUncalibratedRecHitCollection>(edm::InputTag(hitProducer_, hitCollection_))),
      pnDigiToken_(consumes<EcalPnDiodeDigiCollection>(edm::InputTag(digiProducer_, PNdigiCollection_))) {
  //now do what ever initialization is needed
  //get the PN and AB file names
  //get the output file names, digi producers, etc

  //  PNFileName_ = iConfig.getUntrackedParameter<std::string>("PNFileName");
  //  ABFileName_ = iConfig.getUntrackedParameter<std::string>("ABFileName");
}

//
// member functions
//

// ------------ method called to for each event  ------------
void EcalLaserAnalyzerYousi::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;

  //  if ( fPN->IsOpen() ) { edm::LogInfo("EcalLaserAnalyzerYousi") <<"fPN is open in analyze OKAAAAAAAAYYY \n\n"; }

  const edm::Handle<EcalRawDataCollection> &DCCHeaders = iEvent.getHandle(rawDataToken_);

  EcalDCCHeaderBlock::EcalDCCEventSettings settings = DCCHeaders->begin()->getEventSettings();

  int wavelength = settings.wavelength;

  //   std::cout<<"wavelength: "<<wavelength<<"\n\n";

  if (wavelength != 0) {
    return;
  }  //only process blue laser

  const edm::Handle<EBUncalibratedRecHitCollection> &hits = iEvent.getHandle(hitToken_);

  if (!hits.isValid()) {
    edm::LogError("EcalLaserAnalyzerYousi")
        << "Cannot get product:  EBRecHitCollection from: " << hitCollection_ << " - returning.\n\n";
    return;
  }

  const edm::Handle<EcalPnDiodeDigiCollection> &pndigis = iEvent.getHandle(pnDigiToken_);
  if (!pndigis.isValid()) {
    edm::LogError("EcalLaserAnalyzerYousi") << "Cannot get product:  EBdigiCollection from: getHandle - returning.\n\n";
    return;
  }

  Float_t PN_amp[5];

  //do some averaging over each pair of PNs
  for (int j = 0; j < 5; ++j) {
    PN_amp[j] = 0;
    for (int z = 0; z < 2; ++z) {
      FitHist->Reset();
      TF1 peakFit("peakFit", "[0] +[1]*x +[2]*x^2", 30, 50);
      TF1 pedFit("pedFit", "[0]", 0, 5);

      for (int k = 0; k < 50; k++) {
        FitHist->SetBinContent(k, (*pndigis)[j + z * 5].sample(k).adc());
      }
      pedFit.SetParameter(0, 750);
      FitHist->Fit(&pedFit, "RQI");
      Float_t ped = pedFit.GetParameter(0);

      Int_t maxbin = FitHist->GetMaximumBin();
      peakFit.SetRange(FitHist->GetBinCenter(maxbin) - 4 * FitHist->GetBinWidth(maxbin),
                       FitHist->GetBinCenter(maxbin) + 4 * FitHist->GetBinWidth(maxbin));
      peakFit.SetParameters(750, 4, -.05);
      FitHist->Fit(&peakFit, "RQI");
      Float_t max = peakFit.Eval(-peakFit.GetParameter(1) / (2 * peakFit.GetParameter(2)));
      if (ped != max) {
        PN_amp[j] = PN_amp[j] + max - ped;
      } else {
        PN_amp[j] = PN_amp[j] + max;
      }

    }  //end of z loop
    PN_amp[j] = PN_amp[j] / 2.0;
  }  //end of j loop

  //do some real PN, APD calculations

  //FIXME. previously used .info files to get time, what to do now?

  //   TNtuple *Time = new TNtuple("Time", "Time", "Time");
  //   Int_t iTime = Get_Time(Input_File);
  //   Time->Fill(iTime);

  Float_t fTree[7];

  //   b->GetEntry(EVT);
  EBDetId ID;
  Float_t theAPD;
  Float_t thePN;
  Float_t Jitter;
  Float_t Chi2;
  Int_t CN = hits->size();
  // 			cout<<"num CN: "<<CN<<endl;
  for (int j = 0; j < CN; j++) {
    ID = (*hits)[j].id();
    theAPD = (*hits)[j].amplitude();
    Jitter = (*hits)[j].jitter();
    Chi2 = (*hits)[j].chi2();
    thePN = PN_amp[(ID.ic() + 299) / 400];

    // 				cout<<"THE APD: "<<theAPD<<endl;
    // 				cout<<"THE PN: "<<thePN<<endl;

    C_APD[ID.ic() - 1]->Fill(theAPD);
    C_APDPN[ID.ic() - 1]->Fill(theAPD / thePN);
    C_PN[ID.ic() - 1]->Fill(thePN);
    C_J[ID.ic() - 1]->Fill(Jitter);
    C_APDPN_J[ID.ic() - 1]->Fill(Jitter, theAPD / thePN);
    APDPN_J->Fill(Jitter, theAPD / thePN);
    APDPN_C->Fill(Chi2, theAPD / thePN);
    fTree[0] = theAPD;
    fTree[1] = thePN;
    fTree[2] = theAPD / thePN;
    fTree[3] = Jitter;
    fTree[4] = Chi2;
    fTree[5] = (*hits)[j].pedestal();
    fTree[6] = iEvent.id().event();
    C_Tree[ID.ic() - 1]->Fill(fTree);
    if (((ID.ic() - 1) % 20 > 9) || ((ID.ic() - 1) < 100)) {
      peakAPD[0]->Fill(theAPD);
      peakAPDPN[0]->Fill(theAPD / thePN);
      APDPN_J_H[0]->Fill(Jitter, theAPD / thePN);
    } else {
      peakAPD[1]->Fill(theAPD);
      peakAPDPN[1]->Fill(theAPD / thePN);
      APDPN_J_H[1]->Fill(Jitter, theAPD / thePN);
    }
    if ((ID.ic() - 1) < 100) {
      APD_LM[0]->Fill(theAPD);
      APDPN_LM[0]->Fill(theAPD / thePN);
      APDPN_J_LM[0]->Fill(Jitter, theAPD / thePN);
    } else {
      Int_t index;
      if (((ID.ic() - 1) % 20) < 10) {
        index = ((ID.ic() - 101) / 400) * 2 + 1;
        APD_LM[index]->Fill(theAPD);
        APDPN_LM[index]->Fill(theAPD / thePN);
        APDPN_J_LM[index]->Fill(Jitter, theAPD / thePN);
      } else {
        index = ((ID.ic() - 101) / 400) * 2 + 2;
        APD_LM[index]->Fill(theAPD);
        APDPN_LM[index]->Fill(theAPD / thePN);
        APDPN_J_LM[index]->Fill(Jitter, theAPD / thePN);
      }
    }
  }  //end of CN loop

  //now that you got the PN and APD's, make the ntuples. done

  //vec from ROOT version should correspond to hits_itr or something similar. done

  //check WL from PNdiodedigi, should be ==0, o.w (blue data only). don't process. done

  //get PN pulse, and do fitting of pulse. i.e. fill hist with PN.apd() or equivalent. done

  //fit to first 5 for PED, and 30-50 bins for pulse (poly2 for the moment). done
}

// ------------ method called once each job just before starting event loop  ------------
void EcalLaserAnalyzerYousi::beginJob() {
  edm::LogInfo("EcalLaserAnalyzerYousi") << "running laser analyzer \n\n";

  fROOT = new TFile(outFileName_.c_str(), "RECREATE");
  fROOT->cd();

  //init all the histos and files?
  APD = new TH2F("APD", "APD", 85, 0., 85., 20, 0., 20.);
  APD_RMS = new TH2F("APD_RMS", "APD_RMS", 85, 0., 85., 20, 0., 20.);
  APDPN = new TH2F("APDPN", "APDPN", 85, 0., 85., 20, 0., 20.);
  APDPN_RMS = new TH2F("APDPN_RMS", "APDPN_RMS", 85, 0., 85., 20, 0., 20.);
  PN = new TH2F("PN", "PN", 85, 0., 85., 20, 0., 20.);
  APDPN_J = new TH2F("JittervAPDPN", "JittervAPDPN", 250, 3., 7., 250, 1., 2.);
  APDPN_C = new TH2F("Chi2vAPDPN", "Chi2vAPDPN", 250, 0., 50., 250, 0., 5.0);
  FitHist = new TH1F("FitHist", "FitHist", 50, 0, 50);
  Count = new TH2F("Count", "Count", 85, 0., 1., 20, 0., 1.);

  for (int i = 0; i < 1700; i++) {
    std::ostringstream name_1;
    std::ostringstream name_2;
    std::ostringstream name_3;
    std::ostringstream name_4;
    std::ostringstream name_5;
    name_1 << "C_APD_" << i + 1;
    name_2 << "C_APDPN_" << i + 1;
    name_3 << "C_PN_" << i + 1;
    name_4 << "C_J_" << i + 1;
    name_5 << "C_APDPN_J_" << i + 1;
    C_APD[i] = new TH1F(name_1.str().c_str(), name_1.str().c_str(), 2500, 0., 5000.);
    C_APDPN[i] = new TH1F(name_2.str().c_str(), name_2.str().c_str(), 20000, 0., 25.);
    C_PN[i] = new TH1F(name_3.str().c_str(), name_3.str().c_str(), 1000, 0., 4000.);
    C_J[i] = new TH1F(name_4.str().c_str(), name_4.str().c_str(), 250, 3.0, 7.);
    C_APDPN_J[i] = new TH2F(name_5.str().c_str(), name_5.str().c_str(), 250, 3.0, 6., 250, 1., 2.2);
  }

  for (int i = 0; i < 2; i++) {
    std::ostringstream aname_1;
    std::ostringstream aname_2;
    std::ostringstream aname_3;
    aname_1 << "peakAPD_" << i;
    aname_2 << "peakAPDPN_" << i;
    aname_3 << "JittervAPDPN_Half_" << i;
    peakAPD[i] = new TH1F(aname_1.str().c_str(), aname_1.str().c_str(), 1000, 0., 5000.);
    peakAPDPN[i] = new TH1F(aname_2.str().c_str(), aname_2.str().c_str(), 1000, 0., 8.);
    APDPN_J_H[i] = new TH2F(aname_3.str().c_str(), aname_3.str().c_str(), 250, 3., 7., 250, 1., 2.2);
  }

  for (int i = 0; i < 9; i++) {
    std::ostringstream bname_1;
    std::ostringstream bname_2;
    std::ostringstream bname_3;
    bname_1 << "APD_LM_" << i;
    bname_2 << "APDPN_LM_" << i;
    bname_3 << "APDPN_J_LM_" << i;
    APD_LM[i] = new TH1F(bname_1.str().c_str(), bname_1.str().c_str(), 500, 0., 5000.);
    APDPN_LM[i] = new TH1F(bname_2.str().c_str(), bname_2.str().c_str(), 500, 0., 8.);
    APDPN_J_LM[i] = new TH2F(bname_3.str().c_str(), bname_3.str().c_str(), 250, 3., 7., 250, 1., 2.2);
  }

  //get the PN file. or don't get, and read from event.

  //don't need to get AB, it will be read in via framework poolsource = ???

  //configure the final NTuple
  std::ostringstream varlist;
  varlist << "APD:PN:APDPN:Jitter:Chi2:ped:EVT";
  for (int i = 0; i < 1700; i++) {
    std::ostringstream name;
    name << "C_Tree_" << i + 1;
    C_Tree[i] = (TNtuple *)new TNtuple(name.str().c_str(), name.str().c_str(), varlist.str().c_str());
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void EcalLaserAnalyzerYousi::endJob() {
  //write the file (get ouput file name first).
  TFile *fROOT = (TFile *)new TFile(outFileName_.c_str(), "RECREATE");

  //  TDirectory *DIR = fROOT->Get(Run_.c_str());
  TDirectory *DIR;
  //  if(DIR == NULL){
  DIR = fROOT->mkdir(Run_.c_str());
  //  }
  DIR->cd();
  for (int j = 0; j < 1700; j++) {
    Float_t min_r, max_r;
    Float_t RMS, Sigma, K;
    Int_t iCount;
    TF1 *gs3;

    RMS = C_APD[j]->GetRMS();
    APD_RMS->SetBinContent(85 - (j / 20), 20 - (j % 20), RMS);
    Sigma = 999999;
    K = 2.5;
    iCount = 0;
    while (Sigma > RMS) {
      min_r = C_APD[j]->GetBinCenter(C_APD[j]->GetMaximumBin()) - K * RMS;
      max_r = C_APD[j]->GetBinCenter(C_APD[j]->GetMaximumBin()) + K * RMS;
      TF1 gs1("gs1", "gaus", min_r, max_r);
      C_APD[j]->Fit(&gs1, "RQI");
      Sigma = gs1.GetParameter(2);
      K = K * 1.5;
      iCount++;
      if (iCount > 2) {
        C_APD[j]->Fit("gaus", "QI");
        break;
      }
    }

    RMS = C_APDPN[j]->GetRMS();
    APDPN_RMS->SetBinContent(85 - (j / 20), 20 - (j % 20), RMS);
    Sigma = 999999;
    K = 2.5;
    iCount = 0;
    while (Sigma > RMS) {
      min_r = C_APDPN[j]->GetBinCenter(C_APDPN[j]->GetMaximumBin()) - K * RMS;
      max_r = C_APDPN[j]->GetBinCenter(C_APDPN[j]->GetMaximumBin()) + K * RMS;
      TF1 gs2("gs2", "gaus", min_r, max_r);
      C_APDPN[j]->Fit(&gs2, "RQI");
      Sigma = gs2.GetParameter(2);
      K = K * 1.5;
      iCount++;
      if (iCount > 2) {
        C_APDPN[j]->Fit("gaus", "QI");
        break;
      }
    }

    TF1 *newgs1;
    TF1 *newgs2;

    C_PN[j]->Fit("gaus", "Q");
    C_APD[j]->Fit("gaus", "QI");
    C_APDPN[j]->Fit("gaus", "QI");
    C_APD[j]->Write("", TObject::kOverwrite);
    C_APDPN[j]->Write("", TObject::kOverwrite);
    C_PN[j]->Write("", TObject::kOverwrite);
    C_J[j]->Write("", TObject::kOverwrite);
    C_APDPN_J[j]->Write("", TObject::kOverwrite);
    newgs1 = C_APD[j]->GetFunction("gaus");
    newgs2 = C_APDPN[j]->GetFunction("gaus");
    gs3 = C_PN[j]->GetFunction("gaus");
    Float_t theAPD = newgs1->GetParameter(1);
    APD->SetBinContent(85 - (j / 20), 20 - (j % 20), theAPD);
    Float_t theAPDPN = newgs2->GetParameter(1);
    APDPN->SetBinContent(85 - (j / 20), 20 - (j % 20), theAPDPN);
    Float_t thePN = gs3->GetParameter(1);
    //		cout<<"LOOK HERE thePN = "<< thePN<<endl;
    PN->SetBinContent(85 - (j / 20), 20 - (j % 20), thePN);
    C_Tree[j]->Write("", TObject::kOverwrite);
  }

  for (int i = 0; i < 9; i++) {
    APD_LM[i]->Write("", TObject::kOverwrite);
    APDPN_LM[i]->Write("", TObject::kOverwrite);
    APDPN_J_LM[i]->Write("", TObject::kOverwrite);
  }

  //  Time->Write("", TObject::kOverwrite);
  APD->Write("", TObject::kOverwrite);
  APD_RMS->Write("", TObject::kOverwrite);
  APDPN_RMS->Write("", TObject::kOverwrite);
  APDPN->Write("", TObject::kOverwrite);
  APDPN_J->Write("", TObject::kOverwrite);
  APDPN_C->Write("", TObject::kOverwrite);
  PN->Write("", TObject::kOverwrite);
  peakAPD[0]->Write("", TObject::kOverwrite);
  peakAPD[1]->Write("", TObject::kOverwrite);
  peakAPDPN[0]->Write("", TObject::kOverwrite);
  peakAPDPN[1]->Write("", TObject::kOverwrite);
  APDPN_J_H[0]->Write("", TObject::kOverwrite);
  APDPN_J_H[1]->Write("", TObject::kOverwrite);

  // don't Make plots
  //  fROOT->Close();

  //   fPN->Close();
  //   fAPD->Close();

  fROOT->Write();
  //   fROOT->Close();
}

//define this as a plug-in
DEFINE_FWK_MODULE(EcalLaserAnalyzerYousi);
