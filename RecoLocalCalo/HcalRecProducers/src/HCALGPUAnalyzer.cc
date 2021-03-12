// -*- C++ -*-
//
// Package:    ComparisonPlots/HCALGPUAnalyzer
// Class:      HCALGPUAnalyzer
//
/**\class HCALGPUAnalyzer HCALGPUAnalyzer.cc ComparisonPlots/HCALGPUAnalyzer/plugins/HCALGPUAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Mariarosaria D'Alfonso
//         Created:  Mon, 17 Dec 2018 16:22:58 GMT
//
//

// system include files
#include <memory>
#include <string>
#include <map>
#include <iostream>
using namespace std;

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"

#include "TH2F.h"

//
// class declaration
//

class HCALGPUAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit HCALGPUAnalyzer(const edm::ParameterSet &);
  ~HCALGPUAnalyzer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override;

  // ----------member data ---------------------------
  //  void ClearVariables();

  // some variables for storing information
  double Method0Energy, Method0EnergyGPU;
  double RecHitEnergy, RecHitEnergyGPU;
  double RecHitTime, RecHitTimeGPU;
  double iEta, iEtaGPU;
  double iPhi, iPhiGPU;
  int depth, depthGPU;

  TH2F *hEnergy_2dMahi;
  TH2F *hEnergy_2dM0;
  TH2F *hTime_2dMahi;

  TH2F *Unmatched;
  TH2F *Matched;
  TH1F *hEnergy_cpu;
  TH1F *hEnergy_gpu;
  TH1F *hEnergy_cpugpu;
  TH1F *hEnergy_cpugpu_rel;
  TH1F *hEnergyM0_cpu;
  TH1F *hEnergyM0_gpu;
  TH1F *hTime_cpu;
  TH1F *hTime_gpu;

  // create the output file
  edm::Service<TFileService> FileService;
  // create the token to retrieve hit information
  edm::EDGetTokenT<HBHERecHitCollection> hRhToken;
  edm::EDGetTokenT<HBHERecHitCollection> hRhTokenGPU;
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
HCALGPUAnalyzer::HCALGPUAnalyzer(const edm::ParameterSet &iConfig) {
  usesResource("TFileService");

  hRhToken = consumes<HBHERecHitCollection>(iConfig.getUntrackedParameter<string>("HBHERecHits", "hbheprereco"));
  hRhTokenGPU = consumes<HBHERecHitCollection>(
      iConfig.getUntrackedParameter<string>("HBHERecHits", "hcalCPURecHitsProducer:recHitsLegacyHBHE"));

  //

  hEnergy_2dM0 = FileService->make<TH2F>("hEnergy_2dM0", "hEnergy_2dM0", 1000, 0., 100., 1000, 0., 100.);
  hEnergy_2dM0->GetXaxis()->SetTitle("Cpu M0 Energy");
  hEnergy_2dM0->GetYaxis()->SetTitle("GPU M0 Energy");

  hEnergy_2dMahi = FileService->make<TH2F>("hEnergy_2dMahi", "hEnergy_2dMahi", 1000, 0., 100., 1000, 0., 100.);
  hEnergy_2dMahi->GetXaxis()->SetTitle("CPU Energy");
  hEnergy_2dMahi->GetYaxis()->SetTitle("GPU Energy");

  hTime_2dMahi = FileService->make<TH2F>("hTime_2dMahi", "hTime_2dMahi", 250, -12.5, 12.5, 250, -12.5, 12.5);
  hTime_2dMahi->GetXaxis()->SetTitle("Mahi Time CPU");
  hTime_2dMahi->GetYaxis()->SetTitle("Mahi Time GPU");

  //

  hEnergyM0_cpu = FileService->make<TH1F>("hEnergyM0_cpu", "hEnergyM0_cpu", 100, 0., 100.);
  hEnergyM0_cpu->GetXaxis()->SetTitle("CPU Energy");

  hEnergy_cpu = FileService->make<TH1F>("hEnergy_cpu", "hEnergy_cpu", 50, 0., 50.);
  hEnergy_cpu->GetXaxis()->SetTitle("CPU Energy");

  hEnergy_gpu = FileService->make<TH1F>("hEnergy_gpu", "hEnergy_gpu", 50, 0., 50.);
  hEnergy_gpu->GetXaxis()->SetTitle("GPU Energy");

  //

  hEnergy_cpugpu = FileService->make<TH1F>("hEnergy_cpugpu", "hEnergy_cpugpu", 500, -2.5, 2.5);
  hEnergy_cpugpu->GetXaxis()->SetTitle("GPU Energy - CPU Energy [GeV]");
  hEnergy_cpugpu->GetYaxis()->SetTitle("# RecHits");

  hEnergy_cpugpu_rel =
      FileService->make<TH1F>("hEnergy_cpugpu_rel", "hEnergy_cpugpu_rel ( E > 0.005 GeV)", 500, -2.5, 2.5);
  hEnergy_cpugpu_rel->GetXaxis()->SetTitle("(GPU Energy - CPU Energy) / CPU energy");
  hEnergy_cpugpu_rel->GetYaxis()->SetTitle("# RecHits");

  //

  hTime_cpu = FileService->make<TH1F>("hTime_cpu", "hTime_cpu", 50, -25., 25.);
  hTime_cpu->GetXaxis()->SetTitle("CPU Time");

  hTime_gpu = FileService->make<TH1F>("hTime_gpu", "hTime_gpu", 50, -25., 25.);
  hTime_gpu->GetXaxis()->SetTitle("GPU Time");

  Unmatched = FileService->make<TH2F>("Unmatched", "Unmatched (eta,phi)", 100, -50., 50., 85, 0., 85.);
  Matched = FileService->make<TH2F>("Matched", "Matched (eta,phi)", 100, -50., 50., 85, 0., 85.);

  //now do what ever initialization is needed
}

//
// member functions
//

// ------------ method called for each event  ------------
void HCALGPUAnalyzer::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;

  // Read events
  Handle<HBHERecHitCollection> hRecHits;
  iEvent.getByToken(hRhToken, hRecHits);

  Handle<HBHERecHitCollection> hRecHitsGPU;
  iEvent.getByToken(hRhTokenGPU, hRecHitsGPU);

  // Loop over all rechits in one event
  for (int i = 0; i < (int)hRecHits->size(); i++) {
    // get ID information for the reconstructed hit
    HcalDetId detID_rh = (*hRecHits)[i].id().rawId();

    // ID information can get us detector coordinates
    depth = (*hRecHits)[i].id().depth();
    iEta = detID_rh.ieta();
    iPhi = detID_rh.iphi();

    // get some variables
    Method0Energy = (*hRecHits)[i].eraw();
    RecHitEnergy = (*hRecHits)[i].energy();
    RecHitTime = (*hRecHits)[i].time();

    hEnergy_cpu->Fill(RecHitEnergy);
    hTime_cpu->Fill(RecHitTime);

    /*
     cout << "Run " << i << ": ";
     cout << "Method0Energy: " << Method0Energy;
     cout << "RecHitEnergy: " << RecHitEnergy;
     cout << "depth: " << depth;
     cout << "iEta: " << iEta;
     cout << "iPhi: " << iPhi;
     cout << "RecHitTime" << RecHitTime;
     */
  }

  for (int i = 0; i < (int)hRecHitsGPU->size(); i++) {
    // get ID information for the reconstructed hit
    HcalDetId detID_rh = (*hRecHitsGPU)[i].id().rawId();

    // ID information can get us detector coordinates
    depthGPU = (*hRecHitsGPU)[i].id().depth();
    iEtaGPU = detID_rh.ieta();
    iPhiGPU = detID_rh.iphi();

    // get some variables
    Method0EnergyGPU = (*hRecHitsGPU)[i].eraw();
    RecHitEnergyGPU = (*hRecHitsGPU)[i].energy();
    RecHitTimeGPU = (*hRecHitsGPU)[i].time();

    hEnergy_gpu->Fill(RecHitEnergyGPU);
    hTime_gpu->Fill(RecHitTimeGPU);

    /*
     cout << "Run " << i << ": ";
     cout << "Method0Energy: " << Method0EnergyGPU;
     cout << "RecHitEnergy: " << RecHitEnergyGPU;
     cout << "depth: " << depthGPU;
     cout << "iEta: " << iEtaGPU;
     cout << "iPhi: " << iPhiGPU;
     cout << "RecHitTime" << RecHitTimeGPU;
     */
  }

  // Loop over all rechits in one event
  for (int i = 0; i < (int)hRecHits->size(); i++) {
    HcalDetId detID_rh = (*hRecHits)[i].id().rawId();

    bool unmatched = true;
    //     cout << "--------------------------------------------------------" << endl;

    for (int j = 0; j < (int)hRecHitsGPU->size(); j++) {
      HcalDetId detID_gpu = (*hRecHitsGPU)[j].id().rawId();

      if ((detID_rh == detID_gpu)) {
        /*
	 cout << "Mtime(cpu)" << (*hRecHits)[i].time() << endl; 
	 cout << "     Mtime(gpu)" << (*hRecHitsGPU)[j].time() << endl;

	 cout << "M0E(cpu)" << (*hRecHits)[i].eraw() << endl; 
	 cout << "     M0E(gpu)" << (*hRecHitsGPU)[j].eraw() << endl;
	 */

        auto relValue = ((*hRecHitsGPU)[j].energy() - (*hRecHits)[i].energy()) / (*hRecHits)[i].energy();

        hEnergy_2dM0->Fill((*hRecHits)[i].eraw(), (*hRecHitsGPU)[j].eraw());
        hEnergy_2dMahi->Fill((*hRecHits)[i].energy(), (*hRecHitsGPU)[j].energy());
        hEnergy_cpugpu->Fill((*hRecHitsGPU)[j].energy() - (*hRecHits)[i].energy());
        if ((*hRecHits)[i].energy() > 0.005)
          hEnergy_cpugpu_rel->Fill(relValue);
        hTime_2dMahi->Fill((*hRecHits)[i].time(), (*hRecHitsGPU)[j].time());

        /*
	 if((relValue < - 0.9) and ((*hRecHits)[i].energy()>0.005)) {
	   cout << "----------------------------------"<< endl;
	   cout << " detID = " << detID_rh.rawId() << endl;
	   cout << "ME(cpu)" << (*hRecHits)[i].energy() << endl; 
	   cout << "     ME(gpu)" << (*hRecHitsGPU)[j].energy() << endl;
	 }
	 */

        Matched->Fill(detID_rh.ieta(), detID_rh.iphi());

        unmatched = false;
      }
    }

    ///

    if (unmatched) {
      Unmatched->Fill(detID_rh.ieta(), detID_rh.iphi());
      //       cout << "   recHit not matched ="  << detID_rh << "  E(raw)=" << (*hRecHits)[i].eraw() << " E=" << (*hRecHits)[i].energy() << endl;
    }
  }
}

// ------------ method called once each job just before starting event loop  ------------
void HCALGPUAnalyzer::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void HCALGPUAnalyzer::endJob() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HCALGPUAnalyzer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HCALGPUAnalyzer);
