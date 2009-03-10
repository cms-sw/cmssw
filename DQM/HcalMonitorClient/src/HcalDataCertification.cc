// -*- C++ -*-
//
// Package:    DQMO/HcalMonitorClient/HcalDataCertification
// Class:      HcalDataCertification
// 
/**\class HcalDataCertification HcalDataCertification.cc DQM/HcalMonitorClient/src/HcalDataCertification.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  "Igor Vodopiyanov"
//         Created:  Nov-21 2008
// $Id: HcalDataCertification.cc,v 1.11 2008/11/21 01:11:01 ivodop Exp $
//
//

// system include files
#include <memory>
#include <stdio.h>
#include <math.h>
#include <sstream>
#include <fstream>
#include <exception>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// class decleration
//

class HcalDataCertification : public edm::EDAnalyzer {
   public:
      explicit HcalDataCertification(const edm::ParameterSet&);
      ~HcalDataCertification();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      virtual void beginRun(const edm::Run&, const edm::EventSetup&) ;
      virtual void endRun(const edm::Run&, const edm::EventSetup&) ;

   // ----------member data ---------------------------

   edm::ParameterSet conf_;
   DQMStore * dbe;
   edm::Service<TFileService> fs_;

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

HcalDataCertification::HcalDataCertification(const edm::ParameterSet& iConfig):conf_(iConfig)
{
  // now do what ever initialization is needed
}


HcalDataCertification::~HcalDataCertification()
{ 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
HcalDataCertification::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
   
#ifdef THIS_IS_AN_EVENT_EXAMPLE
  Handle<ExampleData> pIn;
  iEvent.getByLabel("example",pIn);
#endif
   
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
  ESHandle<SetupData> pSetup;
  iSetup.get<SetupRecord>().get(pSetup);
#endif

}

// ------------ method called once each job just before starting event loop  ------------
void 
HcalDataCertification::beginJob(const edm::EventSetup&)
{
  dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();
  //std::cout<<"beginJob"<< std::endl;
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HcalDataCertification::endJob() 
{
  //std::cout << ">>> endJob " << std::endl;
}

// ------------ method called just before starting a new run  ------------
void 
HcalDataCertification::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  //std::cout<<"beginRun"<<std::endl;
}

// ------------ method called right after a run ends ------------
void 
HcalDataCertification::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  
  dbe->setCurrentFolder("Hcal");
  std::string currDir = dbe->pwd();
  //std::cout << "--- Current Directory " << currDir << std::endl;
  std::vector<MonitorElement*> mes = dbe->getAllContents("");
  //std::cout << "found " << mes.size() << " monitoring elements:" << std::endl;

  dbe->setCurrentFolder("Hcal/EventInfo/Certification/");
  MonitorElement* Hcal_HB = dbe->bookFloat("Hcal_HB");
  MonitorElement* Hcal_HE = dbe->bookFloat("Hcal_HE");
  MonitorElement* Hcal_HF = dbe->bookFloat("Hcal_HF");
  MonitorElement* Hcal_HO = dbe->bookFloat("Hcal_HO");

  Hcal_HB->Fill(0);
  Hcal_HE->Fill(0);
  Hcal_HF->Fill(0);
  Hcal_HO->Fill(0);

  Int_t ndeadHB,ndeadHE,ndeadHF,ndeadHO,nhotHB,nhotHE,nhotHF,nhotHO;
  ndeadHB=ndeadHE=ndeadHF=ndeadHO=nhotHB=nhotHE=nhotHF=nhotHO=0;
  Int_t nenHB,nenHE,nenHF,nenHO;
  nenHB=nenHE=nenHF=nenHO=0;

  TH1F *hTP;
  TH2F *hhotHB,*hhotHF,*hhotHE,*hhotHO;
  TH2F *hdeadHB,*hdeadHF,*hdeadHE,*hdeadHO;
  TH2F *henergeticHB,*henergeticHE,*henergeticHF,*henergeticHO;


  // ************** Loop over Monitoring Elements and fill working histograms
  Int_t nWorkHist = 0;
  for(std::vector<MonitorElement*>::const_iterator ime = mes.begin(); ime!=mes.end(); ++ime) {
    std::string name = (*ime)->getName();

    // Hcal Nevents by Trig

    if (name == "# TP Digis") {
      hTP = (*ime)->getTH1F();
      //std::cout <<"TP Digis"<<std::endl;
      nWorkHist++;
    }

    // by SubDet

    if (name == "HB_OccupancyMap_HotCell_Threshold0") { // -----------Hot
      hhotHB = (*ime)->getTH2F();
      //std::cout <<"HB_OccupancyMap_HotCell_Threshold0"<<std::endl;
      nWorkHist++;
    }
    if (name == "HE_OccupancyMap_HotCell_Threshold0") { 
      hhotHE = (*ime)->getTH2F();
      //std::cout <<"HE_OccupancyMap_HotCell_Threshold0"<<std::endl;
      nWorkHist++;
    }
    if (name == "HF_OccupancyMap_HotCell_Threshold0") { 
      hhotHF = (*ime)->getTH2F();
      //std::cout <<"HF_OccupancyMap_HotCell_Threshold0"<<std::endl;
      nWorkHist++;
    }
    if (name == "HO_OccupancyMap_HotCell_Threshold0") { 
      hhotHO = (*ime)->getTH2F();
      //std::cout <<"HO_OccupancyMap_HotCell_Threshold0"<<std::endl;
      nWorkHist++;
    }

    if (name == "HB_HotCell_EnergyMap_Thresh0") { // -----------Energetic
      henergeticHB = (*ime)->getTH2F();
      //std::cout <<"HB_HotCell_EnergyMap_Thresh0"<<std::endl;
      nWorkHist++;
    }
    if (name == "HE_HotCell_EnergyMap_Thresh0") { 
      henergeticHE = (*ime)->getTH2F();
      //std::cout <<"HE_HotCell_EnergyMap_Thresh0"<<std::endl;
      nWorkHist++;
    }
    if (name == "HF_HotCell_EnergyMap_Thresh0") { 
      henergeticHF = (*ime)->getTH2F();
      //std::cout <<"HF_HotCell_EnergyMap_Thresh0"<<std::endl;
      nWorkHist++;
    }
    if (name == "HO_HotCell_EnergyMap_Thresh0") { 
      henergeticHO = (*ime)->getTH2F();
      //std::cout <<"HO_HotCell_EnergyMap_Thresh0"<<std::endl;
      nWorkHist++;
    }

    if (name == "HBProblemDeadCells") { // --------------------------Dead
      hdeadHB = (*ime)->getTH2F();
      //std::cout <<"HBProblemDeadCells"<<std::endl;
      nWorkHist++;
    }
    if (name == "HEProblemDeadCells") { 
      hdeadHE = (*ime)->getTH2F();
      //std::cout <<"HEProblemDeadCells"<<std::endl;
      nWorkHist++;
    }
    if (name == "HFProblemDeadCells") { 
      hdeadHF = (*ime)->getTH2F();
      //std::cout <<"HFProblemDeadCells"<<std::endl;
      nWorkHist++;
    }
    if (name == "HOProblemDeadCells") { 
      hdeadHO = (*ime)->getTH2F();
      //std::cout <<"HOProblemDeadCells"<<std::endl;
      nWorkHist++;
    }

  }  // ******** End loop over Monitoring Elements ------------------


  if (nWorkHist<13) {
    edm::LogPrint("HcalDataCertification")<<"N Hist Found ="<<nWorkHist<<" out of 13 => return"<<std::endl;
    return;
  }

  Int_t Nevents = (int) hTP->GetEntries();   // ----- Nevents
  if (Nevents<1) {
    edm::LogPrint("HcalDataCertification")<<"N events ="<<Nevents<<" => return"<<std::endl;
    return;
  }

  TH2F *hdeadratHB = (TH2F*) hdeadHB->Clone(); // ----- dead rate
  TH2F *hdeadratHE = (TH2F*) hdeadHE->Clone(); 
  TH2F *hdeadratHF = (TH2F*) hdeadHF->Clone(); 
  TH2F *hdeadratHO = (TH2F*) hdeadHO->Clone(); 

  TH2F *hhotratHB = (TH2F*) hhotHB->Clone(); // ----- hot rate
  TH2F *hhotratHE = (TH2F*) hhotHE->Clone(); 
  TH2F *hhotratHF = (TH2F*) hhotHF->Clone(); 
  TH2F *hhotratHO = (TH2F*) hhotHO->Clone(); 

  TH2F *henergeticcellHB = (TH2F*) henergeticHB->Clone(); // ----energetic rate
  TH2F *henergeticcellHE = (TH2F*) henergeticHE->Clone(); 
  TH2F *henergeticcellHF = (TH2F*) henergeticHF->Clone(); 
  TH2F *henergeticcellHO = (TH2F*) henergeticHO->Clone(); 

  hdeadratHB->Reset();hhotratHB->Reset();henergeticcellHB->Reset();
  hdeadratHE->Reset();hhotratHE->Reset();henergeticcellHE->Reset();
  hdeadratHF->Reset();hhotratHF->Reset();henergeticcellHF->Reset();
  hdeadratHO->Reset();hhotratHO->Reset();henergeticcellHO->Reset();

  // energetic cells
  for (int ii=1;ii<=henergeticHB->GetNbinsX();ii++) for (int jj=1;jj<=henergeticHB->GetNbinsY();jj++) { 
    if (henergeticHB->GetBinContent(ii,jj)>100) {
      double encell = henergeticHB->GetBinContent(ii,jj);
      double ensumcells = 0;
      int numcells = 0;
      int lmin=jj-2;
      int lmax=jj+2;
      for (int kk=TMath::Max(1,ii-2);kk<=TMath::Min(ii+2,henergeticHB->GetNbinsX());kk++) {
	for (int ll=TMath::Max(1,lmin);ll<=TMath::Min(lmax,henergeticHB->GetNbinsY());ll++) {
	  if (henergeticHB->GetBinContent(kk,ll)>0 && (kk!=ii || ll!=jj)) {
	    ensumcells += henergeticHB->GetBinContent(kk,ll);
	    numcells++;
	  }
	}
	if (numcells>0) if (1000*encell/(ensumcells/numcells+encell)>950) {
	  henergeticcellHB->SetBinContent(ii,jj,1);
	  //cout<<"energetic HB: "<<ii<<" / "<<jj<<"  E="<<encell<<"  "<<ensumcells<<"  "<<numcells<<std::endl;
	}
      }
    }
  }
  for (int ii=1;ii<=henergeticHE->GetNbinsX();ii++) for (int jj=1;jj<=henergeticHE->GetNbinsY();jj++) { // 
    if (henergeticHE->GetBinContent(ii,jj)>100) {
      double encell = henergeticHE->GetBinContent(ii,jj);
      double ensumcells = 0;
      int numcells = 0;
      int lmin=jj-2;
      int lmax=jj+2;
      if (ii<23 || ii>63) {lmin=jj-4; lmax=jj+4;}
      for (int kk=TMath::Max(1,ii-2);kk<=TMath::Min(ii+2,henergeticHE->GetNbinsX());kk++) {
	for (int ll=TMath::Max(1,lmin);ll<=TMath::Min(lmax,henergeticHE->GetNbinsY());ll++) {
	  if (henergeticHE->GetBinContent(kk,ll)>0 && (kk!=ii || ll!=jj)) {
	    ensumcells += henergeticHE->GetBinContent(kk,ll);
	    numcells++;
	  }
	}
      }
      if (numcells>0) if (1000*encell/(ensumcells/numcells+encell)>950) {
	henergeticcellHE->SetBinContent(ii,jj,1);
	//cout<<"energetic HE: "<<ii<<" / "<<jj<<"  E="<<encell<<"  "<<ensumcells<<"  "<<numcells<<std::endl;
      }
    }
  }
  for (int ii=1;ii<=henergeticHF->GetNbinsX();ii++) for (int jj=1;jj<=henergeticHF->GetNbinsY();jj++) { // 
    if (henergeticHF->GetBinContent(ii,jj)>100) {
      double encell = henergeticHF->GetBinContent(ii,jj);
      double ensumcells = 0;
      int numcells = 0;
      int lmin=jj-4;
      int lmax=jj+4;
      if (ii<3 || ii>83) {lmin=jj-8; lmax=jj+8;}
      for (int kk=TMath::Max(1,ii-2);kk<=TMath::Min(ii+2,henergeticHF->GetNbinsX());kk++) {
	for (int ll=TMath::Max(1,lmin);ll<=TMath::Min(lmax,henergeticHF->GetNbinsY());ll++) {
	  if (henergeticHF->GetBinContent(kk,ll)>0 && (kk!=ii || ll!=jj)) {
	    ensumcells += henergeticHF->GetBinContent(kk,ll);
	    numcells++;
	  }
	}
      }
      if (numcells>0) if (1000*encell/(ensumcells/numcells+encell)>950) {
	henergeticcellHF->SetBinContent(ii,jj,1);
	//cout<<"energetic HF: "<<ii<<" / "<<jj<<"  E="<<encell<<"  "<<ensumcells<<"  "<<numcells<<std::endl;
      }
    }
  }
  for (int ii=1;ii<=henergeticHO->GetNbinsX();ii++) for (int jj=1;jj<=henergeticHO->GetNbinsY();jj++) { // 
    if (henergeticHO->GetBinContent(ii,jj)>100) {
      double encell = henergeticHO->GetBinContent(ii,jj);
      double ensumcells = 0;
      int numcells = 0;
      int lmin=jj-2;
      int lmax=jj+2;
      for (int kk=TMath::Max(1,ii-2);kk<=TMath::Min(ii+2,henergeticHO->GetNbinsX());kk++) {
	for (int ll=TMath::Max(1,lmin);ll<=TMath::Min(lmax,henergeticHO->GetNbinsY());ll++) {
	  if (henergeticHO->GetBinContent(kk,ll)>0 && (kk!=ii || ll!=jj)) {
	    ensumcells += henergeticHO->GetBinContent(kk,ll);
	    numcells++;
	  }
	}
      }
      if (numcells>0) if (1000*encell/(ensumcells/numcells+encell)>950) {
	henergeticcellHO->SetBinContent(ii,jj,1);
	//cout<<"energetic HO: "<<ii<<" / "<<jj<<"  E="<<encell<<"  "<<ensumcells<<"  "<<numcells<<std::endl;
      }
    }
  }
  nenHB= (int) henergeticcellHB->Integral();
  nenHE= (int) henergeticcellHE->Integral();
  nenHF= (int) henergeticcellHF->Integral();
  nenHO= (int) henergeticcellHO->Integral();
  //std::cout<<"Energetic HB/HE/HF/HO= "<<nenHB<<" / "<<nenHE<<" / "<<nenHF<<" / "<<nenHO<<std::endl;

  // hot and dead cell rates
  for (int ii=1;ii<=hdeadHB->GetNbinsX();ii++) for (int jj=1;jj<=hdeadHB->GetNbinsY();jj++) { // --- ratio hist
    if (hdeadHB->GetBinContent(ii,jj)>0) {
      hdeadratHB->SetBinContent(ii,jj,hdeadHB->GetBinContent(ii,jj)/Nevents);
      hhotratHB->SetBinContent(ii,jj,hhotHB->GetBinContent(ii,jj)/Nevents);
    }
  }
  for (int ii=1;ii<=hdeadHE->GetNbinsX();ii++) for (int jj=1;jj<=hdeadHE->GetNbinsY();jj++) { 
    if (hdeadHE->GetBinContent(ii,jj)>0) {
      hdeadratHE->SetBinContent(ii,jj,hdeadHE->GetBinContent(ii,jj)/Nevents);
      hhotratHE->SetBinContent(ii,jj,hhotHE->GetBinContent(ii,jj)/Nevents);
    }
  }
  for (int ii=1;ii<=hdeadHF->GetNbinsX();ii++) for (int jj=1;jj<=hdeadHF->GetNbinsY();jj++) { 
    if (hdeadHF->GetBinContent(ii,jj)>0) {
      hdeadratHF->SetBinContent(ii,jj,hdeadHF->GetBinContent(ii,jj)/Nevents);
      hhotratHF->SetBinContent(ii,jj,hhotHF->GetBinContent(ii,jj)/Nevents);
    }
  }
  for (int ii=1;ii<=hdeadHO->GetNbinsX();ii++) for (int jj=1;jj<=hdeadHO->GetNbinsY();jj++) { 
    if (hdeadHO->GetBinContent(ii,jj)>0) {
      hdeadratHO->SetBinContent(ii,jj,hdeadHO->GetBinContent(ii,jj)/Nevents);
      hhotratHO->SetBinContent(ii,jj,hhotHO->GetBinContent(ii,jj)/Nevents);
    }
  }

  for (int ii=1;ii<=hhotHB->GetNbinsX();ii++) for (int jj=1;jj<=hhotHB->GetNbinsY();jj++) { // --- counters
    if (hdeadratHB->GetBinContent(ii,jj)>0.05) ndeadHB++;
    if (hhotratHB->GetBinContent(ii,jj)>0.05 || henergeticcellHB->GetBinContent(ii,jj)>0) nhotHB++;
  }
  for (int ii=1;ii<=hhotHE->GetNbinsX();ii++) for (int jj=1;jj<=hhotHE->GetNbinsY();jj++) { 
    if (hdeadratHE->GetBinContent(ii,jj)>0.05) ndeadHE++;
    if (hhotratHE->GetBinContent(ii,jj)>0.05 || henergeticcellHE->GetBinContent(ii,jj)>0) nhotHE++;
  }
  for (int ii=1;ii<=hhotHF->GetNbinsX();ii++) for (int jj=1;jj<=hhotHF->GetNbinsY();jj++) { 
    if (hdeadratHF->GetBinContent(ii,jj)>0.05) ndeadHF++;
    if (hhotratHF->GetBinContent(ii,jj)>0.05 || henergeticcellHF->GetBinContent(ii,jj)>0) nhotHF++;
  }
  for (int ii=1;ii<=hhotHO->GetNbinsX();ii++) for (int jj=1;jj<=hhotHO->GetNbinsY();jj++) { 
    if (hdeadratHO->GetBinContent(ii,jj)>0.05) ndeadHO++;
    if (hhotratHO->GetBinContent(ii,jj)>0.05 || henergeticcellHO->GetBinContent(ii,jj)>0) nhotHO++;
  }

  //std::cout<<"Dead HB/HE/HF/HO= "<<ndeadHB<<" / "<<ndeadHE<<" / "<<ndeadHF<<" / "<<ndeadHO<<std::endl;
  //std::cout<<"Hot  HB/HE/HF/HO= "<<nhotHB<<" / "<<nhotHE<<" / "<<nhotHF<<" / "<<nhotHO<<std::endl;

  Double_t valdeadHB= 1-ndeadHB/2592.0;
  Double_t valdeadHE= 1-ndeadHE/2592.0;
  Double_t valdeadHF= 1-ndeadHF/1728.0;
  Double_t valdeadHO= 1-ndeadHO/2160.0;

  Double_t valhotHB= 1-nhotHB/2592.0;
  Double_t valhotHE= 1-nhotHE/2592.0;
  Double_t valhotHF= 1-nhotHF/1728.0;
  Double_t valhotHO= 1-nhotHO/2160.0;

  Double_t valHB = valhotHB*valdeadHB;
  Double_t valHE = valhotHE*valdeadHE;
  Double_t valHF = valhotHF*valdeadHF;
  Double_t valHO = valhotHO*valdeadHO;

  Hcal_HB->Fill(valHB);
  Hcal_HE->Fill(valHE);
  Hcal_HF->Fill(valHF);
  Hcal_HO->Fill(valHO);

  //std::cout<<"Dead   HB/HE/HF/HO= "<<valdeadHB<<" / "<<valdeadHE<<" / "<<valdeadHF<<" / "<<valdeadHO<<std::endl;
  //std::cout<<"Hot    HB/HE/HF/HO= "<<valhotHB<<" / "<<valhotHE<<" / "<<valhotHF<<" / "<<valhotHO<<std::endl;
  //std::cout<<"ResVal HB/HE/HF/HO= "<<valHB<<" / "<<valHE<<" / "<<valHF<<" / "<<valHO<<std::endl;

// ---------------------- end of certification

}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalDataCertification);
