// -*- C++ -*-
//
// Package:    DQMAnalyzerSTEP2
// Class:      DQMAnalyzerSTEP2
// 
/**\class DQMAnalyzerSTEP2 DQMAnalyzerSTEP2.cc DQMAnalyzer/DQMAnalyzerSTEP2/plugins/DQMAnalyzerSTEP2.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Cesare Calabria
//         Created:  Wed, 06 Nov 2013 11:28:14 GMT
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <memory>
#include <string>

#include "TEfficiency.h"

//
// class declaration
//

class DQMAnalyzerSTEP2 : public edm::EDAnalyzer {
   public:
      explicit DQMAnalyzerSTEP2(const edm::ParameterSet&);
      ~DQMAnalyzerSTEP2();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

      MonitorElement * ChargeMisIDVsPt;
      MonitorElement * EfficiencyVsPt;
      MonitorElement * InvPtResVsPt;
      MonitorElement * RmsVsPt;
      MonitorElement * EfficiencyVsEta[7];
      MonitorElement * InvPtResVsEta[7];
      MonitorElement * RmsVsEta[7];

   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

      virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------

      typedef std::vector<std::string> vstring;

      std::string globalFolder_;
      vstring localFolder_;
      bool SaveFile_;
      std::string NameFile_;
      DQMStore* dbe_;

};

TEfficiency * calcEff(TH1F * h1, TH1F * h2){

 	TEfficiency* eff = 0;
	//std::cout<<h1->GetEntries()<<" "<<h2->GetEntries()<<std::endl;
	if(TEfficiency::CheckConsistency(*h1,*h2)) eff = new TEfficiency(*h1,*h2);

	return eff;

}

TEfficiency * calcChargeMisID(TH2F * histo){

	TH1D * DeltaChargePercentage = histo->ProjectionX("chargeMisID",1,1);
	DeltaChargePercentage->Reset();
  	TH1F * numGem = (TH1F*)DeltaChargePercentage->Clone();
  	TH1F * denGem = (TH1F*)DeltaChargePercentage->Clone();
  	for(int i=1; i<=histo->GetNbinsX(); i++){

		int num1 = histo->GetBinContent(i,2);
		int num2 = histo->GetBinContent(i,4);
		int num3 = histo->GetBinContent(i,6);

		numGem->SetBinContent(i,num1+num3);
		denGem->SetBinContent(i,num1+num2+num3);
		
		//if((num1+num3) != 0) std::cout<<num1<<" "<<num2<<" "<<num3<<std::endl;

  	}

	TEfficiency * tmp = calcEff(numGem, denGem);
	return tmp;

}

TH1F * convertTEff(TEfficiency * histo){

  	TH1F * tmp = (TH1F*)histo->GetTotalHistogram();

	for(int i=1; i<tmp->GetSize(); i++){

		double eff = histo->GetEfficiency(i);
		double err = (histo->GetEfficiencyErrorLow(i) + histo->GetEfficiencyErrorUp(i))/2;

		//std::cout<<"eff: "<<eff<<" err: "<<err<<std::endl;

		tmp->SetBinContent(i,eff);
		tmp->SetBinError(i,err);

	}
	return tmp;

}

std::vector<TH1D*> produceQoPPlots(TH2F * histo){

	std::vector<TH1D*> tmp;
	tmp.clear();
     	//std::cout<<histo->GetEntries()<<std::endl;

	TH2F * histoClone = (TH2F*)histo->Clone();
	TH1D * projX = histoClone->ProjectionX("projX",1,1);
	TH1D * plotRms = (TH1D*)projX->Clone();
	//std::cout<<projX->GetSize()<<" "<<plotRms->GetSize()<<std::endl;

	for(int i=1; i<histo->GetNbinsX(); i++){

  		TH1D * projY = histoClone->ProjectionY("projY",i,i);
		if((projY->GetEntries()) == 0) continue;

		double mean = projY->GetMean();
		double rms = projY->GetRMS();

        	TF1 *myfitFR = new TF1("myfitFR","gaus", -10, +10);
        	projY->Fit("myfitFR");
        	TF1 *myfit = new TF1("myfit","gaus", mean-2*rms, mean+2*rms);
        	projY->Fit("myfit", "R");

		double sigma = myfit->GetParameter(2);
		double sigmaErr = myfit->GetParError(2);
		double sigmaFR = myfitFR->GetParameter(2);
		//double sigmaErrFR = myfitFR->GetParError(2);
		double delta = abs(sigma - sigmaFR);
		double DeltaSigma = sqrt(delta*delta + sigmaErr*sigmaErr);

		projX->SetBinContent(i,sigma);
		projX->SetBinError(i,DeltaSigma);

		plotRms->SetBinContent(i,rms);

		//std::cout<<i<<" "<<projX->GetBinContent(i)<<" "<<projX->GetBinError(i)<<" "<<plotRms->GetBinContent(i)<<std::endl;
		//std::cout<<i<<" "<<sigma<<" "<<DeltaSigma<<" "<<rms<<std::endl;

	}

	//std::cout<<projX->GetEntries()<<std::endl;
	//std::cout<<plotRms->GetEntries()<<std::endl;
	tmp.push_back(projX);
	tmp.push_back(plotRms);
	return tmp;

}

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
DQMAnalyzerSTEP2::DQMAnalyzerSTEP2(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed

  globalFolder_ = iConfig.getUntrackedParameter<std::string>("GlobalFolder", "GEMBasicPlots/");
  localFolder_ = iConfig.getUntrackedParameter<std::vector<std::string> >("LocalFolder", {"5GeV","10GeV","50GeV","100GeV","200GeV","500GeV","1000GeV"});
  SaveFile_ = iConfig.getUntrackedParameter<bool>("SaveFile", false);
  NameFile_ = iConfig.getUntrackedParameter<std::string>("NameFile","GEMPlots.root");

}


DQMAnalyzerSTEP2::~DQMAnalyzerSTEP2()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
DQMAnalyzerSTEP2::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
}


// ------------ method called once each job just before starting event loop  ------------
void 
DQMAnalyzerSTEP2::beginJob()
{

  dbe_ = edm::Service<DQMStore>().operator->();

}

// ------------ method called once each job just after ending the event loop  ------------
void 
DQMAnalyzerSTEP2::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------

void 
DQMAnalyzerSTEP2::beginRun(edm::Run const&, edm::EventSetup const&)
{

  if(dbe_ == 0) return;

  dbe_->setCurrentFolder(globalFolder_);
  ChargeMisIDVsPt = dbe_->book1D("ChargeMisIDVsPt","Charge Mis-ID Prob. vs p_{T}",261,-2.5,1302.5);
  EfficiencyVsPt = dbe_->book1D("EfficiencyVsPt","Efficiency vs. p_{T}",261,-2.5,1302.5);
  InvPtResVsPt = dbe_->book1D("InvPtResVsPt","q/p core width vs. p_{T}",261,-2.5,1302.5);
  RmsVsPt = dbe_->book1D("RmsVsPt","q/p RMS vs. p_{T}",261,-2.5,1302.5);

  ChargeMisIDVsPt->setAxisTitle("p_{T}^{Sim} [GeV/c]");
  EfficiencyVsPt->setAxisTitle("p_{T}^{Sim} [GeV/c]");
  InvPtResVsPt->setAxisTitle("p_{T}^{Sim} [GeV/c]");
  RmsVsPt->setAxisTitle("p_{T}^{Sim} [GeV/c]");

  for(int i = 0; i < (int)localFolder_.size(); i++){

    	std::stringstream meName;
    	std::stringstream meTitle;
    	meName.str("");
    	meTitle.str("");
    	meName<<"EfficiencyVsEta_"<<localFolder_[i];
    	meTitle<<"Efficiency vs. #eta ("<<localFolder_[i]<<")";
  	EfficiencyVsEta[i] = dbe_->book1D(meName.str(),meTitle.str(),100,-2.5,+2.5);
	EfficiencyVsEta[i]->setAxisTitle("#eta^{Sim}");

    	meName.str("");
    	meTitle.str("");
    	meName<<"InvPtResVsEta_"<<localFolder_[i];
    	meTitle<<"q/p core width vs. #eta ("<<localFolder_[i]<<")";
  	InvPtResVsEta[i] = dbe_->book1D(meName.str(),meTitle.str(),100,-2.5,+2.5);
	InvPtResVsEta[i]->setAxisTitle("#eta^{Sim}");

    	meName.str("");
    	meTitle.str("");
    	meName<<"RmsVsEta_"<<localFolder_[i];
    	meTitle<<"q/p RMS vs. #eta ("<<localFolder_[i]<<")";
  	RmsVsEta[i] = dbe_->book1D(meName.str(),meTitle.str(),100,-2.5,+2.5);
	RmsVsEta[i]->setAxisTitle("#eta^{Sim}");

  }

}


// ------------ method called when ending the processing of a run  ------------

void 
DQMAnalyzerSTEP2::endRun(edm::Run const&, edm::EventSetup const&)
{

   for(int i = 0; i < (int)localFolder_.size(); i++){

	   //Efficiency vs. SimPt

	   std::stringstream meName1;
	   MonitorElement * myMe1;

	   std::stringstream meName2;
	   MonitorElement * myMe2;

	   meName1.str("");
	   meName1<<globalFolder_<<"SingleMu"<<localFolder_[i]<<"/NumSimPt";

	   meName2.str("");
	   meName2<<globalFolder_<<"SingleMu"<<localFolder_[i]<<"/DenSimPt";

	   std::cout<<meName1.str()<<" "<<meName2.str()<<std::endl;

	   myMe1 = dbe_->get(meName1.str());
	   myMe2 = dbe_->get(meName2.str());

	   if(myMe1 && myMe2){

		TH1F * histo1 = myMe1->getTH1F();
		TH1F * histo2 = myMe2->getTH1F();

		TEfficiency * effVsPt = calcEff(histo1, histo2);
		TH1F * effVsPt2 = convertTEff(effVsPt);

		for(int j=1; j<effVsPt2->GetSize(); j++){

			double eff = effVsPt2->GetBinContent(j);
			double err = effVsPt2->GetBinError(j);
			if(eff == 0) continue;
			EfficiencyVsPt->setBinContent(j,eff);
			EfficiencyVsPt->setBinError(j,err);

		}

	   }

	   //Charge mis-ID vs. SimPt

	   std::stringstream meName3;
	   MonitorElement * myMe3;

	   meName3.str("");
	   meName3<<globalFolder_<<"SingleMu"<<localFolder_[i]<<"/DeltaCharge";

	   myMe3 = dbe_->get(meName3.str());

    	   if(myMe3){

		TH2F * histo3 = myMe3->getTH2F();

		TEfficiency * chargeMisID = calcChargeMisID(histo3);
		TH1F * chargeMisID2 = convertTEff(chargeMisID);
		//std::cout<<"charge vs pt"<<std::endl;
		for(int j=1; j<chargeMisID2->GetSize(); j++){

			double eff = chargeMisID2->GetBinContent(j);
			double err = chargeMisID2->GetBinError(j);
			if(eff == 0) continue;
			ChargeMisIDVsPt->setBinContent(j,eff);
			ChargeMisIDVsPt->setBinError(j,err);

		}

	   }

	   //q/p res. and RMS vs. SimPt
		
	   std::stringstream meName4;
	   MonitorElement * myMe4;

	   meName4.str("");
	   meName4<<globalFolder_<<"SingleMu"<<localFolder_[i]<<"/InvPtResVsPt";

	   myMe4 = dbe_->get(meName4.str());

    	   if(myMe4){

	   	std::vector<TH1D*> result;
		result.clear();
		TH2F * histo4 = myMe4->getTH2F();
		//std::cout<<histo4->GetEntries()<<std::endl;

		result = produceQoPPlots(histo4);

		for(int j=1; j<result[0]->GetSize(); j++){

			double sigmaVal = result[0]->GetBinContent(j);
			double sigmaErrVal = result[0]->GetBinError(j);
			double rmsVal = result[1]->GetBinContent(j);
			if(sigmaVal == 0) continue;
			InvPtResVsPt->setBinContent(j,sigmaVal);
			InvPtResVsPt->setBinError(j,sigmaErrVal);
			RmsVsPt->setBinContent(j,rmsVal);

		}

	   }

	   //Efficiency vs. SimEta

	   std::stringstream meName5;
	   MonitorElement * myMe5;

	   std::stringstream meName6;
	   MonitorElement * myMe6;

	   meName5.str("");
	   meName5<<globalFolder_<<"SingleMu"<<localFolder_[i]<<"/NumSimEta";

	   meName6.str("");
	   meName6<<globalFolder_<<"SingleMu"<<localFolder_[i]<<"/DenSimEta";

	   myMe5 = dbe_->get(meName5.str());
	   myMe6 = dbe_->get(meName6.str());

	   if(myMe5 && myMe6){

		TH1F * histo1 = myMe5->getTH1F();
		TH1F * histo2 = myMe6->getTH1F();

		TEfficiency * effVsPt = calcEff(histo1, histo2);
		TH1F * effVsPt2 = convertTEff(effVsPt);

		for(int j=1; j<effVsPt2->GetSize(); j++){

			double eff = effVsPt2->GetBinContent(j);
			double err = effVsPt2->GetBinError(j);
			if(eff == 0) continue;
			EfficiencyVsEta[i]->setBinContent(j,eff);
			EfficiencyVsEta[i]->setBinError(j,err);

		}

	   }

	   //q/p res. and RMS vs. SimEta

	   std::stringstream meName7;
	   MonitorElement * myMe7;

	   meName7.str("");
	   meName7<<globalFolder_<<"SingleMu"<<localFolder_[i]<<"/InvPtResVsEta";

	   myMe7 = dbe_->get(meName7.str());

    	   if(myMe7){

	   	std::vector<TH1D*> result;
		TH2F * histo7 = myMe7->getTH2F();
		//std::cout<<histo7->GetEntries()<<std::endl;

		result = produceQoPPlots(histo7);

		//std::cout<<result[0]->GetEntries()<<" "<<result[1]->GetEntries()<<" "<<result.size()<<std::endl;

		for(int j=1; j<result[0]->GetSize(); j++){

			double sigmaVal = result[0]->GetBinContent(j);
			double sigmaErrVal = result[0]->GetBinError(j);
			double rmsVal = result[1]->GetBinContent(j);
			if(sigmaVal == 0) continue;
			InvPtResVsEta[i]->setBinContent(j,sigmaVal);
			InvPtResVsEta[i]->setBinError(j,sigmaErrVal);
			RmsVsEta[i]->setBinContent(j,rmsVal);

		}

	   }

   }

   if(SaveFile_) dbe_->save(NameFile_);

}

// ------------ method called when starting to processes a luminosity block  ------------
/*
void 
DQMAnalyzerSTEP2::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
DQMAnalyzerSTEP2::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
DQMAnalyzerSTEP2::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DQMAnalyzerSTEP2);
