
#ifndef HLTriggerOffline_HLTEffCalculator_H
#define HLTriggerOffline_HLTEffCalculator_H 


// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "HLTrigger/HLTfilters/interface/HLTHighLevel.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "HLTriggerOffline/Top/interface/TopHLTDQMHelper.h"


#include "TDirectory.h"
#include "TH1F.h"
#include "TVector3.h"
#include "TLorentzVector.h"
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <TMath.h>
#include "TFile.h"
#include "TH1.h"

class EfficiencyHandler{
public:
	EfficiencyHandler(std::string Name, const std::vector<std::string>& pathnames, int verb = 0):name(Name),verbosity(verb){
		int nPaths = (int)pathnames.size();
		efficiencies= new TH1D(name.c_str(),"efficiencies per path",nPaths,-0.5,(double)nPaths-0.5);
		std::stringstream s;
		for(int i = 0; i < nPaths; i++){
			pathNames.push_back(pathnames.at(i));
			efficiencies->GetXaxis()->SetBinLabel(i+1,pathnames.at(i).c_str());
			s.str("");
			s<<"path_"<<i+1;
			numerator.push_back(new TH1D((s.str()+"_num").c_str(),(s.str()+"_num").c_str(),1,-0.5,0.5));
			denominator.push_back(new TH1D((s.str()+"_den").c_str(),(s.str()+"_den").c_str(),1,-0.5,0.5));
		}
	}
	~EfficiencyHandler(){}
	void Fill(const edm::Event& event, const edm::TriggerResults& triggerTable){
		for(unsigned int i = 0; i< pathNames.size(); i++){
			denominator.at(i)->Fill(0);
			if(verbosity > 0)
				std::cout<<pathNames.at(i)<<std::endl;				
			if(acceptHLT(event, triggerTable,pathNames.at(i)))
				numerator.at(i)->Fill(0);
		}
	}
	void WriteAll(TDirectory * d){
		if(d == NULL){
			std::cout<<"NULL directory! Cannot write!"<<std::endl;
			return;
		}
		if((int)efficiencies->GetXaxis()->GetNbins() != (int)denominator.size()){
			std::cout<<"HLT path numbers mismatch!"<<std::endl;
			return;
		}
		for(unsigned int s = 0; s < pathNames.size(); s++){
			double eff = (double)numerator.at(s)->GetEntries();
			eff= eff/(double)denominator.at(s)->GetEntries();
			efficiencies->SetBinContent(s+1, eff);
		}
		(d->mkdir(std::string(name+"_BareNumberHists").c_str()))->cd();
                for(unsigned int s = 0; s < pathNames.size(); s++){
			numerator.at(s)->Write();
			denominator.at(s)->Write();
		}
		d->cd();
		efficiencies->Write();
		d->cd();
	}
private:
	TH1* efficiencies;
	std::vector<TH1*> numerator; //now just a number holder. for possible extention in future
	std::vector<TH1*> denominator; //now just a number holder. for possible extention in future
	std::string name;
	std::vector<std::string> pathNames;
	int verbosity;
};

//
// class decleration
//

class HLTEffCalculator : public edm::EDAnalyzer {
public:
  explicit HLTEffCalculator(const edm::ParameterSet&);
  ~HLTEffCalculator();
  
  
private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
 
 // edm::Service<TFileService> fs;
  
  
  // ----------member data ---------------------------
    
  std::string outputFileName ;
  EfficiencyHandler * myEffHandler;
  edm::InputTag HLTresCollection;        
  int verbosity ;	 
  
};


#endif
