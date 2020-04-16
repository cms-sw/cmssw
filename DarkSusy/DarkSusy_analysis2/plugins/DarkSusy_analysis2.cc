// -*- C++ -*-
//
// Package:    DarkSusy/DarkSusy_analysis2
// Class:      DarkSusy_analysis2
//
/**\class DarkSusy_analysis2 DarkSusy_analysis2.cc DarkSusy/DarkSusy_analysis2/plugins/DarkSusy_analysis2.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Chiara Aime'
//         Created:  Mon, 06 Apr 2020 08:16:58 GMT
//
//


// system include files
#include <memory>
//#include "TCanvas.h"
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 #include "FWCore/Utilities/interface/InputTag.h"
 #include "DataFormats/TrackReco/interface/Track.h"
 #include "DataFormats/TrackReco/interface/TrackFwd.h"
//Chiara's addition 
#include "FWCore/Framework/interface/EventSetup.h" 
#include "FWCore/Framework/interface/ESHandle.h" 
#include "DataFormats/HepMCCandidate/interface/GenParticle.h" //to work with reco::GenParticle
#include "DataFormats/PatCandidates/interface/Muon.h" //to work with pat::Muon
#include "DataFormats/PatCandidates/interface/Jet.h" //to work with pat::Jets
#include "DataFormats/PatCandidates/interface/PackedCandidate.h" //to work with particles inside jets
#include <vector> 
#include <string> 
#include <map>
#include "FWCore/ServiceRegistry/interface/Service.h" // to use TFileService
#include "CommonTools/UtilAlgos/interface/TFileService.h" // to use TFileService
#include "FWCore/Common/interface/TriggerNames.h" // for the trigger
#include "DataFormats/Common/interface/TriggerResults.h" // for the trigger
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h" //for the trigger


//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<>
// This will improve performance in multithreaded jobs.




class DarkSusy_analysis2 : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
   public:
      explicit DarkSusy_analysis2(const edm::ParameterSet&);
      ~DarkSusy_analysis2();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

      // ----------member data ---------------------------
      edm::EDGetTokenT<std::vector<pat::Muon> > patmuonToken;  //used to select what muons to read from configuration file
      edm::EDGetTokenT<std::vector<pat::Jet> > patjetToken;  //used to select what jets to read from configuration file 
      edm::Service<TFileService> fs;
      edm::EDGetTokenT<edm::TriggerResults> triggerToken;
      // ----------histograms ----------------------------
      TH1F *h_genpt; 
      TH1F *h_jetspt; //all jets
      TH1I *h_njets; 
      TH1I *h_nmuons; 
      TH1I *h_matchedmuons; 
      TH1I *h_matchedmuonslj; 
      TH1F *h_leadingjetspt; //just leading jets
      TH1F *h_m1pt; 
      TH1F *h_m2pt; 
      TH1F *h_m3pt;  
      TH1F *h_m4pt; 
      TH1I *h_muonsnumber; 
      //TCanvas *c1;
      std::map<std::string, int> trig_counts;

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
DarkSusy_analysis2::DarkSusy_analysis2(const edm::ParameterSet& iConfig):
  patmuonToken(consumes<std::vector<pat::Muon> >(iConfig.getUntrackedParameter<edm::InputTag>("muonpat"))),
  patjetToken(consumes<std::vector<pat::Jet> >(iConfig.getUntrackedParameter<edm::InputTag>("jetpat"))),
  triggerToken(consumes<edm::TriggerResults>(iConfig.getParameter<edm::InputTag>("trigger")))
{ //now do what ever initialization is needed
    //----------histograms initialization------------------------
    h_genpt = fs->make<TH1F>("Muonpt", "High pt muons", 200, 0.0, 200.0);
    h_jetspt = fs->make<TH1F>("Jetspt", "Jets", 200, 0.0, 200.0);
    h_njets = fs->make<TH1I>("NJets", "number of Jets", 10, -0.5, 9.5);
    h_nmuons = fs->make<TH1I>("NMuons", "number of JMuons", 5, -0.5, 4.5);
    h_leadingjetspt = fs->make<TH1F>("LeadingJets", "Leading Jets pt", 200, 0.0, 200.0);
    h_matchedmuons = fs->make<TH1I>("MatchedMuons", "Matched muons", 5, -0.5, 4.5);
    h_matchedmuonslj = fs->make<TH1I>("MatchedMuonsLJ", "Matched muons in leading jet", 5, -0.5, 4.5);
    h_m1pt = fs->make<TH1F>("Muon1pt", "First muon pt", 100, 0.0, 100.0);
    h_m2pt = fs->make<TH1F>("Muon2pt", "Second muon pt", 100, 0.0, 100.0);
    h_m3pt = fs->make<TH1F>("Muon3pt", "Third muon pt", 100, 0.0, 100.0);
    h_m4pt = fs->make<TH1F>("Muon4pt", "Muon pt", 100, 0.0, 100.0);
    h_muonsnumber = fs->make<TH1I>("Muonsnumber", "number of reconstructed muons", 20, -0.5, 19.5);
    //c1 = fs->make<TCanvas>("c1","c1");
}


DarkSusy_analysis2::~DarkSusy_analysis2()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
DarkSusy_analysis2::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    edm::Handle<std::vector<pat::Muon> > patmuon;
    edm::Handle<std::vector<pat::Jet> > patjet;
    edm::Handle<edm::TriggerResults> triggerResults;
    iEvent.getByToken(patmuonToken, patmuon);
    iEvent.getByToken(patjetToken, patjet);
    iEvent.getByToken(triggerToken, triggerResults);
    std::string m_trigger="muon";

    //---------TRIGGER------------  
    if (triggerResults.isValid()) {
    	const edm::TriggerNames &triggerNames = iEvent.triggerNames(*triggerResults);
    	const std::vector<std::string> &triggerNames_ = triggerNames.triggerNames();
    	for (unsigned int iHLT = 0; iHLT < triggerResults->size(); iHLT++) {
          int hlt = triggerResults->accept(iHLT);
          //std::cout << "Trigger " << triggerNames_[iHLT]<< (triggerResults->accept(iHLT)? "PASS": "fail(not to run)") << std::endl;  
          if (trig_counts.find(triggerNames_[iHLT]) == trig_counts.end()) {trig_counts[triggerNames_[iHLT]]=0;}
	  if (hlt >0){ //if the trigger pass
	    if (triggerNames_[iHLT].find(m_trigger) !=std::string::npos){
	      trig_counts[triggerNames_[iHLT]]++;
	      //std::cout<< "Trigger "<< triggerNames_[iHLT]  <<std::endl;
	    }
	  }
	  /*for(auto elem : trig_counts)
	   {
             std::cout << elem.first << " " << elem.second  << "\n";
	     }*/
	  //std::cout << "Trigger " << triggerNames_[iHLT]<<"----"  << std::endl;
        }
  for(auto elem : trig_counts)
    { if (elem.second!=0)
	{ std::cout << elem.first << " " << elem.second  << "\n";}
	     }

    }
    //---------MUONS------------
    //sort muons by decreasing pt and fill the histogram just with the first four 
    std::vector<float> m_pt;
    std::vector<float> m_eta;
    std::vector<float> m_phi;
    int m_number = 0;
    for (std::vector<pat::Muon>::const_iterator itGen=patmuon->begin(); itGen!=patmuon->end(); ++itGen) {
      float pt = itGen->pt();
      printf("Pt: %6.4f, eta: %6.4f, phi: %6.4f \n", pt, itGen->eta(), itGen->phi());
      m_pt.push_back(pt);
      m_phi.push_back(itGen->phi());
      m_eta.push_back(itGen->eta());
      m_number=m_number+1;
    }
    std::cout << "The number of reconstructed muons is " << m_number << std::endl;
    h_muonsnumber->Fill(m_number);
    std::sort(m_pt.begin(), m_pt.end()); 
    std::reverse(m_pt.begin(), m_pt.end());

    for (unsigned int i=0; i<m_pt.size() && i<4; i++){
	      h_genpt->Fill(m_pt[i]);
    }
    if (m_pt.size() >3) {h_m1pt->Fill(m_pt[0]); h_m2pt->Fill(m_pt[1]); h_m3pt->Fill(m_pt[2]); h_m4pt->Fill(m_pt[3]);}
    else if (m_pt.size()==3) {h_m1pt->Fill(m_pt[0]); h_m2pt->Fill(m_pt[1]); h_m3pt->Fill(m_pt[2]); h_m4pt->Fill(0);}
    else if (m_pt.size()==2) {h_m1pt->Fill(m_pt[0]); h_m2pt->Fill(m_pt[1]); h_m3pt->Fill(0); h_m4pt->Fill(0);}
    else if (m_pt.size()==1) {h_m1pt->Fill(m_pt[0]); h_m2pt->Fill(0); h_m3pt->Fill(0); h_m4pt->Fill(0);}

    //---------JETS------------  
   //count the jets
    int jets = 0;  // this counts the number of jets
    int max_jets = 0; //this is the jet with maximum pt 
    int m_matched = 0;
    int m_matchedlj = 0;
    int m_notmatched = 0;
    std::vector<float> j_pt; 
    float jpt =0;  // this is the maximum pt of the jet 

    for (std::vector<pat::Jet>::const_iterator itJets=patjet->begin(); itJets!=patjet->end(); ++itJets) {
       int nmuons =0; //to count the number of muons in the jets
       jets=jets+1; //counters for jets
       float jetspt = itJets->pt();
       h_jetspt->Fill(jetspt);
       j_pt.push_back(jetspt);
       if (jetspt > jpt) { //to order jets with respect of pt 
         jpt=jetspt;
         max_jets=jets; //I use max_jets to select the leading jet if (jets==maxjets)
       }
       //else {continue;}
       
       //loop on components   
       std::vector daus(itJets->daughterPtrVector());
       std::sort(daus.begin(), daus.end(), [](const reco::CandidatePtr &p1, const reco::CandidatePtr &p2) { return p1->pt() > p2->pt(); }); 
       for (unsigned int k =0; k < daus.size(); k++){
           const pat::PackedCandidate &cand = dynamic_cast<const pat::PackedCandidate &>(*daus[k]);
           //printf("   Jets %3d: constituent %3d: pt %6.4f, pdgId %+3d, eta %6.4f, phi %6.4f \n", jets,k,cand.pt(),cand.pdgId(), cand.eta(), cand.phi());
           if (fabs(cand.pdgId())==13) {
               nmuons=nmuons+1;
               //std::cout << "The muon has pt " << cand.pt() <<std::endl;
               //if (jets==max_jets) { //in the leading jet I control if there is one of the first muon 
                      for (int i=0; i<4; i++){
                           if (fabs(cand.pt()-m_pt[i])<0.01){
                              std::cout << "-----------------Matched muon!!------------------" <<std::endl;
                              //printf("Our muon has  pt %6.4f, eta %6.4f, phi %6.4f \n", m_pt[i], m_eta[i], m_phi[i]);
                              //printf("Jet muon has  pt %6.4f, eta %6.4f, phi %6.4f \n", cand.pt(), cand.eta(), cand.phi());
                              m_matched = m_matched +1;
       			      if (jets==max_jets){m_matchedlj=m_matchedlj+1;}
                           }
    			   else {
			      //std::cout<< "-------no matched muons-----" <<std::endl;
 			      m_notmatched=m_notmatched+1;
			   }
                      }
               //}
           } 
       }
       //std::cout << "The number of muons in the " << jets << " jet is " << nmuons <<std::endl;
       h_nmuons-> Fill(nmuons);
       if (nmuons>=2){
          //std::cout <<"There are at least two muons in this jets" <<std::endl;
       }
 
         /*for (unsigned int i2 = 0, n = daus.size(); i2 < n && i2 <= 3; ++i2) {
                const pat::PackedCandidate &cand = dynamic_cast<const pat::PackedCandidate &>(*daus[i2]);
                printf("   Jets %3d: constituent %3d: pt %6.4f, pdgId %+3d, eta %6.4f, phi %6.4f \n", jets,i2,cand.pt(),cand.pdgId(), cand.eta(), cand.phi());
            }*/
       
    }
    std::cout<<"The number of jets is " <<jets <<std::endl;
    h_njets->Fill(jets); //histogram with the number of jets
    std::cout << "The maximum pt is " <<jpt << " for jet " << max_jets << std::endl;
    //for (unsigned int k =0; k<j_pt.size(); k++)
	//{std::cout << "Non ordinati" << j_pt[k] << std::endl;}
    std::sort(j_pt.begin(), j_pt.end()); 
    std::reverse(j_pt.begin(), j_pt.end());
    for (unsigned int i=0; i<j_pt.size() && i<1; i++){
      h_leadingjetspt->Fill(j_pt[i]);
    }
    //for (unsigned int k =0; k<j_pt.size(); k++)
	//{std::cout << j_pt[k] << std::endl;}
   std::cout<<"The number of muons in jets is " <<m_matched+m_notmatched << "; " << m_matched << " are muons matched while " <<m_matchedlj << " are muons matched in leading jet" <<std::endl;
   h_matchedmuons->Fill(m_matched);
   h_matchedmuonslj->Fill(m_matchedlj);
}

/*#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
}*/


// ------------ method called once each job just before starting event loop  ------------
void
DarkSusy_analysis2::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
DarkSusy_analysis2::endJob()
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
DarkSusy_analysis2::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);

  //Specify that only 'tracks' is allowed
  //To use, remove the default given above and uncomment below
  //ParameterSetDescription desc;
  //desc.addUntracked<edm::InputTag>("tracks","ctfWithMaterialTracks");
  //descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DarkSusy_analysis2);
