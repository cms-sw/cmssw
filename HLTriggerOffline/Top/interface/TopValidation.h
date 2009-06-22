
#ifndef HLTriggerOffline_TopValidation_H
#define HLTriggerOffline_TopValidation_H 


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
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h" 
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h" 

//#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include <DataFormats/HepMCCandidate/interface/GenParticle.h>
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"



#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexSorter.h"
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
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"


//Include DQM core
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"




#include "TH1F.h"
#include "TVector3.h"
#include "TLorentzVector.h"
#include <string.h>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <math.h> 
#include <TMath.h>
#include "TFile.h"



class TH1F;

//
// class decleration
//

class TopValidation : public edm::EDAnalyzer {
public:
  explicit TopValidation(const edm::ParameterSet&);
  ~TopValidation();
  
  
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
 
 // edm::Service<TFileService> fs;
  
  
  // ----------member data ---------------------------
  
  string fOutputFileName ;
  
std::string outFile_, outputFileName;

bool outputMEsInRootFile;



 edm::ParameterSet parameters;

 DQMStore* dbe;
 
 ///// eta and pt histos ////
 
 //  semimu events
 
  MonitorElement*     eta_off_mu;
  MonitorElement*     pt_off_mu;
 

  //  dimu events
 
  MonitorElement*     eta_off_dimu1;
  MonitorElement*     pt_off_dimu1;
  MonitorElement*     eta_off_dimu2;
  MonitorElement*     pt_off_dimu2;

//  semiel events 
  
  MonitorElement*     eta_off_el;
  MonitorElement*     pt_off_el;
 

  //diel events 
 
  MonitorElement*     eta_off_diel1;
  MonitorElement*     pt_off_diel1;
  MonitorElement*     eta_off_diel2;
  MonitorElement*     pt_off_diel2;
 
  //  emu events
 
  MonitorElement*     eta_off_emu_muon;
  MonitorElement*     pt_off_emu_muon;
  MonitorElement*     eta_off_emu_electron;
  MonitorElement*     pt_off_emu_electron;
  
  
//////////////////  
  
  MonitorElement*  h_ptmu1_trig[100];
  MonitorElement*  h_etamu1_trig[100];
  MonitorElement*  h_ptel1_trig[100];
  MonitorElement*  h_etael1_trig[100];
  
  MonitorElement*  h_ptmu1_trig_dimu[100];
  MonitorElement*  h_etamu1_trig_dimu[100];
  MonitorElement*  h_ptel1_trig_diel[100];
  MonitorElement*  h_etael1_trig_diel[100];
  
  MonitorElement*  h_ptmu1_trig_em[100];
  MonitorElement*  h_etamu1_trig_em[100];
  MonitorElement*  h_ptel1_trig_em[100];
  MonitorElement*  h_etael1_trig_em[100];
  
 
 
 
  edm::InputTag inputTag_;
  edm::TriggerNames triggerNames_;
  std::vector<string> hlt_bitnames;
  std::vector<string> hlt_bitnamesMu;
  std::vector<string> hlt_bitnamesEg;
  
  
  //Just a tag for better file organization
  std::string triggerTag_;

                  
 
  
};


#endif
