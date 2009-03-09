#ifndef ElectronTPValidator_H
#define ElectronTPValidator_H
//
// this code was written by Claire Timlin for CMSSW_1_6_7
// and it is imported to CMSSW_2_0_0 by Nikolaos Rompotis 23Apr2008
//           imported to CMSSW_2_0_7 by NR for the CSA08 exercise 29May2008
// 
// 15 July, 2008: Kalanand Mishra, Fermilab imported the file into generic 
// TagAndProbe package for the purpose of comparing the new tag-and-probe 
// method with the earlier method for electron.
//

// system include files
#include <memory>
#include <iostream>


//root include files
#include "TFile.h"
#include "TBranch.h"
#include "TTree.h"
#include "TLorentzVector.h"
#include "TVector.h"
#include "TString.h"
#include "TH2.h"
#include "TH1.h"

//CMSSW include files
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
//
#include "AnalysisDataFormats/Egamma/interface/ElectronIDAssociation.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "AnalysisDataFormats/Egamma/interface/ElectronID.h"

//
// class decleration
//

class ElectronTPValidator : public edm::EDAnalyzer {
   public:
      explicit ElectronTPValidator(const edm::ParameterSet&);
      ~ElectronTPValidator();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      //
      // extra function for the Delta R calculation .......
      double    CalcDR_Val(double, double, double, double);
      // ----------member data ---------------------------
      std::string outputFile_;
      bool useTriggerInfo;
      bool useTagIsolation;
      edm::InputTag MCCollection_;
      edm::InputTag SCCollectionHybrid_;
      edm::InputTag SCCollectionIsland_;
      edm::InputTag ElectronCollection_;
      std::string ElectronLabel_;
      edm::InputTag CtfTrackCollection_;
      edm::InputTag HLTCollection_;
      std::string HLTFilterType_;
      edm::InputTag electronIDAssocProducerRobust_;
      edm::InputTag electronIDAssocProducerLoose_;
      edm::InputTag electronIDAssocProducerTight_;
      edm::InputTag electronIDAssocProducerTightRobust_;
      double TagTIPCut_;
      double LIPCut_;
      //tag and probe  variables:     
      int probe_index_for_tree[4];
     

      //probe SC variables
      double probe_sc_eta_for_tree[4];
      double probe_sc_phi_for_tree[4];
      double probe_sc_et_for_tree[4];
      int probe_sc_pass_fiducial_cut[4]; 
      int probe_sc_pass_et_cut[4]; 

      //probe electron variables
      double probe_ele_eta_for_tree[4];
      double probe_ele_phi_for_tree[4];
      double probe_ele_et_for_tree[4];
      double probe_ele_Xvertex_for_tree[4];
      double probe_ele_Yvertex_for_tree[4];
      double probe_ele_Zvertex_for_tree[4];
      int probe_charge_for_tree[4];

      //tag electron variables
      double tag_ele_eta_for_tree[4];
      double tag_ele_phi_for_tree[4];
      double tag_ele_et_for_tree[4];
      double tag_ele_Xvertex_for_tree[4];
      double tag_ele_Yvertex_for_tree[4];
      double tag_ele_Zvertex_for_tree[4];
      double tag_isolation_value[4]; 
      int tag_charge_for_tree[4];
      int tag_classification_index_for_tree[4]; 
    
      //efficiency cuts
      int probe_ele_pass_fiducial_cut[4]; 
      int probe_ele_pass_et_cut[4]; 
      int probe_pass_recoEle_cut[4]; 
      int probe_pass_iso_cut[4]; 
      double probe_isolation_value[4]; 
      int probe_classification_index_for_tree[4]; 
      int probe_pass_tip_cut[4];
      int probe_pass_id_cut_robust[4];
      int probe_pass_id_cut_loose[4];
      int probe_pass_id_cut_tight[4];
      int probe_pass_id_cut_tight_robust[4];
      int probe_pass_trigger_cut[4];   


      double tag_probe_invariant_mass_for_tree[100];
      double tag_probe_invariant_mass_pass_for_tree[100];
      double sc_eta_for_tree[100];
      double sc_et_for_tree[100];

      
      //event variables
      int numberOfHLTFilterObjects;
      int no_probe_pass_recoEle_cut; 
      int no_probe_sc_pass_fiducial_cut; 
      int no_probe_sc_pass_et_cut; 
      int no_probe_ele_pass_fiducial_cut; 
      int no_probe_ele_pass_et_cut; 
      int no_probe_pass_iso_cut;
      int no_probe_golden; 
      int no_probe_pass_id_cut_robust;
      int no_probe_pass_id_cut_loose;
      int no_probe_pass_id_cut_tight;
      int no_probe_pass_trigger_cut;
      TTree * probe_tree;
      TFile * histofile;

      double BarrelMaxEta;
      double EndcapMinEta;
      double EndcapMaxEta;
      double IsoConeMinDR;
      double IsoConeMaxDR;
      double IsoMaxSumPt;
      double TagElectronMinEt;
      double TagProbeMassMin;
      double TagProbeMassMax;
      double ProbeSCMinEt;
      double ProbeRecoEleSCMaxDE; 
      int  GoldenBarrel;
      int  GoldenEndcap;
      int  BigBremBarrel;
      int  BigBremEndcap;
      int  NarrowBarrel;
      int  NarrowEndcap;
      int  CrackBarrel;
      int  CrackEndcap;
      int  ShowerBarrelMin;
      int  ShowerBarrelMax;
      int  ShowerEndcapMin;
      int  ShowerEndcapMax;
      double ProbeHLTObjMaxDR;
      double TrackInIsoConeMinPt;
      double RecoEleSeedBCMaxDE;
      double GsfTrackMinInnerPt;
        // debugging info: number of tags per event
      int  tag_number_in_event;
      int elec_number_in_event;      
      int elec_1_duplicate_removal;
      int elec_2_isolated;
      int elec_3_HLT_accepted;
      int elec_0_after_cuts_before_iso;
      int elec_0_tip_cut;
      int elec_0_et_cut;
      int elec_0_geom_cut;
      bool UseTransverseVertex_;
      TH1F *h_eta, *h_tip;
      TH1F *h_track_LIP, *h_track_pt;
      std::string      ProbeSC2RecoElecMatchingMethod_;
};

#endif


