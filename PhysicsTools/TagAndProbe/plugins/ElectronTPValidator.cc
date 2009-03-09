// -*- C++ -*-
//
// Package:    ElectronTPValidator
// Class:      ElectronTPValidator
// 
/**\class ElectronTPValidator ElectronTPValidator.cc 
   EgammaAnalysis/EgammaEfficiencyAlgos/src/ElectronTPValidator.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
################################################################################
contacts for this package: #####################################################
#####        Claire.Timlin@cern.ch   and     Nikolaos.Rompotis@cern.ch  ########
################################################################################
Originally written for CMSSW_1_6_7 by Claire Timlin
Imported to CMSSW_2_0_0 by Nikolaos Rompotis 23 April 2008
16.05.08 some comments and final tests (NR)
26.05.08 duplicate removal changes (new code by D. Wardrope) (NR)
28.05.08 change the matching of probe SC to reco Electrons (with detid) (NR)
02.06.08 matching with superCluster() instead of DetId (bug in hybrid)  (NR)
********************************************************************************
<<<<<<<<<<<<<<<<< SUMMARY of Changes since the last version >>>>>>>>>>>>>>>>>>>>
--> Deprecated cfg file  variables related to the old electron selection (golden
          etc) have been removed................................................
--> CalcDR_Val: this function used to be  included in a separate .C  file in the
          interface directory. Now it has been promoted to a class function.....
--> TagTIPCut: A configurable cut at the Tag  transverse  impact  parameter(TIP)
          has been added. You can use it in cfg e.g.:    double  TagTIPCut =0.01
	  If you have not specified this parameter in cgf default value is given
	  999., effectively disabling it........................................
--> LIPCut: There is a LIP cut for the tracks used in tag isolation.  Previously
          it was hardwired to 0.2. Now, this is default value if nothing else is
	  specified in the cfg file, e.g. double LIPCut = 0.2...................
--> Vertex of the Zee pair:  The vertex of the interaction that has produced the 
          ee pair used to be (0,0,vz). In this version this has been  corrected.
	  You can go back to the old way with  bool  UseTransverseVertex = false
	  in your cfg file......................................................
--> HLT matching: reco::HLTFilterObjectWithRefs has been removed from 20X series
          here I access directly TriggerEvent information.   See comments in the
	  code for more information. Configuration file additions are:..........
	  1. InputTag HLTCollection = hltTriggerSummaryAOD     (check your event 
	  content for the exact name of the instance)...........................
	  2. string   HLTFilterType = 
	  "hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter"
	  this instance indicative.   I recommend that you run the commented out
	  piece of code after the check whether trigger info exists in the event
	  for 1 event to find out the exact available names.....................
--> Duplicate Removal change: new code by D. Wardrope is implemented.This allows
          for more than one tracks that lead to the same supercluster.   The new
	  algo keeps the best track for each electron.      The old code is just
	  commented out for convenience.........................................
--> 4 probes per event: the code alows for 4 probes per tag.  A check is  set so
          that the event is skipped when more than 4 probes per event occur  and
	  a warining is print...................................................
--> Way to match a probe SC to a reco Electron: it used to be by matching the SC
          energy to the elec->superCluster()->energy(),but this method was found
	  to fail for 207 samples. detid matching  of the seed  SC has been used
	  here. The old method can be still used by using in the cfg file:......
	  string  ProbeSC2RecoElecMatchingMethod = "scEnergy" (def:"seedDetId" )
...2.06.08: Bug in hybrid algorithm was found that affects detid.Tests show that
          it is safer to use the superCluster reference when comparing.  This is
	  temporary. Tests  have  shown  that  DR  mathcing and superCluster ref 
	  matching agree (Duplicate removal level).In the level sc2reco matching
	  there are differences in DR matching and probe_sc_et(1 per 1000 level)
	  where sometimes probes matched to sc, but the energy  difference probe
	  ->energy - elec->supercluster->energy is very big ~50GeV.All the tests 
	  refer to Zee CSA08 samples............................................

... 15 July, 2008: Kalanand Mishra, Fermilab imported the file into generic TagAndProbe 
    package for the purpose of comparing the new tag-and-probe method with the 
    earlier method for electron.

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<end of summary>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
*/
//
// Original Author:  Claire Timlin
//         Created:  Fri Oct 19 14:10:48 CEST 2007
//
//


#include "PhysicsTools/TagAndProbe/interface/ElectronTPValidator.h" 
// this is included for the CalcDR_Val function
#include <cmath>
#include "HLTrigger/HLTcore/interface/TriggerSummaryAnalyzerAOD.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

ElectronTPValidator::ElectronTPValidator(const edm::ParameterSet& ps)

{
  outputFile_ = ps.getUntrackedParameter<std::string>("outputfile");
  useTriggerInfo = ps.getUntrackedParameter<bool>("useTriggerInfo");
  useTagIsolation = ps.getUntrackedParameter<bool>("useTagIsolation");
  MCCollection_ = ps.getParameter<edm::InputTag>("MCCollection");
  SCCollectionHybrid_ = ps.getParameter<edm::InputTag>("SCCollectionHybrid");
  SCCollectionIsland_ = ps.getParameter<edm::InputTag>("SCCollectionIsland");
  ElectronCollection_= ps.getParameter<edm::InputTag>("ElectronCollection");
  ElectronLabel_= ps.getParameter<std::string>("ElectronLabel");
  CtfTrackCollection_= ps.getParameter<edm::InputTag>("CtfTrackCollection");
  HLTCollection_= ps.getParameter<edm::InputTag>("HLTCollection");
  HLTFilterType_ =ps.getParameter<std::string>("HLTFilterType");
  // electron ID collections
  electronIDAssocProducerRobust_= 
    ps.getParameter<edm::InputTag>("electronIDAssocProducerRobust");
  electronIDAssocProducerLoose_= 
    ps.getParameter<edm::InputTag>("electronIDAssocProducerLoose");
  electronIDAssocProducerTight_= 
    ps.getParameter<edm::InputTag>("electronIDAssocProducerTight");
  electronIDAssocProducerTightRobust_= 
    ps.getParameter<edm::InputTag>("electronIDAssocProducerTightRobust");


  BarrelMaxEta = ps.getUntrackedParameter<double>("BarrelMaxEta");
  EndcapMinEta = ps.getUntrackedParameter<double>("EndcapMinEta");
  EndcapMaxEta = ps.getUntrackedParameter<double>("EndcapMaxEta");
  IsoConeMinDR = ps.getUntrackedParameter<double>("IsoConeMinDR");
  IsoConeMaxDR = ps.getUntrackedParameter<double>("IsoConeMaxDR");
  IsoMaxSumPt = ps.getUntrackedParameter<double>("IsoMaxSumPt");
  TagElectronMinEt = ps.getUntrackedParameter<double>("TagElectronMinEt");
  TagProbeMassMin = ps.getUntrackedParameter<double>("TagProbeMassMin");
  TagProbeMassMax = ps.getUntrackedParameter<double>("TagProbeMassMax");
  ProbeSCMinEt= ps.getUntrackedParameter<double>("ProbeSCMinEt");
  ProbeRecoEleSCMaxDE= ps.getUntrackedParameter<double>("ProbeRecoEleSCMaxDE");
  ProbeHLTObjMaxDR= ps.getUntrackedParameter<double>("ProbeHLTObjMaxDR");
  TrackInIsoConeMinPt = ps.getUntrackedParameter<double>("TrackInIsoConeMinPt");
  RecoEleSeedBCMaxDE = ps.getUntrackedParameter<double>("RecoEleSeedBCMaxDE");


  // Configure TIP cut for the tag
  if (ps.exists("TagTIPCut")) {
    TagTIPCut_ = ps.getParameter<double>("TagTIPCut");}
  else {TagTIPCut_ = 999.;}
  if (ps.exists("LIPCut")) {
    LIPCut_ = ps.getParameter<double>("LIPCut");}
  else {LIPCut_ = 0.2;}

  // Option to use the correct vertex in order to calculate the tag-probe invariant mass
  // The vertex of the interaction that has produced the ee (tag-probe) pair is considered
  //     to be the vertex of the tag (probe is a SC). In the previous version of this code
  //     the vertex was considered to be at (0,0,vz), i.e. ignoring transverse  components
  //     This has been included in this version. For debugging you can specify in your cfg
  //     UseTransverseVertex = false and get the old version results .....................
  UseTransverseVertex_ = true;
  if (ps.exists("UseTransverseVertex")) {
    UseTransverseVertex_ = ps.getParameter<bool>("UseTransverseVertex");
  }
  if (UseTransverseVertex_){
    edm::LogInfo("info") << "Transverse Vertex Correction is on" ; }
  else edm::LogInfo("info") << "Transverse Vertex Correction is OFF!!!!" ; 
  if (ps.exists("ProbeSC2RecoElecMatchingMethod")) {
    ProbeSC2RecoElecMatchingMethod_ =
      ps.getParameter<std::string>("ProbeSC2RecoElecMatchingMethod");
    // check whether the method in the input makes sence to avoid an empty run
    if (ProbeSC2RecoElecMatchingMethod_ != "seedDetId" && 
	ProbeSC2RecoElecMatchingMethod_ != "scEnergy"  &&
	ProbeSC2RecoElecMatchingMethod_ != "superCluster") {
      
      edm::LogInfo("info") << "Error! ProbeSC2RecoElecMatchingMethod in your"
	   << " cfg file is unknown => value changed to superCluster"
	   << " originally was: " << ProbeSC2RecoElecMatchingMethod_
	;
      ProbeSC2RecoElecMatchingMethod_ = "superCluster";
    }
  }
  else ProbeSC2RecoElecMatchingMethod_ = "superCluster";
  edm::LogInfo("info") << "Notice: ProbeSC2RecoElecMatchingMethod is " 
		       << ProbeSC2RecoElecMatchingMethod_;
  
}


ElectronTPValidator::~ElectronTPValidator()
{
   histofile->Close();
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void ElectronTPValidator::analyze(const edm::Event& evt, const edm::EventSetup& es)
{
  edm::LogInfo("") << "Analyse";
  //get MC collection
  edm::Handle<edm::HepMCProduct> pHepMC;
  try{
    evt.getByLabel(MCCollection_, pHepMC); 
  }
  catch (cms::Exception)
    {
      edm::LogError("")<< "Error! Can't get " << MCCollection_.label() << " by label. ";
    }
 //get hybrid sc collection
  edm::Handle<reco::SuperClusterCollection> pSCHybrid;
  try{
    evt.getByLabel(SCCollectionHybrid_, pSCHybrid);
  }
  catch (cms::Exception)
    {
      edm::LogError("")<< "Error! Can't get Hybrid SC Collection by label. ";
    }
  //get island sc collection
  edm::Handle<reco::SuperClusterCollection> pSCIsland;
  try{
    evt.getByLabel(SCCollectionIsland_, pSCIsland);
  }
  catch (cms::Exception)
    {
      edm::LogError("")<< "Error! Can't get Island SC Collection by label. ";
    }
 //get electron collection
  edm::Handle<reco::PixelMatchGsfElectronCollection> pElectrons;
  try{
    evt.getByLabel(ElectronCollection_, pElectrons);
  }
  catch (cms::Exception)
    {
      edm::LogError("")<< "Error! Can't get ElectronCollection by label. ";
    }
//get ctf track collection
  edm::Handle<reco::TrackCollection> pCtfTracks;
  try{
    evt.getByLabel(CtfTrackCollection_, pCtfTracks);
  }
  catch (cms::Exception)
    {
      edm::LogError("")<< "Error! Can't get " << CtfTrackCollection_.label() << " by label. ";
    }

  bool no_trigger_info = false;
//get ctf track collection
// class reco::HLTFilterObjectWithRefs does not exist in CMSSW_2_0_0
//  edm::Handle<reco::HLTFilterObjectWithRefs> pHLT;
//  edm::Handle<reco::ElectronCollection> pHLT;
// for 2_0_7 format and the CSAO8 exercize
  edm::Handle<trigger::TriggerEvent> pHLT;
  //
  // for some reason the method isValid() is a stabler method to catch an error
  // than the previously used try{}catck(){} structure
  //
  evt.getByLabel(HLTCollection_, pHLT);
  if (not pHLT.isValid()){
    no_trigger_info =true;
    //   edm::LogError("")<< "Error! Can't get HLTCollection by label. ";
    //  edm::LogError("")<< "********NO TRIGGER INFO*********** ";
   edm::LogInfo("info")<< "********NO TRIGGER INFO*********** ";
  }   /*    THIS IS FOR ACCESSING TRIGGER DETAILS
  else {
   // this is for debugging. It gives all the contents of the trigger
   edm::LogInfo("debug") << "Used Processname: " << pHLT->usedProcessName() << std::endl;
   const int nO(pHLT->sizeObjects());
   edm::LogInfo("debug") << "Number of TriggerObjects: " << nO << std::endl;
   edm::LogInfo("debug")<< "The TriggerObjects: #, id, pt, eta, phi, mass" << std::endl;
   const trigger::TriggerObjectCollection& TOC(pHLT->getObjects());
   for (int iO=0; iO!=nO; ++iO) {
     const trigger::TriggerObject& TO(TOC[iO]);
     edm::LogInfo("debug") << iO << " " << TO.id() << " " << TO.pt() << " " << TO.eta() << " " << TO.phi() 
     << " " << TO.mass() << std::endl;
   }
   const int nF(pHLT->sizeFilters());
   edm::LogInfo("debug") << "Number of TriggerFilters: " << nF << std::endl;
   edm::LogInfo("debug")  << "The Filters: #, label, #ids/#keys, the id/key pairs" << std::endl;
   for (int iF=0; iF!=nF; ++iF) {
     const trigger::Vids& VIDS (pHLT->filterIds(iF));
     const trigger::Keys& KEYS(pHLT->filterKeys(iF));
     const int nI(VIDS.size());
     const int nK(KEYS.size());
     edm::LogInfo("debug")    << iF << " " << pHLT->filterLabel(iF)
	  << " " << nI << "/" << nK
	  << " the pairs: ";
     //     const int n(max(nI,nK));
     const int n= (nI>nK)? nI:nK;
     for (int i=0; i!=n; ++i) {
      edm::LogInfo("debug")  << " " << VIDS[i] << "/" << KEYS[i];
     }
     edm::LogInfo("debug")   << std::endl;
     //assert (nI==nK);
   
   }
  }
      */
  //OK trigger info may be in the event, but the event may not pass the specific trigger
  // that we want
  const std::string FilterType = HLTFilterType_;
  if (not no_trigger_info) {
    const int nF(pHLT->sizeFilters());
    const int filterInd = pHLT->filterIndex(FilterType);
    if (nF == filterInd) {
      no_trigger_info = true;
      edm::LogInfo("info") << "Trigger Type " << FilterType << " absent in this event";
    }
  }
  
//get electron ID association map for robust ID
  edm::Handle<reco::ElectronIDAssociationCollection> pEleIDRobust;
  try{
    evt.getByLabel(electronIDAssocProducerRobust_, pEleIDRobust);
  }
  catch (cms::Exception)
    {
       edm::LogError("")<< "Error! Can't get Robust electronIDAssocProducer by label. ";
    }


//get electron ID association map for loos ID 
  edm::Handle<reco::ElectronIDAssociationCollection> pEleIDLoose;
  try{
    evt.getByLabel(electronIDAssocProducerLoose_, pEleIDLoose);
  }
  catch (cms::Exception)
    {
       edm::LogError("")<< "Error! Can't get electronIDAssocProducer Loose by label. ";
    }


//get electron ID association map for tight ID 
  edm::Handle<reco::ElectronIDAssociationCollection> pEleIDTight;
  try{
    evt.getByLabel(electronIDAssocProducerTight_, pEleIDTight);
  }
  catch (cms::Exception)
    {
       edm::LogError("")<< "Error! Can't get electronIDAssocProducer Tight by label. ";
    }


//get electron ID association map for a tight form of robust ID (using only 4 cuts) 
  edm::Handle<reco::ElectronIDAssociationCollection> pEleIDTightRobust;
  try{
    evt.getByLabel(electronIDAssocProducerTightRobust_, pEleIDTightRobust);
  }
  catch (cms::Exception)
    {
       edm::LogError("")<< "Error! Can't get electronIDAssocProducer Tight Robust by label. ";
    }


  //****************start analysis***************************************************************
  //... 100 maximum tag-sc 
   for(int i =0; i<100; i++) {
     tag_probe_invariant_mass_for_tree[i] = -99.0;
     tag_probe_invariant_mass_pass_for_tree[i] = -99.0;
     sc_eta_for_tree[i] = -99.0;
     sc_et_for_tree[i] = -99.0;
   }

   for(int i =0; i <4; i++){
      probe_index_for_tree[i] = -99;
      probe_ele_eta_for_tree[i] = -99.0;
      probe_ele_et_for_tree[i] = -99.0;
      probe_ele_phi_for_tree[i] = -99.0;
      probe_ele_Xvertex_for_tree[i] = -99.0;
      probe_ele_Yvertex_for_tree[i] = -99.0;
      probe_ele_Zvertex_for_tree[i] = -99.0;

      probe_sc_eta_for_tree[i] = -99.0;
      probe_sc_et_for_tree[i] = -99.0;
      probe_sc_phi_for_tree[i] = -99.0;

      probe_charge_for_tree[i] = -99;
      probe_sc_pass_fiducial_cut[i] = 0;
      probe_sc_pass_et_cut[i] =0; 
      probe_ele_pass_fiducial_cut[i] = 0;
      probe_ele_pass_et_cut[i] =0; 
      probe_pass_recoEle_cut[i] =0;
      probe_pass_iso_cut[i]=0; 
      probe_classification_index_for_tree[i]=-99; 
      probe_isolation_value[i] = -99.0;
      probe_pass_id_cut_robust[i] =0; 
      probe_pass_id_cut_loose[i] =0; 
      probe_pass_id_cut_tight[i] =0; 
      probe_pass_tip_cut[i] =0; 
      probe_pass_id_cut_tight_robust[i] =0;
      probe_pass_trigger_cut[i]=0;

      tag_ele_eta_for_tree[i] = -99.0;
      tag_ele_phi_for_tree[i] = -99.0;
      tag_ele_et_for_tree[i] = -99.0;
      tag_ele_Xvertex_for_tree[i] = -99.0;
      tag_ele_Yvertex_for_tree[i] = -99.0;
      tag_ele_Zvertex_for_tree[i] = -99.0;
      tag_charge_for_tree[i] = -99;
      tag_classification_index_for_tree[i]=-99; 
      tag_isolation_value[i] =  99.0;
     }
    
      numberOfHLTFilterObjects  =0;
   
      const reco::SuperClusterCollection * SCHybrid = pSCHybrid.product();
      const reco::SuperClusterCollection * SCIsland = pSCIsland.product();
      const reco::PixelMatchGsfElectronCollection * electrons = pElectrons.product();
      const reco::TrackCollection * ctfTracks = pCtfTracks.product();
      // this class has been removed from CMSSW_2_0_0
      //      const reco::HLTFilterObjectWithRefs * HLTObj;
      // replacement 
      //      const reco::ElectronCollection * HLTObj;
      

      // Find how many objects pass the HLTFilterType_ specified in the input
      if(no_trigger_info ==false) {
	//      	HLTObj = pHLT.product();
	//      	numberOfHLTFilterObjects = HLTObj->size();
	//	const int nTrigger(pHLT->sizeObjects());
	// this is the number of HLT pass objects, not electrons necessarily
      	//numberOfHLTFilterObjects = int (nTrigger);
	// the test has been done, so this already  exists
	const int iF = pHLT->filterIndex(HLTFilterType_);
	const trigger::Vids& VIDS (pHLT->filterIds(iF));
	const trigger::Keys& KEYS(pHLT->filterKeys(iF));
	const int nI(VIDS.size());
	const int nK(KEYS.size());
	numberOfHLTFilterObjects = (nI>nK)? nI:nK;
      }
      else numberOfHLTFilterObjects =0;
      // all electrons in this event: usefull for debuging     
      elec_number_in_event = electrons->size();

      //   edm::LogError("") << "ELECTRON COLLECTION SIZE: "  << electrons->size();
      //      edm::LogInfo("") << "Electrons.size() = " << electrons->size();
      std::vector<reco::PixelMatchGsfElectronRef> UniqueElectrons;
      // edm::LogInfo("") << "Starting loop over electrons.";
      int index =0;
      // Loop to check whether there are tracks that share the same SC
      //  LOOP: COMMON SC ********************************************
      // this is the default old method ........................................
      //for(reco::PixelMatchGsfElectronCollection::const_iterator 
      //    elec1 = electrons->begin(); elec1 != electrons->end();++elec1)
      //{
      //  const reco::PixelMatchGsfElectronRef electronRef(pElectrons, index);
      //  
      //  bool duplicate = false;
      //  
      //  for(reco::PixelMatchGsfElectronCollection::const_iterator 
      //	elec2 = electrons->begin(); elec2 != electrons->end();++elec2)
      //    {
      //      if(elec1 != elec2)
      //	{
      //	  DetId id1 =elec1->superCluster()->seed()->getHitsByDetId()[0];
      //	  DetId id2 =elec2->superCluster()->seed()->getHitsByDetId()[0];
      //	  if( id1 == id2)
      //	    {
      //	      duplicate = true;
      //	      if(fabs(elec1->eSuperClusterOverP()-1.) 
      //		 <= fabs(elec2->eSuperClusterOverP()-1.))
      //		{
      //	  UniqueElectrons.push_back(electronRef);
      //		  //edm::LogInfo("")<<"Pushed back electron 1.";
      //		}
      //	    }
      //	}
      //    }
      //  if(duplicate == false) UniqueElectrons.push_back(electronRef);
      //  ++index;
      //  // edm::LogInfo("")<<"end of duplicate loop";
      //}
      //************************************************************************
      // NEW METHOD by D WARDROPE implemented 26.05.08 *************************
      //************* DUPLICATE ******  REMOVAL ********************************
      // 02.06.08: due to a bug in the hybrid algorithm that affects detid *****
      //           we change detid matching to superCluster ref matching *******
      for(reco::GsfElectronCollection::const_iterator 
	    elec = electrons->begin(); elec != electrons->end();++elec) {
	const reco::PixelMatchGsfElectronRef  electronRef(pElectrons, index);
	//Remove duplicate electrons which share a supercluster
	bool duplicate = false;
	reco::GsfElectronCollection::const_iterator BestDuplicate = elec;
	int index2 = 0;
	for(reco::GsfElectronCollection::const_iterator
	      elec2 = electrons->begin();
	    elec2 != electrons->end(); ++elec2)
	  {
	    if(elec != elec2)
	      {
		DetId id1
		  = elec->superCluster()->seed()->hitsAndFractions()[0].first;
		DetId id2
		  = elec2->superCluster()->seed()->hitsAndFractions()[0].first;
		if( elec->superCluster() == elec2->superCluster())
		  {
		    duplicate = true;
		    if(fabs(BestDuplicate->eSuperClusterOverP()-1.)
		       >= fabs(elec2->eSuperClusterOverP()-1.))
		      {
			BestDuplicate = elec2;
		      }
		  }

		//if( id1 == id2)
		//  {
		//    duplicate = true;
		//    if(fabs(BestDuplicate->eSuperClusterOverP()-1.)
		//       >= fabs(elec2->eSuperClusterOverP()-1.))
		//      {
		//	BestDuplicate = elec2;
		//      }
		//  }
	      }
	    ++index2;
	  }
	if(BestDuplicate == elec) UniqueElectrons.push_back(electronRef);
	++index;
      }
      //
      // debugging: store electrons after duplicate removal
      elec_1_duplicate_removal = UniqueElectrons.size();
      //
      // LOOP END : COMMON SC **************************************************
      // *-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-**-*-*-**-*-***-*-**-
      // TAG finder ************************************************************
      //
      // this loops over the     vector<reco::PixelMatchGsfElectronRef> 
      // that was defined / selected in the previous loop and finds the
      // tag through the following procedure:..........................
      // 1. find SC eta + transverse impact parameter (TIP) (this is d0)
      // 2. condition in track Et + SC eta in ECAL + TIP................
      // 3. it calculates the isolation variable and checks if passes...
      // 4. check single electron trigger 
      // 5. check electron id criteria are satisfied
      // So finally we have the tags in  vector<reco::PixelMatchGsfElectronRef> TagElectrons;
      //
      std::vector<reco::PixelMatchGsfElectronRef> TagElectrons;
      //  edm::LogInfo("")<<"after define tag vector";
      int k =0;
      int IsolatedElectrons=0; int HLTElectrons=0, CutsBeforeIso=0;
      int EtCut =0, TIPCut =0, GeomCut =0;
      bool doTrackHistos = true;
      for(std::vector<reco::PixelMatchGsfElectronRef>::const_iterator 
	    Relec = UniqueElectrons.begin(); Relec != UniqueElectrons.end(); ++Relec)
	{
	  //  edm::LogInfo("")<<"loop over unique eles to find tag"; 
	  reco::PixelMatchGsfElectronRef elec;
	  elec = *Relec;
	  // edm::LogInfo("")<<"after accessing electron"; 
	  double tagSCEta = elec->superCluster().get()->eta();
	  double TagTIP = elec->gsfTrack()->d0();
	  // double checkTIP = sqrt(elec->gsfTrack()->dx()
	  //*elec->gsfTrack()->dx() + elec->gsfTrack()->dy()*elec->gsfTrack()->dy());
	  // debugging: Et cut, TIP cut & geometry cut
	  if (elec->et()>TagElectronMinEt) ++EtCut;
	  if (fabs(TagTIP) < TagTIPCut_) {++TIPCut;} 
	  h_tip->Fill(TagTIP);
	  if (fabs(tagSCEta) < BarrelMaxEta ||  
		( fabs(tagSCEta) > EndcapMinEta && fabs(tagSCEta) < EndcapMaxEta)) 
	    {++GeomCut; h_eta->Fill(tagSCEta);}

	  if(elec->et()>TagElectronMinEt && 
	     (fabs(tagSCEta) < BarrelMaxEta ||  
	      ( fabs(tagSCEta) > EndcapMinEta && fabs(tagSCEta) < EndcapMaxEta) )
	     && fabs(TagTIP) < TagTIPCut_ ){
	    ++CutsBeforeIso;
	    //no lets see if this electron is isolated:
	    int iso_int_ele = 0;
	    double sum_of_pt_ele =0.0;
	    double isolation_value_ele = 99.0;
	    if(useTagIsolation == true){
	    for(reco::TrackCollection::const_iterator tr = ctfTracks->begin(); 
		tr != ctfTracks->end();++tr)
	      {
		double LIP = elec->gsfTrack()->dz() - tr->dz();
		//	edm::LogInfo("")<< "LIP: " << LIP ;
		//	edm::LogInfo("")<< "tesy: " << elec->gsfTrack()->vz() - tr->vz() ;
		if (doTrackHistos){
		  h_track_LIP->Fill(LIP); h_track_pt->Fill(tr->pt());
		  doTrackHistos = false;
		}
		// if(tr->pt() >TrackInIsoConeMinPt && fabs(LIP) < 0.2){
		if(tr->pt() >TrackInIsoConeMinPt && fabs(LIP) < LIPCut_){
		  double dr_ctf_ele = 
		    CalcDR_Val(tr->eta()
			       , elec->trackMomentumAtVtx().eta()
			       , tr->phi(), elec->trackMomentumAtVtx().phi());
		  if((dr_ctf_ele>IsoConeMinDR) && (dr_ctf_ele<IsoConeMaxDR)){
		    double cft_pt_2 = (tr->pt())*(tr->pt());
		    sum_of_pt_ele += cft_pt_2;
		  }//end of if in ctf tracks in isolation cone around tag electron candidate sum squared pt values
		}//end of only loop over tracks with Inner Pt >1.5GeV
	      }//end of cft loop to work out isolation value
	    isolation_value_ele = sum_of_pt_ele/
	      (elec->trackMomentumAtVtx().Rho()*elec->trackMomentumAtVtx().Rho());
	    if(isolation_value_ele<IsoMaxSumPt) 
	      {++iso_int_ele; ++IsolatedElectrons; }
	    }//if we want the tag to be isolated
	    else ++iso_int_ele;
	    if(iso_int_ele >0){
	      //edm::LogInfo("")<<"in tigger loop"; 
	      //now let's see if the tag passes the single electron trigger:
	      int trigger_int_ele = 0;
	      if(useTriggerInfo ==false) ++trigger_int_ele;
	      // edm::LogInfo("")<<"USE TRIGGERINFO BOOL"<<no_trigger_info << " , " <<  useTriggerInfo ; 
	      if(no_trigger_info == false && useTriggerInfo ==true ){
		/* The old fashioned way 206
		//	edm::LogInfo("")<<"HLTsize"<<HLTObj->size() ;
	      for(reco::ElectronCollection::const_iterator 
		    HLT1 = HLTObj->begin(); HLT1 != HLTObj->end();++HLT1){
		//edm::LogInfo("")<<"HLT eta"<<HLT1->eta() ;
		double dr_ele_HLT = CalcDR_Val(elec->eta(), HLT1->eta(), elec->phi(), HLT1->phi());
		//edm::LogInfo("")<<"dr iso: " << dr_ele_HLT;
		if(fabs(dr_ele_HLT) < ProbeHLTObjMaxDR) {++trigger_int_ele;
		++HLTElectrons;}
		
	      }//end of loop over HLT filter objects
		*/
		/*
		const int nTriggerObj(pHLT->sizeObjects());
		const trigger::TriggerObjectCollection& TOC(pHLT->getObjects());
		for (int iTrig =0;iTrig != nTriggerObj; ++iTrig) {
		  const trigger::TriggerObject& TO(TOC[iTrig]);
		  if (abs(TO.id())==11 ) { // demand it to be an electron
		    double dr_ele_HLT = CalcDR_Val(elec->eta(),TO.eta(), elec->phi(), TO.phi());
		    if (fabs(dr_ele_HLT) < ProbeHLTObjMaxDR){++trigger_int_ele;
		    ++HLTElectrons;}
		  }
		}
		*/
		// find how many relevant
		const int iF = pHLT->filterIndex(HLTFilterType_);
		// find how many objects there are
		const trigger::Keys& KEYS(pHLT->filterKeys(iF));
		const int nK(KEYS.size());
		// loop over these objects to see whether they match
		const trigger::TriggerObjectCollection& TOC(pHLT->getObjects());
		for (int iTrig = 0;iTrig <nK; ++iTrig ) {
		  const trigger::TriggerObject& TO(TOC[KEYS[iTrig]]);
		  if (abs(TO.id())==11 ) { // demand it to be an electron
		    double dr_ele_HLT = CalcDR_Val(elec->eta(),TO.eta(), elec->phi(), TO.phi());
		    if (fabs(dr_ele_HLT) < ProbeHLTObjMaxDR){++trigger_int_ele;
		    ++HLTElectrons;}
		  }
		}

	      }//checking trigger info is there!
	      // edm::LogInfo("")<<"end of tigger loop"; 
	      if (trigger_int_ele>0){
		//	edm::LogInfo("")<<"in id loop"; 
		//let's aks that the tag passes the ID tool (but no particular class is specified)
		reco::ElectronIDAssociationCollection::const_iterator tagIDAssocItr;
		tagIDAssocItr = pEleIDLoose->find(elec);
		const reco::ElectronIDRef& id_tag = tagIDAssocItr->val;
		bool cutBasedID_tag = false;
		cutBasedID_tag = id_tag->cutBasedDecision();
		if(cutBasedID_tag == true){ 
		//WE HAVE A TAG!!
		TagElectrons.push_back(elec);
		tag_ele_eta_for_tree[k] = elec->eta();
		tag_ele_phi_for_tree[k] = elec->phi();
		tag_ele_et_for_tree[k] = elec->et();
		tag_ele_Xvertex_for_tree[k] = elec->vx();
		tag_ele_Yvertex_for_tree[k] = elec->vy();
		tag_ele_Zvertex_for_tree[k] = elec->vz();
		tag_charge_for_tree[k] = elec->charge();
		tag_classification_index_for_tree[k]=elec->classification(); 
		tag_isolation_value[k] = isolation_value_ele;
		}//end of if tag passes ID 
	      }//end of if tag ele passes the trigger 
	    }//end of is tag ele is isolated
	  }//end of if tag ele has et > min and its associated SC is in the ecal fiducial region
	  ++k;
	  //edm::LogInfo("")<<"exiting loop to find tags";
	}//end of loop over non duplicate electrons to find tags
      elec_2_isolated = IsolatedElectrons;
      elec_3_HLT_accepted = HLTElectrons;
      elec_0_after_cuts_before_iso = CutsBeforeIso;
      elec_0_et_cut = EtCut;
      elec_0_geom_cut = GeomCut;
      elec_0_tip_cut = TIPCut;
      //
      //... by this point we have the tag collection in the vector:.......
      //... vector<reco::PixelMatchGsfElectronRef> TagElectrons ..........
      //...
      //... Now: find  supercluster  probes in the event
      //... define the probe container: vector<reco::SuperClusterRef> ProbeSC
      //... LOOP over all tags............................................
      //...1. check for the available superclusters in the event in the...
      //...  barrel (hybrid) and the endcaps (island) and check the  inv m
      //...  if you find only one SC whose mass is in the limits keep it..
      //...2. run over the SC again only if the probes found were <= 1....
      //...   check whether  there is  corresponding PixelMatchGsfElectron
      //...   in the event and pass the. info in the tree.................
      //...  

      // debug: keep the number of Tags per event:
      tag_number_in_event = TagElectrons.size();
      int mass_iterator = 0;
      int mass_pass_iterator = 0;
      if(TagElectrons.size()>2) std::cout 
	<< "More than 2 tags in event!!!!!! - ignoring event? NO!" << std::endl;
 

      std::vector<reco::SuperClusterRef> ProbeSC;
      for(std::vector<reco::PixelMatchGsfElectronRef>::const_iterator 
	    Rtag = TagElectrons.begin(); Rtag != TagElectrons.end(); ++Rtag)
	{
	  // edm::LogInfo("")<<"entering loop over tags";
	  reco::PixelMatchGsfElectronRef tag;
	  tag = *Rtag;
	  reco::SuperClusterRef probeCandRef;
	  int indexSCH =0;
	  int indexSCI =0;
	  int noProbesPerTag = 0;
	  TVector3 scVect, ZeeVertexVect, probeCandVect;
	  TLorentzVector probeCand, tagLorentzVec;
	  if (not UseTransverseVertex_)
	    ZeeVertexVect.SetXYZ(0.0, 0.0, tag->vz());
	  else
	    ZeeVertexVect.SetXYZ(tag->vx(), tag->vy(), tag->vz());
	  tagLorentzVec.SetPtEtaPhiE(tag->pt(), tag->eta(), tag->phi(), tag->energy());
	  for(reco::SuperClusterCollection::const_iterator 
		scHybrid = SCHybrid->begin(); scHybrid != SCHybrid->end();++scHybrid)
	    {
	      const reco::SuperClusterRef sc(pSCHybrid, indexSCH);
	      double pt = sc->energy()/(cosh(sc->eta()));
	      scVect.SetPtEtaPhi(pt, sc->eta(), sc->phi());
	      probeCandVect = scVect - ZeeVertexVect;
	      double new_pt = sc->energy()/(cosh(probeCandVect.Eta()));
	      probeCand.SetPtEtaPhiE(new_pt, probeCandVect.Eta()
				     , probeCandVect.Phi(), sc->energy()); 
	      TLorentzVector  V = (tagLorentzVec + probeCand);
	      double Mass = V.M();
	      //   if(Mass_hybrid <0) cout << "WARNING!!: TAG - HYBRID PROBE INVARIANT MASS: " 
	      //    << Mass_hybrid << "Probe_Iterator: " << probe_iterator << endl;
	      if(Mass >TagProbeMassMin && Mass < TagProbeMassMax){
		probeCandRef = sc;
		++noProbesPerTag;
	      }
	      ++indexSCH;
	    }//end of loop over hybrid SC to find probe

	  for(reco::SuperClusterCollection::const_iterator 
		scIsland = SCIsland->begin(); scIsland != SCIsland->end();++scIsland)
	    {
	      const reco::SuperClusterRef sc(pSCIsland, indexSCI);
	      double pt = sc->energy()/(cosh(sc->eta()));
	      scVect.SetPtEtaPhi(pt, sc->eta(), sc->phi());
	      probeCandVect = scVect - ZeeVertexVect;
	      double new_pt = sc->energy()/(cosh(probeCandVect.Eta()));
	      probeCand.SetPtEtaPhiE(new_pt, probeCandVect.Eta()
				     , probeCandVect.Phi(), sc->energy()); 
	      TLorentzVector  V = (tagLorentzVec + probeCand);
	      double Mass = V.M();
	      //   if(Mass_hybrid <0) cout << "WARNING!!: TAG - HYBRID PROBE INVARIANT MASS: " 
	      //         << Mass_hybrid << "Probe_Iterator: " << probe_iterator << endl;
	      if(Mass>TagProbeMassMin && Mass< TagProbeMassMax){
		probeCandRef = sc;
		++noProbesPerTag;
	      }
	      ++indexSCI;
	    }//end of loop over hybrid SC to find probe
	  if(noProbesPerTag ==1)ProbeSC.push_back(probeCandRef);
	  //}//end of loop over tags to find probes

      int indexSCH_all =0;
      int indexSCI_all =0;
      if(noProbesPerTag <= 1) {
	for(reco::SuperClusterCollection::const_iterator 
	      scHybrid = SCHybrid->begin(); scHybrid != SCHybrid->end();++scHybrid){
	  const reco::SuperClusterRef sc(pSCHybrid, indexSCH_all); 
	  double pt = sc->energy()/(cosh(sc->eta()));
	  scVect.SetPtEtaPhi(pt, sc->eta(), sc->phi());
	  probeCandVect = scVect - ZeeVertexVect;
	  double new_pt = sc->energy()/(cosh(probeCandVect.Eta()));
	  probeCand.SetPtEtaPhiE(new_pt, probeCandVect.Eta()
				 , probeCandVect.Phi(), sc->energy()); 
	  TLorentzVector  V = (tagLorentzVec + probeCand);
	  double Mass = V.M();
	  // if((Mass_hybrid >0.0) && (Mass_hybrid < 200.0)){
	  if(pt > ProbeSCMinEt &&  fabs(sc->eta()) 
	     < BarrelMaxEta &&  tag->superCluster() != sc ){
	  tag_probe_invariant_mass_for_tree[mass_iterator] = Mass;
	  sc_eta_for_tree[mass_iterator] = sc->eta();
	  sc_et_for_tree[mass_iterator] = pt;
	  std::cout << "the mass of the tag - SC pair is: " << Mass 
		    <<"   ,  SC iterator: " << indexSCH_all 
		    << "  , mass iterator: " << mass_iterator << std::endl;
	  //... in this loop as far as I understand: takes the 
	  //     collection of the UniqueElectrons (=collection of PixelMatchGsfElectronRef)
	  //... and serches whether the supercluster energy of 
	  //     any of them matches the SCHybrid one (outer for loop)
	  //... if they match the tag_probe invariant mass is stored 
	  //     ---??? it is the tag-sc invariant mass for All the SC in the event that
	  //... can be matched to a GsfElectron ==> is it true???? -- why do you write tag_probe??
	  for(std::vector<reco::PixelMatchGsfElectronRef>::const_iterator 
		Relec = UniqueElectrons.begin(); Relec != UniqueElectrons.end(); ++Relec)
	    {
	      reco::PixelMatchGsfElectronRef elec;
	      elec = *Relec;
	      double denergy = sc->energy() - elec->superCluster().get()->energy();
	      if(fabs(denergy) < ProbeRecoEleSCMaxDE){
		tag_probe_invariant_mass_pass_for_tree[mass_iterator] = Mass;
		std::cout << "PASSING PRESEL:  the mass of the tag - ele pair is: " << Mass 
			  <<"   ,  SC iterator: " << indexSCH_all 
			  << "  , mass iterator: " << mass_pass_iterator << std::endl;
		++mass_pass_iterator;
	      }
	    }//end of loop over pixel match gsf electrons
	  ++ mass_iterator;
	  }
	  ++indexSCH_all;
	}//end of loop over Hybrid SC to produce MTP over whole mass range

	for(reco::SuperClusterCollection::const_iterator 
	      scIsland = SCIsland->begin(); scIsland != SCIsland->end();++scIsland){
	  const reco::SuperClusterRef sc(pSCIsland, indexSCI_all);
	  double pt = sc->energy()/(cosh(sc->eta()));
	  scVect.SetPtEtaPhi(pt, sc->eta(), sc->phi());
	  probeCandVect = scVect - ZeeVertexVect;
	  double new_pt = sc->energy()/(cosh(probeCandVect.Eta()));
	  probeCand.SetPtEtaPhiE(new_pt, probeCandVect.Eta()
				 , probeCandVect.Phi(), sc->energy()); 
	  TLorentzVector  V = (tagLorentzVec + probeCand);
	  double Mass = V.M();
	  // if((Mass_island >0.0) && (Mass_island < 200.0)){ 
	  if (pt > ProbeSCMinEt && 
	      fabs(sc->eta()) > EndcapMinEta && fabs(sc->eta()) < EndcapMaxEta 
	      && tag->superCluster() != sc){
	  tag_probe_invariant_mass_for_tree[ mass_iterator] = Mass;
	  sc_eta_for_tree[mass_iterator] = sc->eta();
	  sc_et_for_tree[mass_iterator] = pt;
	  std::cout << "the mass of the tag - SC pair is: " << Mass 
		    <<"   ,  island SC iterator: " << indexSCI_all 
		    << "  , mass iterator: " << mass_iterator << std::endl;
	  for(std::vector<reco::PixelMatchGsfElectronRef>::const_iterator 
		Relec = UniqueElectrons.begin(); Relec != UniqueElectrons.end(); ++Relec)
	    {
	      reco::PixelMatchGsfElectronRef elec;
	      elec = *Relec;
	      double denergy = sc->energy() - elec->superCluster().get()->energy();
	      if(fabs(denergy) < ProbeRecoEleSCMaxDE){
		tag_probe_invariant_mass_pass_for_tree[mass_iterator] = Mass;
		 std::cout << "PASSING PRESEL: the mass of the tag - SC pair is: " << Mass 
			   <<"   ,  island SC iterator: " << indexSCI_all 
			   << "  , mass iterator: " << mass_iterator << std::endl;
		++mass_pass_iterator;
	      }
	    }//end of loop over pixel match gsf electrons
	  ++ mass_iterator;
	  }
	  ++indexSCI_all;
	
	}//end of loop over island endcap SC to produce MTP over whole mass range
      }//end of loop the check there is not more than 1 probe in the defined mass range 
	}//end of loop over tags to find probes
      //
      //
      //...
      //... by this point we have made the Probe SC collection stored in..
      //... vector<reco::SuperClusterRef> ProbeSC  
      //... here we loop over  the Probe SC and fill the probe tree variables
      //
      //
      // EXPLICITLY check about the number of probes in this event:
      //
      int number_of_probes_in_event = ProbeSC.size();
      const int MAX_PROBES_PER_EVENT = 4;
      if (number_of_probes_in_event > MAX_PROBES_PER_EVENT) {
	edm::LogInfo("info") << "More than 4 probes in this event! Skipped!!!";
      }
      int probeIt =0;
      for(std::vector<reco::SuperClusterRef>::const_iterator 
	    Rprobe = ProbeSC.begin(); Rprobe != ProbeSC.end(); ++Rprobe)
	{
	  //edm::LogInfo("")<<"entering loop over probes";
	  if (number_of_probes_in_event > MAX_PROBES_PER_EVENT) break;
	  reco::SuperClusterRef probe;
	  probe = *Rprobe;
	  double probeEt = probe->energy()/(cosh(probe->eta()));
	  probe_sc_eta_for_tree[probeIt] = probe->eta();
	  probe_sc_phi_for_tree[probeIt] = probe->phi();
	  probe_sc_et_for_tree[probeIt] = probeEt;
	  probe_index_for_tree[probeIt] = probeIt;
	  if(fabs(probe->eta()) < BarrelMaxEta ||  (fabs(probe->eta()) > EndcapMinEta 
						    && fabs(probe->eta()) < EndcapMaxEta)){
	    probe_sc_pass_fiducial_cut[probeIt] = 1;
	  }
	  if(probeEt>ProbeSCMinEt){
	    probe_sc_pass_et_cut[probeIt] = 1;
	  }
	  // *******************************************************************
	  // MATCHING Probe SC to reco Electrons *******************************
	  // for the 207 samples the custom way of probe SC matching fails *****
	  //                                                               *****
	  // 1st way: this is the custom way                               *****
	  // 2nd way: geometrical matching       TO BE IMPLEMENTED         *****
	  // 3rd way (DEFAULT) : detid matching seed SC                    *****
	  //********************************************************************
	  reco::PixelMatchGsfElectronRef probeEle;
	  unsigned int recoEle_int =0;
	  //////////////////////////////////////////////////////////////////////
	  if (ProbeSC2RecoElecMatchingMethod_ == "scEnergy") {
	  for(std::vector<reco::PixelMatchGsfElectronRef>::const_iterator 
		Relec = UniqueElectrons.begin(); Relec != UniqueElectrons.end()
		; ++Relec)
	    {
	      reco::PixelMatchGsfElectronRef elec;
	      elec = *Relec;
	      double denergy = probe->energy() - 
		elec->superCluster().get()->energy();
	      if(fabs(denergy) < ProbeRecoEleSCMaxDE){
		recoEle_int = recoEle_int+1;
		probeEle = elec;
	      }
	    }//end of loop over pixel match gsf electrons
	  }
	  else if (ProbeSC2RecoElecMatchingMethod_ == "seedDetId") {
	    DetId id_for_probe = probe->seed()->hitsAndFractions()[0].first;
	    for(std::vector<reco::PixelMatchGsfElectronRef>::const_iterator 
	       Relec = UniqueElectrons.begin(); Relec != UniqueElectrons.end();
	       ++Relec) {
	      reco::PixelMatchGsfElectronRef elec;
	      elec = *Relec; edm::LogInfo("deb") << "Dereferencing successful!";
	      DetId id1 = elec->superCluster()->seed()->hitsAndFractions()[0].first;
	      if (id1 == id_for_probe) { ++recoEle_int;
	      probeEle = elec;
	      }
	    }
	  }
	  else if (ProbeSC2RecoElecMatchingMethod_ == "superCluster") {
	    for(std::vector<reco::PixelMatchGsfElectronRef>::const_iterator 
	       Relec = UniqueElectrons.begin(); Relec != UniqueElectrons.end();
	       ++Relec) {
	      reco::PixelMatchGsfElectronRef elec;
	      elec = *Relec; 
	      if (elec->superCluster() == probe) { ++recoEle_int;
	      probeEle = elec;
	      }
	    }
	  }
	  //////////////////////////////////////////////////////////////////////
	  //********************************************************************
	  if(recoEle_int>0) {
	    probe_pass_recoEle_cut[probeIt] =recoEle_int;
	    probe_charge_for_tree[probeIt] = probeEle->charge();
	    probe_ele_eta_for_tree[probeIt] = probeEle->eta();
	    probe_ele_et_for_tree[probeIt] = probeEle->et();
	    probe_ele_phi_for_tree[probeIt] =probeEle->phi();
	    probe_ele_Xvertex_for_tree[probeIt] =probeEle->vx();
	    probe_ele_Yvertex_for_tree[probeIt] =probeEle->vy();
	    probe_ele_Zvertex_for_tree[probeIt] =probeEle->vz();
	    probe_classification_index_for_tree[probeIt] = probeEle->classification();
	    double ProbeTIP = probeEle->gsfTrack()->d0();
	    //  edm::LogInfo("")<<"Probe TIP" << ProbeTIP;
	    if(fabs(ProbeTIP) < TagTIPCut_) probe_pass_tip_cut[probeIt] = 1;
	    double sum_of_pt_probe_ele =0.0;
	    for(reco::TrackCollection::const_iterator 
		  tr = ctfTracks->begin(); tr != ctfTracks->end();++tr)
	      {
		double LIP = probeEle->gsfTrack()->dz() - tr->dz();
		//if(tr->pt() >TrackInIsoConeMinPt && fabs(LIP) < 0.2){
		if(tr->pt() >TrackInIsoConeMinPt && fabs(LIP) < LIPCut_){
		  double dr_ctf_ele = CalcDR_Val(tr->eta()
						 , probeEle->trackMomentumAtVtx().eta()
						 , tr->phi()
						 , probeEle->trackMomentumAtVtx().phi());
		  if((dr_ctf_ele>IsoConeMinDR) && (dr_ctf_ele<IsoConeMaxDR)){
		    double cft_pt_2 = (tr->pt())*(tr->pt());
		    sum_of_pt_probe_ele += cft_pt_2;
		  }//end of if in ctf tracks in isolation cone around tag electron candidate sum squared pt values
		}//end of only loop over tracks with Inner Pt >1.5GeV
	      }//end of cft loop to work out isolation value
	    double isolation_value_probe_ele = sum_of_pt_probe_ele
	      /(probeEle->trackMomentumAtVtx().Rho()*probeEle->trackMomentumAtVtx().Rho());
	    probe_isolation_value[probeIt] = isolation_value_probe_ele;
	    if(isolation_value_probe_ele<IsoMaxSumPt){
	      probe_pass_iso_cut[probeIt] =1;
	    }
	    
	    // Find entry in robust electron ID map corresponding electron
	    reco::ElectronIDAssociationCollection::const_iterator electronIDAssocItrRobust;
	    electronIDAssocItrRobust = pEleIDRobust->find(probeEle);
	    const reco::ElectronIDRef& id_ele_robust = electronIDAssocItrRobust->val;
	    bool cutBasedIDRobust = false;
	    cutBasedIDRobust = id_ele_robust->cutBasedDecision();
	    if(cutBasedIDRobust == true) probe_pass_id_cut_robust[probeIt] =1;
	    //   std::cout << "cutbased decision robust: " << cutBasedIDRobust << std::endl;
	 
	   
	    // Find entry in loose electron ID map corresponding electron
	    reco::ElectronIDAssociationCollection::const_iterator electronIDAssocItrLoose;
	    electronIDAssocItrLoose = pEleIDLoose->find(probeEle);
	    const reco::ElectronIDRef& id_ele_loose = electronIDAssocItrLoose->val;
	    bool cutBasedIDLoose = false;
	    cutBasedIDLoose = id_ele_loose->cutBasedDecision();
	    if(cutBasedIDLoose == true) probe_pass_id_cut_loose[probeIt] =1;
	    // std::cout << "cutbased decision loose: " << cutBasedIDLoose << std::endl;

	    // Find entry in tight electron ID map corresponding electron
	    reco::ElectronIDAssociationCollection::const_iterator electronIDAssocItrTight;
	    electronIDAssocItrTight = pEleIDTight->find(probeEle);
	    const reco::ElectronIDRef& id_ele_tight = electronIDAssocItrTight->val;
	    bool cutBasedIDTight = false;
	    cutBasedIDTight = id_ele_tight->cutBasedDecision();
	    if(cutBasedIDTight == true) probe_pass_id_cut_tight[probeIt] =1;
	    // std::cout << "cutbased decision tight: " << cutBasedIDTight << std::endl;

	    // Find entry in tight electron ID map corresponding electron
	    reco::ElectronIDAssociationCollection::const_iterator 
	      electronIDAssocItrTightRobust;
	    electronIDAssocItrTightRobust = pEleIDTightRobust->find(probeEle);
	    const reco::ElectronIDRef& 
	      id_ele_tight_robust = electronIDAssocItrTightRobust->val;
	    bool cutBasedIDTightRobust = false;
	    cutBasedIDTightRobust = id_ele_tight_robust->cutBasedDecision();
	    if(cutBasedIDTightRobust == true) probe_pass_id_cut_tight_robust[probeIt] =1;
	    std::cout << "cutbased decision tight robust: " 
		      << cutBasedIDTightRobust << std::endl;
	    //const reco::ClusterShapeRef& shapeRef = getClusterShape(probeEle);
	    double hOverE = probeEle->hadronicOverEm();
	    //double sigmaee = sqrt(shapeRef->covEtaEta());
	    double deltaPhiIn = probeEle->deltaPhiSuperClusterTrackAtVtx();
	    double deltaEtaIn = probeEle->deltaEtaSuperClusterTrackAtVtx();

	    std::cout << "hoe, Dphiin, Detain: " <<  hOverE << deltaPhiIn 
		      <<  deltaEtaIn << std::endl;

	    //no lets ask whether the probe passes the trigger
	    int trigger_int_probe = 0;
	    if(no_trigger_info == false && useTriggerInfo ==true){
	      /*
	      for(reco::ElectronCollection::const_iterator 
		    HLT = HLTObj->begin(); HLT != HLTObj->end();++HLT){
		double dr_ele_HLT = CalcDR_Val(probeEle->eta(), HLT->eta()
					       , probeEle->phi(), HLT->phi());
		if(fabs(dr_ele_HLT) < ProbeHLTObjMaxDR) ++trigger_int_probe;
	      }//end of loop over HLT filter objects
	    }//checking trigger info is there!
	    if(trigger_int_probe>0) probe_pass_trigger_cut[probeIt] = 1;

	      const int nTriggerObj(pHLT->sizeObjects());
	      const trigger::TriggerObjectCollection& TOC(pHLT->getObjects());
	      for (int iTrig =0;iTrig != nTriggerObj; ++iTrig) {
		const trigger::TriggerObject& TO(TOC[iTrig]);
		if (abs(TO.id())==11 ) { // demand it to be an electron
		  double dr_ele_HLT = CalcDR_Val(probeEle->eta(),TO.eta()
						 , probeEle->phi(), TO.phi());
		  if (fabs(dr_ele_HLT) < ProbeHLTObjMaxDR)++trigger_int_probe;
		}
	      }
	      */
	      // find how many relevant
	      const int iF = pHLT->filterIndex(HLTFilterType_);
	      // find how many objects there are
	      const trigger::Keys& KEYS(pHLT->filterKeys(iF));
	      const int nK(KEYS.size());
	      // loop over these objects to see whether they match
	      const trigger::TriggerObjectCollection& TOC(pHLT->getObjects());
	      for (int iTrig = 0;iTrig <nK; ++iTrig ) {
		const trigger::TriggerObject& TO(TOC[KEYS[iTrig]]);
		if (abs(TO.id())==11 ) { // demand it to be an electron
		  double dr_ele_HLT = CalcDR_Val(probeEle->eta(),TO.eta()
						 , probeEle->phi(), TO.phi());
		  if (fabs(dr_ele_HLT) < ProbeHLTObjMaxDR) ++trigger_int_probe;
		}
	      }

	    }
	    if(trigger_int_probe>0) probe_pass_trigger_cut[probeIt] = 1;
	    
	  }//end of if probe SC becomes reconstructed electron
	  ++probeIt; 
	}//end of loop over probes

      probe_tree->Fill();
      
      edm::LogInfo("") << "End of analyse()";
}//end of analyze 


// ------------ method called once each job just before starting event loop  ------------
void 
ElectronTPValidator::beginJob(const edm::EventSetup&)
{
  TString filename_histo = outputFile_;
  //  histofile = TFile::Open(filename_histo,"RECREATE");
  histofile = new TFile(filename_histo,"RECREATE");

  h_eta = new  TH1F("eta","",100,-5,5);
  h_tip = new TH1F("TIP","",100,-0.1,0.1);
  h_eta->GetXaxis()->SetTitle(" tag supercluster #eta");
  h_tip->GetXaxis()->SetTitle(" tag gsfTrack()->d0() #eta");

  h_track_LIP = new TH1F("TrackLIP","Track Collection LIP",100,-0.5,0.5);
  h_track_LIP->GetXaxis()->SetTitle("probeEle->gsfTrack()->dz() - tr->dz() ");
  //
  h_track_pt = new TH1F("TrackPt","Track Pt",100,0.,80.);
  h_track_pt->GetXaxis()->SetTitle("Track p_{T} (GeV)");

  probe_tree =  new TTree("probe_tree","Tree to store probe variables");
  probe_tree->Branch("tag_charge", tag_charge_for_tree, "tag_charge[4]/I");
  probe_tree->Branch("tag_ele_eta",tag_ele_eta_for_tree,"tag_ele_eta[4]/D");
  probe_tree->Branch("tag_ele_phi",tag_ele_phi_for_tree,"tag_ele_phi[4]/D");
  probe_tree->Branch("tag_ele_et",tag_ele_et_for_tree,"tag_ele_et[4]/D");
  probe_tree->Branch("tag_ele_vertex_x",tag_ele_Xvertex_for_tree,"tag_ele_vertex_x[4]/D");
  probe_tree->Branch("tag_ele_vertex_y",tag_ele_Yvertex_for_tree,"tag_ele_vertex_y[4]/D");
  probe_tree->Branch("tag_ele_vertex_z",tag_ele_Zvertex_for_tree,"tag_ele_vertex_z[4]/D");
  probe_tree->Branch("probe_ele_eta",probe_ele_eta_for_tree,"probe_ele_eta[4]/D");
  probe_tree->Branch("probe_ele_phi",probe_ele_phi_for_tree,"probe_ele_phi[4]/D");
  probe_tree->Branch("probe_ele_et",probe_ele_et_for_tree,"probe_ele_et[4]/D");
  probe_tree->Branch("probe_ele_vertex_x",probe_ele_Xvertex_for_tree,"probe_ele_vertex_x[4]/D");
  probe_tree->Branch("probe_ele_vertex_y",probe_ele_Yvertex_for_tree,"probe_ele_vertex_y[4]/D");
  probe_tree->Branch("probe_ele_vertex_z",probe_ele_Zvertex_for_tree,"probe_ele_vertex_z[4]/D");
  probe_tree->Branch("probe_sc_eta",probe_sc_eta_for_tree,"probe_sc_eta[4]/D");
  probe_tree->Branch("probe_sc_phi",probe_sc_phi_for_tree,"probe_sc_phi[4]/D");
  probe_tree->Branch("probe_sc_et",probe_sc_et_for_tree,"probe_sc_et[4]/D");
  probe_tree->Branch("probe_index",probe_index_for_tree,"probe_index[4]/I");
  probe_tree->Branch("probe_trigger_cut",probe_pass_trigger_cut,"probe_trigger_cut[4]/I");
  probe_tree->Branch("probe_charge", probe_charge_for_tree, "probe_charge[4]/I");
  probe_tree->Branch("probe_sc_fiducial_cut",probe_sc_pass_fiducial_cut,"probe_sc_fiducial_cut[4]/I");
  probe_tree->Branch("probe_sc_et_cut",probe_sc_pass_et_cut,"probe_sc_et_cut[4]/I");
  probe_tree->Branch("probe_ele_fiducial_cut",probe_ele_pass_fiducial_cut,"probe_ele_fiducial_cut[4]/I");
  probe_tree->Branch("probe_ele_et_cut",probe_ele_pass_et_cut,"probe_ele_et_cut[4]/I");
  probe_tree->Branch("probe_recoEle_cut",probe_pass_recoEle_cut,"probe_recoEle_cut[4]/I");
  probe_tree->Branch("probe_iso_cut",probe_pass_iso_cut,"probe_iso_cut[4]/I");
  probe_tree->Branch("probe_iso_cut",probe_pass_iso_cut,"probe_iso_cut[4]/I");
  probe_tree->Branch("probe_classification",probe_classification_index_for_tree,"probe_classification[4]/I");
  probe_tree->Branch("probe_isolation_value",probe_isolation_value,"probe_isolation_value[4]/D");
  probe_tree->Branch("tag_isolation_value",tag_isolation_value,"tag_isolation_value[4]/D");
  probe_tree->Branch("tag_classification",tag_classification_index_for_tree,"tag_classification[4]/I");
  probe_tree->Branch("probe_id_cut_robust",probe_pass_id_cut_robust,"probe_id_cut_robust[4]/I");
  probe_tree->Branch("probe_id_cut_loose",probe_pass_id_cut_loose,"probe_id_cut_loose[4]/I");
  probe_tree->Branch("probe_id_cut_tight",probe_pass_id_cut_tight,"probe_id_cut_tight[4]/I");
  probe_tree->Branch("probe_id_cut_tight_robust",probe_pass_id_cut_tight_robust,"probe_id_cut_tight_robust[4]/I");
  probe_tree->Branch("probe_tip_cut",probe_pass_tip_cut,"probe_tip_cut[4]/I");

  probe_tree->Branch("numberOfHLTFilterObjects",&numberOfHLTFilterObjects,"numberOfHLTFilterObjects/I");

  //event variable trees

  probe_tree->Branch("tag_probe_invariant_mass",tag_probe_invariant_mass_for_tree
		     ,"tag_probe_invariant_mass_for_tree[100]/D");
  probe_tree->Branch("tag_probe_invariant_mass_pass"
		     ,tag_probe_invariant_mass_pass_for_tree
		     ,"tag_probe_invariant_mass_pass_for_tree[100]/D");
  probe_tree->Branch("sc_eta",sc_eta_for_tree,"sc_eta_for_tree[100]/D");
  probe_tree->Branch("sc_et",sc_et_for_tree,"sc_et_for_tree[100]/D");
  // debugging info:
  probe_tree->Branch("tag_number_in_event",&tag_number_in_event
		     ,"tag_number_in_event/I");
  probe_tree->Branch("elec_number_in_event",&elec_number_in_event
		     ,"elec_number_in_event/I");
  probe_tree->Branch("elec_1_duplicate_removal",&elec_1_duplicate_removal
		     ,"elec_1_duplicate_removal/I");
  probe_tree->Branch("elec_2_isolated",&elec_2_isolated
		     ,"elec_2_isolated/I");
  probe_tree->Branch("elec_3_HLT_accepted",&elec_3_HLT_accepted
		     ,"elec_3_HLT_accepted/I");
  probe_tree->Branch("elec_0_after_cuts_before_iso",&elec_0_after_cuts_before_iso
		     ,"elec_0_after_cuts_before_iso/I");
  probe_tree->Branch("elec_0_et_cut",&elec_0_et_cut
		     ,"elec_0_et_cut/I");
  probe_tree->Branch("elec_0_tip_cut",&elec_0_tip_cut
		     ,"elec_0_tip_cut/I");
  probe_tree->Branch("elec_0_geom_cut",&elec_0_geom_cut
		     ,"elec_0_geom_cut/I");
}

// ------------ method called once each job just after ending the event loop  ------------
void 
ElectronTPValidator::endJob() {

   probe_tree->Print();
   //   probe_tree->Write();
   h_eta->Print();
   h_tip->Print();
   histofile->Write();
   histofile->Close();
   //  histofile->Write();
   //probe_tree->Write();
   TFile * newfile = new TFile("histos.root","RECREATE");
   h_eta->Write();
   h_tip->Write(); h_track_pt->Write();
   h_track_LIP->Write();
   newfile->Close();
   /*
   TCanvas *c = new TCanvas();
   h_eta->Draw();
   c->Print("eta.gif");
   h_tip->Draw();
   c->Print("tip.gif");
   h_track_pt->Draw();
   c->Print("track_pt.gif");
   h_track_LIP->Draw();
   c->Print("track_lip.gif");
   */

}

// This is just for the DR calculation
double ElectronTPValidator::CalcDR_Val(double eta1, double eta2
				    , double phi1, double phi2)
{
  double deta = eta1 - eta2;
  double dphi = phi1 - phi2;
  if(dphi > acos(-1.0)) dphi -= 2.0*acos(-1.0);
  if(dphi < -1.0* acos(-1.0)) dphi += 2.0*acos(-1.0);
  double dr = sqrt(deta*deta + dphi*dphi);
  return dr;
}


//define this as a plug-in
DEFINE_FWK_MODULE(ElectronTPValidator);
