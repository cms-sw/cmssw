// -*- C++ -*-
//
// Package:    GenPurposeSkimmerAcceptance
// Class:      GenPurposeSkimmerAcceptance
// 
/**\class GenPurposeSkimmerAcceptance GenPurposeSkimmerAcceptance.cc 
EWKSoftware/EWKEleSelectionOpt/src/GenPurposeSkimmerAcceptance.cc

 Description: <one line class summary>
 ===============
 Implementation:
 ===============
   This is a general purpose SkimmerAcceptance
   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     It reads datasets and keeps only the analysis-relevant information 
     and stores it in a simple TTree. 
     Code Inspired by the T&P code by Claire Timlin
     Note: a similar code to read PAT tuples is already available
 
   History:
16.10.08: first version
24.10.08: added ECAL/HCAL isolation + sigma ieta ieta (S. Harper)
30.10.08: all isolations use isodeposits
          all parameters are untracked
18.03.09: modified to store just the 4 highest ET gsf electrons in the event
02.04.09: version for redigi including particle flow MET + gen level MET
04.04.09: version for redigi including tcMET, MET eta dropped
22.04.09: version for redigi including MET Type1 corrections
23.04.09: version completely changes to read from PAT.......................
07.09.09: version for 3_1_2 version
08.09.09: version for 3_1_2 that keeps all the trigger info and reduced
          number of the other collections
28.09.09: version that also stores MC information and SC for the acceptance
          calculation

  Further Information/Inquiries:
   Nikos Rompotis - Imperial College London
   Nikolaos.Rompotis@Cern.ch

*/
//
// Original Author:  Nikolaos Rompotis
//         Created:  Thu Oct 16 17:11:55 CEST 2008
// $Id: GenPurposeSkimmerAcceptance.cc,v 1.1 2009/09/28 13:50:54 rompotis Exp $
//
//

#include "ElectroWeakAnalysis/WENu/interface/GenPurposeSkimmerAcceptance.h"
#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "DataFormats/EgammaReco/interface/PreshowerClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
//#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
GenPurposeSkimmerAcceptance::GenPurposeSkimmerAcceptance(const edm::ParameterSet& ps)

{
//
//   I N P U T      P A R A M E T E R S
//
  // output file name
  outputFile_ = ps.getUntrackedParameter<std::string>("outputfile");
  //
  // Electron Collection
  ElectronCollection_=ps.getUntrackedParameter<edm::InputTag>("ElectronCollection");
  //
  // MC:
  MCCollection_ = ps.getUntrackedParameter<edm::InputTag>("MCCollection");
  MCMatch_Deta_ = ps.getUntrackedParameter<double>("MCMatch_Deta",0.1);
  MCMatch_Dphi_ = ps.getUntrackedParameter<double>("MCMatch_Dphi",0.35);
  //
  // MET Collections:
  MetCollectionTag_ = ps.getUntrackedParameter<edm::InputTag>("MetCollectionTag");
  t1MetCollectionTag_ = ps.getUntrackedParameter<edm::InputTag>("t1MetCollectionTag");
  pfMetCollectionTag_ = ps.getUntrackedParameter<edm::InputTag>("pfMetCollectionTag");
  tcMetCollectionTag_ = ps.getUntrackedParameter<edm::InputTag>("tcMetCollectionTag");
  genMetCollectionTag_ = ps.getUntrackedParameter<edm::InputTag>("genMetCollectionTag");
  t1MetCollectionTagTwiki_ = ps.getUntrackedParameter<edm::InputTag>("t1MetCollectionTagTwiki");
  //
  // HLT parameters:
  // allow info for 2 paths and 2 filters
  // ---------------------------------------------------------------------------
  HLTCollectionE29_= ps.getUntrackedParameter<edm::InputTag>("HLTCollectionE29");
  HLTCollectionE31_= ps.getUntrackedParameter<edm::InputTag>("HLTCollectionE31");
  HLTTriggerResultsE29_ = ps.getUntrackedParameter<edm::InputTag>("HLTTriggerResultsE29");
  HLTTriggerResultsE31_ = ps.getUntrackedParameter<edm::InputTag>("HLTTriggerResultsE31");
  //HLTPath_ = ps.getUntrackedParameter<std::string>("HLTPath","HLT_Ele15_LW_L1R");
  //HLTFilterType_ =ps.getUntrackedParameter<edm::InputTag>("HLTFilterType");
  // all the HLT Paths here:
  // triggers in the  8e29 menu
  HLTPath_[0] = "HLT_Ele10_LW_L1R";
  HLTFilterType_[0] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt10PixelMatchFilter","","HLT8E29");
  HLTPath_[1] = "HLT_Ele10_LW_EleId_L1R";
  HLTFilterType_[1] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDphiFilter","","HLT8E29");
  HLTPath_[2] = "HLT_Ele15_LW_L1R";
  HLTFilterType_[2] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15PixelMatchFilter","","HLT8E29");
  HLTPath_[3] = "HLT_Ele15_SC10_LW_L1R";
  HLTFilterType_[3] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15ESDoubleSC10","","HLT8E29");
  HLTPath_[4] = "HLT_Ele15_SiStrip_L1R";
  HLTFilterType_[4] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronSiStripEt15PixelMatchFilter","","HLT8E29");
  HLTPath_[5] = "HLT_Ele20_LW_L1R";
  HLTFilterType_[5] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15EtFilterESet20","","HLT8E29");
  HLTPath_[6] = "HLT_DoubleEle5_SW_L1R";
  HLTFilterType_[6] = edm::InputTag("hltL1NonIsoHLTNonIsoDoubleElectronEt5PixelMatchFilter","","HLT8E29");
  HLTPath_[7] = "HLT_Ele15_SC10_LW_L1R";
  HLTFilterType_[7] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15ESDoubleSC10","","HLT8E29");
  HLTPath_[8] = "tba";
  HLTFilterType_[8] = edm::InputTag("tba");
  HLTPath_[9] = "tba";
  HLTFilterType_[9] = edm::InputTag("tba");
  // e31 menu
  HLTPath_[10] = "HLT_Ele10_SW_L1R";
  HLTFilterType_[10] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt10PixelMatchFilter","","HLT");
  HLTPath_[11] = "HLT_Ele15_SW_L1R";
  HLTFilterType_[11] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15PixelMatchFilter","","HLT");
  HLTPath_[12] = "HLT_Ele15_SiStrip_L1R"; // <--- same as [4]
  HLTFilterType_[12] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronSiStripEt15PixelMatchFilter","","HLT");
  HLTPath_[13] = "HLT_Ele15_SW_LooseTrackIso_L1R";
  HLTFilterType_[13] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15LTITrackIsolFilter","","HLT");
  HLTPath_[14] = "HLT_Ele15_SW_EleId_L1R";
  HLTFilterType_[14] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15EleIdDphiFilter","","HLT");
  HLTPath_[15] = "HLT_Ele20_SW_L1R";
  HLTFilterType_[15] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt20PixelMatchFilter","","HLT");
  HLTPath_[16] = "HLT_Ele20_SiStrip_L1R";
  HLTFilterType_[16] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronSiStripEt20PixelMatchFilter","","HLT");
  HLTPath_[17] = "HLT_Ele25_SW_L1R";
  HLTFilterType_[17] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15EtFilterESet25","","HLT");
  HLTPath_[18] = "HLT_Ele25_SW_EleId_LooseTrackIso_L1R";
  HLTFilterType_[18] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15EleIdTrackIsolFilterESet25LTI","","HLT");
  HLTPath_[19] = "HLT_DoubleEle10_SW_L1R";
  HLTFilterType_[19] = edm::InputTag("hltL1NonIsoHLTNonIsoDoubleElectronEt10PixelMatchFilter","","HLT");
  HLTPath_[20] = "HLT_Ele15_SC15_SW_EleId_L1R";
  HLTFilterType_[20] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15EleIdESDoubleSC15","","HLT");
  HLTPath_[21] = "HLT_Ele15_SC15_SW_LooseTrackIso_L1R";
  HLTFilterType_[21] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15LTIESDoubleSC15","","HLT");
  HLTPath_[22] = "HLT_Ele20_SC15_SW_L1R";
  HLTFilterType_[22] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt20ESDoubleSC15","","HLT");
  HLTPath_[23] = "tba";
  HLTFilterType_[23] = edm::InputTag("tba");
  HLTPath_[24] = "tba";
  HLTFilterType_[24] = edm::InputTag("tba");
  // matching HLT objects to electrons
  ProbeHLTObjMaxDR= ps.getUntrackedParameter<double>("ProbeHLTObjMaxDR",0.2);
  //
  // ----------------------------------------------------------------------------
  //
  // detector geometry
  //
  BarrelMaxEta = ps.getUntrackedParameter<double>("BarrelMaxEta");
  EndcapMinEta = ps.getUntrackedParameter<double>("EndcapMinEta");
  EndcapMaxEta = ps.getUntrackedParameter<double>("EndcapMaxEta");
  // 

}


GenPurposeSkimmerAcceptance::~GenPurposeSkimmerAcceptance()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
GenPurposeSkimmerAcceptance::analyze(const edm::Event& evt, const edm::EventSetup& es)
{
  // MC Collection ------------------------------------------------
  
  edm::Handle<reco::GenParticleCollection> pGenPart;
  evt.getByLabel(MCCollection_, pGenPart);
  if ( not  pGenPart.isValid() ) {
    std::cout <<"Error! Can't get "<<MCCollection_.label() << std::endl;
    return;
  }
  //
  const reco::GenParticleCollection *McCand = pGenPart.product();
  int const MAX_PROBES_MC = 10;
  // kine ele collection: the first 20 electrons ordered in ET
  for (int i=0; i<MAX_PROBES_MC; ++i) {
    mc_ele_eta[i] = -999;
    mc_ele_phi[i] = -999;
    mc_ele_et[i] = -999;
    mc_ele_vertex_x[i] = -999;
    mc_ele_vertex_y[i] = -999;
    mc_ele_vertex_z[i] = -999;
    mc_ele_mother[i] = -999;
    mc_ele_charge[i] = -999;
    mc_ele_status[i] = -999;
  }
  std::vector<double> Es;
  std::vector<int> EsIndex;
  reco::GenParticleCollection::const_iterator  mcelec;
  int particle_index = 0;
  for (mcelec = McCand->begin(); mcelec !=  McCand->end(); ++mcelec) {
    reco::GenParticle probeEle;
    probeEle = *mcelec;
    if(abs(probeEle.pdgId())==11) { // no status requirement here
      double probeE = probeEle.energy()/cosh(probeEle.eta());
      Es.push_back(probeE);
      EsIndex.push_back(particle_index);
      //std::cout<< "part index: "<<particle_index<<", e="<<probeE<<std::endl;
    }
    ++particle_index;
  }

  const  int mc_particles = (int) Es.size();
  if (mc_particles == 0) return;
  int *mcsorted = new int[mc_particles];
  double *mce = new double[mc_particles];
  for (int i=0; i<mc_particles; ++i)
    mce[i]=Es[i];
  //
  // note on the statuses: final state particles in Pythia have status 1
  //
  // this is from the TWIKI *************************************************
  // 0     null entry
  // 1 	   existing entry ­ not decayed or fragmented, 
  //       represents the final state as given by the generator
  // 2 	   decayed or fragmented entry 
  //       (i.e. decayed particle or parton produced in shower.)
  // 3 	   documentation entry, defined separately from the event history. 
  //       "This includes the two incoming colliding particles and partons 
  //       produced in hard interaction." [ * ] 
  //       (unstable entry produced in Pythia hard interaction ?) 
  // ************************************************************************
  TMath::Sort(mc_particles, mce, mcsorted, true);
  for( int probeIt = 0; probeIt < mc_particles; ++probeIt)
    {
      if (probeIt >= MAX_PROBES_MC) break;
      //
      int elec_index = EsIndex[ mcsorted[probeIt] ];
      reco::GenParticleCollection::const_iterator
	Rprobe = McCand->begin() + elec_index;
      reco::GenParticle probeEle;
      probeEle = *Rprobe;
      //std::cout << "index=" << mcsorted[probeIt] << ", energy: " 
      //		<< probeEle.energy() << std::endl;
      //
      mc_ele_eta[probeIt] = probeEle.eta();
      mc_ele_phi[probeIt] = probeEle.phi();
      mc_ele_et[probeIt] = probeEle.energy()/cosh(probeEle.eta());
      mc_ele_vertex_x[probeIt] = probeEle.vx();
      mc_ele_vertex_y[probeIt] = probeEle.vy();
      mc_ele_vertex_z[probeIt] = probeEle.vz();
      mc_ele_charge[probeIt] = probeEle.charge();
      mc_ele_status[probeIt] = probeEle.status();
      const  reco::Candidate *mother;
      mother = probeEle.mother();
      mc_ele_mother[probeIt] =   probeEle.mother()->pdgId();
    }

  delete []  mcsorted;
  delete []  mce;
  //
  // SC collections *********************************************************
  //
  // collections that we store:
  //0 recoCaloClusters "hybridSuperClusters" "hybridBarrelBasicClusters" "RECO"
  //1 recoSuperClusters "correctedHybridSuperClusters" "" "RECO"
  //2 recoSuperClusters "correctedMulti5x5SuperClustersWithPreshower" "" "RECO"
  //3 recoSuperClusters "hybridSuperClusters" "" "RECO"
  //4 recoSuperClusters "multi5x5SuperClusters" 
  //                                       "multi5x5EndcapSuperClusters" "RECO"
  //5 recoSuperClusters "multi5x5SuperClustersWithPreshower" "" "RECO"
  //6 recoPreshowerClusters "multi5x5SuperClustersWithPreshower" 
  //                                                "preshowerXClusters" "RECO"
  //7 recoPreshowerClusters "multi5x5SuperClustersWithPreshower" 
  //                                                "preshowerYClusters" "RECO"
  //
  //edm::InputTag sc0_tag("hybridSuperClusters", "hybridBarrelBasicClusters", 
  //			"RECO");
  edm::InputTag sc1_tag("correctedHybridSuperClusters", "", "RECO");
  edm::InputTag sc2_tag("correctedMulti5x5SuperClustersWithPreshower", "",
			"RECO");
  edm::InputTag sc3_tag("hybridSuperClusters", "", "RECO");
  edm::InputTag sc4_tag("multi5x5SuperClusters", "multi5x5EndcapSuperClusters",
			"RECO");
  edm::InputTag sc5_tag("multi5x5SuperClustersWithPreshower", "", "RECO");
  //edm::InputTag sc6_tag("multi5x5SuperClustersWithPreshower",
  //			"preshowerXClusters", "RECO");
  //edm::InputTag sc7_tag("multi5x5SuperClustersWithPreshower",
  //			"preshowerYClusters", "RECO");
  //
  // storing 8 highest ET SC from each category
  const int MAX_SC = 8;
  //
  // CASE 0 ------------------------------------------------------------------
  /*
  for (int i=0; i<MAX_SC; ++i) {
    sc0_eta[i] = -99;
    sc0_phi[i] = -99;
    sc0_et[i] = -99;
  }
  //
  edm::Handle<reco::CaloClusterCollection> pSC0;
  evt.getByLabel(sc0_tag, pSC0);
  const reco::CaloClusterCollection *sc0 = pSC0.product();
  std::vector<double> Es_sc0;
  std::vector<int> EsIndex_sc0;
  reco::CaloClusterCollection::const_iterator  sc0It;
  particle_index = 0;
  for (sc0It = sc0->begin(); sc0It !=  sc0->end(); ++sc0It) {
    reco::CaloCluster probeEle;
    probeEle = *sc0It;
    double probeE = probeEle.energy()/cosh(probeEle.eta());
    Es_sc0.push_back(probeE);
    EsIndex_sc0.push_back(particle_index);
    ++particle_index;
  }
  const  int sc0_particles = (int) Es_sc0.size();
  if (sc0_particles > 0) {
    int *mcsorted_sc0 = new int[sc0_particles];
    double *mce_sc0 = new double[sc0_particles];
    for (int i=0; i<sc0_particles; ++i)
      mce_sc0[i]=Es_sc0[i];
    //
    TMath::Sort(sc0_particles, mce_sc0, mcsorted_sc0, true);
    //
    for( int probeIt = 0; probeIt < sc0_particles; ++probeIt)
      {
	if (probeIt >= MAX_SC) break;
	//
	int elec_index = EsIndex_sc0[ mcsorted_sc0[probeIt] ];
	reco::CaloClusterCollection::const_iterator
	  Rprobe = sc0->begin() + elec_index;
	reco::CaloCluster probeEle;
	probeEle = *Rprobe;
	sc0_eta[probeIt] = probeEle.eta();
	sc0_phi[probeIt] = probeEle.phi();
	sc0_et[probeIt] = probeEle.energy()/cosh(probeEle.eta());
	//std::cout << "index=" << mcsorted_sc0[probeIt] << ", energy: " 
	//  << probeEle.energy() << std::endl;
      }
    //std::cout << "outside the loop" << std::endl;
    delete [] mcsorted_sc0;
    delete [] mce_sc0;
  }
  */
  //
  // CASE 1 ------------------------------------------------------------------
  for (int i=0; i<MAX_SC; ++i) {
    sc1_eta[i] = -99;
    sc1_phi[i] = -99;
    sc1_et[i] = -99;
  }
  //
  edm::Handle<reco::SuperClusterCollection> pSC1;
  evt.getByLabel(sc1_tag, pSC1);
  const reco::SuperClusterCollection *sc1 = pSC1.product();
  std::vector<double> Es_sc1;
  std::vector<int> EsIndex_sc1;
  reco::SuperClusterCollection::const_iterator  sc1It;
  particle_index = 0;
  for (sc1It = sc1->begin(); sc1It !=  sc1->end(); ++sc1It) {
    reco::SuperCluster probeEle;
    probeEle = *sc1It;
    double probeE = probeEle.energy()/cosh(probeEle.eta());
    Es_sc1.push_back(probeE);
    EsIndex_sc1.push_back(particle_index);
    ++particle_index;
  }
  const  int sc1_particles = (int) Es_sc1.size();
  if (sc1_particles > 0) {
    int *mcsorted_sc1 = new int[sc1_particles];
    double *mce_sc1 = new double[sc1_particles];
    for (int i=0; i<sc1_particles; ++i)
      mce_sc1[i]=Es_sc1[i];
    //
    TMath::Sort(sc1_particles, mce_sc1, mcsorted_sc1, true);
    //
    for( int probeIt = 0; probeIt < sc1_particles; ++probeIt)
      {
	if (probeIt >= MAX_SC) break;
	//
	int elec_index = EsIndex_sc1[ mcsorted_sc1[probeIt] ];
	reco::SuperClusterCollection::const_iterator
	  Rprobe = sc1->begin() + elec_index;
	reco::SuperCluster probeEle;
	probeEle = *Rprobe;
	sc1_eta[probeIt] = probeEle.eta();
	sc1_phi[probeIt] = probeEle.phi();
	sc1_et[probeIt] = probeEle.energy()/cosh(probeEle.eta());
	//std::cout << "index=" << mcsorted_sc1[probeIt] << ", energy: " 
	//  << probeEle.energy() << std::endl;
      }
    //std::cout << "outside the loop" << std::endl;
    delete [] mcsorted_sc1;
    delete [] mce_sc1;
  }

  // CASE 2 ------------------------------------------------------------------
  for (int i=0; i<MAX_SC; ++i) {
    sc2_eta[i] = -99;
    sc2_phi[i] = -99;
    sc2_et[i] = -99;
  }
  //
  edm::Handle<reco::SuperClusterCollection> pSC2;
  evt.getByLabel(sc2_tag, pSC2);
  const reco::SuperClusterCollection *sc2 = pSC2.product();
  std::vector<double> Es_sc2;
  std::vector<int> EsIndex_sc2;
  reco::SuperClusterCollection::const_iterator  sc2It;
  particle_index = 0;
  for (sc2It = sc2->begin(); sc2It !=  sc2->end(); ++sc2It) {
    reco::SuperCluster probeEle;
    probeEle = *sc2It;
    double probeE = probeEle.energy()/cosh(probeEle.eta());
    Es_sc2.push_back(probeE);
    EsIndex_sc2.push_back(particle_index);
    ++particle_index;
  }
  const  int sc2_particles = (int) Es_sc2.size();
  if (sc2_particles > 0) {
    int *mcsorted_sc2 = new int[sc2_particles];
    double *mce_sc2 = new double[sc2_particles];
    for (int i=0; i<sc2_particles; ++i)
      mce_sc2[i]=Es_sc2[i];
    //
    TMath::Sort(sc2_particles, mce_sc2, mcsorted_sc2, true);
    //
    for( int probeIt = 0; probeIt < sc2_particles; ++probeIt)
      {
	if (probeIt >= MAX_SC) break;
	//
	int elec_index = EsIndex_sc2[ mcsorted_sc2[probeIt] ];
	reco::SuperClusterCollection::const_iterator
	  Rprobe = sc2->begin() + elec_index;
	reco::SuperCluster probeEle;
	probeEle = *Rprobe;
	sc2_eta[probeIt] = probeEle.eta();
	sc2_phi[probeIt] = probeEle.phi();
	sc2_et[probeIt] = probeEle.energy()/cosh(probeEle.eta());
	//std::cout << "index=" << mcsorted_sc2[probeIt] << ", energy: " 
	//  << probeEle.energy() << std::endl;
      }
    //std::cout << "outside the loop" << std::endl;
    delete [] mcsorted_sc2;
    delete [] mce_sc2;
  }


  //
  // CASE 3 ------------------------------------------------------------------
  for (int i=0; i<MAX_SC; ++i) {
    sc3_eta[i] = -99;
    sc3_phi[i] = -99;
    sc3_et[i] = -99;
  }
  //
  edm::Handle<reco::SuperClusterCollection> pSC3;
  evt.getByLabel(sc3_tag, pSC3);
  const reco::SuperClusterCollection *sc3 = pSC3.product();
  std::vector<double> Es_sc3;
  std::vector<int> EsIndex_sc3;
  reco::SuperClusterCollection::const_iterator  sc3It;
  particle_index = 0;
  for (sc3It = sc3->begin(); sc3It !=  sc3->end(); ++sc3It) {
    reco::SuperCluster probeEle;
    probeEle = *sc3It;
    double probeE = probeEle.energy()/cosh(probeEle.eta());
    Es_sc3.push_back(probeE);
    EsIndex_sc3.push_back(particle_index);
    ++particle_index;
  }
  const  int sc3_particles = (int) Es_sc3.size();
  if (sc3_particles > 0) {
    int *mcsorted_sc3 = new int[sc3_particles];
    double *mce_sc3 = new double[sc3_particles];
    for (int i=0; i<sc3_particles; ++i)
      mce_sc3[i]=Es_sc3[i];
    //
    TMath::Sort(sc3_particles, mce_sc3, mcsorted_sc3, true);
    //
    for( int probeIt = 0; probeIt < sc3_particles; ++probeIt)
      {
	if (probeIt >= MAX_SC) break;
	//
	int elec_index = EsIndex_sc3[ mcsorted_sc3[probeIt] ];
	reco::SuperClusterCollection::const_iterator
	  Rprobe = sc3->begin() + elec_index;
	reco::SuperCluster probeEle;
	probeEle = *Rprobe;
	sc3_eta[probeIt] = probeEle.eta();
	sc3_phi[probeIt] = probeEle.phi();
	sc3_et[probeIt] = probeEle.energy()/cosh(probeEle.eta());
	//std::cout << "index=" << mcsorted_sc3[probeIt] << ", energy: " 
	//  << probeEle.energy() << std::endl;
      }
    //std::cout << "outside the loop" << std::endl;
    delete [] mcsorted_sc3;
    delete [] mce_sc3;
  }


  // CASE 4 ------------------------------------------------------------------
  for (int i=0; i<MAX_SC; ++i) {
    sc4_eta[i] = -99;
    sc4_phi[i] = -99;
    sc4_et[i] = -99;
  }
  //
  edm::Handle<reco::SuperClusterCollection> pSC4;
  evt.getByLabel(sc4_tag, pSC4);
  const reco::SuperClusterCollection *sc4 = pSC4.product();
  std::vector<double> Es_sc4;
  std::vector<int> EsIndex_sc4;
  reco::SuperClusterCollection::const_iterator  sc4It;
  particle_index = 0;
  for (sc4It = sc4->begin(); sc4It !=  sc4->end(); ++sc4It) {
    reco::SuperCluster probeEle;
    probeEle = *sc4It;
    double probeE = probeEle.energy()/cosh(probeEle.eta());
    Es_sc4.push_back(probeE);
    EsIndex_sc4.push_back(particle_index);
    ++particle_index;
  }
  const  int sc4_particles = (int) Es_sc4.size();
  if (sc4_particles > 0) {
    int *mcsorted_sc4 = new int[sc4_particles];
    double *mce_sc4 = new double[sc4_particles];
    for (int i=0; i<sc4_particles; ++i)
      mce_sc4[i]=Es_sc4[i];
    //
    TMath::Sort(sc4_particles, mce_sc4, mcsorted_sc4, true);
    //
    for( int probeIt = 0; probeIt < sc4_particles; ++probeIt)
      {
	if (probeIt >= MAX_SC) break;
	//
	int elec_index = EsIndex_sc4[ mcsorted_sc4[probeIt] ];
	reco::SuperClusterCollection::const_iterator
	  Rprobe = sc4->begin() + elec_index;
	reco::SuperCluster probeEle;
	probeEle = *Rprobe;
	sc4_eta[probeIt] = probeEle.eta();
	sc4_phi[probeIt] = probeEle.phi();
	sc4_et[probeIt] = probeEle.energy()/cosh(probeEle.eta());
	//std::cout << "index=" << mcsorted_sc4[probeIt] << ", energy: " 
	//  << probeEle.energy() << std::endl;
      }
    //std::cout << "outside the loop" << std::endl;
    delete [] mcsorted_sc4;
    delete [] mce_sc4;
  }


  //

  // CASE 5 ------------------------------------------------------------------
  for (int i=0; i<MAX_SC; ++i) {
    sc5_eta[i] = -99;
    sc5_phi[i] = -99;
    sc5_et[i] = -99;
  }
  //
  edm::Handle<reco::SuperClusterCollection> pSC5;
  evt.getByLabel(sc5_tag, pSC5);
  const reco::SuperClusterCollection *sc5 = pSC5.product();
  std::vector<double> Es_sc5;
  std::vector<int> EsIndex_sc5;
  reco::SuperClusterCollection::const_iterator  sc5It;
  particle_index = 0;
  for (sc5It = sc5->begin(); sc5It !=  sc5->end(); ++sc5It) {
    reco::SuperCluster probeEle;
    probeEle = *sc5It;
    double probeE = probeEle.energy()/cosh(probeEle.eta());
    Es_sc5.push_back(probeE);
    EsIndex_sc5.push_back(particle_index);
    ++particle_index;
  }
  const  int sc5_particles = (int) Es_sc5.size();
  if (sc5_particles > 0) {
    int *mcsorted_sc5 = new int[sc5_particles];
    double *mce_sc5 = new double[sc5_particles];
    for (int i=0; i<sc5_particles; ++i)
      mce_sc5[i]=Es_sc5[i];
    //
    TMath::Sort(sc5_particles, mce_sc5, mcsorted_sc5, true);
    //
    for( int probeIt = 0; probeIt < sc5_particles; ++probeIt)
      {
	if (probeIt >= MAX_SC) break;
	//
	int elec_index = EsIndex_sc5[ mcsorted_sc5[probeIt] ];
	reco::SuperClusterCollection::const_iterator
	  Rprobe = sc5->begin() + elec_index;
	reco::SuperCluster probeEle;
	probeEle = *Rprobe;
	sc5_eta[probeIt] = probeEle.eta();
	sc5_phi[probeIt] = probeEle.phi();
	sc5_et[probeIt] = probeEle.energy()/cosh(probeEle.eta());
	//std::cout << "index=" << mcsorted_sc5[probeIt] << ", energy: " 
	//  << probeEle.energy() << std::endl;
      }
    //std::cout << "outside the loop" << std::endl;
    delete [] mcsorted_sc5;
    delete [] mce_sc5;
  }



  //


  //
  // CASE 6 ------------------------------------------------------------------
  /*
  for (int i=0; i<MAX_SC; ++i) {
    sc6_eta[i] = -99;
    sc6_phi[i] = -99;
    sc6_et[i] = -99;
  }
  //
  edm::Handle<reco::PreshowerClusterCollection> pSC6;
  evt.getByLabel(sc6_tag, pSC6);
  const reco::PreshowerClusterCollection *sc6 = pSC6.product();
  std::vector<double> Es_sc6;
  std::vector<int> EsIndex_sc6;
  reco::PreshowerClusterCollection::const_iterator  sc6It;
  particle_index = 0;
  for (sc6It = sc6->begin(); sc6It !=  sc6->end(); ++sc6It) {
    reco::PreshowerCluster probeEle;
    probeEle = *sc6It;
    double probeE = probeEle.energy()/cosh(probeEle.eta());
    Es_sc6.push_back(probeE);
    EsIndex_sc6.push_back(particle_index);
    ++particle_index;
  }
  const  int sc6_particles = (int) Es_sc6.size();
  if (sc6_particles > 0) {
    int *mcsorted_sc6 = new int[sc6_particles];
    double *mce_sc6 = new double[sc6_particles];
    for (int i=0; i<sc6_particles; ++i)
      mce_sc6[i]=Es_sc6[i];
    //
    TMath::Sort(sc6_particles, mce_sc6, mcsorted_sc6, true);
    //
    for( int probeIt = 0; probeIt < sc6_particles; ++probeIt)
      {
	if (probeIt >= MAX_SC) break;
	//
	int elec_index = EsIndex_sc6[ mcsorted_sc6[probeIt] ];
	reco::PreshowerClusterCollection::const_iterator
	  Rprobe = sc6->begin() + elec_index;
	reco::PreshowerCluster probeEle;
	probeEle = *Rprobe;
	sc6_eta[probeIt] = probeEle.eta();
	sc6_phi[probeIt] = probeEle.phi();
	sc6_et[probeIt] = probeEle.energy()/cosh(probeEle.eta());
	//std::cout << "index=" << mcsorted_sc6[probeIt] << ", energy: " 
	//  << probeEle.energy() << std::endl;
      }
    //std::cout << "outside the loop" << std::endl;
    delete [] mcsorted_sc6;
    delete [] mce_sc6;
  }
  */
  //

  // CASE 7 ------------------------------------------------------------------
  /*
  for (int i=0; i<MAX_SC; ++i) {
    sc7_eta[i] = -99;
    sc7_phi[i] = -99;
    sc7_et[i] = -99;
  }
  //
  edm::Handle<reco::PreshowerClusterCollection> pSC7;
  evt.getByLabel(sc7_tag, pSC7);
  const reco::PreshowerClusterCollection *sc7 = pSC7.product();
  std::vector<double> Es_sc7;
  std::vector<int> EsIndex_sc7;
  reco::PreshowerClusterCollection::const_iterator  sc7It;
  particle_index = 0;
  for (sc7It = sc7->begin(); sc7It !=  sc7->end(); ++sc7It) {
    reco::PreshowerCluster probeEle;
    probeEle = *sc7It;
    double probeE = probeEle.energy()/cosh(probeEle.eta());
    Es_sc7.push_back(probeE);
    EsIndex_sc7.push_back(particle_index);
    ++particle_index;
  }
  const  int sc7_particles = (int) Es_sc7.size();
  if (sc7_particles > 0) {
    int *mcsorted_sc7 = new int[sc7_particles];
    double *mce_sc7 = new double[sc7_particles];
    for (int i=0; i<sc7_particles; ++i)
      mce_sc7[i]=Es_sc7[i];
    //
    TMath::Sort(sc7_particles, mce_sc7, mcsorted_sc7, true);
    //
    for( int probeIt = 0; probeIt < sc7_particles; ++probeIt)
      {
	if (probeIt >= MAX_SC) break;
	//
	int elec_index = EsIndex_sc7[ mcsorted_sc7[probeIt] ];
	reco::PreshowerClusterCollection::const_iterator
	  Rprobe = sc7->begin() + elec_index;
	reco::PreshowerCluster probeEle;
	probeEle = *Rprobe;
	sc7_eta[probeIt] = probeEle.eta();
	sc7_phi[probeIt] = probeEle.phi();
	sc7_et[probeIt] = probeEle.energy()/cosh(probeEle.eta());
	//std::cout << "index=" << mcsorted_sc7[probeIt] << ", energy: " 
	//  << probeEle.energy() << std::endl;
      }
    //std::cout << "outside the loop" << std::endl;
    delete [] mcsorted_sc7;
    delete [] mce_sc7;
  }
  */
  //
  //
  //
  // end SC collections *****************************************************
  // *-*-*-*-*-*-*--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
  // GsF Electron Collection ---------------------------------------
  edm::Handle<pat::ElectronCollection> pElectrons;

  try{
    evt.getByLabel(ElectronCollection_, pElectrons);
  }
  catch (cms::Exception)
    {
      edm::LogError("")<< "Error! Can't get ElectronCollection by label. ";
    }
  // ***********************************************************************
  // check which trigger has accepted the event ****************************
  // ***********************************************************************
  //
  // path allocation: first 10 paths belong to the low lum menu, the rest
  // in the high lum one
  //
  // Low Luminosity Menu (8e29)
  //
  edm::Handle<edm::TriggerResults> HLTResultsE29;
  evt.getByLabel(HLTTriggerResultsE29_, HLTResultsE29);
  if (not HLTResultsE29.isValid()) {
    std::cout << "HLT Results with label: " << HLTTriggerResultsE29_ 
	      << " not found" << std::endl;
    return;
  }
  //
  edm::Handle<trigger::TriggerEvent> pHLTe29;
  evt.getByLabel(HLTCollectionE29_, pHLTe29);
  if (not pHLTe29.isValid()) {
    std::cout << "HLT Results with label: " << HLTCollectionE29_
	      << " not found" << std::endl;
    return;
  }
  //
  int sum = 0;
  //
  for (int iT=0; iT<10; ++iT) {
    event_HLTPath[iT] = 0;
    numberOfHLTFilterObjects[iT] =0;
    //
    edm::TriggerNames triggerNames;
    triggerNames.init(*HLTResultsE29);
    unsigned int trigger_size = HLTResultsE29->size();
    unsigned int trigger_position = triggerNames.triggerIndex(HLTPath_[iT]);
    if (trigger_position < trigger_size ) 
      event_HLTPath[iT] = (int) HLTResultsE29->accept(trigger_position);
    //
    numberOfHLTFilterObjects[iT] = 0;
    // check explicitly that the filter is there
    const int nF(pHLTe29->sizeFilters());
    const int filterInd = pHLTe29->filterIndex(HLTFilterType_[iT]);
    if (nF != filterInd) {
      const trigger::Vids& VIDS (pHLTe29->filterIds(filterInd));
      const trigger::Keys& KEYS(pHLTe29->filterKeys(filterInd));
      const int nI(VIDS.size());
      const int nK(KEYS.size());
      numberOfHLTFilterObjects[iT] = (nI>nK)? nI:nK;
    }
    //if (iT==2) // HLT_Ele15_LW_L1R only this trigger is required
      sum += numberOfHLTFilterObjects[iT];
  }
  //
  // High Luminosity Menu (1e31) DISABLED - only low lumi level
  //
  edm::Handle<edm::TriggerResults> HLTResultsE31;
  evt.getByLabel(HLTTriggerResultsE31_, HLTResultsE31);
  if (not HLTResultsE31.isValid()) {
      std::cout << "HLT Results with label: " << HLTTriggerResultsE31_ 
            << " not found" << std::endl;
    return;
  }
  ////
  edm::Handle<trigger::TriggerEvent> pHLTe31;
  evt.getByLabel(HLTCollectionE31_, pHLTe31);
  if (not pHLTe31.isValid()) {
    std::cout << "HLT Results with label: " << HLTCollectionE31_
  	      << " not found" << std::endl;
    return;
  }
  ////
  for (int iT=10; iT<25; ++iT) {
    event_HLTPath[iT] = 0;
    numberOfHLTFilterObjects[iT] =0;
    //
    edm::TriggerNames triggerNames;
    triggerNames.init(*HLTResultsE31);
    unsigned int trigger_size = HLTResultsE31->size();
    unsigned int trigger_position = triggerNames.triggerIndex(HLTPath_[iT]);
    if (trigger_position < trigger_size ) 
      event_HLTPath[iT] = (int) HLTResultsE31->accept(trigger_position);
    //
    numberOfHLTFilterObjects[iT] = 0;
    // check explicitly that the filter is there
    const int nF(pHLTe31->sizeFilters());
    const int filterInd = pHLTe31->filterIndex(HLTFilterType_[iT]);
    if (nF != filterInd) {
      const trigger::Vids& VIDS (pHLTe31->filterIds(filterInd));
      const trigger::Keys& KEYS(pHLTe31->filterKeys(filterInd));
      const int nI(VIDS.size());
      const int nK(KEYS.size());
      numberOfHLTFilterObjects[iT] = (nI>nK)? nI:nK;
    }
    // not needed
    sum += numberOfHLTFilterObjects[iT];
  }
  if (sum == 0) { 
    //std::cout << "No trigger found in this event..." << std::endl;
    return;
  }
  //std::cout << "HLT objects: #" << sum << std::endl;
  // *********************************************************************
  // MET Collections:
  //
  edm::Handle<reco::CaloMETCollection> caloMET;
  evt.getByLabel(MetCollectionTag_, caloMET);  
  //
  edm::Handle<pat::METCollection> t1MET;
  evt.getByLabel(t1MetCollectionTag_, t1MET);
  //
  edm::Handle<pat::METCollection> twikiT1MET;
  evt.getByLabel(t1MetCollectionTagTwiki_, twikiT1MET);
  //
  edm::Handle<reco::METCollection> tcMET;
  evt.getByLabel(tcMetCollectionTag_, tcMET);
  //
  edm::Handle<reco::PFMETCollection> pfMET;
  evt.getByLabel(pfMetCollectionTag_, pfMET);
  //
  edm::Handle<reco::GenMETCollection> genMET;
  evt.getByLabel(genMetCollectionTag_, genMET);
  //
  // initialize the MET variables ........................................
  event_MET     = -99.;   event_MET_phi = -99.;    event_MET_sig = -99.;
  event_tcMET   = -99.;   event_tcMET_phi = -99.;  event_tcMET_sig = -99.;
  event_pfMET   = -99.;   event_pfMET_phi = -99.;  event_pfMET_sig = -99.;
  event_t1MET   = -99.;   event_t1MET_phi = -99.;  event_t1MET_sig = -99.;
  event_twikiT1MET   = -99.;   event_twikiT1MET_phi = -99.;  event_twikiT1MET_sig = -99.;
  event_genMET  = -99.;   event_genMET_phi= -99.;  event_genMET_sig = -99.;
  //
  // get the values, if they are available
  if ( caloMET.isValid() ) {
    const reco::CaloMETRef MET(caloMET, 0);
    event_MET = MET->et();  event_MET_phi = MET->phi();
    event_MET_sig = MET->mEtSig();
  }
  else {
    std::cout << "caloMET not valid: input Tag: " << MetCollectionTag_
	      << std::endl;
  }
  if ( tcMET.isValid() ) {
    const reco::METRef MET(tcMET, 0);
    event_tcMET = MET->et();  event_tcMET_phi = MET->phi();
    event_tcMET_sig = MET->mEtSig();
  }
  if ( pfMET.isValid() ) {
    const reco::PFMETRef MET(pfMET, 0);
    event_pfMET = MET->et();  event_pfMET_phi = MET->phi();
    event_pfMET_sig = MET->mEtSig();
  }
  if ( t1MET.isValid() ) {
    const pat::METRef MET(t1MET, 0);
    event_t1MET = MET->et();  event_t1MET_phi = MET->phi();
    event_t1MET_sig = MET->mEtSig();
  }
  if ( twikiT1MET.isValid() ) {
    const pat::METRef MET(twikiT1MET, 0);
    event_twikiT1MET = MET->et();  event_twikiT1MET_phi = MET->phi();
    event_twikiT1MET_sig = MET->mEtSig();
  }

  if ( genMET.isValid() ) {
    const reco::GenMETRef MET(genMET, 0);
    event_genMET = MET->et();  event_genMET_phi = MET->phi();
    event_genMET_sig = MET->mEtSig();
  }

  //  std::cout << "t1MET: " << event_t1MET  << " twikiT1MET: " 
  //	    << event_twikiT1MET  << ", calo="<<event_MET  << std::endl;

  //
  /// -*-*-*-*-*--*-*-*-*-*-*-*-*-*-*-*-*--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*-*
  const int MAX_PROBES = 4;
  for(int i =0; i < MAX_PROBES; i++){
    probe_ele_eta_for_tree[i] = -99.0;
    probe_ele_et_for_tree[i] = -99.0;
    probe_ele_phi_for_tree[i] = -99.0;
    probe_ele_Xvertex_for_tree[i] = -99.0;
    probe_ele_Yvertex_for_tree[i] = -99.0;
    probe_ele_Zvertex_for_tree[i] = -99.0;
    probe_ele_tip[i] = -999.;    

    probe_sc_eta_for_tree[i] = -99.0;
    probe_sc_et_for_tree[i] = -99.0;
    probe_sc_phi_for_tree[i] = -99.0;
    
    probe_charge_for_tree[i] = -99;
    probe_sc_pass_fiducial_cut[i] = 0;
    probe_classification_index_for_tree[i]=-99; 
    //
    // probe isolation values ............
    probe_isolation_value[i] = 999.0;
    probe_iso_user[i] = 999.0;
    probe_ecal_isolation_value[i] = 999;
    probe_ecal_iso_user[i] = 999;
    probe_hcal_isolation_value[i] = 999;
    probe_hcal_iso_user[i] = 999;

    probe_ele_hoe[i]  = 999.;
    probe_ele_shh[i]  = 999.;
    probe_ele_sihih[i] = 999.;
    probe_ele_dhi[i]  = 999.;
    probe_ele_dfi[i]  = 999.;
    probe_ele_eop[i]  = 999.;
    probe_ele_pin[i]  = 999.;
    probe_ele_pout[i] = 999.;
    probe_ele_e5x5[i] = 999.;
    probe_ele_e2x5[i] = 999.;
    probe_ele_e1x5[i] = 999.;

    //
    //
    for (int j=0; j<25; ++j) {
      probe_pass_trigger_cut[i][j]=0;
    }
    //probe_hlt_matched_dr[i]=0;
    probe_mc_matched[i] = 0;
    probe_mc_matched_deta[i] = 999.;
    probe_mc_matched_dphi[i] = 999.;
    probe_mc_matched_denergy[i] = 999.;
    probe_mc_matched_mother[i] = 999;
    //
    //
  }
  const pat::ElectronCollection *electrons= pElectrons.product();
  

  elec_number_in_event = electrons->size();
  //  std::cout << "In this event " << elec_number_in_event << " were found" << std::endl;
  if (elec_number_in_event == 0) return;
 
  std::vector<pat::ElectronRef> UniqueElectrons;
  // edm::LogInfo("") << "Starting loop over electrons.";
  int index =0;
  //***********************************************************************
  // NEW METHOD by D WARDROPE implemented 26.05.08 ************************
  //************* DUPLICATE ******  REMOVAL *******************************
  // 02.06.08: due to a bug in the hybrid algorithm that affects detid ****
  //           we change detid matching to superCluster ref matching ******
  for(pat::ElectronCollection::const_iterator 
	elec = electrons->begin(); elec != electrons->end();++elec) {
    const pat::ElectronRef  electronRef(pElectrons, index);
    //Remove duplicate electrons which share a supercluster
    bool duplicate = false;
    pat::ElectronCollection::const_iterator BestDuplicate = elec;
    int index2 = 0;
    for(pat::ElectronCollection::const_iterator
	  elec2 = electrons->begin();
	elec2 != electrons->end(); ++elec2)
      {
	if(elec != elec2)
	  {
	    if( elec->superCluster() == elec2->superCluster())
	      {
		duplicate = true;
		if(fabs(BestDuplicate->eSuperClusterOverP()-1.)
		   >= fabs(elec2->eSuperClusterOverP()-1.))
		  {
		    BestDuplicate = elec2;
		  }
	      }
	  }
	++index2;
      }
    if(BestDuplicate == elec) UniqueElectrons.push_back(electronRef);
    ++index;
  }
  //
  // debugging: store electrons after duplicate removal
  elec_1_duplicate_removal = UniqueElectrons.size();
  //std::cout << "In this event there are " << elec_1_duplicate_removal 
  //   	    << " electrons" << std::endl;
  //
  //
  // duplicate removal is done now:
  //           the electron collection is in UniqueElectrons
  //
  // run over probes - now probe electrons and store
  //
  // the electron collection is now 
  // vector<reco::PixelMatchGsfElectronRef>   UniqueElectrons
  std::vector<double> ETs;
  std::vector<pat::ElectronRef>::const_iterator  elec;
  for (elec = UniqueElectrons.begin(); elec !=  UniqueElectrons.end(); ++elec) {
    pat::ElectronRef probeEle;
    probeEle = *elec;
    double probeEt = probeEle->caloEnergy()/(cosh(probeEle->caloPosition().eta()));
    ETs.push_back(probeEt);

  }
  int *sorted = new int[elec_1_duplicate_removal];
  double *et = new double[elec_1_duplicate_removal];
  //std::cout << "Elecs: " << elec_1_duplicate_removal << std::endl;
  for (int i=0; i<elec_1_duplicate_removal; ++i) {
    et[i] = ETs[i];
    //std::cout << "et["<< i << "]=" << et[i] << std::endl;
  }
  // array sorted now has the indices of the highest ET electrons
  TMath::Sort(elec_1_duplicate_removal, et, sorted, true);
  //
  //
  for( int probeIt = 0; probeIt < elec_1_duplicate_removal; ++probeIt)
    {
      //std::cout<<"sorted["<< probeIt<< "]=" << sorted[probeIt] << std::endl;
      // break if you have more than the appropriate number of electrons
      if (probeIt >= MAX_PROBES) break;
      //
      int elec_index = sorted[probeIt];
      std::vector<pat::ElectronRef>::const_iterator
	Rprobe = UniqueElectrons.begin() + elec_index;
      //
      pat::ElectronRef probeEle;
      probeEle = *Rprobe;
      double probeEt = probeEle->caloEnergy()/(cosh(probeEle->caloPosition().eta()));
      probe_sc_eta_for_tree[probeIt] = probeEle->caloPosition().eta();
      probe_sc_phi_for_tree[probeIt] = probeEle->caloPosition().phi();
      probe_sc_et_for_tree[probeIt] = probeEt;
      // fiducial cut ...............................
      if(fabs(probeEle->caloPosition().eta()) < BarrelMaxEta || 
	 (fabs(probeEle->caloPosition().eta()) > EndcapMinEta && 
	  fabs(probeEle->caloPosition().eta()) < EndcapMaxEta)){
	probe_sc_pass_fiducial_cut[probeIt] = 1;
      }
      //
      probe_charge_for_tree[probeIt] = probeEle->charge();
      probe_ele_eta_for_tree[probeIt] = probeEle->eta();
      probe_ele_et_for_tree[probeIt] = probeEle->et();
      probe_ele_phi_for_tree[probeIt] =probeEle->phi();
      probe_ele_Xvertex_for_tree[probeIt] =probeEle->vx();
      probe_ele_Yvertex_for_tree[probeIt] =probeEle->vy();
      probe_ele_Zvertex_for_tree[probeIt] =probeEle->vz();
      probe_classification_index_for_tree[probeIt] = 
	probeEle->classification();
      double ProbeTIP = probeEle->gsfTrack()->d0();
      probe_ele_tip[probeIt] = ProbeTIP;
      // isolation ..................................
      // these are the default values: trk 03, ecal, hcal 04
      // I know that there is a more direct way, but in this way it
      // is clearer what you get each time :P
      probe_isolation_value[probeIt] = probeEle->dr03IsolationVariables().tkSumPt;
      probe_ecal_isolation_value[probeIt] = probeEle->dr04IsolationVariables().ecalRecHitSumEt;
      probe_hcal_isolation_value[probeIt] = 
	probeEle->dr04IsolationVariables().hcalDepth1TowerSumEt + 
	probeEle->dr04IsolationVariables().hcalDepth2TowerSumEt;
      // one extra isos:
      probe_iso_user[probeIt] = probeEle->dr04IsolationVariables().tkSumPt;
      probe_ecal_iso_user[probeIt] = probeEle->dr03IsolationVariables().ecalRecHitSumEt;
      probe_hcal_iso_user[probeIt] = 
	probeEle->dr03IsolationVariables().hcalDepth1TowerSumEt + 
	probeEle->dr03IsolationVariables().hcalDepth2TowerSumEt;
      // ele id variables
      double hOverE = probeEle->hadronicOverEm();
      double deltaPhiIn = probeEle->deltaPhiSuperClusterTrackAtVtx();
      double deltaEtaIn = probeEle->deltaEtaSuperClusterTrackAtVtx();
      double eOverP = probeEle->eSuperClusterOverP();
      double pin  = probeEle->trackMomentumAtVtx().R(); 
      double pout = probeEle->trackMomentumOut().R(); 
      double sigmaee = probeEle->scSigmaEtaEta();
      double sigma_IetaIeta = probeEle->scSigmaIEtaIEta();
      // correct if in endcaps
      if( fabs (probeEle->caloPosition().eta()) > 1.479 )  {
	sigmaee = sigmaee - 0.02*(fabs(probeEle->caloPosition().eta()) -2.3);
      }
      //
      //double e5x5, e2x5Right, e2x5Left, e2x5Top, e2x5Bottom, e1x5;
      double e5x5, e2x5, e1x5;
      e5x5 = probeEle->scE5x5();
      e1x5 = probeEle->scE1x5();
      e2x5 = probeEle->scE2x5Max();
      //
      // electron ID variables
      probe_ele_hoe[probeIt] = hOverE;
      probe_ele_shh[probeIt] = sigmaee;
      probe_ele_sihih[probeIt] = sigma_IetaIeta;
      probe_ele_dfi[probeIt] = deltaPhiIn;
      probe_ele_dhi[probeIt] = deltaEtaIn;
      probe_ele_eop[probeIt] = eOverP;
      probe_ele_pin[probeIt] = pin;
      probe_ele_pout[probeIt] = pout;
      probe_ele_e5x5[probeIt] = e5x5;
      probe_ele_e2x5[probeIt] = e2x5;
      probe_ele_e1x5[probeIt] = e1x5;
 
      //
      // HLT filter ------------------------------------------------------
      //
      //
      // low luminosity filters
      for (int filterNum=0; filterNum<10; ++filterNum) {
	int trigger_int_probe = 0;
	
	//double hlt_matched_dr   = -1.;
	const int nF(pHLTe29->sizeFilters());
	//
	// default (tag) trigger filter
	//
	// find how many relevant
	const int iF = pHLTe29->filterIndex(HLTFilterType_[filterNum]);
	// loop over these objects to see whether they match
	const trigger::TriggerObjectCollection& TOC(pHLTe29->getObjects());
	if (nF != iF) {
	  // find how many objects there are
	  const trigger::Keys& KEYS(pHLTe29->filterKeys(iF));
	  const int nK(KEYS.size());
	  for (int iTrig = 0;iTrig <nK; ++iTrig ) {
	    const trigger::TriggerObject& TO(TOC[KEYS[iTrig]]);
	    //std::cout << "--> filter: "<< HLTFilterType_[filterNum]  <<" TO id: " << TO.id() << std::endl;
	    // this is better to be left out: HLT matching is with an HLT object
	    // and we don't care what this object is
	    //if (abs(TO.id())==11 ) { // demand it to be an electron
	    double dr_ele_HLT = 
	      reco::deltaR(probeEle->eta(), probeEle->phi(), TO.eta(), TO.phi());
	    if (fabs(dr_ele_HLT) < ProbeHLTObjMaxDR) {++trigger_int_probe;
	    //hlt_matched_dr = dr_ele_HLT;
	    }
	    //}
	  }
	}
	//
	if(trigger_int_probe>0) probe_pass_trigger_cut[probeIt][filterNum] = 1;
	//probe_hlt_matched_dr[probeIt] = hlt_matched_dr;
      }
      // high lumi filters
      for (int filterNum=10; filterNum<25; ++filterNum) {
      	int trigger_int_probe = 0;
      	
      	//double hlt_matched_dr   = -1.;
      	const int nF(pHLTe31->sizeFilters());
      	//
      	// default (tag) trigger filter
      	//
      	// find how many relevant
      	const int iF = pHLTe31->filterIndex(HLTFilterType_[filterNum]);
      	// loop over these objects to see whether they match
      	const trigger::TriggerObjectCollection& TOC(pHLTe31->getObjects());
	if (nF != iF) {
	  // find how many objects there are
	  const trigger::Keys& KEYS(pHLTe31->filterKeys(iF));
	  const int nK(KEYS.size());
	  for (int iTrig = 0;iTrig <nK; ++iTrig ) {
	    const trigger::TriggerObject& TO(TOC[KEYS[iTrig]]);
	    //if (abs(TO.id())==11 ) { // demand it to be an electron
	    double dr_ele_HLT = 
	      reco::deltaR(probeEle->eta(), probeEle->phi(), TO.eta(), TO.phi());
	    if (fabs(dr_ele_HLT) < ProbeHLTObjMaxDR) {++trigger_int_probe;
	    //hlt_matched_dr = dr_ele_HLT;
	    }
	  }
	}
      
	//
	if(trigger_int_probe>0) probe_pass_trigger_cut[probeIt][filterNum] = 1;
	//probe_hlt_matched_dr[probeIt] = hlt_matched_dr;
      }
      // ------------------------------------------------------------------
      //
      // MC Matching ......................................................
      // check whether these electrons are matched to a MC electron
      
      int mc_index = 0;
      int matched = 0; int mother_id = 999;
      double deta_matched = 999.;      double dphi_matched = 999.;
      double denergy_matched = 999.;
      for(reco::GenParticleCollection::const_iterator   McParticle = 
	    McCand->begin(); McParticle != McCand->end();  ++McParticle)
	{
	  // check only for electrons
	  if(abs(McParticle->pdgId())==11 && McParticle->status()==1) {
	    mc_index++;
	    // check whether it matches a gsf electron
	    double deta = McParticle->eta() - probeEle->eta();
	    double dphi = McParticle->phi() - probeEle->phi();
	    if ( fabs(deta) < MCMatch_Deta_  && fabs(dphi) < MCMatch_Dphi_){
	      ++matched;
	      deta_matched = deta; dphi_matched = dphi;
	      denergy_matched = McParticle->energy() - probeEle->caloEnergy();
	      // find the mother of the MC electron
	      const reco::Candidate *mum;
	      bool mother_finder = true;
	      if (abs(McParticle->mother()->pdgId()) != 11)
		mum = McParticle->mother();
	      else if (abs(McParticle->mother()->mother()->pdgId())!= 11)
		mum = McParticle->mother()->mother();
	      else {
		edm::LogInfo("info") << "Going too far to find the mum";
		mother_finder = false;
	      }		 
	      if (mother_finder) {
		mother_id = mum->pdgId();
	      }
	    }
	  }
	}
      probe_mc_matched[probeIt] = matched;
      probe_mc_matched_deta[probeIt] = deta_matched;
      probe_mc_matched_dphi[probeIt] = dphi_matched;
      probe_mc_matched_denergy[probeIt] = denergy_matched;
      probe_mc_matched_mother[probeIt] = mother_id;
      
    }
  
  probe_tree->Fill();
  ++ tree_fills_;
  delete []  sorted;
  delete []  et;
}


// ------------ method called once each job just before starting event loop  --
void 
GenPurposeSkimmerAcceptance::beginJob(const edm::EventSetup&)
{
  //std::cout << "In beginJob()" << std::endl;
  TString filename_histo = outputFile_;
  histofile = new TFile(filename_histo,"RECREATE");
  tree_fills_ = 0;

  probe_tree =  new TTree("probe_tree","Tree to store probe variables");

  //probe_tree->Branch("probe_ele_eta",probe_ele_eta_for_tree,"probe_ele_eta[4]/D");
  //probe_tree->Branch("probe_ele_phi",probe_ele_phi_for_tree,"probe_ele_phi[4]/D");
  //probe_tree->Branch("probe_ele_et",probe_ele_et_for_tree,"probe_ele_et[4]/D");
  probe_tree->Branch("probe_ele_tip",probe_ele_tip,"probe_ele_tip[4]/D");
  probe_tree->Branch("probe_ele_vertex_x",probe_ele_Xvertex_for_tree,
		     "probe_ele_vertex_x[4]/D");
  probe_tree->Branch("probe_ele_vertex_y",probe_ele_Yvertex_for_tree,
		     "probe_ele_vertex_y[4]/D");
  probe_tree->Branch("probe_ele_vertex_z",probe_ele_Zvertex_for_tree,
		     "probe_ele_vertex_z[4]/D");
  probe_tree->Branch("probe_sc_eta",probe_sc_eta_for_tree,"probe_sc_eta[4]/D");
  probe_tree->Branch("probe_sc_phi",probe_sc_phi_for_tree,"probe_sc_phi[4]/D");
  probe_tree->Branch("probe_sc_et",probe_sc_et_for_tree,"probe_sc_et[4]/D");

  // trigger related variables
  probe_tree->Branch("probe_trigger_cut",probe_pass_trigger_cut,"probe_trigger_cut[4][25]/I");
  //probe_tree->Branch("probe_hlt_matched_dr", probe_hlt_matched_dr,"probe_hlt_matched_dr[4]/D");
  // mc matching to electrons
  probe_tree->Branch("probe_mc_matched",probe_mc_matched,"probe_mc_matched[4]/I");
  //probe_tree->Branch("probe_mc_matched_deta",probe_mc_matched_deta,
  //	     "probe_mc_matched_deta[4]/D");
  //probe_tree->Branch("probe_mc_matched_dphi",probe_mc_matched_dphi,
  //		     "probe_mc_matched_dphi[4]/D");
  //probe_tree->Branch("probe_mc_matched_denergy",probe_mc_matched_denergy,
  //		     "probe_mc_matched_denergy[4]/D");
  probe_tree->Branch("probe_mc_matched_mother",probe_mc_matched_mother,
  		     "probe_mc_matched_mother[4]/I");
  //
  probe_tree->Branch("probe_charge",probe_charge_for_tree,"probe_charge[4]/I");
  //probe_tree->Branch("probe_sc_fiducial_cut",probe_sc_pass_fiducial_cut,
  //		     "probe_sc_fiducial_cut[4]/I");



  //probe_tree->Branch("probe_classification",
  //	    probe_classification_index_for_tree,"probe_classification[4]/I");
  //
  // Isolation related variables ........................................
  //
  probe_tree->Branch("probe_isolation_value",probe_isolation_value, "probe_isolation_value[4]/D");
  probe_tree->Branch("probe_ecal_isolation_value",probe_ecal_isolation_value, "probe_ecal_isolation_value[4]/D");
  probe_tree->Branch("probe_hcal_isolation_value",probe_hcal_isolation_value,"probe_hcal_isolation_value[4]/D");
  //
  probe_tree->Branch("probe_iso_user",     probe_iso_user,      "probe_iso_user[4]/D");
  probe_tree->Branch("probe_ecal_iso_user",probe_ecal_iso_user, "probe_ecal_iso_user[4]/D");
  probe_tree->Branch("probe_hcal_iso_user",probe_hcal_iso_user, "probe_hcal_iso_user[4]/D");

  //......................................................................
  // Electron ID Related variables .......................................
  probe_tree->Branch("probe_ele_hoe",probe_ele_hoe, "probe_ele_hoe[4]/D");
  //probe_tree->Branch("probe_ele_shh",probe_ele_shh, "probe_ele_shh[4]/D");
  probe_tree->Branch("probe_ele_sihih",probe_ele_sihih,"probe_ele_sihih[4]/D");
  probe_tree->Branch("probe_ele_dfi",probe_ele_dfi, "probe_ele_dfi[4]/D");
  probe_tree->Branch("probe_ele_dhi",probe_ele_dhi, "probe_ele_dhi[4]/D");
  probe_tree->Branch("probe_ele_eop",probe_ele_eop, "probe_ele_eop[4]/D");
  probe_tree->Branch("probe_ele_pin",probe_ele_pin, "probe_ele_pin[4]/D");
  probe_tree->Branch("probe_ele_pout",probe_ele_pout, "probe_ele_pout[4]/D");
  // probe_tree->Branch("probe_ele_e5x5",probe_ele_e5x5, "probe_ele_e5x5[4]/D");
  //probe_tree->Branch("probe_ele_e2x5",probe_ele_e2x5, "probe_ele_e2x5[4]/D");
  //probe_tree->Branch("probe_ele_e1x5",probe_ele_e1x5, "probe_ele_e1x5[4]/D");

  //.......................................................................
  //
  // each entry for each trigger path
  probe_tree->Branch("event_HLTPath",event_HLTPath,"event_HLTPath[25]/I");
  probe_tree->Branch("numberOfHLTFilterObjects", numberOfHLTFilterObjects,
		     "numberOfHLTFilterObjects[25]/I");
  //
  // debugging info:
  //probe_tree->Branch("elec_number_in_event",&elec_number_in_event,"elec_number_in_event/I");
  probe_tree->Branch("elec_1_duplicate_removal",&elec_1_duplicate_removal,"elec_1_duplicate_removal/I");
  //

  // Missing ET in the event
  probe_tree->Branch("event_MET",&event_MET,"event_MET/D");
  probe_tree->Branch("event_MET_phi",&event_MET_phi,"event_MET_phi/D");
  //  probe_tree->Branch("event_MET_sig",&event_MET_sig,"event_MET_sig/D");

  probe_tree->Branch("event_tcMET",&event_tcMET,"event_tcMET/D");
  probe_tree->Branch("event_tcMET_phi",&event_tcMET_phi,"event_tcMET_phi/D");
  //  probe_tree->Branch("event_tcMET_sig",&event_tcMET_sig,"event_tcMET_sig/D");

  probe_tree->Branch("event_pfMET",&event_pfMET,"event_pfMET/D");
  probe_tree->Branch("event_pfMET_phi",&event_pfMET_phi,"event_pfMET_phi/D");
  //  probe_tree->Branch("event_pfMET_sig",&event_pfMET_sig,"event_pfMET_sig/D");

  probe_tree->Branch("event_genMET",&event_genMET,"event_genMET/D");
  probe_tree->Branch("event_genMET_phi",&event_genMET_phi, "event_genMET_phi/D");
  //  probe_tree->Branch("event_genMET_sig",&event_genMET_sig, "event_genMET_sig/D");
  //..... type 1 corrected MET
  probe_tree->Branch("event_t1MET", &event_t1MET, "event_t1MET/D");
  probe_tree->Branch("event_t1MET_phi", &event_t1MET_phi,
		     "event_t1MET_phi/D");
  //probe_tree->Branch("event_t1MET_sig",&event_t1MET_sig,"event_t1MET_sig/D");
  //
  probe_tree->Branch("event_twikiT1MET", &event_twikiT1MET, "event_twikiT1MET/D");
  probe_tree->Branch("event_twikiT1MET_phi", &event_twikiT1MET_phi,
		     "event_twikiT1MET_phi/D");

  // extras for MC particles
  probe_tree->Branch("mc_ele_eta", mc_ele_eta, "mc_ele_eta[10]/D");
  probe_tree->Branch("mc_ele_phi", mc_ele_phi, "mc_ele_phi[10]/D");
  probe_tree->Branch("mc_ele_et", mc_ele_et, "mc_ele_et[10]/D");
  probe_tree->Branch("mc_ele_vertex_x",mc_ele_vertex_x,"mc_ele_vertex_x[10]/D");
  probe_tree->Branch("mc_ele_vertex_y",mc_ele_vertex_y,"mc_ele_vertex_y[10]/D");
  probe_tree->Branch("mc_ele_vertex_z",mc_ele_vertex_z,"mc_ele_vertex_z[10]/D");
  probe_tree->Branch("mc_ele_mother", mc_ele_mother, "mc_ele_mother[10]/I");
  probe_tree->Branch("mc_ele_charge", mc_ele_charge, "mc_ele_charge[10]/I");
  probe_tree->Branch("mc_ele_status", mc_ele_status, "mc_ele_status[10]/I");

  // SC information
  //probe_tree->Branch("sc0_eta", sc0_eta, "sc0_eta[8]/D");
  //probe_tree->Branch("sc0_phi", sc0_phi, "sc0_phi[8]/D");
  //probe_tree->Branch("sc0_et", sc0_et, "sc0_et[8]/D");
  //
  probe_tree->Branch("sc1_eta", sc1_eta, "sc1_eta[8]/D");
  probe_tree->Branch("sc1_phi", sc1_phi, "sc1_phi[8]/D");
  probe_tree->Branch("sc1_et", sc1_et, "sc1_et[8]/D");
  //
  probe_tree->Branch("sc2_eta", sc2_eta, "sc2_eta[8]/D");
  probe_tree->Branch("sc2_phi", sc2_phi, "sc2_phi[8]/D");
  probe_tree->Branch("sc2_et", sc2_et, "sc2_et[8]/D");
  //
  probe_tree->Branch("sc3_eta", sc3_eta, "sc3_eta[8]/D");
  probe_tree->Branch("sc3_phi", sc3_phi, "sc3_phi[8]/D");
  probe_tree->Branch("sc3_et", sc3_et, "sc3_et[8]/D");
  //
  probe_tree->Branch("sc4_eta", sc4_eta, "sc4_eta[8]/D");
  probe_tree->Branch("sc4_phi", sc4_phi, "sc4_phi[8]/D");
  probe_tree->Branch("sc4_et", sc4_et, "sc4_et[8]/D");
  //
  probe_tree->Branch("sc5_eta", sc5_eta, "sc5_eta[8]/D");
  probe_tree->Branch("sc5_phi", sc5_phi, "sc5_phi[8]/D");
  probe_tree->Branch("sc5_et", sc5_et, "sc5_et[8]/D");
  //
  //probe_tree->Branch("sc6_eta", sc6_eta, "sc6_eta[8]/D");
  //probe_tree->Branch("sc6_phi", sc6_phi, "sc6_phi[8]/D");
  //probe_tree->Branch("sc6_et", sc6_et, "sc6_et[8]/D");
  //
  //probe_tree->Branch("sc7_eta", sc7_eta, "sc7_eta[8]/D");
  //probe_tree->Branch("sc7_phi", sc7_phi, "sc7_phi[8]/D");
  //probe_tree->Branch("sc7_et", sc7_et, "sc7_et[8]/D");


}

// ------------ method called once each job just after ending the event loop  -
void 
GenPurposeSkimmerAcceptance::endJob() {
  //std::cout << "In endJob()" << std::endl;
  if (tree_fills_ == 0) {
    std::cout << "Empty tree: no output..." << std::endl;
    return;
  }
  //probe_tree->Print();
  histofile->Write();
  histofile->Close();

}


//define this as a plug-in
DEFINE_FWK_MODULE(GenPurposeSkimmerAcceptance);
