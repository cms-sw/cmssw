// -*- C++ -*-
//
// Package:    testEcalClusterSeverityAlgo
// Class:      testEcalClusterSeverityAlgo
// 
/**\class testEcalClusterSeverityAlgo testEcalClusterSeverityAlgo.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  "Paolo Meridiani CERN CMG"
// $Id: testEcalClusterSeverityAlgo.cc,v 1.6 2011/11/12 11:05:33 sani Exp $



// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// to access recHits and BasicClusters
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"



// to use the cluster tools
#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterSeverityLevelAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"

#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

#include "CLHEP/Units/PhysicalConstants.h"

#include "TTree.h"
#include "TFile.h"

class testEcalClusterSeverityAlgo : public edm::EDAnalyzer 
{
  
public:
  
  explicit testEcalClusterSeverityAlgo(const edm::ParameterSet&);
  ~testEcalClusterSeverityAlgo();
  
  edm::InputTag barrelClusterCollection_;
  edm::InputTag endcapClusterCollection_;
  edm::InputTag reducedBarrelRecHitCollection_;
  edm::InputTag reducedEndcapRecHitCollection_;
  edm::InputTag mcTruthCollection_;
  
  struct ClusterSeverityTreeContent
  {
#define NMAXOBJ 50
    //MC Variables     
    int nSc;
    float scE[NMAXOBJ];
    float scEta[NMAXOBJ];
    float scPhi[NMAXOBJ];
    float scGoodFraction[NMAXOBJ];
    float scFracAroundClosProb[NMAXOBJ];
    float scClosProbEta[NMAXOBJ];
    float scClosProbPhi[NMAXOBJ];
    float mcE[NMAXOBJ];
    float mcEta[NMAXOBJ];
    float mcPhi[NMAXOBJ];
  };

  static void setBranchAddresses(TTree* chain, ClusterSeverityTreeContent& treeVars)    
  {
    chain -> SetBranchAddress("nSc",         &treeVars.nSc);
    chain -> SetBranchAddress("scE", treeVars.scE);
    chain -> SetBranchAddress("scEta", treeVars.scEta);
    chain -> SetBranchAddress("scPhi", treeVars.scPhi);
    chain -> SetBranchAddress("scGoodFraction", treeVars.scGoodFraction);
    chain -> SetBranchAddress("scFracAroundClosProb", treeVars.scFracAroundClosProb);
    chain -> SetBranchAddress("scClosProbEta", treeVars.scClosProbEta);
    chain -> SetBranchAddress("scClosProbPhi", treeVars.scClosProbPhi);
    chain -> SetBranchAddress("mcE", treeVars.mcE);
    chain -> SetBranchAddress("mcEta", treeVars.mcEta);
    chain -> SetBranchAddress("mcPhi", treeVars.mcPhi);
  }

  static void setBranches(TTree* chain, ClusterSeverityTreeContent& treeVars)
  {
    chain -> Branch("nSc",         &treeVars.nSc , "nSc/I");
    chain -> Branch("scE", treeVars.scE, "scE[nSc]/F");
    chain -> Branch("scEta", treeVars.scEta, "scEta[nSc]/F");
    chain -> Branch("scPhi", treeVars.scPhi, "scPhi[nSc]/F");
    chain -> Branch("scGoodFraction", treeVars.scGoodFraction, "scGoodFraction[nSc]/F");
    chain -> Branch("scFracAroundClosProb", treeVars.scFracAroundClosProb, "scFracAroundClosProb[nSc]/F");
    chain -> Branch("scClosProbEta", treeVars.scClosProbEta, "scClosProbEta[nSc]/F");
    chain -> Branch("scClosProbPhi", treeVars.scClosProbPhi, "scClosProbPhi[nSc]/F");
    chain -> Branch("mcE", treeVars.mcE, "mcE[nSc]/F");
    chain -> Branch("mcEta", treeVars.mcEta, "mcEta[nSc]/F");
    chain -> Branch("mcPhi", treeVars.mcPhi, "mcPhi[nSc]/F");
  }

  void initializeBranches(TTree* chain, ClusterSeverityTreeContent& treeVars)
  {
    
    treeVars.nSc = 0;
    for(int i = 0; i < NMAXOBJ; ++i)
      {
	treeVars.scE[i]=0;
	treeVars.scEta[i]=0;
	treeVars.scPhi[i]=0;
	treeVars.scGoodFraction[i]=0;
	treeVars.scFracAroundClosProb[i]=0;
	treeVars.scClosProbEta[i]=0;
	treeVars.scClosProbPhi[i]=0;
	treeVars.mcE[i]=0;
	treeVars.mcEta[i]=0;
	treeVars.mcPhi[i]=0;
      }
  }

private:

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  std::string outputFile_;
  TFile *treeFile_;
  TTree *tree_;
  ClusterSeverityTreeContent myTreeVariables_;

};



testEcalClusterSeverityAlgo::testEcalClusterSeverityAlgo(const edm::ParameterSet& ps)
{
        barrelClusterCollection_ = ps.getParameter<edm::InputTag>("barrelClusterCollection");
        endcapClusterCollection_ = ps.getParameter<edm::InputTag>("endcapClusterCollection");
        reducedBarrelRecHitCollection_ = ps.getParameter<edm::InputTag>("reducedBarrelRecHitCollection");
        reducedEndcapRecHitCollection_ = ps.getParameter<edm::InputTag>("reducedEndcapRecHitCollection");
	mcTruthCollection_ = ps.getParameter<edm::InputTag>("mcTruthCollection");
        outputFile_ = ps.getParameter<std::string>("outputFile");
	treeFile_ = new TFile(outputFile_.c_str(),"RECREATE");
	treeFile_->cd();
	// Initialize Tree
	tree_ = new TTree ( "ClusterSeverityAnalysis","ClusterSeverityAnalysis" ) ;
	setBranches (tree_, myTreeVariables_) ;
        
}



testEcalClusterSeverityAlgo::~testEcalClusterSeverityAlgo()
{
}



void testEcalClusterSeverityAlgo::analyze(const edm::Event& ev, const edm::EventSetup& es)
{
  initializeBranches(tree_, myTreeVariables_);

  edm::Handle<edm::HepMCProduct> hepMC;
  ev.getByLabel(mcTruthCollection_,hepMC);
  const HepMC::GenEvent *myGenEvent = hepMC->GetEvent();

  edm::ESHandle<EcalSeverityLevelAlgo> sevlvh;  
  es.get<EcalSeverityLevelAlgoRcd>().get(sevlvh);  
  const EcalSeverityLevelAlgo * sevLv= sevlvh.product();

  edm::Handle< reco::SuperClusterCollection > pEBClusters;
        ev.getByLabel( barrelClusterCollection_, pEBClusters );
        const reco::SuperClusterCollection *ebClusters = pEBClusters.product();

        edm::Handle< reco::SuperClusterCollection > pEEClusters;
        ev.getByLabel( endcapClusterCollection_, pEEClusters );
        //const reco::SuperClusterCollection *eeClusters = pEEClusters.product();

        edm::Handle< EcalRecHitCollection > pEBRecHits;
        ev.getByLabel( reducedBarrelRecHitCollection_, pEBRecHits );
        const EcalRecHitCollection *ebRecHits = pEBRecHits.product();

        //edm::Handle< EcalRecHitCollection > pEERecHits;
        //ev.getByLabel( reducedEndcapRecHitCollection_, pEERecHits );
        //const EcalRecHitCollection *eeRecHits = pEERecHits.product();

        //edm::ESHandle<CaloGeometry> pGeometry;
        //es.get<CaloGeometryRecord>().get(pGeometry);
        //const CaloGeometry *geometry = pGeometry.product();

        edm::ESHandle<CaloTopology> pTopology;
        es.get<CaloTopologyRecord>().get(pTopology);
        const CaloTopology *topology = pTopology.product();

	
	//        std::cout << "========== BARREL ==========" << std::endl;
	int problematicSC=0;
        for (reco::SuperClusterCollection::const_iterator it = ebClusters->begin(); it != ebClusters->end(); ++it ) 
	  {
	    //apply an et Cut
	    if ((*it).energy()/cosh((*it).eta())<15.)
	      continue;
	    //  	    if ( fabs(EcalClusterSeverityLevelAlgo::goodFraction( *it, *ebRecHits, *theEcalChStatus) - 1. ) < 1e-6 && EcalClusterSeverityLevelAlgo::closestProblematic( *it, *ebRecHits, *theEcalChStatus,topology).null() ) 
	    //  	      continue;
	    // 	    std::cout << "seed" << EBDetId(EcalClusterTools::getMaximum(*it, ebRecHits).first) << std::endl;
	    // 	    std::cout << "goodFraction" << EcalClusterSeverityLevelAlgo::goodFraction( *it, *ebRecHits, *theEcalChStatus) << std::endl;
	    // 	    std::cout << "closestProblematicDetId" << EBDetId(EcalClusterSeverityLevelAlgo::closestProblematic( *it, *ebRecHits, *theEcalChStatus,topology)) << std::endl;
	    // 	    std::cout << "(deta,dphi)" << "(" << EcalClusterSeverityLevelAlgo::etaphiDistanceClosestProblematic( *it, *ebRecHits, *theEcalChStatus,topology).first << "," <<EcalClusterSeverityLevelAlgo::etaphiDistanceClosestProblematic( *it, *ebRecHits, *theEcalChStatus,topology).second << ")" << std::endl;
	    
	    HepMC::GenParticle* bestMcMatch=0;
	    for ( HepMC::GenEvent::particle_const_iterator mcIter=myGenEvent->particles_begin(); mcIter != myGenEvent->particles_end(); mcIter++ ) {
	      
	      // select electrons
	      if ( abs((*mcIter)->pdg_id()) == 11 )
		{
		
		  // single primary electrons or electrons from Zs or Ws
		  HepMC::GenParticle* mother = 0;
		  if ( (*mcIter)->production_vertex() )  {
		    if ( (*mcIter)->production_vertex()->particles_begin(HepMC::parents) !=
			 (*mcIter)->production_vertex()->particles_end(HepMC::parents))
		      mother = *((*mcIter)->production_vertex()->particles_begin(HepMC::parents));
		  }
		  if ( (
			(mother == 0) || 
			((mother != 0) && (mother->pdg_id() == 23)) ||
			((mother != 0) && (mother->pdg_id() == 32)) ||
			((mother != 0) && (fabs(mother->pdg_id()) == 24))
			)
		       ) 
		    {
		    
		    HepMC::GenParticle* genPc=(*mcIter);
		    HepMC::FourVector pAssSim = genPc->momentum();


		    //bool okGsfFound = false;
		    double ScOkRatio = 999999.;
		    
		    
		    
		    
		    double dphi = (*it).phi()-pAssSim.phi();
		    if (fabs(dphi)>CLHEP::pi)
		      dphi = dphi < 0? (CLHEP::twopi) + dphi : dphi - CLHEP::twopi;
		    double deltaR = sqrt(pow(((*it).eta()-pAssSim.eta()),2) + pow(dphi,2));
		    if ( deltaR < 0.15 )
		      {
			
			double tmpScRatio = (*it).energy()/pAssSim.t();
			if ( fabs(tmpScRatio-1) < fabs(ScOkRatio-1) ) {
			  ScOkRatio = tmpScRatio;
			  bestMcMatch=genPc;
			  //okGsfFound = true;
			}
		      }
		    
		  }
		}
	    }
	  	  
	    if (!bestMcMatch)
	      continue;
	    
	    myTreeVariables_.scE[problematicSC]=(*it).energy();
	    myTreeVariables_.scEta[problematicSC]=(*it).eta();
	    myTreeVariables_.scPhi[problematicSC]=(*it).phi();
	    myTreeVariables_.scGoodFraction[problematicSC]=EcalClusterSeverityLevelAlgo::goodFraction( *it, *ebRecHits, *sevLv);
	    myTreeVariables_.scFracAroundClosProb[problematicSC]=EcalClusterSeverityLevelAlgo::fractionAroundClosestProblematic( *it, *ebRecHits,topology,*sevLv);
	    std::pair<int,int> distanceClosestProblematic=EcalClusterSeverityLevelAlgo::etaphiDistanceClosestProblematic( *it, *ebRecHits,topology,*sevLv);
	    myTreeVariables_.scClosProbEta[problematicSC]=distanceClosestProblematic.first;
	    myTreeVariables_.scClosProbPhi[problematicSC]=distanceClosestProblematic.second;
	    myTreeVariables_.mcE[problematicSC]=bestMcMatch->momentum().t();
	    myTreeVariables_.mcEta[problematicSC]=bestMcMatch->momentum().eta();
	    myTreeVariables_.mcPhi[problematicSC]=bestMcMatch->momentum().phi();
	    ++problematicSC;
	  }
	myTreeVariables_.nSc=problematicSC;
	tree_ -> Fill();
}


void testEcalClusterSeverityAlgo::endJob() {
  treeFile_->cd();
  tree_->Write () ;
  treeFile_->Close();
}

//define this as a plug-in
DEFINE_FWK_MODULE(testEcalClusterSeverityAlgo);
