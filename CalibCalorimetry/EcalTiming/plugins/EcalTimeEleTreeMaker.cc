// -*- C++ -*-
//
// Package:   EcalTimeEleTreeMaker
// Class:     EcalTimeEleTreeMaker
//
/**\class EcalTimeEleTreeMaker EcalTimeEleTreeMaker.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
 */
//
// Skeleton Derived from an example by:  
// Authors:                              Giovanni Franzoni (UMN)
//         Created:  Mo Apr 18 5:46:22 CEST 2008
// $Id: EcalTimeEleTreeMaker.cc,v 1.11 2011/11/22 17:07:35 franzoni Exp $
//
// Maintained by Tambe E. Norbert (UMN) 2012-2014
//
//



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// NOTE it's assumed that the events fed into this Ntuple producer have passed the W/Z DPG skim selections specified here: 
// http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/DPGAnalysis/Skims/python/WZinterestingEventFilter_cfi.py?revision=1.1&view=markup
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <algorithm>

#include "CalibCalorimetry/EcalTiming/plugins/EcalTimeEleTreeMaker.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalTools.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"

// vertex stuff
#include <DataFormats/VertexReco/interface/VertexFwd.h>
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"


#include <vector>
#include <iostream>
#include <iterator>

using namespace cms ;
using namespace edm ;
using namespace std ;

#define FILL_SHAPE_VARS 0

 


EcalTimeEleTreeMaker::EcalTimeEleTreeMaker (const edm::ParameterSet& iConfig) :
  barrelEcalRecHitCollection_              (iConfig.getParameter<edm::InputTag> ("barrelEcalRecHitCollection")),
  endcapEcalRecHitCollection_              (iConfig.getParameter<edm::InputTag> ("endcapEcalRecHitCollection")),
  barrelBasicClusterCollection_            (iConfig.getParameter<edm::InputTag> ("barrelBasicClusterCollection")),
  endcapBasicClusterCollection_            (iConfig.getParameter<edm::InputTag> ("endcapBasicClusterCollection")),
  barrelSuperClusterCollection_            (iConfig.getParameter<edm::InputTag> ("barrelSuperClusterCollection")),
  endcapSuperClusterCollection_            (iConfig.getParameter<edm::InputTag> ("endcapSuperClusterCollection")),
  patElectrons_                            (iConfig.getParameter<edm::InputTag> ("patElectrons")),
  vertexCollection_                        (iConfig.getParameter<edm::InputTag> ("vertexCollection")),
  l1GMTReadoutRecTag_                      (iConfig.getParameter<std::string> ("l1GlobalReadoutRecord")),
  gtRecordCollectionTag_                   (iConfig.getParameter<std::string> ("GTRecordCollection")),
  runNum_                                  (iConfig.getParameter<int> ("runNum")),
  eleIdCuts_                               (iConfig.getParameter< std::vector<int> > ("eleIdCuts")),
  elePtCut_                                (iConfig.getParameter<double> ("elePtCut")),
  scHighEtaEEPtCut_                        (iConfig.getParameter<double> ("scHighEtaEEPtCut")),
  fileName_                                (iConfig.getParameter<std::string> ("OutfileName")),
  workingPoint_                            (iConfig.getParameter<std::string> ("eleWorkingPoint")),
  naiveId_ (0)              
{
  // TrackAssociator parameters // gfwork: can we remove this? 
  edm::ParameterSet trkParameters = iConfig.getParameter<edm::ParameterSet> ("TrackAssociatorParameters") ;
  trackParameters_.loadParameters ( trkParameters ) ;
  //trackAssociator_.useDefaultPropagator () ;

  // Create File
  fileName_ += "_"+intToString (runNum_)+".root" ;
  file_ = new TFile (fileName_.c_str () , "RECREATE") ;
  file_->cd () ;

  // Initialize Tree
  tree_ = new TTree ( "EcalTimeAnalysis","EcalTimeAnalysis" ) ;

  setBranches (tree_, myTreeVariables_) ;

}


// -----------------------------------------------------------------------------------------


EcalTimeEleTreeMaker::~EcalTimeEleTreeMaker ()
{
}


// -----------------------------------------------------------------------------------------


void EcalTimeEleTreeMaker::analyze (const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  ++naiveId_ ;
    

  // Geometry
  edm::ESHandle<CaloGeometry> pGeometry ;
  iSetup.get<CaloGeometryRecord> ().get (pGeometry) ;
  const CaloGeometry * theGeometry = pGeometry.product () ;
  // Topology
  edm::ESHandle<CaloTopology> pCaloTopology ;
  iSetup.get<CaloTopologyRecord> ().get (pCaloTopology) ;
  const CaloTopology * theCaloTopology = pCaloTopology.product () ;

  // Ecal barrel RecHits 
  edm::Handle<EcalRecHitCollection> pBarrelEcalRecHits ;
  const EcalRecHitCollection* theBarrelEcalRecHits = 0;
  if( iEvent.getByLabel (barrelEcalRecHitCollection_, pBarrelEcalRecHits) && pBarrelEcalRecHits.isValid ())
    {
      theBarrelEcalRecHits = pBarrelEcalRecHits.product () ;   
    }
  if (! (pBarrelEcalRecHits.isValid ()) )
    {
      LogWarning ("EcalTimeEleTreeMaker") << barrelEcalRecHitCollection_ 
				     << " not available" ;
      return ;
    }
  
  
  // Ecal endcap RecHits
  edm::Handle<EcalRecHitCollection> pEndcapEcalRecHits ;
  const EcalRecHitCollection* theEndcapEcalRecHits = 0;
  if( iEvent.getByLabel (endcapEcalRecHitCollection_, pEndcapEcalRecHits) && pEndcapEcalRecHits.isValid ())
    {
      theEndcapEcalRecHits = pEndcapEcalRecHits.product () ;   
    }
  
  if (! (pEndcapEcalRecHits.isValid ()))
    {
      LogWarning ("EcalTimeEleTreeMaker") << endcapEcalRecHitCollection_ 
                                     << " not available" ;
      return ;
    }
  

  // Endcap SuperClusters
  edm::Handle<reco::SuperClusterCollection> pEndcapSuperClusters ;
  iEvent.getByLabel (endcapSuperClusterCollection_, pEndcapSuperClusters) ;
  const reco::SuperClusterCollection* theEndcapSuperClusters = pEndcapSuperClusters.product () ;
    if (! (pEndcapSuperClusters.isValid ()) )
    {
      LogWarning ("EcalTimeTreeMaker") << endcapSuperClusterCollection_ 
                                     << " not available" ;
      return ;
    }


  // ClusterShapes
  EcalClusterLazyTools* lazyTools = new EcalClusterLazyTools(iEvent, iSetup, barrelEcalRecHitCollection_, endcapEcalRecHitCollection_);


  // Xtal - TkLength map
  std::map<int,float> XtalMap ;
  std::map<int,float> XtalMapCurved ;
  std::map<int,float> XtalMapCurved_high ;
  std::map<int,float> XtalMapCurved_low ;

  // GFdoc initialize variables to 0/false
  initializeBranches(tree_, myTreeVariables_);

  myTreeVariables_.bx          = iEvent.bunchCrossing();
  myTreeVariables_.lumiSection = iEvent.id().luminosityBlock();
  myTreeVariables_.unixTime    = iEvent.eventAuxiliary().time().unixTime();
  myTreeVariables_.orbit       = iEvent.orbitNumber();
  
  myTreeVariables_.runId         = iEvent.id ().run () ;
  myTreeVariables_.eventId       = iEvent.id ().event () ;
  myTreeVariables_.eventNaiveId  = naiveId_ ;
  myTreeVariables_.timeStampLow  = ( 0xFFFFFFFF & iEvent.time ().value () ) ;
  myTreeVariables_.timeStampHigh = ( iEvent.time ().value () >> 32 ) ;

  /**Get Collection of GSFElectrons if using **/
  edm::Handle<reco::GsfElectronCollection> gsfElectrons;
  iEvent.getByLabel("gsfElectrons", gsfElectrons);
  if (!gsfElectrons.isValid()) {
    std::cerr << "Error! can't get the product ElectronCollection: " << std::endl;
  } else{
    // std::cerr << "BRAVO! GoT the product ElectronCollection. " << std::endl;
  }


  // get collection of pat electrons
  edm::Handle<pat::ElectronCollection> patElectrons;
  iEvent.getByLabel(patElectrons_, patElectrons);
  if ( ! patElectrons.isValid()) {      
    std::cout << "[EcalTimeEleTreeMaker] No electrons found in this event with tag "  << patElectrons << std::endl;     return ; }
  else {      if( 0 ) std::cout  << "[EcalTimeEleTreeMaker] got electrons collection with tag: " << patElectrons_ << "\t"<< patElectrons << std::endl; }
  const pat::ElectronCollection& patElecs = *(patElectrons.product());
  


  // RECO vertices
  Handle<reco::VertexCollection> recVtxs;
  iEvent.getByLabel(vertexCollection_, recVtxs);
  const reco::VertexCollection * theRecVtxs = recVtxs.product();


  // GFdoc GT information 
  dump3Ginfo(iEvent, iSetup, myTreeVariables_) ;
  
  dumpBarrelClusterInfo(iEvent,theGeometry, theCaloTopology,
			patElecs,
			theBarrelEcalRecHits, 
			lazyTools, XtalMap, XtalMapCurved, myTreeVariables_) ;
  dumpEndcapClusterInfo(iEvent,theGeometry, theCaloTopology,
			patElecs,
			theEndcapSuperClusters,
			theEndcapEcalRecHits, 
			lazyTools, 
			XtalMap, XtalMapCurved, myTreeVariables_) ;
  
  dumpVertexInfo(theRecVtxs, myTreeVariables_);

  tree_ -> Fill();
  delete lazyTools;
}






// -----------------------------------------------------------------------------------------

void EcalTimeEleTreeMaker::endJob ()
{
  using namespace std ;
  file_->cd () ;
  tree_->Write () ;
  file_->Close () ;

}


// -----------------------------------------------------------------------------------------

void EcalTimeEleTreeMaker::beginRun(edm::Run const &, edm::EventSetup const & eventSetup)
{
  // IC's
  eventSetup.get<EcalIntercalibConstantsRcd>().get(ical);
  // ADCtoGeV
  eventSetup.get<EcalADCToGeVConstantRcd>().get(agc);
  // transp corrections
  eventSetup.get<EcalLaserDbRecord>().get(laser);
}



// -----------------------------------------------------------------------------------------

std::string EcalTimeEleTreeMaker::intToString (int num)
{
    using namespace std ;
    ostringstream myStream ;
    myStream << num << flush ;
    return (myStream.str ()) ; //returns the string form of the stringstream object
}






// -----------------------------------------------------------------------------------------

void EcalTimeEleTreeMaker::dumpBarrelClusterInfo (const edm::Event& iEvent,
						  const CaloGeometry * theGeometry,
						  const CaloTopology * theCaloTopology,
						  const pat::ElectronCollection& patElecs,
						  const EcalRecHitCollection* theBarrelEcalRecHits,
						  EcalClusterLazyTools* lazyTools,
						  const std::map<int,float> & XtalMap, //GFdoc unclear
						  const std::map<int,float> & XtalMapCurved, //GFdoc unclear
						  EcalTimeTreeContent & myTreeVariables_)
{            
  // get number of of objects already present in the tree (none if dumpBarrelClusterInfo is called first)
  int numberOfSuperClusters = myTreeVariables_.nSuperClusters;
  int numberOfClusters      = myTreeVariables_.nClusters ;

  const EcalIntercalibConstantMap& icalMap = ical->getMap();
  float adcToGeV = float(agc->getEBValue());

  // loop over the pat electrons
  pat::ElectronCollection::const_iterator eleIt;
  for (eleIt=patElecs.begin(); eleIt!=patElecs.end() && numberOfSuperClusters<MAXSC; eleIt++) {
    
    reco::SuperClusterRef sclusRef=eleIt->superCluster();
    if (sclusRef.isNull()) {        std::cout  << "[EcalTimeEleTreeMaker] reference to superlcuster is NULL: we have a problem" << std::endl;        continue; }
    
    reco::SuperClusterRef sclus=eleIt->superCluster();
    
    double eta_det=sclus.get()->eta();
    if( 0 ) std::cout  << "[EcalTimeEleTreeMaker] BEF got ONE pat electron with SC-pt: " 
    		       << eleIt->pt() << " eta: " << eta_det << " and with ID : " <<  eleIt->electronID( workingPoint_.c_str()  ) 
		       << std::endl; 

    // remove electrons which are too soft
    if (eleIt->pt() < elePtCut_) continue;
    
    // remove electrons that don't pass the desired selections
    vector<int>::iterator itEleId = find ( eleIdCuts_.begin(), eleIdCuts_.end(), eleIt->electronID( workingPoint_.c_str()) ) ;
    if( itEleId == eleIdCuts_.end()) continue;

    if( 0 ) std::cout  << "[EcalTimeEleTreeMaker] AFT got ONE pat electron with SC-pt: " 
    		       << eleIt->pt() << " eta: " << eta_det << " and with ID : " <<  eleIt->electronID( workingPoint_.c_str()  ) 
		       << "\n"<< std::endl; 
        
        
    myTreeVariables_.nSuperClusters       +=1;
    myTreeVariables_.nBarrelSuperClusters +=1;
    
    if(!eleIt->isEB()) {
      continue;
    } else {
    }
    
      myTreeVariables_.nClustersInSuperCluster[numberOfSuperClusters] = sclus.get()-> clustersSize () ;
      myTreeVariables_.superClusterEta[numberOfSuperClusters] = sclus.get()-> position ().eta () ;
      myTreeVariables_.superClusterPhi[numberOfSuperClusters] = sclus.get()-> position ().phi () ;
      myTreeVariables_.superClusterX[numberOfSuperClusters] = sclus.get()-> position ().x () ;
      myTreeVariables_.superClusterY[numberOfSuperClusters] = sclus.get()-> position ().y () ;
      myTreeVariables_.superClusterZ[numberOfSuperClusters] = sclus.get()-> position ().z () ;
      myTreeVariables_.superClusterVertexX[numberOfSuperClusters] = eleIt->trackPositionAtVtx().x() ;
      myTreeVariables_.superClusterVertexY[numberOfSuperClusters] = eleIt->trackPositionAtVtx().y() ;
      myTreeVariables_.superClusterVertexZ[numberOfSuperClusters] = eleIt->trackPositionAtVtx().z() ;
      myTreeVariables_.superClusterRawEnergy[numberOfSuperClusters] = sclus.get()-> rawEnergy () ;
      myTreeVariables_.superClusterPhiWidth[numberOfSuperClusters] = sclus.get()-> phiWidth () ;
      myTreeVariables_.superClusterEtaWidth[numberOfSuperClusters] = sclus.get()-> etaWidth () ;

      numberOfSuperClusters++ ;
     
      
      ///////////////////////////////////////////////////////////////////////////////////////
      // loop on barrel basic clusters which are inside SC from an electron
      for ( reco::CaloCluster_iterator  bClus =  sclus.get()->clustersBegin()  ; 
	    bClus != sclus.get()->clustersEnd()  && numberOfClusters<MAXC;  
	    ++bClus) // loop on barrel Bclusters
	{        
	  // std::cout << "bc number: " << (bClus-sclus.get()->clustersBegin()) << std::endl;
	  
	  double energy = (*bClus)->energy () ;
	  double phi    = (*bClus)->phi () ;
	  double eta    = (*bClus)->eta () ;
	  double sinTheta         = fabs( sin( 2 *atan( exp(-1*(*bClus)->eta()) ) ) );
	  double transverseEnergy = (*bClus)->energy () * sinTheta;
	  double time = -1000.0 ; // gfdoc: work on this to provide a combination of crystals?
	  double ampli = 0. ;
	  double secondMin = 0. ;
	  double secondTime = -1000. ;
	  int numberOfXtalsInCluster=0 ;//counter for all xtals in cluster
	  int numXtalsinCluster = 0 ;   //xtals in cluster above 3sigma noise  
	  EBDetId maxDet ;
	  EBDetId secDet ;
	  
	  std::vector<std::pair<DetId, float> > clusterDetIds = (*bClus)->hitsAndFractions() ; //get these from the cluster
	  
	  //myTreeVariables_.xtalIndexInCluster[numberOfClusters] = numberOfXtals ;
	  
	  //////////////////////////////////////////////////////////////////////////////////////
	  //loop on xtals in cluster
	  for (std::vector<std::pair<DetId, float> >::const_iterator detitr = clusterDetIds.begin () ; 
	       detitr != clusterDetIds.end ()  && numberOfXtalsInCluster<MAXXTALINC;// && numberOfXtals<MAXXTAL ; 
	       ++detitr)// loop on rechics of barrel basic clusters
	    {
	      //Here I use the "find" on a digi collection... I have been warned...   (GFdoc: ??)
	      // GFdoc: check if DetId belongs to ECAL; if so, find it among those if this basic cluster
	      if ( (detitr -> first).det () != DetId::Ecal) 
		{ 
		  std::cout << " det is " << (detitr -> first).det () << " (and not DetId::Ecal)" << std::endl ;
		  continue ;
		}
	      if ( (detitr -> first).subdetId () != EcalBarrel) 
		{
		  std::cout << " subdet is " << (detitr -> first).subdetId () << " (and not EcalBarrel)" << std::endl ; 
		  continue ; 
		}
	      
	      // GFdoc now find it!
	      EcalRecHitCollection::const_iterator thishit = theBarrelEcalRecHits->find ( (detitr -> first) ) ;
	      if (thishit == theBarrelEcalRecHits->end ()) 
		{
		  continue ;
		}
	      //The checking above should no longer be needed...... 
	      //as only those in the cluster would already have rechits..
	      
	      // GFdoc this is one crystal in the basic cluster
	      EcalRecHit myhit = (*thishit) ;
	      
	      // SIC Feb 14 2011 -- Add check on RecHit flags (takes care of spike cleaning in 42X)
	      if( !( myhit.checkFlag(EcalRecHit::kGood) ||
		     myhit.checkFlag(EcalRecHit::kOutOfTime) ||
		     myhit.checkFlag(EcalRecHit::kPoorCalib) 
		     ) 
		  )
		continue;
	      
	      
	      // thisamp is the EB amplitude of the current rechit
	      double thisamp  = myhit.energy () ;
	      double thisChi2 = myhit.chi2 ();
	      double thisOutOfTimeChi2 = myhit.outOfTimeChi2 ();
	      
	      EcalIntercalibConstantMap::const_iterator icalit = icalMap.find(detitr->first);
	      
	      EcalIntercalibConstant icalconst = 1;
	      if( icalit!=icalMap.end() ) {
		icalconst = (*icalit);
	      } else {
		edm::LogError("EcalTimeEleTreeMaker") << "No intercalib const found for xtal "
						      << (detitr->first).rawId();
	      }
	      // get laser coefficient
	      float lasercalib = 1.;
	      lasercalib = laser->getLaserCorrection( detitr->first, iEvent.time());
	      //std::cout << "GF EE debug: " << lasercalib << std::endl;
	      // discard rechits with A/sigma < 12
	      if ( thisamp/(icalconst*lasercalib*adcToGeV) < (1.1*12) ) continue;
	      
	      if (thisamp > 0.027) //cut on energy->number of crystals in cluster above 3sigma noise; gf: desirable?
		{ 
		  numXtalsinCluster++ ; 
		}
	      if (thisamp > secondMin) 
		{
		  secondMin = thisamp ; 
		  secondTime = myhit.time () ; 
		  secDet = (EBDetId) (detitr -> first) ;
		}
	      if (secondMin > ampli) 
		{
		  std::swap (ampli, secondMin) ; 
		  std::swap (time, secondTime) ; 
		  std::swap (maxDet, secDet) ;
		}

      
	      if(myhit.isTimeErrorValid())
		myTreeVariables_.xtalInBCTimeErr[numberOfClusters][numberOfXtalsInCluster]= myhit.timeError();
	      else
		myTreeVariables_.xtalInBCTimeErr[numberOfClusters][numberOfXtalsInCluster]= -999999;
	      
	      // xtal variables inside a barrel basic cluster 
	      myTreeVariables_.xtalInBCEnergy[numberOfClusters][numberOfXtalsInCluster]=       (float) thisamp;
	      myTreeVariables_.xtalInBCTime[numberOfClusters][numberOfXtalsInCluster]=         (float) (myhit.time ()); 
	      myTreeVariables_.xtalInBCHashedIndex[numberOfClusters][numberOfXtalsInCluster]=  EBDetId (detitr -> first).hashedIndex () ; 
	      myTreeVariables_.xtalInBCIEta[numberOfClusters][numberOfXtalsInCluster]=         EBDetId((detitr -> first)).ieta();
	      myTreeVariables_.xtalInBCIPhi[numberOfClusters][numberOfXtalsInCluster]=         EBDetId((detitr -> first)).iphi();
	      myTreeVariables_.xtalInBCIx[numberOfClusters][numberOfXtalsInCluster]=           -999999; 
	      myTreeVariables_.xtalInBCIy[numberOfClusters][numberOfXtalsInCluster]=           -999999; 
	      myTreeVariables_.xtalInBCFlag[numberOfClusters][numberOfXtalsInCluster]=         myhit.recoFlag(); 
	      myTreeVariables_.xtalInBCAmplitudeADC[numberOfClusters][numberOfXtalsInCluster]= (float) thisamp/(icalconst*lasercalib*adcToGeV);
	      myTreeVariables_.xtalInBCChi2[numberOfClusters][numberOfXtalsInCluster]=         thisChi2;
	      myTreeVariables_.xtalInBCOutOfTimeChi2[numberOfClusters][numberOfXtalsInCluster]=thisOutOfTimeChi2;
	      // note: SwissCross = 1 - E4/E1   
	      myTreeVariables_.xtalInBCSwissCross[numberOfClusters][numberOfXtalsInCluster] = EcalTools::swissCross(detitr->first,*theBarrelEcalRecHits,0.5);
	      
	      GlobalPoint pos = theGeometry->getPosition((myhit).detid());
	      myTreeVariables_.xtalInBCEta[numberOfClusters][numberOfXtalsInCluster]=      pos.eta();
	      myTreeVariables_.xtalInBCPhi[numberOfClusters][numberOfXtalsInCluster]=      pos.phi();

	      numberOfXtalsInCluster++ ;
	      
	    } //end loop on rechits within barrel basic clusters
	  //////////////////////////////////////////////////////
	  
	  float E2 = (float)ampli + (float)secondMin ;
	  
	  //Set some more values for this cluster
	  myTreeVariables_.clusterEnergy[numberOfClusters] = (float) energy ;
	  myTreeVariables_.clusterTransverseEnergy[numberOfClusters] = (float) transverseEnergy ;
	  myTreeVariables_.clusterTime[numberOfClusters] = (float) time ;
	  myTreeVariables_.clusterEta[numberOfClusters] = (float) eta ;
	  myTreeVariables_.clusterPhi[numberOfClusters] = (float) phi ;
	  myTreeVariables_.clusterE1[numberOfClusters] = (float) ampli ;
	  myTreeVariables_.clusterE2[numberOfClusters] = (float) E2 ;
	  myTreeVariables_.clusterXtalsAbove3Sigma[numberOfClusters] = numXtalsinCluster ;
	  myTreeVariables_.clusterXtals[numberOfClusters] = clusterDetIds.size () ;
	  myTreeVariables_.clusterMaxId[numberOfClusters] =  maxDet.rawId () ;
	  myTreeVariables_.clusterMaxId[numberOfClusters] =  secDet.rawId () ;
	  myTreeVariables_.nXtalsInCluster[numberOfClusters]= numberOfXtalsInCluster ;    
	  
	  // (basic) cluster shapes for barrel
	  if(FILL_SHAPE_VARS) {
	    myTreeVariables_.clusterE2x2[numberOfClusters] = lazyTools -> e2x2(*(*bClus));
	    myTreeVariables_.clusterE3x2[numberOfClusters] = lazyTools -> e3x2(*(*bClus));
	    myTreeVariables_.clusterE3x3[numberOfClusters] = lazyTools -> e3x3(*(*bClus));
	    myTreeVariables_.clusterE4x4[numberOfClusters] = lazyTools -> e4x4(*(*bClus));
	    myTreeVariables_.clusterE5x5[numberOfClusters] = lazyTools -> e5x5(*(*bClus));
	    
	    myTreeVariables_.clusterE2x5Right[numberOfClusters] = lazyTools -> e2x5Right(*(*bClus));
	    myTreeVariables_.clusterE2x5Left[numberOfClusters] = lazyTools -> e2x5Left(*(*bClus));
	    myTreeVariables_.clusterE2x5Top[numberOfClusters] = lazyTools -> e2x5Top(*(*bClus));
	    myTreeVariables_.clusterE2x5Bottom[numberOfClusters] = lazyTools -> e2x5Bottom(*(*bClus));
	    
	    myTreeVariables_.clusterCovEtaEta[numberOfClusters] = (lazyTools -> covariances(*(*bClus)))[0];
	    myTreeVariables_.clusterCovPhiPhi[numberOfClusters] = (lazyTools -> covariances(*(*bClus)))[2];
	    myTreeVariables_.clusterCovEtaPhi[numberOfClusters] = (lazyTools -> covariances(*(*bClus)))[1];
	    
	    myTreeVariables_.clusterLat[numberOfClusters] = (lazyTools -> lat(*(*bClus)))[2];
	    myTreeVariables_.clusterEtaLat[numberOfClusters] = (lazyTools -> lat(*(*bClus)))[0];
	    myTreeVariables_.clusterPhiLat[numberOfClusters] = (lazyTools -> lat(*(*bClus)))[1];
	    
	    myTreeVariables_.clusterZernike20[numberOfClusters] = lazyTools -> zernike20(*(*bClus));
	    myTreeVariables_.clusterZernike42[numberOfClusters] = lazyTools -> zernike42(*(*bClus));
	  }
	  
	  numberOfClusters++ ;
	  
	} // loop on BC inside SC
    }// loop over electrons-SC
  
  //number of clusters / xtals for the whole event
  myTreeVariables_.nClusters = numberOfClusters ;
  
  return ;
} // end dumpBarrelClusterInfo  


// -----------------------------------------------------------------------------------------

void EcalTimeEleTreeMaker::dumpEndcapClusterInfo (const edm::Event& iEvent,
						  const CaloGeometry * theGeometry,
						  const CaloTopology * theCaloTopology,
						  const pat::ElectronCollection& patElecs,
						  const reco::SuperClusterCollection* theEndcapSuperClusters,
						  const EcalRecHitCollection* theEndcapEcalRecHits,
						  EcalClusterLazyTools* lazyTools,
						  const std::map<int,float> & XtalMap,
						  const std::map<int,float> & XtalMapCurved,
						  EcalTimeTreeContent & myTreeVariables_)
{
  // counters come from the ntuple are to account for what was added in dumpBarrelClusterInf
  int numberOfSuperClusters = myTreeVariables_.nSuperClusters;
  int numberOfClusters      = myTreeVariables_.nClusters;

  const EcalIntercalibConstantMap& icalMap = ical->getMap();
  float adcToGeV = float(agc->getEEValue());


  //loop on all endcap superclusters in event
  for (reco::SuperClusterCollection::const_iterator sclus = theEndcapSuperClusters->begin () ; 
       sclus != theEndcapSuperClusters->end ()  && numberOfSuperClusters<MAXSC; 
       ++sclus) 
    {//loop on SC's
      
      bool acceptScInHighEta(false);
      //////////////////////////////////////////////////////////////////////////////////////////////////
      // if supercluster in the forward region of EE and softer than scHighEtaEEPtCut, don't consider it 
      if( fabs(sclus->position().eta()) > 2.5   &&   ( sclus->energy()/cosh(sclus->position().eta()) ) > scHighEtaEEPtCut_ ) {
	acceptScInHighEta=true;
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////
      // look for matching electron which fulfills requirements
      bool foundMatcgingElectron(false);
      pat::ElectronCollection::const_iterator eleIt;
      pat::ElectronCollection::const_iterator It;
      for (It=patElecs.begin(); It!=patElecs.end(); It++) 
	{ 
	  // remove electrons which are too soft
	  if (It->pt() < elePtCut_)     continue;

	  // remove electrons that don't pass the desired selections
	  vector<int>::iterator itEleId = find ( eleIdCuts_.begin(), eleIdCuts_.end(), It->electronID( workingPoint_.c_str()) ) ;
	  if( itEleId == eleIdCuts_.end()) continue;
	  //std::cout <<  "position difference is: " << sqrt( ( sclus->position() - It->superCluster()->position() ).Mag2() )   << std::endl;
	  // do matching by position, 1 cm
	  if ( 	      sqrt( ( sclus->position() - It->superCluster()->position() ).Mag2() ) < 1  	      ) 
	    {
	      foundMatcgingElectron = true;
	      myTreeVariables_.superClusterVertexX[numberOfSuperClusters] = It->trackPositionAtVtx().x() ;
	      myTreeVariables_.superClusterVertexY[numberOfSuperClusters] = It->trackPositionAtVtx().y() ;
	      myTreeVariables_.superClusterVertexZ[numberOfSuperClusters] = It->trackPositionAtVtx().z() ;
	      break;
	    }
	}// end loop over electrons
      
      if( (!foundMatcgingElectron) && (!acceptScInHighEta)) continue;
      

      myTreeVariables_.nSuperClusters       +=1;
      myTreeVariables_.nEndcapSuperClusters +=1;
      
      myTreeVariables_.nClustersInSuperCluster[numberOfSuperClusters] = sclus-> clustersSize () ;
      
      myTreeVariables_.superClusterEta[numberOfSuperClusters] = sclus-> position ().eta () ;
      myTreeVariables_.superClusterPhi[numberOfSuperClusters] = sclus-> position ().phi () ;
      myTreeVariables_.superClusterX[numberOfSuperClusters] = sclus-> position ().x () ;
      myTreeVariables_.superClusterY[numberOfSuperClusters] = sclus-> position ().y () ;
      myTreeVariables_.superClusterZ[numberOfSuperClusters] = sclus-> position ().z () ;
      myTreeVariables_.superClusterRawEnergy[numberOfSuperClusters] = sclus-> energy () ;
      myTreeVariables_.superClusterPhiWidth[numberOfSuperClusters] = sclus-> phiWidth () ;
      myTreeVariables_.superClusterEtaWidth[numberOfSuperClusters] = sclus-> etaWidth () ;

      numberOfSuperClusters++ ;
  
      ///////////////////////////////////////////////////////////////////////////////////////
      // loop on barrel basic clusters which are inside SC from an electron
      for ( reco::CaloCluster_iterator  bClus =  sclus->clustersBegin()  ; 
	    bClus != sclus->clustersEnd()  && numberOfClusters<MAXC;  
	    ++bClus) // loop on barrel Bclusters
	{        
         double energy = (*bClus)->energy () ;
         double phi    = (*bClus)->phi () ;
         double eta    = (*bClus)->eta () ;
	 double sinTheta         = fabs( sin( 2 *atan( exp(-1*(*bClus)->eta()) ) ) );
	 double transverseEnergy = (*bClus)->energy () * sinTheta;
         double time = -1000.0 ;  // gfdoc: work on this to provide a combination of crystals?
         double ampli = 0. ;
         double secondMin = 0. ;
         double secondTime = -1000. ;
         int numberOfXtalsInCluster=0 ;//counter for all xtals in cluster
         int numXtalsinCluster = 0 ;//xtals in cluster above 3sigma noise  
         EEDetId maxDet ;
         EEDetId secDet ;
         
         std::vector<std::pair<DetId, float> > clusterDetIds = (*bClus)->hitsAndFractions() ; //get these from the cluster

        //loop on xtals in cluster
         for (std::vector<std::pair<DetId, float> >::const_iterator detitr = clusterDetIds.begin () ; 
              detitr != clusterDetIds.end () && numberOfXtalsInCluster<MAXXTALINC;// && numberOfXtals<MAXXTAL ; 
              ++detitr)// loop on rechits of endcap basic clusters
           {
             if ( (detitr -> first).det () != DetId::Ecal) 
               {                   continue ;                }
             if ( (detitr -> first).subdetId () != EcalEndcap) 
               {                  continue ;                 }
             EcalRecHitCollection::const_iterator thishit = theEndcapEcalRecHits->find ( (detitr -> first) ) ;
             if (thishit == theEndcapEcalRecHits->end ()) 
               {                  continue ;                }
             //The checking above should no longer be needed...... 
             //as only those in the cluster would already have rechits..
             
             EcalRecHit myhit = (*thishit) ;
             
             // SIC Feb 14 2011 -- Add check on RecHit flags (takes care of spike cleaning in 42X)
             if( !( myhit.checkFlag(EcalRecHit::kGood) ||
                    myhit.checkFlag(EcalRecHit::kOutOfTime) ||
                    myhit.checkFlag(EcalRecHit::kPoorCalib) 
		    ) 
		 )
                 continue;


	     // thisamp is the EE amplitude of the current rechit
	     double thisamp  = myhit.energy () ;
	     double thistime = myhit.time ();
	     double thisChi2 = myhit.chi2 ();
	     double thisOutOfTimeChi2 = myhit.outOfTimeChi2 ();

    
             EcalIntercalibConstantMap::const_iterator icalit = icalMap.find(detitr->first);
             EcalIntercalibConstant icalconst = 1;
             if( icalit!=icalMap.end() ) {
               icalconst = (*icalit);
             } else {
               edm::LogError("EcalTimeEleTreeMaker") << "No intercalib const found for xtal "
                 << (detitr->first).rawId();
             }
	     // get laser coefficient
	     float lasercalib = 1.;
	     lasercalib = laser->getLaserCorrection( detitr->first, iEvent.time());
	     //std::cout << "GF EB debug: " << lasercalib << std::endl;
	     // don't store rechits with A/sigma < 12
	     if ( thisamp/(icalconst*lasercalib*adcToGeV) < (2.2*12) ) continue;

             if (thisamp > 0.027) //cut on energy->number of crystals in cluster above 3sigma noise
               { 
                 numXtalsinCluster++ ; //xtals in cluster above 3sigma noise  
               }
	     
	     // GFdoc: 'second' keeps track of current cry if it has larger energy than previous
             if (thisamp > secondMin) 
               {
                 secondMin = thisamp ; 
                 secondTime = myhit.time () ; 
                 secDet = (EEDetId) (detitr -> first) ;
               }
             if (secondMin > ampli) 
               {
                 std::swap (ampli, secondMin) ; 
                 std::swap (time, secondTime) ; 
                 std::swap (maxDet, secDet) ;
               }


             if(myhit.isTimeErrorValid())
               myTreeVariables_.xtalInBCTimeErr[numberOfClusters][numberOfXtalsInCluster]= myhit.timeError();
             else
               myTreeVariables_.xtalInBCTimeErr[numberOfClusters][numberOfXtalsInCluster]= -999999;

	     // xtal variables inside an endcap basic cluster 
	      myTreeVariables_.xtalInBCEnergy[numberOfClusters][numberOfXtalsInCluster]=      (float) thisamp;
	      myTreeVariables_.xtalInBCTime[numberOfClusters][numberOfXtalsInCluster]=        (float) thistime;
	      myTreeVariables_.xtalInBCHashedIndex[numberOfClusters][numberOfXtalsInCluster]= EEDetId (detitr -> first).hashedIndex () ; 
	      myTreeVariables_.xtalInBCIEta[numberOfClusters][numberOfXtalsInCluster]=-999999;
	      myTreeVariables_.xtalInBCIPhi[numberOfClusters][numberOfXtalsInCluster]=-999999;
	      myTreeVariables_.xtalInBCIx[numberOfClusters][numberOfXtalsInCluster]=          EEDetId((detitr -> first)).ix();
	      myTreeVariables_.xtalInBCIy[numberOfClusters][numberOfXtalsInCluster]=          EEDetId((detitr -> first)).iy();
	      myTreeVariables_.xtalInBCFlag[numberOfClusters][numberOfXtalsInCluster]=         myhit.recoFlag(); 
	      myTreeVariables_.xtalInBCAmplitudeADC[numberOfClusters][numberOfXtalsInCluster]=      (float) thisamp/(icalconst*lasercalib*adcToGeV);
              myTreeVariables_.xtalInBCChi2[numberOfClusters][numberOfXtalsInCluster]=         thisChi2;
              myTreeVariables_.xtalInBCOutOfTimeChi2[numberOfClusters][numberOfXtalsInCluster]=thisOutOfTimeChi2;
              myTreeVariables_.xtalInBCSwissCross[numberOfClusters][numberOfXtalsInCluster] =
                EcalTools::swissCross(detitr->first,*theEndcapEcalRecHits,0.5);

	      GlobalPoint pos = theGeometry->getPosition((myhit).detid());
	      myTreeVariables_.xtalInBCEta[numberOfClusters][numberOfXtalsInCluster]=      pos.eta();
	      myTreeVariables_.xtalInBCPhi[numberOfClusters][numberOfXtalsInCluster]=      pos.phi();

             numberOfXtalsInCluster++ ;
     
           } //end loop on rechics within endcap basic clusters
         //////////////////////////////////////////////////////
	 
         float E2 = (float)ampli + (float)secondMin ;
         
         //Set some more values for this cluster
         myTreeVariables_.clusterEnergy[numberOfClusters] = (float) energy ;
         myTreeVariables_.clusterTransverseEnergy[numberOfClusters] = (float) transverseEnergy ;
         myTreeVariables_.clusterTime[numberOfClusters] = (float) time ;
         myTreeVariables_.clusterEta[numberOfClusters] = (float) eta ;
         myTreeVariables_.clusterPhi[numberOfClusters] = (float) phi ;
         myTreeVariables_.clusterE1[numberOfClusters] = (float) ampli ;
         myTreeVariables_.clusterE2[numberOfClusters] = (float) E2 ;
         myTreeVariables_.clusterXtalsAbove3Sigma[numberOfClusters] = numXtalsinCluster ;
         myTreeVariables_.clusterXtals[numberOfClusters] = clusterDetIds.size () ;
         myTreeVariables_.clusterMaxId[numberOfClusters] =  maxDet.rawId () ;
         myTreeVariables_.clusterMaxId[numberOfClusters] =  secDet.rawId () ;
         myTreeVariables_.nXtalsInCluster[numberOfClusters]=numberOfXtalsInCluster ;    


	 // (basic) cluster shapes for endcap
	 if(FILL_SHAPE_VARS) {
	   myTreeVariables_.clusterE2x2[numberOfClusters] = lazyTools -> e2x2(*(*bClus));
	   //myTreeVariables_.clusterE2x2[numberOfClusters] = lazyTools -> e2x2(*(*bClus));
	   myTreeVariables_.clusterE3x2[numberOfClusters] = lazyTools -> e3x2(*(*bClus));
	   myTreeVariables_.clusterE3x3[numberOfClusters] = lazyTools -> e3x3(*(*bClus));
	   myTreeVariables_.clusterE4x4[numberOfClusters] = lazyTools -> e4x4(*(*bClus));
	   myTreeVariables_.clusterE5x5[numberOfClusters] = lazyTools -> e5x5(*(*bClus));
	   
	   myTreeVariables_.clusterE2x5Right[numberOfClusters] = lazyTools -> e2x5Right(*(*bClus));
	   myTreeVariables_.clusterE2x5Left[numberOfClusters] = lazyTools -> e2x5Left(*(*bClus));
	   myTreeVariables_.clusterE2x5Top[numberOfClusters] = lazyTools -> e2x5Top(*(*bClus));
	   myTreeVariables_.clusterE2x5Bottom[numberOfClusters] = lazyTools -> e2x5Bottom(*(*bClus));
	   
	   myTreeVariables_.clusterCovEtaEta[numberOfClusters] = (lazyTools -> covariances(*(*bClus)))[0];
	   myTreeVariables_.clusterCovPhiPhi[numberOfClusters] = (lazyTools -> covariances(*(*bClus)))[2];
	   myTreeVariables_.clusterCovEtaPhi[numberOfClusters] = (lazyTools -> covariances(*(*bClus)))[1];
	   
	   myTreeVariables_.clusterLat[numberOfClusters] = (lazyTools -> lat(*(*bClus)))[2];
	   myTreeVariables_.clusterEtaLat[numberOfClusters] = (lazyTools -> lat(*(*bClus)))[0];
	   myTreeVariables_.clusterPhiLat[numberOfClusters] = (lazyTools -> lat(*(*bClus)))[1];
	   
	   myTreeVariables_.clusterZernike20[numberOfClusters] = lazyTools -> zernike20(*(*bClus));
	   myTreeVariables_.clusterZernike42[numberOfClusters] = lazyTools -> zernike42(*(*bClus));
	 }

         numberOfClusters++ ;

	}//end endcap basic cluster loop belonging to a SC
    }// end loop over electrons-SC

  //number of clusters / xtals for the whole event
  myTreeVariables_.nClusters  = numberOfClusters ;
  
  return ;
} // end dumpEndcapClusterInfo  



void
EcalTimeEleTreeMaker::dumpVertexInfo(const reco::VertexCollection* recVtxs, EcalTimeTreeContent & myTreeVariables_){

  // std::cout << "number of vertices: " << recVtxs->size() << std::endl;
  int thisVertex=0;
  myTreeVariables_.nVertices= recVtxs->size();

  // make sure you don't exceed with number of vertices the available slots in the ntuple format (thisVertex) 
  for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
      v!=recVtxs->end() && thisVertex<MAXVTX; 
      ++v){
    myTreeVariables_.vtxNTracks[thisVertex]=v->tracksSize();
    myTreeVariables_.vtxIsFake[thisVertex]= v->isFake();
    myTreeVariables_.vtxChi2[thisVertex]=   v->chi2();
    myTreeVariables_.vtxNdof[thisVertex]=   v->ndof();
    myTreeVariables_.vtxX[thisVertex]=      v->x();
    myTreeVariables_.vtxDx[thisVertex]=     v->xError();
    myTreeVariables_.vtxY[thisVertex]=      v->y();
    myTreeVariables_.vtxDy[thisVertex]=     v->yError();
    myTreeVariables_.vtxZ[thisVertex]=      v->z();
    myTreeVariables_.vtxDz[thisVertex]=     v->zError();
    
    thisVertex++;
  }// end loop on vertices

}


// GFdoc GT information, at present anf +-1 bx
void
EcalTimeEleTreeMaker::dump3Ginfo (const edm::Event& iEvent, 
                             const edm::EventSetup& iSetup,
                             EcalTimeTreeContent & myTreeVariables_)
{
  std::vector<bool> l1Triggers = determineTriggers (iEvent, iSetup);      
  myTreeVariables_.isECALL1 = l1Triggers[4];
  myTreeVariables_.isHCALL1 = l1Triggers[3];
  myTreeVariables_.isRPCL1 = l1Triggers[2];
  myTreeVariables_.isCSCL1 = l1Triggers[1];
  myTreeVariables_.isDTL1 = l1Triggers[0];
  
  int nbunches = 1;
  for(int bx = -nbunches; bx <=nbunches; bx++){
    std::vector<bool> l1Triggers = determineTriggers (iEvent, iSetup, bx);      
    myTreeVariables_.isECALL1Bx[ bx + nbunches] = l1Triggers[4];
    myTreeVariables_.isHCALL1Bx[ bx + nbunches] = l1Triggers[3];
    myTreeVariables_.isRPCL1Bx [ bx + nbunches] = l1Triggers[2];
    myTreeVariables_.isCSCL1Bx [ bx + nbunches] = l1Triggers[1];
    myTreeVariables_.isDTL1Bx  [ bx + nbunches] = l1Triggers[0];
  }

  // Fill the trigger bit arrays, from EcalTimingAnalysis
  myTreeVariables_.l1NActiveTriggers = 0 ;
  myTreeVariables_.l1NActiveTechTriggers = 0;
  //I (Jason) Decided ONLY to look at the L1 triggers that took part in the decision, not just the ACTIVE triggers
  // HOPEFULLY this wasn't a bad decision
  edm::Handle< L1GlobalTriggerReadoutRecord > gtRecord;
  iEvent.getByLabel(gtRecordCollectionTag_, gtRecord);
  DecisionWord dWord = gtRecord->decisionWord();   // this will get the decision word *before* masking disabled bits
  int iBit = -1;
  for (std::vector<bool>::iterator itBit = dWord.begin(); itBit != dWord.end(); ++itBit)
  {
    iBit++;
    if (*itBit)
    {
      myTreeVariables_.l1ActiveTriggers[myTreeVariables_.l1NActiveTriggers] = iBit ;
      myTreeVariables_.l1NActiveTriggers++;
    }
  }

  TechnicalTriggerWord tw = gtRecord->technicalTriggerWord();
  if ( ! tw.empty() )
  {
    // loop over dec. bit to get total rate (no overlap)
    for ( int itechbit = 0; itechbit < 64; ++itechbit )
    {
      myTreeVariables_.l1ActiveTechTriggers[myTreeVariables_.l1NActiveTechTriggers] = 0; // ADD THIS 

      if ( tw[itechbit] )
      {
        myTreeVariables_.l1ActiveTechTriggers[myTreeVariables_.l1NActiveTechTriggers] = itechbit;
        myTreeVariables_.l1NActiveTechTriggers++;
      }

    }
  }
 
  return ;
} //PG dump3Ginfo  



// -------------------------------------------------------------------------------------------------------------

//PG stolen from Toyoko's EcalCosmicHists, PGP changed it (added Bx information)
std::vector<bool> 
EcalTimeEleTreeMaker::determineTriggers (const edm::Event& iEvent, const edm::EventSetup& eventSetup, int Bx)
{
  std::vector<bool> l1Triggers; //DT,CSC,RPC,HCAL,ECAL
                                //0 , 1 , 2 , 3  , 4
  for(int i=0;i<5;i++)
    l1Triggers.push_back(false);
  
  Handle<L1GlobalTriggerReadoutRecord> L1GTRR;
  iEvent.getByLabel(l1GMTReadoutRecTag_,L1GTRR);
  
  //Ecal
  edm::ESHandle<L1GtTriggerMenu> menuRcd;
  eventSetup.get<L1GtTriggerMenuRcd>().get(menuRcd) ;
  const L1GtTriggerMenu* menu = menuRcd.product();
  edm::Handle< L1GlobalTriggerReadoutRecord > gtRecord;
  iEvent.getByLabel( edm::InputTag("gtDigis"), gtRecord);
  // Get dWord after masking disabled bits
  const DecisionWord dWord = gtRecord->decisionWord(Bx);
  
  bool l1SingleEG1 = menu->gtAlgorithmResult("L1_SingleEG1", dWord);
  bool l1SingleEG5 = menu->gtAlgorithmResult("L1_SingleEG5", dWord);
  bool l1SingleEG8 = menu->gtAlgorithmResult("L1_SingleEG8", dWord);
  bool l1SingleEG10 = menu->gtAlgorithmResult("L1_SingleEG10", dWord);
  bool l1SingleEG12 = menu->gtAlgorithmResult("L1_SingleEG12", dWord);
  bool l1SingleEG15 = menu->gtAlgorithmResult("L1_SingleEG15", dWord);
  bool l1SingleEG20 = menu->gtAlgorithmResult("L1_SingleEG20", dWord);
  bool l1SingleEG25 = menu->gtAlgorithmResult("L1_SingleEG25", dWord);
  bool l1DoubleNoIsoEGBTBtight = menu->gtAlgorithmResult("L1_DoubleNoIsoEG_BTB_tight", dWord);
  bool l1DoubleNoIsoEGBTBloose = menu->gtAlgorithmResult("L1_DoubleNoIsoEG_BTB_loose ", dWord);
  bool l1DoubleNoIsoEGTopBottom = menu->gtAlgorithmResult("L1_DoubleNoIsoEGTopBottom", dWord);
  bool l1DoubleNoIsoEGTopBottomCen  = menu->gtAlgorithmResult("L1_DoubleNoIsoEGTopBottomCen", dWord);
  bool l1DoubleNoIsoEGTopBottomCen2  = menu->gtAlgorithmResult("L1_DoubleNoIsoEGTopBottomCen2", dWord);
  bool l1DoubleNoIsoEGTopBottomCenVert  = menu->gtAlgorithmResult("L1_DoubleNoIsoEGTopBottomCenVert", dWord);
  
  l1Triggers[4] = l1SingleEG1 || l1SingleEG5 || l1SingleEG8 || l1SingleEG10 || l1SingleEG12 || l1SingleEG15
    || l1SingleEG20 || l1SingleEG25 || l1DoubleNoIsoEGBTBtight || l1DoubleNoIsoEGBTBloose
    || l1DoubleNoIsoEGTopBottom || l1DoubleNoIsoEGTopBottomCen || l1DoubleNoIsoEGTopBottomCen2
    || l1DoubleNoIsoEGTopBottomCenVert;

  
  L1GlobalTriggerReadoutRecord const* gtrr = L1GTRR.product();
  
  for(int ibx=-1; ibx<=1; ibx++) {
    bool hcal_top = false;
    bool hcal_bot = false;
    const L1GtPsbWord psb = gtrr->gtPsbWord(0xbb0d,ibx);
    std::vector<int> valid_phi;
    if((psb.aData(4)&0x3f) >= 1) {valid_phi.push_back( (psb.aData(4)>>10)&0x1f ); }
    if((psb.bData(4)&0x3f) >= 1) {valid_phi.push_back( (psb.bData(4)>>10)&0x1f ); }
    if((psb.aData(5)&0x3f) >= 1) {valid_phi.push_back( (psb.aData(5)>>10)&0x1f ); }
    if((psb.bData(5)&0x3f) >= 1) {valid_phi.push_back( (psb.bData(5)>>10)&0x1f ); }
    std::vector<int>::const_iterator iphi;
    for(iphi=valid_phi.begin(); iphi!=valid_phi.end(); iphi++) {
      //std::cout << "Found HCAL mip with phi=" << *iphi << " in bx wrt. L1A = " << ibx << std::endl;
      if(*iphi<9) hcal_top=true;
      if(*iphi>8) hcal_bot=true;
    }
    if(ibx==0 && hcal_top && hcal_bot) l1Triggers[3]=true;
  }

  return l1Triggers;
}
// -------------------------------------------------------------------------------------------------------------
