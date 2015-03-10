// -*- C++ -*-
//
// Package:   EcalTimeTreeMaker
// Class:     EcalTimeTreeMaker
//
/**\class EcalTimeTreeMaker EcalTimeTreeMaker.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
 */
//
// Authors:                              Seth Cooper, Giovanni Franzoni (UMN)
//         Created:  Mo Jul 14 5:46:22 CEST 2008
// $Id: EcalTimeTreeMaker.cc,v 1.14 2011/11/22 17:07:35 franzoni Exp $
//
//

 
#include "CalibCalorimetry/EcalTiming/plugins/EcalTimeTreeMaker.h"

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

#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"

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

EcalTimeTreeMaker::EcalTimeTreeMaker (const edm::ParameterSet& iConfig) :
  barrelEcalRecHitCollection_              (iConfig.getParameter<edm::InputTag> ("barrelEcalRecHitCollection")),
  endcapEcalRecHitCollection_              (iConfig.getParameter<edm::InputTag> ("endcapEcalRecHitCollection")),
  barrelBasicClusterCollection_            (iConfig.getParameter<edm::InputTag> ("barrelBasicClusterCollection")),
  endcapBasicClusterCollection_            (iConfig.getParameter<edm::InputTag> ("endcapBasicClusterCollection")),
  barrelSuperClusterCollection_            (iConfig.getParameter<edm::InputTag> ("barrelSuperClusterCollection")),
  endcapSuperClusterCollection_            (iConfig.getParameter<edm::InputTag> ("endcapSuperClusterCollection")),
  muonCollection_                          (iConfig.getParameter<edm::InputTag> ("muonCollection")),
  vertexCollection_                        (iConfig.getParameter<edm::InputTag> ("vertexCollection")),
  l1GMTReadoutRecTag_   (iConfig.getUntrackedParameter<std::string> ("L1GlobalReadoutRecord","gtDigis")),
  gtRecordCollectionTag_ (iConfig.getUntrackedParameter<std::string> ("GTRecordCollection","")),
  runNum_               (iConfig.getParameter<int> ("runNum")),
  minEtEB_              (iConfig.getParameter<double> ("minEtEB")),
  minEtEE_              (iConfig.getParameter<double> ("minEtEE")),
  fileName_             (iConfig.getUntrackedParameter<std::string> ("fileName", std::string ("EcalTimeTreeMaker"))),
  naiveId_ (0)              

{
  // TrackAssociator parameters // gfwork: can we remove this? 
  edm::ParameterSet trkParameters = iConfig.getParameter<edm::ParameterSet> ("TrackAssociatorParameters") ;
  trackParameters_.loadParameters ( trkParameters ) ;

  // Create File
  fileName_ += "_"+intToString (runNum_)+".root" ;
  file_ = new TFile (fileName_.c_str () , "RECREATE") ;
  file_->cd () ;

  // Initialize Tree
  tree_ = new TTree ( "EcalTimeAnalysis","EcalTimeAnalysis" ) ;
  setBranches (tree_, myTreeVariables_) ;

}


// -----------------------------------------------------------------------------------------


EcalTimeTreeMaker::~EcalTimeTreeMaker ()
{
}


// -----------------------------------------------------------------------------------------


void EcalTimeTreeMaker::analyze (const edm::Event& iEvent, const edm::EventSetup& iSetup)
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
      LogWarning ("EcalTimeTreeMaker") << barrelEcalRecHitCollection_ 
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
      LogWarning ("EcalTimeTreeMaker") << endcapEcalRecHitCollection_ 
                                     << " not available" ;
      return ;
    }
  
  // GFdo switch to appropriate clusters here (basic instead of super) 
  // Barrel SuperClusters
  edm::Handle<reco::SuperClusterCollection> pBarrelSuperClusters ;
  iEvent.getByLabel (barrelSuperClusterCollection_, pBarrelSuperClusters) ;
  const reco::SuperClusterCollection* theBarrelSuperClusters = pBarrelSuperClusters.product () ;

  if (! (pBarrelSuperClusters.isValid ()) )
    {
      LogWarning ("EcalTimeTreeMaker") << barrelSuperClusterCollection_ 
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
  
  
  // Barrel BasicClusters
  edm::Handle<reco::BasicClusterCollection> pBarrelBasicClusters ;
  iEvent.getByLabel (barrelBasicClusterCollection_, pBarrelBasicClusters) ;
  const reco::BasicClusterCollection* theBarrelBasicClusters = pBarrelBasicClusters.product () ;

  if (! (pBarrelBasicClusters.isValid ()) )
    {
      LogWarning ("EcalTimeTreeMaker") << barrelBasicClusterCollection_ 
                                     << " not available" ;
      return ;
    }

  // Endcap BasicClusters
  edm::Handle<reco::BasicClusterCollection> pEndcapBasicClusters ;
  iEvent.getByLabel (endcapBasicClusterCollection_, pEndcapBasicClusters) ;
  const reco::BasicClusterCollection* theEndcapBasicClusters = pEndcapBasicClusters.product () ;
  
  if (! (pEndcapBasicClusters.isValid ()) )
    {
      LogWarning ("EcalTimeTreeMaker") << endcapBasicClusterCollection_ 
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

  Handle<reco::VertexCollection> recVtxs;
  iEvent.getByLabel(vertexCollection_, recVtxs);
  const reco::VertexCollection * theRecVtxs = recVtxs.product();

  dump3Ginfo(iEvent, iSetup, myTreeVariables_) ;
  
  dumpBarrelClusterInfo(iEvent, theGeometry, theCaloTopology,
			theBarrelEcalRecHits, 
			theBarrelBasicClusters, theBarrelSuperClusters, lazyTools, XtalMap, XtalMapCurved, myTreeVariables_) ;
  dumpEndcapClusterInfo(iEvent, theGeometry, theCaloTopology,
			theEndcapEcalRecHits, 
			theEndcapBasicClusters, theEndcapSuperClusters, lazyTools, XtalMap, XtalMapCurved, myTreeVariables_) ;
  
  dumpVertexInfo(theRecVtxs, myTreeVariables_);

  tree_ -> Fill();
  delete lazyTools;
}






// -----------------------------------------------------------------------------------------

void EcalTimeTreeMaker::endJob ()
{
  using namespace std ;
  file_->cd () ;
  tree_->Write () ;
  file_->Close () ;

}


// -----------------------------------------------------------------------------------------

void EcalTimeTreeMaker::beginRun(edm::Run const &, edm::EventSetup const & eventSetup)
{
  // IC's
  eventSetup.get<EcalIntercalibConstantsRcd>().get(ical);
  // ADCtoGeV
  eventSetup.get<EcalADCToGeVConstantRcd>().get(agc);
  // transp corrections
  eventSetup.get<EcalLaserDbRecord>().get(laser);
}



// -----------------------------------------------------------------------------------------

std::string EcalTimeTreeMaker::intToString (int num)
{
    using namespace std ;
    ostringstream myStream ;
    myStream << num << flush ;
    return (myStream.str ()) ; //returns the string form of the stringstream object
}






// -----------------------------------------------------------------------------------------

void EcalTimeTreeMaker::dumpBarrelClusterInfo (const edm::Event& iEvent,
     				             const CaloGeometry * theGeometry,
					     const CaloTopology * theCaloTopology,
					     const EcalRecHitCollection* theBarrelEcalRecHits,
					     const reco::BasicClusterCollection* theBarrelBasicClusters,
					     const reco::SuperClusterCollection* theBarrelSuperClusters,
					     EcalClusterLazyTools* lazyTools,
					     const std::map<int,float> & XtalMap, //GFdoc unclear
					     const std::map<int,float> & XtalMapCurved, //GFdoc unclear
					     EcalTimeTreeContent & myTreeVarnumberOfSuperClustersiables_)
{            
  // get number of of objects already present in the tree (none if dumpBarrelClusterInfo is called first)
  // this will be the index looping over the SC collection;
  int numberOfSuperClusters = myTreeVariables_.nSuperClusters;
  // this will be the index looping over the BC collection;
  int numberOfClusters      = myTreeVariables_.nClusters;

  const EcalIntercalibConstantMap& icalMap = ical->getMap();
  float adcToGeV = float(agc->getEBValue());
  
  //number of superClusters in event (collection = vector!)
  myTreeVariables_.nSuperClusters += theBarrelSuperClusters->size () ;
  myTreeVariables_.nBarrelSuperClusters = theBarrelSuperClusters->size () ;
  
  /////////////////////////////////////////////////////////////////////////////////////////
  //loop on all superclusters in event
  for (reco::SuperClusterCollection::const_iterator sclus = theBarrelSuperClusters->begin () ; 
       sclus != theBarrelSuperClusters->end ()  && numberOfSuperClusters<MAXSC;
       ++sclus) 
    {//start Loop on SC's
  
      //    int numberOfXtalsInSuperCluster = 0 ;//counter for all xtals in supercluster 
      
      myTreeVariables_.nClustersInSuperCluster[numberOfSuperClusters] = sclus -> clustersSize () ;
      //myTreeVariables_.superClusterType[numberOfSuperClusters] = 0 ;
      
      myTreeVariables_.superClusterEta[numberOfSuperClusters] = sclus -> position ().eta () ;
      myTreeVariables_.superClusterPhi[numberOfSuperClusters] = sclus -> position ().phi () ;
      myTreeVariables_.superClusterX[numberOfSuperClusters] = sclus -> position ().x () ;
      myTreeVariables_.superClusterY[numberOfSuperClusters] = sclus -> position ().y () ;
      myTreeVariables_.superClusterZ[numberOfSuperClusters] = sclus -> position ().z () ;
      myTreeVariables_.superClusterRawEnergy[numberOfSuperClusters] = sclus -> rawEnergy () ;
      myTreeVariables_.superClusterPhiWidth[numberOfSuperClusters] = sclus -> phiWidth () ;
      myTreeVariables_.superClusterEtaWidth[numberOfSuperClusters] = sclus -> etaWidth () ;

      numberOfSuperClusters++ ;
      
    } //end supercluster loop
 // GF - FIXME : at some point we may want to loop only over BC's which are included inside a give SC
 // more natural. But even what we have (i.e. two separate loops) should not generate inconsistencies
 

   ///////////////////////////////////////////////////////////////////////////////////////
   // loop on barrel basic clusters (independently of loop on SC)
   for (reco::BasicClusterCollection::const_iterator clus = theBarrelBasicClusters->begin () ; 
	clus != theBarrelBasicClusters->end ()  && numberOfClusters<MAXC;  
	++clus) // loop on barrel Bclusters
     {        
       double energy = (clus)->energy () ;
       double phi    = (clus)->phi () ;
       double eta    = (clus)->eta () ;
       if  ( energy/cosh(eta) <minEtEB_ ) continue;
       double sinTheta         = fabs( sin( 2 *atan( exp(-1*(clus)->eta()) ) ) );
       double transverseEnergy = (clus)->energy () * sinTheta;
       double time = -1000.0 ; // gfdoc: work on this to provide a combination of crystals?
       double ampli = 0. ;
       double secondMin = 0. ;
       double secondTime = -1000. ;
       int numberOfXtalsInCluster=0 ;//counter for all xtals in cluster
       int numXtalsinCluster = 0 ;   //xtals in cluster above 3sigma noise  
       EBDetId maxDet ;
       EBDetId secDet ;
       
       // GFdoc clusterDetIds holds crystals that participate to this basic cluster 
       std::vector<std::pair<DetId, float> > clusterDetIds = (clus)->hitsAndFractions() ; //get these from the cluster
       
       //////////////////////////////////////////////////////////////////////////////////////
       //loop on xtals in cluster
       for (std::vector<std::pair<DetId, float> >::const_iterator detitr = clusterDetIds.begin () ; 
	    detitr != clusterDetIds.end ()  && numberOfXtalsInCluster<MAXXTALINC; // && numberOfXtals<MAXXTAL ; 
	    ++detitr)// loop on rechics of barrel basic clusters
	 {

	   // Here I use the "find" on a digi collection... I have been warned...   (GFdoc: ??)
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
	   //The checking above should no longer be needed... 
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
	   double thistime = myhit.time ();
	   double thisChi2 = myhit.chi2 ();
	   double thisOutOfTimeChi2 = myhit.outOfTimeChi2 ();

	   
	   EcalIntercalibConstantMap::const_iterator icalit = icalMap.find(detitr->first);
	   EcalIntercalibConstant icalconst = 1;
	   if( icalit!=icalMap.end() ) {
	     icalconst = (*icalit);
	   } else {
	     edm::LogError("EcalTimeTreeMaker") << "No intercalib const found for xtal "
						<< (detitr->first).rawId();
	   }
	   
	   // get laser coefficient
	   float lasercalib = 1.;
	   lasercalib = laser->getLaserCorrection( detitr->first, iEvent.time());
	   // std::cout << "GF EB debug: " << lasercalib << std::endl;

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
	   myTreeVariables_.xtalInBCTime[numberOfClusters][numberOfXtalsInCluster]=         (float) thistime; 
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
           myTreeVariables_.xtalInBCSwissCross[numberOfClusters][numberOfXtalsInCluster] =
             EcalTools::swissCross(detitr->first,*theBarrelEcalRecHits,0.5);

  
	   GlobalPoint pos = theGeometry->getPosition((myhit).detid());
	   myTreeVariables_.xtalInBCEta[numberOfClusters][numberOfXtalsInCluster]=      pos.eta();
	   myTreeVariables_.xtalInBCPhi[numberOfClusters][numberOfXtalsInCluster]=      pos.phi();
	   
	   numberOfXtalsInCluster++ ; // increment number of crystals in basic cluster
	   
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
	 myTreeVariables_.clusterE2x2[numberOfClusters] = lazyTools -> e2x2(*(clus));
	 myTreeVariables_.clusterE3x2[numberOfClusters] = lazyTools -> e3x2(*(clus));
	 myTreeVariables_.clusterE3x3[numberOfClusters] = lazyTools -> e3x3(*(clus));
	 myTreeVariables_.clusterE4x4[numberOfClusters] = lazyTools -> e4x4(*(clus));
	 myTreeVariables_.clusterE5x5[numberOfClusters] = lazyTools -> e5x5(*(clus));
	 
	 myTreeVariables_.clusterE2x5Right[numberOfClusters]  = lazyTools -> e2x5Right(*(clus));
	 myTreeVariables_.clusterE2x5Left[numberOfClusters]   = lazyTools -> e2x5Left(*(clus));
	 myTreeVariables_.clusterE2x5Top[numberOfClusters]    = lazyTools -> e2x5Top(*(clus));
	 myTreeVariables_.clusterE2x5Bottom[numberOfClusters] = lazyTools -> e2x5Bottom(*(clus));
	 
	 myTreeVariables_.clusterCovEtaEta[numberOfClusters] = (lazyTools -> covariances(*(clus)))[0];
	 myTreeVariables_.clusterCovPhiPhi[numberOfClusters] = (lazyTools -> covariances(*(clus)))[2];
	 myTreeVariables_.clusterCovEtaPhi[numberOfClusters] = (lazyTools -> covariances(*(clus)))[1];
	 
	 myTreeVariables_.clusterLat[numberOfClusters] = (lazyTools -> lat(*(clus)))[2];
	 myTreeVariables_.clusterEtaLat[numberOfClusters] = (lazyTools -> lat(*(clus)))[0];
	 myTreeVariables_.clusterPhiLat[numberOfClusters] = (lazyTools -> lat(*(clus)))[1];
	 
	 myTreeVariables_.clusterZernike20[numberOfClusters] = lazyTools -> zernike20(*(clus));
	 myTreeVariables_.clusterZernike42[numberOfClusters] = lazyTools -> zernike42(*(clus));
       }
       
       numberOfClusters++ ;
       
     } //end (BASIC)cluster loop
   
   
   //number of Bclusters already stored in the ntuple
   myTreeVariables_.nClusters = numberOfClusters;
   //myTreeVariables_.nXtals    = numberOfXtals ; - obsolete

   return ;
} // end dumpBarrelClusterInfo  


// -----------------------------------------------------------------------------------------

void EcalTimeTreeMaker::dumpEndcapClusterInfo (const edm::Event& iEvent,
                                             const CaloGeometry * theGeometry,
					     const CaloTopology * theCaloTopology,
					     const EcalRecHitCollection* theEndcapEcalRecHits,
					     const reco::BasicClusterCollection* theEndcapBasicClusters,
					     const reco::SuperClusterCollection* theEndcapSuperClusters,
					     EcalClusterLazyTools* lazyTools,
					     const std::map<int,float> & XtalMap,
					     const std::map<int,float> & XtalMapCurved,
					     EcalTimeTreeContent & myTreeVariables_)
{
  // counters come from the ntuple are to account for what was added in dumpBarrelClusterInf
  int numberOfSuperClusters = myTreeVariables_.nSuperClusters;
  int numberOfClusters      = myTreeVariables_.nClusters;
  // int numberOfXtals         = myTreeVariables_.nXtals ; // this is number of crystals associated to any cluster

  const EcalIntercalibConstantMap& icalMap = ical->getMap();
  float adcToGeV = float(agc->getEEValue());
  
  
  //number of superClusters in event (collection = vector!)
  myTreeVariables_.nSuperClusters += theEndcapSuperClusters->size () ;
  myTreeVariables_.nEndcapSuperClusters = theEndcapSuperClusters->size () ;
  
  //loop on all endcap superclusters in event
  for (reco::SuperClusterCollection::const_iterator sclus = theEndcapSuperClusters->begin () ; 
       sclus != theEndcapSuperClusters->end ()  && numberOfSuperClusters<MAXSC; 
       ++sclus) 
    {//loop on SC's
      
      myTreeVariables_.nClustersInSuperCluster[numberOfSuperClusters] = sclus -> clustersSize () ;
     
      myTreeVariables_.superClusterEta[numberOfSuperClusters] = sclus -> position ().eta () ;
      myTreeVariables_.superClusterPhi[numberOfSuperClusters] = sclus -> position ().phi () ;
      myTreeVariables_.superClusterX[numberOfSuperClusters] = sclus -> position ().x () ;
      myTreeVariables_.superClusterY[numberOfSuperClusters] = sclus -> position ().y () ;
      myTreeVariables_.superClusterZ[numberOfSuperClusters] = sclus -> position ().z () ;
      myTreeVariables_.superClusterRawEnergy[numberOfSuperClusters] = sclus -> energy () ;
      myTreeVariables_.superClusterPhiWidth[numberOfSuperClusters] = sclus -> phiWidth () ;
      myTreeVariables_.superClusterEtaWidth[numberOfSuperClusters] = sclus -> etaWidth () ;

      numberOfSuperClusters++ ;
  
    } //end endcap supercluster loopsuperClusterEnergySum
  

  ///////////////////////////////////////////////////////////////////////////////////////
  // independent loop on endcap basic clusters
  // numberOfClusters takes into account for what was added in dumpBarrelClusterInfo

  for (reco::BasicClusterCollection::const_iterator clus = theEndcapBasicClusters->begin () ; 
       clus != theEndcapBasicClusters->end () && numberOfClusters<MAXC;
       ++clus) // loop on endcap Bclusters
    {        

         double energy = (clus)->energy () ;
         double phi    = (clus)->phi () ;
         double eta    = (clus)->eta () ;
	 if  ( energy/cosh(eta) <minEtEE_ ) continue;
	 double sinTheta         = fabs( sin( 2 *atan( exp(-1*(clus)->eta()) ) ) );
	 double transverseEnergy = (clus)->energy () * sinTheta;
         double time = -1000.0 ;  // gfdoc: work on this to provide a combination of crystals?
         double ampli = 0. ;
         double secondMin = 0. ;
         double secondTime = -1000. ;
         int numberOfXtalsInCluster=0 ;//counter for all xtals in cluster
         int numXtalsinCluster = 0 ;//xtals in cluster above 3sigma noise  
         EEDetId maxDet ;
         EEDetId secDet ;
         
         std::vector<std::pair<DetId, float> > clusterDetIds = (clus)->hitsAndFractions() ; //get these from the cluster

        //loop on xtals in cluster
         for (std::vector<std::pair<DetId, float> >::const_iterator detitr = clusterDetIds.begin () ; 
              detitr != clusterDetIds.end () && numberOfXtalsInCluster<MAXXTALINC;// && numberOfXtals<MAXXTAL ; 
              ++detitr)// loop on rechits of endcap basic clusters
           {
             //Here I use the "find" on a digi collection... I have been warned...
             if ( (detitr -> first).det () != DetId::Ecal) 
               { 
                 std::cout << " det is " << (detitr -> first).det () << std::endl ;
                 continue ;
               }
             if ( (detitr -> first).subdetId () != EcalEndcap) 
               {
                 std::cout << " subdet is " << (detitr -> first).subdetId () << std::endl ; 
                 continue ; 
               }
             EcalRecHitCollection::const_iterator thishit = theEndcapEcalRecHits->find ( (detitr -> first) ) ;
             if (thishit == theEndcapEcalRecHits->end ()) 
               {
                 continue ;
               }
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

	     // get laser coefficient
	     float lasercalib = 1.;
	     lasercalib = laser->getLaserCorrection( detitr->first, iEvent.time());
	     //std::cout << "GF EE debug: " << lasercalib << std::endl;
	     
             EcalIntercalibConstantMap::const_iterator icalit = icalMap.find(detitr->first);
             EcalIntercalibConstant icalconst = 1;
             if( icalit!=icalMap.end() ) {
               icalconst = (*icalit);
             } else {
               edm::LogError("EcalTimeTreeMaker") << "No intercalib const found for xtal "
						  << (detitr->first).rawId();
             }
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

              
             numberOfXtalsInCluster++ ; // increment number of crystals in basic cluster
	     
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
         myTreeVariables_.clusterE2x2[numberOfClusters] = lazyTools -> e2x2(*(clus));

         myTreeVariables_.clusterE3x2[numberOfClusters] = lazyTools -> e3x2(*(clus));
         myTreeVariables_.clusterE3x3[numberOfClusters] = lazyTools -> e3x3(*(clus));
         myTreeVariables_.clusterE4x4[numberOfClusters] = lazyTools -> e4x4(*(clus));
         myTreeVariables_.clusterE5x5[numberOfClusters] = lazyTools -> e5x5(*(clus));

         myTreeVariables_.clusterE2x5Right[numberOfClusters] = lazyTools -> e2x5Right(*(clus));
         myTreeVariables_.clusterE2x5Left[numberOfClusters] = lazyTools -> e2x5Left(*(clus));
         myTreeVariables_.clusterE2x5Top[numberOfClusters] = lazyTools -> e2x5Top(*(clus));
         myTreeVariables_.clusterE2x5Bottom[numberOfClusters] = lazyTools -> e2x5Bottom(*(clus));

         myTreeVariables_.clusterCovEtaEta[numberOfClusters] = (lazyTools -> covariances(*(clus)))[0];
         myTreeVariables_.clusterCovPhiPhi[numberOfClusters] = (lazyTools -> covariances(*(clus)))[2];
         myTreeVariables_.clusterCovEtaPhi[numberOfClusters] = (lazyTools -> covariances(*(clus)))[1];

         myTreeVariables_.clusterLat[numberOfClusters] = (lazyTools -> lat(*(clus)))[2];
         myTreeVariables_.clusterEtaLat[numberOfClusters] = (lazyTools -> lat(*(clus)))[0];
         myTreeVariables_.clusterPhiLat[numberOfClusters] = (lazyTools -> lat(*(clus)))[1];

         myTreeVariables_.clusterZernike20[numberOfClusters] = lazyTools -> zernike20(*(clus));
         myTreeVariables_.clusterZernike42[numberOfClusters] = lazyTools -> zernike42(*(clus));
    
         numberOfClusters++ ;

    }//end endcap basic cluster loop
     
  //number of B clusters already stored in the ntuple
  myTreeVariables_.nClusters  = numberOfClusters ;
  //myTreeVariables_.nXtals     = numberOfXtals ;

  return ;
} // end dumpEndcapClusterInfo  



void
EcalTimeTreeMaker::dumpVertexInfo(const reco::VertexCollection* recVtxs, EcalTimeTreeContent & myTreeVariables_){

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
EcalTimeTreeMaker::dump3Ginfo (const edm::Event& iEvent, 
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

  //SIC - July 5 2010
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
EcalTimeTreeMaker::determineTriggers (const edm::Event& iEvent, const edm::EventSetup& eventSetup, int Bx)
{
  std::vector<bool> l1Triggers; //DT,CSC,RPC,HCAL,ECAL
                                //0 , 1 , 2 , 3  , 4
  for(int i=0;i<5;i++)
    l1Triggers.push_back(false);

  // get hold of L1GlobalReadoutRecord
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
  //cout << "l1Triggers[3]" << l1Triggers[3] << endl;

  //cout << "l1Triggers: "; 
  //for(int i=0;i<5;i++)
  //  cout << l1Triggers[i];
  //cout << endl;
  return l1Triggers;
}
// -------------------------------------------------------------------------------------------------------------
