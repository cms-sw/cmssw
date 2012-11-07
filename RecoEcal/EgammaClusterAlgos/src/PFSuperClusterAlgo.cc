//#include "RecoParticleFlow/PFClusterProducer/interface/PFSuperClusterAlgo.h"
#include "RecoEcal/EgammaClusterAlgos/interface/PFSuperClusterAlgo.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFClusterWidthAlgo.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "RecoEcal/EgammaCoreTools/interface/Mustache.h"
#include "Math/GenVector/VectorUtil.h"
#include "TFile.h"
#include "TH2F.h"
#include "TROOT.h"
#include "TMath.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <stdexcept>
#include <string>
#include <sstream>

using namespace std;

PFSuperClusterAlgo::PFSuperClusterAlgo()

{

}


void PFSuperClusterAlgo::doClustering(const edm::Handle<reco::PFClusterCollection> & pfclustersHandle, std::auto_ptr< reco::BasicClusterCollection > & basicClusters_p, boost::shared_ptr<PFEnergyCalibration> thePFEnergyCalibration_, int detector){

  //Similar to hybrid e/gamma, algorithm but same in barrel and endcap
  //0- Calibrate PFCluster energies
  //1- first look for the seed pfclusters
  //2- around the seed, open the (etawidth, phiwidth) window
  //3- take all the pfclusters passing the threshold in this window and remove them from the list of available pfclusters
  //4- go to next seed if not already clustered
  
  //Setting parameters
  if (detector==0) {
    threshPFClusterSeed_ = threshPFClusterSeedBarrel_;
    threshPFCluster_ = threshPFClusterBarrel_;
    etawidthSuperCluster_ = etawidthSuperClusterBarrel_;
    phiwidthSuperCluster_ = phiwidthSuperClusterBarrel_;
  }
  if (detector==1) {
    threshPFClusterSeed_ = threshPFClusterSeedEndcap_;
    threshPFCluster_ = threshPFClusterEndcap_;
    etawidthSuperCluster_ = etawidthSuperClusterEndcap_;
    phiwidthSuperCluster_ = phiwidthSuperClusterEndcap_;
  }

  if (verbose_) {
    if (detector==0) cout << "PFSuperClusterAlgo::doClustering in EB" << endl;
    if (detector==1) cout << "PFSuperClusterAlgo::doClustering in EE" << endl;
  }

  //cleaning vectors
  pfClusters_.clear();
  basicClusters_.clear();

  basicClusterPtr_.clear();

  scPFseedIndex_.clear();
  seedCandidateIndex_.clear();
  pfClusterIndex_.clear();

  EclustersPS1.clear();
  EclustersPS2.clear();

  allPfClusterCalibratedEnergy_.clear();
  pfClusterCalibratedEnergy_.clear();

  seedCandidateCollection.clear();
  pfClusterAboveThresholdCollection.clear();


  int layer;
  if (detector==0) layer = PFLayer::ECAL_BARREL;
  if (detector==1) layer = PFLayer::ECAL_ENDCAP;

  //Select PF clusters available for the clustering
  for (uint i=0; i<pfclustersHandle->size(); i++){

    if (verbose_) cout << "PFCluster i="<<i<<" energy="<<(*pfclustersHandle)[i].energy()<<endl;
    
    if ((*pfclustersHandle)[i].layer()==layer){

      const reco::PFCluster & myPFCluster (*(reco::PFClusterRef(pfclustersHandle, i)));
      double PFClusterCalibratedEnergy = thePFEnergyCalibration_->energyEm(myPFCluster,EclustersPS1,EclustersPS2,applyCrackCorrections_);

      allPfClusterCalibratedEnergy_.push_back(PFClusterCalibratedEnergy);

      //select PFClusters seeds
      if (PFClusterCalibratedEnergy > threshPFClusterSeed_){
	const reco::PFClusterRef seedCandidate = reco::PFClusterRef(pfclustersHandle, i);
	seedCandidateCollection.push_back(seedCandidate);
      }

      //select PFClusters above thresholds
      if (PFClusterCalibratedEnergy > threshPFCluster_){
	const reco::PFClusterRef pfClusterAboveThreshold = reco::PFClusterRef(pfclustersHandle, i);
	pfClusterAboveThresholdCollection.push_back(pfClusterAboveThreshold);
	pfClusterIndex_.push_back(i);
      }

    }

  }

  //Sort seeds by energy
  sort(seedCandidateCollection.begin(), seedCandidateCollection.end(), less_magPF());
 
  if (verbose_) cout << "After sorting"<<endl;
  for (uint is=0; is<seedCandidateCollection.size(); is++) {
    if (verbose_) cout << "SeedPFCluster "<<is<< " energy="<<seedCandidateCollection[is]->energy()<<endl;

    for (uint i=0; i<pfclustersHandle->size(); i++){
      if (seedCandidateCollection[is]->energy()==(*pfclustersHandle)[i].energy()) {
	seedCandidateIndex_.push_back(i);
	if (verbose_) cout << "seedCandidateIndex_[is]="<<seedCandidateIndex_[is]<<endl;
      }
    }
  }


  //Keep pfclusters within etawidth-phiwidth window around seeds

  isSeedUsed = new int[seedCandidateCollection.size()];
  for (uint is=0; is<seedCandidateCollection.size(); is++) isSeedUsed[is] = 0;

  isPFclusterUsed = new int[pfClusterAboveThresholdCollection.size()];
  for (uint j=0; j<pfClusterAboveThresholdCollection.size(); j++) isPFclusterUsed[j] = 0;

  isClusterized = new bool[pfclustersHandle->size()];
  for (uint i=0; i<pfclustersHandle->size(); i++) isClusterized[i] = false;

  nSuperClusters = 0;

  //run over the seed candidates
  for (uint is=0; is<seedCandidateCollection.size(); is++){

    if (verbose_) cout << "is="<<is<<" seedCandidate Energy="<<seedCandidateCollection[is]->energy() <<" eta="<< seedCandidateCollection[is]->eta()<< " phi="<< seedCandidateCollection[is]->phi()<< endl;

    if (isClusterized[seedCandidateIndex_[is]]==true) {
     if (verbose_) cout << "This seed cluster (energy="<<(*pfclustersHandle)[seedCandidateIndex_[is]].energy() <<") is already used, switching to next one"  << endl;
      continue;
    }
    isSeedUsed[is]=0;

    //check if the pfclusters can belong to the seeded supercluster
    for (uint j=0; j<pfClusterAboveThresholdCollection.size(); j++){
      
      if (isPFclusterUsed[j]==1) {
	if (verbose_) cout << "This PFCluster (energy="<<pfClusterAboveThresholdCollection[j]->energy() <<") is already used" << endl;
	continue;
      }
      
      //checking if the pfcluster is inside the eta/phi box around the seed
      if (fabs(seedCandidateCollection[is]->eta()-pfClusterAboveThresholdCollection[j]->eta())<etawidthSuperCluster_
	  && fabs(seedCandidateCollection[is]->phi()-pfClusterAboveThresholdCollection[j]->phi())<phiwidthSuperCluster_
		  ){

	isSeedUsed[is]++;
	if (isSeedUsed[is]==1){
	  pfClusters_.push_back(std::vector<const reco::PFCluster *>());
	  basicClusters_.push_back(reco::BasicClusterCollection());
	  pfClusterCalibratedEnergy_.push_back(std::vector<double>());
	}

	isPFclusterUsed[j]=1;
	
	//add pfcluster to collection of basicclusters
	createBasicCluster(pfClusterAboveThresholdCollection[j], basicClusters_[nSuperClusters], pfClusters_[nSuperClusters]);

	const reco::PFCluster & myPFCluster (*(pfClusterAboveThresholdCollection[j]));
	double PFClusterCalibratedEnergy = thePFEnergyCalibration_->energyEm(myPFCluster,EclustersPS1,EclustersPS2,applyCrackCorrections_);
	pfClusterCalibratedEnergy_[nSuperClusters].push_back(PFClusterCalibratedEnergy);

	if (pfClusterAboveThresholdCollection[j]->energy()==seedCandidateCollection[is]->energy()) {
	  scPFseedIndex_.push_back(basicClusters_[nSuperClusters].size()-1);
	}

	if (verbose_) cout << "Use PFCluster "<<j<<" eta="<< pfClusterAboveThresholdCollection[j]->eta()<< "phi="<< pfClusterAboveThresholdCollection[j]->phi()<<" energy="<< pfClusterAboveThresholdCollection[j]->energy()<<" calibEnergy="<< pfClusterCalibratedEnergy_[nSuperClusters][basicClusters_[nSuperClusters].size()-1]<<endl;
	
	isClusterized[pfClusterIndex_[j]] = true;
	
      }
    }
    
    
    //If the seed was used, store the basic clusters
    if (isSeedUsed[is]>0) {

      if (verbose_) cout << "New supercluster, number "<<nSuperClusters<<" having "<< basicClusters_[nSuperClusters].size()<< " basicclusters"<<endl;
      if (verbose_) for (uint i=0; i<basicClusters_[nSuperClusters].size(); i++) cout << "BC "<<i<<" energy="<<basicClusters_[nSuperClusters][i].energy()<<endl;

      basicClusters_p->insert(basicClusters_p->end(),basicClusters_[nSuperClusters].begin(), basicClusters_[nSuperClusters].end());

      if (verbose_) cout << "basicClusters_p filled" << endl;

      nSuperClusters++;
    }
    
  }

  if (verbose_) {
    if (detector==0) cout << "Leaving doClustering in EB (nothing more to clusterize)"<<endl;
    if (detector==1) cout << "Leaving doClustering in EE (nothing more to clusterize)"<<endl;
  }


  //Deleting objects
  delete[] isSeedUsed;
  delete[] isPFclusterUsed;
  delete[] isClusterized;

  return;
}


void PFSuperClusterAlgo::createBasicCluster(const reco::PFClusterRef & myPFClusterRef, 
					      reco::BasicClusterCollection & basicClusters, 
					      std::vector<const reco::PFCluster *> & pfClusters) const
{

  //cout << "Inside PFSuperClusterAlgo::createBasicCluster"<<endl;

  if(myPFClusterRef.isNull()) return;  

  const reco::PFCluster & myPFCluster (*myPFClusterRef);
  pfClusters.push_back(&myPFCluster);



  basicClusters.push_back(reco::CaloCluster(//coCandidate.rawEcalEnergy(),
					    myPFCluster.energy(),
					    myPFCluster.position(),
					    myPFCluster.caloID(),
					    myPFCluster.hitsAndFractions(),
					    myPFCluster.algo(),
					    myPFCluster.seed()));


}

void PFSuperClusterAlgo::createBasicClusterPtrs(const edm::OrphanHandle<reco::BasicClusterCollection> & basicClustersHandle )
{
  unsigned size=nSuperClusters;
  unsigned basicClusterCounter=0;
  basicClusterPtr_.resize(size);

  for(unsigned is=0;is<size;++is) // loop on superclusters
    {
      unsigned nbc=basicClusters_[is].size();
      for(unsigned ibc=0;ibc<nbc;++ibc) // loop on basic clusters
	{
	  reco::CaloClusterPtr bcPtr(basicClustersHandle,basicClusterCounter);
	  basicClusterPtr_[is].push_back(bcPtr);
	  ++basicClusterCounter;
	}
    }
}


void PFSuperClusterAlgo::createSuperClusters(reco::SuperClusterCollection &superClusters, bool doEEwithES) const{

  unsigned ns=nSuperClusters;
  for(unsigned is=0;is<ns;++is)
    {

      // Computes energy position a la e/gamma 
      double sclusterE=0;
      double posX=0.;
      double posY=0.;
      double posZ=0.;
      
      double correctedEnergy = 0;
      double correctedEnergyWithES = 0;

      unsigned nbasics=basicClusters_[is].size();
      for(unsigned ibc=0;ibc<nbasics;++ibc)
	{

	  if (doMustacheCut_ && insideMust_[is][ibc] == 0) continue; //Cleaning of PU clusters outside Mustache area
	  
	  double BCenergy = basicClusters_[is][ibc].energy();
	  sclusterE += BCenergy;

	  //Use PFCluster calibrated energy
	  correctedEnergy += pfClusterCalibratedEnergy_[is][ibc];
	  if (doEEwithES) correctedEnergyWithES += pfClusterCalibratedEnergyWithES_[is][ibc];

	  posX += BCenergy * basicClusters_[is][ibc].position().X();
	  posY += BCenergy * basicClusters_[is][ibc].position().Y();
	  posZ += BCenergy * basicClusters_[is][ibc].position().Z();	  
	}
      posX /=sclusterE;
      posY /=sclusterE;
      posZ /=sclusterE;
          

      // compute the width
      PFClusterWidthAlgo pfwidth(pfClusters_[is]);
      
      //create the supercluster
      double corrEnergy = correctedEnergy;
      if (doEEwithES) corrEnergy = correctedEnergyWithES;
      reco::SuperCluster mySuperCluster(corrEnergy,math::XYZPoint(posX,posY,posZ));

      if (verbose_) {
	if (!doEEwithES) cout << "Supercluster "<<is<< " eta="<< mySuperCluster.eta() <<" phi="<< mySuperCluster.phi()<< " rawEnergy="<<sclusterE<<" correctedEnergy="<<correctedEnergy <<endl;
	if (doEEwithES) cout << "Supercluster "<<is<< " eta="<< mySuperCluster.eta() <<" phi="<< mySuperCluster.phi()<< " rawEnergy="<<sclusterE<<" correctedEnergy="<<correctedEnergy <<" correctedEnergyWithES="<<correctedEnergyWithES<<endl;
      }

      
      if(nbasics)
	{
	  if (verbose_) std::cout << "Seed cluster energy=" << basicClusters_[is][scPFseedIndex_[is]].energy() << std::endl;
	  mySuperCluster.setSeed(basicClusterPtr_[is][scPFseedIndex_[is]]);
	}
      else
	{
	  mySuperCluster.setSeed(reco::CaloClusterPtr());
	}
      // the seed should be the first basic cluster


      for(unsigned ibc=0;ibc<nbasics;++ibc)
	{
	  mySuperCluster.addCluster(basicClusterPtr_[is][ibc]);
	  //	  std::cout <<"Adding Ref to SC " << basicClusterPtr_[is][ibc].index() << std::endl;
	  const std::vector< std::pair<DetId, float> > & v1 = basicClusters_[is][ibc].hitsAndFractions();
	  for( std::vector< std::pair<DetId, float> >::const_iterator diIt = v1.begin();
	       diIt != v1.end();
	       ++diIt ) {
	    mySuperCluster.addHitAndFraction(diIt->first,diIt->second);
	  } // loop over rechits      
	}      

      /*
      //Could consider adding the preshower clusters
      unsigned nps=preshowerClusterPtr_[is].size();
      for(unsigned ips=0;ips<nps;++ips)
	{
	  mySuperCluster.addPreshowerCluster(preshowerClusterPtr_[is][ips]);
	}
      */

      // Set the preshower energy
      if (doEEwithES) mySuperCluster.setPreshowerEnergy(correctedEnergyWithES-correctedEnergy);
      else mySuperCluster.setPreshowerEnergy(0.);

      // Set the cluster width
      mySuperCluster.setEtaWidth(pfwidth.pflowEtaWidth());
      mySuperCluster.setPhiWidth(pfwidth.pflowPhiWidth());
      // Force the computation of rawEnergy_ of the reco::SuperCluster
      mySuperCluster.rawEnergy();

      superClusters.push_back(mySuperCluster);
      
    }
}

void PFSuperClusterAlgo::storeSuperClusters(const edm::OrphanHandle<reco::BasicClusterCollection> & basicClustersHandle, std::auto_ptr< reco::SuperClusterCollection > & pfSuperClusters_p)
{

  //Find PU clusters lying outside Mustache area
  findClustersOutsideMustacheArea();

  //Create basic clusters and superclusters
  createBasicClusterPtrs(basicClustersHandle);
  superClusters_.clear();
  createSuperClusters(superClusters_, false);

  if (verbose_) cout << "Created "<<superClusters_.size()<< " pfSuperClusters"<<endl;

  //storing superclusters
  pfSuperClusters_p->insert(pfSuperClusters_p->end(), superClusters_.begin(), superClusters_.end());

  return;
}

void PFSuperClusterAlgo::matchSCtoESclusters(const edm::Handle<reco::PFClusterCollection> & pfclustersHandle, std::auto_ptr< reco::SuperClusterCollection > & pfSuperClustersWithES_p, boost::shared_ptr<PFEnergyCalibration> thePFEnergyCalibration_, int detector)
{

  if (verbose_) cout << "matchSCtoESclusters" << endl;


  if (detector==0) return;

  //std::vector<reco::PFClusterRef> pfESClusterAboveThresholdCollection;
  pfClusterCalibratedEnergyWithES_.clear();
  pfESClusterAboveThresholdCollection.clear();

  //Select the preshower pfclusters above thresholds
  for (uint i=0; i<pfclustersHandle->size(); i++){
    
    if ((*pfclustersHandle)[i].layer()==PFLayer::PS1 || (*pfclustersHandle)[i].layer()==PFLayer::PS2){
      
      if (verbose_) cout << "ES PFCluster i="<<i<<" energy="<<(*pfclustersHandle)[i].energy()<<endl;
      
      if ((*pfclustersHandle)[i].energy()>threshPFClusterES_){
	const reco::PFClusterRef pfESClusterAboveThreshold = reco::PFClusterRef(pfclustersHandle, i);
	pfESClusterAboveThresholdCollection.push_back(pfESClusterAboveThreshold);
      }

    }

  }


  //for each preshower pfcluster get the associated EE pfcluster if existing

  double dist = -1;
  double distmin = 1000;
  int iscsel = -1;
  int ibcsel = -1;

  //These vectors will relate the EE clusters in the SC to the ES clusters (needed for calibration)
  SCBCtoESenergyPS1 = new std::vector<double>*[pfClusters_.size()]; 
  SCBCtoESenergyPS2 = new std::vector<double>*[pfClusters_.size()];
  for (uint isc=0; isc<pfClusters_.size(); isc++) {
    SCBCtoESenergyPS1[isc] = new std::vector<double>[pfClusters_[isc].size()];
    SCBCtoESenergyPS2[isc] = new std::vector<double>[pfClusters_[isc].size()];
  }

  for (uint ies=0; ies<pfESClusterAboveThresholdCollection.size(); ies++){ //loop over the ES pfclusters above the threshold

    distmin = 1000;
    iscsel = -1;
    ibcsel = -1;

    for (uint isc=0; isc<pfClusters_.size(); isc++){ //loop over the superclusters
      for (uint ibc=0; ibc<pfClusters_[isc].size(); ibc++){ //loop over the basic clusters inside the SC

	if (pfClusters_[isc][ibc]->layer()!=PFLayer::ECAL_ENDCAP) continue;
	if (pfClusters_[isc][ibc]->eta()*pfESClusterAboveThresholdCollection[ies]->eta()<0) continue; //same side of the EE
	
	double dphi= fabs(pfClusters_[isc][ibc]->phi()-pfESClusterAboveThresholdCollection[ies]->phi()); 
	if (dphi>TMath::Pi()) dphi-= TMath::TwoPi();
	double deta=fabs(pfClusters_[isc][ibc]->eta()-pfESClusterAboveThresholdCollection[ies]->eta());
	//if (fabs(deta)>0.4 || fabs(dphi)>1.0) continue;
	if (fabs(deta)>0.3 || fabs(dphi)>0.6) continue; //geometrical matching to speed up the timing
	
	dist = LinkByRecHit::testECALAndPSByRecHit( *(pfClusters_[isc][ibc]), *(pfESClusterAboveThresholdCollection[ies]), false); //matches EE and ES cluster
      
	if (dist!=-1){
	  if (verbose_) cout << "isc="<<isc<<" ibc="<<ibc<< " ies="<<ies<< " ESenergy="<< pfESClusterAboveThresholdCollection[ies]->energy()<<" dist="<<dist<<endl;
	  
	  if (dist<distmin){
	    distmin = dist;
	    iscsel = isc;
	    ibcsel = ibc;
	  }
	}
	
      }
    }


    //Store energies of the ES clusters associated to BC of the SC
    if (iscsel!=-1 && ibcsel!=-1){
      if (verbose_) cout << "Associate ESpfcluster ies="<<ies<<" to BC "<<ibcsel<<" in SC"<<iscsel<<endl;

      if (pfESClusterAboveThresholdCollection[ies]->layer()==PFLayer::PS1) {
	SCBCtoESenergyPS1[iscsel][ibcsel].push_back(pfESClusterAboveThresholdCollection[ies]->energy());
      }
      if (pfESClusterAboveThresholdCollection[ies]->layer()==PFLayer::PS2) {
	SCBCtoESenergyPS2[iscsel][ibcsel].push_back(pfESClusterAboveThresholdCollection[ies]->energy());
      }
    }

  }


  //Compute the calibrated pfcluster energy, including EE+ES calibration. 
   for (uint isc=0; isc<pfClusters_.size(); isc++){
      for (uint ibc=0; ibc<pfClusters_[isc].size(); ibc++){
	if (pfClusters_[isc][ibc]->layer()!=PFLayer::ECAL_ENDCAP) continue;

	pfClusterCalibratedEnergyWithES_.push_back(std::vector<double>());

	const reco::PFCluster & myPFCluster (*(pfClusters_[isc][ibc]));
	double PFClusterCalibratedEnergy = thePFEnergyCalibration_->energyEm(myPFCluster,SCBCtoESenergyPS1[isc][ibc],SCBCtoESenergyPS2[isc][ibc],applyCrackCorrections_);
	if (verbose_) cout << "isc="<<isc<<" ibc="<<ibc<<" EEenergy="<<pfClusters_[isc][ibc]->energy()<<" calibEnergyWithoutES="<< pfClusterCalibratedEnergy_[isc][ibc] << " calibEnergyWithES="<<PFClusterCalibratedEnergy <<endl;
	pfClusterCalibratedEnergyWithES_[isc].push_back(PFClusterCalibratedEnergy);

	
	if (verbose_){
	  double PS1energy=0;
	  double PS2energy=0;
	  for (uint ies=0; ies<SCBCtoESenergyPS1[isc][ibc].size(); ies++) PS1energy+=SCBCtoESenergyPS1[isc][ibc][ies];
	  for (uint ies=0; ies<SCBCtoESenergyPS2[isc][ibc].size(); ies++) PS2energy+=SCBCtoESenergyPS2[isc][ibc][ies];
	  cout << "isc="<<isc<<" ibc="<<ibc<<" EEenergy="<<pfClusters_[isc][ibc]->energy()<<" PS1energy="<< PS1energy<<" PS2energy="<<PS2energy<<" calibEnergyWithoutES="<< pfClusterCalibratedEnergy_[isc][ibc] << " calibEnergyWithES="<<PFClusterCalibratedEnergy <<endl;
	}
	
      }
   }

   
   //Store EE+preshower superclusters
   if (verbose_) cout << "Store EE+preshower superclusters" << endl;
   superClusters_.clear();

   createSuperClusters(superClusters_, true);

   pfSuperClustersWithES_p->insert(pfSuperClustersWithES_p->end(), superClusters_.begin(), superClusters_.end());
   

   //Deleting objects
   for (uint isc=0; isc<pfClusters_.size(); isc++) {
     delete[] SCBCtoESenergyPS1[isc];
     delete[] SCBCtoESenergyPS2[isc];
   }
   delete SCBCtoESenergyPS1;
   delete SCBCtoESenergyPS2;

  return;
}

void PFSuperClusterAlgo::findClustersOutsideMustacheArea(){

  //Find PF cluster outside the Mustache area

  if (!doMustacheCut_) return;

  //if (verbose_) cout << "findClustersOutsideMustacheArea" << endl;

  insideMust_.clear();
  //outsideMust_.clear();

  reco::Mustache PFSCMustache;
  
  //if (verbose_) cout << "Mustache object created" << endl;

  std::vector<unsigned int> insideMustList;
  std::vector<unsigned int> outsideMustList;

  for (uint isc=0; isc<basicClusters_.size(); isc++){
    
    //if (verbose_) cout << "isc="<<isc<<endl;
 
    insideMust_.push_back(std::vector<unsigned int>());
    //outsideMust_.push_back(std::vector<unsigned int>());

    insideMustList.clear();
    outsideMustList.clear();

    //Find the pfclusters inside/outside the mustache
    PFSCMustache.MustacheClust(basicClusters_[isc],
			       insideMustList, 
			       outsideMustList);

    for (unsigned int ibc=0; ibc<basicClusters_[isc].size(); ibc++) insideMust_[isc].push_back(1);
      
    for (unsigned int iclus=0; iclus<outsideMustList.size(); iclus++) {
      if (verbose_) cout << "isc="<<isc<<" outsideMustList[iclus]="<<outsideMustList[iclus]<<" outside mustache, energy="<< basicClusters_[isc][outsideMustList[iclus]]<<endl;
      insideMust_[isc][outsideMustList[iclus]] = 0;
    }
    
  }



  return;
}
