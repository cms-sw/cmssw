//#include "RecoParticleFlow/PFClusterProducer/interface/PFECALBoxSuperClusterAlgo.h"
#include "RecoEcal/EgammaClusterAlgos/interface/PFECALBoxSuperClusterAlgo.h"
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

PFECALBoxSuperClusterAlgo::PFECALBoxSuperClusterAlgo()

{

}


void PFECALBoxSuperClusterAlgo::doClustering(const edm::Handle<reco::PFClusterCollection> & pfclustersHandle, std::auto_ptr< reco::BasicClusterCollection > & basicClusters_p, boost::shared_ptr<PFEnergyCalibration> thePFEnergyCalibration_, int detector){

  //Similar to hybrid e/gamma, algorithm but same in barrel and endcap
  //0- Calibrate PFCluster energies
  //1- first look for the seed pfclusters
  //2- around the seed, open the (etawidth, phiwidth) window
  //3- take all the pfclusters passing the threshold in this window and remove them from the list of available pfclusters
  //4- go to next seed if not already clustered
  
  const reco::PFClusterCollection& pfclusters = *pfclustersHandle.product();
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
    if (detector==0) cout << "PFECALBoxSuperClusterAlgo::doClustering in EB" << endl;
    if (detector==1) cout << "PFECALBoxSuperClusterAlgo::doClustering in EE" << endl;
  }

  //cleaning vectors
  pfClusters_.clear();
  basicClusters_.clear();

  basicClusterPtr_.clear();

  scPFseedIndex_.clear();
  seedCandidateIndex_.clear();
  pfClusterIndex_.clear();

  unsigned int nClusters = pfclusters.size();
  allPfClusterCalibratedEnergy_.resize(nClusters);
  allPfClusterCalibratedEnergy_.assign(nClusters, 0.0);

  pfClusterCalibratedEnergy_.clear();

  seedCandidateCollection.clear();
  pfClusterAboveThresholdCollection.clear();


  int layer;
  if (detector==0) layer = PFLayer::ECAL_BARREL;
  if (detector==1) layer = PFLayer::ECAL_ENDCAP;

  //Select PF clusters available for the clustering
  for (unsigned int i=0; i<pfclusters.size(); i++){
    const reco::PFCluster & myPFCluster = pfclusters[i];
    if (verbose_) cout << "PFCluster i="<<i<<" energy="<<myPFCluster.energy()<<endl;
    
    if (myPFCluster.layer()==layer){

      double PFClusterCalibratedEnergy = thePFEnergyCalibration_->energyEm(myPFCluster,0.0,0.0,applyCrackCorrections_);

      allPfClusterCalibratedEnergy_[i]= PFClusterCalibratedEnergy;

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
  for (unsigned int is=0; is<seedCandidateCollection.size(); is++) {
    if (verbose_) cout << "SeedPFCluster "<<is<< " energy="<<seedCandidateCollection[is]->energy()<<endl;

    for (unsigned int i=0; i<pfclusters.size(); i++){
      if (seedCandidateCollection[is].key()==i) {
	seedCandidateIndex_.push_back(i);
	if (verbose_) cout << "seedCandidateIndex_[is]="<<seedCandidateIndex_[is]<<endl;
      }
    }
  }


  //Keep pfclusters within etawidth-phiwidth window around seeds

  std::vector<int> isSeedUsed(seedCandidateCollection.size(),0);

  std::vector<int> isPFclusterUsed(pfClusterAboveThresholdCollection.size(), 0);

  std::vector<bool> isClusterized(pfclusters.size(), false);

  nSuperClusters = 0;

  //run over the seed candidates
  for (unsigned int is=0; is<seedCandidateCollection.size(); is++){

    if (verbose_) cout << "is="<<is<<" seedCandidate Energy="<<seedCandidateCollection[is]->energy() <<" eta="<< seedCandidateCollection[is]->eta()<< " phi="<< seedCandidateCollection[is]->phi()<< endl;

    if (isClusterized[seedCandidateIndex_[is]]==true) {
     if (verbose_) cout << "This seed cluster (energy="<<(*pfclustersHandle)[seedCandidateIndex_[is]].energy() <<") is already used, switching to next one"  << endl;
      continue;
    }
    isSeedUsed[is]=0;

    double seedEta = seedCandidateCollection[is]->eta();
    double seedPhi = seedCandidateCollection[is]->phi();
    //check if the pfclusters can belong to the seeded supercluster
    for (unsigned int j=0; j<pfClusterAboveThresholdCollection.size(); j++){
      
      reco::PFClusterRef myPFClusterRef = pfClusterAboveThresholdCollection[j];
      const reco::PFCluster & myPFCluster (*myPFClusterRef);
      if (isPFclusterUsed[j]==1) {
	if (verbose_) cout << "This PFCluster (energy="<<myPFCluster.energy() <<") is already used" << endl;
	continue;
      }
      
      //checking if the pfcluster is inside the eta/phi box around the seed
      if (fabs(seedEta-myPFCluster.eta())<etawidthSuperCluster_
	  && fabs(seedPhi-myPFCluster.phi())<phiwidthSuperCluster_ //FIXME wraparound
		  ){

	isSeedUsed[is]++;
	if (isSeedUsed[is]==1){
	  pfClusters_.push_back(std::vector<const reco::PFCluster *>());
	  basicClusters_.push_back(reco::BasicClusterCollection());
	  pfClusterCalibratedEnergy_.push_back(std::vector<double>());
	}

	isPFclusterUsed[j]=1;
	
	//add pfcluster to collection of basicclusters
	createBasicCluster(myPFClusterRef, basicClusters_[nSuperClusters], pfClusters_[nSuperClusters]);

	double PFClusterCalibratedEnergy = allPfClusterCalibratedEnergy_[myPFClusterRef.key()];
	  //thePFEnergyCalibration_->energyEm(myPFCluster,0.0,0.0,applyCrackCorrections_);
	pfClusterCalibratedEnergy_[nSuperClusters].push_back(PFClusterCalibratedEnergy);

	if (myPFClusterRef==seedCandidateCollection[is]) {
	  scPFseedIndex_.push_back(basicClusters_[nSuperClusters].size()-1);
	}

	if (verbose_) cout << "Use PFCluster "<<j<<" eta="<< myPFCluster.eta()
			   << "phi="<< myPFCluster.phi()
			   <<" energy="<< myPFCluster.energy()
			   <<" calibEnergy="<< pfClusterCalibratedEnergy_[nSuperClusters][basicClusters_[nSuperClusters].size()-1]<<endl;
	
	isClusterized[pfClusterIndex_[j]] = true;
	
      }
    }
    
    
    //If the seed was used, store the basic clusters
    if (isSeedUsed[is]>0) {

      if (verbose_) cout << "New supercluster, number "<<nSuperClusters<<" having "<< basicClusters_[nSuperClusters].size()<< " basicclusters"<<endl;
      if (verbose_) for (unsigned int i=0; i<basicClusters_[nSuperClusters].size(); i++) cout << "BC "<<i<<" energy="<<basicClusters_[nSuperClusters][i].energy()<<endl;

      basicClusters_p->insert(basicClusters_p->end(),basicClusters_[nSuperClusters].begin(), basicClusters_[nSuperClusters].end());

      if (verbose_) cout << "basicClusters_p filled" << endl;

      nSuperClusters++;
    }
    
  }

  if (verbose_) {
    if (detector==0) cout << "Leaving doClustering in EB (nothing more to clusterize)"<<endl;
    if (detector==1) cout << "Leaving doClustering in EE (nothing more to clusterize)"<<endl;
  }


  return;
}


void PFECALBoxSuperClusterAlgo::createBasicCluster(const reco::PFClusterRef & myPFClusterRef, 
					      reco::BasicClusterCollection & basicClusters, 
					      std::vector<const reco::PFCluster *> & pfClusters) const
{

  //cout << "Inside PFECALBoxSuperClusterAlgo::createBasicCluster"<<endl;

  if(myPFClusterRef.isNull()) return;  

  const reco::PFCluster & myPFCluster (*myPFClusterRef);
  pfClusters.push_back(&myPFCluster);



  basicClusters.push_back(myPFCluster);
  /*
    reco::CaloCluster(//coCandidate.rawEcalEnergy(),
    myPFCluster.energy(),
    myPFCluster.position(),
    myPFCluster.caloID(),
    myPFCluster.hitsAndFractions(),
    myPFCluster.algo(),
    myPFCluster.seed()));
  */

}

void PFECALBoxSuperClusterAlgo::createBasicClusterPtrs(const edm::OrphanHandle<reco::BasicClusterCollection> & basicClustersHandle )
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


void PFECALBoxSuperClusterAlgo::createSuperClusters(reco::SuperClusterCollection &superClusters, bool doEEwithES) const{

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

void PFECALBoxSuperClusterAlgo::storeSuperClusters(const edm::OrphanHandle<reco::BasicClusterCollection> & basicClustersHandle, std::auto_ptr< reco::SuperClusterCollection > & pfSuperClusters_p)
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

void PFECALBoxSuperClusterAlgo::matchSCtoESclusters(const edm::Handle<reco::PFClusterCollection> & pfclustersHandle, std::auto_ptr< reco::SuperClusterCollection > & pfSuperClustersWithES_p, boost::shared_ptr<PFEnergyCalibration> thePFEnergyCalibration_, int detector)
{

  if (verbose_) cout << "matchSCtoESclusters" << endl;


  if (detector==0) return;

  const reco::PFClusterCollection& pfclusters = *pfclustersHandle.product();
  std::vector<const reco::PFCluster*> pfESClusterAboveThresholdCollection;
  pfClusterCalibratedEnergyWithES_.clear();
  //  pfESClusterAboveThresholdCollection.clear();

  //Select the preshower pfclusters above thresholds
  typedef reco::PFClusterCollection::const_iterator PFCI;
  unsigned int i=0;
  for (PFCI cluster = pfclusters.begin(), cEnd = pfclusters.end(); cluster!=cEnd; ++cluster,++i){
    
    if (cluster->layer()==PFLayer::PS1 || cluster->layer()==PFLayer::PS2){
      
      if (verbose_) cout << "ES PFCluster i="<<i<<" energy="<<cluster->energy()<<endl;
      
      if (cluster->energy()>threshPFClusterES_){
	pfESClusterAboveThresholdCollection.push_back(&*cluster);
      }

    }

  }

  unsigned int nESaboveThreshold = pfESClusterAboveThresholdCollection.size();


  //for each preshower pfcluster get the associated EE pfcluster if existing

  double dist = -1;
  double distmin = 1000;
  int iscsel = -1;
  int ibcsel = -1;

  unsigned int nSCs = pfClusters_.size();
 
  //These vectors will relate the EE clusters in the SC to the ES clusters (needed for calibration)
  unsigned int maxSize = 0;
  for (unsigned int isc=0; isc<nSCs; isc++) {
    unsigned int iscSize = pfClusters_[isc].size();
    if (maxSize < iscSize) maxSize = iscSize;
  }

  //cache some values instead of recomputing ntimes
  std::vector<double > SCBCtoESenergyPS1(nSCs*maxSize, 0);
  std::vector<double > SCBCtoESenergyPS2(nSCs*maxSize, 0);

  std::vector<double> bcEtas(nSCs*maxSize,0);
  std::vector<double> bcPhis(nSCs*maxSize,0);
  for (unsigned int isc=0; isc<nSCs; isc++) {
    for (unsigned int ibc=0, nBCs = pfClusters_[isc].size(); ibc<nBCs; ibc++){
      unsigned int indBC = isc*maxSize + ibc;
      bcEtas[indBC] = pfClusters_[isc][ibc]->eta();
      bcPhis[indBC] = pfClusters_[isc][ibc]->phi();
    }
  }
  for (unsigned int ies=0;  ies<nESaboveThreshold; ies++){ //loop over the ES pfclusters above the threshold

    distmin = 1000;
    iscsel = -1;
    ibcsel = -1;
    
    const reco::PFCluster* pfes(pfESClusterAboveThresholdCollection[ies]);
    double pfesEta = pfes->eta();
    double pfesPhi = pfes->phi();
    for (unsigned int isc=0; isc<nSCs; isc++){ //loop over the superclusters
      for (unsigned int ibc=0, nBCs = pfClusters_[isc].size(); ibc<nBCs; ibc++){ //loop over the basic clusters inside the SC
	unsigned int indBC = isc*maxSize + ibc;
	const reco::PFCluster* bcPtr = pfClusters_[isc][ibc];

	if (bcPtr->layer()!=PFLayer::ECAL_ENDCAP) continue;
	double bcEta = bcEtas[indBC];
	double deta=fabs(bcEta-pfesEta);
	if (bcEta*pfesEta<0 || fabs(deta)>0.3) continue; //same side of the EE
	
	double bcPhi = bcPhis[indBC];
	double dphi= fabs(bcPhi-pfesPhi); 
	if (dphi>TMath::Pi()) dphi-= TMath::TwoPi();
	//if (fabs(deta)>0.4 || fabs(dphi)>1.0) continue;
	if (fabs(dphi)>0.6) continue; //geometrical matching to speed up the timing
	
	dist = LinkByRecHit::testECALAndPSByRecHit( *(bcPtr), *(pfes), false); //matches EE and ES cluster
      
	if (dist!=-1){
	  if (verbose_) cout << "isc="<<isc<<" ibc="<<ibc<< " ies="<<ies<< " ESenergy="<< pfes->energy()<<" dist="<<dist<<endl;
	  
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
      unsigned int indBCsel = iscsel*maxSize + ibcsel;
      if (pfes->layer()==PFLayer::PS1) {
	SCBCtoESenergyPS1[indBCsel]+=pfes->energy();
      }
      if (pfes->layer()==PFLayer::PS2) {
	SCBCtoESenergyPS2[indBCsel]+=pfes->energy();
      }
    }

  }


  //Compute the calibrated pfcluster energy, including EE+ES calibration. 
   for (unsigned int isc=0; isc<nSCs; isc++){
      pfClusterCalibratedEnergyWithES_.push_back(std::vector<double>());
      for (unsigned int ibc=0; ibc<pfClusters_[isc].size(); ibc++){
	const reco::PFCluster & myPFCluster (*(pfClusters_[isc][ibc]));
	if (myPFCluster.layer()!=PFLayer::ECAL_ENDCAP) continue;

	unsigned int indBC = isc*maxSize + ibc;
	double PFClusterCalibratedEnergy = 
	  thePFEnergyCalibration_->energyEm(myPFCluster,SCBCtoESenergyPS1[indBC],SCBCtoESenergyPS2[indBC],applyCrackCorrections_);
	if (verbose_) cout << "isc="<<isc<<" ibc="<<ibc<<" EEenergy="<<myPFCluster.energy()
			   <<" calibEnergyWithoutES="<< pfClusterCalibratedEnergy_[isc][ibc] 
			   << " calibEnergyWithES="<<PFClusterCalibratedEnergy <<endl;
	pfClusterCalibratedEnergyWithES_[isc].push_back(PFClusterCalibratedEnergy);
	
	if (verbose_){
	  cout << "isc="<<isc<<" ibc="<<ibc<<" EEenergy="<<myPFCluster.energy()
	       <<" PS1energy="<< SCBCtoESenergyPS1[indBC]<<" PS2energy="<<SCBCtoESenergyPS2[indBC]
	       <<" calibEnergyWithoutES="<< pfClusterCalibratedEnergy_[isc][ibc] << " calibEnergyWithES="<<PFClusterCalibratedEnergy <<endl;
	}
	
      }
   }

   
   //Store EE+preshower superclusters
   if (verbose_) cout << "Store EE+preshower superclusters" << endl;
   superClusters_.clear();

   createSuperClusters(superClusters_, true);

   pfSuperClustersWithES_p->insert(pfSuperClustersWithES_p->end(), superClusters_.begin(), superClusters_.end());
   
  return;
}

void PFECALBoxSuperClusterAlgo::findClustersOutsideMustacheArea(){

  //Find PF cluster outside the Mustache area

  if (!doMustacheCut_) return;

  //if (verbose_) cout << "findClustersOutsideMustacheArea" << endl;

  insideMust_.clear();
  //outsideMust_.clear();

  reco::Mustache PFSCMustache;
  
  //if (verbose_) cout << "Mustache object created" << endl;

  std::vector<unsigned int> insideMustList;
  std::vector<unsigned int> outsideMustList;

  for (unsigned int isc=0; isc<basicClusters_.size(); isc++){
    
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
