#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoParticleFlow/PFProducer/plugins/PFElectronTranslator.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFClusterWidthAlgo.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"


PFElectronTranslator::PFElectronTranslator(const edm::ParameterSet & iConfig) {
  inputTagPFCandidates_ 
    = iConfig.getParameter<edm::InputTag>("PFCandidate");
  inputTagGSFTracks_
    = iConfig.getParameter<edm::InputTag>("GSFTracks");

  PFBasicClusterCollection_ = iConfig.getParameter<std::string>("PFBasicClusters");
  PFPreshowerClusterCollection_ = iConfig.getParameter<std::string>("PFPreshowerClusters");
  PFSuperClusterCollection_ = iConfig.getParameter<std::string>("PFSuperClusters");
  PFMVAValueMap_ = iConfig.getParameter<std::string>("ElectronMVA");
  PFSCValueMap_ = iConfig.getParameter<std::string>("ElectronSC");
  MVACut_ = (iConfig.getParameter<edm::ParameterSet>("MVACutBlock")).getParameter<double>("MVACut");

  produces<reco::BasicClusterCollection>(PFBasicClusterCollection_); 
  produces<reco::PreshowerClusterCollection>(PFPreshowerClusterCollection_); 
  produces<reco::SuperClusterCollection>(PFSuperClusterCollection_); 
  produces<edm::ValueMap<float> >(PFMVAValueMap_);
  produces<edm::ValueMap<reco::SuperClusterRef> >(PFSCValueMap_);
}

PFElectronTranslator::~PFElectronTranslator() {}

void PFElectronTranslator::beginRun(edm::Run& run,const edm::EventSetup & es) {}

void PFElectronTranslator::produce(edm::Event& iEvent,  
				    const edm::EventSetup& iSetup) { 
  
  std::auto_ptr<reco::SuperClusterCollection> 
    superClusters_p(new reco::SuperClusterCollection);

  std::auto_ptr<reco::BasicClusterCollection> 
    basicClusters_p(new reco::BasicClusterCollection);

  std::auto_ptr<reco::PreshowerClusterCollection>
    psClusters_p(new reco::PreshowerClusterCollection);
  
  std::auto_ptr<edm::ValueMap<float> > mvaMap_p(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler mvaFiller(*mvaMap_p);

  std::auto_ptr<edm::ValueMap<reco::SuperClusterRef> > 
    scMap_p(new edm::ValueMap<reco::SuperClusterRef>());
  edm::ValueMap<reco::SuperClusterRef>::Filler scRefFiller(*scMap_p);
  

  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  bool status=fetchCandidateCollection(pfCandidates, 
				       inputTagPFCandidates_, 
				       iEvent );
  
  // clear the vectors
  GsfTrackRef_.clear();
  basicClusters_.clear();
  pfClusters_.clear();
  preshowerClusters_.clear();
  superClusters_.clear();
  basicClusterPtr_.clear();
  preshowerClusterPtr_.clear();
  gsfPFCandidateIndex_.clear();
  scMap_.clear();
  gsfMvaMap_.clear();
 
  // loop on the candidates 
  //CC@@
  // we need first to create AND put the SuperCluster, 
  // basic clusters and presh clusters collection 
  // in order to get a working Handle
  unsigned ncand=(status)?pfCandidates->size():0;
  unsigned iGSF=0;
  for( unsigned i=0; i<ncand; ++i ) {

    const reco::PFCandidate& cand = (*pfCandidates)[i];    
    if(cand.particleId()!=reco::PFCandidate::e) continue; 
    if(cand.gsfTrackRef().isNull()) continue;
    // Note that -1 will still cut some total garbage candidates 
    // Fill the MVA map
    gsfMvaMap_[cand.gsfTrackRef()]=cand.mva_e_pi();	  
    if(cand.mva_e_pi()<MVACut_) continue;

    GsfTrackRef_.push_back(cand.gsfTrackRef());
    gsfPFCandidateIndex_.push_back(i);
    
    basicClusters_.push_back(reco::BasicClusterCollection());
    pfClusters_.push_back(std::vector<const reco::PFCluster *>());
    preshowerClusters_.push_back(reco::PreshowerClusterCollection());

    for(unsigned iele=0; iele<cand.elementsInBlocks().size(); ++iele) {
      // first get the block 
      reco::PFBlockRef blockRef = cand.elementsInBlocks()[iele].first;
      //
      unsigned elementIndex = cand.elementsInBlocks()[iele].second;
      // check it actually exists 
      if(blockRef.isNull()) continue;
      
      // then get the elements of the block
      const edm::OwnVector< reco::PFBlockElement >&  elements = (*blockRef).elements();
      
      const reco::PFBlockElement & pfbe (elements[elementIndex]); 
      // The first ECAL element should be the cluster associated to the GSF; defined as the seed
      if(pfbe.type()==reco::PFBlockElement::ECAL)
	{	  
	  //	  const reco::PFCandidate * coCandidate = &cand;
	  // the Brem photons are saved as daughter PFCandidate; this 
	  // is convenient to access the corrected energy
	  //	  std::cout << " Found candidate "  << correspondingDaughterCandidate(coCandidate,pfbe) << " " << coCandidate << std::endl;
	  createBasicCluster(pfbe,basicClusters_[iGSF],pfClusters_[iGSF],correspondingDaughterCandidate(cand,pfbe));
	}
      if(pfbe.type()==reco::PFBlockElement::PS1)
	{
	  createPreshowerCluster(pfbe,preshowerClusters_[iGSF],1);
	}
      if(pfbe.type()==reco::PFBlockElement::PS2)
	{
	  createPreshowerCluster(pfbe,preshowerClusters_[iGSF],2);
	}      
	  
    }   // loop on the elements

    // save the basic clusters
    basicClusters_p->insert(basicClusters_p->end(),basicClusters_[iGSF].begin(), basicClusters_[iGSF].end());
    // save the preshower clusters
    psClusters_p->insert(psClusters_p->end(),preshowerClusters_[iGSF].begin(),preshowerClusters_[iGSF].end());

    ++iGSF;
  } // loop on PFCandidates

  
   //Save the basic clusters and get an handle as to be able to create valid Refs (thanks to Claude)
  //  std::cout << " Number of basic clusters " << basicClusters_p->size() << std::endl;
  const edm::OrphanHandle<reco::BasicClusterCollection> bcRefProd = 
    iEvent.put(basicClusters_p,PFBasicClusterCollection_);

  //preshower clusters
  const edm::OrphanHandle<reco::PreshowerClusterCollection> psRefProd = 
    iEvent.put(psClusters_p,PFPreshowerClusterCollection_);

  // now that the Basic clusters are in the event, the Ref can be created
  createBasicClusterPtrs(bcRefProd);
  // now that the preshower clusters are in the event, the Ref can be created
  createPreshowerClusterPtrs(psRefProd);
  
  // and now the Super cluster can be created with valid references  
  if(status) createSuperClusters(*pfCandidates,*superClusters_p);
  
  // Let's put the super clusters in the event
  const edm::OrphanHandle<reco::SuperClusterCollection> scRefProd = iEvent.put(superClusters_p,PFSuperClusterCollection_); 
  // create the super cluster Ref
  createSuperClusterGsfMapRefs(scRefProd);
  
  
  fillMVAValueMap(iEvent,mvaFiller);
  mvaFiller.fill();

  fillSCRefValueMap(iEvent,scRefFiller);
  scRefFiller.fill();

  // MVA map
  iEvent.put(mvaMap_p,PFMVAValueMap_);
  // Gsf-SC map
  iEvent.put(scMap_p,PFSCValueMap_);
}



bool PFElectronTranslator::fetchCandidateCollection(edm::Handle<reco::PFCandidateCollection>& c, 
					      const edm::InputTag& tag, 
					      const edm::Event& iEvent) const {  
  bool found = iEvent.getByLabel(tag, c);

  if(!found)
    {
      std::ostringstream  err;
      err<<" cannot get PFCandidates: "
	 <<tag<<std::endl;
      edm::LogError("PFElectronTranslator")<<err.str();
    }
  return found;
      
}

void PFElectronTranslator::fetchGsfCollection(edm::Handle<reco::GsfTrackCollection>& c, 
					      const edm::InputTag& tag, 
					      const edm::Event& iEvent) const {  
  bool found = iEvent.getByLabel(tag, c);
  
  if(!found ) {
    std::ostringstream  err;
    err<<" cannot get GSFTracks: "
       <<tag<<std::endl;
    edm::LogError("PFElectronTranslator")<<err.str();
    throw cms::Exception( "MissingProduct", err.str());
  }  
}

// The basic cluster is a copy of the PFCluster -> the energy is not corrected 
// It should be possible to get the corrected energy (including the associated PS energy)
// from the PFCandidate daugthers ; Needs some work 
void PFElectronTranslator::createBasicCluster(const reco::PFBlockElement & PFBE, 
					      reco::BasicClusterCollection & basicClusters, 
					      std::vector<const reco::PFCluster *> & pfClusters,
					      const reco::PFCandidate & coCandidate) const
{
  reco::PFClusterRef myPFClusterRef= PFBE.clusterRef();
  if(myPFClusterRef.isNull()) return;  

  const reco::PFCluster & myPFCluster (*myPFClusterRef);
  pfClusters.push_back(&myPFCluster);
//  std::cout << " Creating BC " << myPFCluster.energy() << " " << coCandidate.ecalEnergy() <<" "<<  coCandidate.rawEcalEnergy() <<std::endl;
//  std::cout << " # hits " << myPFCluster.hitsAndFractions().size() << std::endl;

//  basicClusters.push_back(reco::CaloCluster(myPFCluster.energy(),
  basicClusters.push_back(reco::CaloCluster(coCandidate.rawEcalEnergy(),
					    myPFCluster.position(),
					    myPFCluster.caloID(),
					    myPFCluster.hitsAndFractions(),
					    myPFCluster.algo(),
					    myPFCluster.seed()));
}


void PFElectronTranslator::createPreshowerCluster(const reco::PFBlockElement & PFBE, reco::PreshowerClusterCollection& preshowerClusters,unsigned plane) const
{
  reco::PFClusterRef  myPFClusterRef= PFBE.clusterRef();
  preshowerClusters.push_back(reco::PreshowerCluster(myPFClusterRef->energy(),myPFClusterRef->position(),
					       myPFClusterRef->hitsAndFractions(),plane));
}

void PFElectronTranslator::createBasicClusterPtrs(const edm::OrphanHandle<reco::BasicClusterCollection> & basicClustersHandle )
{
  unsigned size=GsfTrackRef_.size();
  unsigned basicClusterCounter=0;
  basicClusterPtr_.resize(size);

  for(unsigned iGSF=0;iGSF<size;++iGSF) // loop on tracks
    {
      unsigned nbc=basicClusters_[iGSF].size();
      for(unsigned ibc=0;ibc<nbc;++ibc) // loop on basic clusters
	{
	  //	  std::cout <<  "Track "<< iGSF << " ref " << basicClusterCounter << std::endl;
	  reco::CaloClusterPtr bcPtr(basicClustersHandle,basicClusterCounter);
	  basicClusterPtr_[iGSF].push_back(bcPtr);
	  ++basicClusterCounter;
	}
    }
}

void PFElectronTranslator::createPreshowerClusterPtrs(const edm::OrphanHandle<reco::PreshowerClusterCollection> & preshowerClustersHandle )
{
  unsigned size=GsfTrackRef_.size();
  unsigned psClusterCounter=0;
  preshowerClusterPtr_.resize(size);

  for(unsigned iGSF=0;iGSF<size;++iGSF) // loop on tracks
    {
      unsigned nbc=preshowerClusters_[iGSF].size();
      for(unsigned ibc=0;ibc<nbc;++ibc) // loop on basic clusters
	{
	  //	  std::cout <<  "Track "<< iGSF << " ref " << basicClusterCounter << std::endl;
	  reco::CaloClusterPtr psPtr(preshowerClustersHandle,psClusterCounter);
	  preshowerClusterPtr_[iGSF].push_back(psPtr);
	  ++psClusterCounter;
	}
    }
}

void PFElectronTranslator::createSuperClusterGsfMapRefs(const edm::OrphanHandle<reco::SuperClusterCollection> & superClustersHandle )
{
  unsigned size=GsfTrackRef_.size();

  for(unsigned iGSF=0;iGSF<size;++iGSF) // loop on tracks
    {
      edm::Ref<reco::SuperClusterCollection> scRef(superClustersHandle,iGSF);
      scMap_[GsfTrackRef_[iGSF]]=scRef;
    }
}


void PFElectronTranslator::fillMVAValueMap(edm::Event& iEvent, edm::ValueMap<float>::Filler & filler) const
{
  edm::Handle<reco::GsfTrackCollection> gsfTracks;
  fetchGsfCollection(gsfTracks,
		     inputTagGSFTracks_,
		     iEvent);
  unsigned ngsf=gsfTracks->size();
  std::vector<float> values;
  for(unsigned igsf=0;igsf<ngsf;++igsf)
    {
      reco::GsfTrackRef theTrackRef(gsfTracks, igsf);
      std::map<reco::GsfTrackRef,float>::const_iterator itcheck=gsfMvaMap_.find(theTrackRef);
      if(itcheck==gsfMvaMap_.end())
	{
	  //	  edm::LogWarning("PFElectronTranslator") << "MVA Map, missing GSF track ref " << std::endl;
	  values.push_back(-99.);
	  //	  std::cout << " Push_back -99. " << std::endl;
	}
      else
	{
	  values.push_back(itcheck->second);      
	}
    }
  filler.insert(gsfTracks,values.begin(),values.end());
}


void PFElectronTranslator::fillSCRefValueMap(edm::Event& iEvent, 
					     edm::ValueMap<reco::SuperClusterRef>::Filler & filler) const
{
  edm::Handle<reco::GsfTrackCollection> gsfTracks;
  fetchGsfCollection(gsfTracks,
		     inputTagGSFTracks_,
		     iEvent);
  unsigned ngsf=gsfTracks->size();
  std::vector<reco::SuperClusterRef> values;
  for(unsigned igsf=0;igsf<ngsf;++igsf)
    {
      reco::GsfTrackRef theTrackRef(gsfTracks, igsf);
      std::map<reco::GsfTrackRef,reco::SuperClusterRef>::const_iterator itcheck=scMap_.find(theTrackRef);
      if(itcheck==scMap_.end())
	{
	  //	  edm::LogWarning("PFElectronTranslator") << "SCRef Map, missing GSF track ref" << std::endl;
	  values.push_back(reco::SuperClusterRef());
	}
      else
	{
	  values.push_back(itcheck->second);      
	}
    }
  filler.insert(gsfTracks,values.begin(),values.end());
}


void PFElectronTranslator::createSuperClusters(const reco::PFCandidateCollection & pfCand,
					       reco::SuperClusterCollection &superClusters) const
{
  unsigned nGSF=GsfTrackRef_.size();
  for(unsigned iGSF=0;iGSF<nGSF;++iGSF)
    {

      // Computes energy position a la e/gamma 
      double sclusterE=0;
      double posX=0.;
      double posY=0.;
      double posZ=0.;
      
      unsigned nbasics=basicClusters_[iGSF].size();
      for(unsigned ibc=0;ibc<nbasics;++ibc)
	{
	  double e = basicClusters_[iGSF][ibc].energy();
	  sclusterE += e;
	  posX += e * basicClusters_[iGSF][ibc].position().X();
	  posY += e * basicClusters_[iGSF][ibc].position().Y();
	  posZ += e * basicClusters_[iGSF][ibc].position().Z();	  
	}
      posX /=sclusterE;
      posY /=sclusterE;
      posZ /=sclusterE;
      
      if(pfCand[gsfPFCandidateIndex_[iGSF]].gsfTrackRef()!=GsfTrackRef_[iGSF])
	{
	  edm::LogError("PFElectronTranslator") << " Major problem in PFElectron Translator" << std::endl;
	}
      
      // compute the width
      PFClusterWidthAlgo pfwidth(pfClusters_[iGSF]);
      
      double correctedEnergy=pfCand[gsfPFCandidateIndex_[iGSF]].ecalEnergy();
      reco::SuperCluster mySuperCluster(correctedEnergy,math::XYZPoint(posX,posY,posZ));
      // protection against empty basic cluster collection ; the value is -2 in this case
      if(nbasics)
	{
//	  std::cout << "SuperCluster creation; energy " << pfCand[gsfPFCandidateIndex_[iGSF]].ecalEnergy();
//	  std::cout << " " <<   pfCand[gsfPFCandidateIndex_[iGSF]].rawEcalEnergy() << std::endl;
//	  std::cout << "Seed energy from basic " << basicClusters_[iGSF][0].energy() << std::endl;
	  mySuperCluster.setSeed(basicClusterPtr_[iGSF][0]);
	}
      else
	{
	  //	  std::cout << "SuperCluster creation ; seed energy " << 0 << std::endl;
//	  std::cout << "SuperCluster creation ; energy " << pfCand[gsfPFCandidateIndex_[iGSF]].ecalEnergy();
//	  std::cout << " " <<   pfCand[gsfPFCandidateIndex_[iGSF]].rawEcalEnergy() << std::endl;
//	  std::cout << " No seed found " << 0 << std::endl;	  
//	  std::cout << " MVA " << pfCand[gsfPFCandidateIndex_[iGSF]].mva_e_pi() << std::endl;
	  mySuperCluster.setSeed(reco::CaloClusterPtr());
	}
      // the seed should be the first basic cluster

      for(unsigned ibc=0;ibc<nbasics;++ibc)
	{
	  mySuperCluster.addCluster(basicClusterPtr_[iGSF][ibc]);
	  //	  std::cout <<"Adding Ref to SC " << basicClusterPtr_[iGSF][ibc].index() << std::endl;
	  const std::vector< std::pair<DetId, float> > & v1 = basicClusters_[iGSF][ibc].hitsAndFractions();
	  //	  std::cout << " Number of cells " << v1.size() << std::endl;
	  for( std::vector< std::pair<DetId, float> >::const_iterator diIt = v1.begin();
	       diIt != v1.end();
	       ++diIt ) {
	    //	    std::cout << " Adding DetId " << (diIt->first).rawId() << " " << diIt->second << std::endl;
	    mySuperCluster.addHitAndFraction(diIt->first,diIt->second);
	  } // loop over rechits      
	}      

      unsigned nps=preshowerClusterPtr_[iGSF].size();
      for(unsigned ips=0;ips<nps;++ips)
	{
	  mySuperCluster.addPreshowerCluster(preshowerClusterPtr_[iGSF][ips]);
	}
      

      // Set the preshower energy
      mySuperCluster.setPreshowerEnergy(pfCand[gsfPFCandidateIndex_[iGSF]].pS1Energy()+
					pfCand[gsfPFCandidateIndex_[iGSF]].pS2Energy());

      // Set the cluster width
      mySuperCluster.setEtaWidth(pfwidth.pflowEtaWidth());
      mySuperCluster.setPhiWidth(pfwidth.pflowPhiWidth());
      // Force the computation of rawEnergy_ of the reco::SuperCluster
      mySuperCluster.rawEnergy();
      superClusters.push_back(mySuperCluster);
   }
}


const reco::PFCandidate & PFElectronTranslator::correspondingDaughterCandidate(const reco::PFCandidate & cand, const reco::PFBlockElement & pfbe) const
{
  unsigned refindex=pfbe.index();
  //  std::cout << " N daughters " << cand.numberOfDaughters() << std::endl;
  reco::PFCandidate::const_iterator myDaughterCandidate=cand.begin();
  reco::PFCandidate::const_iterator itend=cand.end();

  for(;myDaughterCandidate!=itend;++myDaughterCandidate)
    {
      const reco::PFCandidate * myPFCandidate = (const reco::PFCandidate*)&*myDaughterCandidate;
      if(myPFCandidate->elementsInBlocks().size()!=1)
	{
	  //	  std::cout << " Daughter with " << myPFCandidate.elementsInBlocks().size()<< " element in block " << std::endl;
	  return cand;
	}
      if(myPFCandidate->elementsInBlocks()[0].second==refindex) 
	{
	  //	  std::cout << " Found it " << cand << std::endl;
	  return *myPFCandidate;
	}      
    }
  return cand;
}

