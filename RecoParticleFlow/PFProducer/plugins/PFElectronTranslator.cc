#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoParticleFlow/PFProducer/plugins/PFElectronTranslator.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFClusterWidthAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/Mustache.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateElectronExtra.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"


PFElectronTranslator::PFElectronTranslator(const edm::ParameterSet & iConfig) {
  inputTagPFCandidates_ 
    = iConfig.getParameter<edm::InputTag>("PFCandidate");
  inputTagPFCandidateElectrons_ 
    = iConfig.getParameter<edm::InputTag>("PFCandidateElectron");
  inputTagGSFTracks_
    = iConfig.getParameter<edm::InputTag>("GSFTracks");

  bool useIsolationValues = iConfig.getParameter<bool>("useIsolationValues") ;
  if ( useIsolationValues ) {
        if( ! iConfig.exists("isolationValues") )
                throw cms::Exception("PFElectronTranslator|InternalError")
                        <<"Missing ParameterSet isolationValues" ;
        else {
  		edm::ParameterSet isoVals  = 
			iConfig.getParameter<edm::ParameterSet> ("isolationValues");
  		inputTagIsoVals_.push_back
			(isoVals.getParameter<edm::InputTag>("pfSumChargedHadronPt"));
  		inputTagIsoVals_.push_back
			(isoVals.getParameter<edm::InputTag>("pfSumPhotonEt"));
  		inputTagIsoVals_.push_back
			(isoVals.getParameter<edm::InputTag>("pfSumNeutralHadronEt"));
  		inputTagIsoVals_.push_back
			(isoVals.getParameter<edm::InputTag>("pfSumPUPt"));
	}
  }

  PFBasicClusterCollection_ = iConfig.getParameter<std::string>("PFBasicClusters");
  PFPreshowerClusterCollection_ = iConfig.getParameter<std::string>("PFPreshowerClusters");
  PFSuperClusterCollection_ = iConfig.getParameter<std::string>("PFSuperClusters");
  GsfElectronCoreCollection_ = iConfig.getParameter<std::string>("PFGsfElectronCore");
  GsfElectronCollection_ = iConfig.getParameter<std::string>("PFGsfElectron");
  
  PFMVAValueMap_ = iConfig.getParameter<std::string>("ElectronMVA");
  PFSCValueMap_ = iConfig.getParameter<std::string>("ElectronSC");
  MVACut_ = (iConfig.getParameter<edm::ParameterSet>("MVACutBlock")).getParameter<double>("MVACut");
  checkStatusFlag_ = iConfig.getParameter<bool>("CheckStatusFlag");
  
  if (iConfig.exists("emptyIsOk")) emptyIsOk_ = iConfig.getParameter<bool>("emptyIsOk");
  else emptyIsOk_=false;

  produces<reco::BasicClusterCollection>(PFBasicClusterCollection_); 
  produces<reco::PreshowerClusterCollection>(PFPreshowerClusterCollection_); 
  produces<reco::SuperClusterCollection>(PFSuperClusterCollection_); 
  produces<reco::GsfElectronCoreCollection>(GsfElectronCoreCollection_);
  produces<reco::GsfElectronCollection>(GsfElectronCollection_);
  produces<edm::ValueMap<float> >(PFMVAValueMap_);
  produces<edm::ValueMap<reco::SuperClusterRef> >(PFSCValueMap_);
}

PFElectronTranslator::~PFElectronTranslator() {}

void PFElectronTranslator::produce(edm::Event& iEvent,  
				    const edm::EventSetup& iSetup) { 
  
  auto gsfElectronCores_p = std::make_unique<reco::GsfElectronCoreCollection>();

  auto gsfElectrons_p = std::make_unique<reco::GsfElectronCollection>();

  auto superClusters_p = std::make_unique<reco::SuperClusterCollection>();

  auto basicClusters_p = std::make_unique<reco::BasicClusterCollection>();

  auto psClusters_p = std::make_unique<reco::PreshowerClusterCollection>();
  
  auto mvaMap_p = std::make_unique<edm::ValueMap<float>>();
  edm::ValueMap<float>::Filler mvaFiller(*mvaMap_p);

  auto scMap_p = std::make_unique<edm::ValueMap<reco::SuperClusterRef>>();
  edm::ValueMap<reco::SuperClusterRef>::Filler scRefFiller(*scMap_p);
  

  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  bool status=fetchCandidateCollection(pfCandidates, 
				       inputTagPFCandidates_, 
				       iEvent );

  IsolationValueMaps isolationValues(inputTagIsoVals_.size());
  for (size_t j = 0; j<inputTagIsoVals_.size(); ++j) {
    iEvent.getByLabel(inputTagIsoVals_[j], isolationValues[j]);
  }


  // clear the vectors
  GsfTrackRef_.clear();
  CandidatePtr_.clear();
  ambiguousGsfTracks_.clear();
  kfTrackRef_.clear();
  basicClusters_.clear();
  pfClusters_.clear();
  preshowerClusters_.clear();
  superClusters_.clear();
  basicClusterPtr_.clear();
  preshowerClusterPtr_.clear();
  gsfPFCandidateIndex_.clear();
  gsfElectronCoreRefs_.clear();
  scMap_.clear();

 
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
    if(cand.mva_e_pi()<MVACut_) continue;
    
    // Check the status flag
    if(checkStatusFlag_ && !cand.electronExtraRef()->electronStatus(reco::PFCandidateElectronExtra::Selected)) {
      continue;
    }

    GsfTrackRef_.push_back(cand.gsfTrackRef());
    kfTrackRef_.push_back(cand.trackRef());
    gsfPFCandidateIndex_.push_back(i);

    reco::PFCandidatePtr ptrToPFElectron(pfCandidates,i);
    //CandidatePtr_.push_back(ptrToPFElectron->sourceCandidatePtr(0));
    CandidatePtr_.push_back(ptrToPFElectron);    

    basicClusters_.push_back(reco::BasicClusterCollection());
    pfClusters_.push_back(std::vector<const reco::PFCluster *>());
    preshowerClusters_.push_back(reco::PreshowerClusterCollection());
    ambiguousGsfTracks_.push_back(std::vector<reco::GsfTrackRef>());

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
      if(pfbe.type()==reco::PFBlockElement::GSF)
	{
	  getAmbiguousGsfTracks(pfbe,ambiguousGsfTracks_[iGSF]);
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
    iEvent.put(std::move(basicClusters_p),PFBasicClusterCollection_);

  //preshower clusters
  const edm::OrphanHandle<reco::PreshowerClusterCollection> psRefProd = 
    iEvent.put(std::move(psClusters_p),PFPreshowerClusterCollection_);

  // now that the Basic clusters are in the event, the Ref can be created
  createBasicClusterPtrs(bcRefProd);
  // now that the preshower clusters are in the event, the Ref can be created
  createPreshowerClusterPtrs(psRefProd);
  
  // and now the Super cluster can be created with valid references  
  if(status) createSuperClusters(*pfCandidates,*superClusters_p);
  
  // Let's put the super clusters in the event
  const edm::OrphanHandle<reco::SuperClusterCollection> scRefProd = iEvent.put(std::move(superClusters_p),PFSuperClusterCollection_); 
  // create the super cluster Ref
  createSuperClusterGsfMapRefs(scRefProd);

  // Now create the GsfElectronCoers
  createGsfElectronCores(*gsfElectronCores_p);
  // Put them in the as to get to be able to build a Ref
  const edm::OrphanHandle<reco::GsfElectronCoreCollection> gsfElectronCoreRefProd = 
    iEvent.put(std::move(gsfElectronCores_p),GsfElectronCoreCollection_);

  // now create the Refs 
  createGsfElectronCoreRefs(gsfElectronCoreRefProd);

  // now make the GsfElectron
  createGsfElectrons(*pfCandidates,isolationValues,*gsfElectrons_p);
  iEvent.put(std::move(gsfElectrons_p),GsfElectronCollection_);

  fillMVAValueMap(iEvent,mvaFiller);
  mvaFiller.fill();

  fillSCRefValueMap(iEvent,scRefFiller);
  scRefFiller.fill();

  // MVA map
  iEvent.put(std::move(mvaMap_p),PFMVAValueMap_);
  // Gsf-SC map
  iEvent.put(std::move(scMap_p),PFSCValueMap_);


  
}



bool PFElectronTranslator::fetchCandidateCollection(edm::Handle<reco::PFCandidateCollection>& c, 
					      const edm::InputTag& tag, 
					      const edm::Event& iEvent) const {  
  bool found = iEvent.getByLabel(tag, c);

  if(!found && !emptyIsOk_)
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
  const reco::PFClusterRef& myPFClusterRef= PFBE.clusterRef();
  if(myPFClusterRef.isNull()) return;  

  const reco::PFCluster & myPFCluster (*myPFClusterRef);
  pfClusters.push_back(&myPFCluster);
//  std::cout << " Creating BC " << myPFCluster.energy() << " " << coCandidate.ecalEnergy() <<" "<<  coCandidate.rawEcalEnergy() <<std::endl;
//  std::cout << " # hits " << myPFCluster.hitsAndFractions().size() << std::endl;

//  basicClusters.push_back(reco::CaloCluster(myPFCluster.energy(),
  basicClusters.push_back(reco::CaloCluster(
					    //	    myPFCluster.energy(),
					    coCandidate.rawEcalEnergy(),
					    myPFCluster.position(),
					    myPFCluster.caloID(),
					    myPFCluster.hitsAndFractions(),
					    myPFCluster.algo(),
					    myPFCluster.seed()));
}


void PFElectronTranslator::createPreshowerCluster(const reco::PFBlockElement & PFBE, reco::PreshowerClusterCollection& preshowerClusters,unsigned plane) const
{
  const reco::PFClusterRef&  myPFClusterRef= PFBE.clusterRef();
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


void PFElectronTranslator::fillMVAValueMap(edm::Event& iEvent, edm::ValueMap<float>::Filler & filler) 
{
  gsfMvaMap_.clear();
  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  bool status=fetchCandidateCollection(pfCandidates, 
				       inputTagPFCandidateElectrons_, 
				       iEvent );
  
  unsigned ncand=(status)?pfCandidates->size():0;
  for( unsigned i=0; i<ncand; ++i ) {
    
    const reco::PFCandidate& cand = (*pfCandidates)[i];    
    if(cand.particleId()!=reco::PFCandidate::e) continue; 
    if(cand.gsfTrackRef().isNull()) continue;
    // Fill the MVA map
    gsfMvaMap_[cand.gsfTrackRef()]=cand.mva_e_pi();	  
  }
  
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
	  //	  std::cout <<  " Value " << itcheck->second << std::endl;
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

void PFElectronTranslator::createGsfElectronCores(reco::GsfElectronCoreCollection & gsfElectronCores) const {
  unsigned nGSF=GsfTrackRef_.size();
  for(unsigned iGSF=0;iGSF<nGSF;++iGSF)
    {
      reco::GsfElectronCore myElectronCore(GsfTrackRef_[iGSF]);
      myElectronCore.setCtfTrack(kfTrackRef_[iGSF],-1.);
      std::map<reco::GsfTrackRef,reco::SuperClusterRef>::const_iterator 
	itcheck=scMap_.find(GsfTrackRef_[iGSF]);
      if(itcheck!=scMap_.end())
	myElectronCore.setParentSuperCluster(itcheck->second);
      gsfElectronCores.push_back(myElectronCore);
    }
}

void PFElectronTranslator::createGsfElectronCoreRefs(const edm::OrphanHandle<reco::GsfElectronCoreCollection> & gsfElectronCoreHandle) {
  unsigned size=GsfTrackRef_.size();
  
  for(unsigned iGSF=0;iGSF<size;++iGSF) // loop on tracks
    {
      edm::Ref<reco::GsfElectronCoreCollection> elecCoreRef(gsfElectronCoreHandle,iGSF);
      gsfElectronCoreRefs_.push_back(elecCoreRef);
    }  
}

void PFElectronTranslator::getAmbiguousGsfTracks(const reco::PFBlockElement & PFBE, std::vector<reco::GsfTrackRef>& tracks) const {
  const reco::PFBlockElementGsfTrack *  GsfEl =  dynamic_cast<const reco::PFBlockElementGsfTrack*>(&PFBE);
  if(GsfEl==nullptr) return;
  const std::vector<reco::GsfPFRecTrackRef>& ambPFRecTracks(GsfEl->GsftrackRefPF()->convBremGsfPFRecTrackRef());
  unsigned ntracks=ambPFRecTracks.size();
  for(unsigned it=0;it<ntracks;++it) {
    tracks.push_back(ambPFRecTracks[it]->gsfTrackRef());
  }
}


void PFElectronTranslator::createGsfElectrons(const reco::PFCandidateCollection & pfcand, 
					      const IsolationValueMaps& isolationValues,
					      reco::GsfElectronCollection &gsfelectrons) {
  unsigned size=GsfTrackRef_.size();
  
  for(unsigned iGSF=0;iGSF<size;++iGSF) // loop on tracks
    {
      const reco::PFCandidate& pfCandidate(pfcand[gsfPFCandidateIndex_[iGSF]]);
      // Electron
      reco::GsfElectron myElectron(gsfElectronCoreRefs_[iGSF]);
      // Warning set p4 error ! 
      myElectron.setP4(reco::GsfElectron::P4_PFLOW_COMBINATION, pfCandidate.p4(),pfCandidate.deltaP(), true);
      
      // MVA inputs
      reco::GsfElectron::MvaInput myMvaInput;
      myMvaInput.earlyBrem = pfCandidate.electronExtraRef()->mvaVariable(reco::PFCandidateElectronExtra::MVA_FirstBrem);
      myMvaInput.lateBrem = pfCandidate.electronExtraRef()->mvaVariable(reco::PFCandidateElectronExtra::MVA_LateBrem);
      myMvaInput.deltaEta = pfCandidate.electronExtraRef()->mvaVariable(reco::PFCandidateElectronExtra::MVA_DeltaEtaTrackCluster);
      myMvaInput.sigmaEtaEta = pfCandidate.electronExtraRef()->sigmaEtaEta();
      myMvaInput.hadEnergy = pfCandidate.electronExtraRef()->hadEnergy();

      // Mustache
      reco::Mustache myMustache;
      myMustache.MustacheID(*(myElectron. parentSuperCluster()), myMvaInput.nClusterOutsideMustache, myMvaInput.etOutsideMustache );

      myElectron.setMvaInput(myMvaInput);

      // MVA output
      reco::GsfElectron::MvaOutput myMvaOutput;
      myMvaOutput.status = pfCandidate.electronExtraRef()->electronStatus();
      myMvaOutput.mva_e_pi = pfCandidate.mva_e_pi();
      myElectron.setMvaOutput(myMvaOutput);
      
      // ambiguous tracks
      unsigned ntracks=ambiguousGsfTracks_[iGSF].size();
      for(unsigned it=0;it<ntracks;++it) {
	myElectron.addAmbiguousGsfTrack(ambiguousGsfTracks_[iGSF][it]);
      }

      // isolation
      if( !isolationValues.empty() ) {
      	reco::GsfElectron::PflowIsolationVariables myPFIso;
      	myPFIso.sumChargedHadronPt=(*isolationValues[0])[CandidatePtr_[iGSF]];
      	myPFIso.sumPhotonEt=(*isolationValues[1])[CandidatePtr_[iGSF]];
      	myPFIso.sumNeutralHadronEt=(*isolationValues[2])[CandidatePtr_[iGSF]];
      	myPFIso.sumPUPt=(*isolationValues[3])[CandidatePtr_[iGSF]];      
      	myElectron.setPfIsolationVariables(myPFIso);
      }

      gsfelectrons.push_back(myElectron);
    }

}

  
