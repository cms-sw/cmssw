#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoParticleFlow/PFProducer/plugins/PFPhotonTranslator.h"

#include "RecoParticleFlow/PFClusterTools/interface/PFClusterWidthAlgo.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCore.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCoreFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
//#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "RecoEgamma/PhotonIdentification/interface/PhotonIsolationCalculator.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"

//#include "Geometry/Records/interface/CaloGeometryRecord.h"
//#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
//#include "Geometry/CaloTopology/interface/CaloTopology.h"
//#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
//#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
//#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
//#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
//#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
//#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
//#include "RecoEgamma/PhotonIdentification/interface/PhotonIsolationCalculator.h"
//#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h"
//#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h" 
//#include "CondFormats/EcalObjects/interface/EcalFunctionParameters.h" 
//#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidatePhotonExtra.h"

#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "RecoEcal/EgammaCoreTools/interface/Mustache.h"

#include <Math/VectorUtil.h>
#include <vector>
#include "TLorentzVector.h"
#include "TMath.h"

using namespace edm;
using namespace std;
using namespace reco;

using namespace ROOT::Math::VectorUtil;
typedef math::XYZTLorentzVector LorentzVector;
typedef math::XYZPoint Point;
typedef math::XYZVector Vector;


PFPhotonTranslator::PFPhotonTranslator(const edm::ParameterSet & iConfig) {

  //std::cout << "PFPhotonTranslator" << std::endl;

  inputTagPFCandidates_ 
    = iConfig.getParameter<edm::InputTag>("PFCandidate");
  
  edm::ParameterSet isoVals  = iConfig.getParameter<edm::ParameterSet> ("isolationValues");
  inputTagIsoVals_.push_back(isoVals.getParameter<edm::InputTag>("pfChargedHadrons"));
  inputTagIsoVals_.push_back(isoVals.getParameter<edm::InputTag>("pfPhotons"));
  inputTagIsoVals_.push_back(isoVals.getParameter<edm::InputTag>("pfNeutralHadrons"));
  

  PFBasicClusterCollection_ = iConfig.getParameter<std::string>("PFBasicClusters");
  PFPreshowerClusterCollection_ = iConfig.getParameter<std::string>("PFPreshowerClusters");
  PFSuperClusterCollection_ = iConfig.getParameter<std::string>("PFSuperClusters");
  PFConversionCollection_ = iConfig.getParameter<std::string>("PFConversionCollection");
  PFPhotonCoreCollection_ = iConfig.getParameter<std::string>("PFPhotonCores");
  PFPhotonCollection_ = iConfig.getParameter<std::string>("PFPhotons");

  EGPhotonCollection_ = iConfig.getParameter<std::string>("EGPhotons");

  vertexProducer_   = iConfig.getParameter<std::string>("primaryVertexProducer");

  barrelEcalHits_   = iConfig.getParameter<edm::InputTag>("barrelEcalHits");
  endcapEcalHits_   = iConfig.getParameter<edm::InputTag>("endcapEcalHits");

  hcalTowers_ = iConfig.getParameter<edm::InputTag>("hcalTowers");
  hOverEConeSize_   = iConfig.getParameter<double>("hOverEConeSize");

  if (iConfig.exists("emptyIsOk")) emptyIsOk_ = iConfig.getParameter<bool>("emptyIsOk");
  else emptyIsOk_=false;

  produces<reco::BasicClusterCollection>(PFBasicClusterCollection_); 
  produces<reco::PreshowerClusterCollection>(PFPreshowerClusterCollection_); 
  produces<reco::SuperClusterCollection>(PFSuperClusterCollection_); 
  produces<reco::PhotonCoreCollection>(PFPhotonCoreCollection_);
  produces<reco::PhotonCollection>(PFPhotonCollection_); 
  produces<reco::ConversionCollection>(PFConversionCollection_);
}

PFPhotonTranslator::~PFPhotonTranslator() {}

void PFPhotonTranslator::produce(edm::Event& iEvent,  
				    const edm::EventSetup& iSetup) { 

  //cout << "NEW EVENT"<<endl;

  std::auto_ptr<reco::BasicClusterCollection> 
    basicClusters_p(new reco::BasicClusterCollection);

  std::auto_ptr<reco::PreshowerClusterCollection>
    psClusters_p(new reco::PreshowerClusterCollection);

  /*
  std::auto_ptr<reco::ConversionCollection>
    SingleLeg_p(new reco::ConversionCollection);
  */

  reco::SuperClusterCollection outputSuperClusterCollection;
  reco::ConversionCollection outputOneLegConversionCollection;
  reco::PhotonCoreCollection outputPhotonCoreCollection;
  reco::PhotonCollection outputPhotonCollection;

  outputSuperClusterCollection.clear();
  outputOneLegConversionCollection.clear();
  outputPhotonCoreCollection.clear();
  outputPhotonCollection.clear();


  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  bool status=fetchCandidateCollection(pfCandidates, 
				       inputTagPFCandidates_, 
				       iEvent );

  edm::Handle<reco::PhotonCollection> egPhotons;
  iEvent.getByLabel(EGPhotonCollection_, egPhotons);
  

  Handle<reco::VertexCollection> vertexHandle;

  
  IsolationValueMaps isolationValues(inputTagIsoVals_.size());
  for (size_t j = 0; j<inputTagIsoVals_.size(); ++j) {
    iEvent.getByLabel(inputTagIsoVals_[j], isolationValues[j]);
  }
  

  // clear the vectors
  photPFCandidateIndex_.clear();
  basicClusters_.clear();
  pfClusters_.clear();
  preshowerClusters_.clear();
  superClusters_.clear();
  basicClusterPtr_.clear();
  preshowerClusterPtr_.clear();
  CandidatePtr_.clear();
  egSCRef_.clear();
  egPhotonRef_.clear();
  pfPhotonMva_.clear();
  energyRegression_.clear();
  energyRegressionError_.clear();
  pfConv_.clear();
  pfSingleLegConv_.clear();
  pfSingleLegConvMva_.clear();
  conv1legPFCandidateIndex_.clear();
  conv2legPFCandidateIndex_.clear();

  // loop on the candidates 
  //CC@@
  // we need first to create AND put the SuperCluster, 
  // basic clusters and presh clusters collection 
  // in order to get a working Handle
  unsigned ncand=(status)?pfCandidates->size():0;

  unsigned iphot=0;
  unsigned iconv1leg=0;
  unsigned iconv2leg=0;

  for( unsigned i=0; i<ncand; ++i ) {

    const reco::PFCandidate& cand = (*pfCandidates)[i];    
    if(cand.particleId()!=reco::PFCandidate::gamma) continue;
    //cout << "cand.mva_nothing_gamma()="<<cand. mva_nothing_gamma()<<endl;
    if(cand. mva_nothing_gamma()>0.001)//Found PFPhoton with PFPhoton Extras saved
      {

	//cout << "NEW PHOTON" << endl;

	//std::cout << "nDoubleLegConv="<<cand.photonExtraRef()->conversionRef().size()<<std::endl;

	if (cand.photonExtraRef()->conversionRef().size()>0){

	  pfConv_.push_back(reco::ConversionRefVector());

	  const reco::ConversionRefVector & doubleLegConvColl = cand.photonExtraRef()->conversionRef();
	  for (unsigned int iconv=0; iconv<doubleLegConvColl.size(); iconv++){
	    pfConv_[iconv2leg].push_back(doubleLegConvColl[iconv]);
	  }

	  conv2legPFCandidateIndex_.push_back(iconv2leg);
	  iconv2leg++;
	}
	else conv2legPFCandidateIndex_.push_back(-1);

	const std::vector<reco::TrackRef> & singleLegConvColl = cand.photonExtraRef()->singleLegConvTrackRef();
	const std::vector<float>& singleLegConvCollMva = cand.photonExtraRef()->singleLegConvMva();
	
	//std::cout << "nSingleLegConv=" <<singleLegConvColl.size() << std::endl;

	if (singleLegConvColl.size()>0){

	  pfSingleLegConv_.push_back(std::vector<reco::TrackRef>());
	  pfSingleLegConvMva_.push_back(std::vector<float>());

          //cout << "nTracks="<< singleLegConvColl.size()<<endl;
          for (unsigned int itk=0; itk<singleLegConvColl.size(); itk++){
            //cout << "Track pt="<< singleLegConvColl[itk]->pt() <<endl;

            pfSingleLegConv_[iconv1leg].push_back(singleLegConvColl[itk]);
            pfSingleLegConvMva_[iconv1leg].push_back(singleLegConvCollMva[itk]);
          }


	  conv1legPFCandidateIndex_.push_back(iconv1leg);
	
	  iconv1leg++;
	}
	else conv1legPFCandidateIndex_.push_back(-1);
	
      }

    photPFCandidateIndex_.push_back(i);
    pfPhotonMva_.push_back(cand.mva_nothing_gamma());
    energyRegression_.push_back(cand.photonExtraRef()->MVAGlobalCorrE());
    energyRegressionError_.push_back(cand.photonExtraRef()->MVAGlobalCorrEError());
    basicClusters_.push_back(reco::BasicClusterCollection());
    pfClusters_.push_back(std::vector<const reco::PFCluster *>());
    preshowerClusters_.push_back(reco::PreshowerClusterCollection());
    superClusters_.push_back(reco::SuperClusterCollection());

    reco::PFCandidatePtr ptrToPFPhoton(pfCandidates,i);
    CandidatePtr_.push_back(ptrToPFPhoton);  
    egSCRef_.push_back(cand.superClusterRef());
    //std::cout << "PFPhoton cand " << iphot << std::endl;

    int iegphot=0;
    for (reco::PhotonCollection::const_iterator gamIter = egPhotons->begin(); gamIter != egPhotons->end(); ++gamIter){
      if (cand.superClusterRef()==gamIter->superCluster()){
	reco::PhotonRef PhotRef(reco::PhotonRef(egPhotons, iegphot));
	egPhotonRef_.push_back(PhotRef);
      }
      iegphot++;
    }


    //std::cout << "Cand elements in blocks : " << cand.elementsInBlocks().size() << std::endl;

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

	  //std::cout << "BlockElement ECAL" << std::endl;
	  // the Brem photons are saved as daughter PFCandidate; this 
	  // is convenient to access the corrected energy
	  //	  std::cout << " Found candidate "  << correspondingDaughterCandidate(coCandidate,pfbe) << " " << coCandidate << std::endl;
	  createBasicCluster(pfbe,basicClusters_[iphot],pfClusters_[iphot],correspondingDaughterCandidate(cand,pfbe));
	}
      if(pfbe.type()==reco::PFBlockElement::PS1)
	{
	  //std::cout << "BlockElement PS1" << std::endl;
	  createPreshowerCluster(pfbe,preshowerClusters_[iphot],1);
	}
      if(pfbe.type()==reco::PFBlockElement::PS2)
	{
	  //std::cout << "BlockElement PS2" << std::endl;
	  createPreshowerCluster(pfbe,preshowerClusters_[iphot],2);
	}    


    }   // loop on the elements

        // save the basic clusters
    basicClusters_p->insert(basicClusters_p->end(),basicClusters_[iphot].begin(), basicClusters_[iphot].end());
    // save the preshower clusters
    psClusters_p->insert(psClusters_p->end(),preshowerClusters_[iphot].begin(),preshowerClusters_[iphot].end());

    ++iphot;

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
  //if(status) createSuperClusters(*pfCandidates,*superClusters_p);
  if(status) createSuperClusters(*pfCandidates,outputSuperClusterCollection);
  
  //std::cout << "nb superclusters in collection : "<<outputSuperClusterCollection.size()<<std::endl;

  // Let's put the super clusters in the event
  std::auto_ptr<reco::SuperClusterCollection> superClusters_p(new reco::SuperClusterCollection(outputSuperClusterCollection));  
  const edm::OrphanHandle<reco::SuperClusterCollection> scRefProd = iEvent.put(superClusters_p,PFSuperClusterCollection_); 


  /*
  int ipho=0;
  for (reco::SuperClusterCollection::const_iterator gamIter = scRefProd->begin(); gamIter != scRefProd->end(); ++gamIter){
    std::cout << "SC i="<<ipho<<" energy="<<gamIter->energy()<<std::endl;
    ipho++;
  }
  */


  //1-leg conversions


  if (status) createOneLegConversions(scRefProd, outputOneLegConversionCollection);


  std::auto_ptr<reco::ConversionCollection> SingleLeg_p(new reco::ConversionCollection(outputOneLegConversionCollection));  
  const edm::OrphanHandle<reco::ConversionCollection> ConvRefProd = iEvent.put(SingleLeg_p,PFConversionCollection_);
  /*
  int iconv = 0;
  for (reco::ConversionCollection::const_iterator convIter = ConvRefProd->begin(); convIter != ConvRefProd->end(); ++convIter){

    std::cout << "OneLegConv i="<<iconv<<" nTracks="<<convIter->nTracks()<<" EoverP="<<convIter->EoverP() <<std::endl;
    std::vector<edm::RefToBase<reco::Track> > convtracks = convIter->tracks();
    for (unsigned int itk=0; itk<convtracks.size(); itk++){
      std::cout << "Track pt="<< convtracks[itk]->pt() << std::endl;
    }  

    iconv++;
  }
  */

  //create photon cores
  //if(status) createPhotonCores(pfCandidates, scRefProd, *photonCores_p);
  if(status) createPhotonCores(scRefProd, ConvRefProd, outputPhotonCoreCollection);
  
  //std::cout << "nb photoncores in collection : "<<outputPhotonCoreCollection.size()<<std::endl;

  // Put the photon cores in the event
  std::auto_ptr<reco::PhotonCoreCollection> photonCores_p(new reco::PhotonCoreCollection(outputPhotonCoreCollection));  
  //std::cout << "photon core collection put in auto_ptr"<<std::endl;
  const edm::OrphanHandle<reco::PhotonCoreCollection> pcRefProd = iEvent.put(photonCores_p,PFPhotonCoreCollection_); 
  
  //std::cout << "photon core have been put in the event"<<std::endl;
  /*
  int ipho=0;
  for (reco::PhotonCoreCollection::const_iterator gamIter = pcRefProd->begin(); gamIter != pcRefProd->end(); ++gamIter){
    std::cout << "PhotonCore i="<<ipho<<" energy="<<gamIter->pfSuperCluster()->energy()<<std::endl;
    //for (unsigned int i=0; i<)

    std::cout << "PhotonCore i="<<ipho<<" nconv2leg="<<gamIter->conversions().size()<<" nconv1leg="<<gamIter->conversionsOneLeg().size()<<std::endl;

    const reco::ConversionRefVector & conv = gamIter->conversions();
    for (unsigned int iconv=0; iconv<conv.size(); iconv++){
      cout << "2-leg iconv="<<iconv<<endl;
      cout << "2-leg nTracks="<<conv[iconv]->nTracks()<<endl;
      cout << "2-leg EoverP="<<conv[iconv]->EoverP()<<endl;
      cout << "2-leg ConvAlgorithm="<<conv[iconv]->algo()<<endl;
    }

    const reco::ConversionRefVector & convbis = gamIter->conversionsOneLeg();
    for (unsigned int iconv=0; iconv<convbis.size(); iconv++){
      cout << "1-leg iconv="<<iconv<<endl;
      cout << "1-leg nTracks="<<convbis[iconv]->nTracks()<<endl;
      cout << "1-leg EoverP="<<convbis[iconv]->EoverP()<<endl;
      cout << "1-leg ConvAlgorithm="<<convbis[iconv]->algo()<<endl;
    }

    ipho++;
  }
  */

  //load vertices
  reco::VertexCollection vertexCollection;
  bool validVertex=true;
  iEvent.getByLabel(vertexProducer_, vertexHandle);
  if (!vertexHandle.isValid()) {
    edm::LogWarning("PhotonProducer") << "Error! Can't get the product primary Vertex Collection "<< "\n";
    validVertex=false;
  }
  if (validVertex) vertexCollection = *(vertexHandle.product());

  /*
  //load Ecal rechits
  bool validEcalRecHits=true;
  Handle<EcalRecHitCollection> barrelHitHandle;
  EcalRecHitCollection barrelRecHits;
  iEvent.getByLabel(barrelEcalHits_, barrelHitHandle);
  if (!barrelHitHandle.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the product "<<barrelEcalHits_.label();
    validEcalRecHits=false; 
  }
  if (  validEcalRecHits)  barrelRecHits = *(barrelHitHandle.product());
  
  Handle<EcalRecHitCollection> endcapHitHandle;
  iEvent.getByLabel(endcapEcalHits_, endcapHitHandle);
  EcalRecHitCollection endcapRecHits;
  if (!endcapHitHandle.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the product "<<endcapEcalHits_.label();
    validEcalRecHits=false; 
  }
  if( validEcalRecHits) endcapRecHits = *(endcapHitHandle.product());

  //load detector topology & geometry
  iSetup.get<CaloGeometryRecord>().get(theCaloGeom_);

  edm::ESHandle<CaloTopology> pTopology;
  iSetup.get<CaloTopologyRecord>().get(theCaloTopo_);
  const CaloTopology *topology = theCaloTopo_.product();

  // get Hcal towers collection 
  Handle<CaloTowerCollection> hcalTowersHandle;
  iEvent.getByLabel(hcalTowers_, hcalTowersHandle);
  */

  //create photon collection
  //if(status) createPhotons(vertexCollection, pcRefProd, topology, &barrelRecHits, &endcapRecHits, hcalTowersHandle, isolationValues, outputPhotonCollection);
  if(status) createPhotons(vertexCollection, egPhotons, pcRefProd, isolationValues, outputPhotonCollection);

  // Put the photons in the event
  std::auto_ptr<reco::PhotonCollection> photons_p(new reco::PhotonCollection(outputPhotonCollection));  
  //std::cout << "photon collection put in auto_ptr"<<std::endl;
  const edm::OrphanHandle<reco::PhotonCollection> photonRefProd = iEvent.put(photons_p,PFPhotonCollection_); 
  //std::cout << "photons have been put in the event"<<std::endl;
  
  /*
  ipho=0;
  for (reco::PhotonCollection::const_iterator gamIter = photonRefProd->begin(); gamIter != photonRefProd->end(); ++gamIter){
    std::cout << "Photon i="<<ipho<<" pfEnergy="<<gamIter->pfSuperCluster()->energy()<<std::endl;
    
    const reco::ConversionRefVector & conv = gamIter->conversions();
    cout << "conversions obtained : conv.size()="<< conv.size()<<endl;
    for (unsigned int iconv=0; iconv<conv.size(); iconv++){
      cout << "iconv="<<iconv<<endl;
      cout << "nTracks="<<conv[iconv]->nTracks()<<endl;
      cout << "EoverP="<<conv[iconv]->EoverP()<<endl;
      
      cout << "Vtx x="<<conv[iconv]->conversionVertex().x() << " y="<< conv[iconv]->conversionVertex().y()<<" z="<<conv[iconv]->conversionVertex().z()<< endl;
      cout << "VtxError x=" << conv[iconv]->conversionVertex().xError() << endl;
      
      std::vector<edm::RefToBase<reco::Track> > convtracks = conv[iconv]->tracks();
      //cout << "nTracks="<< convtracks.size()<<endl;
      for (unsigned int itk=0; itk<convtracks.size(); itk++){
	double convtrackpt = convtracks[itk]->pt();
	const edm::RefToBase<reco::Track> & mytrack = convtracks[itk];
	cout << "Track pt="<< convtrackpt <<endl;
	cout << "Track origin="<<gamIter->conversionTrackOrigin(mytrack)<<endl;
      }
    }
    
    //1-leg
    const reco::ConversionRefVector & convbis = gamIter->conversionsOneLeg();
    cout << "conversions obtained : conv.size()="<< convbis.size()<<endl;
    for (unsigned int iconv=0; iconv<convbis.size(); iconv++){
      cout << "iconv="<<iconv<<endl;
      cout << "nTracks="<<convbis[iconv]->nTracks()<<endl;
      cout << "EoverP="<<convbis[iconv]->EoverP()<<endl;
      
      cout << "Vtx x="<<convbis[iconv]->conversionVertex().x() << " y="<< convbis[iconv]->conversionVertex().y()<<" z="<<convbis[iconv]->conversionVertex().z()<< endl;
      cout << "VtxError x=" << convbis[iconv]->conversionVertex().xError() << endl;
       
      std::vector<edm::RefToBase<reco::Track> > convtracks = convbis[iconv]->tracks();
      //cout << "nTracks="<< convtracks.size()<<endl;
      for (unsigned int itk=0; itk<convtracks.size(); itk++){
	double convtrackpt = convtracks[itk]->pt();
	cout << "Track pt="<< convtrackpt <<endl;
	cout << "Track origin="<<gamIter->conversionTrackOrigin((convtracks[itk]))<<endl;
      }
    }
    ipho++;
  }
  */
  
}

bool PFPhotonTranslator::fetchCandidateCollection(edm::Handle<reco::PFCandidateCollection>& c, 
					      const edm::InputTag& tag, 
					      const edm::Event& iEvent) const {  
  bool found = iEvent.getByLabel(tag, c);

  if(!found && !emptyIsOk_)
    {
      std::ostringstream  err;
      err<<" cannot get PFCandidates: "
	 <<tag<<std::endl;
      edm::LogError("PFPhotonTranslator")<<err.str();
    }
  return found;
      
}

// The basic cluster is a copy of the PFCluster -> the energy is not corrected 
// It should be possible to get the corrected energy (including the associated PS energy)
// from the PFCandidate daugthers ; Needs some work 
void PFPhotonTranslator::createBasicCluster(const reco::PFBlockElement & PFBE, 
					      reco::BasicClusterCollection & basicClusters, 
					      std::vector<const reco::PFCluster *> & pfClusters,
					      const reco::PFCandidate & coCandidate) const
{
  reco::PFClusterRef myPFClusterRef = PFBE.clusterRef();
  if(myPFClusterRef.isNull()) return;  

  const reco::PFCluster & myPFCluster (*myPFClusterRef);
  pfClusters.push_back(&myPFCluster);
  //std::cout << " Creating BC " << myPFCluster.energy() << " " << coCandidate.ecalEnergy() <<" "<<  coCandidate.rawEcalEnergy() <<std::endl;
  //std::cout << " # hits " << myPFCluster.hitsAndFractions().size() << std::endl;

//  basicClusters.push_back(reco::CaloCluster(myPFCluster.energy(),
  basicClusters.push_back(reco::CaloCluster(//coCandidate.rawEcalEnergy(),
					    myPFCluster.energy(),
					    myPFCluster.position(),
					    myPFCluster.caloID(),
					    myPFCluster.hitsAndFractions(),
					    myPFCluster.algo(),
					    myPFCluster.seed()));
}

void PFPhotonTranslator::createPreshowerCluster(const reco::PFBlockElement & PFBE, reco::PreshowerClusterCollection& preshowerClusters,unsigned plane) const
{
  reco::PFClusterRef  myPFClusterRef= PFBE.clusterRef();
  preshowerClusters.push_back(reco::PreshowerCluster(myPFClusterRef->energy(),myPFClusterRef->position(),
					       myPFClusterRef->hitsAndFractions(),plane));
}

void PFPhotonTranslator::createBasicClusterPtrs(const edm::OrphanHandle<reco::BasicClusterCollection> & basicClustersHandle )
{
  unsigned size=photPFCandidateIndex_.size();
  unsigned basicClusterCounter=0;
  basicClusterPtr_.resize(size);

  for(unsigned iphot=0;iphot<size;++iphot) // loop on tracks
    {
      unsigned nbc=basicClusters_[iphot].size();
      for(unsigned ibc=0;ibc<nbc;++ibc) // loop on basic clusters
	{
	  //	  std::cout <<  "Track "<< iGSF << " ref " << basicClusterCounter << std::endl;
	  reco::CaloClusterPtr bcPtr(basicClustersHandle,basicClusterCounter);
	  basicClusterPtr_[iphot].push_back(bcPtr);
	  ++basicClusterCounter;
	}
    }
}

void PFPhotonTranslator::createPreshowerClusterPtrs(const edm::OrphanHandle<reco::PreshowerClusterCollection> & preshowerClustersHandle )
{
  unsigned size=photPFCandidateIndex_.size();
  unsigned psClusterCounter=0;
  preshowerClusterPtr_.resize(size);

  for(unsigned iphot=0;iphot<size;++iphot) // loop on tracks
    {
      unsigned nbc=preshowerClusters_[iphot].size();
      for(unsigned ibc=0;ibc<nbc;++ibc) // loop on basic clusters
	{
	  //	  std::cout <<  "Track "<< iGSF << " ref " << basicClusterCounter << std::endl;
	  reco::CaloClusterPtr psPtr(preshowerClustersHandle,psClusterCounter);
	  preshowerClusterPtr_[iphot].push_back(psPtr);
	  ++psClusterCounter;
	}
    }
}

void PFPhotonTranslator::createSuperClusters(const reco::PFCandidateCollection & pfCand,
					       reco::SuperClusterCollection &superClusters) const
{
  unsigned nphot=photPFCandidateIndex_.size();
  for(unsigned iphot=0;iphot<nphot;++iphot)
    {

      //cout << "SC iphot=" << iphot << endl;

      // Computes energy position a la e/gamma 
      double sclusterE=0;
      double posX=0.;
      double posY=0.;
      double posZ=0.;
      
      unsigned nbasics=basicClusters_[iphot].size();
      for(unsigned ibc=0;ibc<nbasics;++ibc)
	{
	  //cout << "BC in SC : iphot="<<iphot<<endl;
	  
	  double e = basicClusters_[iphot][ibc].energy();
	  sclusterE += e;
	  posX += e * basicClusters_[iphot][ibc].position().X();
	  posY += e * basicClusters_[iphot][ibc].position().Y();
	  posZ += e * basicClusters_[iphot][ibc].position().Z();	  
	}
      posX /=sclusterE;
      posY /=sclusterE;
      posZ /=sclusterE;
      
      /*
      if(pfCand[gsfPFCandidateIndex_[iphot]].gsfTrackRef()!=GsfTrackRef_[iphot])
	{
	  edm::LogError("PFElectronTranslator") << " Major problem in PFElectron Translator" << std::endl;
	}
      */      

      // compute the width
      PFClusterWidthAlgo pfwidth(pfClusters_[iphot]);
      
      double correctedEnergy=pfCand[photPFCandidateIndex_[iphot]].ecalEnergy();
      reco::SuperCluster mySuperCluster(correctedEnergy,math::XYZPoint(posX,posY,posZ));
      // protection against empty basic cluster collection ; the value is -2 in this case
      if(nbasics)
	{
//	  std::cout << "SuperCluster creation; energy " << pfCand[gsfPFCandidateIndex_[iphot]].ecalEnergy();
//	  std::cout << " " <<   pfCand[gsfPFCandidateIndex_[iphot]].rawEcalEnergy() << std::endl;
//	  std::cout << "Seed energy from basic " << basicClusters_[iphot][0].energy() << std::endl;
	  mySuperCluster.setSeed(basicClusterPtr_[iphot][0]);
	}
      else
	{
	  //	  std::cout << "SuperCluster creation ; seed energy " << 0 << std::endl;
	  //std::cout << "SuperCluster creation ; energy " << pfCand[photPFCandidateIndex_[iphot]].ecalEnergy();
	  //std::cout << " " <<   pfCand[photPFCandidateIndex_[iphot]].rawEcalEnergy() << std::endl;
//	  std::cout << " No seed found " << 0 << std::endl;	  
//	  std::cout << " MVA " << pfCand[gsfPFCandidateIndex_[iphot]].mva_e_pi() << std::endl;
	  mySuperCluster.setSeed(reco::CaloClusterPtr());
	}
      // the seed should be the first basic cluster

      for(unsigned ibc=0;ibc<nbasics;++ibc)
	{
	  mySuperCluster.addCluster(basicClusterPtr_[iphot][ibc]);
	  //	  std::cout <<"Adding Ref to SC " << basicClusterPtr_[iphot][ibc].index() << std::endl;
	  const std::vector< std::pair<DetId, float> > & v1 = basicClusters_[iphot][ibc].hitsAndFractions();
	  //	  std::cout << " Number of cells " << v1.size() << std::endl;
	  for( std::vector< std::pair<DetId, float> >::const_iterator diIt = v1.begin();
	       diIt != v1.end();
	       ++diIt ) {
	    //	    std::cout << " Adding DetId " << (diIt->first).rawId() << " " << diIt->second << std::endl;
	    mySuperCluster.addHitAndFraction(diIt->first,diIt->second);
	  } // loop over rechits      
	}      

      unsigned nps=preshowerClusterPtr_[iphot].size();
      for(unsigned ips=0;ips<nps;++ips)
	{
	  mySuperCluster.addPreshowerCluster(preshowerClusterPtr_[iphot][ips]);
	}
      

      // Set the preshower energy
      mySuperCluster.setPreshowerEnergy(pfCand[photPFCandidateIndex_[iphot]].pS1Energy()+
					pfCand[photPFCandidateIndex_[iphot]].pS2Energy());

      // Set the cluster width
      mySuperCluster.setEtaWidth(pfwidth.pflowEtaWidth());
      mySuperCluster.setPhiWidth(pfwidth.pflowPhiWidth());
      // Force the computation of rawEnergy_ of the reco::SuperCluster
      mySuperCluster.rawEnergy();

      //cout << "SC energy="<< mySuperCluster.energy()<<endl;

      superClusters.push_back(mySuperCluster);
      //std::cout << "nb super clusters in collection : "<<superClusters.size()<<std::endl;
    }
}

void PFPhotonTranslator::createOneLegConversions(const edm::OrphanHandle<reco::SuperClusterCollection> & superClustersHandle, reco::ConversionCollection &oneLegConversions)
{

  //std::cout << "createOneLegConversions" << std::endl;

    unsigned nphot=photPFCandidateIndex_.size();
    for(unsigned iphot=0;iphot<nphot;++iphot)
      {

	//if (conv1legPFCandidateIndex_[iphot]==-1) cout << "No OneLegConversions to add"<<endl;
	//else std::cout << "Phot "<<iphot<< " nOneLegConversions to add : "<<pfSingleLegConv_[conv1legPFCandidateIndex_[iphot]].size()<<endl;


	if (conv1legPFCandidateIndex_[iphot]>-1){

	  for (unsigned iConv=0; iConv<pfSingleLegConv_[conv1legPFCandidateIndex_[iphot]].size(); iConv++){

	    reco::CaloClusterPtrVector scPtrVec;
	    std::vector<reco::CaloClusterPtr>matchingBC;
	    math::Error<3>::type error;
	    const reco::Vertex  * convVtx = new reco::Vertex(pfSingleLegConv_[conv1legPFCandidateIndex_[iphot]][iConv]->innerPosition(), error);
	    
	    //cout << "Vtx x="<<convVtx->x() << " y="<< convVtx->y()<<" z="<<convVtx->z()<< endl;
	    //cout << "VtxError x=" << convVtx->xError() << endl;

	    std::vector<reco::TrackRef> OneLegConvVector;
	    OneLegConvVector.push_back(pfSingleLegConv_[conv1legPFCandidateIndex_[iphot]][iConv]);
	    
	    reco::CaloClusterPtrVector clu=scPtrVec;
	    std::vector<reco::TrackRef> tr=OneLegConvVector;
	    std::vector<math::XYZPointF>trackPositionAtEcalVec;
	    std::vector<math::XYZPointF>innPointVec;
	    std::vector<math::XYZVectorF>trackPinVec;
	    std::vector<math::XYZVectorF>trackPoutVec;
	    math::XYZPointF trackPositionAtEcal(pfSingleLegConv_[conv1legPFCandidateIndex_[iphot]][iConv]->
						outerPosition().X(), 
						pfSingleLegConv_[conv1legPFCandidateIndex_[iphot]][iConv]->
						outerPosition().Y(),
						pfSingleLegConv_[conv1legPFCandidateIndex_[iphot]][iConv]->
						outerPosition().Z());
	    math::XYZPointF innPoint(pfSingleLegConv_[conv1legPFCandidateIndex_[iphot]][iConv]->
				     innerPosition().X(), 
				     pfSingleLegConv_[conv1legPFCandidateIndex_[iphot]][iConv]->
				     innerPosition().Y(),
				     pfSingleLegConv_[conv1legPFCandidateIndex_[iphot]][iConv]->
				     innerPosition().Z());
	    math::XYZVectorF trackPin(pfSingleLegConv_[conv1legPFCandidateIndex_[iphot]][iConv]->
				     innerMomentum().X(), 
				     pfSingleLegConv_[conv1legPFCandidateIndex_[iphot]][iConv]->
				     innerMomentum().Y(),
				     pfSingleLegConv_[conv1legPFCandidateIndex_[iphot]][iConv]->
				     innerMomentum().Z());
	    math::XYZVectorF trackPout(pfSingleLegConv_[conv1legPFCandidateIndex_[iphot]][iConv]->
				      outerMomentum().X(), 
				      pfSingleLegConv_[conv1legPFCandidateIndex_[iphot]][iConv]->
				      outerMomentum().Y(),
				      pfSingleLegConv_[conv1legPFCandidateIndex_[iphot]][iConv]->
				      outerMomentum().Z());
	    float DCA=pfSingleLegConv_[conv1legPFCandidateIndex_[iphot]][iConv]->d0();
	    trackPositionAtEcalVec.push_back(trackPositionAtEcal);
	    innPointVec.push_back(innPoint);
	    trackPinVec.push_back(trackPin);
	    trackPoutVec.push_back(trackPout);
	    std::vector< float > OneLegMvaVector;
	    reco::Conversion myOneLegConversion(scPtrVec, 
						OneLegConvVector,
						trackPositionAtEcalVec,
						*convVtx,
						matchingBC,
						DCA,
						innPointVec,
						trackPinVec,
						trackPoutVec,
						pfSingleLegConvMva_[conv1legPFCandidateIndex_[iphot]][iConv],			  
						reco::Conversion::pflow);
	    OneLegMvaVector.push_back(pfSingleLegConvMva_[conv1legPFCandidateIndex_[iphot]][iConv]);
	    myOneLegConversion.setOneLegMVA(OneLegMvaVector);
	    //reco::Conversion myOneLegConversion(scPtrVec, 
	    //OneLegConvVector, *convVtx, reco::Conversion::pflow);
	    
	    /*
	    std::cout << "One leg conversion created" << endl;
	    std::vector<edm::RefToBase<reco::Track> > convtracks = myOneLegConversion.tracks();
	    const std::vector<float> mvalist = myOneLegConversion.oneLegMVA();

	    cout << "nTracks="<< convtracks.size()<<endl;
	    for (unsigned int itk=0; itk<convtracks.size(); itk++){
	      //double convtrackpt = convtracks[itk]->pt();
	      std::cout << "Track pt="<< convtracks[itk]->pt() << std::endl;
	      std::cout << "Track mva="<< mvalist[itk] << std::endl;
	    }   
	    */
	    oneLegConversions.push_back(myOneLegConversion);
	    
	    //cout << "OneLegConv added"<<endl;
	    
	  }
	}
      }
}


void PFPhotonTranslator::createPhotonCores(const edm::OrphanHandle<reco::SuperClusterCollection> & superClustersHandle, const edm::OrphanHandle<reco::ConversionCollection> & oneLegConversionHandle, reco::PhotonCoreCollection &photonCores)
{
  
  //std::cout << "createPhotonCores" << std::endl;

  unsigned nphot=photPFCandidateIndex_.size();

  unsigned i1legtot = 0;

  for(unsigned iphot=0;iphot<nphot;++iphot)
    {
      //std::cout << "iphot="<<iphot<<std::endl;

      reco::PhotonCore myPhotonCore;

      reco::SuperClusterRef SCref(reco::SuperClusterRef(superClustersHandle, iphot));
      
      myPhotonCore.setPFlowPhoton(true);
      myPhotonCore.setStandardPhoton(false);
      myPhotonCore.setPflowSuperCluster(SCref);
      myPhotonCore.setSuperCluster(egSCRef_[iphot]);

      reco::ElectronSeedRefVector pixelSeeds = egPhotonRef_[iphot]->electronPixelSeeds();
      for (unsigned iseed=0; iseed<pixelSeeds.size(); iseed++){
	myPhotonCore.addElectronPixelSeed(pixelSeeds[iseed]);
      }

      //cout << "PhotonCores : SC OK" << endl;

      //cout << "conv1legPFCandidateIndex_[iphot]="<<conv1legPFCandidateIndex_[iphot]<<endl;
      //cout << "conv2legPFCandidateIndex_[iphot]="<<conv2legPFCandidateIndex_[iphot]<<endl;

      if (conv1legPFCandidateIndex_[iphot]>-1){
	for (unsigned int iConv=0; iConv<pfSingleLegConv_[conv1legPFCandidateIndex_[iphot]].size(); iConv++){
	  
	  const reco::ConversionRef & OneLegRef(reco::ConversionRef(oneLegConversionHandle, i1legtot));
	  myPhotonCore.addOneLegConversion(OneLegRef);
	  
	  //cout << "PhotonCores : 1-leg OK" << endl;
	  /*
	  cout << "Testing 1-leg :"<<endl;
	  const reco::ConversionRefVector & conv = myPhotonCore.conversionsOneLeg();
	  for (unsigned int iconv=0; iconv<conv.size(); iconv++){
	    cout << "Testing 1-leg : iconv="<<iconv<<endl;
	    cout << "Testing 1-leg : nTracks="<<conv[iconv]->nTracks()<<endl;
	    cout << "Testing 1-leg : EoverP="<<conv[iconv]->EoverP()<<endl;
	    std::vector<edm::RefToBase<reco::Track> > convtracks = conv[iconv]->tracks();
	    for (unsigned int itk=0; itk<convtracks.size(); itk++){
	      //double convtrackpt = convtracks[itk]->pt();
	      std::cout << "Testing 1-leg : Track pt="<< convtracks[itk]->pt() << std::endl;
	      // std::cout << "Track mva="<< mvalist[itk] << std::endl;
	    }  
	  }
	  */

	  i1legtot++;
	}
      }

      if (conv2legPFCandidateIndex_[iphot]>-1){
	for(unsigned int iConv=0; iConv<pfConv_[conv2legPFCandidateIndex_[iphot]].size(); iConv++) {

	  const reco::ConversionRef & TwoLegRef(pfConv_[conv2legPFCandidateIndex_[iphot]][iConv]);
	  myPhotonCore.addConversion(TwoLegRef);

	}
	//cout << "PhotonCores : 2-leg OK" << endl;

	/*
	cout << "Testing 2-leg :"<<endl;
	const reco::ConversionRefVector & conv = myPhotonCore.conversions();
	for (unsigned int iconv=0; iconv<conv.size(); iconv++){
	  cout << "Testing 2-leg : iconv="<<iconv<<endl;
	  cout << "Testing 2-leg : nTracks="<<conv[iconv]->nTracks()<<endl;
	  cout << "Testing 2-leg : EoverP="<<conv[iconv]->EoverP()<<endl;
	  std::vector<edm::RefToBase<reco::Track> > convtracks = conv[iconv]->tracks();
	  for (unsigned int itk=0; itk<convtracks.size(); itk++){
	    //double convtrackpt = convtracks[itk]->pt();
	    std::cout << "Testing 2-leg : Track pt="<< convtracks[itk]->pt() << std::endl;
	    // std::cout << "Track mva="<< mvalist[itk] << std::endl;
	  }  
	}
	*/
      }

      photonCores.push_back(myPhotonCore);
      
    }

  //std::cout << "end of createPhotonCores"<<std::endl;
}


void PFPhotonTranslator::createPhotons(reco::VertexCollection &vertexCollection, edm::Handle<reco::PhotonCollection> &egPhotons, const edm::OrphanHandle<reco::PhotonCoreCollection> & photonCoresHandle, const IsolationValueMaps& isolationValues, reco::PhotonCollection &photons) 
{

  //cout << "createPhotons" << endl;
  
  unsigned nphot=photPFCandidateIndex_.size();

  for(unsigned iphot=0;iphot<nphot;++iphot)
    {
      //std::cout << "iphot="<<iphot<<std::endl;

      reco::PhotonCoreRef PCref(reco::PhotonCoreRef(photonCoresHandle, iphot));

      math::XYZPoint vtx(0.,0.,0.);
      if (vertexCollection.size()>0) vtx = vertexCollection.begin()->position();
      //std::cout << "vtx made" << std::endl;

      math::XYZVector direction =  PCref->pfSuperCluster()->position() - vtx;

      //It could be that pfSC energy gives not the best resolution : use smaller agregates for some cases ?
      math::XYZVector P3 = direction.unit() * PCref->pfSuperCluster()->energy();
      LorentzVector P4(P3.x(), P3.y(), P3.z(), PCref->pfSuperCluster()->energy());

      reco::Photon myPhoton(P4, PCref->pfSuperCluster()->position(), PCref, vtx);
      //cout << "photon created"<<endl;



      reco::Photon::ShowerShape  showerShape;
      reco::Photon::FiducialFlags fiducialFlags;
      reco::Photon::IsolationVariables isolationVariables03;
      reco::Photon::IsolationVariables isolationVariables04;

      showerShape.e1x5= egPhotonRef_[iphot]->e1x5();
      showerShape.e2x5= egPhotonRef_[iphot]->e2x5();
      showerShape.e3x3= egPhotonRef_[iphot]->e3x3();
      showerShape.e5x5= egPhotonRef_[iphot]->e5x5();
      showerShape.maxEnergyXtal =  egPhotonRef_[iphot]->maxEnergyXtal();
      showerShape.sigmaEtaEta =    egPhotonRef_[iphot]->sigmaEtaEta();
      showerShape.sigmaIetaIeta =  egPhotonRef_[iphot]->sigmaIetaIeta();
      showerShape.hcalDepth1OverEcal = egPhotonRef_[iphot]->hadronicDepth1OverEm();
      showerShape.hcalDepth2OverEcal = egPhotonRef_[iphot]->hadronicDepth2OverEm();
      myPhoton.setShowerShapeVariables ( showerShape ); 
	  
      fiducialFlags.isEB = egPhotonRef_[iphot]->isEB();
      fiducialFlags.isEE = egPhotonRef_[iphot]->isEE();
      fiducialFlags.isEBEtaGap = egPhotonRef_[iphot]->isEBEtaGap();
      fiducialFlags.isEBPhiGap = egPhotonRef_[iphot]->isEBPhiGap();
      fiducialFlags.isEERingGap = egPhotonRef_[iphot]->isEERingGap();
      fiducialFlags.isEEDeeGap = egPhotonRef_[iphot]->isEEDeeGap();
      fiducialFlags.isEBEEGap = egPhotonRef_[iphot]->isEBEEGap();
      myPhoton.setFiducialVolumeFlags ( fiducialFlags );

      isolationVariables03.ecalRecHitSumEt = egPhotonRef_[iphot]->ecalRecHitSumEtConeDR03();
      isolationVariables03.hcalTowerSumEt = egPhotonRef_[iphot]->hcalTowerSumEtConeDR03();
      isolationVariables03.hcalDepth1TowerSumEt = egPhotonRef_[iphot]->hcalDepth1TowerSumEtConeDR03();
      isolationVariables03.hcalDepth2TowerSumEt = egPhotonRef_[iphot]->hcalDepth2TowerSumEtConeDR03();
      isolationVariables03.trkSumPtSolidCone = egPhotonRef_[iphot]->trkSumPtSolidConeDR03();
      isolationVariables03.trkSumPtHollowCone = egPhotonRef_[iphot]->trkSumPtHollowConeDR03();
      isolationVariables03.nTrkSolidCone = egPhotonRef_[iphot]->nTrkSolidConeDR03();
      isolationVariables03.nTrkHollowCone = egPhotonRef_[iphot]->nTrkHollowConeDR03();
      isolationVariables04.ecalRecHitSumEt = egPhotonRef_[iphot]->ecalRecHitSumEtConeDR04();
      isolationVariables04.hcalTowerSumEt = egPhotonRef_[iphot]->hcalTowerSumEtConeDR04();
      isolationVariables04.hcalDepth1TowerSumEt = egPhotonRef_[iphot]->hcalDepth1TowerSumEtConeDR04();
      isolationVariables04.hcalDepth2TowerSumEt = egPhotonRef_[iphot]->hcalDepth2TowerSumEtConeDR04();
      isolationVariables04.trkSumPtSolidCone = egPhotonRef_[iphot]->trkSumPtSolidConeDR04();
      isolationVariables04.trkSumPtHollowCone = egPhotonRef_[iphot]->trkSumPtHollowConeDR04();
      isolationVariables04.nTrkSolidCone = egPhotonRef_[iphot]->nTrkSolidConeDR04();
      isolationVariables04.nTrkHollowCone = egPhotonRef_[iphot]->nTrkHollowConeDR04();
      myPhoton.setIsolationVariables(isolationVariables04, isolationVariables03);
     
	  
      

      reco::Photon::PflowIsolationVariables myPFIso;
      myPFIso.chargedHadronIso=(*isolationValues[0])[CandidatePtr_[iphot]];
      myPFIso.photonIso=(*isolationValues[1])[CandidatePtr_[iphot]];
      myPFIso.neutralHadronIso=(*isolationValues[2])[CandidatePtr_[iphot]];   
      myPhoton.setPflowIsolationVariables(myPFIso);
      
      reco::Photon::PflowIDVariables myPFVariables;

      reco::Mustache myMustache;
      myMustache.MustacheID(*(myPhoton.pfSuperCluster()), myPFVariables.nClusterOutsideMustache, myPFVariables.etOutsideMustache );
      myPFVariables.mva = pfPhotonMva_[iphot];
      myPhoton.setPflowIDVariables(myPFVariables);

      //cout << "chargedHadronIso="<<myPhoton.chargedHadronIso()<<" photonIso="<<myPhoton.photonIso()<<" neutralHadronIso="<<myPhoton.neutralHadronIso()<<endl;
      
      // set PF-regression energy
      myPhoton.setCorrectedEnergy(reco::Photon::regression2,energyRegression_[iphot],energyRegressionError_[iphot],false);
      

      /*
      if (basicClusters_[iphot].size()>0){
      // Cluster shape variables
      //Algorithms from EcalClusterTools could be adapted to PF photons ? (not based on 5x5 BC)
      //It happens that energy computed in eg e5x5 is greater than pfSC energy (EcalClusterTools gathering energies from adjacent crystals even if not belonging to the SC)
      const EcalRecHitCollection* hits = 0 ;
      int subdet = PCref->pfSuperCluster()->seed()->hitsAndFractions()[0].first.subdetId();
      if (subdet==EcalBarrel) hits = barrelRecHits;
      else if  (subdet==EcalEndcap) hits = endcapRecHits;
      const CaloGeometry* geometry = theCaloGeom_.product();

      float maxXtal =   EcalClusterTools::eMax( *(PCref->pfSuperCluster()->seed()), &(*hits) );
      //cout << "maxXtal="<<maxXtal<<endl;
      float e1x5    =   EcalClusterTools::e1x5(  *(PCref->pfSuperCluster()->seed()), &(*hits), &(*topology)); 
      //cout << "e1x5="<<e1x5<<endl;
      float e2x5    =   EcalClusterTools::e2x5Max(  *(PCref->pfSuperCluster()->seed()), &(*hits), &(*topology)); 
      //cout << "e2x5="<<e2x5<<endl;
      float e3x3    =   EcalClusterTools::e3x3(  *(PCref->pfSuperCluster()->seed()), &(*hits), &(*topology)); 
      //cout << "e3x3="<<e3x3<<endl;
      float e5x5    =   EcalClusterTools::e5x5( *(PCref->pfSuperCluster()->seed()), &(*hits), &(*topology)); 
      //cout << "e5x5="<<e5x5<<endl;
      std::vector<float> cov =  EcalClusterTools::covariances( *(PCref->pfSuperCluster()->seed()), &(*hits), &(*topology), geometry); 
      float sigmaEtaEta = sqrt(cov[0]);
      //cout << "sigmaEtaEta="<<sigmaEtaEta<<endl;
      std::vector<float> locCov =  EcalClusterTools::localCovariances( *(PCref->pfSuperCluster()->seed()), &(*hits), &(*topology)); 
      float sigmaIetaIeta = sqrt(locCov[0]);
      //cout << "sigmaIetaIeta="<<sigmaIetaIeta<<endl;
      //float r9 =e3x3/(PCref->pfSuperCluster()->rawEnergy());


      // calculate HoE
      const CaloTowerCollection* hcalTowersColl = hcalTowersHandle.product();
      EgammaTowerIsolation towerIso1(hOverEConeSize_,0.,0.,1,hcalTowersColl) ;  
      EgammaTowerIsolation towerIso2(hOverEConeSize_,0.,0.,2,hcalTowersColl) ;  
      double HoE1=towerIso1.getTowerESum(&(*PCref->pfSuperCluster()))/PCref->pfSuperCluster()->energy();
      double HoE2=towerIso2.getTowerESum(&(*PCref->pfSuperCluster()))/PCref->pfSuperCluster()->energy(); 
      //cout << "HoE1="<<HoE1<<endl;
      //cout << "HoE2="<<HoE2<<endl;  

      reco::Photon::ShowerShape  showerShape;
      showerShape.e1x5= e1x5;
      showerShape.e2x5= e2x5;
      showerShape.e3x3= e3x3;
      showerShape.e5x5= e5x5;
      showerShape.maxEnergyXtal =  maxXtal;
      showerShape.sigmaEtaEta =    sigmaEtaEta;
      showerShape.sigmaIetaIeta =  sigmaIetaIeta;
      showerShape.hcalDepth1OverEcal = HoE1;
      showerShape.hcalDepth2OverEcal = HoE2;
      myPhoton.setShowerShapeVariables ( showerShape ); 
      //cout << "shower shape variables filled"<<endl;
      }
      */
      

      photons.push_back(myPhoton);

    }

  //std::cout << "end of createPhotons"<<std::endl;
}


const reco::PFCandidate & PFPhotonTranslator::correspondingDaughterCandidate(const reco::PFCandidate & cand, const reco::PFBlockElement & pfbe) const
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

