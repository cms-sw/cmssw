#include <iostream>
#include <vector>
#include <memory>
/** \class GEDPhotonCoreProducer
 **  
 **
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/GEDPhotonCoreProducer.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtraFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include <Math/VectorUtil.h>
#include <vector>
#include "TLorentzVector.h"
#include "TMath.h"

GEDPhotonCoreProducer::GEDPhotonCoreProducer(const edm::ParameterSet& config) : 
  conf_(config)

{

  // use onfiguration file to setup input/output collection names
  pfEgammaCandidates_ = 
    consumes<reco::PFCandidateCollection>(conf_.getParameter<edm::InputTag>("pfEgammaCandidates"));
  pixelSeedProducer_ = 
    consumes<reco::ElectronSeedCollection>(conf_.getParameter<edm::InputTag>("pixelSeedProducer"));

  GEDPhotonCoreCollection_ = conf_.getParameter<std::string>("gedPhotonCoreCollection");



  // Register the product
  produces<reco::PhotonCoreCollection>(GEDPhotonCoreCollection_);
  produces<reco::ConversionCollection>(PFConversionCollection_);

}

GEDPhotonCoreProducer::~GEDPhotonCoreProducer() {}




void GEDPhotonCoreProducer::produce(edm::Event &theEvent, const edm::EventSetup& theEventSetup) {


  using namespace edm;
  //  nEvt_++;

  reco::PhotonCoreCollection outputPhotonCoreCollection;
  std::auto_ptr< reco::PhotonCoreCollection > outputPhotonCoreCollection_p(new reco::PhotonCoreCollection);

  reco::ConversionCollection outputOneLegConversionCollection;
  std::auto_ptr<reco::ConversionCollection> SingleLeg_p(new reco::ConversionCollection(outputOneLegConversionCollection));  

  // Get the  PF refined cluster  collection
  Handle<reco::PFCandidateCollection> pfCandidateHandle;
  theEvent.getByToken(pfEgammaCandidates_,pfCandidateHandle);
  if (!pfCandidateHandle.isValid()) {
    edm::LogError("GEDPhotonCoreProducer") 
      << "Error! Can't get the pfEgammaCandidates";
  }


 // Get ElectronPixelSeeds
  validPixelSeeds_=true;
  Handle<reco::ElectronSeedCollection> pixelSeedHandle;
  reco::ElectronSeedCollection pixelSeeds;
  theEvent.getByToken(pixelSeedProducer_, pixelSeedHandle);
  if (!pixelSeedHandle.isValid()) {
    validPixelSeeds_=false;
  }


 
  //  std::cout <<  "  GEDPhotonCoreProducer::produce input PFcandidate size " <<   pfCandidateHandle->size() << std::endl;


  // Loop over PF candidates and get only photons
  reco::ElectronSeedCollection::const_iterator pixelSeedItr;
  for(unsigned int lCand=0; lCand < pfCandidateHandle->size(); lCand++) {
    reco::PFCandidateRef candRef (reco::PFCandidateRef(pfCandidateHandle,lCand));

    // Retrieve stuff from the pfPhoton
    reco::PFCandidateEGammaExtraRef pfPhoRef =  candRef->egammaExtraRef();
    reco::SuperClusterRef  refinedSC= pfPhoRef->superClusterRef();
    reco::SuperClusterRef  boxSC= pfPhoRef->superClusterPFECALRef();
    const reco::ConversionRefVector & doubleLegConv = pfPhoRef->conversionRef();
    reco::CaloClusterPtr refinedSCPtr= edm::refToPtr(refinedSC);

    //    std::cout << "newCandidate  doubleLegConv="<<doubleLegConv.size()<< std::endl;
    //std::cout << "newCandidate  singleLegConv="<<  pfPhoRef->singleLegConvTrackRef().size()<< std::endl;

    //////////
    reco::PhotonCore newCandidate;
    newCandidate.setPFlowPhoton(true);
    newCandidate.setStandardPhoton(false);
    newCandidate.setSuperCluster(refinedSC);
    newCandidate.setParentSuperCluster(boxSC);
    // fill conversion infos
    

    for(unsigned int lConv=0; lConv < doubleLegConv.size(); lConv++) {
      newCandidate.addConversion(doubleLegConv[lConv]);
    } 

    //    std::cout << "newCandidate pf refined SC energy="<< newCandidate.superCluster()->energy()<<std::endl;
    //std::cout << "newCandidate pf SC energy="<< newCandidate.parentSuperCluster()->energy()<<std::endl;
    //std::cout << "newCandidate  nconv2leg="<<newCandidate.conversions().size()<< std::endl;

    if ( validPixelSeeds_) {
      for( unsigned int icp = 0;  icp < pixelSeedHandle->size(); icp++) {
        reco::ElectronSeedRef cpRef(pixelSeedHandle,icp);
        if ( boxSC.isNonnull() && boxSC.id() == cpRef->caloCluster().id() && boxSC.key() == cpRef->caloCluster().key() ) {
          newCandidate.addElectronPixelSeed(cpRef);     
        }
      } 
    }



    //
    createSingleLegConversions(refinedSCPtr,  pfPhoRef->singleLegConvTrackRef(), pfPhoRef->singleLegConvMva(), outputOneLegConversionCollection); 
    outputPhotonCoreCollection.push_back(newCandidate);
  }

  SingleLeg_p->assign(outputOneLegConversionCollection.begin(),outputOneLegConversionCollection.end()); 
  const edm::OrphanHandle<reco::ConversionCollection> singleLegConvOrhpHandle = theEvent.put(SingleLeg_p,PFConversionCollection_);

  //std::cout <<  "  GEDPhotonCoreProducer::produce orphanHandle to single legs " <<  singleLegConvOrhpHandle->size() << std::endl;
  //std::cout <<  "  GEDPhotonCoreProducer::produce photon size " <<  outputPhotonCoreCollection.size() << std::endl;

  

  int ipho=0;
  for (reco::PhotonCoreCollection::iterator gamIter=outputPhotonCoreCollection.begin(); gamIter != outputPhotonCoreCollection.end(); ++gamIter){

    for( unsigned int icp = 0;  icp <  singleLegConvOrhpHandle->size(); icp++) {
      const reco::ConversionRef cpRef(reco::ConversionRef(singleLegConvOrhpHandle,icp));
      if ( !cpRef->caloCluster().size()) continue; 
      if (!( gamIter->superCluster().id() == cpRef->caloCluster()[0].id() && gamIter->superCluster().key() == cpRef->caloCluster()[0].key() )) continue; 
      gamIter->addOneLegConversion(cpRef);     
    }
    // debug
    //    std::cout << "PhotonCoreCollection i="<<ipho<<" pf refined SC energy="<<gamIter->superCluster()->energy()<<std::endl;
    //std::cout << "PhotonCoreCollection i="<<ipho<<" pf SC energy="<<gamIter->parentSuperCluster()->energy()<<std::endl;
    //std::cout << "PhotonCoreCollection i="<<ipho<<" nconv2leg="<<gamIter->conversions().size()<<" nconv1leg="<<gamIter->conversionsOneLeg().size()<<std::endl;
    ipho++;
  }



  // put the product in the event
  //  edm::LogInfo("GEDPhotonCoreProducer") << " Put in the event " << iSC << " Photon Candidates \n";
  outputPhotonCoreCollection_p->assign(outputPhotonCoreCollection.begin(),outputPhotonCoreCollection.end());
  theEvent.put( outputPhotonCoreCollection_p, GEDPhotonCoreCollection_);
  


  
} 


void GEDPhotonCoreProducer::createSingleLegConversions(reco::CaloClusterPtr sc,  const std::vector<reco::TrackRef>&  conv, const std::vector<float>& mva,  reco::ConversionCollection &oneLegConversions) {
  // this method translates the single track into the Conversion Data Format

  math::Error<3>::type error;
  for (unsigned int itk=0; itk<conv.size(); itk++){
    const reco::Vertex convVtx(conv[itk]->innerPosition(), error);
    std::vector<reco::TrackRef> OneLegConvVector;
    OneLegConvVector.push_back(conv[itk]);
    std::vector< float > OneLegMvaVector;
    OneLegMvaVector.push_back(mva[itk]);
    std::vector<reco::CaloClusterPtr> dummymatchingBC;
    reco::CaloClusterPtrVector scPtrVec;
    scPtrVec.push_back(sc);


    std::vector<math::XYZPointF>trackPositionAtEcalVec;
    std::vector<math::XYZPointF>innPointVec;
    std::vector<math::XYZVectorF>trackPinVec;
    std::vector<math::XYZVectorF>trackPoutVec;
    math::XYZPointF trackPositionAtEcal(conv[itk]->outerPosition().X(), conv[itk]->outerPosition().Y(), conv[itk]->outerPosition().Z());
    trackPositionAtEcalVec.push_back(trackPositionAtEcal);

    math::XYZPointF innPoint(conv[itk]->innerPosition().X(), conv[itk]->innerPosition().Y(), conv[itk]->innerPosition().Z());
    innPointVec.push_back(innPoint);

    math::XYZVectorF trackPin(conv[itk]->innerMomentum().X(), conv[itk]->innerMomentum().Y(), conv[itk]->innerMomentum().Z());
    trackPinVec.push_back(trackPin);

    math::XYZVectorF trackPout(conv[itk]->outerMomentum().X(), conv[itk]->outerMomentum().Y(), conv[itk]->outerMomentum().Z());
    trackPoutVec.push_back( trackPout );

    float DCA = conv[itk]->d0() ;
    reco::Conversion singleLegConvCandidate(scPtrVec, 
					OneLegConvVector,
					trackPositionAtEcalVec,
					convVtx,
					dummymatchingBC,
					DCA,
					innPointVec,
					trackPinVec,
					trackPoutVec,
					mva[itk],			  
					reco::Conversion::pflow);
    singleLegConvCandidate.setOneLegMVA(OneLegMvaVector); 
    oneLegConversions.push_back(singleLegConvCandidate);

  } 



}
