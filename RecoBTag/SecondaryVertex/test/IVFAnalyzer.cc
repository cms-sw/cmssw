#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoBTag/SecondaryVertex/interface/SecondaryVertex.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"

class IVFAnalyzer : public edm::EDAnalyzer
{
public:
  explicit IVFAnalyzer(const edm::ParameterSet&);
  ~IVFAnalyzer(){}
  
  // auxiliary class holding simulated primary vertices
  
  
private:
  virtual void beginJob() {}
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() {}
  
 
  
  
};


IVFAnalyzer::IVFAnalyzer(const edm::ParameterSet& pSet){

}


void IVFAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup){


  using namespace reco;

  // some analysis on the inclusive vertices

  // get the IVF collection

 edm::Handle<VertexCollection> BtoCharmIVF;
 event.getByLabel("bToCharmDecayVertexMerged" , BtoCharmIVF);


  // get the IVF with B to charm merging
 
  std::cout<<"BtoCharm vertices size = " << BtoCharmIVF->size() << "  :" << std::endl;
  for(std::vector<reco::Vertex>::const_iterator sv = BtoCharmIVF->begin();
      sv != BtoCharmIVF->end(); ++sv) {
 
    std::cout<<"  pos: x="<< (*sv).position().x()  << " y="<< (*sv).position().y() << " z=" << (*sv).position().z() <<" ntracks=" << (*sv).nTracks() 
	     << " MASS = " << (*sv).p4().M() << " px="<< (*sv).p4().X() << " py="<<(*sv).p4().Y()<<" pz="<<(*sv).p4().Z()<< std::endl;

     for(reco::Vertex::trackRef_iterator ti = sv->tracks_begin(); ti!= sv->tracks_end(); ti++){
      if( sv->trackWeight(*ti)>0.5)  std::cout<<"   track px = " << sv->refittedTrack(*ti).px() << std::endl;
     }
    
  }
  

  // ***************************************
  // now access the double b-tagger prototype
  edm::Handle<reco::JetTagCollection> bTagHandle;
  event.getByLabel( "simpleInclusiveSecondaryVertexHighEffBJetTags", bTagHandle);
  const reco::JetTagCollection & tagColl = *(bTagHandle.product());

  edm::Handle<reco::JetTagCollection> doubleTagHandle;
  event.getByLabel( "doubleSecondaryVertexHighEffBJetTags", doubleTagHandle);
  const reco::JetTagCollection & doubleTagColl = *(doubleTagHandle.product());

  for(reco::JetTagCollection::const_iterator it = tagColl.begin() ; it!=tagColl.end() ; ++it)
  {
    std::cout<<" simple SSV b-tag discriminator = " << it->second << std::endl;
    if(it->second > -0.5) std::cout<<" **** THIS JET HAS ONE VERTEX ****"<<std::endl;
  }


  for(reco::JetTagCollection::const_iterator it = doubleTagColl.begin() ; it!=doubleTagColl.end() ; ++it)
  {
    std::cout<<" DOUBLE b-tag discriminator = " << it->second << std::endl;
    if(it->second > -0.5) std::cout<<" **** THIS JET HAS TWO VERTICES AND IS DOUBLE TAGGED !!! ****"<<std::endl;
  }
 
}


DEFINE_FWK_MODULE(IVFAnalyzer);
