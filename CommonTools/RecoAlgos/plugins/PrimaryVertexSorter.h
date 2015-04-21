#ifndef CommonTools_RecoAlgos_PrimaryVertexSorter_
#define CommonTools_RecoAlgos_PrimaryVertexSorter_

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Association.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "CommonTools/RecoAlgos/interface/PrimaryVertexAssignment.h"
#include "CommonTools/RecoAlgos/interface/PrimaryVertexSorting.h"

/**\class PrimaryVertexSorter
 * \author Andrea Rizzi

*/


template <class ParticlesCollection>

class PrimaryVertexSorter : public edm::stream::EDProducer<> {
 public:

  typedef edm::Association<reco::VertexCollection> CandToVertex;
  typedef edm::ValueMap<int> CandToVertexQuality;
  typedef edm::ValueMap<float> VertexScore;

  typedef ParticlesCollection PFCollection;

  explicit PrimaryVertexSorter(const edm::ParameterSet&);

  ~PrimaryVertexSorter() {}

  virtual void produce(edm::Event&, const edm::EventSetup&) override;

 private:

  PrimaryVertexAssignment    assignmentAlgo_;
  PrimaryVertexSorting       sortingAlgo_;

  /// Candidates to be analyzed
  edm::EDGetTokenT<PFCollection>   tokenCandidates_;

  /// vertices
  edm::EDGetTokenT<reco::VertexCollection>   tokenVertices_;
  edm::EDGetTokenT<edm::View<reco::Candidate> >   tokenJets_;

  bool produceOriginalMapping_;
  bool produceSortedVertices_;
  bool producePFPileUp_;
  bool producePFNoPileUp_;
  int  qualityCut_;
  bool useMET_;
};



#include "DataFormats/VertexReco/interface/Vertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"

// #include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"


using namespace std;
using namespace edm;
using namespace reco;

template <class ParticlesCollection>
PrimaryVertexSorter<ParticlesCollection>::PrimaryVertexSorter(const edm::ParameterSet& iConfig) :
  assignmentAlgo_(iConfig.getParameterSet("assignment")),
  sortingAlgo_(iConfig.getParameterSet("sorting")),
  tokenCandidates_(consumes<ParticlesCollection>(iConfig.getParameter<InputTag>("particles"))),
  tokenVertices_(consumes<VertexCollection>(iConfig.getParameter<InputTag>("vertices"))),
  tokenJets_(consumes<edm::View<reco::Candidate> > (iConfig.getParameter<InputTag>("jets"))),
  produceOriginalMapping_(iConfig.getParameter<bool>("produceAssociationToOriginalVertices")),
  produceSortedVertices_(iConfig.getParameter<bool>("produceSortedVertices")),
  producePFPileUp_(iConfig.getParameter<bool>("producePileUpCollection")),
  producePFNoPileUp_(iConfig.getParameter<bool>("produceNoPileUpCollection")),
  qualityCut_(iConfig.getParameter<int>("qualityForPrimary")),
  useMET_(iConfig.getParameter<bool>("usePVMET"))
{


  if(produceOriginalMapping_){
      produces< CandToVertex> ("original");
      produces< CandToVertexQuality> ("original");
      produces< VertexScore> ("original");
  }
  if(produceSortedVertices_){
      produces< reco::VertexCollection> ();
      produces< CandToVertex> ();
      produces< CandToVertexQuality> ();
      produces< VertexScore> ();
  }

  if(producePFPileUp_){
      if(produceOriginalMapping_)
            produces< PFCollection> ("originalPileUp");
      if(produceSortedVertices_)
            produces< PFCollection> ("PileUp");
  }

  if(producePFNoPileUp_){
      if(produceOriginalMapping_)
            produces< PFCollection> ("originalNoPileUp");
      if(produceSortedVertices_)
            produces< PFCollection> ("NoPileUp");
  }


}





template <class ParticlesCollection>
void PrimaryVertexSorter<ParticlesCollection>::produce(Event& iEvent,  const EventSetup& iSetup) {


  Handle<edm::View<reco::Candidate> > jets;
  iEvent.getByToken( tokenJets_, jets);

  edm::ESHandle<TransientTrackBuilder> builder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", builder);


  Handle<VertexCollection> vertices;
  iEvent.getByToken( tokenVertices_, vertices);

  Handle<ParticlesCollection> particlesHandle;
  iEvent.getByToken( tokenCandidates_, particlesHandle);

  ParticlesCollection particles = *particlesHandle.product();
  std::vector<int> pfToPVVector;
  std::vector<PrimaryVertexAssignment::Quality> pfToPVQualityVector;
  //reverse mapping
  std::vector< std::vector<int> > pvToPFVector(vertices->size());
  std::vector< std::vector<const reco::Candidate *> > pvToCandVector(vertices->size());
  std::vector< std::vector<PrimaryVertexAssignment::Quality> > pvToPFQualityVector(vertices->size());
  std::vector<float> vertexScoreOriginal(vertices->size());
  std::vector<float> vertexScore(vertices->size());

    for(auto const & pf : particles) {
    std::pair<int,PrimaryVertexAssignment::Quality> vtxWithQuality=assignmentAlgo_.chargedHadronVertex(*vertices,pf,*jets,*builder);
    pfToPVVector.push_back(vtxWithQuality.first); 
    pfToPVQualityVector.push_back(vtxWithQuality.second); 
  }

  //Invert the mapping
  for(size_t i = 0; i < pfToPVVector.size();i++)
  {
    auto pv = pfToPVVector[i];
    auto qual = pfToPVQualityVector[i];
    if(pv >=0 and qual >= qualityCut_){
       pvToPFVector[pv].push_back(i);
//    std::cout << i << std::endl;
//     const typename  ParticlesCollection::value_type & cp = particles[i];
//     std::cout << "CP " << &cp <<  std::endl;
       pvToCandVector[pv].push_back( &particles[i] );
       pvToPFQualityVector[pv].push_back(qual);
    }
  }

  //Use multimap for sorting of indices
  std::multimap<float,int> scores;
  for(unsigned int i=0;i<vertices->size();i++){
     float s=sortingAlgo_.score((*vertices)[i],pvToCandVector[i],useMET_);
     vertexScoreOriginal[i]=s;
     scores.insert(std::pair<float,int>(-s,i));    
  }

  //create indices
  std::vector<int> oldToNew(vertices->size()),  newToOld(vertices->size());
  size_t newIdx=0;
  for(auto const &  idx :  scores)
  {
//    std::cout << newIdx << " score: " << idx.first << " oldidx: " << idx.second << " "<< producePFPileUp_ << std::endl;
    vertexScore[newIdx]=-idx.first;
    oldToNew[idx.second]=newIdx;
    newToOld[newIdx]=idx.second;
    newIdx++;
  }
  



  if(produceOriginalMapping_){
    auto_ptr< CandToVertex>  pfCandToOriginalVertexOutput( new CandToVertex(vertices) );
    auto_ptr< CandToVertexQuality>  pfCandToOriginalVertexQualityOutput( new CandToVertexQuality() );
    CandToVertex::Filler cand2VertexFiller(*pfCandToOriginalVertexOutput);
    CandToVertexQuality::Filler cand2VertexQualityFiller(*pfCandToOriginalVertexQualityOutput);

    cand2VertexFiller.insert(particlesHandle,pfToPVVector.begin(),pfToPVVector.end());
    cand2VertexQualityFiller.insert(particlesHandle,pfToPVQualityVector.begin(),pfToPVQualityVector.end());

    cand2VertexFiller.fill();
    cand2VertexQualityFiller.fill();
    iEvent.put( pfCandToOriginalVertexOutput ,"original");
    iEvent.put( pfCandToOriginalVertexQualityOutput ,"original");

    auto_ptr< VertexScore>  vertexScoreOriginalOutput( new VertexScore );
    VertexScore::Filler vertexScoreOriginalFiller(*vertexScoreOriginalOutput);
    vertexScoreOriginalFiller.insert(vertices,vertexScoreOriginal.begin(),vertexScoreOriginal.end());
    vertexScoreOriginalFiller.fill();
    iEvent.put( vertexScoreOriginalOutput ,"original");
 
  }

  if(produceSortedVertices_){
      std::vector<int> pfToSortedPVVector;
//      std::vector<int> pfToSortedPVQualityVector;
      for(size_t i=0;i<pfToPVVector.size();i++) {
        pfToSortedPVVector.push_back(oldToNew[pfToPVVector[i]]);
//        pfToSortedPVQualityVector.push_back(pfToPVQualityVector[i]); //same as old!
      }

      auto_ptr< reco::VertexCollection>  sortedVerticesOutput( new reco::VertexCollection );
      for(size_t i=0;i<vertices->size();i++){
         sortedVerticesOutput->push_back((*vertices)[newToOld[i]]); 
      }
    edm::OrphanHandle<reco::VertexCollection> oh = iEvent.put( sortedVerticesOutput);
    auto_ptr< CandToVertex>  pfCandToVertexOutput( new CandToVertex(oh) );
    auto_ptr< CandToVertexQuality>  pfCandToVertexQualityOutput( new CandToVertexQuality() );
    CandToVertex::Filler cand2VertexFiller(*pfCandToVertexOutput);
    CandToVertexQuality::Filler cand2VertexQualityFiller(*pfCandToVertexQualityOutput);

    cand2VertexFiller.insert(particlesHandle,pfToSortedPVVector.begin(),pfToSortedPVVector.end());
    cand2VertexQualityFiller.insert(particlesHandle,pfToPVQualityVector.begin(),pfToPVQualityVector.end());

    cand2VertexFiller.fill();
    cand2VertexQualityFiller.fill();
    iEvent.put( pfCandToVertexOutput );
    iEvent.put( pfCandToVertexQualityOutput );

    auto_ptr< VertexScore>  vertexScoreOutput( new VertexScore );
    VertexScore::Filler vertexScoreFiller(*vertexScoreOutput);
    vertexScoreFiller.insert(oh,vertexScore.begin(),vertexScore.end());
    vertexScoreFiller.fill();
    iEvent.put( vertexScoreOutput);


  }


  auto_ptr< PFCollection >  pfCollectionNOPUOriginalOutput( new PFCollection );
  auto_ptr< PFCollection >  pfCollectionNOPUOutput( new PFCollection );
  auto_ptr< PFCollection >  pfCollectionPUOriginalOutput( new PFCollection );
  auto_ptr< PFCollection >  pfCollectionPUOutput( new PFCollection );

  for(size_t i=0;i<particles.size();i++) {
    auto pv = pfToPVVector[i];
    auto qual = pfToPVQualityVector[i];


     if(producePFNoPileUp_ && produceSortedVertices_) 
         if(pv == newToOld[0] and qual >= qualityCut_) 
                   pfCollectionNOPUOutput->push_back(particles[i]);

     if(producePFPileUp_ && produceSortedVertices_) 
         if(pv != newToOld[0] and qual >= qualityCut_) 
                   pfCollectionPUOutput->push_back(particles[i]);

     if(producePFNoPileUp_ && produceOriginalMapping_) 
         if(pv == 0 and qual >= qualityCut_) 
                   pfCollectionNOPUOriginalOutput->push_back(particles[i]);

     if(producePFPileUp_ && produceOriginalMapping_) 
         if(pv != 0 and qual >= qualityCut_) 
                   pfCollectionPUOriginalOutput->push_back(particles[i]);

  }              
  if(producePFNoPileUp_ && produceSortedVertices_) iEvent.put(pfCollectionNOPUOutput,"NoPileUp" );
  if(producePFPileUp_ && produceSortedVertices_) iEvent.put(pfCollectionPUOutput, "PileUp");
  if(producePFNoPileUp_ && produceOriginalMapping_) iEvent.put(pfCollectionNOPUOriginalOutput,"originalNoPileUp" );
  if(producePFPileUp_ && produceOriginalMapping_) iEvent.put(pfCollectionPUOriginalOutput,"originalPileUp" );
  

} 


#endif
