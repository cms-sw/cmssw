#include "HLTrigger/btau/interface/HLTDisplacedmumuFilter.h"

#include <iostream>

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

//#include "DataFormats/Common/interface/AssociationVector.h"


using namespace edm;
using namespace reco;
using namespace std; 

//typedef edm::AssociationVector<CandidateCollection, VertexCollection > CandidateVertexCollection;

//
// constructors and destructor
//
HLTDisplacedmumuFilter::HLTDisplacedmumuFilter(const edm::ParameterSet& iConfig)
{
	//now do what ever initialization is needed
	minLxySignificance_ = iConfig.getParameter<double>("MinLxySignificance");       
	maxNormalisedChi2_ = iConfig.getParameter<double>("MaxNormalisedChi2");  
	minCosinePointingAngle_ = iConfig.getParameter<double>("MinCosinePointingAngle");
	src_ = iConfig.getParameter<edm::InputTag>("Src");
	produces<VertexCollection>();
	produces<HLTFilterObjectWithRefs>();
// 	produces<CandidateVertexCollection>();
}


HLTDisplacedmumuFilter::~HLTDisplacedmumuFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


// ------------ method called once each job just before starting event loop  ------------
void HLTDisplacedmumuFilter::beginJob(const edm::EventSetup&)
{

}

// ------------ method called once each job just after ending the event loop  ------------
void HLTDisplacedmumuFilter::endJob() 
{
 	
}

// ------------ method called on each new Event  ------------
bool HLTDisplacedmumuFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

			
	//get the jpsi collection 
	Handle<CandidateCollection> candCollection;
	iEvent.getByLabel(src_,candCollection);
	
	// All HLT filters must create and fill an HLT filter object,
	// recording any reconstructed physics objects satisfying (or not)
	// this HLT filter, and place it in the Event.

	// The filter object
	auto_ptr<HLTFilterObjectWithRefs> filterproduct (new HLTFilterObjectWithRefs(path(),module()));
	
	// reference to the candidate to be stored in the HLT filter object
	RefToBase<Candidate> candref;
	
	//get the transient track builder:
	edm::ESHandle<TransientTrackBuilder> theB;
	iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theB);
	std::auto_ptr<VertexCollection> vertexCollection(new VertexCollection());
// 	std::auto_ptr<CandidateVertexCollection> candidatevertexCollection( new CandidateVertexCollection( CandidateRefProd( candCollection )));
	if(candCollection->size()>0) {
		CandidateCollection::const_iterator it=(*candCollection).begin();
		
		for (;it!=(*candCollection).end();it++) {
		
			if ((*it).numberOfDaughters()==2){
			
				vector<TransientTrack> t_tks;
				
				Candidate::const_iterator dit=(*it).begin();
				
				for (;dit!=(*it).end();dit++){
				
				const RecoCandidate * cand = dynamic_cast<const RecoCandidate *> (&(*dit));
				TrackRef tk = (cand)->track();
				
				TransientTrack ttkp   = (*theB).build(&tk);
				
				t_tks.push_back(ttkp);
				
				}
				
				// Call the KalmanVertexFitter if there are 2 tracks
				if (t_tks.size()==2){
				
					KalmanVertexFitter kvf;
					TransientVertex tv = kvf.vertex(t_tks);
					if (tv.isValid()) {
						Vertex vertex = tv;
					
						// get vertex position and error to calculate the decay length significance
						GlobalPoint v = tv.position();
						GlobalError err = tv.positionError();
						float lxy = v.perp();
						float lxyerr = sqrt(err.rerr(v));
						
						// get normalizes chi2
						float normChi2 = tv.normalisedChiSquared();
					
						//calculate the angle between the decay length and the mumu momentum
						Vertex::Point vperp(v.x(),v.y(),0.);
						math::XYZVector pperp((*it).momentum().x(),(*it).momentum().y(),0.);
					
						float cosAlpha = vperp.Dot(pperp)/(vperp.R()*pperp.R());
						
						if( (cosAlpha > minCosinePointingAngle_) && (normChi2 < maxNormalisedChi2_) && ( lxy/lxyerr > minLxySignificance_ ) ){
							
							// put vertex in the event
							vertexCollection->push_back(vertex);
// 							candidatevertexCollection->push_back(vertex);
							iEvent.put(vertexCollection);
// 							iEvent.put(candidatevertexCollection);
							
							// put filter object into the Event
							candref=RefToBase<Candidate>( Ref<CandidateCollection>
                     		(candCollection,distance(candCollection->begin(), it)));
							filterproduct->putParticle(candref);
   							iEvent.put(filterproduct);
							
							return true;
						}
					}
				}
			}	
		}	
	} 
	return false;
}

