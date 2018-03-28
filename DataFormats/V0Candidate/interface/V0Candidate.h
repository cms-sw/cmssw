#ifndef CANDIDATE_V0CANDIDATE_H
#define CANDIDATE_V0CANDIDATE_H
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"

namespace reco{

 class V0Candidate : public CompositeCandidate {


  public:
  V0Candidate() : CompositeCandidate() { }
  V0Candidate( Charge q, const LorentzVector & p4, 
	       const Point & vtx = Point( 0, 
					  0, 
					  0 ) ) : CompositeCandidate(q, p4,
								     vtx) { }
    const Vertex & getRecoVertex() const { return recoVertex; }
    const Vertex::CovarianceMatrix vtxCovariance() { 
      return recoVertex.covariance(); 
    }
    void setRecoVertex( const Vertex & vtxIn );
    //    virtual int pdgId() const { return PDGid; }
    //    void setPdgId( const int & Id ) { PDGid = Id; }
  private:
    Vertex recoVertex;
    //    int PDGid;
  };

}

#endif
