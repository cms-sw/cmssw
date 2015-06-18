#include "CommonTools/RecoAlgos/interface/PrimaryVertexAssignment.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "TrackingTools/IPTools/interface/IPTools.h"



std::pair<int,PrimaryVertexAssignment::Quality>
PrimaryVertexAssignment::chargedHadronVertex( const reco::VertexCollection& vertices,
                                   const reco::TrackRef& trackRef,
                                   const reco::Track* track,
                                   const edm::View<reco::Candidate>& jets,
                                   const TransientTrackBuilder& builder) const {

  int iVertex = -1;
  size_t index=0;
  typedef reco::VertexCollection::const_iterator IV;
  typedef reco::Vertex::trackRef_iterator IT;
  float bestweight=0;
  for( auto const & vtx : vertices) {
      float w = vtx.trackWeight(trackRef);
      if (w > bestweight){
          bestweight=w;
          iVertex=index;
        }        
      index++;
  }


  if(iVertex >= 0 ) return std::pair<int,PrimaryVertexAssignment::Quality>(iVertex,PrimaryVertexAssignment::UsedInFit);

    double dzmin = 1e99;
    int vtxIdMinDz = -1;
    for(IV iv=vertices.begin(); iv!=vertices.end(); ++iv) {
      double dz = std::abs(track->dz(iv->position()));
      if(dz<dzmin) {
        dzmin = dz;
        vtxIdMinDz = iv-vertices.begin();
      }
   }
  // first use "closest in Z" with tight cuts (targetting primary particles)
    if(vtxIdMinDz>=0 and (dzmin < maxDzForPrimaryAssignment_ or dzmin/track->dzError() < maxDzSigForPrimaryAssignment_ ))
    {
        iVertex=vtxIdMinDz;
    }
  if(iVertex >= 0 ) return std::pair<int,PrimaryVertexAssignment::Quality>(iVertex,PrimaryVertexAssignment::PrimaryDz);

  // if track not assigned yet, it could be a b-decay secondary , use jet axis dist criterion
    // first find the closest jet within maxJetDeltaR_
    int jetIdx = -1;
    double minDeltaR = maxJetDeltaR_;
    for(edm::View<reco::Candidate>::const_iterator ij=jets.begin(); ij!=jets.end(); ++ij)
    {
      if( ij->pt() < minJetPt_ ) continue; // skip jets below the jet Pt threshold

      double deltaR = reco::deltaR( *ij, *track );
      if( deltaR < minDeltaR )
      {
        minDeltaR = deltaR;
        jetIdx = std::distance(jets.begin(), ij);
      }
    }
    // if jet found
    if( jetIdx!=-1 )
    {
      reco::TransientTrack transientTrack = builder.build(*track);
      GlobalVector direction(jets.at(jetIdx).px(), jets.at(jetIdx).py(), jets.at(jetIdx).pz());
      // find the vertex with the smallest distanceToJetAxis that is still within maxDistaneToJetAxis_
      int vtxIdx = -1;
      double minDistanceToJetAxis = maxDistanceToJetAxis_;
      for(IV iv=vertices.begin(); iv!=vertices.end(); ++iv)
      {
        // only check for vertices that are close enough in Z and for tracks that have not too high dXY
        if(std::abs(track->dz(iv->position())) > maxDzForJetAxisAssigment_ || std::abs(track->dxy(iv->position())) > maxDxyForJetAxisAssigment_) 
          continue;

        double distanceToJetAxis = IPTools::jetTrackDistance(transientTrack, direction, *iv).second.value();
        if( distanceToJetAxis < minDistanceToJetAxis )
        {
          minDistanceToJetAxis = distanceToJetAxis;
          vtxIdx = std::distance(vertices.begin(), iv);
        }
      }
      if( vtxIdx>=0 )
      {
        iVertex=vtxIdx;
      }
    }
  if(iVertex >= 0 )
     return std::pair<int,PrimaryVertexAssignment::Quality>(iVertex,PrimaryVertexAssignment::BTrack);

  // if the track is not compatible with other PVs but is compatible with the BeamSpot, we may simply have not reco'ed the PV!
  //  we still point it to the closest in Z, but flag it as possible orphan-primary
  if(std::abs(track->dxy(vertices[0].position()))<maxDxyForNotReconstructedPrimary_ && std::abs(track->dxy(vertices[0].position())/track->dxyError())<maxDxySigForNotReconstructedPrimary_)
     return std::pair<int,PrimaryVertexAssignment::Quality>(vtxIdMinDz,PrimaryVertexAssignment::NotReconstructedPrimary);
 
  //FIXME: here we could better handle V0s and NucInt

  // all other tracks could be non-B secondaries and we just attach them with closest Z
  if(vtxIdMinDz>=0)
     return std::pair<int,PrimaryVertexAssignment::Quality>(vtxIdMinDz,PrimaryVertexAssignment::OtherDz);
  //If for some reason even the dz failed (when?) we consider the track not assigned
  return std::pair<int,PrimaryVertexAssignment::Quality>(-1,PrimaryVertexAssignment::Unassigned);
}


