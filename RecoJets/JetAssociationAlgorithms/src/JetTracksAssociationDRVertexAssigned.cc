// Associate jets with tracks by simple "dR" criteria
// Fedor Ratnikov (UMd), Aug. 28, 2007
// $Id: JetTracksAssociationDRVertexAssigned.cc,v 1.2 2011/11/29 10:04:47 srappocc Exp $

#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationDRVertexAssigned.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

JetTracksAssociationDRVertexAssigned::JetTracksAssociationDRVertexAssigned (double fDr) 
: mDeltaR2Threshold (fDr*fDr)
{}

void JetTracksAssociationDRVertexAssigned::produce (reco::JetTracksAssociation::Container* fAssociation, 
					 const std::vector <edm::RefToBase<reco::Jet> >& fJets,
					 const std::vector <reco::TrackRef>& fTracks,
                                         const reco::VertexCollection& vertices) const 
{
  // cache tracks kinematics
  std::vector <math::RhoEtaPhiVector> trackP3s;
  std::map <int,double> trackvert;

 // std::cout<<" Number of vertices "<<vertices.size()<<std::endl;

  trackP3s.reserve (fTracks.size());
  for (unsigned i = 0; i < fTracks.size(); ++i) {
    const reco::Track* track = &*(fTracks[i]);
    trackP3s.push_back (math::RhoEtaPhiVector (track->p(),track->eta(), track->phi()));
        
 // OK: Look for the tracks not associated with vertices 

      const reco::TrackBaseRef ttr1(fTracks[i]);

   
      int trackhasvert = -1;
      for( reco::VertexCollection::const_iterator iv = vertices.begin(); iv != vertices.end(); iv++) {
      std::vector<reco::TrackBaseRef>::const_iterator rr = 
                                                           find((*iv).tracks_begin(),
                                                                (*iv).tracks_end(),
                                                                               ttr1);
           if( rr != (*iv).tracks_end() ) { 
                      trackhasvert = 1;
                      trackvert[i] = (*iv).position().z();
        // std::cout<<" Z "<<i<<" "<<trackhasvert<<" "<<(*iv).position().z()<<std::endl;             
                      break;
           }
      } // all vertices  
     if(trackhasvert < 0) {
    // Take impact parameter of the track as vertex position
       math::XYZPoint ppt(0.,0.,0.);
       trackvert[i] = track->dz(ppt); 
   //    std::cout<<" Z "<<i<<" "<<trackhasvert<<" "<<track->dz(ppt)<<" "<<track->vz()<<" "<<track->vx()<<" "<<
   //    track->vy()<<" "<<track->pz()<<std::endl;
     }
 // OK 
  }  // tracks

  for (unsigned j = 0; j < fJets.size(); ++j) { 

    reco::TrackRefVector assoTracks;
    const reco::Jet* jet = &*(fJets[j]); 
    double jetEta = jet->eta();
    double jetPhi = jet->phi();
    double neweta = 0;
    for (unsigned t = 0; t < fTracks.size(); ++t) {

      std::map<int, double>::iterator cur  = trackvert.find(t);
      if(cur != trackvert.end()) { 
               neweta = jet->physicsEta((*cur).second,jetEta);            
      } else {
	neweta = jetEta; 
	//std::cout<<" Lost track - not in map "<<std::endl;
      }

      //std::cout<<" Old eta-new eta "<<(*cur).second<<" "<<jetEta<<" "<<neweta<<" Track "<<t<<" "<<trackP3s[t].eta()<<std::endl;

      double dR2 = deltaR2 (neweta, jetPhi, trackP3s[t].eta(), trackP3s[t].phi());
      if (dR2 < mDeltaR2Threshold)  assoTracks.push_back (fTracks[t]);
    }

    reco::JetTracksAssociation::setValue (fAssociation, fJets[j], assoTracks);
  }

}
