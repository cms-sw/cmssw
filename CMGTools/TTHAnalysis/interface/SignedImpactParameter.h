#ifndef SignedImpactParmeter_h
#define SignedImpactParmeter_h

#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"

#if defined( __GXX_EXPERIMENTAL_CXX0X__)
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#endif

struct MagneticField;


class SignedImpactParameter {
    public:
        SignedImpactParameter() ;
        ~SignedImpactParameter() ;

        Measurement1D signedIP3D(const reco::Track &tk, const reco::Vertex &vtx, const reco::Track::Vector jetdir) const ;
        Measurement1D signedIP3D(const reco::Track &tk, const reco::VertexCompositePtrCandidate &vtx, const reco::Track::Vector jetdir) const ;
        Measurement1D signedIP2D(const reco::Track &tk, const reco::Vertex &vtx, const reco::Track::Vector jetdir) const ;
        Measurement1D signedIP2D(const reco::Track &tk, const reco::VertexCompositePtrCandidate &vtx, const reco::Track::Vector jetdir) const ;
        Measurement1D IP3D(const reco::Track &tk, const reco::Vertex &vtx) const ;
        Measurement1D IP3D(const reco::Track &tk, const reco::VertexCompositePtrCandidate &vtx) const ;
        Measurement1D IP2D(const reco::Track &tk, const reco::Vertex &vtx) const ;
        Measurement1D IP2D(const reco::Track &tk, const reco::VertexCompositePtrCandidate &vtx) const ;

        std::pair<double,double> twoTrackChi2(const reco::Track &tk1, const reco::Track &tk2) const ;

        //For the vertex related variables
        //A = selectedLeptons[0], B = selectedLeptons[1], C = selectedLeptons[2], D = selectedLeptons[3]
       
        #if defined( __GXX_EXPERIMENTAL_CXX0X__)
        //Helping functions
        std::vector<reco::TransientTrack> ttrksf(const reco::Track &trkA, const reco::Track &trkB, const reco::Track &trkC, const reco::Track &trkD, int nlep) const;  
        std::vector<reco::TransientTrack> ttrksbuthef(const reco::Track &trkA, const reco::Track &trkB, const reco::Track &trkC, const reco::Track &trkD, int nlep, int iptrk) const; 
        reco::TransientTrack thettrkf(const reco::Track &trkA, const reco::Track &trkB, const reco::Track &trkC, const reco::Track &trkD, int nlep, int iptrk) const;
        #endif

        //Variables related to IP
        //Of one lepton w.r.t. the PV of the event
        std::pair<double,double> absIP3D(const reco::Track &trk, const reco::Vertex &pv) const;
        //Of one lepton w.r.t. the PV of the PV of the other leptons only
        std::pair<double,double> absIP3Dtrkpvtrks(const reco::Track &trkA, const reco::Track &trkB, const reco::Track &trkC, const reco::Track &trkD, int nlep, int iptrk) const;
       //Variables related to chi2
       std::pair<double,double> chi2pvtrks(const reco::Track &trkA, const reco::Track &trkB, const reco::Track &trkC, const reco::Track &trkD, int nlep) const;

        Measurement1D vertexD3d(const reco::VertexCompositePtrCandidate &sv, const reco::Vertex &pv) const ;
        Measurement1D vertexDxy(const reco::VertexCompositePtrCandidate &sv, const reco::Vertex &pv) const ;
        float vertexDdotP(const reco::VertexCompositePtrCandidate &sv, const reco::Vertex &pv) const ;
    private:
        //MagneticField *bfield_;
        static MagneticField *paramField_;
};


#endif
