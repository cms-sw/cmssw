#include "CMGTools/TTHAnalysis/interface/SignedImpactParameter.h"
#include "MagneticField/UniformEngine/src/UniformMagneticField.h"
#include "MagneticField/ParametrizedEngine/src/OAEParametrizedMagneticField.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include <vector>
#include <TMath.h>
#include <Math/SVector.h>

MagneticField *SignedImpactParameter::paramField_ = 0;

SignedImpactParameter::SignedImpactParameter() {
}

SignedImpactParameter::~SignedImpactParameter() {
}

//Signed 3D IP

Measurement1D 
SignedImpactParameter::signedIP3D(const reco::Track &tk, const reco::Vertex &vtx, const reco::Track::Vector jetdir) const {
    if (paramField_ == 0) paramField_ = new OAEParametrizedMagneticField("3_8T");
    reco::TransientTrack ttk(tk,paramField_);
    return IPTools::signedImpactParameter3D(ttk, GlobalVector(jetdir.X(),jetdir.Y(),jetdir.Z()), vtx).second;
}

Measurement1D 
SignedImpactParameter::signedIP3D(const reco::Track &tk, const reco::VertexCompositePtrCandidate &sv, const reco::Track::Vector jetdir) const {
    reco::Vertex::CovarianceMatrix csv; sv.fillVertexCovariance(csv);
    reco::Vertex svtx(sv.vertex(), csv);
    return signedIP3D(tk, svtx, jetdir);
}

//Signed 2D IP

Measurement1D 
SignedImpactParameter::signedIP2D(const reco::Track &tk, const reco::Vertex &vtx, const reco::Track::Vector jetdir) const {
    if (paramField_ == 0) paramField_ = new OAEParametrizedMagneticField("3_8T");
    reco::TransientTrack ttk(tk,paramField_);
    return IPTools::signedTransverseImpactParameter(ttk, GlobalVector(jetdir.X(),jetdir.Y(),jetdir.Z()), vtx).second;
}

Measurement1D 
SignedImpactParameter::signedIP2D(const reco::Track &tk, const reco::VertexCompositePtrCandidate &sv, const reco::Track::Vector jetdir) const {
    reco::Vertex::CovarianceMatrix csv; sv.fillVertexCovariance(csv);
    reco::Vertex svtx(sv.vertex(), csv);
    return signedIP2D(tk, svtx, jetdir);
}


//3D IP

Measurement1D 
SignedImpactParameter::IP3D(const reco::Track &tk, const reco::Vertex &vtx) const {
    if (paramField_ == 0) paramField_ = new OAEParametrizedMagneticField("3_8T");
    reco::TransientTrack ttk(tk,paramField_);
    return IPTools::absoluteImpactParameter3D(ttk,vtx).second;
}

Measurement1D 
SignedImpactParameter::IP3D(const reco::Track &tk, const reco::VertexCompositePtrCandidate &sv) const {
    reco::Vertex::CovarianceMatrix csv; sv.fillVertexCovariance(csv);
    reco::Vertex svtx(sv.vertex(), csv);
    return IP3D(tk, svtx);
}

//2D IP

Measurement1D 
SignedImpactParameter::IP2D(const reco::Track &tk, const reco::Vertex &vtx) const {
    if (paramField_ == 0) paramField_ = new OAEParametrizedMagneticField("3_8T");
    reco::TransientTrack ttk(tk,paramField_);
    return IPTools::absoluteTransverseImpactParameter(ttk,vtx).second;
}

Measurement1D 
SignedImpactParameter::IP2D(const reco::Track &tk, const reco::VertexCompositePtrCandidate &sv) const {
    reco::Vertex::CovarianceMatrix csv; sv.fillVertexCovariance(csv);
    reco::Vertex svtx(sv.vertex(), csv);
    return IP2D(tk, svtx);
}



std::pair<double,double>
SignedImpactParameter::twoTrackChi2(const reco::Track &tk1, const reco::Track &tk2) const {
    if (paramField_ == 0) paramField_ = new OAEParametrizedMagneticField("3_8T");
    std::vector<reco::TransientTrack> ttks;
    ttks.push_back(reco::TransientTrack(tk1,paramField_));
    ttks.push_back(reco::TransientTrack(tk2,paramField_));
    KalmanVertexFitter vtxFitter;
    TransientVertex myVertex = vtxFitter.vertex(ttks);
    return std::make_pair(myVertex.totalChiSquared(),myVertex.degreesOfFreedom());  
}

//Helping functions
std::vector<reco::TransientTrack> SignedImpactParameter::ttrksf(const reco::Track &trkA, const reco::Track &trkB, const reco::Track &trkC, const reco::Track &trkD, int nlep) const {
  std::vector<reco::TransientTrack> ttrks;
 if (paramField_ == 0) paramField_ = new OAEParametrizedMagneticField("3_8T");
 if(nlep==2){
  ttrks.push_back(reco::TransientTrack(trkA,paramField_));
  ttrks.push_back(reco::TransientTrack(trkB,paramField_));
 }else if(nlep==3){
  ttrks.push_back(reco::TransientTrack(trkA,paramField_));
  ttrks.push_back(reco::TransientTrack(trkB,paramField_));
  ttrks.push_back(reco::TransientTrack(trkC,paramField_));
 }else if(nlep==4){
  ttrks.push_back(reco::TransientTrack(trkA,paramField_));
  ttrks.push_back(reco::TransientTrack(trkB,paramField_));
  ttrks.push_back(reco::TransientTrack(trkC,paramField_));
  ttrks.push_back(reco::TransientTrack(trkD,paramField_));
 }
 return ttrks;
}

std::vector<reco::TransientTrack> SignedImpactParameter::ttrksbuthef(const reco::Track &trkA, const reco::Track &trkB, const reco::Track &trkC, const reco::Track &trkD, int nlep, int iptrk) const {
  std::vector<reco::TransientTrack> ttrks;
 if (paramField_ == 0) paramField_ = new OAEParametrizedMagneticField("3_8T");
 if(nlep==3){
  if(iptrk==0){
   ttrks.push_back(reco::TransientTrack(trkB,paramField_));
   ttrks.push_back(reco::TransientTrack(trkC,paramField_));
  }else if(iptrk==1){
   ttrks.push_back(reco::TransientTrack(trkA,paramField_));
   ttrks.push_back(reco::TransientTrack(trkC,paramField_));
  }else if(iptrk==2){
   ttrks.push_back(reco::TransientTrack(trkA,paramField_));
   ttrks.push_back(reco::TransientTrack(trkB,paramField_));
  } 
 }else if(nlep==4){
  if(iptrk==0){
   ttrks.push_back(reco::TransientTrack(trkB,paramField_));
   ttrks.push_back(reco::TransientTrack(trkC,paramField_));
   ttrks.push_back(reco::TransientTrack(trkD,paramField_));
  }else if(iptrk==1){
   ttrks.push_back(reco::TransientTrack(trkA,paramField_));
   ttrks.push_back(reco::TransientTrack(trkC,paramField_));
   ttrks.push_back(reco::TransientTrack(trkD,paramField_));
  }else if(iptrk==2){
   ttrks.push_back(reco::TransientTrack(trkA,paramField_));
   ttrks.push_back(reco::TransientTrack(trkB,paramField_));
   ttrks.push_back(reco::TransientTrack(trkD,paramField_));
  }else if(iptrk==3){
   ttrks.push_back(reco::TransientTrack(trkA,paramField_));
   ttrks.push_back(reco::TransientTrack(trkB,paramField_));
   ttrks.push_back(reco::TransientTrack(trkC,paramField_));
  }
 }
 return ttrks;
}

reco::TransientTrack SignedImpactParameter::thettrkf(const reco::Track &trkA, const reco::Track &trkB, const reco::Track &trkC, const reco::Track &trkD, int nlep, int iptrk) const {
 reco::TransientTrack thettrk; 
 if (paramField_ == 0) paramField_ = new OAEParametrizedMagneticField("3_8T");
 if(nlep==3){
  if(iptrk==0){
   reco::TransientTrack tmpttrk(trkA,paramField_); thettrk = tmpttrk;   
  }else if(iptrk==1){
   reco::TransientTrack tmpttrk(trkB,paramField_); thettrk = tmpttrk;
  }else if(iptrk==2){
   reco::TransientTrack tmpttrk(trkC,paramField_); thettrk = tmpttrk;
  } 
 }else if(nlep==4){
  if(iptrk==0){
   reco::TransientTrack tmpttrk(trkA,paramField_); thettrk = tmpttrk;
  }else if(iptrk==1){
   reco::TransientTrack tmpttrk(trkB,paramField_); thettrk = tmpttrk;
  }else if(iptrk==2){
   reco::TransientTrack tmpttrk(trkC,paramField_); thettrk = tmpttrk;
  }else if(iptrk==3){
   reco::TransientTrack tmpttrk(trkD,paramField_); thettrk = tmpttrk;
  }
 }
 return thettrk;
}

//Variables related to IP
//Of one lepton w.r.t. the PV of the event
std::pair<double,double> SignedImpactParameter::absIP3D(const reco::Track &trk, const reco::Vertex &pv) const {
 if (paramField_ == 0) paramField_ = new OAEParametrizedMagneticField("3_8T");
 reco::TransientTrack ttrk(trk,paramField_);
 Measurement1D aIP3Dtrk = IPTools::absoluteImpactParameter3D(ttrk, pv).second; 
 return std::make_pair(aIP3Dtrk.value(), aIP3Dtrk.error()); 
}
//Of one lepton w.r.t. the PV of the PV of the other leptons only
std::pair<double,double> SignedImpactParameter::absIP3Dtrkpvtrks(const reco::Track &trkA, const reco::Track &trkB, const reco::Track &trkC, const reco::Track &trkD, int nlep, int iptrk) const {
 //Take the transient tracks
 reco::TransientTrack thettrk = thettrkf(trkA,trkB,trkC,trkD,nlep,iptrk); 
 std::vector<reco::TransientTrack> ttrks = ttrksbuthef(trkA,trkB,trkC,trkD,nlep,iptrk);
 //Build new vertex
 KalmanVertexFitter vtxFitter;
 TransientVertex trkspv = vtxFitter.vertex(ttrks);
 //Measure 3DIP
 Measurement1D aIP3Dtrk = IPTools::absoluteImpactParameter3D(thettrk,reco::Vertex(trkspv)).second;
 return std::make_pair(aIP3Dtrk.value(), aIP3Dtrk.error());
} 

//Variables related to chi2
std::pair<double,double> SignedImpactParameter::chi2pvtrks(const reco::Track &trkA, const reco::Track &trkB, const reco::Track &trkC, const reco::Track &trkD, int nlep) const {
 //Take transient tracks
  std::vector<reco::TransientTrack> ttrks = ttrksf(trkA,trkB,trkC,trkD,nlep);
 //Build new vertex
 KalmanVertexFitter vtxFitter;
 TransientVertex trkspv = vtxFitter.vertex(ttrks);
 //Take interested values
 return std::make_pair(trkspv.totalChiSquared(),trkspv.degreesOfFreedom());  
}

Measurement1D
SignedImpactParameter::vertexD3d(const reco::VertexCompositePtrCandidate &svcand, const reco::Vertex &pv) const {
    VertexDistance3D dist;
    reco::Vertex::CovarianceMatrix csv; svcand.fillVertexCovariance(csv);
    reco::Vertex svtx(svcand.vertex(), csv);
    return dist.distance(svtx, pv);
}

Measurement1D
SignedImpactParameter::vertexDxy(const reco::VertexCompositePtrCandidate &svcand, const reco::Vertex &pv) const {
    VertexDistanceXY dist;
    reco::Vertex::CovarianceMatrix csv; svcand.fillVertexCovariance(csv);
    reco::Vertex svtx(svcand.vertex(), csv);
    return dist.distance(svtx, pv);
}

float SignedImpactParameter::vertexDdotP(const reco::VertexCompositePtrCandidate &sv, const reco::Vertex &pv) const {
    reco::Candidate::Vector p = sv.momentum();
    reco::Candidate::Vector d(sv.vx() - pv.x(), sv.vy() - pv.y(), sv.vz() - pv.z());
    return p.Unit().Dot(d.Unit());
}
