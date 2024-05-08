#ifndef __L1Trigger_L1TTrackMatch_DisplacedVertexProducer_h__
#define __L1Trigger_L1TTrackMatch_DisplacedVertexProducer_h__

#include "DataFormats/L1Trigger/interface/DisplacedVertex.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTTrackAssociationMap.h"
#include "TMath.h" 
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <valarray>

using namespace std;

class Track_Parameters
{
public:
  float pt;
  float d0;
  float dxy = -99999;
  float z0;
  float eta;
  float phi;
  float charge;
  float rho;
  int index;
  int pdgid = -99999;
  float vx;
  float vy;
  float vz;
  Track_Parameters* tp;
  float x0;
  float y0;
  int nstubs;
  float chi2rphi;
  float chi2rz;
  float bendchi2;
  float MVA1;
  float MVA2;

  float dist_calc(float x_dv, float y_dv, float x, float y){
    dxy = TMath::Sqrt((x_dv-x)*(x_dv-x) + (y_dv-y)*(y_dv-y));
    return dxy;
  }
  float x(float phi_T=0){
    return (-charge * rho * TMath::Sin(phi - charge*phi_T) + (d0 + charge * rho) * TMath::Sin(phi));
  }
  float y(float phi_T=0){
    return ( charge * rho * TMath::Cos(phi - charge*phi_T) - (d0 + charge * rho) * TMath::Cos(phi));
  }
  float z(float phi_T=0){
    float theta = 2 * TMath::ATan(TMath::Exp(-eta));
    return (z0 + rho*phi_T/TMath::Tan(theta));
  }
  float deltaPhi_T(Double_t phi1, Double_t phi2)
  {
    Double_t dPhi = phi1 - phi2;
    if (dPhi >= TMath::Pi())
      dPhi -= 2. * TMath::Pi();
    if (dPhi <= -TMath::Pi())
      dPhi += 2. * TMath::Pi();
    return dPhi;
  }
  float phi_T(float x, float y){
    float num = x - (d0 + charge * rho) * TMath::Sin(phi);
    float den = y + (d0 + charge * rho) * TMath::Cos(phi);
    return ((phi-TMath::ATan2(num,-den))/charge);
  }
  float z(float x, float y){
    float t = std::sinh(eta);
    float r = TMath::Sqrt(pow(x,2)+pow(y,2));
    return (z0+(t*r*(1+(pow(d0,2)/pow(r,2))+(1.0/6.0)*pow(r/(2*rho),2)))); // can do higher order terms if necessary from displaced math
  }
  Track_Parameters(float pt_in, float d0_in, float z0_in, float eta_in, float phi_in, int pdgid_in, float vx_in, float vy_in, float vz_in, float charge_in=0, int index_in=-1, Track_Parameters* tp_in=nullptr, int nstubs_in=0, float chi2rphi_in=0, float chi2rz_in=0, float bendchi2_in=0, float MVA1_in=0, float MVA2_in=0)
  {
    pt = pt_in;
    d0 = d0_in;
    z0 = z0_in;
    eta = eta_in;
    phi = phi_in;
    if(charge_in > 0){
      charge = 1;
    }
    else if (charge_in < 0){
      charge = -1;
    }
    else{
      charge = 0;
    }
    index = index_in;
    pdgid = pdgid_in;
    vx = vx_in;
    vy = vy_in;
    vz = vz_in;
    tp = tp_in;
    rho = fabs(1/charge_in);
    x0 = (rho+charge*d0)*TMath::Cos(phi-(charge*TMath::Pi()/2));
    y0 = (rho+charge*d0)*TMath::Sin(phi-(charge*TMath::Pi()/2));
    nstubs = nstubs_in;
    chi2rphi = chi2rphi_in;
    chi2rz = chi2rz_in;
    bendchi2 = bendchi2_in;
    MVA1 = MVA1_in;
    MVA2 = MVA2_in;
  }
  Track_Parameters(){};
  ~Track_Parameters(){};
};

constexpr bool operator==(const Track_Parameters* lhs, const Track_Parameters& rhs)
{
  return (lhs->pt==rhs.pt && lhs->d0==rhs.d0 && lhs->z0==rhs.z0 && lhs->eta==rhs.eta && lhs->phi==rhs.phi);
}
constexpr bool operator==(const Track_Parameters& lhs, const Track_Parameters* rhs)
{
  return (lhs.pt==rhs->pt && lhs.d0==rhs->d0 && lhs.z0==rhs->z0 && lhs.eta==rhs->eta && lhs.phi==rhs->phi);
}
constexpr bool operator==(const Track_Parameters& lhs, const Track_Parameters& rhs)
{
  return (lhs.pt==rhs.pt && lhs.d0==rhs.d0 && lhs.z0==rhs.z0 && lhs.eta==rhs.eta && lhs.phi==rhs.phi);
}

std::valarray<float> calcPVec(Track_Parameters a, double_t v_x, double_t v_y)
{
  std::valarray<float> r_vec = {float(v_x)-a.x0,float(v_y)-a.y0};
  std::valarray<float> p_vec = {-r_vec[1],r_vec[0]};
  if(a.charge>0){
    p_vec *= -1;
  }
  p_vec /= TMath::Sqrt(pow(p_vec[0],2)+pow(p_vec[1],2));
  p_vec *= a.pt;
  return p_vec;
}

class Vertex_Parameters
{
public:
  Double_t x_dv;
  Double_t y_dv;
  Double_t z_dv;
  float score;
  Track_Parameters a;
  Track_Parameters b;
  int inTraj;
  bool matched = false;
  std::vector<Track_Parameters> tracks = {};
  float p_mag;
  float p2_mag;
  float openingAngle;
  float R_T;
  float cos_T;
  float alpha_T;
  float d_T;
  float chi2rphidofSum;
  float chi2rzdofSum;
  float bendchi2Sum;
  float MVA1Sum;
  float MVA2Sum;
  int numStubsSum;
  float delta_z;
  float delta_eta;
  float phi;
  Vertex_Parameters(Double_t x_dv_in, Double_t y_dv_in, Double_t z_dv_in, Track_Parameters a_in, Track_Parameters b_in, float score_in=-1, int inTraj_in=4):
    a(a_in),
    b(b_in)
  {
    x_dv = x_dv_in;
    y_dv = y_dv_in;
    z_dv = z_dv_in;
    score = score_in;
    tracks.push_back(a_in);
    tracks.push_back(b_in);
    inTraj = inTraj_in;
    std::valarray<float> p_trk_1 = calcPVec(a_in,x_dv_in,y_dv_in);
    std::valarray<float> p_trk_2 = calcPVec(b_in,x_dv_in,y_dv_in);
    std::valarray<float> p_tot = p_trk_1+p_trk_2;
    p_mag = TMath::Sqrt(pow(p_tot[0],2)+pow(p_tot[1],2));
    openingAngle = (p_trk_1[0]*p_trk_2[0]+p_trk_1[1]*p_trk_2[1]) / (TMath::Sqrt(pow(p_trk_1[0],2)+pow(p_trk_1[1],2))*TMath::Sqrt(pow(p_trk_2[0],2)+pow(p_trk_2[1],2)));
    R_T = TMath::Sqrt(pow(x_dv_in,2)+pow(y_dv_in,2));
    cos_T = (p_tot[0]*x_dv_in+p_tot[1]*y_dv_in)/(R_T*TMath::Sqrt(pow(p_tot[0],2)+pow(p_tot[1],2)));
    alpha_T = acos(cos_T);
    phi = atan2(p_tot[1],p_tot[0]);
    d_T = fabs(cos(phi)*y_dv_in-sin(phi)*x_dv_in);
    int ndof_1 = 2 * a_in.nstubs - 5;
    float chi2rphidof_1 = a_in.chi2rphi / ndof_1;
    float chi2rzdof_1 = a_in.chi2rz / ndof_1;
    float bendchi2_1 = a_in.bendchi2;
    int ndof_2 = 2 * b_in.nstubs - 5;
    float chi2rphidof_2 = b_in.chi2rphi / ndof_2;
    float chi2rzdof_2 = b_in.chi2rz / ndof_2;
    float bendchi2_2 = b_in.bendchi2;
    chi2rphidofSum = chi2rphidof_1 + chi2rphidof_2;
    chi2rzdofSum = chi2rzdof_1 + chi2rzdof_2;
    bendchi2Sum = bendchi2_1 + bendchi2_2;
    MVA1Sum = a_in.MVA1 + b_in.MVA1;
    MVA2Sum = a_in.MVA2 + b_in.MVA2;
    numStubsSum = a_in.nstubs + b_in.nstubs;
    p2_mag = pow(a_in.pt,2)+pow(b_in.pt,2);
    delta_z = fabs(a_in.z(x_dv_in,y_dv_in)-b_in.z(x_dv_in,y_dv_in));
    delta_eta = fabs(a_in.eta-b_in.eta);
  }

  void addTrack(Track_Parameters trk){
    tracks.push_back(trk);
    std::valarray<float> p_tot = {0,0};
    for(auto track : tracks){
      p_tot+= calcPVec(track,x_dv,y_dv);
    }
    p_mag = TMath::Sqrt(pow(p_tot[0],2)+pow(p_tot[1],2));
    cos_T = (p_tot[0]*x_dv+p_tot[1]*y_dv)/(R_T*TMath::Sqrt(pow(p_tot[0],2)+pow(p_tot[1],2)));
    alpha_T = acos(cos_T);
    phi = atan2(p_tot[1],p_tot[0]);
    d_T = fabs(cos(phi)*y_dv-sin(phi)*x_dv);
    int ndof = 2 * trk.nstubs - 5;
    float chi2rphidof = trk.chi2rphi / ndof;
    float chi2rzdof = trk.chi2rz / ndof;
    float bendchi2 = trk.bendchi2;
    chi2rphidofSum+= chi2rphidof;
    chi2rzdofSum+= chi2rzdof;
    bendchi2Sum+= bendchi2;
    numStubsSum+= trk.nstubs;
    p2_mag+= pow(trk.pt,2);
    MVA1Sum+= trk.MVA1;
    MVA2Sum+= trk.MVA2;
  }

  Vertex_Parameters(){};
  ~Vertex_Parameters(){};
};

constexpr bool operator==(const Vertex_Parameters& lhs, const Vertex_Parameters& rhs)
{
  return (lhs.x_dv==rhs.x_dv && lhs.y_dv==rhs.y_dv && lhs.z_dv==rhs.z_dv);
}

class DisplacedVertexProducer : public edm::global::EDProducer<> {
 public:
  explicit DisplacedVertexProducer(const edm::ParameterSet&);
  ~DisplacedVertexProducer() override {}

 private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

 private:
  const edm::EDGetTokenT<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>> ttTrackMCTruthToken_;
  const edm::EDGetTokenT<std::vector<TTTrack<Ref_Phase2TrackerDigi_> >> trackToken_;
  const std::string outputTrackCollectionName_;
  const std::string qualityAlgorithm_;
  const std::string ONNXmodel_;
  const std::string ONNXInputName_;
  const std::vector<std::string> featureNames_;
  std::unique_ptr<cms::Ort::ONNXRuntime> runTime_;
};

#endif
