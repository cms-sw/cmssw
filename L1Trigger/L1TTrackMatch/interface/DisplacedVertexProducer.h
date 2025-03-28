#ifndef __L1Trigger_L1TTrackMatch_DisplacedVertexProducer_h__
#define __L1Trigger_L1TTrackMatch_DisplacedVertexProducer_h__

#include "DataFormats/L1Trigger/interface/DisplacedVertex.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/Associations/interface/TTTrackAssociationMap.h"
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <valarray>
#include <ap_int.h>
#include "conifer.h"

using namespace std;

class Track_Parameters {
public:
  float pt;
  float d0;
  float z0;
  float eta;
  float phi;
  float charge;
  float rho;
  int index;
  float x0;
  float y0;
  int nstubs;
  float chi2rphi;
  float chi2rz;
  float bendchi2;
  float MVA1;

  float trackZAtVertex(float x, float y) {
    float t = sinh(eta);
    float r = sqrt(pow(x, 2) + pow(y, 2));
    return (z0 +
            (t * r *
             (1 + (pow(d0, 2) / pow(r, 2)) +
              (1.0 / 6.0) * pow(r / (2 * rho), 2))));  // can do higher order terms if necessary from displaced math
  }
  Track_Parameters(float pt_in,
                   float d0_in,
                   float z0_in,
                   float eta_in,
                   float phi_in,
                   float rho_in,
                   int index_in,
                   int nstubs_in,
                   float chi2rphi_in,
                   float chi2rz_in,
                   float bendchi2_in,
                   float MVA1_in) {
    pt = pt_in;
    d0 = d0_in;
    z0 = z0_in;
    eta = eta_in;
    phi = phi_in;
    if (rho_in > 0) {
      charge = 1;
    } else if (rho_in < 0) {
      charge = -1;
    } else {
      charge = 0;
    }
    index = index_in;
    rho = fabs(rho_in);
    x0 = (rho + charge * d0) * cos(phi - (charge * numbers::pi / 2));
    y0 = (rho + charge * d0) * sin(phi - (charge * numbers::pi / 2));
    nstubs = nstubs_in;
    chi2rphi = chi2rphi_in;
    chi2rz = chi2rz_in;
    bendchi2 = bendchi2_in;
    MVA1 = MVA1_in;
  }
  Track_Parameters() {};
  ~Track_Parameters() {};
};

inline std::valarray<float> calcPVec(Track_Parameters a, double_t v_x, double_t v_y) {
  std::valarray<float> r_vec = {float(v_x) - a.x0, float(v_y) - a.y0};
  std::valarray<float> p_vec = {-r_vec[1], r_vec[0]};
  if (a.charge > 0) {
    p_vec *= -1;
  }
  if ((p_vec[0] != 0.0) || (p_vec[1] != 0.0)) {
    p_vec /= sqrt(pow(p_vec[0], 2) + pow(p_vec[1], 2));
  }
  p_vec *= a.pt;
  return p_vec;
}

class Vertex_Parameters {
public:
  Double_t x_dv;
  Double_t y_dv;
  Double_t z_dv;
  Track_Parameters a;
  Track_Parameters b;
  std::vector<Track_Parameters> tracks = {};
  float p_mag;
  float p2_mag;
  float openingAngle = -999.0;
  float R_T;
  float cos_T = -999.0;
  float d_T;
  float delta_z;
  float phi;
  Vertex_Parameters(Double_t x_dv_in, Double_t y_dv_in, Double_t z_dv_in, Track_Parameters a_in, Track_Parameters b_in)
      : a(a_in), b(b_in) {
    x_dv = x_dv_in;
    y_dv = y_dv_in;
    z_dv = z_dv_in;
    tracks.push_back(a_in);
    tracks.push_back(b_in);
    std::valarray<float> p_trk_1 = calcPVec(a_in, x_dv_in, y_dv_in);
    std::valarray<float> p_trk_2 = calcPVec(b_in, x_dv_in, y_dv_in);
    std::valarray<float> p_tot = p_trk_1 + p_trk_2;
    p_mag = sqrt(pow(p_tot[0], 2) + pow(p_tot[1], 2));
    if (((p_trk_1[0] != 0.0) || (p_trk_1[1] != 0.0)) && ((p_trk_2[0] != 0.0) || (p_trk_2[1] != 0.0))) {
      openingAngle = (p_trk_1[0] * p_trk_2[0] + p_trk_1[1] * p_trk_2[1]) /
                     (sqrt(pow(p_trk_1[0], 2) + pow(p_trk_1[1], 2)) * sqrt(pow(p_trk_2[0], 2) + pow(p_trk_2[1], 2)));
    }
    R_T = sqrt(pow(x_dv_in, 2) + pow(y_dv_in, 2));
    if ((R_T != 0.0) && ((p_tot[0] != 0.0) || (p_tot[1] != 0.0))) {
      cos_T = (p_tot[0] * x_dv_in + p_tot[1] * y_dv_in) / (R_T * sqrt(pow(p_tot[0], 2) + pow(p_tot[1], 2)));
    }
    phi = atan2(p_tot[1], p_tot[0]);
    d_T = fabs(cos(phi) * y_dv_in - sin(phi) * x_dv_in);
    p2_mag = pow(a_in.pt, 2) + pow(b_in.pt, 2);
    delta_z = fabs(a_in.trackZAtVertex(x_dv_in, y_dv_in) - b_in.trackZAtVertex(x_dv_in, y_dv_in));
  }

  Vertex_Parameters() {};
  ~Vertex_Parameters() {};
};

class DisplacedVertexProducer : public edm::global::EDProducer<> {
public:
  explicit DisplacedVertexProducer(const edm::ParameterSet &);
  ~DisplacedVertexProducer() override = default;
  typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;
  double FloatPtFromBits(const L1TTTrackType &) const;
  double FloatEtaFromBits(const L1TTTrackType &) const;
  double FloatPhiFromBits(const L1TTTrackType &) const;
  double FloatZ0FromBits(const L1TTTrackType &) const;
  double FloatD0FromBits(const L1TTTrackType &) const;
  int ChargeFromBits(const L1TTTrackType &) const;

private:
  const edm::EDGetTokenT<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>> ttTrackMCTruthToken_;
  const edm::EDGetTokenT<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>> trackToken_;
  const edm::EDGetTokenT<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>> trackGTTToken_;
  const std::string outputVertexCollectionName_;
  const edm::FileInPath model_;
  const bool runEmulation_;
  const edm::ParameterSet cutSet_;
  const double chi2rzMax_, promptMVAMin_, ptMin_, etaMax_, dispD0Min_, promptMVADispTrackMin_, overlapEtaMin_,
      overlapEtaMax_;
  const int overlapNStubsMin_;
  const double diskEtaMin_, diskD0Min_, barrelD0Min_, RTMin_, RTMax_;
};

#endif
