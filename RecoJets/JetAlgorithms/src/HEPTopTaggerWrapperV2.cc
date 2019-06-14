// 2011 Christopher Vermilion
//
//----------------------------------------------------------------------
//  This file is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 3 of the License, or
//  (at your option) any later version.
//
//  This file is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  The GNU General Public License is available at
//  http://www.gnu.org/licenses/gpl.html or you can write to the Free Software
//  Foundation, Inc.:
//      59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//----------------------------------------------------------------------

#include "RecoJets/JetAlgorithms/interface/HEPTopTaggerWrapperV2.h"

#include "fastjet/Error.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/PseudoJet.hh"
#include "fastjet/tools/Pruner.hh"
#include "fastjet/tools/Filter.hh"

#include <cmath>
#include <limits>
#include <cassert>
using namespace std;

#include "RecoJets/JetAlgorithms/interface/HEPTopTaggerV2.h"

FASTJET_BEGIN_NAMESPACE

// Expected R_min for tops (as function of filtered initial fatjet pT in GeV (using CA, R=0.2, n=10)
// From ttbar sample, phys14, n20, bx25
// matched to hadronically decaying top with delta R < 1.2 and true top pT > 200
// Cuts are: fW < 0.175 and  m_top = 120..220
// Input objects are packed pfCandidates with CHS
double R_min_expected_function(double x) {
  if (x > 700)
    x = 700;

  double A = -9.42052;
  double B = 0.202773;
  double C = 4498.45;
  double D = -1.05737e+06;
  double E = 9.95494e+07;

  return A + B * sqrt(x) + C / x + D / (x * x) + E / (x * x * x);
}

//------------------------------------------------------------------------
// returns the tagged PseudoJet if successful, 0 otherwise
//  - jet   the PseudoJet to tag
PseudoJet HEPTopTaggerV2::result(const PseudoJet &jet) const {
  // make sure that there is a "regular" cluster sequence associated
  // with the jet. Note that we also check it is valid (to avoid a
  // more criptic error later on)
  if (!jet.has_valid_cluster_sequence()) {
    throw Error("HEPTopTagger can only be applied on jets having an associated (and valid) ClusterSequence");
  }

  external::HEPTopTaggerV2 tagger(jet);

  external::HEPTopTaggerV2 best_tagger;

  // translate the massRatioWidth (which should be the half-width given in %)
  // to values useful for the A-shape cuts
  double mw_over_mt = 80.4 / 172.3;
  double ratio_min = mw_over_mt * (100. - massRatioWidth_) / 100.;
  double ratio_max = mw_over_mt * (100. + massRatioWidth_) / 100.;

  // Unclustering, Filtering & Subjet Settings
  tagger.set_max_subjet_mass(subjetMass_);
  tagger.set_mass_drop_threshold(muCut_);
  tagger.set_filtering_R(filtR_);
  tagger.set_filtering_n(filtN_);
  tagger.set_filtering_minpt_subjet(minSubjetPt_);

  // Optimal R
  tagger.do_optimalR(DoOptimalR_);
  tagger.set_optimalR_reject_minimum(optRrejectMin_);

  // How to select among candidates
  tagger.set_mode((external::Mode)mode_);

  // Requirements to accept a candidate
  tagger.set_top_minpt(minCandPt_);
  tagger.set_top_mass_range(minCandMass_, maxCandMass_);
  tagger.set_mass_ratio_cut(minM23Cut_, minM13Cut_, maxM13Cut_);
  tagger.set_mass_ratio_range(ratio_min, ratio_max);

  // Set function to calculate R_min_expected
  tagger.set_optimalR_calc_fun(R_min_expected_function);

  double qweight = -1;
  double qepsilon = -1;
  double qsigmaM = -1;

  if (DoQjets_) {
    int niter(100);
    double q_zcut(0.1);
    double q_dcut_fctr(0.5);
    double q_exp_min(0.);
    double q_exp_max(0.);
    double q_rigidity(0.1);
    double q_truncation_fctr(0.0);

    double weight_q1 = -1.;
    double m_sum = 0.;
    double m2_sum = 0.;
    int qtags = 0;

    tagger.set_qjets(q_zcut, q_dcut_fctr, q_exp_min, q_exp_max, q_rigidity, q_truncation_fctr);
    tagger.set_qjets_rng(engine_);
    tagger.do_qjets(true);
    tagger.run();

    for (int iq = 0; iq < niter; iq++) {
      tagger.run();
      if (tagger.is_tagged()) {
        qtags++;
        m_sum += tagger.t().m();
        m2_sum += tagger.t().m() * tagger.t().m();
        if (tagger.q_weight() > weight_q1) {
          best_tagger = tagger;
          weight_q1 = tagger.q_weight();
        }
      }
    }

    tagger = best_tagger;
    qweight = weight_q1;
    qepsilon = float(qtags) / float(niter);

    // calculate width of tagged mass distribution if we have at least one candidate
    if (qtags > 0) {
      double mean_m = m_sum / qtags;
      double mean_m2 = m2_sum / qtags;
      qsigmaM = sqrt(mean_m2 - mean_m * mean_m);
    }
  } else {
    tagger.run();
  }

  // Requires:
  //   - top mass window
  //   - mass ratio cuts
  //   - minimal candidate pT
  // If this is not intended: use loose top mass and ratio windows
  if (!tagger.is_tagged())
    return PseudoJet();

  // create the result and its structure
  const JetDefinition::Recombiner *rec = jet.associated_cluster_sequence()->jet_def().recombiner();

  const vector<PseudoJet> &subjets = tagger.top_subjets();
  assert(subjets.size() == 3);

  PseudoJet non_W = subjets[0];
  PseudoJet W1 = subjets[1];
  PseudoJet W2 = subjets[2];
  PseudoJet W = join(subjets[1], subjets[2], *rec);

  PseudoJet result = join<HEPTopTaggerV2Structure>(W1, W2, non_W, *rec);
  HEPTopTaggerV2Structure *s = (HEPTopTaggerV2Structure *)result.structure_non_const_ptr();

  s->_fj_mass = jet.m();
  s->_fj_pt = jet.perp();
  s->_fj_eta = jet.eta();
  s->_fj_phi = jet.phi();

  s->_top_mass = tagger.t().m();
  s->_pruned_mass = tagger.pruned_mass();
  s->_unfiltered_mass = tagger.unfiltered_mass();
  s->_fRec = tagger.f_rec();
  s->_mass_ratio_passed = tagger.is_masscut_passed();

  if (DoOptimalR_) {
    s->_tau1Unfiltered = tagger.nsub_unfiltered(1);
    s->_tau2Unfiltered = tagger.nsub_unfiltered(2);
    s->_tau3Unfiltered = tagger.nsub_unfiltered(3);
    s->_tau1Filtered = tagger.nsub_filtered(1);
    s->_tau2Filtered = tagger.nsub_filtered(2);
    s->_tau3Filtered = tagger.nsub_filtered(3);
  }

  s->_qweight = qweight;
  s->_qepsilon = qepsilon;
  s->_qsigmaM = qsigmaM;

  if (DoOptimalR_) {
    s->_ropt = tagger.Ropt();
    s->_roptCalc = tagger.Ropt_calc();
    s->_ptForRoptCalc = tagger.pt_for_Ropt_calc();
  } else {
    s->_ropt = -1;
    s->_roptCalc = -1;
    s->_ptForRoptCalc = -1;
  }

  // Removed selectors as all cuts are applied in HTT

  return result;
}

FASTJET_END_NAMESPACE
