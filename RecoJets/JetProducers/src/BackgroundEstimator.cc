#include "RecoJets/JetProducers/interface/BackgroundEstimator.h"
#include <fastjet/ClusterSequenceAreaBase.hh>
#include <iostream>

using namespace fastjet;
using namespace std;

//---------------------------------------------------------------------
// \class BackgroundEstimator
// Class to estimate the density of the background per unit area
//
// The default behaviour of this class is to compute the global 
// properties of the background as it is done in ClusterSequenceArea.
// On top of that, we provide methods to specify an explicit set of
// jets to use or a list of jets to exclude.
// We also provide all sorts of additional information regarding
// the background estimation like the jets that have been used or
// the number of pure-ghost jets.
//---------------------------------------------------------------------

// default ctor
//  - csa      the ClusterSequenceArea to use
//  - range    the range over which jets will be considered
BackgroundEstimator::BackgroundEstimator(const ClusterSequenceAreaBase &csa, const RangeDefinition &range)
  : _csa(csa), _range(range){
  reset();
}

// default dtor
BackgroundEstimator::~BackgroundEstimator(){

}

// reset to default values
// set the list of included jets to the inclusive jets and clear the excluded ones
void BackgroundEstimator::reset(){
  // set the list of included jets to the inclusive jets 
  _included_jets = _csa.inclusive_jets();
  _all_from_inclusive = true;
  //set_included_jets(_csa.inclusive_jets());

  // clear the list of explicitly excluded jets
  _excluded_jets.clear();
  //set_excluded_jets(vector<PseudoJet>());

  // set the remaining default parameters
  set_use_area_4vector();  // true by default

  // reset the computed values
  _median_rho = _sigma = _mean_area = 0.0;
  _n_jets_used = _n_jets_excluded = _n_empty_jets = 0;
  _empty_area = 0.0;
  _uptodate = false;
}


// do the actual job
void BackgroundEstimator::_compute(){
  //TODO: check that the alg is OK for median computation
  //_check_jet_alg_good_for_median();

  // fill the vector of pt/area with the jets 
  //  - in included_jets
  //  - not in excluded_jets
  //  - in the range
  vector<double> pt_over_areas;
  double total_area  = 0.0;
  
  _n_jets_used = 0;
  _n_jets_excluded = 0;

  for (unsigned i = 0; i < _included_jets.size(); i++) {
    const PseudoJet & current_jet = _included_jets[i];

    // check that the jet is not explicitly excluded
    // we'll compare them using their cluster_history_index
    bool excluded = false;
    int ref_idx = current_jet.cluster_hist_index();
    for (unsigned int j = 0; j < _excluded_jets.size(); j++)
      excluded |= (_excluded_jets[j].cluster_hist_index() == ref_idx);

    // check if the jet is in the range
    if (_range.is_in_range(current_jet)){
      if (excluded){
	// keep track of the explicitly excluded jets
	_n_jets_excluded++;
      } else {
	double this_area = (_use_area_4vector) 
	  ? _csa.area_4vector(current_jet).perp()
	  : _csa.area(current_jet); 
	
	pt_over_areas.push_back(current_jet.perp()/this_area);
	total_area  += this_area;
	_n_jets_used++;
      }
    }
  }
  
  // there is nothing inside our region, so answer will always be zero
  if (pt_over_areas.size() == 0) {
    _median_rho = 0.0;
    _sigma      = 0.0;
    _mean_area  = 0.0;
    return;
  }

  // get median (pt/area) [this is the "old" median definition. It considers
  // only the "real" jets in calculating the median, i.e. excluding the
  // only-ghost ones; it will be supplemented with more info below]
  sort(pt_over_areas.begin(), pt_over_areas.end());

  // determine the number of empty jets
  _empty_area = 0.0;
  _n_empty_jets = 0.0;
  if (_csa.has_explicit_ghosts()) {
    _empty_area = 0.0;
    _n_empty_jets = 0;
  } else if (_all_from_inclusive) {
    _empty_area = _csa.empty_area(_range);
    _n_empty_jets = _csa.n_empty_jets(_range);
  } else {
    _empty_area = _csa.empty_area_from_jets(_included_jets, _range);
    _mean_area = total_area / _n_jets_used; // temporary value
    _n_empty_jets = _empty_area / _mean_area;
  }

  double total_njets = _n_jets_used + _n_empty_jets;
  total_area  += _empty_area;


  // now get the median & error, accounting for empty jets
  // define the fractions of distribution at median, median-1sigma
  double posn[2] = {0.5, (1.0-0.6827)/2.0};
  double res[2];

  for (int i = 0; i < 2; i++) {
    double nj_median_pos = (total_njets-1)*posn[i] - _n_empty_jets;
    double nj_median_ratio;
    if (nj_median_pos >= 0 && pt_over_areas.size() > 1) {
      int int_nj_median = int(nj_median_pos);
      nj_median_ratio =
        pt_over_areas[int_nj_median] * (int_nj_median+1-nj_median_pos)
        + pt_over_areas[int_nj_median+1] * (nj_median_pos - int_nj_median);
    } else {
      nj_median_ratio = 0.0;
    }
    res[i] = nj_median_ratio;
  }

  // store the results
  double error  = res[0] - res[1];
  _median_rho = res[0];
  _mean_area  = total_area / total_njets;
  _sigma      = error * sqrt(_mean_area);

  // record that the computation has been performed  
  _uptodate = true;
}




