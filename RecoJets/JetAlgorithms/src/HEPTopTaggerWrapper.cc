// Copyright (c) 2011 Christopher Vermilion
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

#include "RecoJets/JetAlgorithms/interface/HEPTopTaggerWrapper.h"

#include <fastjet/Error.hh>
#include <fastjet/JetDefinition.hh>
#include <fastjet/ClusterSequence.hh>

#include <limits>
#include <cassert>
using namespace std;

// namespace hack so that this tagger can have the same name as the core code
namespace external {
#include "RecoJets/JetAlgorithms/interface/HEPTopTagger.h"
}

FASTJET_BEGIN_NAMESPACE

//------------------------------------------------------------------------
// returns the tagged PseudoJet if successful, 0 otherwise
//  - jet   the PseudoJet to tag
PseudoJet HEPTopTagger::result(const PseudoJet & jet) const{
  // make sure that there is a "regular" cluster sequence associated
  // with the jet. Note that we also check it is valid (to avoid a
  // more criptic error later on)
  if (!jet.has_valid_cluster_sequence()){
    throw Error("HEPTopTagger can only be applied on jets having an associated (and valid) ClusterSequence");
  }

  external::HEPTopTagger tagger(*jet.associated_cluster_sequence(), jet);
  tagger.set_top_range(0.0, 10000.0); // don't do top mass cut; this can be applied later
  tagger.set_mass_drop_threshold(_mass_drop_threshold);
  tagger.set_max_subjet_mass(_max_subjet_mass);

  tagger.run_tagger();
  
  // check that we passed the tagger; if not return a blank PseudoJet
  if (_use_subjet_mass_cuts) {
    if (!tagger.is_masscut_passed()) // encompasses is_maybe_top() plus subjet mass cuts
      return PseudoJet();
  } else if (!tagger.is_maybe_top())
    return PseudoJet();
  
  // create the result and its structure
  const JetDefinition::Recombiner *rec
    = jet.associated_cluster_sequence()->jet_def().recombiner();

  const vector<PseudoJet>& subjets = tagger.top_subjets();
  assert(subjets.size() == 3);

  PseudoJet non_W = subjets[0];
  PseudoJet W1 = subjets[1];
  PseudoJet W2 = subjets[2];
  PseudoJet W = join(subjets[1], subjets[2], *rec);


  PseudoJet result = join<HEPTopTaggerStructure>( W1, W2, non_W, *rec);
  //HEPTopTaggerStructure *s = (HEPTopTaggerStructure*) result.structure_non_const_ptr();
//  s->_cos_theta_w = _cos_theta_W(result);

  // Check selectors to see if identified top, W pass and cuts
  //
  // Note that we could perhaps ensure this cut before constructing
  // the result structure but this has the advantage that the top
  // 4-vector is already available and does not have to de re-computed
  if (! _top_selector.pass(result) || ! _W_selector.pass(W)) {
    result *= 0.0;
  }

  return result;
}




FASTJET_END_NAMESPACE
