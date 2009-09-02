///////////////////////////////////////////////////////////////////////////////
///
// PrunedRecombPlugin.cc
// Last update: 5/28/09 CKV
// PrunedRecomb version: 0.2.0
// Author: Christopher Vermilion <verm@u.washington.edu>
//
// Implements the PrunedRecombPlugin class.  See PrunedRecombPlugin.h for a 
//   description.
//
///////////////////////////////////////////////////////////////////////////////

#include "PrunedRecombPlugin.h"

#include <sstream>
#include <cmath>
using namespace std;

FASTJET_BEGIN_NAMESPACE      // defined in fastjet/internal/base.hh

string PrunedRecombPlugin::description () const {
	ostringstream desc;
  
	desc << "Pruned jet algorithm \n"
	     << "----------------------- \n"
	     << "Jet finder: " << _find_definition.description()
	     << "\n----------------------- \n"
	     << "Prune jets with: " << _prune_definition.description()
	     << "\n----------------------- \n"
		  << "Pruning parameters: "
	     << "zcut = " << _zcut << ", "
	     << "Rcut_factor = " << _Rcut_factor << "\n"
		  << "----------------------- \n" ;

  return desc.str();
}

// Meat of the plugin.  This function takes the input particles from input_seq
//   and calls plugin_record_{ij, iB}_recombination repeatedly such that
//   input_seq holds a cluster sequence corresponding to pruned jets
// This function first creates unpruned jets using _find_definition, then uses
//   the prune_definition, along with the PrunedRecombiner, on each jet to
//   produce a pruned jet.  For each of these jets, output_mergings() is called,
//   which reads the history of the pruned sequence and calls the
//   plugin_record functions of input_seq to match this history.
// Branches that are pruned appear in the history as PseudoJets with no
//   children.  For this reason, only inclusive_jets() is sensible to use with
//   the final ClusterSequence.  The substructure, e.g., constituents() of a
//   pruned jet will not include the pruned away branchings.
void PrunedRecombPlugin::run_clustering(ClusterSequence & input_seq) const {
		
	vector<PseudoJet> inputs = input_seq.jets();
	// Record user_index's so we can match PJ's in pruned jets to PJ's in 
	//   input_seq.
	// Use i+1 to distinguish from the default, which in some places appears to
	//   be 0.
	for (unsigned int i = 0; i < inputs.size(); i++)
		inputs[i].set_user_index(i+1);
		
	// ClusterSequence for initial (unpruned) jet finding
	ClusterSequence unpruned_seq(inputs, _find_definition);
	
	// now, for each jet, construct a pruned version:
	vector<PseudoJet> unpruned_jets
	                    = sorted_by_pt(unpruned_seq.inclusive_jets(0.0));
	for (unsigned int i = 0; i < unpruned_jets.size(); i++) {
		double angle = _characteristic_angle(unpruned_jets[i], unpruned_seq);
		// PrunedRecombiner is just DefaultRecombiner but vetoes on recombinations
		//   that fail a pruning test.  Note that Rcut is proportional to the
		//   characteristic angle of the jet.
		JetDefinition::Recombiner *pruned_recombiner = 
		                          new PrunedRecombiner(_zcut, _Rcut_factor*angle);
		_prune_definition.set_recombiner(pruned_recombiner);
		ClusterSequence pruned_seq
		                       (unpruned_seq.constituents(unpruned_jets[i]), 
		                        _prune_definition);
		delete pruned_recombiner;
		
		// Cluster all the particles in this jet into 1 pruned jet.
		// It is possible, though rare, to have a particle or two not clustered 
		//   into the jet even with R = pi/2.  It doesn't have to be the first 
		//  element of pruned_jets!  Sorting by pT and taking [0] should work...
		vector<PseudoJet> pruned_jets = pruned_seq.inclusive_jets(0.0);
		PseudoJet pruned_jet = sorted_by_pt(pruned_jets)[0];

		_output_mergings(pruned_seq, input_seq);
	}
}


// Takes the merging structure of "in_seq" and feeds this into "out_seq" using 
//   _plugin_record_{ij,iB}_recombination().
// PJ's in the input CS should have user_index's that are their (index+1) in 
//	  _jets() in the output CS (the output CS should be the input CS to
//	  run_clustering()).
// This allows us to build up the same jet in out_seq as already exists in
//   in_seq.								 														 
void PrunedRecombPlugin::_output_mergings(ClusterSequence & in_seq,
                                               ClusterSequence & out_seq) const
{
	vector<PseudoJet> p = in_seq.jets();

	// get the history from in_seq
	const vector<ClusterSequence::history_element> & hist = in_seq.history();
	vector<ClusterSequence::history_element>::const_iterator
		iter = hist.begin();
	
	// skip particle input elements
	while (iter->parent1 == ClusterSequence::InexistentParent)
		iter++;
	
	// Walk through history.  PseudoJets in in_seq should have a user_index
	//   corresponding to their index in out_seq.  Note that when we create a 
	//   new PJ via record_ij we need to set the user_index of our local copy.
	for (; iter != hist.end(); iter++) {
		int new_jet_index = -1;
		int i1 = p[hist[iter->parent1].jetp_index].user_index() - 1;
		if (iter->parent2 == ClusterSequence::BeamJet) {
			out_seq.plugin_record_iB_recombination(i1, iter->dij);
		} else {
			int i2 = p[hist[iter->parent2].jetp_index].user_index() - 1;
			
			// Check if either parent is equal to the child, indicating pruning.
			//  There is probably a smarter way to keep track of this.
			if (p[iter->jetp_index].e() - p[hist[iter->parent1].jetp_index].e() == 0)
				// pruned away parent2 -- just give child parent1's index
				p[iter->jetp_index].set_user_index(p[hist[iter->parent1].jetp_index].user_index());
			else if (p[iter->jetp_index].e() - p[hist[iter->parent2].jetp_index].e() == 0)
				// pruned away parent1 -- just give child parent2's index
				p[iter->jetp_index].set_user_index(p[hist[iter->parent2].jetp_index].user_index());
			else {
				// no pruning -- record combination and index for child
				out_seq.plugin_record_ij_recombination(i1, i2, iter->dij,
			                                       new_jet_index);
				p[iter->jetp_index].set_user_index(new_jet_index + 1);
			}
		}
	}
}



// Helper function to find the characteristic angle of a given jet.
// There are a few reasonable choices for this; we use 2*m/pT.
//   If no parents, return -1.0.
double PrunedRecombPlugin::_characteristic_angle (const PseudoJet & jet, 
                                           const ClusterSequence & seq) const
{
	PseudoJet p1, p2;
	if (! seq.has_parents(jet, p1, p2))
		return -1.0;
	else
		//return sqrt(p1.squared_distance(p2));  // DeltaR
		return 2.0*jet.m()/jet.perp();
}


FASTJET_END_NAMESPACE      // defined in fastjet/internal/base.hh
