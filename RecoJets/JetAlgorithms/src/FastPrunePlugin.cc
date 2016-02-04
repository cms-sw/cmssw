// $Id: FastPrunePlugin.cc,v 1.1 2011/04/25 04:19:54 srappocc Exp $
///////////////////////////////////////////////////////////////////////////////
//
// Implements the FastPrunePlugin class.  See FastPrunePlugin.hh for a
//   description.
//
///////////////////////////////////////////////////////////////////////////////

#include "RecoJets/JetAlgorithms/interface/FastPrunePlugin.hh"

#include <sstream>
#include <cmath>
#include <vector>
#include <algorithm>
using namespace std;

using namespace fastjet;

FastPrunePlugin::FastPrunePlugin (const JetDefinition & find_definition,
                                  const JetDefinition & prune_definition,
                                  const double & zcut,
                                  const double & Rcut_factor) :
		_find_definition(find_definition),
		_prune_definition(prune_definition),
		_minpT(20.),
		_unpruned_seq(NULL),
		_pruned_recombiner(0),
		_cut_setter(new DefaultCutSetter(zcut, Rcut_factor))
{
	// If the passed prune_definition (copied into _prune_def) has an external
	// recombiner, use it.  Otherwise, create our own DefaultRecombiner.  We
	// could just point to _prune_definition's DefaultRecombiner, but FastJet
	// actually sets this to use external_scheme later when we set
	// _prune_definition's Recombiner to be a PrunedRecombiner.  (Calling
	// JetDefinition::set_recombiner() also calls jetdef._default_recombiner = 
	// DefaultRecombiner(external_scheme) as a sanity check to keep you from
	// using it.)
	RecombinationScheme scheme = _prune_definition.recombination_scheme();
	if (scheme == external_scheme)
		_pruned_recombiner = new PrunedRecombiner(_prune_definition.recombiner(), zcut, 0.0);
	else
		_pruned_recombiner = new PrunedRecombiner(scheme, zcut, 0.0);
}

FastPrunePlugin::FastPrunePlugin (const JetDefinition & find_definition,
                                  const JetDefinition & prune_definition,
                                  const JetDefinition::Recombiner* recomb,
                                  const double & zcut,
                                  const double & Rcut_factor) :
		_find_definition(find_definition),
		_prune_definition(prune_definition),
		_minpT(20.),
		_unpruned_seq(NULL),
		_pruned_recombiner(new PrunedRecombiner(recomb, zcut, 0.0)),
		_cut_setter(new DefaultCutSetter(zcut, Rcut_factor))
{}

FastPrunePlugin::FastPrunePlugin (const JetDefinition & find_definition,
                                  const JetDefinition & prune_definition,
                                  CutSetter* const cut_setter) :
		_find_definition(find_definition),
		_prune_definition(prune_definition),
		_minpT(20.),
		_unpruned_seq(NULL),
		_pruned_recombiner(0),
		_cut_setter(cut_setter)
{
	// See comments in first constructor
	RecombinationScheme scheme = _prune_definition.recombination_scheme();
	if (scheme == external_scheme)
		_pruned_recombiner = new PrunedRecombiner(_prune_definition.recombiner());
	else
		_pruned_recombiner = new PrunedRecombiner(scheme);
}

FastPrunePlugin::FastPrunePlugin (const JetDefinition & find_definition,
                                  const JetDefinition & prune_definition,
                                  CutSetter* const cut_setter,
                                  const JetDefinition::Recombiner* recomb) :
		_find_definition(find_definition),
		_prune_definition(prune_definition),
		_minpT(20.),
		_unpruned_seq(NULL),
		_pruned_recombiner(new PrunedRecombiner(recomb)),
		_cut_setter(cut_setter)
{}

string FastPrunePlugin::description () const {
	ostringstream desc;

	desc << "Pruned jet algorithm \n"
	     << "----------------------- \n"
	     << "Jet finder: " << _find_definition.description()
	     << "\n----------------------- \n"
	     << "Prune jets with: " << _prune_definition.description()
	     << "\n----------------------- \n"
	     << "Pruning parameters: "
	     << "zcut = " << _cut_setter->zcut << ", "
	     << "Rcut_factor = ";
	if (DefaultCutSetter *cs = dynamic_cast<DefaultCutSetter*>(_cut_setter))
		desc << cs->Rcut_factor;
	else
		desc << "[dynamic]";
	desc << "\n"
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
void FastPrunePlugin::run_clustering(ClusterSequence & input_seq) const {

	// this will be filled in the output_mergings() step
	_pruned_subjets.clear();

	vector<PseudoJet> inputs = input_seq.jets();
	// Record user_index's so we can match PJ's in pruned jets to PJ's in
	//   input_seq.
	// Use i+1 to distinguish from the default, which in some places appears to
	//   be 0.
	// Note that we're working with a local copy of the input jets, so we don't
	//   change the indices in input_seq.
	for (unsigned int i = 0; i < inputs.size(); i++)
		inputs[i].set_user_index(i+1);

	// ClusterSequence for initial (unpruned) jet finding
	if(_unpruned_seq) delete _unpruned_seq;
	_unpruned_seq = new ClusterSequence(inputs, _find_definition);

	// now, for each jet, construct a pruned version:
	vector<PseudoJet> unpruned_jets = 
	                  sorted_by_pt(_unpruned_seq->inclusive_jets(_minpT));

	for (unsigned int i = 0; i < unpruned_jets.size(); i++) {
		_cut_setter->SetCuts(unpruned_jets[i], *_unpruned_seq);
		_pruned_recombiner->reset(_cut_setter->zcut, _cut_setter->Rcut);
		_prune_definition.set_recombiner(_pruned_recombiner);

		// temp way to get constituents, to compare to new version
		vector<PseudoJet> constituents;
		for (size_t j = 0; j < inputs.size(); ++j)
			if (_unpruned_seq->object_in_jet(inputs[j], unpruned_jets[i])) {
				constituents.push_back(inputs[j]);
			}
		ClusterSequence pruned_seq(constituents, _prune_definition);
		
		vector<int> pruned_PJs = _pruned_recombiner->pruned_pseudojets();
		_output_mergings(pruned_seq, pruned_PJs, input_seq);
	}
}


// Takes the merging structure of "in_seq" and feeds this into "out_seq" using 
//   _plugin_record_{ij,iB}_recombination().
// PJ's in the input CS should have user_index's that are their (index+1) in 
//	  _jets() in the output CS (the output CS should be the input CS to
//	  run_clustering()).
// This allows us to build up the same jet in out_seq as already exists in
//   in_seq.								 														 
void FastPrunePlugin::_output_mergings(ClusterSequence & in_seq,
                                       vector<int> & pruned_pseudojets,
                                       ClusterSequence & out_seq) const {
	// vector to store the pruned subjets for this jet
	vector<PseudoJet> temp_pruned_subjets;

	vector<PseudoJet> p = in_seq.jets();

	// sort this vector so we can binary search it
	sort(pruned_pseudojets.begin(), pruned_pseudojets.end());

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
		int jet_index = iter->jetp_index;
		int parent1 = iter->parent1;
		int parent2 = iter->parent2;
		int parent1_index = p[hist[iter->parent1].jetp_index].user_index() - 1;

		if (parent2 == ClusterSequence::BeamJet) {
			out_seq.plugin_record_iB_recombination(parent1_index, iter->dij);
		} else {
			int parent2_index = p[hist[parent2].jetp_index].user_index() - 1;
			
			// Check if either parent is stored in pruned_pseudojets
			//   Note that it is the history index that is stored
			if (binary_search(pruned_pseudojets.begin(), pruned_pseudojets.end(),
                        parent2)) {
				// pruned away parent2 -- just give child parent1's index
				p[jet_index].set_user_index(p[hist[parent1].jetp_index].user_index());
				temp_pruned_subjets.push_back(out_seq.jets()[parent2_index]);
			} else if (binary_search(pruned_pseudojets.begin(), pruned_pseudojets.end(),
                             parent1)) {
				// pruned away parent1 -- just give child parent2's index
				p[jet_index].set_user_index(p[hist[parent2].jetp_index].user_index());
				temp_pruned_subjets.push_back(out_seq.jets()[parent1_index]);
			} else {
				// no pruning -- record combination and index for child
				out_seq.plugin_record_ij_recombination(parent1_index, parent2_index,
                                               iter->dij, new_jet_index);
				p[jet_index].set_user_index(new_jet_index + 1);
			}
		}
	}
	
	_pruned_subjets.push_back(temp_pruned_subjets);
}


