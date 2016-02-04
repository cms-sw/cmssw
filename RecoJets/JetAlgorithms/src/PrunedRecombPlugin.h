/*

Original source:
http://www.phys.washington.edu/groups/lhcti/pruning/FastJetPlugin/
for physics, see
http://arxiv.org/abs/0903.5081

Some minor adaptions for integration into CMSSW by Jochen Ott.

*/
///////////////////////////////////////////////////////////////////////////////
//
// PrunedRecombPlugin.hh
// Last update: 5/29/09 CKV
// PrunedRecomb version: 0.2.0
// Author: Christopher Vermilion <verm@u.washington.edu>
//
// -------------------
//
// Defines the PrunedRecomb Plugin to FastJet.  This is the "pruned" version of
//   a recombination algorithm specifed in the constructor (Cambridge/Aachen and
//   kT are both sensible choices), as described in arXiv:0903.5081.
//
// The helper class PrunedRecombiner is also defined below.
//
// When a ClusterSequence created with this plugin calls "run_clustering" on a
//   set of input particles, the following occurs:
//
// 1) Normal jets are formed using the JetDefintion _find_definition.
// 2) For each of these jets, create a second ClusterSequence with the JetDef.
//      _prune_definition and the Recombiner PrunedRecombiner.
//      See comments below for more on how this works.  The pruned jet is the
//      original jet with certain branches removed.
// 3) This creates a pruned jet.  For each of these, the helper function
//      output_mergings takes the cluster sequence from the pruned jet and
//      feeds it into the input ClusterSequence with the 
//      plugin_record_{ij,iB}_recombination functions.  This leaves the
//      ClusterSequence holding pruned versions of all the original jets.
// 4) Pruned away branches are marked in the history as entries that are not
//      not subsequently recombined (i.e., child == Invalid).  This means that
//      only inclusive_jets(ptcut) makes sense.  Future versions of this plugins
//      will hopefully include an Extras() object with the ClusterSequence so 
//      that the pruned branches can be associated with the points they were 
//      pruned from.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef RecoJetsJetAlgorithms_PRUNEDRECOMBPLUGIN_H_
#define RecoJetsJetAlgorithms_PRUNEDRECOMBPLUGIN_H_

#include "fastjet/ClusterSequence.hh"
#include "fastjet/JetDefinition.hh"

#include <string>

FASTJET_BEGIN_NAMESPACE      // defined in fastjet/internal/base.hh

// forward declaration since we use this in the plugin
//class PrunedRecombiner;

class PrunedRecombPlugin : public JetDefinition::Plugin {
public:

	/// constructor for the PrunedRecombPlugin, whose arguments have the
	/// following meaning:
	///
	///   - find_definition, which is a JetDefinition object, specifies which
	///     jet algorithm to run to find jets.
	///
	///   - prune_definition is the algorithm used when pruning the found jet
	///     (typically the same as find_algorithm).  Also a JetDefinition.
	///     The PrunedRecombiner will be used in this definition.
	///
	///   - zcut and Rcut_factor are parameters to the pruning procedure.  
	///     Jet alg. prune_algorithm is run as normal, but with infinite R
	///     (actually R = pi/2, since FastJet asserts that R <= pi/2), and
	///     recombinations with DeltaR > Rcut and z < zcut are vetoed, and the
	///     softer PseudoJet is discarded.  (It will appear in the history as a
	///     PJ that never merged again -- it is not merged with the beam!
	///     inclusive_jets(ptCut) should be used to access the pruned jets.
	///     Rcut is chosen for each jet according to
	///           Rcut = Rcut_factor * 2*m/pT, 
	///     where m and pT correspond to the four-momentum of the found jet.
	///
	///
	PrunedRecombPlugin (const JetDefinition & find_definition,
	                    const JetDefinition & prune_definition,
							  const double & zcut = 0.1,
							  const double & Rcut_factor = 0.5) :
		_find_definition(find_definition),
		_prune_definition(prune_definition),
		_zcut(zcut),
		_Rcut_factor(Rcut_factor)
		{}
	
	// The things that are required by base class.
	virtual std::string description () const;
	virtual void run_clustering(ClusterSequence &) const;
	
	// Access to parameters.
	inline virtual double R() const {return _find_definition.R();}
	inline double zcut() const {return _zcut;}
	inline double Rcut_factor() const {return _Rcut_factor;}

private:

	JetDefinition _find_definition;
	mutable JetDefinition _prune_definition;  // mutable so we can change its Recombiner*
	double _zcut;
	double _Rcut_factor;

	// Helper function to find the characteristic angle of a given jet.
	// There are a few reasonable choices for this; we use 2*m/pT.
	//   If no parents, return -1.0.
	double _characteristic_angle (const PseudoJet & jet,
	                       const ClusterSequence & seq) const;
	
	// Takes the merging structure of "in_seq" and feeds this into
	//  "out_seq" using _plugin_record_{ij,iB}_recombination()
	void _output_mergings(ClusterSequence & in_seq, 
	                           ClusterSequence & out_seq) const;

};



///////////////////////////////////////////////////////////////////////////////
//
// The PrunedRecombiner class.  This class extends the DefaultRecombiner
//   class to include a "pruning test".  If this test on a recombination fails,
//   the recombination does not occur.  This happens in the following way:
// 1) The "new" PseudoJet is set equal to the higher-pT parent.
// 2) The lower-pT parent is effectively discarded from the algorithm.
// 3) Not implemented yet: some method of keeping track of what was pruned when.
//
///////////////////////////////////////////////////////////////////////////////


class PrunedRecombiner : public JetDefinition::DefaultRecombiner {
public:
	PrunedRecombiner(const double & zcut = 0.1, const double & Rcut = 0.5,
	                 RecombinationScheme recomb_scheme = E_scheme) :
	   JetDefinition::DefaultRecombiner(recomb_scheme), _zcut(zcut), _Rcut(Rcut) 
		{}
		
	virtual std::string description() const;
	
	/// recombine pa and pb and put result into pab
	virtual void recombine(const PseudoJet & pa, const PseudoJet & pb, 
                           PseudoJet & pab) const;
		
private:
	/// tests whether pa and pb should be recombined or vetoed
	int _pruning_test(const PseudoJet & pa, const PseudoJet & pb) const;

	double _zcut; // zcut parameter to the pruning test
	double _Rcut; // Rcut parameter to the pruning test
};



FASTJET_END_NAMESPACE      // defined in fastjet/internal/base.hh



#endif 
