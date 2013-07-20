// $Id: FastPrunePlugin.hh,v 1.1 2011/04/25 04:19:54 srappocc Exp $
///////////////////////////////////////////////////////////////////////////////
//
// Defines the FastPrune Plugin to FastJet.  This is the "pruned" version of
//   a recombination algorithm specifed in the constructor (Cambridge/Aachen and
//   kT are both sensible choices), as described in arXiv:0912.0033.
//
// The helper class PrunedRecombiner is defined in PrunedRecombiner.hh.
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
//      only inclusive_jets(ptcut) makes sense.  Future versions of this plugin
//      will hopefully include an Extras() object with the ClusterSequence so
//      that the pruned branches can be associated with the points they were
//      pruned from.
//
// An external Recombiner can be passed to the constructor: the
//   PrunedRecombiner will only determine whether to prune a recombination; the
//   external Recombiner will perform the actual recombination.  If a
//   Recombiner is not explicitly passed, the Recombiner from the pruned_def is
//   used -- this will be DefaultRecombiner if not otherwise set.
//
// New in v0.3.4: the plugin now uses the helper class CutSetter to store
//   zcut and Rcut; the user can create their own derived class to set these
//   on a jet-by-jet basis.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef __FASTPRUNEPLUGIN_HH__
#define __FASTPRUNEPLUGIN_HH__

#include "RecoJets/JetAlgorithms/interface/PrunedRecombiner.hh"

#include "fastjet/ClusterSequence.hh"
#include "fastjet/JetDefinition.hh"

#include <string>

FASTJET_BEGIN_NAMESPACE      // defined in fastjet/internal/base.hh


class FastPrunePlugin : public JetDefinition::Plugin {
public:

	class CutSetter;

	/// Constructors for the FastPrunePlugin, whose arguments have the
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
	///  - recomb specifies the user's own Recombiner class, useful for
	///    implementing, say, flavor recombination.  The internal PrunedRecombiner
	///    class only checks if pruning should be done.  If this is not passed,
	///    the Recombiner from prune_definition is used instead.
	///
	/// Note that the PrunedRecombiner is set up in the constructor, but specific
	/// values of Rcut are selected jet-by-jet in run_clustering().
	///
	FastPrunePlugin (const JetDefinition & find_definition,
	                 const JetDefinition & prune_definition,
	                 const double & zcut = 0.1,
	                 const double & Rcut_factor = 0.5);

	FastPrunePlugin (const JetDefinition & find_definition,
	                 const JetDefinition & prune_definition,
	                 const JetDefinition::Recombiner* recomb,
	                 const double & zcut = 0.1,
	                 const double & Rcut_factor = 0.5);

	/// Two new constructors that allow you to pass your own CutSetter.
	///  This lets you define zcut and Rcut on a jet-by-jet basis.
	///  See below for CutSetter definition.
	FastPrunePlugin (const JetDefinition & find_definition,
	                 const JetDefinition & prune_definition,
	                 CutSetter* const cut_setter);

	FastPrunePlugin (const JetDefinition & find_definition,
	                 const JetDefinition & prune_definition,
	                 CutSetter* const cut_setter,
	                 const JetDefinition::Recombiner* recomb);

	
	// Get the set of pruned away PseudoJets (valid in pruned ClusterSequence)
	// Each vector corresponds to one of the unpruned jets, in pT order
	virtual std::vector<std::vector<PseudoJet> > pruned_subjets() const {return _pruned_subjets;}

	// The things that are required by base class.
	virtual std::string description () const;
	virtual void run_clustering(ClusterSequence &) const;

	// Sets minimum pT for unpruned jets (default is 20 GeV)
	//  (Just to save time by not pruning jets we don't care about)
	virtual void set_unpruned_minpT(double pt) {_minpT = pt;}
	
	// Access to parameters.
	virtual double R() const {return _find_definition.R();}
	virtual double zcut() const {return _cut_setter->zcut;}
	virtual double Rcut() const {return _cut_setter->Rcut;}
	// only meaningful for DefaultCutSetter:
	virtual double Rcut_factor() const {
		if (DefaultCutSetter *cs = dynamic_cast<DefaultCutSetter*>(_cut_setter))
			return cs->Rcut_factor;
		else return -1.0;
	}

	// ***** DEPRECATED **** -- use FastPruneTool instead; this will be removed
	//  in a future release!
	// Access to unpruned cluster sequence that gets made along the way
	//  This should only be used after run_clustering() has been called,
	//  i.e., after the plugin has been used to find jets.  I've added
	//  this for speed, so you don't have to duplicate the effort of
	//  finding unpruned jets.
	virtual ClusterSequence *get_unpruned_sequence() const {return _unpruned_seq;}

	virtual ~FastPrunePlugin() {delete _unpruned_seq; delete _pruned_recombiner;
                              delete _cut_setter;}

protected:

	JetDefinition _find_definition;
	mutable JetDefinition _prune_definition; // mutable: we change its Recombiner*
	double _minpT; // minimum pT for unpruned jets

	mutable std::vector<std::vector<PseudoJet> > _pruned_subjets;
	mutable ClusterSequence* _unpruned_seq;
	
	mutable PrunedRecombiner* _pruned_recombiner;
	
	// Takes the merging structure of "in_seq" and feeds this into
	//  "out_seq" using _plugin_record_{ij,iB}_recombination()
	void _output_mergings(ClusterSequence & in_seq,
	                      std::vector<int> & pruned_pseudojets,
	                      ClusterSequence & out_seq) const;

	// this class stores info on how to dynamically set zcut and Rcut
	//  const because we won't change it
	CutSetter* _cut_setter;

public:

	/// A helper class to define cuts, possibly on a jet-by-jet basis
	class CutSetter {
	public:
		CutSetter(const double & z = 0.1, const double & R = 0.5)
      : zcut(z), Rcut(R) {}
		double zcut, Rcut;
		virtual void SetCuts(const PseudoJet &jet,
		                     const ClusterSequence &clust_seq) = 0;
		virtual ~CutSetter() {}
	};

	/// Default CutSetter implementation: never changes zcut,
	///   and Rcut = Rcut_factor * 2m/pT for a given jet
	class DefaultCutSetter : public CutSetter {
	public:
		DefaultCutSetter(const double & z = 0.1, const double &Rcut_fact = 0.5)
			: CutSetter(z), Rcut_factor(Rcut_fact) {}
		double Rcut_factor;
		virtual void SetCuts(const PseudoJet &jet,
		                     const ClusterSequence &clust_seq) {
			PseudoJet p1, p2;
			if (! clust_seq.has_parents(jet, p1, p2))
				Rcut = 0.0;
			else
				Rcut = Rcut_factor*2.0*jet.m()/jet.perp();
		}
	};

};


FASTJET_END_NAMESPACE      // defined in fastjet/internal/base.hh



#endif  // __FASTPRUNEPLUGIN_HH__
