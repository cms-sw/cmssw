// $Id: PrunedRecombiner.hh,v 1.1 2011/04/25 04:19:54 srappocc Exp $
///////////////////////////////////////////////////////////////////////////////
//
// The PrunedRecombiner class.  This class extends any Recombiner
//   class to include a "pruning test".  If this test on a recombination fails,
//   the recombination does not occur.  This happens in the following way:
// 1) The "new" PseudoJet is set equal to the higher-pT parent.
// 2) The lower-pT parent is effectively discarded from the algorithm.
// 3) Not implemented yet: some method of keeping track of what was pruned when.
//
// New:  You must pass a Recombiner-derived object that will do the actual
//   recombining; PrunedRecombiner only checks whether to prune or not.  This is
//   useful if you want to implement, say, flavor recombination.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef __PRUNEDRECOMBINER_HH__
#define __PRUNEDRECOMBINER_HH__

#include "fastjet/ClusterSequence.hh"
#include "fastjet/JetDefinition.hh"

#include <string>

FASTJET_BEGIN_NAMESPACE      // defined in fastjet/internal/base.hh


class PrunedRecombiner : public JetDefinition::Recombiner {
public:
	PrunedRecombiner(const JetDefinition::Recombiner *recomb,
	                 const double & zcut = 0.1, const double & Rcut = 0.5) :
		_zcut(zcut), _Rcut(Rcut), _recombiner(recomb) {}
	
	PrunedRecombiner(const RecombinationScheme scheme,
	                 const double & zcut = 0.1, const double & Rcut = 0.5) :
		_zcut(zcut), _Rcut(Rcut), _recombiner(0), _default_recombiner(scheme) {
		_recombiner = &_default_recombiner;
	}

	virtual std::string description() const;
	
	// recombine pa and pb and put result into pab
	virtual void recombine(const PseudoJet & pa, const PseudoJet & pb, 
	                       PseudoJet & pab) const;

	std::vector<int> pruned_pseudojets() { return _pruned_pseudojets; }

	// resets pruned_pseudojets vector, parameters
	void reset(const double & zcut, const double & Rcut);

	virtual ~PrunedRecombiner() {}
		
private:
	// tests whether pa and pb should be recombined or vetoed
	int _pruning_test(const PseudoJet & pa, const PseudoJet & pb) const;

	double _zcut; // zcut parameter to the pruning test
	double _Rcut; // Rcut parameter to the pruning test

	// vector that holds cluster_history_indices of pruned pj's
	mutable std::vector<int> _pruned_pseudojets;
	
	// points to the "real" external recombiner
	const JetDefinition::Recombiner* _recombiner;
	
	// Use this if we are only passed a recombination scheme.
	// A DefaultRecombiner is so small it's not worth dealing with whether we own
	// the recombiner or not...
	JetDefinition::DefaultRecombiner _default_recombiner;
};



FASTJET_END_NAMESPACE      // defined in fastjet/internal/base.hh



#endif  // __PRUNEDRECOMBINER_HH__
