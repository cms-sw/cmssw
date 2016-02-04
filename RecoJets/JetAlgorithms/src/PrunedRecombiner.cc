///////////////////////////////////////////////////////////////////////////////
//
// PrunedRecombiner.cc
// Last update: 5/28/09 CKV
// PrunedRecomb version: 0.2.0
// Author: Christopher Vermilion <verm@u.washington.edu>
//
// Implements the PrunedRecombiner class.  See PrunedRecombiner.hh
//
///////////////////////////////////////////////////////////////////////////////

#include "PrunedRecombPlugin.h"

#include <string>
#include <sstream>
#include <iostream>

FASTJET_BEGIN_NAMESPACE

std::string PrunedRecombiner::description() const
{
	std::ostringstream s;
	s << "Pruned " << JetDefinition::DefaultRecombiner::description()
	  << ", with zcut = " << _zcut << " and Rcut = " << _Rcut;
	return s.str();
}	
	
/// Recombine pa and pb and put result into pab.
/// If pruning test is true (the recombination is vetoed) the harder
///   parent is merged with a 0 PseudoJet. (Check that this is an identity in
///   all recombination schemes!)
void PrunedRecombiner::recombine(const PseudoJet & pa, const PseudoJet & pb, 
                           PseudoJet & pab) const
{
	//std::cout << "In PR::recombine()\n";
	PseudoJet p0(0.0, 0.0, 0.0, 0.0);
	// test if recombination should be pruned
	switch ( _pruning_test(pa, pb) ) {
		case 1:
			// MAKE RECORD OF PB BEING PRUNED
			JetDefinition::DefaultRecombiner::recombine(pa, p0, pab);
			break;
		case 2:
			// MAKE RECORD OF PA BEING PRUNED
			JetDefinition::DefaultRecombiner::recombine(pb, p0, pab);
			break;
		default: 
			// if no pruning, do regular combination
			JetDefinition::DefaultRecombiner::recombine(pa, pb, pab);
	}
}

		
// Function to test if two pseudojets should be merged -- ie, to selectively
//   veto mergings.  Should provide possibility to provide this function...
//   Return codes:
//
//   0: Merge.  (Currently, anything other than 1 or 2 will do.)
//   1: Don't merge; keep pa
//   2: Don't merge; keep pb 
//
int PrunedRecombiner::_pruning_test(const PseudoJet & pa, const PseudoJet & pb) const
{
	// create the new jet by recombining the first two, using the normal
	//   recombination scheme
	PseudoJet newjet;	
	JetDefinition::DefaultRecombiner::recombine(pa, pb, newjet);

	double minpT = pa.perp();
	int hard = 2;  // harder pj is pj2
	double tmp = pb.perp();
	if (tmp < minpT) {
		minpT = tmp;
		hard = 1;  // harder pj is pj1
	}
	
	if ( pa.squared_distance(pb) < _Rcut*_Rcut 
	      || minpT > _zcut * newjet.perp() )
		return 0;
	else
		return hard;
}


FASTJET_END_NAMESPACE

