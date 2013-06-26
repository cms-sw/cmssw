// $Id: PrunedRecombiner.cc,v 1.2 2011/04/25 04:19:54 srappocc Exp $
///////////////////////////////////////////////////////////////////////////////
//
// Implements the PrunedRecombiner class.  See PrunedRecombiner.hh
//
///////////////////////////////////////////////////////////////////////////////

#include "RecoJets/JetAlgorithms/interface/PrunedRecombiner.hh"

#include <sstream>

using namespace fastjet;

std::string PrunedRecombiner::description() const
{
	std::ostringstream s;
	s << "Pruned " << _recombiner->description()
	  << ", with zcut = " << _zcut << " and Rcut = " << _Rcut;
	return s.str();
}	
	
// Recombine pa and pb and put result into pab.
// If pruning test is true (the recombination is vetoed) the harder
//   parent is merged with a 0 PseudoJet. (Check that this is an identity in
//   all recombination schemes!)
// When a branch is pruned, its cluster_hist_index is stored in 
//   _pruned_pseudojets for later use.
void PrunedRecombiner::recombine(const PseudoJet & pa, const PseudoJet & pb, 
                           PseudoJet & pab) const
{
	PseudoJet p0(0.0, 0.0, 0.0, 0.0);
	// test if recombination should be pruned
	switch ( _pruning_test(pa, pb) ) {
		case 1:
			_pruned_pseudojets.push_back(pb.cluster_hist_index());
			_recombiner->recombine(pa, p0, pab);
			break;
		case 2:
			_pruned_pseudojets.push_back(pa.cluster_hist_index());
			_recombiner->recombine(pb, p0, pab);
			break;
		default: 
			// if no pruning, do regular combination
			_recombiner->recombine(pa, pb, pab);
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
	_recombiner->recombine(pa, pb, newjet);

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


void PrunedRecombiner::reset(const double & zcut, const double & Rcut)
{
	_pruned_pseudojets.clear();
	_zcut = zcut;
	_Rcut = Rcut;
}
