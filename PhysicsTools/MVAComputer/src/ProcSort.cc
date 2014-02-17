// -*- C++ -*-
//
// Package:     MVAComputer
// Class  :     ProcSort
// 

// Implementation:
//     Sorts the input variables. Each input variable must appear in the
//     multiplicity as the others. One variable is the sorting "leader" by
//     which all variables are reordered the same way. The ordering is
//     determined by either ascending or descending order of the leader.
//
// Author:      Christophe Saout
// Created:     Sun Sep 16 14:52 CEST 2007
// $Id: ProcSort.cc,v 1.3 2009/06/03 09:50:14 saout Exp $
//

#include <stdlib.h>
#include <algorithm>
#include <functional>
#include <iterator>
#include <vector>

#include <boost/iterator/transform_iterator.hpp>

#include "PhysicsTools/MVAComputer/interface/VarProcessor.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"

using namespace PhysicsTools;

namespace { // anonymous

class ProcSort : public VarProcessor {
    public:
	typedef VarProcessor::Registry::Registry<ProcSort,
					Calibration::ProcSort> Registry;

	ProcSort(const char *name,
	         const Calibration::ProcSort *calib,
	         const MVAComputer *computer);
	virtual ~ProcSort() {}

	virtual void configure(ConfIterator iter, unsigned int n);
	virtual void eval(ValueIterator iter, unsigned int n) const;
	virtual std::vector<double> deriv(
				ValueIterator iter, unsigned int n) const;

    private:
	unsigned int	leader;
	bool		descending;
};

static ProcSort::Registry registry("ProcSort");

ProcSort::ProcSort(const char *name,
                   const Calibration::ProcSort *calib,
                   const MVAComputer *computer) :
	VarProcessor(name, calib, computer),
	leader(calib->sortByIndex),
	descending(calib->descending)
{
}

void ProcSort::configure(ConfIterator iter, unsigned int n)
{
	if (leader >= n)
		return;

	iter << iter;
	while(iter)
		iter << iter++(Variable::FLAG_ALL);
}

namespace { // anonymous
	struct LeaderLookup : public std::unary_function<int, double> {
		inline LeaderLookup() {}
		inline LeaderLookup(const double *values) : values(values) {}

		inline double operator () (int index) const
		{ return values[index]; }

		const double	*values;
	};
} // anonymous namespace

void ProcSort::eval(ValueIterator iter, unsigned int n) const
{
	ValueIterator leaderIter = iter;
	for(unsigned int i = 0; i < leader; i++, leaderIter++);
	unsigned int size = leaderIter.size();
	LeaderLookup lookup(leaderIter.begin());

	int *sort = (int*)alloca(size * sizeof(int));
	for(unsigned int i = 0; i < size; i++)
		sort[i] = (int)i;

	boost::transform_iterator<LeaderLookup, int*> begin(sort, lookup);
	boost::transform_iterator<LeaderLookup, int*> end = begin;

	for(unsigned int i = 0; i < size; i++, end++) {
		unsigned int pos = std::lower_bound(begin, end,
		                                    leaderIter[i]) - begin;   
		std::memmove(sort + (pos + 1), sort + pos,
		             (i - pos) * sizeof(*sort));
		sort[pos] = i;
	}

	if (descending)
		std::reverse(sort, sort + size);

	for(unsigned int i = 0; i < size; i++)
		iter << (double)sort[i];
	iter();

	while(iter) {
		for(unsigned int i = 0; i < size; i++)
			iter << iter[sort[i]];
		iter();
		iter++;
	}
}

std::vector<double> ProcSort::deriv(ValueIterator iter, unsigned int n) const
{
	unsigned int in = 0;
	for(ValueIterator iter2 = iter; iter2; ++iter2)
		in += iter2.size();

	ValueIterator leaderIter = iter;
	for(unsigned int i = 0; i < leader; i++, leaderIter++);
	unsigned int size = leaderIter.size();
	LeaderLookup lookup(leaderIter.begin());

	std::vector<int> sort;
	for(unsigned int i = 0; i < size; i++)
		sort.push_back((int)i);

	boost::transform_iterator<LeaderLookup,
	                          std::vector<int>::const_iterator > begin(
							sort.begin(), lookup);
	boost::transform_iterator<LeaderLookup,
	                          std::vector<int>::const_iterator > end =
									begin;

	for(unsigned int i = 0; i < size; i++, end++) {
		unsigned int pos = std::lower_bound(begin, end,
		                                    leaderIter[i]) - begin;
		std::memmove(&sort.front() + (pos + 1), &sort.front() + pos,
		             (i - pos) * sizeof(sort.front()));
		sort[pos] = i;
	}

	if (descending)
		std::reverse(sort.begin(), sort.end());

	std::vector<double> result(size * in, 0.0);

	for(unsigned int pos = 0; iter; pos += (iter++).size()) {
		for(unsigned int i = 0; i < size; i++) {
			unsigned int row = result.size();
			result.resize(row + in);
			result[row + pos + sort[i]] = 1.0;
		}
	}

	return result;
}

} // anonymous namespace
