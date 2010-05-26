#ifndef RecoBTag_Analysis_Matching_h
#define RecoBTag_Analysis_Matching_h

#include <functional>
#include <algorithm>
#include <vector>
#include <set>

#include "SimpleMatrix.h"

namespace btag {

template<typename Delta>
class Matching {
    public:
	typedef typename SimpleMatrix<Delta>::size_type index_type;

	template<typename V1, typename V2, class Separation>
	Matching(const V1 &v1, const V2 &v2, Separation separation) :
		matrix(v1.size(), v2.size()),
		matched1(v1.size(), false),
		matched2(v2.size(), false)
	{
		index_type i = 0;
		for(typename V1::const_iterator iter1 = v1.begin();
		    iter1 != v1.end(); ++iter1, i++) {
			index_type j = 0;
			for(typename V2::const_iterator iter2 = v2.begin();
			    iter2 != v2.end(); ++iter2, j++)
				matrix(i, j) = separation(*iter1, *iter2);
					
		}
	}

	struct Match {
		typedef typename Matching::index_type index_type;

		inline Match(index_type i1, index_type i2) :
			index1(i1), index2(i2) {}

		index_type	index1, index2;
	};

	inline Delta delta(index_type index1, index_type index2) const
	{ return matrix(index1, index2); }

	inline Delta delta(Match match) const
	{ return matrix(match.index1, match.index2); }

    private:
	template<class SortComparator>
	struct Comparator : public std::binary_function<Delta, Delta, bool> {
		typedef typename Matching::index_type index_type;

		inline Comparator(const SimpleMatrix<Delta> &matrix,
		                  SortComparator &sort) :
			matrix(matrix), sort(sort) {}

		inline bool operator () (index_type i1, index_type i2) const
		{ return sort(matrix[i1], matrix[i2]); }

		const SimpleMatrix<Delta>	&matrix;
		SortComparator			&sort;
	};

	struct AlwaysTrue : public std::unary_function<Delta, bool> {
		inline bool operator () (Delta dummy) { return true; }
	};

    public:
	template<class SortComparator, class CutCriterion>
	std::vector<Match> match(
			SortComparator sortComparator = SortComparator(),
			CutCriterion cutCriterion = CutCriterion())
	{
		std::vector<index_type> matches(matrix.size());
		for(index_type i = 0; i != matrix.size(); i++)
			matches[i] = i;

		std::sort(matches.begin(), matches.end(),
		          Comparator<SortComparator>(matrix, sortComparator));

		std::vector<Match> result;
		result.reserve(std::min(matrix.rows(), matrix.cols()));
		for(typename std::vector<index_type>::const_iterator iter =
			matches.begin(); iter != matches.end(); ++iter) {

			index_type row = matrix.row(*iter);
			index_type col = matrix.col(*iter);
			if (matched1[row] || matched2[col])
				continue;

			if (!cutCriterion(matrix[*iter]))
				continue;

			matched1[row] = true;
			matched2[col] = true;
			result.push_back(Match(row, col));
		}

		return result;
	}

	template<class SortComparator>
	inline std::vector<Match> match()
	{ return match<SortComparator, AlwaysTrue>(); }

	inline std::vector<Match> match()
	{ return match<std::less<Delta>, AlwaysTrue>(); }

	inline bool isMatched1st(index_type index) { return matched1[index]; }
	inline bool isMatched2nd(index_type index) { return matched2[index]; }

    private:
	SimpleMatrix<Delta>	matrix;
	std::vector<bool>	matched1, matched2;
};

} // namespace btag

#endif // GeneratorEvent_Analysis_Matching_h
