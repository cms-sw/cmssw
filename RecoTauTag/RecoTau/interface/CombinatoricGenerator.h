#ifndef RecoTauTag_RecoTau_CombinatoricGenerator_h
#define RecoTauTag_RecoTau_CombinatoricGenerator_h

#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/filter_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <vector>
#include <set>
#include <memory>
#include <iostream>

/*
 * CombinatoricGenerator
 *
 * Author: Evan K. Friis (UC Davis)
 *
 * Generic classes to compute combinatoric subsets of collections.
 *
 * Example of use:
 *
 * vector<int> collection = [0, 1, 2, 3, 4, 5];
 *
 * typedef CombinatoricGenerator<std::vector<int> Generator;
 *
 * // Select three element combinations of collection
 * Generator generator(collection.begin(), collection.end(), 3);
 *
 * for(Generator::iterator combo = generator.begin(); combo != generator.end(); ++combo)
 * {
 *     for(Generator::combo_iterator element = combo->begin(); element != combo->end(); ++element)
 *     {
 *         cout << *element << " ";
 *     }
 *     cout << endl;
 * }
 *
 * Outputs:
 * 0 1 2
 * 0 1 3
 * 0 1 4
 * ...
 * 3 4 5
 *
 *
 */

namespace reco { namespace tau {

template <typename T>
  class Combinatoric
  {
    /* Combinatoric<T>
     *
     * Class that represents a subset of values from a collection of type T.
     *
     * The values belonging to the current combination subset can be accessed with the iterators
     * combo_begin() and combo_end()
     *
     * The values in the collection not in the subset can be accessed with
     * remainder_begin() and remainder_end()
     *
     */

    // Iterator over input collection
    typedef typename T::const_iterator value_iter;
    typedef typename T::value_type value_type;
    typedef size_t index_type;
    typedef typename std::vector<index_type> indices_collection;
    typedef typename indices_collection::const_iterator index_iter;
    typedef typename std::set<index_type> indices_set;

    class IndexInSet {
      /* Determine if a given index is in a set of indices */
      public:
        IndexInSet(const indices_set& combo, bool negate):
          combo_(combo),negate_(negate){}
        bool operator()(const index_type& index) const {
          return (negate_) ? !combo_.count(index) : combo_.count(index);
        }
      private:
        indices_set combo_;
        bool negate_;
    };

    class ValueAccessor : public std::unary_function<index_type const&, const value_type&>
    {
      /* Class to extract a value from a collection given the beginning of the collection
       * and the index into the colleciton */
      public:
        ValueAccessor(const value_iter& begin):
          begin_(begin){}
        const value_type& operator()(index_type const & index) const {
          return *(begin_+index);
        }
      private:
        value_iter begin_;
    };

    public:

    Combinatoric(const value_iter& begin, const indices_collection& indices,
        const indices_collection& combo, bool done):
      begin_(begin), combo_(combo.begin(), combo.end()),
      comboSet_(combo.begin(), combo.end()),
      indices_(indices),
      done_(done),
      valueAccessor_(begin), inChoice_(comboSet_, false),
      notInChoice_(comboSet_, true) {}

    typedef typename boost::filter_iterator<IndexInSet, index_iter> ComboIter;
    typedef typename boost::transform_iterator<ValueAccessor, ComboIter> ValueIter;

    /// The first element in the selected subset
    ValueIter combo_begin() const { return ValueIter(ComboIter(inChoice_, indices_.begin(), indices_.end()), valueAccessor_); }
    /// One past the last element in the selected subset
    ValueIter combo_end() const { return ValueIter(ComboIter(inChoice_, indices_.end(), indices_.end()), valueAccessor_); }

    /// The first element in the non-selected subset
    ValueIter remainder_end() const { return ValueIter(ComboIter(notInChoice_, indices_.end(), indices_.end()), valueAccessor_); }
    /// One past the last element in the non-selected subset
    ValueIter remainder_begin() const { return ValueIter(ComboIter(notInChoice_, indices_.begin(), indices_.end()), valueAccessor_); }

    /// Build the next cominatoric subset after the current one
    Combinatoric<T> next() const
    {
      //std::cout << "building next: " << std::endl;
      //std::cout << "current " << std::endl;
      //std::copy(combo_.begin(), combo_.end(), std::ostream_iterator<int>(std::cout, " "));
      //std::cout << std::endl;

      indices_collection newCombo(combo_);

      // Find the first value that can be updated (starting from the back
      // max value for index is (nElements-1) - (distance from end)
      // Examples:
      //    189 -> 289 (will be updated to 234)
      //    179 -> 189
      //    123 -> 124
      size_t distanceFromEnd = 0;
      size_t nElements = indices_.size();
      indices_collection::reverse_iterator pos = newCombo.rbegin();
      for(; pos != newCombo.rend(); ++pos, ++distanceFromEnd)
      {
        // Check if we can update this element to the next value (without overflow)
        // If so, increment it it, then quit
        if( *pos < (nElements - 1 - distanceFromEnd) )
        {
          (*pos)++;
          break;
        } else {
          // Set to 'unset value' - we need to update it!
          (*pos) = nElements;
        }
      }

      //std::cout << "after update " << std::endl;
      //std::copy(newCombo.begin(), newCombo.end(), std::ostream_iterator<int>(std::cout, " "));
      //std::cout << std::endl;

      // forward_pos points to the element *after* pos
      indices_collection::iterator forward_pos = pos.base();
      // Only do the updates if we have not reached all the way to the beginning!
      // this indicates we are at the end of the iteration.  We return a combo
      // with all values at nElements to flag that we are at the end
      bool done = true;
      if (forward_pos != newCombo.begin()) {
        // Everything after pos needs to be updated.  i.e. 159 -> 167
        index_type next_pos_value = (*pos)+1;
        //std::cout << "next pos: " << next_pos_value << std::endl;
        done = false;
        for (; forward_pos != newCombo.end(); ++forward_pos, ++next_pos_value) {
          *forward_pos = next_pos_value;
        }
      }

      //std::cout << "final " << std::endl;
      //std::copy(newCombo.begin(), newCombo.end(), std::ostream_iterator<int>(std::cout, " "));
      //std::cout << std::endl;

      //return std::unique_ptr<Combinatoric<T> >(new Combinatoric<T>(begin_, indices_, newCombo));
      return Combinatoric<T>(begin_, indices_, newCombo, done);
    }

    // Check if iteration is done
    bool done() const { return done_; }

    /// Return the set of selected indices
    const indices_set& combo() const { return comboSet_; }

    /// Comparison to another combination
    bool operator==(const Combinatoric<T>& rhs) const
    {
      return (this->combo() == rhs.combo() && this->done() == rhs.done());
    }

    private:
    // Beginning and ending of the vector of iterators that point to our
    // input collection
    value_iter begin_;
    indices_collection combo_;
    indices_set comboSet_;
    indices_collection indices_;
    bool done_;
    ValueAccessor valueAccessor_;
    IndexInSet inChoice_;
    IndexInSet notInChoice_;
  };

template<typename T>
  class CombinatoricIterator : public boost::iterator_facade<
                               CombinatoricIterator<T>,
                               const Combinatoric<T>,
                               boost::forward_traversal_tag>
{
  /* An iterator over Combinatorics */
  public:
    typedef Combinatoric<T> value_type;
    explicit CombinatoricIterator(const Combinatoric<T>& c):node_(c) {}

  private:
    friend class boost::iterator_core_access;

    bool equal(CombinatoricIterator const& other) const {
      return this->node_ == other.node_;
    }

    void increment() {
      node_ = node_.next();
    }

    const Combinatoric<T>& dereference() const {
      return node_;
    }

    Combinatoric<T> node_;
};

template<typename T>
  class CombinatoricGenerator
  {
    /* CombinatoricGenerator
     *
     * Generate combinatoric subsets of a collection of type T.
     *
     * This classes begin() and end() functions return iterators of size
     * <choose> the first and last combinatoric subsets in the input collection
     * range defined by <begin> and <end>.
     */

    // Iterator over input collection
    typedef typename T::const_iterator value_iter;
    typedef typename T::value_type value_type;
    typedef size_t index_type;
    typedef typename std::vector<index_type> indices_collection;
    typedef typename indices_collection::iterator index_iter;
    typedef typename std::set<index_type> indices_set;

    public:

    typedef std::unique_ptr<CombinatoricIterator<T> > CombIterPtr;
    typedef CombinatoricIterator<T> iterator;
    typedef typename iterator::value_type::ValueIter combo_iterator;

    explicit CombinatoricGenerator(const value_iter& begin, const value_iter& end, size_t choose)
    {
      // Make beginning and ending index collections
      indices_collection initialCombo(choose);
      indices_collection finalCombo(choose);

      size_t totalElements = end-begin;

      if (choose <= totalElements) {
        indices_collection allIndices(totalElements);
        for(size_t i=0; i < totalElements; ++i)
        {
          allIndices[i] = i;
        }

        for(size_t i=0; i < choose; ++i)
        {
          initialCombo[i] = i;
          // End conditions each is set at nElements
          finalCombo[i] = totalElements;
        }

        beginning_ = CombIterPtr(new CombinatoricIterator<T>(Combinatoric<T>(
                begin, allIndices, initialCombo, false)));

        ending_ = CombIterPtr(new CombinatoricIterator<T>(Combinatoric<T>(
                begin, allIndices, finalCombo, true)));
      } else {
        // We don't have enough in the collection to return [choose] items.
        // Return an empty collection
        beginning_ = CombIterPtr(new CombinatoricIterator<T>(Combinatoric<T>(
                begin, indices_collection(), indices_collection(), true)));

        ending_ = CombIterPtr(new CombinatoricIterator<T>(Combinatoric<T>(
                begin, indices_collection(), indices_collection(), true)));
      }
    }

    CombinatoricIterator<T> begin() {
      return *beginning_;
    }

    CombinatoricIterator<T> end() {
      return *ending_;
    }

    private:
    CombIterPtr beginning_;
    CombIterPtr ending_;
  };

} } // end namespace reco::tau
#endif
