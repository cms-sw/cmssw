#ifndef GENERS_CP_COLUMN_ITERATOR_HH_
#define GENERS_CP_COLUMN_ITERATOR_HH_

#include <climits>
#include <iterator>
#include "Alignment/Geners/interface/IOException.hh"

#include "Alignment/Geners/interface/IOReferredType.hh"

namespace gs {
    //
    // The following works as a simple forward iterator for ColumnPacker
    // (cycles over the contents of a single column). It is effectively
    // const, as it can not modify the underlying packer data.
    //
    template<unsigned long N, typename Packer, typename StoragePtr>
    class CP_column_iterator
    {
    public:
        typedef typename IOReferredType<StoragePtr>::type value_type;
        typedef value_type* pointer;
        typedef value_type& reference;
        typedef std::ptrdiff_t difference_type;
        typedef std::forward_iterator_tag iterator_category;

        inline CP_column_iterator(const Packer& packer, StoragePtr s,
                                  const unsigned long row)
            : packer_(packer), s_(s), row_(row)
        {
            if (row_ > packer_.nRows())
                row_ = packer_.nRows();
        }

        inline reference operator*() const
        {
            packer_.template fetchItem<N>(row_, s_);
            return *s_;
        }

        inline pointer operator->() const
        {
            packer_.template fetchItem<N>(row_, s_);
            return &*s_;
        }

        inline CP_column_iterator& operator++()
            {++row_; return *this;}

        inline CP_column_iterator operator++(int)
            {CP_column_iterator tmp(*this); ++row_; return tmp;}

        inline bool operator==(const CP_column_iterator& r) const
            {return row_ == r.row_;}

        inline bool operator!=(const CP_column_iterator& r) const
            {return row_ != r.row_;}

        inline bool operator<(const CP_column_iterator& r) const
            {return row_ < r.row_;}

    private:
        const Packer& packer_;
        StoragePtr s_;
        unsigned long row_;
    };

    template<unsigned long N, typename Packer, typename StoragePtr>
    inline CP_column_iterator<N, Packer, StoragePtr>
    CP_column_begin(const Packer& packer, StoragePtr s,
                    const unsigned long row=0UL)
    {
        return CP_column_iterator<N, Packer, StoragePtr>(packer, s, row);
    }

    template<unsigned long N, typename Packer, typename StoragePtr>
    inline CP_column_iterator<N, Packer, StoragePtr>
    CP_column_end(const Packer& packer, StoragePtr s)
    {
        return CP_column_iterator<N, Packer, StoragePtr>(packer, s, ULONG_MAX);
    }
}

#endif // GENERS_CP_COLUMN_ITERATOR_HH_

