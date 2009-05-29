#include <algorithm>
#include "DataFormats/Common/interface/MultiAssociation.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>

using edm::helper::IndexRangeAssociation;
using edm::ProductID;

IndexRangeAssociation::range
IndexRangeAssociation::get(const ProductID &id, size_t key) const {
    typedef IndexRangeAssociation::id_offset_vector::const_iterator iter;
    iter pos = std::lower_bound(id_offsets_.begin(), id_offsets_.end(), id, IDComparator()); 
    if ((pos == id_offsets_.end()) || (pos->first != id)) {
        throw cms::Exception("Bad Key") << "Product ID " << id << " not found in this IndexRangeAssociation\n";
    }
    // === Do we want this check ? I would say yes, even if it costs some extra CPU cycles
    if ((pos + 1 != id_offsets_.end()) && (pos->second + key >= (pos+1)->second)) {
        throw cms::Exception("Bad Offset") << "Key " << key << " goes beyond bounds " 
                    << ((pos+1)->second - pos->second) 
                    << " of this key collection within IndexRangeAssociation\n";
    }
    // === End check
    offset_vector::const_iterator offs = ref_offsets_.begin() + pos->second + key;
    if (offs >= ref_offsets_.end()-1) {
        throw cms::Exception("Bad Offset") << "Key " << key << " goes beyond bounds " << ref_offsets_.size()-1 << " of this IndexRangeAssociation\n";
    }
    return range(*offs,*(offs+1));
}

bool
IndexRangeAssociation::contains(ProductID id) const {
    typedef IndexRangeAssociation::id_offset_vector::const_iterator iter;
    iter pos = std::lower_bound(id_offsets_.begin(), id_offsets_.end(), id, IDComparator());
    return (pos != id_offsets_.end()) && (pos->first == id);
}

void
IndexRangeAssociation::swap(IndexRangeAssociation &other) {
    if (isFilling_ || other.isFilling_) throw cms::Exception("Busy") << "Can't swap an IndexRangeAssociation while it's being filled!\n";
    id_offsets_.swap(other.id_offsets_);
    ref_offsets_.swap(other.ref_offsets_);
}

IndexRangeAssociation::FastFiller::FastFiller(IndexRangeAssociation &assoc, ProductID id, size_t size) :
    assoc_(assoc), id_(id), 
    start_(assoc.ref_offsets_.empty() ? 0 : assoc.ref_offsets_.size() - 1), // must skip the end marker element
    end_(start_ + size),
    lastKey_(-1)
{
    if (assoc_.isFilling_) throw cms::Exception("Unsupported Operation") << 
        "IndexRangeAssociation::FastFiller: you already have one active filler for this map.\n";

    // Look if the key is there, or find the right place to insert it
    typedef IndexRangeAssociation::id_offset_vector::iterator iter;
    iter pos = std::lower_bound(assoc_.id_offsets_.begin(), assoc_.id_offsets_.end(), id, IndexRangeAssociation::IDComparator());

    // Check for duplicate ProductID
    if ((pos != assoc_.id_offsets_.end()) && (pos->first == id)) throw cms::Exception("Duplicated Key") << 
        "IndexRangeAssociation::FastFiller: there is already an entry for ProductID " << id << " in this map.\n";

    // Lock the map    
    assoc_.isFilling_ = true;

    // Insert the key, keeping id_offsets_ sorted
    assoc_.id_offsets_.insert(pos, IndexRangeAssociation::id_off_pair(id, start_));

    int lastEnd = (assoc_.ref_offsets_.empty() ? 0 : assoc_.ref_offsets_.back());
    assoc_.ref_offsets_.resize(end_ + 1, -1);
    assoc_.ref_offsets_.back() = lastEnd;
}

IndexRangeAssociation::FastFiller::~FastFiller() {
    // I have to consolidate, replacing "-1" with the correct end offset.
    // I can start from the end, as I know that the last item is never -1
    typedef IndexRangeAssociation::offset_vector::iterator IT;
    //std::cout << "Fixupping [" << start_ << ", " << end_ << "]" << std::endl;
    //for(IT i = assoc_.ref_offsets_.begin() + start_; i <= assoc_.ref_offsets_.begin() + end_; ++i) { std::cout << "  - " << *i << std::endl; }
    IT top = assoc_.ref_offsets_.begin() + start_;
    IT it  = assoc_.ref_offsets_.begin() + end_;
    int offset = *it;
    for (--it; it >= top; --it) {
        if (*it == -1) {
            //std::cout << " > replace *it " << *it << " with offset " << offset << " at " << (it - top) << std::endl;
            *it = offset; // replace -1 with real end offset
        } else {
            //std::cout << " > replace offset " << offset << " with *it " << *it << " at " << (it - top) << std::endl;
            offset = *it;           // take as new end offset for the preceding "-1"s
        }
    }
    assoc_.isFilling_ = false; // unlock
    //std::cout << "Fixupped [" << start_ << ", " << end_ << "]" << std::endl;
    //for(IT i = assoc_.ref_offsets_.begin() + start_; i <= assoc_.ref_offsets_.begin() + end_; ++i) { std::cout << "  - " << *i << std::endl; }
}

void
IndexRangeAssociation::FastFiller::insert(edm::ProductID id, size_t key, size_t startingOffset, size_t size) {
    if (id != id_) IndexRangeAssociation::throwUnexpectedProductID(id,id_,"FastFiller::insert");
    if (int(key) <= lastKey_) throw cms::Exception("Bad Key") << 
            "IndexRangeAssociation::FastFiller: you must fill this in strict key order\n" << 
            "\tLast key = " << lastKey_ << ", this key = " << key << "\n";
    if (key >= end_)  throw cms::Exception("Bad Key") <<
            "IndexRangeAssociation::FastFiller: key index out of bounds for this collection\n" << 
            "\tKey = " << key << ", bound = " << end_ << "\n";
    if ((assoc_.ref_offsets_.back() != 0) && (int(startingOffset) != assoc_.ref_offsets_.back())) 
        throw cms::Exception("Bad Offset") <<
            "IndexRangeAssociation::FastFiller: The start for this key is not the end of the preceding key.\n" << 
            "\tThis offset = " << startingOffset << ", last key = " << lastKey_ << 
            ", last end offset = " << assoc_.ref_offsets_.back() << "\n";
    assoc_.ref_offsets_[start_ + key] = startingOffset; 
    lastKey_ = key;
    assoc_.ref_offsets_.back() += size;
}

void
IndexRangeAssociation::throwUnexpectedProductID(ProductID found, ProductID expected, const char *where) {
    throw cms::Exception("Unexpected ProductID") << where <<
          ": found product id " << found << ", while expecting " << expected << ".\n" << 
          "Make sure you're not mismatching references from different collections.\n";
}
