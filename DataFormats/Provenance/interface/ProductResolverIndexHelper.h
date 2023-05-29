#ifndef DataFormats_Provenance_ProductResolverIndexHelper_h
#define DataFormats_Provenance_ProductResolverIndexHelper_h

/** \class edm::ProductResolverIndexHelper

This class assigns and gets the ProductResolverIndex
associated with a type, module label, instance, and
process.  The ProductResolverIndex is used to tell the
Principal where to store a ProductResolver and how to find
it quickly.

One can also look up the same ProductResolverIndex's using
the type or base type of an element in a container in the
product (if the product is a container). In this case the
KindOfType argument to the Principal::getByLabel function
is ELEMENT_TYPE, whereas normally it is PRODUCT_TYPE.

There are also special ProductResolverIndex's generated
where the process name is empty. These indexes refer
to a special ProductResolvers that search for a matching
product from the most recent process that has a matching
type, label and instance. There is ProductResolverIndex
generated for each type/label/instance combination which
has at least one entry in the tables in this class.
Both PRODUCT_TYPEs and ELEMENT_TYPEs get these special
ProductResolvers.

The ProductResolverIndex for a particular product
will not change during a process after the ProductRegistry
has been frozen. Nor will any of the other member data
of this class. Multiple threads can access it concurrently
without problems. The ProductResolverIndexes can be safely
cached in InputTags and possibly other places, because
they never change within a process. The ProductResolverIndex
for a particular product is not intended to be persistent
and will be different in different processes.

The ProductResolverIndex is used to order the placement of
the ProductResolvers in the Principal that are either
present in the input or produced in the current process.
Be aware that there are other ProductResolvers for products
that come after ProductResolvers placed by this class.
For example, the placement of dropped products
is not handled by this class, instead by the ProductRegistry.
The reason for this distinction is that those other
ProductResolvers can change and be added as a process runs.
The content of this class never changes after the
ProductRegistry is frozen.

\author W. David Dagenhart, created 10 December, 2012 

*/

#include "FWCore/Utilities/interface/ProductResolverIndex.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <iosfwd>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>

namespace edm {

  class TypeWithDict;

  namespace productholderindexhelper {
    // The next function supports views. For the given wrapped type,
    // which must be Wrapper<T>,
    // this function returns the type of the contained type of T.
    // If the type is not a recongnized container, it returns
    // a TypeID(typeid(void)).
    TypeID getContainedTypeFromWrapper(TypeID const& wrappedtypeID, std::string const& className);

    // The next function supports views. For the given type T,
    // this function returns the type of the contained type of T.
    // If the type is not a recongnized container, it returns
    // a TypeID(typeid(void)).
    // This calls getContainedTypefromWrapped internally
    // If the TypeID for the wrapped type is already available,
    // it is faster to call getContainedTypeFromWrapper directly.
    TypeID getContainedType(TypeID const& typeID);

    bool typeIsViewCompatible(TypeID const& requestedViewType,
                              TypeID const& wrappedtypeID,
                              std::string const& className);
  }  // namespace productholderindexhelper

  class ProductResolverIndexHelper {
  public:
    ProductResolverIndexHelper();

    // The accessors below return a ProductResolverIndex that matches
    // the arguments or a set of matching indexes using the Matches
    // class. A returned index can have a value that indicates that it
    // is invalid or ambiguous and the client should check for these
    // values before using the index (see ProductResolverIndex.h).

    // If no matches are found or the ProductResolverIndexHelper
    // has not been frozen yet, then an invalid index or a Matches
    // object with numberOfMatches equal to 0 will be returned.

    // The ambiguous values can occur when the kind of type is ELEMENT_TYPE
    // type. ELEMENT_TYPE is the type or base type of an object
    // stored in a container that is the product. These types are used
    // with data requests that use Views. For PRODUCT_TYPE types, ambiguity
    // is not possible.

    // The next function returns the index for the one group that
    // matches the type, module label, instance, and process.
    // The 3 pointer arguments must point to C style strings terminated
    // by a '\0' with the one following possible exception. If the
    // process pointer is null or the process string empty, then
    // this returns the index of the special ProductResolver that
    // knows how to search for the product matching the type, module
    // label, and instance and which is from the most recent process.
    ProductResolverIndex index(KindOfType kindOfType,
                               TypeID const& typeID,
                               char const* moduleLabel,
                               char const* instance,
                               char const* process = nullptr) const;

    using ModulesToIndiciesMap =
        std::unordered_multimap<std::string, std::tuple<TypeID const*, const char*, ProductResolverIndex>>;
    ModulesToIndiciesMap indiciesForModulesInProcess(const std::string& iProcessName) const;

    class Matches {
    public:
      Matches(ProductResolverIndexHelper const* productResolverIndexHelper,
              unsigned int startInIndexAndNames,
              unsigned int numberOfMatches);

      ProductResolverIndex index(unsigned int i) const;
      unsigned int numberOfMatches() const { return numberOfMatches_; }
      bool isFullyResolved(unsigned int i) const;
      char const* moduleLabel(unsigned int i) const;
      char const* productInstanceName(unsigned int i) const;
      char const* processName(unsigned int i) const;

    private:
      ProductResolverIndexHelper const* productResolverIndexHelper_;
      unsigned int startInIndexAndNames_;
      unsigned int numberOfMatches_;
    };

    // Return ProductResolverIndex's for all product holders that
    // match the type, module label, and product instance name.
    // The pointer arguments must be C style strings terminated
    // by a '\0'.
    Matches relatedIndexes(KindOfType kindOfType,
                           TypeID const& typeID,
                           char const* moduleLabel,
                           char const* instance) const;

    // Return indexes for all groups that match the type.
    Matches relatedIndexes(KindOfType kindOfType, TypeID const& typeID) const;

    // This will throw if called after the object is frozen.
    // The typeID must be for a type with a dictionary
    // (the calling function is expected to check that)
    // The pointer arguments must point at C style strings
    // terminated by '\0'.
    // 1. This creates an entry and new ProductResolverIndex for
    // the product if it does not already exist. If it
    // does exist then it throws.
    // 2. If it does not already exist, this will create an
    // entry and new ProductResolverIndex for the ProductResolver
    // that will search for the matching type, label, and
    // instance for the most recent process (internally indicated
    // by an empty process string).
    // This will then loop over the contained class (if it exists)
    // and the base classes of the contained class.
    // 1. If the matching type, label, instance, and process
    // already exist then that entry is modified and marked
    // ambiguous. If not, it inserts an entry which uses the same
    // ProductResolverIndex as the containing product.
    // 2. If it does not already exist it inserts a new
    // entry with a new ProductResolverIndex for the case
    // which searches for the most recent process.
    ProductResolverIndex insert(TypeID const& typeID,
                                char const* moduleLabel,
                                char const* instance,
                                char const* process,
                                TypeID const& containedTypeID,
                                std::vector<TypeID>* baseTypesOfContainedType);

    ProductResolverIndex insert(TypeID const& typeID,
                                char const* moduleLabel,
                                char const* instance,
                                char const* process);

    // Before the object is frozen the accessors above will
    // fail to find a match. Once frozen, no more new entries
    // can be added with insert.
    void setFrozen();

    std::vector<std::string> const& lookupProcessNames() const;

    class Range {
    public:
      Range(unsigned int begin, unsigned int end) : begin_(begin), end_(end) {}
      unsigned int begin() const { return begin_; }
      unsigned int end() const { return end_; }

    private:
      unsigned int begin_;
      unsigned int end_;
    };

    class IndexAndNames {
    public:
      IndexAndNames(ProductResolverIndex index, unsigned int start, unsigned int startProcess)
          : index_(index), startInBigNamesContainer_(start), startInProcessNames_(startProcess) {}
      ProductResolverIndex index() const { return index_; }
      unsigned int startInBigNamesContainer() const { return startInBigNamesContainer_; }
      unsigned int startInProcessNames() const { return startInProcessNames_; }

    private:
      ProductResolverIndex index_;
      unsigned int startInBigNamesContainer_;
      unsigned int startInProcessNames_;
    };

    unsigned int beginElements() const { return beginElements_; }
    std::vector<TypeID> const& sortedTypeIDs() const { return sortedTypeIDs_; }
    std::vector<Range> const& ranges() const { return ranges_; }
    std::vector<IndexAndNames> const& indexAndNames() const { return indexAndNames_; }
    std::vector<char> const& processNames() const { return processNames_; }

    // The next few functions are intended for internal use
    // but are public so tests can use them also.

    unsigned int indexToIndexAndNames(KindOfType kindOfType,
                                      TypeID const& typeID,
                                      char const* moduleLabel,
                                      char const* instance,
                                      char const* process) const;

    // Returns the index into sortedTypeIDs_. Returns the maximum unsigned
    // int value if the type is not there.
    unsigned int indexToType(KindOfType kindOfType, TypeID const& typeID) const;

    // Returns the index of the process name in processNames_. Returns the
    // maximum unsigned int value if the process name is not found.
    unsigned int processIndex(char const* process) const;

    // This will throw if it detects problems, but unless there is a bug
    // there should never be any. Mostly it is checking for things which
    // might cause out of bounds errors when accessing the vectors.
    void sanityCheck() const;

    ProductResolverIndex nextIndexValue() const { return nextIndexValue_; }

    // For debugging only
    void print(std::ostream& os) const;

  private:
    // Next available value for a ProductResolverIndex. This just
    // increments by one each time a new value is assigned.
    ProductResolverIndex nextIndexValue_;

    // This is an index into sortedTypeIDs_ that tells where
    // the entries corresponding to types of elements in containers
    // start.
    unsigned int beginElements_;

    // Sorted by putting all the product entries first and
    // then the element entries.  Then sorted by TypeID value.
    // Most lookups start here by finding the TypeID and
    // then using its position to find the corresponding Range
    // in the ranges_ vector below.
    std::vector<TypeID> sortedTypeIDs_;

    // There is a one to one correspondence between this vector
    // and sortedTypeIDs_ and the corresponding elements appear
    // in the same order. Each Range object holds the beginning and
    // end of the corresponding elements in the indexAndNames_
    // vector below.
    std::vector<Range> ranges_;

    // Each element of this vector contains a ProductResolverIndex.
    // It also contains indexes into vectors of characters that
    // hold the corresponding moduleLabel, instance, and process
    // name. Note this is sorted with product entries first then
    // element entries, then by TypeID, then by moduleLabel,
    // then by instance, and finally by process. Note that in
    // a subset of entries that have matching type/label/instance
    // the entry with an empty process name will always be the
    // first process and startInProcessNames_ will always be 0
    // for the empty process name.
    std::vector<IndexAndNames> indexAndNames_;

    // These contain C style strings terminated
    // by '\0' all concatenated together. In the
    // first vector the strings come in pairs,
    // always module label then instance.  The pairs
    // are ordered the same as the IndexAndNames
    // vector. Within a TypeID, the moduleLabel/instance
    // pair is not duplicated if it appears multiple times.
    // The second vector contains all the process names
    // in alphabetical order with no duplication. The
    // first element is always the empty string.
    std::vector<char> bigNamesContainer_;
    std::vector<char> processNames_;

    // Duplicates the entries in processNames_ in
    // a convenient format.
    std::vector<std::string> lookupProcessNames_;

    // The rest of the data members are for temporary use
    // while the data structure is being filled.

    class Item {
    public:
      Item(KindOfType kindOfType,
           TypeID const& typeID,
           std::string const& moduleLabel,
           std::string const& instance,
           std::string const& process,
           ProductResolverIndex index);
      KindOfType kindOfType() const { return kindOfType_; }
      TypeID const& typeID() const { return typeID_; }
      std::string const& moduleLabel() const { return moduleLabel_; }
      std::string const& instance() const { return instance_; }
      std::string const& process() const { return process_; }
      ProductResolverIndex index() const { return index_; }

      void clearProcess() { process_.clear(); }
      void setIndex(ProductResolverIndex v) { index_ = v; }

      bool operator<(Item const& right) const;

    private:
      KindOfType kindOfType_;
      TypeID typeID_;
      std::string moduleLabel_;
      std::string instance_;
      std::string process_;
      ProductResolverIndex index_;
    };

    edm::propagate_const<std::unique_ptr<std::set<Item>>> items_;

    edm::propagate_const<std::unique_ptr<std::set<std::string>>> processItems_;
  };
}  // namespace edm
#endif
