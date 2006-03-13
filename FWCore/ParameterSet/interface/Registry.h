#ifndef ParameterSet_Registry_h
#define ParameterSet_Registry_h

// ----------------------------------------------------------------------
// $Id: Registry.h,v 1.5 2006/03/08 22:14:52 wmtan Exp $
//
// Declaration for pset::Registry. This is an implementation detail of
// the ParameterSet library.
//
// A Registry is used to keep track of the persistent form of all
// ParameterSets used a given program, so that they may be retrieved by
// ParameterSetID, and so they may be written to persistent storage.
// ----------------------------------------------------------------------

#include <map>
#include <ostream>

#include "DataFormats/Common/interface/ParameterSetID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm
{
  namespace pset
  {

    class Registry
    {
    public:
      //typedef std::set<edm::ParameterSet> collection_type;
      typedef std::map<edm::ParameterSetID, edm::ParameterSet> collection_type;
      typedef collection_type::const_iterator const_iterator;
      typedef collection_type::size_type size_type;

      static Registry* instance();

      /// Retrieve the ParameterSet with the given ID.
      /// If we return 'true', then 'result' carries the ParameterSet.
      /// If we return 'false, no ParameterSet was found, and
      /// the value of 'result' is undefined.

      // We could add another version that returns the ParameterSet (as
      // the function return value), and which throws on lookup failure.
      bool getParameterSet(edm::ParameterSetID const& id,
  			 edm::ParameterSet & result) const;


      /// Insert the *tracked parts* of the given ParameterSet into the
      /// Registry. If there was already a ParameterSet with the same
      /// ID, we don't change itw. This should be OK, since it should
      /// have the same contents if the ID is the same.
      /// Return 'true' if we really added the new ParameterSet, and
      /// 'false' if the ParameterSet was already present.
      bool insertParameterSet(edm::ParameterSet const& p);

      /// Return the number of contained ParameterSets.
      size_type size() const;

      /// Allow iteration through the contents of the Registry.
      const_iterator begin() const;
      const_iterator end() const;

      /// Print the contents of this Registry to the given ostream.
      void print(std::ostream& os) const;

    private:
      Registry();
      Registry(Registry const&); // not implemented
      Registry& operator=(Registry const&); // not implemented
      ~Registry();

      collection_type  psets_;

      static Registry* instance_;
    };

    // Free functions associated with Registry.

    /// Print the contents of the given Registry to the given ostream.
    std::ostream&
    operator<< (std::ostream& os, Registry const& reg);

    void loadAllNestedParameterSets(edm::ParameterSet const& main);



  }  // namespace pset

}  // namespace edm


#endif
