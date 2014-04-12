// -*- C++ -*-
// 
//

// class ParameterCollector provides a tool to parse blocks of PSets
// like "PythiaParameters" where one can list blocks of vstrings to be
// parsed inside another vstring named "parameterSets".  It is also
// extended to allow nesting of blocks using "+block" statements
// (something which is heavily used for Herwig++).
// arbitrary vstring blocks can also be explicitly retrieved by name.

#ifndef gen_ParameterCollector_h
#define gen_ParameterCollector_h

#include <ostream>
#include <vector>
#include <string>
#include <map>

#include <boost/iterator/iterator_facade.hpp>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace gen {

class ParameterCollector {
public:
   ParameterCollector();
   ParameterCollector(const edm::ParameterSet &pset);
   ~ParameterCollector();

   // this iterator makes begin()/end() look like it was
   // looping over a simple vector<string>

   class const_iterator :
      public boost::iterator_facade<const_iterator, const std::string,
                                    boost::forward_traversal_tag> {
   public:
      const_iterator() : collector_(0), dump_(0) {}

   protected:
      friend class ParameterCollector;

      inline const_iterator(const ParameterCollector *collector,
                            std::vector<std::string>::const_iterator begin,
                            std::vector<std::string>::const_iterator end,
                            bool special = false, std::ostream *dump = 0);

   private:
      friend class boost::iterator_core_access;

      void increment();
      const std::string &dereference() const { return cache_; }
      bool equal(const const_iterator &other) const
      { return iter_ == other.iter_; }

      void next();

      typedef std::pair<std::vector<std::string>::const_iterator,
                        std::vector<std::string>::const_iterator> IterPair;

      const ParameterCollector	*collector_;
      std::ostream		*dump_;
      bool			special_;
      std::vector<IterPair>	iter_;
      std::string		cache_;
   };

   // start iterating over blocks listed in "parameterSets"
   const_iterator begin() const;
   const_iterator begin(std::ostream &dump) const;

   // start iterating over contents of this particular vstring block
   const_iterator begin(const std::string &block) const;
   const_iterator begin(const std::string &block, std::ostream &dump) const;

   // the iterator to mark the end of a loop
   const_iterator end() const { return const_iterator(); }

   // this replaces ${...} with environment variables
   // needed for ThePEG because you need full path to repository
   static std::string resolve(const std::string &line);

private:
   friend class const_iterator;

   std::map<std::string, std::vector<std::string> > contents_;
};

} // namespace gen

#endif // gen_ParameterCollector_h
