#ifndef FWCore_ParameterSet_ParameterSetDescription_h
#define FWCore_ParameterSet_ParameterSetDescription_h
// -*- C++ -*-
//
// Package:     ParameterSet
// Class  :     ParameterSetDescription
// 
/**\class ParameterSetDescription ParameterSetDescription.h FWCore/ParameterSet/interface/ParameterSetDescription.h

 Description: Used to describe the allowed values in a ParameterSet

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Jul 31 15:18:40 EDT 2007
// $Id: ParameterSetDescription.h,v 1.1 2007/09/17 21:04:37 chrjones Exp $
//

#include "FWCore/ParameterSet/interface/ParameterDescriptionTemplate.h"

#include <boost/shared_ptr.hpp>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

namespace edm {
  class ParameterSetDescription
  {

  public:
    typedef std::vector<boost::shared_ptr<ParameterDescription> > Parameters;
    typedef Parameters::const_iterator parameter_const_iterator;
        
    ParameterSetDescription();
    virtual ~ParameterSetDescription();

    ///allow any parameter label/value pairs
    void setAllowAnything();
      
    // This is set only for parameterizables which have not set their descriptions.
    // This should only be called to allow backwards compatibility.
    void setUnknown();
      
    template<class T>
      boost::shared_ptr<ParameterDescription> add(const std::string& iLabel, T const& value) {
      boost::shared_ptr<ParameterDescription> ptr(new ParameterDescriptionTemplate<T>(iLabel, true, value));
      parameters_.push_back(ptr);
      return ptr;
    }

    template<class T>
    boost::shared_ptr<ParameterDescription> addUntracked(const std::string& iLabel, T const& value) {
      boost::shared_ptr<ParameterDescription> ptr(new ParameterDescriptionTemplate<T>(iLabel, false, value));
      parameters_.push_back(ptr);
      return ptr;
    }

    //Throws a cms::Exception if invalid
    void validate(const edm::ParameterSet& ) const;

    bool anythingAllowed() const { return anythingAllowed_; }
    bool isUnknown() const { return unknown_; }

    parameter_const_iterator parameter_begin() const {
      return parameters_.begin();
    }

    parameter_const_iterator parameter_end() const {
      return parameters_.end();
    }

  private:

    bool anythingAllowed_;
    bool unknown_;
    Parameters parameters_;
  };
}
#endif
