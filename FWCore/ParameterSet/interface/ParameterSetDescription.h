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
// $Id$
//

// system include files
#include <vector>
#include <boost/shared_ptr.hpp>

// user include files

// forward declarations
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ParameterSet/interface/ParameterDescriptionTemplate.h"

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
      
      /**This is set only for parameterizables which have not set their descriptions.
        This should only be called to allow backwards compatibility.
        */
      void setUnknown();
      
      template<class T>
        void add(const std::string& iLabel) {
          parameters_.push_back( boost::shared_ptr<ParameterDescription>(new ParameterDescriptionTemplate<T>(iLabel,true) ) );
        }

      template<class T>
        void addUntracked(const std::string& iLabel) {
          parameters_.push_back( new ParameterDescriptionTemplate<T>(iLabel,false) );
        }
      // ---------- const member functions ---------------------
      //Throws a cms::Exception if invalid
      void validate(const edm::ParameterSet& ) const;

      
      bool anythingAllowed() const {
        return anythingAllowed_;
      }
      
      bool isUnknown() const {
        return unknown_;
      }
      
      parameter_const_iterator parameter_begin() const {
        return parameters_.begin();
      }
      parameter_const_iterator parameter_end() const {
        return parameters_.end();
      }
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      //ParameterSetDescription(const ParameterSetDescription&); // stop default

      //const ParameterSetDescription& operator=(const ParameterSetDescription&); // stop default

      // ---------- member data --------------------------------
        bool anythingAllowed_;
        bool unknown_;
        Parameters parameters_;
};

}
#endif
