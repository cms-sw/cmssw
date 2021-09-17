#ifndef Framework_es_Label_h
#define Framework_es_Label_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     es_Label
//
/**\class es_Label es_Label.h FWCore/Framework/interface/es_Label.h

   Description: Used to assign labels to data items produced by an ESProducer

   Usage:
   See the header file for ESProducer for detail examples

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Sep 30 09:35:20 EDT 2005
//

// system include files
#include <memory>
#include <string>
#include <string_view>
#include <vector>

// user include files
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"

// forward declarations

namespace edm::es {

  template <typename T, int ILabel>
  struct L {
    using element_type = T;

    L() = default;
    explicit L(std::shared_ptr<T> iP) : product_{std::move(iP)} {}
    explicit L(std::unique_ptr<T> iP) : product_{std::move(iP)} {}
    explicit L(T* iP) : product_(iP) {}

    T& operator*() { return *product_; }
    T* operator->() { return product_.get(); }
    T const& operator*() const { return *product_; }
    T const* operator->() const { return product_.get(); }
    mutable std::shared_ptr<T> product_{nullptr};
  };

  template <int ILabel, typename T>
  L<T, ILabel> l(std::shared_ptr<T>& iP) {
    return L<T, ILabel>{iP};
  }

  struct Label {
    Label() = default;
    Label(const char* iLabel) : default_{iLabel} {}
    Label(const std::string& iString) : default_{iString} {}
    Label(const std::string& iString, unsigned int const iIndex) : labels_(iIndex + 1, def()) {
      labels_[iIndex] = iString;
    }

    Label& operator()(const std::string& iString, unsigned int const iIndex) {
      if (iIndex == labels_.size()) {
        labels_.push_back(iString);
      } else if (iIndex > labels_.size()) {
        std::vector<std::string> temp(iIndex + 1, def());
        copy_all(labels_, temp.begin());
        labels_.swap(temp);
      } else {
        if (labels_[iIndex] != def()) {
          Exception e(errors::Configuration, "Duplicate Label");
          e << "The index " << iIndex << " was previously assigned the label \"" << labels_[iIndex]
            << "\" and then was later assigned \"" << iString << "\"";
          e.raise();
        }
        labels_[iIndex] = iString;
      }
      return *this;
    }
    Label& operator()(int iIndex, const std::string& iString) { return (*this)(iString, iIndex); }

    static const std::string& def() {
      static const std::string s_def("\n\t");
      return s_def;
    }

    std::vector<std::string> labels_{};
    std::string default_{};
  };

  inline Label label(const std::string& iString, int iIndex) { return Label(iString, iIndex); }
  inline Label label(int iIndex, const std::string& iString) { return Label(iString, iIndex); }
}  // namespace edm::es

#endif
