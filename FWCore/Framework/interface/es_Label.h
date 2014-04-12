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
#include "boost/shared_ptr.hpp"
#include <string>
#include <vector>

// user include files
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"

// forward declarations

namespace edm {
   namespace es{
      template<typename T, int ILabel>
      struct L {
         typedef T element_type;
         
         L() : product_() {}
         explicit L(boost::shared_ptr<T> iP) : product_(iP) {}
         explicit L(T* iP) : product_(iP) {}
         
         T& operator*() { return *product_;}
         T* operator->() { return product_.get(); }
         mutable boost::shared_ptr<T> product_;
      };
      template<int ILabel,typename T>
         L<T,ILabel> l(boost::shared_ptr<T>& iP) { 
            L<T,ILabel> temp(iP);
            return temp;
         }
      struct Label {
         Label() : labels_(), default_() {}
         Label(const char* iLabel) : labels_(), default_(iLabel) {}
         Label(const std::string& iString) : labels_(), default_(iString) {}
         Label(const std::string& iString, unsigned int iIndex) : 
           labels_(iIndex+1,def()), default_() {labels_[iIndex] = iString;}
         
         Label& operator()(const std::string& iString, unsigned int iIndex) {
            if(iIndex==labels_.size()){
               labels_.push_back(iString);
            } else if(iIndex > labels_.size()) {
               std::vector<std::string> temp(iIndex+1,def());
               copy_all(labels_, temp.begin());
               labels_.swap(temp);
            } else {
               if( labels_[iIndex] != def() ) {
                  Exception e(errors::Configuration,"Duplicate Label");
                  e <<"The index "<<iIndex<<" was previously assigned the label \""
                    <<labels_[iIndex]<<"\" and then was later assigned \""
                    <<iString<<"\"";
                  e.raise();
               }
               labels_[iIndex] = iString;
            }
            return *this;
         }
         Label& operator()(int iIndex, const std::string& iString) {
            return (*this)(iString, iIndex);
         }
         
         static const std::string& def() {
            static const std::string s_def("\n\t");
            return s_def;
         }
         std::vector<std::string> labels_;
         std::string default_;
      };
      
      inline Label label(const std::string& iString, int iIndex) {
         return Label(iString, iIndex);
      }
      inline Label label(int iIndex, const std::string& iString) {
         return Label(iString, iIndex);
      }
   }
}   

#endif
