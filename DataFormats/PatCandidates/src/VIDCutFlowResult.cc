#include "DataFormats/PatCandidates/interface/VIDCutFlowResult.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace {
  static const std::string empty_str("");
}

namespace vid {
  const std::string& CutFlowResult::getNameAtIndex(const unsigned idx) const {
    for( const auto& value : name_to_index_ ) {
      if( value.second == idx ) return value.first;
    }
    throw cms::Exception("IndexNotFound")
      << "index = " << idx << " has no corresponding cut name!";
    return empty_str;
  }

  bool CutFlowResult::getCutResultByIndex(const unsigned idx) const {
    if( idx >= name_to_index_.size() ) {
        throw cms::Exception("OutOfBounds")
          << idx << " is out of bounds for this cut flow!";
    }
    return getCutBit(idx); 
  }

  bool CutFlowResult::getCutResultByName(const std::string& name) const {
    auto idx = name_to_index_.find(name);
    if( idx == name_to_index_.end() ) {
      throw cms::Exception("UnknownName")
        << "Cut name: " << name 
        << " is not known for this cutflow!";
    }
    return getCutBit(idx->second);
  }

  double CutFlowResult::getValueCutUpon(const unsigned idx) const {
    if( idx >= name_to_index_.size() ) {
        throw cms::Exception("OutOfBounds")
          << idx << " is out of bounds for this cut flow!";
    }
    return getCutValue(idx);    
  }
  
  double CutFlowResult::getValueCutUpon(const std::string& name) const {
    auto idx = name_to_index_.find(name);
    if( idx == name_to_index_.end() ) {
      throw cms::Exception("UnknownName")
        << "Cut name: " << name 
        << " is not known for this cutflow!";
    }
    return getCutValue(idx->second);
  }

  CutFlowResult CutFlowResult::
  getCutFlowResultMasking(const std::vector<unsigned>& idxs) const {
    unsigned bitmap = bitmap_;
    for( const unsigned idx : idxs ) {
      if( idx >= name_to_index_.size() ) {
        throw cms::Exception("OutOfBounds")
          << idx << " is out of bounds for this cut flow!";
      }
      bitmap = bitmap | 1 << idx;
    }
    return CutFlowResult(name_,name_to_index_,bitmap,values_);
  }

  CutFlowResult CutFlowResult::
  getCutFlowResultMasking(const std::vector<std::string>& names) const {
    unsigned bitmap = bitmap_;
    for( const std::string& name : names ) {
      auto idx = name_to_index_.find(name);
      if( idx == name_to_index_.end() ) {
        throw cms::Exception("UnknownName")
          << "Cut name: " << name 
          << " is not known for this cutflow!";
      }
      bitmap = bitmap | 1 << idx->second;
    }
    return CutFlowResult(name_,name_to_index_,bitmap,values_);
  }

  CutFlowResult CutFlowResult::
  getCutFlowResultMasking(const unsigned idx) const {
    unsigned bitmap = bitmap_;   
    if( idx >= name_to_index_.size() ) {
      throw cms::Exception("OutOfBounds")
        << idx << " is out of bounds for this cut flow!";
    }
    bitmap = bitmap | 1 << idx;
    return CutFlowResult(name_,name_to_index_,bitmap,values_);
  }
  
  CutFlowResult CutFlowResult::
  getCutFlowResultMasking(const std::string& name) const {
    unsigned bitmap = bitmap_;
    auto idx = name_to_index_.find(name);
    if( idx == name_to_index_.end() ) {
      throw cms::Exception("UnknownName")
        << "Cut name: " << name 
        << " is not known for this cutflow!";
    }
    bitmap = bitmap | 1 << idx->second;
    return CutFlowResult(name_,name_to_index_,bitmap,values_);
  }
}
