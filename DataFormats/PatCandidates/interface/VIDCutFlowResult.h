#ifndef __DataFormats_PatCandidates_VIDResult_H__
#define __DataFormats_PatCandidates_VIDResult_H__

#include <map>
#include <vector>
#include <string>

namespace vid {
  class CutFlowResult {    
  public:
    template<class T> friend class VersionedSelector;

    CutFlowResult() : bitmap_(0) {}
    CutFlowResult(const std::string& name,
                  const std::map<std::string,unsigned>& n2idx,
                  unsigned bitmap, 
                  const std::vector<double>& values) : 
      name_(name),
      bitmap_(bitmap), 
      values_(values), 
      name_to_index_(n2idx) {}
    
    const std::string& cutFlowName()   const { return name_; }
    bool               cutFlowPassed() const { 
      const unsigned all_pass = name_to_index_.size()-1;
      return (all_pass&bitmap_) == all_pass; 
    } 
    size_t             cutFlowSize()   const { return name_to_index_.size(); } 

    const std::string& getNameAtIndex(const unsigned idx) const;
    
    bool getCutResultByIndex(const unsigned idx) const;
    bool getCutResultByName(const std::string& name) const;

    double getValueCutUpon(const unsigned idx) const;
    double getValueCutUpon(const std::string& name) const;

    CutFlowResult getCutFlowResultMasking(const unsigned idx) const;
    CutFlowResult getCutFlowResultMasking(const std::string& name) const;
    
    CutFlowResult getCutFlowResultMasking(const std::vector<unsigned>& idxs) const;
    CutFlowResult getCutFlowResultMasking(const std::vector<std::string>& names) const;

  private:
    std::string name_;
    unsigned bitmap_;
    std::vector<double> values_;
    std::map<std::string,unsigned> name_to_index_;

    bool getCutBit(const unsigned idx) const { 
      return (bool)(0x1&(bitmap_>>idx)); 
    }

    bool getCutValue(const unsigned idx) const {
      return values_[idx];
    } 
  };
}

#endif
