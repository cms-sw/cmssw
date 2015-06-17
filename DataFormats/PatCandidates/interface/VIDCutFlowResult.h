#ifndef __DataFormats_PatCandidates_VIDResult_H__
#define __DataFormats_PatCandidates_VIDResult_H__

#include <unordered_map>
#include <vector>
#include <string>

namespace vid {
  class CutFlowResult {    
  public:
    template<class T> friend class VersionedSelector;

    CutFlowResult() : bitmap_(0) {}
    
    bool cutflowPassed() const { return (bool)bitmap_; } 
    size_t cutflowSize() const { return name_to_index_.size(); } 

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
    unsigned bitmap_;
    std::vector<double> values_;
    std::unordered_map<std::string,unsigned> name_to_index_;

    bool getCutBit(const unsigned idx) const { 
      return (bool)(0x1&(bitmap_>>idx)); 
    }

    bool getCutValue(const unsigned idx) const {
      return values_[idx];
    }

    CutFlowResult(const std::unordered_map<std::string,unsigned>& n2idx,
                  unsigned bitmap, 
                  const std::vector<double>& values) : 
    bitmap_(bitmap), values_(values), name_to_index_(n2idx) {}
    
  };
}

#endif
