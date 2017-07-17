#ifndef __DataFormats_PatCandidates_VIDResult_H__
#define __DataFormats_PatCandidates_VIDResult_H__

#include <map>
#include <vector>
#include <string>

/*********
 *
 * Class: vid::CutFlowResult
 * Author: L. Gray (FNAL)
 * 
 * A synthesis of the output of a VID selector into an easily
 * manipulated class, such that cuts can be masked and sidebands
 * created without major intervention on the part of the person
 * doing analysis.
 *
 * The class is self-describing and the original cut-names, the
 * values cut upon, the result of each cut used, and final cutflow 
 * decision can be accessed with this class. Using this information
 * the cut flow can be masked interactively by the user, allowing
 * for a large degree of flexibility at the analysis level.
 *
 *********/

namespace vid {
  class CutFlowResult {    
    template<class T> friend class VersionedSelector;
  public:    

    CutFlowResult() : bitmap_(0) {}
    CutFlowResult(const std::string& name,
                  const std::string& hash,
                  const std::map<std::string,unsigned>& n2idx,                   
                  const std::vector<double>& values,
                  unsigned bitmap,
                  unsigned mask = 0);

    // get the original name of this cutflow
    const std::string& cutFlowName()   const { return name_; }
    // get the md5 hash for this cutflow
    const std::string& cutFlowHash()   const { return hash_; }
    // did this cutflow (in its current state!) pass?
    bool               cutFlowPassed() const { 
      const unsigned all_pass = (1 << indices_.size()) - 1;
      return (all_pass&bitmap_) == all_pass; 
    } 
    // how many cuts in this cutflow?
    size_t             cutFlowSize()   const { return indices_.size(); } 
    
    // get the name of a cut in the cutflow
    // indexed by order it was executed
    const std::string& getNameAtIndex(const unsigned idx) const;
    
    // get the individual cut result (pass/fail) either by name or by index
    bool getCutResultByIndex(const unsigned idx) const;
    bool getCutResultByName(const std::string& name) const;

    // return true if the cut as index/name is masked out
    bool isCutMasked(const unsigned idx) const;
    bool isCutMasked(const std::string& name) const;

    // get the value of variable that was cut on, either by name or by index
    double getValueCutUpon(const unsigned idx) const;
    double getValueCutUpon(const std::string& name) const;

    // create a new copy of this cutflow masking out the listed cuts
    // can be done either by name or by index
    CutFlowResult getCutFlowResultMasking(const unsigned idx) const;
    CutFlowResult getCutFlowResultMasking(const std::string& name) const;
    
    CutFlowResult getCutFlowResultMasking(const std::vector<unsigned>& idxs) const;
    CutFlowResult getCutFlowResultMasking(const std::vector<std::string>& names) const;

  private:
    std::string name_, hash_;
    unsigned bitmap_, mask_;
    std::vector<double> values_;
    std::vector<std::string> names_;
    std::vector<unsigned> indices_;

    CutFlowResult(const std::string& name,
                  const std::string& hash,
                  const std::vector<std::string>& names,
                  const std::vector<unsigned>& indices,
                  const std::vector<double>& values,
                  unsigned bitmap,
                  unsigned mask) :
      name_(name),
      hash_(hash),
      bitmap_(bitmap),
      mask_(mask),
      values_(values),
      names_(names),
      indices_(indices) {}
      

    bool getMaskBit(const unsigned idx) const {
      return (bool)(0x1&(mask_>>idx));
    }

    bool getCutBit(const unsigned idx) const { 
      return (bool)(0x1&(bitmap_>>idx)); 
    }

    double getCutValue(const unsigned idx) const {
      return values_[idx];
    } 
  };
}

#endif
