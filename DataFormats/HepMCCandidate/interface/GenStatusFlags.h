#ifndef HepMCCandidate_GenStatusFlags_h
#define HepMCCandidate_GenStatusFlags_h
/** \class reco::GenStatusFlags
 *
 * enum for generator status flags
 *
 * \author: Josh Bendavid
 *
 */

#include <bitset>

namespace reco {

  struct GenStatusFlags {
    
    enum StatusBits {
      kIsPrompt = 0,
      kIsTauDecayProduct,
      kIsPromptTauDecayProduct,
      kIsDirectTauDecayProduct,
      kIsDirectPromptTauDecayProduct,
      kIsMuonDecayProduct,
      kIsPromptMuonDecayProduct,
      kIsDirectHadronDecayProduct,
      kIsHardProcess,
      kFromHardProcess,
      kIsLastCopy
    };
    
    bool isPrompt() const { return flags_[kIsPrompt]; }
    void setIsPrompt(bool b) { flags_[kIsPrompt] = b; }    
    
    bool isTauDecayProduct() const { return flags_[kIsTauDecayProduct]; }
    void setIsTauDecayProduct(bool b) { flags_[kIsTauDecayProduct] = b; }    
    
    bool isPromptTauDecayProduct() const { return flags_[kIsPromptTauDecayProduct]; }
    void setIsPromptTauDecayProduct(bool b) { flags_[kIsPromptTauDecayProduct] = b; }    
    
    bool isDirectTauDecayProduct() const { return flags_[kIsDirectTauDecayProduct]; }
    void setIsDirectTauDecayProduct(bool b) { flags_[kIsDirectTauDecayProduct] = b; }    
    
    bool isDirectPromptTauDecayProduct() const { return flags_[kIsDirectPromptTauDecayProduct]; }
    void setIsDirectPromptTauDecayProduct(bool b) { flags_[kIsDirectPromptTauDecayProduct] = b; }    
    
    bool isMuonDecayProduct() const { return flags_[kIsMuonDecayProduct]; }
    void setIsMuonDecayProduct(bool b) { flags_[kIsMuonDecayProduct] = b; }    
    
    bool isPromptMuonDecayProduct() const { return flags_[kIsPromptMuonDecayProduct]; }
    void setIsPromptMuonDecayProduct(bool b) { flags_[kIsPromptMuonDecayProduct] = b; }    
    
    bool isDirectHadronDecayProduct() const { return flags_[kIsDirectHadronDecayProduct]; }
    void setIsDirectHadronDecayProduct(bool b) { flags_[kIsDirectHadronDecayProduct] = b; }    
    
    bool isHardProcess() const { return flags_[kIsHardProcess]; }
    void setIsHardProcess(bool b) { flags_[kIsHardProcess] = b; }    
    
    bool fromHardProcess() const { return flags_[kFromHardProcess]; }
    void setFromHardProcess(bool b) { flags_[kFromHardProcess] = b; }    
    
    bool isLastCopy() const { return flags_[kIsLastCopy]; }
    void setIsLastCopy(bool b) { flags_[kIsLastCopy] = b; }    
    
    std::bitset<11> flags_;
  };

}

#endif
