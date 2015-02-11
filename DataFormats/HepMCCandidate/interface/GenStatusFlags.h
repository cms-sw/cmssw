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
      kIsHardProcessTauDecayProduct,
      kIsDirectHardProcessTauDecayProduct,
      kIsLastCopy
    };
    
    //is particle prompt (not from hadron, muon, or tau decay)
    bool isPrompt() const { return flags_[kIsPrompt]; }
    void setIsPrompt(bool b) { flags_[kIsPrompt] = b; }    
    
    //this particle is a direct or indirect tau decay product
    bool isTauDecayProduct() const { return flags_[kIsTauDecayProduct]; }
    void setIsTauDecayProduct(bool b) { flags_[kIsTauDecayProduct] = b; }    
    
    //this particle is a direct or indirect decay product of a prompt tau
    bool isPromptTauDecayProduct() const { return flags_[kIsPromptTauDecayProduct]; }
    void setIsPromptTauDecayProduct(bool b) { flags_[kIsPromptTauDecayProduct] = b; }    
    
    //this particle is a direct tau decay product
    bool isDirectTauDecayProduct() const { return flags_[kIsDirectTauDecayProduct]; }
    void setIsDirectTauDecayProduct(bool b) { flags_[kIsDirectTauDecayProduct] = b; }    
    
    //this particle is a direct decay product from a prompt tau 
    bool isDirectPromptTauDecayProduct() const { return flags_[kIsDirectPromptTauDecayProduct]; }
    void setIsDirectPromptTauDecayProduct(bool b) { flags_[kIsDirectPromptTauDecayProduct] = b; }    
    
    //this particle is a direct or indirect muon decay product
    bool isMuonDecayProduct() const { return flags_[kIsMuonDecayProduct]; }
    void setIsMuonDecayProduct(bool b) { flags_[kIsMuonDecayProduct] = b; }    
    
    //this particle is a direct or indirect decay product of a prompt muon
    bool isPromptMuonDecayProduct() const { return flags_[kIsPromptMuonDecayProduct]; }
    void setIsPromptMuonDecayProduct(bool b) { flags_[kIsPromptMuonDecayProduct] = b; }    
    
    //this particle is a direct decay product from a hadron
    bool isDirectHadronDecayProduct() const { return flags_[kIsDirectHadronDecayProduct]; }
    void setIsDirectHadronDecayProduct(bool b) { flags_[kIsDirectHadronDecayProduct] = b; }    
    
    //this particle is part of the hard process
    bool isHardProcess() const { return flags_[kIsHardProcess]; }
    void setIsHardProcess(bool b) { flags_[kIsHardProcess] = b; }    
    
    //this particle is the direct descendant of a hard process particle of the same pdg id
    bool fromHardProcess() const { return flags_[kFromHardProcess]; }
    void setFromHardProcess(bool b) { flags_[kFromHardProcess] = b; }    
    
    //this particle is a direct or indirect decay product of a tau
    //from the hard process
    bool isHardProcessTauDecayProduct() const { return flags_[kIsHardProcessTauDecayProduct]; }
    void setIsHardProcessTauDecayProduct(bool b) { flags_[kIsHardProcessTauDecayProduct] = b; }    
    
    //this particle is a direct decay product of a tau
    //from the hard process
    bool isDirectHardProcessTauDecayProduct() const { return flags_[kIsDirectHardProcessTauDecayProduct]; }
    void setIsDirectHardProcessTauDecayProduct(bool b) { flags_[kIsDirectHardProcessTauDecayProduct] = b; }     
    
    //this particle is the last copy of the particle in the chain with the same pdg id
    //(and therefore is more likely, but not guaranteed, to carry the final physical momentum)    
    bool isLastCopy() const { return flags_[kIsLastCopy]; }
    void setIsLastCopy(bool b) { flags_[kIsLastCopy] = b; }    
    
    std::bitset<13> flags_;
  };

}

#endif
