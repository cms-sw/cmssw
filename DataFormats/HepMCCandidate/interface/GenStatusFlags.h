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
      kIsDecayedLeptonHadron,
      kIsTauDecayProduct,
      kIsPromptTauDecayProduct,
      kIsDirectTauDecayProduct,
      kIsDirectPromptTauDecayProduct,
      kIsDirectHadronDecayProduct,
      kIsHardProcess,
      kFromHardProcess,
      kIsHardProcessTauDecayProduct,
      kIsDirectHardProcessTauDecayProduct,
      kFromHardProcessBeforeFSR,
      kIsFirstCopy,
      kIsLastCopy,
      kIsLastCopyBeforeFSR
    };
    
    /////////////////////////////////////////////////////////////////////////////
    //these are robust, generator-independent functions for categorizing
    //mainly final state particles, but also intermediate hadrons/taus    
    
    //is particle prompt (not from hadron, muon, or tau decay)
    bool isPrompt() const { return flags_[kIsPrompt]; }
    void setIsPrompt(bool b) { flags_[kIsPrompt] = b; }    
    
    //is particle a decayed hadron, muon, or tau (does not include resonance decays like W,Z,Higgs,top,etc)
    //This flag is equivalent to status 2 in the current HepMC standard
    //but older generators (pythia6, herwig6) predate this and use status 2 also for other intermediate
    //particles/states    
    bool isDecayedLeptonHadron() const { return flags_[kIsDecayedLeptonHadron]; }
    void setIsDecayedLeptonHadron(bool b) { flags_[kIsDecayedLeptonHadron] = b; }        
    
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
    
    //this particle is a direct decay product from a hadron
    bool isDirectHadronDecayProduct() const { return flags_[kIsDirectHadronDecayProduct]; }
    void setIsDirectHadronDecayProduct(bool b) { flags_[kIsDirectHadronDecayProduct] = b; }    
    
    /////////////////////////////////////////////////////////////////////////////
    //these are generator history-dependent functions for tagging particles
    //associated with the hard process
    //Currently implemented for Pythia 6 and Pythia 8 status codes and history   
    //and may not have 100% consistent meaning across all types of processes
    //Users are strongly encouraged to stick to the more robust flags above    
    
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
    
    //this particle is the direct descendant of a hard process particle of the same pdg id
    //For outgoing particles the kinematics are those before QCD or QED FSR
    //This corresponds roughly to status code 3 in pythia 6    
    bool fromHardProcessBeforeFSR() const { return flags_[kFromHardProcessBeforeFSR]; }
    void setFromHardProcessBeforeFSR(bool b) { flags_[kFromHardProcessBeforeFSR] = b; }    
    
    //this particle is the first copy of the particle in the chain with the same pdg id 
    bool isFirstCopy() const { return flags_[kIsFirstCopy]; }
    void setIsFirstCopy(bool b) { flags_[kIsFirstCopy] = b; }
    
    //this particle is the last copy of the particle in the chain with the same pdg id
    //(and therefore is more likely, but not guaranteed, to carry the final physical momentum)    
    bool isLastCopy() const { return flags_[kIsLastCopy]; }
    void setIsLastCopy(bool b) { flags_[kIsLastCopy] = b; }
    
    //this particle is the last copy of the particle in the chain with the same pdg id
    //before QED or QCD FSR
    //(and therefore is more likely, but not guaranteed, to carry the momentum after ISR)  
    bool isLastCopyBeforeFSR() const { return flags_[kIsLastCopyBeforeFSR]; }
    void setIsLastCopyBeforeFSR(bool b) { flags_[kIsLastCopyBeforeFSR] = b; }
    
    std::bitset<15> flags_;
  };

}

#endif
