#ifndef gen_JetMatchingHook_h
#define gen_JetMatchingHook_h

#include "Pythia8/Pythia.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/LHECommonBlocks.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

#include "GeneratorInterface/PartonShowerVeto/interface/JetMatching.h"

// forward declaration
class Py8toJetInput;

class JetMatchingHook : public Pythia8::UserHooks {
public:
  JetMatchingHook(const edm::ParameterSet&, Pythia8::Info*);
  ~JetMatchingHook() override;

  //
  // Julia Yarba, Jan.8, 2013
  // The "Early" option will work with Pythia8.170 or higher;
  // for lower versions, please use just VetoPartonLevel
  //
  // virtual bool canVetoPartonLevelEarly() { return true; }
  // virtual bool doVetoPartonLevelEarly( const Pythia8::Event& event );
  bool canVetoPartonLevel() override { return true; }
  bool doVetoPartonLevel(const Pythia8::Event& event) override;

  void setEventNumber(int ievt) {
    fEventNumber = ievt;
    return;
  }

  virtual void init(lhef::LHERunInfo* runInfo);
  bool initAfterBeams() override {
    if (fIsInitialized)
      return true;
    fJetMatching->initAfterBeams();
    fIsInitialized = true;
    return true;
  }
  void resetMatchingStatus() {
    fJetMatching->resetMatchingStatus();
    return;
  }
  virtual void beforeHadronization(lhef::LHEEvent* lhee);

protected:
  JetMatchingHook() : UserHooks() {}

  void setLHERunInfo(lhef::LHERunInfo* lheri) {
    fRunBlock = lheri;
    if (fRunBlock == nullptr)
      return;
    const lhef::HEPRUP* heprup = fRunBlock->getHEPRUP();
    lhef::CommonBlocks::fillHEPRUP(heprup);
    return;
  }
  void setLHEEvent(lhef::LHEEvent* lhee) {
    fEventBlock = lhee;
    if (fEventBlock == nullptr)
      return;
    const lhef::HEPEUP* hepeup = fEventBlock->getHEPEUP();
    lhef::CommonBlocks::fillHEPEUP(hepeup);
    return;
  }

  // private:

  lhef::LHERunInfo* fRunBlock;
  lhef::LHEEvent* fEventBlock;
  int fEventNumber;

  Pythia8::Info* fInfoPtr;

  gen::JetMatching* fJetMatching;
  Py8toJetInput* fJetInputFill;

  //void setJetAlgoInput( const Pythia8::Event& );
  //int getAncestor( int, const Pythia8::Event& );

  bool fIsInitialized;
};

#endif  // gen_JetMatchingHook_h
