//to allow combining multiple user hooks

#include "Pythia8/UserHooks.h"
#include "Pythia8/StringFragmentation.h"

class MultiUserHook : public Pythia8::UserHooks {
public:
  // Constructor and destructor.
  MultiUserHook() {}

  unsigned int nHooks() const { return hooks_.size(); }
  void addHook(Pythia8::UserHooks *hook) { hooks_.push_back(hook); }

  // Initialisation after beams have been set by Pythia::init().
  bool initAfterBeams() override {
    bool test = true;
    for (Pythia8::UserHooks *hook : hooks_) {
      //      hook->initPtr(infoPtr,
      //                    settingsPtr,
      //                    particleDataPtr,
      //                    rndmPtr,
      //                    beamAPtr,
      //                    beamBPtr,
      //                    beamPomAPtr,
      //                    beamPomBPtr,
      //                    coupSMPtr,
      //                    partonSystemsPtr,
      //                    sigmaTotPtr);
      test &= hook->initAfterBeams();
    }

    //Certain types of hooks can only be handled by one UserHook at a time.
    //Check for this and return an error if needed
    int nCanVetoPT = 0;
    int nCanVetoStep = 0;
    int nCanVetoMPIStep = 0;
    int nCanSetResonanceScale = 0;
    int nCanEnhance = 0;
    for (Pythia8::UserHooks *hook : hooks_) {
      if (hook->canVetoPT())
        ++nCanVetoPT;
      if (hook->canVetoStep())
        ++nCanVetoStep;
      if (hook->canVetoMPIStep())
        ++nCanVetoMPIStep;
      if (hook->canSetResonanceScale())
        ++nCanSetResonanceScale;
      if (hook->canEnhanceEmission() || hook->canEnhanceTrial())
        ++nCanEnhance;
    }

    if (nCanVetoPT > 1) {
      infoPtr->errorMsg(
          "Error in MultiUserHook::initAfterBeams "
          "multiple UserHooks with canVetoPT() not allowed");
      test = false;
    }
    if (nCanVetoStep > 1) {
      infoPtr->errorMsg(
          "Error in MultiUserHook::initAfterBeams "
          "multiple UserHooks with canVetoStep() not allowed");
      test = false;
    }
    if (nCanVetoMPIStep > 1) {
      infoPtr->errorMsg(
          "Error in MultiUserHook::initAfterBeams "
          "multiple UserHooks with canVetoMPIStep() not allowed");
      test = false;
    }
    if (nCanSetResonanceScale > 1) {
      infoPtr->errorMsg(
          "Error in MultiUserHook::initAfterBeams "
          "multiple UserHooks with canSetResonanceScale() not allowed");
      test = false;
    }
    if (nCanEnhance > 1) {
      infoPtr->errorMsg(
          "Error in MultiUserHook::initAfterBeams "
          "multiple UserHooks with canEnhanceEmission() or canEnhanceTrial() not allowed");
      test = false;
    }

    return test;
  }

  // Possibility to modify cross section of process.
  bool canModifySigma() override {
    bool test = false;
    for (Pythia8::UserHooks *hook : hooks_) {
      test |= hook->canModifySigma();
    }
    return test;
  }

  // Multiplicative factor modifying the cross section of a hard process.
  double multiplySigmaBy(const Pythia8::SigmaProcess *sigmaProcessPtr,
                         const Pythia8::PhaseSpace *phaseSpacePtr,
                         bool inEvent) override {
    double factor = 1.;
    for (Pythia8::UserHooks *hook : hooks_) {
      if (hook->canModifySigma())
        factor *= hook->multiplySigmaBy(sigmaProcessPtr, phaseSpacePtr, inEvent);
    }
    return factor;
  }

  // Possibility to bias selection of events, compensated by a weight.
  bool canBiasSelection() override {
    bool test = false;
    for (Pythia8::UserHooks *hook : hooks_) {
      test |= hook->canBiasSelection();
    }
    return test;
  }

  // Multiplicative factor in the phase space selection of a hard process.
  double biasSelectionBy(const Pythia8::SigmaProcess *sigmaProcessPtr,
                         const Pythia8::PhaseSpace *phaseSpacePtr,
                         bool inEvent) override {
    double factor = 1.;
    for (Pythia8::UserHooks *hook : hooks_) {
      if (hook->canBiasSelection())
        factor *= hook->biasSelectionBy(sigmaProcessPtr, phaseSpacePtr, inEvent);
    }
    return factor;
  };

  // Event weight to compensate for selection weight above.
  double biasedSelectionWeight() override {
    double factor = 1.;
    for (Pythia8::UserHooks *hook : hooks_) {
      if (hook->canBiasSelection())
        factor *= hook->biasedSelectionWeight();
    }
    return factor;
  };

  // Possibility to veto event after process-level selection.
  bool canVetoProcessLevel() override {
    bool test = false;
    for (Pythia8::UserHooks *hook : hooks_) {
      test |= hook->canVetoProcessLevel();
    }
    return test;
  }

  // Decide whether to veto current process or not, based on process record.
  // Usage: doVetoProcessLevel( process).
  bool doVetoProcessLevel(Pythia8::Event &event) override {
    bool test = false;
    for (Pythia8::UserHooks *hook : hooks_) {
      if (hook->canVetoProcessLevel())
        test |= hook->doVetoProcessLevel(event);
    }
    return test;
  }

  // Possibility to veto resonance decay chain.
  bool canVetoResonanceDecays() override {
    bool test = false;
    for (Pythia8::UserHooks *hook : hooks_) {
      test |= hook->canVetoResonanceDecays();
    }
    return test;
  }

  // Decide whether to veto current resonance decay chain or not, based on
  // process record. Usage: doVetoProcessLevel( process).
  bool doVetoResonanceDecays(Pythia8::Event &event) override {
    bool test = false;
    for (Pythia8::UserHooks *hook : hooks_) {
      if (hook->canVetoResonanceDecays())
        test |= hook->doVetoResonanceDecays(event);
    }
    return test;
  }

  // Possibility to veto MPI + ISR + FSR evolution and kill event,
  // making decision at a fixed pT scale. Useful for MLM-style matching.
  bool canVetoPT() override {
    bool test = false;
    for (Pythia8::UserHooks *hook : hooks_) {
      test |= hook->canVetoPT();
    }
    return test;
  }

  // Transverse-momentum scale for veto test. (only one hook allowed)
  double scaleVetoPT() override {
    for (Pythia8::UserHooks *hook : hooks_) {
      if (hook->canVetoPT())
        return hook->scaleVetoPT();
    }
    return 0.;
  };

  // Decide whether to veto current event or not, based on event record.
  // Usage: doVetoPT( iPos, event), where iPos = 0: no emissions so far;
  // iPos = 1/2/3 joint evolution, latest step was MPI/ISR/FSR;
  // iPos = 4: FSR only afterwards; iPos = 5: FSR in resonance decay.
  bool doVetoPT(int iPos, const Pythia8::Event &event) override {
    for (Pythia8::UserHooks *hook : hooks_) {
      if (hook->canVetoPT())
        return hook->doVetoPT(iPos, event);
    }
    return false;
  }

  // Possibility to veto MPI + ISR + FSR evolution and kill event,
  // making decision after fixed number of ISR or FSR steps.
  bool canVetoStep() override {
    bool test = false;
    for (Pythia8::UserHooks *hook : hooks_) {
      test |= hook->canVetoStep();
    }
    return test;
  }

  // Up to how many ISR + FSR steps of hardest interaction should be checked.
  int numberVetoStep() override {
    for (Pythia8::UserHooks *hook : hooks_) {
      if (hook->canVetoStep())
        return hook->numberVetoStep();
    }
    return 1;
  };

  // Decide whether to veto current event or not, based on event record.
  // Usage: doVetoStep( iPos, nISR, nFSR, event), where iPos as above,
  // nISR and nFSR number of emissions so far for hard interaction only.
  bool doVetoStep(int iPos, int nISR, int nFSR, const Pythia8::Event &event) override {
    for (Pythia8::UserHooks *hook : hooks_) {
      if (hook->canVetoStep())
        return hook->doVetoStep(iPos, nISR, nFSR, event);
    }
    return false;
  }

  // Possibility to veto MPI + ISR + FSR evolution and kill event,
  // making decision after fixed number of MPI steps.
  bool canVetoMPIStep() override {
    bool test = false;
    for (Pythia8::UserHooks *hook : hooks_) {
      test |= hook->canVetoMPIStep();
    }
    return test;
  }

  // Up to how many MPI steps should be checked.
  int numberVetoMPIStep() override {
    for (Pythia8::UserHooks *hook : hooks_) {
      if (hook->canVetoMPIStep())
        return hook->numberVetoMPIStep();
    }
    return 1;
  };

  // Decide whether to veto current event or not, based on event record.
  // Usage: doVetoMPIStep( nMPI, event), where nMPI is number of MPI's so far.
  bool doVetoMPIStep(int nMPI, const Pythia8::Event &event) override {
    for (Pythia8::UserHooks *hook : hooks_) {
      if (hook->canVetoMPIStep())
        return hook->doVetoMPIStep(nMPI, event);
    }
    return false;
  }

  // Possibility to veto event after ISR + FSR + MPI in parton level,
  // but before beam remnants and resonance decays.
  bool canVetoPartonLevelEarly() override {
    bool test = false;
    for (Pythia8::UserHooks *hook : hooks_) {
      test |= hook->canVetoPartonLevelEarly();
    }
    return test;
  }

  // Decide whether to veto current partons or not, based on event record.
  // Usage: doVetoPartonLevelEarly( event).
  bool doVetoPartonLevelEarly(const Pythia8::Event &event) override {
    bool test = false;
    for (Pythia8::UserHooks *hook : hooks_) {
      if (hook->canVetoPartonLevelEarly())
        test |= hook->doVetoPartonLevelEarly(event);
    }
    return test;
  }

  // Retry same ProcessLevel with a new PartonLevel after a veto in
  // doVetoPT, doVetoStep, doVetoMPIStep or doVetoPartonLevelEarly
  // if you overload this method to return true.
  bool retryPartonLevel() override {
    bool test = false;
    for (Pythia8::UserHooks *hook : hooks_) {
      test |= hook->retryPartonLevel();
    }
    return test;
  }

  // Possibility to veto event after parton-level selection.
  bool canVetoPartonLevel() override {
    bool test = false;
    for (Pythia8::UserHooks *hook : hooks_) {
      test |= hook->canVetoPartonLevel();
    }
    return test;
  }

  // Decide whether to veto current partons or not, based on event record.
  // Usage: doVetoPartonLevel( event).
  bool doVetoPartonLevel(const Pythia8::Event &event) override {
    bool test = false;
    for (Pythia8::UserHooks *hook : hooks_) {
      if (hook->canVetoPartonLevel())
        test |= hook->doVetoPartonLevel(event);
    }
    return test;
  }

  // Possibility to set initial scale in TimeShower for resonance decay.
  bool canSetResonanceScale() override {
    bool test = false;
    for (Pythia8::UserHooks *hook : hooks_) {
      test |= hook->canSetResonanceScale();
    }
    return test;
  }

  // Initial scale for TimeShower evolution.
  // Usage: scaleResonance( iRes, event), where iRes is location
  // of decaying resonance in the event record.
  // Only one UserHook setting the resonance scale is allowed
  double scaleResonance(int iRes, const Pythia8::Event &event) override {
    for (Pythia8::UserHooks *hook : hooks_) {
      if (hook->canSetResonanceScale())
        return hook->scaleResonance(iRes, event);
    }
    return 0.;
  };

  // Possibility to veto an emission in the ISR machinery.
  bool canVetoISREmission() override {
    bool test = false;
    for (Pythia8::UserHooks *hook : hooks_) {
      test |= hook->canVetoISREmission();
    }
    return test;
  }

  // Decide whether to veto current emission or not, based on event record.
  // Usage: doVetoISREmission( sizeOld, event, iSys) where sizeOld is size
  // of event record before current emission-to-be-scrutinized was added,
  // and iSys is the system of the radiation (according to PartonSystems).
  bool doVetoISREmission(int sizeOld, const Pythia8::Event &event, int iSys) override {
    bool test = false;
    for (Pythia8::UserHooks *hook : hooks_) {
      if (hook->canVetoISREmission())
        test |= hook->doVetoISREmission(sizeOld, event, iSys);
    }
    return test;
  }

  // Possibility to veto an emission in the FSR machinery.
  bool canVetoFSREmission() override {
    bool test = false;
    for (Pythia8::UserHooks *hook : hooks_) {
      test |= hook->canVetoFSREmission();
    }
    return test;
  }

  // Decide whether to veto current emission or not, based on event record.
  // Usage: doVetoFSREmission( sizeOld, event, iSys, inResonance) where
  // sizeOld is size of event record before current emission-to-be-scrutinized
  // was added, iSys is the system of the radiation (according to
  // PartonSystems), and inResonance is true if the emission takes place in a
  // resonance decay.
  bool doVetoFSREmission(int sizeOld, const Pythia8::Event &event, int iSys, bool inResonance = false) override {
    bool test = false;
    for (Pythia8::UserHooks *hook : hooks_) {
      if (hook->canVetoFSREmission())
        test |= hook->doVetoFSREmission(sizeOld, event, iSys, inResonance);
    }
    return test;
  }

  // Possibility to veto an MPI.
  bool canVetoMPIEmission() override {
    bool test = false;
    for (Pythia8::UserHooks *hook : hooks_) {
      test |= hook->canVetoMPIEmission();
    }
    return test;
  }

  // Decide whether to veto an MPI based on event record.
  // Usage: doVetoMPIEmission( sizeOld, event) where sizeOld
  // is size of event record before the current MPI.
  bool doVetoMPIEmission(int sizeOld, const Pythia8::Event &event) override {
    bool test = false;
    for (Pythia8::UserHooks *hook : hooks_) {
      if (hook->canVetoMPIEmission())
        test |= hook->doVetoMPIEmission(sizeOld, event);
    }
    return test;
  }

  // Possibility to reconnect colours from resonance decay systems.
  bool canReconnectResonanceSystems() override {
    bool test = false;
    for (Pythia8::UserHooks *hook : hooks_) {
      test |= hook->canReconnectResonanceSystems();
    }
    return test;
  }

  // Do reconnect colours from resonance decay systems.
  // Usage: doVetoFSREmission( oldSizeEvt, event)
  // where oldSizeEvent is the event size before resonance decays.
  // Should normally return true, while false means serious failure.
  // Value of PartonLevel:earlyResDec determines where method is called.
  bool doReconnectResonanceSystems(int oldSizeEvt, Pythia8::Event &event) override {
    bool test = true;
    for (Pythia8::UserHooks *hook : hooks_) {
      if (hook->canReconnectResonanceSystems())
        test &= hook->doReconnectResonanceSystems(oldSizeEvt, event);
    }
    return test;
  }

  // Enhance emission rates (sec. 4 in EPJC (2013) 73).
  bool canEnhanceEmission() override {
    bool test = false;
    for (Pythia8::UserHooks *hook : hooks_) {
      test |= hook->canEnhanceEmission();
    }
    return test;
  }

  // Bookkeeping of weights for enhanced actual or trial emissions
  // (sec. 3 in EPJC (2013) 73).
  bool canEnhanceTrial() override {
    bool test = false;
    for (Pythia8::UserHooks *hook : hooks_) {
      test |= hook->canEnhanceTrial();
    }
    return test;
  }

  double enhanceFactor(std::string str) override {
    for (Pythia8::UserHooks *hook : hooks_) {
      if (hook->canEnhanceEmission() || hook->canEnhanceTrial())
        return hook->enhanceFactor(str);
    }
    return 1.;
  };

  double vetoProbability(std::string str) override {
    for (Pythia8::UserHooks *hook : hooks_) {
      if (hook->canEnhanceEmission() || hook->canEnhanceTrial())
        return hook->vetoProbability(str);
    }
    return 0.;
  };

  // Can change fragmentation parameters.
  bool canChangeFragPar() override {
    bool test = false;
    for (Pythia8::UserHooks *hook : hooks_) {
      test |= hook->canChangeFragPar();
    }
    return test;
  }

  // Do change fragmentation parameters.
  // Input: flavPtr, zPtr, pTPtr, idEnd, m2Had, iParton.
  bool doChangeFragPar(Pythia8::StringFlav *flavPtr,
                       Pythia8::StringZ *zPtr,
                       Pythia8::StringPT *pTPtr,
                       int idEnd,
                       double m2Had,
                       std::vector<int> iParton,
                       const Pythia8::StringEnd *SE) override {
    bool test = true;
    for (Pythia8::UserHooks *hook : hooks_) {
      if (hook->canChangeFragPar())
        test &= hook->doChangeFragPar(flavPtr, zPtr, pTPtr, idEnd, m2Had, iParton, SE);
    }
    return test;
  }

  // Do a veto on a hadron just before it is added to the final state.
  bool doVetoFragmentation(Pythia8::Particle part, const Pythia8::StringEnd *SE) override {
    bool test = false;
    for (Pythia8::UserHooks *hook : hooks_) {
      if (hook->canChangeFragPar())
        test |= hook->doVetoFragmentation(part, SE);
    }
    return test;
  }

  //--------------------------------------------------------------------------

private:
  std::vector<Pythia8::UserHooks *> hooks_;
};
