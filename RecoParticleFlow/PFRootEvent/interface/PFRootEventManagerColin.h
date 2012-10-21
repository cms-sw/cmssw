#ifndef Demo_PFRootEvent_PFRootEventManagerColin_h
#define Demo_PFRootEvent_PFRootEventManagerColin_h

#include "RecoParticleFlow/PFRootEvent/interface/PFRootEventManager.h"

class NeutralEvent {
 public:
  NeutralEvent() {
    reset();
  }

  virtual ~NeutralEvent() {};

  virtual void reset() {
    eECAL = -1;
    eHCAL = -1;
    eNeutral = -1;
    etaECAL = -10;
    etaHCAL = -10;
    etaNeutral = -10;
    nECAL = -1;
    nHCAL = -1;
    rCode = 0;
  }

  double eECAL;         
  double eHCAL;     
  double eNeutral;
  double etaECAL;         
  double etaHCAL;     
  double etaNeutral;
  int    nECAL;
  int    nHCAL;
  int    rCode;
};


class TauEvent : public NeutralEvent {
 public:
  TauEvent() : 
    NeutralEvent() {
    reset();
  }

  virtual ~TauEvent() {};

  void reset() {
    NeutralEvent::reset();
    pTrack = -1;
    ptTrack = -1;
    etaTrack = -10;
    pHadron = -1;
    eEcalHadron = -1;
    chi2ECAL = -1;
  }
  
  
  double pTrack;
  double ptTrack;
  double etaTrack;
  double pHadron;
  double eEcalHadron;
  double chi2ECAL;
};


class PFRootEventManagerColin : public PFRootEventManager {
  
 public:
  PFRootEventManagerColin(const char* file);

  ~PFRootEventManagerColin();

  void readSpecificOptions(const char* file);

  bool processEntry(int entry);
  bool processNeutral(); 
  bool processHIGH_E_TAUS();
 
  
  void write();


  enum Mode {
    Neutral=0,
    HIGH_E_TAUS=1
  };

  TTree          *outTreeMy_;
  NeutralEvent   *neutralEvent_;
  TauEvent       *tauEvent_;
  int             mode_;
};

#endif
