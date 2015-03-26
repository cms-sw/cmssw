//--------------------------------------------------------------------------
//
// Environment:
//      This software is part of the EvtGen package developed jointly
//      for the BaBar and CLEO collaborations.  If you use all or part
//      of it, please give an appropriate acknowledgement.
//
// Copyright Information:
//      Copyright (C) 1998      Caltech, UCSB
//
// Module: EvtGen/EvtLb2plnuLCSR.hh
//
// Description:Implementation of the Lb2plnuLCSR model
// Class to handle semileptonic Lb -> p l nu decays using the using form factor predictions from Light Cone Sum Rules.
// 
//
// Modification history:
//
//    William Sutcliffe     July 27, 2013     Module created
//
//------------------------------------------------------------------------

#ifndef EVTLB2PMUNULCSR_HH
#define EVTLB2PMUNULCSR_HH

#include "EvtGenBase/EvtDecayAmp.hh"
#include "EvtGenBase/EvtSemiLeptonicFF.hh"
#include "GeneratorInterface/EvtGenInterface/interface/EvtGenUserModels/EvtSLBaryonAmp.hh"

class EvtParticle;

class EvtLb2plnuLCSR:public  EvtDecayAmp  {

public:

  EvtLb2plnuLCSR();
  virtual ~EvtLb2plnuLCSR();

  std::string getName();
  EvtDecayBase* clone();

  void decay(EvtParticle *p);
  void initProbMax();
  void init();

private:
  EvtSemiLeptonicFF *ffmodel;
  EvtSLBaryonAmp *calcamp;
};

#endif

