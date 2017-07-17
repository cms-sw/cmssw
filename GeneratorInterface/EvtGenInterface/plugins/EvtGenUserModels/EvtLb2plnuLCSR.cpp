

//
// Environment:
//      This software is part of the EvtGen package developed jointly
//      for the BaBar and CLEO collaborations.  If you use all or part
//      of it, please give an appropriate acknowledgement.
//
// Copyright Information:
//      Copyright (C) 1998      Caltech, UCSB
//
// Module: EvtLb2plnuLCSR.cc
//
// Description: Routine to implement Lb->p l nu semileptonic decays using form factor
//              predictions form Light Cone Sum Rules.  The form factors are from 
//              A. Khodjamirian, C. Klein, T. Mannel and Y.-M. Wang, arXiv.1108.2971 (2011)
//              
//
// Modification history:
//
//   William Sutcliffe     27/07/2013        Module created
//
//
//------------------------------------------------------------------------

#include "EvtGenBase/EvtPatches.hh"
#include <stdlib.h>
#include "EvtGenBase/EvtParticle.hh"
#include "EvtGenBase/EvtGenKine.hh"
#include "EvtGenBase/EvtPDL.hh"
#include "EvtGenBase/EvtReport.hh"
#include "GeneratorInterface/EvtGenInterface/interface/EvtGenUserModels/EvtLb2plnuLCSR.hh"
#include "EvtGenBase/EvtConst.hh"
#include "EvtGenBase/EvtIdSet.hh"
#include <string>
#include "GeneratorInterface/EvtGenInterface/interface/EvtGenUserModels/EvtLb2plnuLCSRFF.hh"

using namespace std;
#ifdef D0
#undef D0
#endif
EvtLb2plnuLCSR::EvtLb2plnuLCSR():
  ffmodel(0)
  ,calcamp(0)
{}


EvtLb2plnuLCSR::~EvtLb2plnuLCSR() {
  delete ffmodel;
  ffmodel=0;
  delete calcamp;
  calcamp=0;
}

std::string EvtLb2plnuLCSR::getName(){
  
  return "Lb2plnuLCSR";     
  
}



EvtDecayBase* EvtLb2plnuLCSR::clone(){
  
  return new EvtLb2plnuLCSR;
  
}

void EvtLb2plnuLCSR::decay( EvtParticle *p ){
  
  //This is a kludge to avoid warnings because the K_2* mass becomes to large.
  static EvtIdSet regenerateMasses("K_2*+","K_2*-","K_2*0","anti-K_2*0",
				   "K_1+","K_1-","K_10","anti-K_10",
				   "D'_1+","D'_1-","D'_10","anti-D'_10");
  
  if (regenerateMasses.contains(getDaug(0))){
    p->resetFirstOrNot();
  }
  
  p->initializePhaseSpace(getNDaug(),getDaugs());
  
  EvtComplex r00(getArg(0), 0.0 );
  EvtComplex r01(getArg(1), 0.0 );
  EvtComplex r10(getArg(2), 0.0 );
  EvtComplex r11(getArg(3), 0.0 );

  calcamp->CalcAmp(p,_amp2,ffmodel, r00, r01, r10, r11);
  
}

void EvtLb2plnuLCSR::initProbMax() {


  static EvtId LAMB=EvtPDL::getId("Lambda_b0");
  static EvtId LAMBB=EvtPDL::getId("anti-Lambda_b0");
  static EvtId PRO=EvtPDL::getId("p+");
  static EvtId PROB=EvtPDL::getId("anti-p-");
  
  EvtId parnum,barnum;
  
  parnum = getParentId();
  barnum = getDaug(0);

  if( (parnum==LAMB && barnum==PRO) || (parnum==LAMBB && barnum==PROB) ) 
  {
      setProbMax(22000.0);
  }
  else
  {
    EvtGenReport(EVTGEN_ERROR,"EvtGen") << "Decay does not have Lb->p setting ProbMax = 0 "
			   <<endl;
    setProbMax(0.0);
  }
  
}

void EvtLb2plnuLCSR::init(){
  
  if (getNArg()!=4) {

    EvtGenReport(EVTGEN_ERROR,"EvtGen") << "EvtLb2plnuLCSR generator expected "
			   << " 4 arguments but found:"<<getNArg()<<endl;
    EvtGenReport(EVTGEN_ERROR,"EvtGen") << "Will terminate execution!"<<endl;
    ::abort();

  }

  if ( getNDaug()!=3 ) {
     EvtGenReport(EVTGEN_ERROR,"EvtGen") 
       << "Wrong number of daughters in EvtLb2plnu.cc " 
       << " 3 daughters expected but found: "<<getNDaug()<<endl;
     EvtGenReport(EVTGEN_ERROR,"EvtGen") << "Will terminate execution!"<<endl;
     ::abort();
  }


  //We expect the parent to be a dirac particle 
  //and the daughters to be X lepton neutrino

  EvtSpinType::spintype parenttype=EvtPDL::getSpinType(getParentId());
  EvtSpinType::spintype baryontype=EvtPDL::getSpinType(getDaug(0));
  EvtSpinType::spintype leptontype=EvtPDL::getSpinType(getDaug(1));
  EvtSpinType::spintype neutrinotype=EvtPDL::getSpinType(getDaug(2));

  if ( parenttype != EvtSpinType::DIRAC ) {
    EvtGenReport(EVTGEN_ERROR,"EvtGen") << "EvtLb2plnuLCSR generator expected "
                           << " a DIRAC parent, found:"<<
                           EvtPDL::name(getParentId())<<endl;
    EvtGenReport(EVTGEN_ERROR,"EvtGen") << "Will terminate execution!"<<endl;
    ::abort();
  }
  if ( leptontype != EvtSpinType::DIRAC ) {
    EvtGenReport(EVTGEN_ERROR,"EvtGen") << "EvtLb2plnuLCSR generator expected "
                           << " a DIRAC 2nd daughter, found:"<<
                           EvtPDL::name(getDaug(1))<<endl;
    EvtGenReport(EVTGEN_ERROR,"EvtGen") << "Will terminate execution!"<<endl;
    ::abort();
  }
  if ( neutrinotype != EvtSpinType::NEUTRINO ) {
    EvtGenReport(EVTGEN_ERROR,"EvtGen") << "EvtLb2plnuLCSR generator expected "
                           << " a NEUTRINO 3rd daughter, found:"<<
                           EvtPDL::name(getDaug(2))<<endl;
    EvtGenReport(EVTGEN_ERROR,"EvtGen") << "Will terminate execution!"<<endl;
    ::abort();
  }

  //set ffmodel
  ffmodel = new EvtLb2plnuLCSRFF;

  if ( baryontype==EvtSpinType::DIRAC) 
  { 
    calcamp = new EvtSLBaryonAmp; 
  }
  else {
    EvtGenReport(EVTGEN_ERROR,"EvtGen") 
      << "Wrong baryon spin type in EvtLb2plnuLCSR.cc " 
      << "Expected spin type " << EvtSpinType::DIRAC 
      << ", found spin type " << baryontype <<endl;
    EvtGenReport(EVTGEN_ERROR,"EvtGen") << "Will terminate execution!" <<endl;
     ::abort();
  }
  
}

