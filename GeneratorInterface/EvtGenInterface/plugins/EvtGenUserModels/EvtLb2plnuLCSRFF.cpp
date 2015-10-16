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
// Module: EvtLb2plnuLCSRFF.cc
//
// Description: Routine to implement Lb->p l nu form factors
//              according to predictions from LCSR
//
// Modification history:
//
//   William Sutcliffe     27/07/2013        Module created
//                                      
//
//--------------------------------------------------------------------------
#include "EvtGenBase/EvtPatches.hh"
#include "EvtGenBase/EvtReport.hh"
#include "GeneratorInterface/EvtGenInterface/interface/EvtGenUserModels/EvtLb2plnuLCSRFF.hh"
#include "EvtGenBase/EvtPDL.hh"
#include "EvtGenBase/EvtId.hh"
#include "EvtGenBase/EvtIdSet.hh"
#include "EvtGenBase/EvtConst.hh"
#include <string>
#include <math.h>
#include <stdlib.h>
using std::endl;

void EvtLb2plnuLCSRFF::getdiracff(EvtId parent, EvtId daught,
				double q2, double /* mass */ , 
				double *f1, double *f2, double *f3, 
				double *g1, double *g2, double *g3 ) {


  // Define Event IDs for Lb and p 
  static EvtId LAMB=EvtPDL::getId("Lambda_b0");
  static EvtId LAMBB=EvtPDL::getId("anti-Lambda_b0");
  static EvtId PRO=EvtPDL::getId("p+");
  static EvtId PROB=EvtPDL::getId("anti-p-");

  if( (parent==LAMB && daught==PRO) 
       || (parent==LAMBB && daught==PROB) ) 
  {
      // Calculate Lb->p form factors based on LCSR predictions
      // Predictions taken from A. Khodjamirian, C. Klein, T. Mannel and Y.-M. Wang, arXiv.1108.2971 (2011)

      double MLamB = EvtPDL::getMass(parent);
      double MPro = EvtPDL::getMass(daught);

      double tplus = (MLamB + MPro) * (MLamB + MPro);      
      double tminus = (MLamB - MPro) * (MLamB - MPro);      
      double t0 = tplus - sqrt(tplus - tminus) * sqrt(tplus + 6);
      double z = (sqrt(tplus - q2) - sqrt(tplus - t0))/(sqrt(tplus - q2) + sqrt(tplus - t0));
      double z0 = (sqrt(tplus) - sqrt(tplus - t0))/(sqrt(tplus) + sqrt(tplus - t0));
      
      // FF parameters
      double f10 = 0.14;
      double bf1 = -1.49;
      double f20 = -0.054;
      double bf2 = -14.0;
      double g10 = 0.14;
      double bg1 = -4.05;
      double g20 = -0.028;
      double bg2 = -20.2;

      //FF paramterisation
      double F1 = (f10 / ( 1.0 - q2/ (5.325 * 5.325)))*(1.0 + bf1 * (z - z0 ) ); 
      double F2 = (f20 / ( 1.0 - q2/ (5.325 * 5.325)))*(1.0 + bf2 * (z - z0 ) );
      double G1 = (g10 / ( 1.0 - q2/ (5.723 * 5.723)))*(1.0 + bg1 * (z - z0 ) ); 
      double G2 = (g20 / ( 1.0 - q2/ (5.723 * 5.723)))*(1.0 + bg2 * (z - z0 ) );

      *f1  = F1 - (MLamB + MPro)*F2/MLamB;
      *f2  = F2;
      *f3  = MPro*(F2)/MLamB;
      *g1  =  G1 - (MLamB - MPro)*G2/MLamB;
      *g2  = -G2;
      *g3  = -MPro*G2/MLamB;

  }
  else 
  {
  EvtGenReport(EVTGEN_ERROR,"EvtGen") << "Only Lb -> p transitions allowed in EvtLb2plnuLCSRFF.\n";  
  ::abort();
  }

  return ;
}


void EvtLb2plnuLCSRFF::getraritaff( EvtId , EvtId ,
				  double , double , 
				  double* , double* , double* , double*, 
				  double* , double* , double* , double*  ) {

  EvtGenReport(EVTGEN_ERROR,"EvtGen") << "Not implemented :getraritaff in EvtLb2plnuLCSRFF.\n";  
  ::abort();

}

void EvtLb2plnuLCSRFF::getscalarff(EvtId, EvtId, double, double, double*, double*) {

  EvtGenReport(EVTGEN_ERROR,"EvtGen") << "Not implemented :getscalarff in EvtLb2plnuLCSRFF.\n";  
  ::abort();

}

void EvtLb2plnuLCSRFF::getvectorff(EvtId, EvtId, double, double, double*, double*,
				 double*, double*) {

  EvtGenReport(EVTGEN_ERROR,"EvtGen") << "Not implemented :getvectorff in EvtLb2plnuLCSRFF.\n";  
  ::abort();

}

void EvtLb2plnuLCSRFF::gettensorff(EvtId, EvtId, double, double, double*, double*,
				 double*, double*) {

  EvtGenReport(EVTGEN_ERROR,"EvtGen") << "Not implemented :gettensorff in EvtLb2plnuLCSRFF.\n";  
  ::abort();

}

void EvtLb2plnuLCSRFF::getbaryonff(EvtId, EvtId, double, double, double*, 
				 double*, double*, double*){
  
  EvtGenReport(EVTGEN_ERROR,"EvtGen") << "Not implemented :getbaryonff in EvtLb2plnuLCSRFF.\n";  
  ::abort();

}
