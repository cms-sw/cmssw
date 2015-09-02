//--------------------------------------------------------------------------
//
// Environment:
//      This software is part of the EvtGen package developed jointly
//      for the BaBar and CLEO collaborations.  If you use all or part
//      of it, please give an appropriate acknowledgement.
//
// Copyright Information: See EvtGen/COPYRIGHT
//      Copyright (C) 1998      Caltech, UCSB
//
// Module: EvtSLBaryonAmp.cc
//
// Description: Routine to implement semileptonic decays of Dirac baryons
//
// Modification history:
//
//    R.J. Tesarek     May 28, 2004     Module created
//    Karen Gibson     1/20/2006        Module updated for 1/2+->1/2+,
//                                      1/2+->1/2-, 1/2+->3/2- Lambda decays
//
//--------------------------------------------------------------------------

#include "EvtGenBase/EvtPatches.hh"
#include "EvtGenBase/EvtParticle.hh"
#include "EvtGenBase/EvtGenKine.hh"
#include "EvtGenBase/EvtPDL.hh"
#include "EvtGenBase/EvtReport.hh"
#include "EvtGenBase/EvtTensor4C.hh"
#include "EvtGenBase/EvtVector4C.hh"
#include "EvtGenBase/EvtDiracSpinor.hh"
#include "EvtGenBase/EvtDiracParticle.hh"
#include "EvtGenBase/EvtRaritaSchwinger.hh"
#include "GeneratorInterface/EvtGenInterface/interface/EvtGenUserModels/EvtSLBaryonAmp.hh"
#include "EvtGenBase/EvtId.hh"
#include "EvtGenBase/EvtAmp.hh"
#include "EvtGenBase/EvtSemiLeptonicFF.hh"
#include "EvtGenBase/EvtGammaMatrix.hh"

#include <stdlib.h>

using std::endl;


EvtSLBaryonAmp::~EvtSLBaryonAmp(){
}


void EvtSLBaryonAmp::CalcAmp( EvtParticle *parent,
					EvtAmp& amp,
					EvtSemiLeptonicFF *FormFactors ) {

  static EvtId EM=EvtPDL::getId("e-");
  static EvtId MUM=EvtPDL::getId("mu-");
  static EvtId TAUM=EvtPDL::getId("tau-");
  static EvtId EP=EvtPDL::getId("e+");
  static EvtId MUP=EvtPDL::getId("mu+");
  static EvtId TAUP=EvtPDL::getId("tau+");

 
  //Add the lepton and neutrino 4 momenta to find q2

  EvtVector4R q = parent->getDaug(1)->getP4() 
                    + parent->getDaug(2)->getP4(); 
  double q2 = (q.mass2());

  double f1v,f1a,f2v,f2a;
  double m_meson = parent->getDaug(0)->mass();

  FormFactors->getbaryonff(parent->getId(),
			   parent->getDaug(0)->getId(),
                           q2,
                           m_meson,
                           &f1v, 
                           &f1a, 
                           &f2v, 
                           &f2a);

  EvtVector4R p4b;
  p4b.set(parent->mass(),0.0,0.0,0.0);
  
  EvtVector4C temp_00_term1;
  EvtVector4C temp_00_term2;
  
  EvtVector4C temp_01_term1;
  EvtVector4C temp_01_term2;
  
  EvtVector4C temp_10_term1;
  EvtVector4C temp_10_term2;
  
  EvtVector4C temp_11_term1;
  EvtVector4C temp_11_term2;
  
  EvtDiracSpinor p0=parent->sp(0);
  EvtDiracSpinor p1=parent->sp(1);
  
  EvtDiracSpinor d0=parent->getDaug(0)->spParent(0);
  EvtDiracSpinor d1=parent->getDaug(0)->spParent(1);
  
  temp_00_term1.set(0,f1v*(d0*(EvtGammaMatrix::g0()*p0)));
  temp_00_term2.set(0,f1a*(d0*((EvtGammaMatrix::g0()*EvtGammaMatrix::g5())*p0)));
  temp_01_term1.set(0,f1v*(d0*(EvtGammaMatrix::g0()*p1)));
  temp_01_term2.set(0,f1a*(d0*((EvtGammaMatrix::g0()*EvtGammaMatrix::g5())*p1)));
  temp_10_term1.set(0,f1v*(d1*(EvtGammaMatrix::g0()*p0)));
  temp_10_term2.set(0,f1a*(d1*((EvtGammaMatrix::g0()*EvtGammaMatrix::g5())*p0)));
  temp_11_term1.set(0,f1v*(d1*(EvtGammaMatrix::g0()*p1)));
  temp_11_term2.set(0,f1a*(d1*((EvtGammaMatrix::g0()*EvtGammaMatrix::g5())*p1)));
  
  temp_00_term1.set(1,f1v*(d0*(EvtGammaMatrix::g1()*p0)));
  temp_00_term2.set(1,f1a*(d0*((EvtGammaMatrix::g1()*EvtGammaMatrix::g5())*p0)));
  temp_01_term1.set(1,f1v*(d0*(EvtGammaMatrix::g1()*p1)));
  temp_01_term2.set(1,f1a*(d0*((EvtGammaMatrix::g1()*EvtGammaMatrix::g5())*p1)));
  temp_10_term1.set(1,f1v*(d1*(EvtGammaMatrix::g1()*p0)));
  temp_10_term2.set(1,f1a*(d1*((EvtGammaMatrix::g1()*EvtGammaMatrix::g5())*p0)));
  temp_11_term1.set(1,f1v*(d1*(EvtGammaMatrix::g1()*p1)));
  temp_11_term2.set(1,f1a*(d1*((EvtGammaMatrix::g1()*EvtGammaMatrix::g5())*p1)));
  
  temp_00_term1.set(2,f1v*(d0*(EvtGammaMatrix::g2()*p0)));
  temp_00_term2.set(2,f1a*(d0*((EvtGammaMatrix::g2()*EvtGammaMatrix::g5())*p0)));
  temp_01_term1.set(2,f1v*(d0*(EvtGammaMatrix::g2()*p1)));
  temp_01_term2.set(2,f1a*(d0*((EvtGammaMatrix::g2()*EvtGammaMatrix::g5())*p1)));
  temp_10_term1.set(2,f1v*(d1*(EvtGammaMatrix::g2()*p0)));
  temp_10_term2.set(2,f1a*(d1*((EvtGammaMatrix::g2()*EvtGammaMatrix::g5())*p0)));
  temp_11_term1.set(2,f1v*(d1*(EvtGammaMatrix::g2()*p1)));
  temp_11_term2.set(2,f1a*(d1*((EvtGammaMatrix::g2()*EvtGammaMatrix::g5())*p1)));
  
  temp_00_term1.set(3,f1v*(d0*(EvtGammaMatrix::g3()*p0)));
  temp_00_term2.set(3,f1a*(d0*((EvtGammaMatrix::g3()*EvtGammaMatrix::g5())*p0)));
  temp_01_term1.set(3,f1v*(d0*(EvtGammaMatrix::g3()*p1)));
  temp_01_term2.set(3,f1a*(d0*((EvtGammaMatrix::g3()*EvtGammaMatrix::g5())*p1)));
  temp_10_term1.set(3,f1v*(d1*(EvtGammaMatrix::g3()*p0)));
  temp_10_term2.set(3,f1a*(d1*((EvtGammaMatrix::g3()*EvtGammaMatrix::g5())*p0)));
  temp_11_term1.set(3,f1v*(d1*(EvtGammaMatrix::g3()*p1)));
  temp_11_term2.set(3,f1a*(d1*((EvtGammaMatrix::g3()*EvtGammaMatrix::g5())*p1)));
  


  EvtVector4C l1,l2;

  EvtId l_num = parent->getDaug(1)->getId();
  if (l_num==EM||l_num==MUM||l_num==TAUM){

    l1=EvtLeptonVACurrent(parent->getDaug(1)->spParent(0),
                          parent->getDaug(2)->spParentNeutrino());
    l2=EvtLeptonVACurrent(parent->getDaug(1)->spParent(1),
                          parent->getDaug(2)->spParentNeutrino());
  }
  else{
    if (l_num==EP||l_num==MUP||l_num==TAUP){
    l1=EvtLeptonVACurrent(parent->getDaug(2)->spParentNeutrino(),
			    parent->getDaug(1)->spParent(0));
    l2=EvtLeptonVACurrent(parent->getDaug(2)->spParentNeutrino(),
			    parent->getDaug(1)->spParent(1));
    }
    else{
      report(ERROR,"EvtGen") << "Wrong lepton number"<<endl;
    }
  }

  amp.vertex(0,0,0,l1.cont(temp_00_term1+temp_00_term2));
  amp.vertex(0,0,1,l2.cont(temp_00_term1+temp_00_term2));

  amp.vertex(0,1,0,l1.cont(temp_01_term1+temp_01_term2));
  amp.vertex(0,1,1,l2.cont(temp_01_term1+temp_01_term2));

  amp.vertex(1,0,0,l1.cont(temp_10_term1+temp_10_term2));
  amp.vertex(1,0,1,l2.cont(temp_10_term1+temp_10_term2));

  amp.vertex(1,1,0,l1.cont(temp_11_term1+temp_11_term2));
  amp.vertex(1,1,1,l2.cont(temp_11_term1+temp_11_term2));

  return;
}



double EvtSLBaryonAmp::CalcMaxProb( EvtId parent, EvtId baryon, 
					      EvtId lepton, EvtId nudaug,
					      EvtSemiLeptonicFF *FormFactors,
					      EvtComplex r00, EvtComplex r01, 
					      EvtComplex r10, EvtComplex r11) {

  //This routine takes the arguements parent, baryon, and lepton
  //number, and a form factor model, and returns a maximum
  //probability for this semileptonic form factor model.  A
  //brute force method is used.  The 2D cos theta lepton and
  //q2 phase space is probed.

  //Start by declaring a particle at rest.

  //It only makes sense to have a scalar parent.  For now. 
  //This should be generalized later.

  //  EvtScalarParticle *scalar_part;
  //  scalar_part=new EvtScalarParticle;

  EvtDiracParticle *dirac_part;
  EvtParticle *root_part;
  dirac_part=new EvtDiracParticle;

  //cludge to avoid generating random numbers!
  //  scalar_part->noLifeTime();
  dirac_part->noLifeTime();

  EvtVector4R p_init;
  
  p_init.set(EvtPDL::getMass(parent),0.0,0.0,0.0);
  //  scalar_part->init(parent,p_init);
  //  root_part=(EvtParticle *)scalar_part;
  //  root_part->set_type(EvtSpinType::SCALAR);

  dirac_part->init(parent,p_init);
  root_part=(EvtParticle *)dirac_part;
  root_part->setDiagonalSpinDensity();      

  EvtParticle *daughter, *lep, *trino;
  
  EvtAmp amp;

  EvtId listdaug[3];
  listdaug[0] = baryon;
  listdaug[1] = lepton;
  listdaug[2] = nudaug;

  amp.init(parent,3,listdaug);

  root_part->makeDaughters(3,listdaug);
  daughter=root_part->getDaug(0);
  lep=root_part->getDaug(1);
  trino=root_part->getDaug(2);

  //cludge to avoid generating random numbers!
  daughter->noLifeTime();
  lep->noLifeTime();
  trino->noLifeTime();


  //Initial particle is unpolarized, well it is a scalar so it is 
  //trivial
  EvtSpinDensity rho;
  rho.setDiag(root_part->getSpinStates());
  
  double mass[3];
  
  double m = root_part->mass();
  
  EvtVector4R p4baryon, p4lepton, p4nu, p4w;
  double q2max;

  double q2, elepton, plepton;
  int i,j;
  double erho,prho,costl;

  double maxfoundprob = 0.0;
  double prob = -10.0;
  int massiter;

  for (massiter=0;massiter<3;massiter++){

    mass[0] = EvtPDL::getMass(baryon);
    mass[1] = EvtPDL::getMass(lepton);
    mass[2] = EvtPDL::getMass(nudaug);
    if ( massiter==1 ) {
      mass[0] = EvtPDL::getMinMass(baryon);
    }
    if ( massiter==2 ) {
      mass[0] = EvtPDL::getMaxMass(baryon);
    }

    q2max = (m-mass[0])*(m-mass[0]);
    
    //loop over q2

    for (i=0;i<25;i++) {
      q2 = ((i+0.5)*q2max)/25.0;
      
      erho = ( m*m + mass[0]*mass[0] - q2 )/(2.0*m);
      
      prho = sqrt(erho*erho-mass[0]*mass[0]);
      
      p4baryon.set(erho,0.0,0.0,-1.0*prho);
      p4w.set(m-erho,0.0,0.0,prho);
      
      //This is in the W rest frame
      elepton = (q2+mass[1]*mass[1])/(2.0*sqrt(q2));
      plepton = sqrt(elepton*elepton-mass[1]*mass[1]);
      
      double probctl[3];

      for (j=0;j<3;j++) {
	
        costl = 0.99*(j - 1.0);
	
	//These are in the W rest frame. Need to boost out into
	//the B frame.
        p4lepton.set(elepton,0.0,
		  plepton*sqrt(1.0-costl*costl),plepton*costl);
        p4nu.set(plepton,0.0,
		 -1.0*plepton*sqrt(1.0-costl*costl),-1.0*plepton*costl);

	EvtVector4R boost((m-erho),0.0,0.0,1.0*prho);
        p4lepton=boostTo(p4lepton,boost);
        p4nu=boostTo(p4nu,boost);

	//Now initialize the daughters...

        daughter->init(baryon,p4baryon);
        lep->init(lepton,p4lepton);
        trino->init(nudaug,p4nu);

        CalcAmp(root_part,amp,FormFactors,r00,r01,r10,r11);

	//Now find the probability at this q2 and cos theta lepton point
        //and compare to maxfoundprob.

	//Do a little magic to get the probability!!
	prob = rho.normalizedProb(amp.getSpinDensity());

	probctl[j]=prob;
      }

      //probclt contains prob at ctl=-1,0,1.
      //prob=a+b*ctl+c*ctl^2

      double a=probctl[1];
      double b=0.5*(probctl[2]-probctl[0]);
      double c=0.5*(probctl[2]+probctl[0])-probctl[1];

      prob=probctl[0];
      if (probctl[1]>prob) prob=probctl[1];
      if (probctl[2]>prob) prob=probctl[2];

      if (fabs(c)>1e-20){
	double ctlx=-0.5*b/c;
	if (fabs(ctlx)<1.0){
	  double probtmp=a+b*ctlx+c*ctlx*ctlx;
	  if (probtmp>prob) prob=probtmp;
	} 

      }

      //report(DEBUG,"EvtGen") << "prob,probctl:"<<prob<<" "
      //			    << probctl[0]<<" "
      //			    << probctl[1]<<" "
      //			    << probctl[2]<<std::endl;

      if ( prob > maxfoundprob ) {
	maxfoundprob = prob; 
      }

    }
    if ( EvtPDL::getWidth(baryon) <= 0.0 ) {
      //if the particle is narrow dont bother with changing the mass.
      massiter = 4;
    }

  }
  root_part->deleteTree();  

  maxfoundprob *=1.1;
  return maxfoundprob;
  
}
void EvtSLBaryonAmp::CalcAmp(EvtParticle *parent,
				       EvtAmp& amp,
				       EvtSemiLeptonicFF *FormFactors,
				       EvtComplex r00, EvtComplex r01, 
				       EvtComplex r10, EvtComplex r11) {
  //  Leptons
  static EvtId EM=EvtPDL::getId("e-");
  static EvtId MUM=EvtPDL::getId("mu-");
  static EvtId TAUM=EvtPDL::getId("tau-");
  //  Anti-Leptons
  static EvtId EP=EvtPDL::getId("e+");
  static EvtId MUP=EvtPDL::getId("mu+");
  static EvtId TAUP=EvtPDL::getId("tau+");

  //  Baryons
  static EvtId LAMCP=EvtPDL::getId("Lambda_c+");
  static EvtId LAMC1P=EvtPDL::getId("Lambda_c(2593)+");
  static EvtId LAMC2P=EvtPDL::getId("Lambda_c(2625)+");
  static EvtId LAMB=EvtPDL::getId("Lambda_b0");
  static EvtId PRO=EvtPDL::getId("p+");
  static EvtId N1440=EvtPDL::getId("N(1440)+");
  static EvtId N1520=EvtPDL::getId("N(1520)+");
  static EvtId N1535=EvtPDL::getId("N(1535)+");
  static EvtId N1720=EvtPDL::getId("N(1720)+");
  static EvtId N1650=EvtPDL::getId("N(1650)+");
  static EvtId N1700=EvtPDL::getId("N(1700)+");
  static EvtId N1710=EvtPDL::getId("N(1710)+");
  static EvtId N1875=EvtPDL::getId("N(1875)+");
  static EvtId N1900=EvtPDL::getId("N(1900)+");

  // Anti-Baryons
  static EvtId LAMCM=EvtPDL::getId("anti-Lambda_c-");
  static EvtId LAMC1M=EvtPDL::getId("anti-Lambda_c(2593)-");
  static EvtId LAMC2M=EvtPDL::getId("anti-Lambda_c(2625)-");
  static EvtId LAMBB=EvtPDL::getId("anti-Lambda_b0");
  static EvtId PROB=EvtPDL::getId("anti-p-");
  static EvtId N1440B=EvtPDL::getId("anti-N(1440)-");
  static EvtId N1520B=EvtPDL::getId("anti-N(1520)-");
  static EvtId N1535B=EvtPDL::getId("anti-N(1535)-");
  static EvtId N1720B=EvtPDL::getId("anti-N(1720)-");
  static EvtId N1650B=EvtPDL::getId("anti-N(1650)-");
  static EvtId N1700B=EvtPDL::getId("anti-N(1700)-");
  static EvtId N1710B=EvtPDL::getId("anti-N(1710)-");
  static EvtId N1875B=EvtPDL::getId("anti-N(1875)-");
  static EvtId N1900B=EvtPDL::getId("anti-N(1900)-");

  // Set the spin density matrix of the parent baryon
  EvtSpinDensity rho;
  rho.setDim(2);
  rho.set(0,0,r00);
  rho.set(0,1,r01);

  rho.set(1,0,r10);
  rho.set(1,1,r11);

  EvtVector4R vector4P = parent->getP4Lab();
  double pmag = vector4P.d3mag();
  double cosTheta = vector4P.get(3)/pmag;
  
  double theta = acos(cosTheta);
  double phi = atan2(vector4P.get(2), vector4P.get(1));
  
  parent->setSpinDensityForwardHelicityBasis(rho,phi,theta, 0.0);
  //parent->setSpinDensityForward(rho);

  // Set the four momentum of the parent baryon in it's rest frame
  EvtVector4R p4b;
  p4b.set(parent->mass(), 0.0,0.0,0.0);

  // Get the four momentum of the daughter baryon in the parent's rest frame
  EvtVector4R p4daught = parent->getDaug(0)->getP4();
  
  // Add the lepton and neutrino 4 momenta to find q (q^2)
  EvtVector4R q = parent->getDaug(1)->getP4() 
    + parent->getDaug(2)->getP4();
  
  double q2 = q.mass2();


  EvtId l_num = parent->getDaug(1)->getId();
  EvtId bar_num = parent->getDaug(0)->getId();
  EvtId par_num = parent->getId();
  
  double baryonmass = parent->getDaug(0)->mass();
  
  // Handle spin-1/2 daughter baryon Dirac spinor cases
  if( EvtPDL::getSpinType(parent->getDaug(0)->getId())==EvtSpinType::DIRAC ) {

    // Set the form factors
    double f1,f2,f3,g1,g2,g3;
    FormFactors->getdiracff( par_num,
			     bar_num,
			     q2,
			     baryonmass,
			     &f1, &f2, &f3,
			     &g1, &g2, &g3);
    
    const double form_fact[6] = {f1, f2, f3, g1, g2, g3};
    
    EvtVector4C b11, b12, b21, b22, l1, l2;

    //  Lepton Current
    if(l_num==EM || l_num==MUM || l_num==TAUM){

      l1=EvtLeptonVACurrent(parent->getDaug(1)->spParent(0),
			    parent->getDaug(2)->spParentNeutrino());
      l2=EvtLeptonVACurrent(parent->getDaug(1)->spParent(1),
			    parent->getDaug(2)->spParentNeutrino());
      
    } else if (l_num==EP || l_num==MUP || l_num==TAUP) {

      l1=EvtLeptonVACurrent(parent->getDaug(2)->spParentNeutrino(),
			    parent->getDaug(1)->spParent(0));
      l2=EvtLeptonVACurrent(parent->getDaug(2)->spParentNeutrino(),
			    parent->getDaug(1)->spParent(1));
      
    } else {
      report(ERROR,"EvtGen")<< "Wrong lepton number \n";
      ::abort();
    }

    // Baryon current

    // Flag for particle/anti-particle parent, daughter with same/opp. parity
    // pflag = 0 => particle, same parity parent, daughter
    // pflag = 1 => particle, opp. parity parent, daughter
    // pflag = 2 => anti-particle, same parity parent, daughter
    // pflag = 3 => anti-particle, opp. parity parent, daughter

    int pflag = 0;

    // Handle 1/2+ -> 1/2+ first
    if ( (par_num==LAMB && bar_num==LAMCP) 
	 || (par_num==LAMBB && bar_num==LAMCM)
         || (par_num==LAMB && bar_num==PRO )
	 || (par_num==LAMBB && bar_num==PROB)
         || (par_num==LAMB && bar_num==N1440 )
	 || (par_num==LAMBB && bar_num==N1440B)
         || (par_num==LAMB && bar_num==N1710 )
	 || (par_num==LAMBB && bar_num==N1710B)
	 
	 ) {

      // Set particle/anti-particle flag
      if (bar_num==LAMCP || bar_num==PRO || bar_num==N1440 || bar_num==N1710)
	pflag = 0;
      else if (bar_num==LAMCM || bar_num==PROB || bar_num==N1440B || bar_num==N1710B)
	pflag = 2;

      b11=EvtBaryonVACurrent(parent->getDaug(0)->spParent(0),
			     parent->sp(0),
			     p4b, p4daught, form_fact, pflag);
      b21=EvtBaryonVACurrent(parent->getDaug(0)->spParent(0),
			     parent->sp(1),
			     p4b, p4daught, form_fact, pflag);
      b12=EvtBaryonVACurrent(parent->getDaug(0)->spParent(1),
			     parent->sp(0),
			     p4b, p4daught, form_fact, pflag);
      b22=EvtBaryonVACurrent(parent->getDaug(0)->spParent(1),
			     parent->sp(1),
			     p4b, p4daught, form_fact, pflag);
    }

    // Handle 1/2+ -> 1/2- second
    else if( 
	    (par_num==LAMB && bar_num==LAMC1P) 
	     || (par_num==LAMBB && bar_num==LAMC1M)
	     || (par_num==LAMB && bar_num==N1535) 
	     || (par_num==LAMBB && bar_num==N1535B)
	     || (par_num==LAMB && bar_num==N1650) 
	     || (par_num==LAMBB && bar_num==N1650B)
	   ) {
      
      // Set particle/anti-particle flag
      if (bar_num==LAMC1P || bar_num == N1535 || bar_num == N1650)
	pflag = 1;
      else if (bar_num==LAMC1M || bar_num == N1535B || bar_num == N1650B)
	pflag = 3;

      b11=EvtBaryonVACurrent((parent->getDaug(0)->spParent(0)),
			     (EvtGammaMatrix::g5()*parent->sp(0)),
			     p4b, p4daught, form_fact, pflag);
      b21=EvtBaryonVACurrent((parent->getDaug(0)->spParent(0)),
			     (EvtGammaMatrix::g5()*parent->sp(1)),
			     p4b, p4daught, form_fact, pflag);
      b12=EvtBaryonVACurrent((parent->getDaug(0)->spParent(1)),
			     (EvtGammaMatrix::g5()*parent->sp(0)),
			     p4b, p4daught, form_fact, pflag);
      b22=EvtBaryonVACurrent((parent->getDaug(0)->spParent(1)),
			     (EvtGammaMatrix::g5()*parent->sp(1)),
			     p4b, p4daught, form_fact, pflag);
      
    }

    else {
      report(ERROR,"EvtGen") << "Dirac semilep. baryon current " 
			     << "not implemented for this decay sequence." 
			     << std::endl;
      ::abort();
    }
     
    amp.vertex(0,0,0,l1*b11);
    amp.vertex(0,0,1,l2*b11);
    
    amp.vertex(1,0,0,l1*b21);
    amp.vertex(1,0,1,l2*b21);
    
    amp.vertex(0,1,0,l1*b12);
    amp.vertex(0,1,1,l2*b12);
    
    amp.vertex(1,1,0,l1*b22);
    amp.vertex(1,1,1,l2*b22);

  }
  
  // Need special handling for the spin-3/2 daughter baryon 
  // Rarita-Schwinger spinor cases
  else if( EvtPDL::getSpinType(parent->getDaug(0)->getId())==EvtSpinType::RARITASCHWINGER ) {
    
    // Set the form factors
    double f1,f2,f3,f4,g1,g2,g3,g4;
    FormFactors->getraritaff( par_num,
			      bar_num,
			      q2,
			      baryonmass,
			      &f1, &f2, &f3, &f4,
			      &g1, &g2, &g3, &g4);
    
    const double form_fact[8] = {f1, f2, f3, f4, g1, g2, g3, g4};
    
    EvtId l_num = parent->getDaug(1)->getId();
    
    EvtVector4C b11, b12, b21, b22, b13, b23, b14, b24, l1, l2;

    //  Lepton Current
    if (l_num==EM || l_num==MUM || l_num==TAUM) {
      //  Lepton Current
      l1=EvtLeptonVACurrent(parent->getDaug(1)->spParent(0),
			    parent->getDaug(2)->spParentNeutrino());
      l2=EvtLeptonVACurrent(parent->getDaug(1)->spParent(1),
			    parent->getDaug(2)->spParentNeutrino());
    }
    else if (l_num==EP || l_num==MUP || l_num==TAUP) {
      l1=EvtLeptonVACurrent(parent->getDaug(2)->spParentNeutrino(),
			    parent->getDaug(1)->spParent(0));
      l2=EvtLeptonVACurrent(parent->getDaug(2)->spParentNeutrino(),
			    parent->getDaug(1)->spParent(1));
      
    } else {
      report(ERROR,"EvtGen")<< "Wrong lepton number \n";
    }
      
    //  Baryon Current
    // Declare particle, anti-particle flag, same/opp. parity
    // pflag = 0 => particle
    // pflag = 1 => anti-particle
    int pflag = 0;
    
    // Handle cases of 1/2+ -> 3/2- or 3/2+
    if ( (par_num==LAMB && bar_num==LAMC2P) 
	    ||(par_num==LAMB && bar_num==N1720 )
	    ||(par_num==LAMB && bar_num==N1520 )
	    ||(par_num==LAMB && bar_num==N1700 )
	    ||(par_num==LAMB && bar_num==N1875 )
	    ||(par_num==LAMB && bar_num==N1900 )
       ) {
      // Set flag for particle case
      pflag = 0;
    }
    else if (
	    (par_num==LAMBB && bar_num==LAMC2M)
	    ||(par_num==LAMBB && bar_num==N1520B )
	    ||(par_num==LAMBB && bar_num==N1700B )
	    ||(par_num==LAMBB && bar_num==N1875B )
	    )
    {
    // Set flag for anti-particle opposite parity case
      pflag = 1;
    }
    // Handle anti-particle case for 1/2+ -> 3/2+
    else if (
	   ( par_num==LAMBB && bar_num==N1720B)
	    || (par_num==LAMBB && bar_num==N1900B)
	    ) {
      pflag = 2;
    }
    else {
      report(ERROR,"EvtGen") << "Rarita-Schwinger semilep. baryon current " 
			     << "not implemented for this decay sequence." 
			     << std::endl;
      ::abort();
    }
     
    // Baryon current
    b11=EvtBaryonVARaritaCurrent(parent->getDaug(0)->spRSParent(0),
				 parent->sp(0),
				 p4b, p4daught, form_fact, pflag);
    b21=EvtBaryonVARaritaCurrent(parent->getDaug(0)->spRSParent(0),
				 parent->sp(1),
				 p4b, p4daught, form_fact, pflag);
    
    b12=EvtBaryonVARaritaCurrent(parent->getDaug(0)->spRSParent(1),
				 parent->sp(0),
				 p4b, p4daught, form_fact, pflag);
    b22=EvtBaryonVARaritaCurrent(parent->getDaug(0)->spRSParent(1),
				 parent->sp(1),
				 p4b, p4daught, form_fact, pflag);
    
    b13=EvtBaryonVARaritaCurrent(parent->getDaug(0)->spRSParent(2),
				 parent->sp(0),
				 p4b, p4daught, form_fact, pflag);
    b23=EvtBaryonVARaritaCurrent(parent->getDaug(0)->spRSParent(2),
				 parent->sp(1),
				 p4b, p4daught, form_fact, pflag);
    
    b14=EvtBaryonVARaritaCurrent(parent->getDaug(0)->spRSParent(3),
				 parent->sp(0),
				 p4b, p4daught, form_fact, pflag);
    b24=EvtBaryonVARaritaCurrent(parent->getDaug(0)->spRSParent(3),
				 parent->sp(1),
				 p4b, p4daught, form_fact, pflag);
    
    amp.vertex(0,0,0,l1*b11);
    amp.vertex(0,0,1,l2*b11);
    
    amp.vertex(1,0,0,l1*b21);
    amp.vertex(1,0,1,l2*b21);
    
    amp.vertex(0,1,0,l1*b12);
    amp.vertex(0,1,1,l2*b12);
    
    amp.vertex(1,1,0,l1*b22);
    amp.vertex(1,1,1,l2*b22);
    
    amp.vertex(0,2,0,l1*b13);
    amp.vertex(0,2,1,l2*b13);
    
    amp.vertex(1,2,0,l1*b23);
    amp.vertex(1,2,1,l2*b23);

    amp.vertex(0,3,0,l1*b14);
    amp.vertex(0,3,1,l2*b14);
    
    amp.vertex(1,3,0,l1*b24);
    amp.vertex(1,3,1,l2*b24);

  }

}
  

EvtVector4C EvtSLBaryonAmp::EvtBaryonVACurrent( const EvtDiracSpinor& Bf,
							  const EvtDiracSpinor& Bi, 
							  EvtVector4R parent, 
							  EvtVector4R daught, 
							  const double *ff,
							  int pflag) {

  // flag == 0 => particle
  // flag == 1 => particle, opposite parity 
  // flag == 2 => anti-particle, same parity 
  // flag == 3 => anti-particle, opposite parity 

  // particle
  EvtComplex cv = EvtComplex(1.0, 0.);
  EvtComplex ca = EvtComplex(1.0, 0.);

  EvtComplex cg0 = EvtComplex(1.0, 0.);
  EvtComplex cg5 = EvtComplex(1.0, 0.);

  // antiparticle- same parity parent & daughter
  if( pflag == 2 ) {
    cv = EvtComplex(-1.0, 0.);
    ca = EvtComplex(1.0, 0.);

    cg0 =  EvtComplex(1.0, 0.0);
    // Changed cg5 from -i to -1 as appears to fix particle - anti-particle discrepency
    cg5 =  EvtComplex(-1.0,0.0 );
  }
  // antiparticle- opposite parity parent & daughter
  else if( pflag == 3) {
    cv = EvtComplex(1.0, 0.);
    ca = EvtComplex(-1.0, 0.);

    // Changed cg0 from -i to -1 as appears to fix particle - anti-particle discrepency
    cg0 =  EvtComplex(-1.0, 0.0);
    cg5 =  EvtComplex(1.0, 0.0);
  }

  EvtVector4C t[6];


  

  // Term 1 = \bar{u}(p',s')*(F_1(q^2)*\gamma_{mu})*u(p,s)
  t[0] = cv*EvtLeptonVCurrent( Bf, Bi);

  // Term 2 = \bar{u}(p',s')*(F_2(q^2)*(p_{mu}/m_{\Lambda_Q}))*u(p,s)
  t[1] = cg0*EvtLeptonSCurrent( Bf, Bi ) * (parent/parent.mass());

  // Term 3 = \bar{u}(p',s')*(F_3(q^2)*(p'_{mu}/m_{\Lambda_q}))*u(p,s)
  t[2] = cg0*EvtLeptonSCurrent( Bf, Bi ) * (daught/daught.mass());

  // Term 4 = \bar{u}(p',s')*(G_1(q^2)*\gamma_{mu}*\gamma_5)*u(p,s)
  t[3] = ca*EvtLeptonACurrent( Bf, Bi);

  // Term 5 =  \bar{u}(p',s')*(G_2(q^2)*(p_{mu}/m_{\Lambda_Q})*\gamma_5)*u(p,s)
  t[4] = cg5*EvtLeptonPCurrent( Bf, Bi ) * (parent/parent.mass());

  // Term 6 = \bar{u}(p',s')*(G_3(q^2)*(p'_{mu}/m_{\Lambda_q})*\gamma_5)*u(p,s)
  t[5] = cg5*EvtLeptonPCurrent( Bf, Bi ) * (daught/daught.mass());

  // Sum the individual terms
  EvtVector4C current = (ff[0]*t[0] + ff[1]*t[1] + ff[2]*t[2]
			 - ff[3]*t[3] - ff[4]*t[4] - ff[5]*t[5]);
  
  return current;
}

EvtVector4C EvtSLBaryonAmp::EvtBaryonVARaritaCurrent( const EvtRaritaSchwinger& Bf,
								const EvtDiracSpinor& Bi, 
								EvtVector4R parent, 
								EvtVector4R daught, 
								const double *ff,
								int pflag) {

  // flag == 0 => particle
  // flag == 1 => anti-particle

  // particle
  EvtComplex cv = EvtComplex(1.0, 0.);
  EvtComplex ca = EvtComplex(1.0, 0.);

  EvtComplex cg0 = EvtComplex(1.0, 0.);
  EvtComplex cg5 = EvtComplex(1.0, 0.);

  // antiparticle opposite parity
  if( pflag == 1 ) {
    cv = EvtComplex(-1.0, 0.);
    ca = EvtComplex(1.0, 0.);
 
    cg0 =  EvtComplex(1.0, 0.0);
    cg5 =  EvtComplex(-1.0, 0.0);
 }
  // antiparticle same parity
  else if( pflag == 2) {
    cv = EvtComplex(1.0, 0.);
    ca = EvtComplex(-1.0, 0.);

    cg0 =  EvtComplex(-1.0, 0.0);
    cg5 =  EvtComplex(1.0, 0.0);
  }

  EvtVector4C t[8];
  EvtTensor4C id;
  id.setdiag(1.0,1.0,1.0,1.0);

  EvtDiracSpinor tmp;
  for(int i=0;i<4;i++){
    tmp.set_spinor(i,Bf.getVector(i)*parent);
  }

  EvtVector4C v1,v2;
  for(int i=0;i<4;i++){
    v1.set(i,EvtLeptonSCurrent(Bf.getSpinor(i),Bi));
    v2.set(i,EvtLeptonPCurrent(Bf.getSpinor(i),Bi));
  }

  // Term 1 = \bar{u}^{\alpha}(p',s')*(p_{\alpha}/m_{\Lambda_Q})*(F_1(q^2)*\gamma_{mu})*u(p,s)
  t[0] = (cv/parent.mass()) * EvtLeptonVCurrent(tmp, Bi);

  // Term 2 
  // = \bar{u}^{\alpha}(p',s')*(p_{\alpha}/m_{\Lambda_Q})*(F_2(q^2)*(p_{mu}/m_{\Lambda_Q}))*u(p,s)
  t[1] = ((cg0/parent.mass()) * EvtLeptonSCurrent(tmp, Bi)) * (parent/parent.mass());

  // Term 3 
  // = \bar{u}^{\alpha}(p',s')*(p_{\alpha}/m_{\Lambda_Q})*(F_3(q^2)*(p'_{mu}/m_{\Lambda_q}))*u(p,s)
  t[2] = ((cg0/parent.mass()) * EvtLeptonSCurrent(tmp, Bi)) * (daught/daught.mass());

  // Term 4 = \bar{u}^{\alpha}(p',s')*(F_4(q^2)*g_{\alpha,\mu})*u(p,s)
  t[3] = cg0*(id.cont2(v1));

  // Term 5 
  // = \bar{u}^{\alpha}(p',s')*(p_{\alpha}/m_{\Lambda_Q})*(G_1(q^2)*\gamma_{mu}*\gamma_5)*u(p,s)
  t[4] = (ca/parent.mass()) * EvtLeptonACurrent(tmp, Bi);

  // Term 6 
  // = \bar{u}^{\alpha}(p',s')*(p_{\alpha}/m_{\Lambda_Q})
  //      *(G_2(q^2)*(p_{mu}/m_{\Lambda_Q})*\gamma_5)*u(p,s)
  t[5] = ((cg5/parent.mass()) * EvtLeptonPCurrent(tmp, Bi)) * (parent/parent.mass());

  // Term 7 
  // = \bar{u}^{\alpha}(p',s')*(p_{\alpha}/m_{\Lambda_Q})
  //      *(G_3(q^2)*(p'_{mu}/m_{\Lambda_q})*\gamma_5)*u(p,s)
  t[6] = ((cg5/parent.mass()) * EvtLeptonPCurrent(tmp, Bi)) * (daught/daught.mass());

  // Term 8 = \bar{u}^{\alpha}(p',s')*(G_4(q^2)*g_{\alpha,\mu}*\gamma_5))*u(p,s)
  t[7] = cg5*(id.cont2(v2));

  // Sum the individual terms
  EvtVector4C current = (ff[0]*t[0] + ff[1]*t[1] + ff[2]*t[2] + ff[3]*t[3]
			 - ff[4]*t[4] - ff[5]*t[5] - ff[6]*t[6] - ff[7]*t[7]);
  
  return current;
}
