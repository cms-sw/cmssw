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
// Module: EvtPHOTOS.cc
//
// Description: This routine takes the particle *p and applies
//              the PHOTOS package to generate final state radiation
//              on the produced mesons.
//
// Modification history:
//
//    RYD     October 1, 1997        Module created
//
//------------------------------------------------------------------------
// 
#include "EvtGenBase/EvtPatches.hh"
#include "EvtGenBase/EvtParticle.hh"
#include "EvtGenBase/EvtPhotonParticle.hh"
#include "EvtGenBase/EvtPDL.hh"
#include "EvtGenModels/EvtPHOTOS.hh"

#ifdef WIN32
extern "C" {
  void __stdcall BEGEVTGENSTOREX(int *,int *,int *);
  /*                              int *,int *,int *,int *,
                                double *,double *,double *, 
                                double *,double *,double *, 
                                double *,double *,double *);
  void __stdcall BEGEVTGENGETX(int *,int *,int *,int *,
			      int *,int *,int *,int *,
			      double *,double *,double *, 
			      double *,double *,double *, 
			      double *,double *,double *);
  void __stdcall HEPLST(int *);
  void __stdcall PHOTOS(int *);
  void __stdcall PHOINI( ) ;*/
}
#else
extern "C" void begevtgenstorex_(int *,int *,int *);
/*                                int *,int *,int *,int *,
                                double *,double *,double *, 
                                double *,double *,double *, 
                                double *,double *,double *);

extern "C" void begevtgengetx_(int *,int *,int *,int *,
			      int *,int *,int *,int *,
			      double *,double *,double *, 
			      double *,double *,double *, 
			      double *,double *,double *); */

// extern "C" void heplst_(int *);

extern "C" void myphotos_(int *,int *, int *, int [], int [],
			  int [][2], int [][2], double [][5], 
                          double [][5], int *, double [][5]);

extern "C" void phoini_();
#endif

void EvtPHOTOS::doRadCorr( EvtParticle *p){

  static int first=1;

  //added by Lange Jan4,2000
  static EvtId GAMM=EvtPDL::getId("gamma");

  if (first) {

    first=0;
#ifdef WIN32
    PHOINI() ;
#else
    phoini_();
    std::cout << "New PHOTOS implementation is ON" << std::endl; 
#endif
  }

  int entry;
  int thenevhep;
  int thenhep, thenhepout;
  int theisthep[4000];
  int theidhep[4000];
  int thejmohep[4000][2];
  int thejdahep[4000][2];
  double thephep[4000][5];
  double thephepout[4000][5];
  double thevhep[4000][5];

  /* eventnum,numparticle,istat,partnum,mother;
     int daugfirst,dauglast; */

  double px,py,pz,e;

  int numparticle, numparticlephotos;

  thephep[0][0]=0.0;
  thephep[0][1]=0.0;
  thephep[0][2]=0.0;
  thephep[0][3]=p->mass();
  thephep[0][4]=p->mass();
  thevhep[0][0]=0.0;
  thevhep[0][1]=0.0;
  thevhep[0][2]=0.0;
  thevhep[0][3]=0.0;
  
  entry=1;
  thenevhep=1;
  numparticle=1;
  theisthep[0]=2;
  theidhep[0]=EvtPDL::getStdHep(p->getId());
  thejmohep[0][0]=0;
  thejmohep[0][1]=0;
  thejdahep[0][0]=2;
  thejdahep[0][1]=1+p->getNDaug();

#ifdef WIN32
  BEGEVTGENSTOREX(&entry,&thejdahep[0][0],&thejdahep[0][1]);
#else
  begevtgenstorex_(&entry,&thejdahep[0][0],&thejdahep[0][1]);
#endif

  int i;

  for(i=0;i<p->getNDaug();i++){

    thephep[numparticle][0]=p->getDaug(i)->getP4().get(1);
    thephep[numparticle][1]=p->getDaug(i)->getP4().get(2);
    thephep[numparticle][2]=p->getDaug(i)->getP4().get(3);
    thephep[numparticle][3]=p->getDaug(i)->getP4().get(0);
    thephep[numparticle][4]=p->getDaug(i)->mass();
    thevhep[numparticle][0]=0.0;
    thevhep[numparticle][1]=0.0;
    thevhep[numparticle][2]=0.0;
    thevhep[numparticle][3]=0.0;
    
    entry+=1;
    thenevhep=1;
    theisthep[numparticle]=1;
    theidhep[numparticle]=EvtPDL::getStdHep(p->getDaug(i)->getId());
    thejmohep[numparticle][0]=1;
    thejmohep[numparticle][1]=0;
    thejdahep[numparticle][0]=0;
    thejdahep[numparticle][1]=0;

#ifdef WIN32
    BEGEVTGENSTOREX(&entry,&thejdahep[numparticle][0],&thejdahep[numparticle][1]);
#else
    begevtgenstorex_(&entry,&thejdahep[numparticle][0],&thejdahep[numparticle][1]);
#endif
    numparticle+=1;   
  }
  
  thenhep = numparticle;

  //can't use heplst since the common block used by the BaBar
  //implementation of PHOTOS  is renamed due to real*4 vs real*8
  //problems.

  //int mlst=1;

  //heplst_(&mlst);

  entry=1;
  // std::cout << "New PHOTOS implementation is ON" << std::endl; 

  //  report(INFO,"EvtGen") << "Doing photos " << 
  // EvtPDL::name(p->getId()) << std::endl;
#ifdef WIN32
  MYPHOTOS(&entry, &thenevhep, &thenhep, theisthep, theidhep,
	   thejmohep, thejdahep, thephep, thevhep, &thenhepout, thephepout );
#else
  myphotos_(&entry, &thenevhep, &thenhep, theisthep, theidhep,
	    thejmohep, thejdahep, thephep, thevhep, &thenhepout, thephepout);
#endif
  //  report(INFO,"EvtGen") << "done\n";
  /*
#ifdef WIN32
  BEGEVTGENGETX(&entry,&eventnum,&numparticlephotos,&istat,&partnum,
		    &mother,&daugfirst,&dauglast,
		    &px,&py,&pz,&e,&m,&x,&y,&z,&t);
#else
  begevtgengetx_(&entry,&eventnum,&numparticlephotos,&istat,&partnum,
		    &mother,&daugfirst,&dauglast,
		    &px,&py,&pz,&e,&m,&x,&y,&z,&t);
#endif    

  //report(INFO,"EvtGen") << "numparticlephotos:"<<numparticlephotos
  // <<std::endl;
  
  */
  numparticlephotos = thenhepout;

  if (numparticle==numparticlephotos) return;

  // std::cout << "**********" << std::endl;
  // std::cout << "Particles after: " << numparticlephotos << std::endl;
  // std::cout << "Particles before: " << numparticle << std::endl;
  // std::cout << "N. Generated photons: " << numparticlephotos-numparticle << std::endl;
  EvtVector4R new4mom;

  // int np;
  // std::cout << "Other particles: " << std::endl;
  for(i=0;i<p->getNDaug();i++){

    entry=i+1;
    /* if (abs(theidhep[0]) == 511 && numparticlephotos-numparticle == 2) {
      std::cout << "Particle " << entry << ": " << theidhep[entry] << " " << thephepout[entry][0] << " " << thephepout[entry][1]  << " " << thephepout[entry][2] << std::endl;
      } */
    /* #ifdef WIN32
    BEGEVTGENGETX(&entry,&eventnum,&np,&istat,&partnum,
		    &mother,&daugfirst,&dauglast,
		    &px,&py,&pz,&e,&m,&x,&y,&z,&t);
#else
    begevtgengetx_(&entry,&eventnum,&np,&istat,&partnum,
		    &mother,&daugfirst,&dauglast,
		    &px,&py,&pz,&e,&m,&x,&y,&z,&t);
#endif
    */

    px = thephepout[entry][0];
    py = thephepout[entry][1];
    pz = thephepout[entry][2];

    //this is needed to ensure that photos does not
    //change the masses. But it will violate energy conservation!
    double mp=p->getDaug(i)->mass();
    e=sqrt(mp*mp+px*px+py*py+pz*pz);
        
    new4mom.set(e,px,py,pz);

    p->getDaug(i)->setP4(new4mom);

  }

  for(entry=numparticle;entry<numparticlephotos;entry++){

    /* #ifdef WIN32
    BEGEVTGENGETX(&entry,&eventnum,&np,&istat,&partnum,
		    &mother,&daugfirst,&dauglast,
		    &px,&py,&pz,&e,&m,&x,&y,&z,&t);
#else
    begevtgengetx_(&entry,&eventnum,&np,&istat,&partnum,
		    &mother,&daugfirst,&dauglast,
		    &px,&py,&pz,&e,&m,&x,&y,&z,&t);
		    #endif */
        
    px = thephepout[entry][0];
    py = thephepout[entry][1];
    pz = thephepout[entry][2];
    e  = thephepout[entry][3];
    new4mom.set(e,px,py,pz);

    /* if (abs(theidhep[0]) == 511 && numparticlephotos-numparticle==2) {
      std::cout << "Particle " << entry << ": " << theidhep[entry] << " " << thephepout[entry][0] << " " << thephepout[entry][1]  << " " << thephepout[entry][2] << std::endl;
      } */
    //new4mom.dump();

    EvtPhotonParticle* gamma;
    gamma=new EvtPhotonParticle;
    gamma->init(GAMM,new4mom);
    //    report(INFO,"EvtGen") << gamma << " " << p << " "<< px << " " 
    //<< py << " " << pz << " " << p->getNDaug() << " " << 
    //EvtPDL::name(p->getId())<<" " << entry << " " <<numparticlephotos
    //<<std::endl;
    gamma->addDaug(p);

//    p->getDaug(i)->set_type(EvtSpinType::PHOTON);

  }
  
  return ;
}

