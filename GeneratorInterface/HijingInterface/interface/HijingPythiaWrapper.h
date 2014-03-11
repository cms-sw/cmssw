#ifndef gen_HijingPythiaWrapper_h
#define gen_HijingPythiaWrapper_h

#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Service.h"
#include "HepMC/PythiaWrapper6_4.h"

#include "CLHEP/Random/RandomEngine.h"

 extern "C"
 {
    void   py1ent_(int& ip, int& kf, double& pe, double& the, double& phi);
    double pymass_(int& );
    void   pyexec_();
    int    pycomp_(int& );
    void   pyglfr_();
    void   pyglrhad_();
    void   pystlfr_();
    void   pystrhad_();
    void   pygive_(const char*, int );
    void   pydecy_( int& ip ) ;
    void   pyrobo_( int&, int&, double&, double&, double&, double&, double& );
    
    void   txgive_(const char*, int );
    void   txgive_init_(void);
    
    static bool call_pygive(const std::string &line)
    {
       int numWarn = pydat1.mstu[26];    // # warnings
       int numErr = pydat1.mstu[22];     // # errors
       
       pygive_(line.c_str(), line.length());
       
       return pydat1.mstu[26] == numWarn &&
	  pydat1.mstu[22] == numErr;
    }
 }
 
#define PYCOMP pycomp_
extern "C" {
   int PYCOMP(int& length);
}

#define LUGIVE pygive_
extern "C" {
   void LUGIVE(const char*,int length);
}

/*
extern "C" {
   double ran_(int*){
      return gen::pyr_(0);
   }
}
*/

float ranff_(unsigned int *iseed)
{
   (*iseed) = (69069 * (*iseed) + 1) & 0xffffffffUL;
   return (*iseed) / 4294967296.0;
}


CLHEP::HepRandomEngine* hijRandomEngine;

extern "C"
{
   float gen::hijran_(int *idummy)
   {
      return hijRandomEngine->flat();
   }
}




extern "C" {
   float ran_(unsigned int* iseed){
      return hijRandomEngine->flat();
      //      return ranff_(iseed);
      //      return gen::pyr_(0);
   }
}

extern "C" {
   float rlu_(unsigned int* iseed){
      return hijRandomEngine->flat();
      //      return ranff_(iseed);
      //      return gen::pyr_(0);
   }
}






/*

#include "CLHEP/Random/RandomEngine.h"
extern CLHEP::HepRandomEngine* randomEngine;
extern "C" {
   double pyr_(int* idummy);
}

CLHEP::HepRandomEngine* randomEngine;

double pyr_(int *idummy)
{
   // getInstance will throw if no one used enter/leave
   // or this is the wrong caller class, like e.g. Herwig6Instance
   return randomEngine->flat(); 
}

*/

#endif
