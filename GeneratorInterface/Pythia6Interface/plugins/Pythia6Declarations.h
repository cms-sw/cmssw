#ifndef gen_Pythia6Declarations_h
#define gen_Pythia6Declarations_h


#include "HepMC/PythiaWrapper6_2.h"

namespace gen
{

extern "C"
{
   void   py1ent_(int& ip, int& kf, double& pe, double& the, double& phi);
   double pymass_(int& kf);
   void   pyexec_();
   int    pycomp_(int& ip);
   void   pyglfr_();
   void   pyglrhad_();
   void   pystlfr_();
   void   pystrhad_();
   void   pygive_(const char *line, int length);
   
   void   txgive_(const char *line, int length);
   void   txgive_init_(void);
    
   static bool call_pygive(const std::string &line)
   {
      int numWarn = pydat1.mstu[26];	// # warnings
      int numErr = pydat1.mstu[22];	// # errors

      pygive_(line.c_str(), line.length());

      return pydat1.mstu[26] == numWarn &&
             pydat1.mstu[22] == numErr;
   }

}

}

#endif
