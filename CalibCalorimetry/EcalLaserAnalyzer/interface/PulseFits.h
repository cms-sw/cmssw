//-----------------------------------------------------------------------
//----------------------------------------------------------------------
// File PulseFits.h

#ifndef PulseFits_H
#define PulseFits_H

using namespace std;

class PulseFits
{
 public:
  // Default Constructor, mainly for Root
  PulseFits();

  // Destructor: Does nothing
  virtual ~PulseFits() ;

  // Compute amplitude of a channel

  virtual double doFit(double *){return 0;};
  

  virtual double getAmpl(){return 0;};
  virtual double getTime(){return 0;};

  // ClassDef(PulseFits,1)     //!< The processed part of the class is persistant
} ;

#endif



//-----------------------------------------------------------------------
//----------------------------------------------------------------------
