// PulseFit.h
//
// Class which computes 
//
// last change : $Date: 2012/02/09 10:07:33 $
// by          : $Author: eulisse $
//

#ifndef PulseFit_H
#define PulseFit_H


#include "TObject.h"
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/Shape.h>

class PulseFit: public TObject
{
 public:
  // Default Constructor, mainly for Root
  PulseFit() ;

  // Destructor: Does nothing?
  virtual ~PulseFit() ;

  // Get reconstructed values
  double getAmplitude() const ;
  double getTime() const ;
  double getPedestal() const ;
  double getChi2() const ;
  
  //! return the cristal number (supermodule convention [0-1699])
  int getSmCrystalNb() const ;
  
  //! set the cristal number (supermodule convention [0-1699])
  void setSmCrystalNb(const int & crystalNb) ;

 protected:
  double amplitude_ ;    /// amplitude of the pulse
  double time_ ;         /// position (in clock unit) of the maximum of the pulse
  double pedestal_ ;     /// remaining pedestal 
  double chi2_ ; 	 /// chi2 of the fit
  int smCrystalNb_ ;     /// cristal number in the supermodule
  
  //H4Analysis * h4ana_ ;  //!< pointer to current analysis
  //H4Shape * shape_ ;     //!< pointer to current shape

  // ClassDef(PulseFit,1)     //!< The processed part of the class is persistant
} ;

#endif
