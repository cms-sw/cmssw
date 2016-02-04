#ifndef EcalIndexingTools_h
#define EcalIndexingTools_h

#include <iostream>
/* ******************************************
 * Alessandro Palma 19/03/2008
 ********************************************/

// $Id: EcalIndexingTools.h,v 1.1 2008/04/28 16:59:17 palmale Exp $

class EcalIndexingTools
{
  
 private:
  
  EcalIndexingTools():totNumberOfBins_(-1), nBinsEt_(-1),  nBinsEta_(-1), maxEta_(-1.), maxEt_(-1.), minEta_(-1.), minEt_(-1.){}; 
  
  static EcalIndexingTools *instance_;
  
  int totNumberOfBins_, nBinsEt_, nBinsEta_;
  
  double maxEta_, maxEt_, minEta_, minEt_;

 public:
  
  ~EcalIndexingTools() {};

  static EcalIndexingTools* getInstance () {
    if (instance_ == 0 ){
      instance_ = new EcalIndexingTools();

      std::cout<< "[EcalIndexingTools* getInstance ()] new EcalIndexingTools created "<< std::endl;
     
    }
    return instance_;
  }
  
  
  int getNumberOfChannels(){return totNumberOfBins_;};
  
  double getEtaMax(){return maxEta_;};
  
  int getProgressiveIndex( double , double );
  
  void setBinRange( int, double, double, int, double, double);
  

  
};


#endif
