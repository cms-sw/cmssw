#include "Calibration/Tools/interface/EcalIndexingTools.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

EcalIndexingTools* EcalIndexingTools::instance_ = nullptr;
 

void EcalIndexingTools::setBinRange(int nBinsEta, double minEta, double maxEta, int nBinsEt, double minEt, double maxEt){
  
  //  std::cout<< "[EcalIndexingTools::setBinRange] setting values "<< nBinsEta << " " << minEta << " " << maxEta << std::endl; 

  totNumberOfBins_ = nBinsEta*nBinsEt;
  
  nBinsEt_ = nBinsEt;
  nBinsEta_ = nBinsEta;
  
  minEt_ = minEt;
  minEta_ = minEta;
  maxEt_ = maxEt;
  maxEta_ = maxEta;
  
  //  std::cout<< "[EcalIndexingTools::setBinRange] read back values: "<< nBinsEta_ << " " << minEta_ << " " << maxEta_ << std::endl; 
  
  return;
  
}


int EcalIndexingTools::getProgressiveIndex(double myEta, double myEt){
  

  std::cout << "minEt_ " << minEt_ <<std::endl;
  std::cout << "minEta_ " << minEta_ <<std::endl;
  std::cout << "maxEt_ " << maxEt_ <<std::endl;
  std::cout << "maxEta_ " << maxEta_ <<std::endl;
  
  ///INITIALIZE BOUNDARIES
  
  double BoundaryEt[100] = {-99.};
  double BoundaryEta[100] = {-99.};
  
  for( int i = 0; i < ( nBinsEt_ + 1 ); i++ ){
    
    BoundaryEt[i] = minEt_ +  i * ( (maxEt_ - minEt_)/ (double)nBinsEt_ );
    //    std::cout << "i " << i << " BoundaryEt[i] "<< BoundaryEt[i] <<std::endl;

  }
  
  for( int i = 0; i < ( nBinsEta_ + 1 ); i++ ){
    
    BoundaryEta[i] = minEta_ +  i * ( (maxEta_ - minEta_)/ (double)nBinsEta_ );
    //std::cout << "i " << i << " BoundaryEta[i] "<< BoundaryEta[i] <<std::endl;
  
  }
  
  ////FIND ETA BIN AND ET BIN, SEPARATELY
  int etBin(-1);
  int etaBin(-1);
  
  for( int i = 0; i <  nBinsEta_ ; i++ ){
    if( myEta > BoundaryEta[i] && 
	myEta <= BoundaryEta[i+1] )
      etaBin=i;
  }
  
  for( int i = 0; i <  nBinsEt_ ; i++ ){
    if( myEt > BoundaryEt[i] && 
	myEt <= BoundaryEt[i+1] )
      etBin=i;
  }

  // std::cout << " myEta "<< myEta << " myEt "<< myEt << " etaBin "<< etaBin << " etBin "<< etBin << std::endl;
  
  /////////////FIND UNIQUE PROGRESSIVE INDEX
  
  int in =  etaBin * nBinsEta_ + etBin;
  
  //std::cout << "Progressive index " << in << std::endl;

  return in;
  
}

