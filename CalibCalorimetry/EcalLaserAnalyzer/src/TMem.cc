/* 
 *  \class TMem
 *
 *  $Date: 2012/02/09 10:08:10 $
 *  \author: Julie Malcles - CEA/Saclay
 */

#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TMem.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/ME.h>

#include <TMath.h>
#include <iostream>
using namespace std;

//ClassImp(TMem)


// Default Constructor...
TMem::TMem()
{
  init(610);
}

// Constructor...
TMem::TMem( int fedid )
{
  init(fedid);
}

// Destructor
TMem::~TMem()
{
}

void TMem::init(int fedid) 
{
  _fedid=fedid;
  _memFromDcc=ME::memFromDcc(_fedid);
}

bool TMem::isMemRelevant(int mem){

  bool isMemOK=false;
  for (unsigned int imem=0;imem<_memFromDcc.size();imem++){
    if(mem == _memFromDcc[imem]) {
      isMemOK=true;
      imem=_memFromDcc.size();
    }
  }
  return isMemOK;
}

int TMem::Mem(int lmr, int n){
  
  std::pair<int,int> mempair=ME::memFromLmr(lmr);
  if(n==0) return mempair.first;
  else return mempair.second;
  
}
