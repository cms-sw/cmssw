/* 
 *  \class TMem
 *
 *  $Date: 2008/04/28 15:04:33 $
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
  
  pair<int,int> mempair=ME::memFromLmr(lmr);
  if(n==0) return mempair.first;
  else return mempair.second;
  
}
