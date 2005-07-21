#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/ExprAlgo/interface/AlgoImpl.h"
#include "DetectorDescription/ExprAlgo/interface/AlgoPos.h"

AlgoImpl::AlgoImpl(AlgoPos * al, std::string label)
 : ParS_(al->ParS_),
   ParE_(al->ParE_),
   start_(al->start_), end_(al->end_), incr_(al->incr_),
   curr_(al->curr_), count_(al->count_),
   terminate_(al->terminate_), err_(al->err_),
   label_(label)
{
 // DCOUT('E', "AlgoImpl ctor called with label=" << label << " AlgoPos.ddname=" << al->ddname() );
  al->registerAlgo(this);
} 


AlgoImpl::~AlgoImpl()
{ }


int AlgoImpl::copyno() const
{
  return incr_ ? curr_ : count_ ;
}  


void AlgoImpl::terminate()
{
  terminate_ = true;
}


void AlgoImpl::checkTermination()
{
  terminate();
}  
  
#include <cstdio>
std::string AlgoImpl::d2s(double x)
{
  char buffer [25]; 
  sprintf(buffer,"%g",x);
  return std::string(buffer);
}
      
