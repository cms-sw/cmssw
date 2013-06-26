#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/ExprAlgo/interface/AlgoImpl.h"
#include "DetectorDescription/ExprAlgo/interface/AlgoPos.h"
#include <cstdio>

AlgoImpl::AlgoImpl( AlgoPos * al, std::string label )
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

AlgoImpl::~AlgoImpl( void )
{ }

int
AlgoImpl::copyno( void ) const
{
  return incr_ ? curr_ : count_ ;
}  

void
AlgoImpl::terminate( void )
{
  terminate_ = true;
}

void
AlgoImpl::checkTermination( void )
{
  terminate();
}  
  
std::string
AlgoImpl::d2s( double x )
{
  char buffer [25]; 
  int len = snprintf( buffer, 25, "%g", x );
  if( len >= 25 )
    edm::LogError( "DoubleToString" ) << "Length truncated (from " << len << ")";
  return std::string( buffer );
}
      
