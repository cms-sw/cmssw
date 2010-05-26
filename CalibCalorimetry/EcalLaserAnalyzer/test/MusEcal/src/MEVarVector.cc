#include <cassert>
#include <math.h>
#include <iostream>
using namespace std;

#include "MEVarVector.hh"

ClassImp(MEVarVector)

MEVarVector::MEVarVector( int size ) : _size(  size )
{
}

MEVarVector::~MEVarVector() 
{
  for( MusEcal::VarVecTimeMap::iterator it=_map.begin(); it!=_map.end(); it++ )
    {
      delete it->second;
    }
}

void
MEVarVector::addTime( ME::Time t )
{
  if( _map.count( t ) !=0 ) return;
  MusEcal::VarVec* vec_ = 
    new MusEcal::VarVec( _size, make_pair<float,bool>( 0., true ) );
  _map[t] = vec_;
}

void
MEVarVector::setVal( ME::Time t, int ii, float val, bool check )
{
  if( check ) addTime( t );
  MusEcal::VarVec& vec_ = *(_map[t]);
  vec_[ii].first = val;
}

void
MEVarVector::setFlag( ME::Time t, int ii, bool flag, bool check )
{
  if( check ) addTime( t );
  MusEcal::VarVec& vec_ = *(_map[t]);
  vec_[ii].second = flag;
}

void
MEVarVector::setValAndFlag( ME::Time t, int ii, float val, bool flag, bool check )
{
  setVal( t, ii, val, check );
  setFlag( t, ii, flag, check );
}

void
MEVarVector::getTime( vector< ME::Time >& time, 
		      const METimeInterval* interval )
{
  time.clear();
  for( MusEcal::VarVecTimeMap::iterator it=_map.begin(); 
       it!=_map.end(); it++ )
    {
      ME::Time t_ = it->first;
      if( interval!=0 )
	{
	  if( t_<interval->firstTime() || t_>interval->lastTime() ) continue;
	}
      time.push_back( t_ );
    }
}

void
MEVarVector::getValAndFlag( int ii, 
			    const vector< ME::Time >& time, 
			    vector< float >& val,
			    vector< bool >& flag )
{
  val.clear();
  flag.clear();
  for( unsigned int itime=0; itime<time.size(); itime++ )
    {
      ME::Time t_ = time[itime];
      float val_(0.);
      bool flag_(true);
      assert( getValByTime( t_, ii, val_, flag_ ) );
      val.push_back( val_ );
      flag.push_back( flag_ );
    }
}

void
MEVarVector::getTimeValAndFlag( int ii, 
				vector< ME::Time >& time, 
				vector< float >& val,
				vector< bool >& flag,
				const METimeInterval* interval )
{
  getTime( time, interval );
  val.clear();
  flag.clear();
  getValAndFlag( ii, time, val, flag );
}

bool
MEVarVector::getValByTime(  ME::Time time, int ii, 
			    float& val, bool& flag ) 
{
  val=0;
  flag=false;
  if( _map.count( time )==0 ) return false;
  MusEcal::VarVec* vec_ = _map[time];
  val=(*vec_)[ii].first;
  flag=(*vec_)[ii].second;
  return true;
}

