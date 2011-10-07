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
MEVarVector::getClosestValid( ME::Time timeref, int ii,  vector< ME::Time >& time, float &val, bool &flag )
{
  time.clear();
  val=0.;
  flag=false;

  // cout<<" === > Inside getClosestValid "<< endl;
  ME::Time previousValidKey, nextValidKey;
  float minPrev=99999999.;
  float minNext=99999999.;
  bool prevFound=false;
  bool nextFound=false;
  float prevVal=0;float nextVal=0;
  float prevDiff=0;float nextDiff=0;

  for( MusEcal::VarVecTimeMap::iterator it=_map.begin(); 
       it!=_map.end(); it++ )
    {
      ME::Time t_ = it->first;
      float v; bool f;

      getValByTime( t_, ii, v, f );
      float diff=ME::timeDiff( t_, timeref, ME::iMinute );

      //  cout<<" === > diff: "<<diff<< " "<< minPrev<<" "<<minNext<<" "<<prevDiff<<" "<<nextDiff<<endl;
      if(f && diff<0. && TMath::Abs(diff)<minPrev){
	prevVal=v;
	minPrev=diff;
	previousValidKey=t_;
	prevFound=true;
	prevDiff=TMath::Abs(diff);
      }
      if(f && diff>=0. && TMath::Abs(diff)<minNext){
	nextVal=v;
	minNext=diff;
	nextValidKey=t_;
	nextFound=true;
	nextDiff=TMath::Abs(diff);
      }
    }

  if( prevFound && nextFound ){
    flag=true;
    val=prevVal+((nextVal-prevVal)/(prevDiff+nextDiff))*prevDiff;
    time.push_back(previousValidKey);
    time.push_back(nextValidKey);
  }else if( prevFound && prevDiff<400.){
    flag=true;
    val=prevVal;  
    time.push_back(previousValidKey);
    time.push_back(previousValidKey);
  }else if( nextFound && nextDiff<400.){
    flag=true;
    val=nextVal;  
    time.push_back(nextValidKey);
    time.push_back(nextValidKey);
  }else{
    flag=false;
    val=0.;
    time.push_back(timeref);
    time.push_back(timeref);
  }
  
  // cout<<" === > ... done "<< val<<" "<< flag<<" "<< time[0]<<" "<< time[1]<<endl;
}

void
MEVarVector::getClosestValidInFuture( ME::Time timeref, int ii,   ME::Time& time, float &val, bool &flag )
{
  val=0.;
  flag=false;
  time=0.;
  
  ME::Time nextValidKey;
  float minNext=99999999.;
  bool  nextFound=false;
  float nextDiff=0;
  float nextVal=0;

  for( MusEcal::VarVecTimeMap::iterator it=_map.begin(); 
       it!=_map.end(); it++ )
    {
      ME::Time t_ = it->first;
      float v; bool f;

      getValByTime( t_, ii, v, f );
      float diff=ME::timeDiff( t_, timeref, ME::iMinute );

      if(f && diff>0 && TMath::Abs(diff)<minNext){
	nextVal=v;
	minNext=diff;
	nextValidKey=t_;
	nextFound=true;
	nextDiff=TMath::Abs(diff);
      }
    }

  if( nextFound ){
    flag=true;
    val=nextVal;
    time=nextValidKey;
  }
  
  
}
void
MEVarVector::getClosestValidInPast( ME::Time timeref, int ii,   ME::Time& time, float &val, bool &flag )
{
  val=0.;
  flag=false;
  time=0.;
  
  ME::Time previousValidKey;
  float minPrevious=99999999.;
  bool  prevFound=false;
  float prevDiff=0;
  float prevVal=0;

  for( MusEcal::VarVecTimeMap::iterator it=_map.begin(); 
       it!=_map.end(); it++ )
    {
      ME::Time t_ = it->first;
      float v; bool f;

      getValByTime( t_, ii, v, f );
      float diff=ME::timeDiff( t_, timeref, ME::iMinute );

      if(f && diff<0 && TMath::Abs(diff)<minPrevious){
	prevVal=v;
	minPrevious=diff;
	previousValidKey=t_;
	prevFound=true;
	prevDiff=TMath::Abs(diff);
      }
    }

  if( prevFound ){
    flag=true;
    val=prevVal;
    time=previousValidKey;
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
MEVarVector::getValFlagAndNorm( int ii, 
			    const vector< ME::Time >& time, 
			    vector< float >& val,
			    vector< bool >& flag,
				double& norm	)
{
  val.clear();
  flag.clear();
  norm=0.0;

  double normS=0.0; int normC=0;
  for( unsigned int itime=0; itime<time.size(); itime++ )
    {
      ME::Time t_ = time[itime];
      float val_(0.);
      bool flag_(true);
      assert( getValByTime( t_, ii, val_, flag_ ) );
      val.push_back( val_ );
      flag.push_back( flag_ ); 

      if(flag_) {
	normS+=val_;
	normC++;
      }
    }
  
  if(normC!=0) norm=normS/double(normC);
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

void
MEVarVector::getTimeValFlagAndNorm( int ii, 
				vector< ME::Time >& time, 
				vector< float >& val,
				vector< bool >& flag,
				    double& norm,
				    const METimeInterval* interval )
{
  
  getTimeValAndFlag( ii, time, val, flag , interval);
  double normS=0.0; int normC=0;
  for(unsigned int i=0;i<time.size();i++){
    if(flag[i]) {
      normS+=double(val[i]);
      normC++;
    }
  }
  
  if(normC!=0) norm=normS/double(normC);
  else norm=0.0;
  
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

void
MEVarVector::getNormsInInterval(int ii, unsigned int nbuf, unsigned int nave, 
				const METimeInterval* timeInterval,
				std::vector< double >&  norm,
				std::vector< bool >&  normflag)
{
  norm.clear();			
  normflag.clear();
  
  double normbeg=0.0;
  bool flagbeg=true;
  unsigned int cbeg=0;
  std::vector< ME::Time > time;
  std::vector< float > val;
  std::vector< bool > flag;
  if(nbuf<1) nbuf=1;
  if(nave<2) nave=2;

  getTimeValAndFlag( ii, time, val, flag, timeInterval);
  
  for(unsigned int jj=nbuf;jj<time.size();jj++){
    if(flag[jj] && cbeg<nave){
      normbeg+=val[jj];
      cbeg++;
    }
  }
  if(cbeg!=0) {
    normbeg/=double(cbeg);
    flagbeg=true;
  }  else {
    normbeg=1.0;
    flagbeg=false;
  }
  norm.push_back(normbeg);
  normflag.push_back(flagbeg);
   
  double normend=0.0;
  bool flagend=true;
  unsigned int cend=0;
  for(int jj=time.size()-nbuf;jj>=0;jj--){
    if(flag[jj] && cend<nave){
      normend+=val[jj];
      cend++;
    }
  }	
  if(cend!=0){
    normend/=double(cend);
    flagend=true;
  }else{
    normend=1.0;
    flagend=false;
  }
  norm.push_back(normend);
  normflag.push_back(flagend);	 
  
}

void
MEVarVector::getNormsInInterval(int ivar,int irms, int inevt,
				unsigned int nbuf, unsigned int nave, 
				const METimeInterval* timeInterval,
				std::vector< double >&  norm,
				std::vector< double >&  enorm,
				std::vector< bool >&  normflag)
{
  norm.clear();	
  enorm.clear();			
  normflag.clear();
  
  double normbeg=0.0;
  double enormbeg=0.0;
  bool flagbeg=true;
  unsigned int    cbeg=0;
  double sbeg=0;
  std::vector< ME::Time > time;
  std::vector< float > val;
  std::vector< float > rms;
  std::vector< float > nevt;
  std::vector< bool > flag;

  if(nbuf<1) nbuf=1;
  if(nave<2) nave=2;

  double wi=0.0;
  double err=0.0;
    
  getTimeValAndFlag( ivar, time, val, flag, timeInterval);
  getTimeValAndFlag( irms, time, rms, flag, timeInterval);
  getTimeValAndFlag( inevt, time, nevt, flag, timeInterval);

  //cout<< " Entering getnormsininterval nbuf=" <<nbuf<< " nave=" << nave<<" ivar=" <<ivar<<" irms=" <<irms<<" inevt=" <<inevt<<endl;

  //cout<< "getnormsininterval size check: "<<val.size()<<" "<<rms.size()<<" "<<nevt.size()<<" "<<time.size()<<endl;


  for(unsigned int jj=nbuf;jj<time.size();jj++){
    
    //if(cbeg<nave) cout<< " check  time:"<<time[jj]<<" valbeg[" <<jj<<"]=" <<val[jj]<<"  rms=" <<rms[jj]<<"  nevt=" <<
    // nevt[jj]<<" flag="<< flag[jj]<<" cbeg="<<cbeg<<" ivar=" <<ivar<<" irms=" <<irms<<" inevt=" <<inevt;
    if(flag[jj] && nevt[jj] >100.0 && rms[jj]>0.0 && cbeg<nave){
      // if(cbeg<nave)cout<< " ==> passed "<< endl;
      err=rms[jj]/sqrt(nevt[jj]);
      wi=1/err;
      normbeg+=val[jj]*wi;
      sbeg+=wi; 
      enormbeg+=wi*wi;
      cbeg++;
      //}else{
      //if(cbeg<nave) cout<< " ==> failed "<< endl;
    }
  }
  
  
  if(cbeg!=0 && sbeg!=0.0 && enormbeg!=0.0 ) {
    normbeg/=double(sbeg);
    enormbeg=sqrt(enormbeg);
    enormbeg=1.0/enormbeg;
    //    cout<< " GETNORMS1 normbeg=" <<normbeg<<" enormbeg=" <<enormbeg<<" cbeg=" <<cbeg<<endl;
    flagbeg=true;
  }  else {
    normbeg=1.0;
    enormbeg=0.0;
    flagbeg=false;
    //    cout<< " GETNORMS1 nothing cbeg=" <<cbeg<<" sbeg="<<sbeg<<" enormbeg=" <<enormbeg<<endl;
  }
  norm.push_back(normbeg);
  enorm.push_back(enormbeg);
  normflag.push_back(flagbeg);
  
  double normend=0.0;
  double enormend=0.0;
  bool flagend=true;
  unsigned int cend=0;
  double send=0;
  wi=0.0;
  err=0.0;

  for(unsigned int jj=time.size()-nbuf;jj>=0;jj--){
    if(flag[jj] && nevt[jj] >100.0 && rms[jj]>0.0 && cend<nave){ 
      err=rms[jj]/sqrt(nevt[jj]);
      wi=1/err;
      normend+=val[jj]*wi;
      send+=wi; 
      enormend+=wi*wi;
      cend++;
    }
  }	
  if(cend!=0 && send!=0.0 && enormend!=0.0 ){
    normend/=double(send);
    enormend=sqrt(enormend);
    enormend=1.0/enormend;
    //    cout<< " GETNORMS2 normend=" <<normend<<" enormend=" <<enormend<<" cend=" <<cend<<endl;
    flagend=true;
  }else{
    normend=1.0;
    flagend=false;
    enormend=0.0;
    //    cout<< " GETNORMS2 nothing cend=" <<cend<<" send="<<send<<" enormend=" <<enormend<<endl;
  }
  norm.push_back(normend);
  enorm.push_back(enormend);
  normflag.push_back(flagend);	 
  
}

// void
// MEVarVector::getVarInInterval( const METimeInterval& interval, 
// 			       int var, 
// 			       int& nrun, int& ngood, double& norm, double& min, double& max,
// 			  vector< double >& val,
// 			  vector< bool >& good, vector< double >& error,
// 			  bool clear )
// {

//   if( clear )
//     {
//       nrun  = 0;
//       ngood = 0;
//       norm  = 0;
//       min   = 1000000;
//       max   = 0;
//       val.clear();
//       good.clear();
//       error.clear();
//     }

//   ME::Time firstKey = interval.firstKey();
//   ME::Time lastKey  = interval.lastKey();

//   int nrun_    =  0;
//   int ngood_   =  0;
//   double norm_ =  0;
//   double min_  =  1000000;
//   double max_  =  0;
//   MERunIterator it;
//   for( it=runMgr->from( firstKey ); it!=runMgr->to( lastKey ); it++ )
//     {
//       ME::Time key_ = it->first;
//       double val_(0.), error_(0.), extra_(0.);
//       bool good_(true);
//       if( !getVarByKey( key_, type, var, val_, good_, error_, extra_ ) ) continue;
//       nrun_++;
//       if( good_ )
// 	{
// 	  norm_ += val_;
// 	  if( val_<min_ ) min_ = val_;
// 	  if( val_>max_ ) max_ = val_;
// 	  ngood_++;
// 	}
//       val.push_back( val_ );
//       good.push_back( good_ );
//       error.push_back( error_ );
//     }
//   if( ngood_!=0 ) norm_ /= ngood_;

//   nrun += nrun_;
//   if( min_<min ) min = min_;
//   if( max_>max ) max = max_;
//   int ngoodtot = ngood + ngood_;
//   if( ngoodtot>0 ) norm = ( ngood*norm + ngood_*norm_ )/ngoodtot;
//   ngood += ngood_;

// }

// // same, without filling vectors
// void
// MELeaf::getNormInInterval( const MEKeyInterval& interval, 
// 			   int type, int var, 
// 			   int& nrun, int& ngood, double& norm, double& rms, double& min, double& max )
// {

//   MERunManager* runMgr = _me->_runMgr[type];

//   ME::Time firstKey = interval.firstKey();
//   ME::Time lastKey  = interval.lastKey();

//   ngood = 0;
//   nrun =  0;
//   norm =  0;
//   rms  =  0;
//   min  =  1000000;
//   max  =  0;
//   MERunIterator it;
//   for( it=runMgr->from( firstKey ); it!=runMgr->to( lastKey ); it++ )
//     {
//       ME::Time key_ = it->first;
//       bool good_(true);
//       double val_(0.), error_(0.), extra_(0.);
//       if( !getVarByKey( key_, type, var, val_, good_, error_, extra_ ) ) continue;
//       nrun++;
//       if( good_ )
// 	{
// 	  norm += val_;
// 	  rms  += val_*val_;
// 	  if( val_<min ) min = val_;
// 	  if( val_>max ) max = val_;
// 	  ngood++;
// 	}
//     }
//   if( ngood!=0 ) 
//     {
//       norm /= ngood;
//       rms  /= ngood;
//       rms  -= norm*norm;
//       if( rms>0 ) rms = sqrt( rms );
//       else rms = 0;
//     }
//   //  cout << firstKey << "(" << runMgr->altKey(firstKey) << ") --> " << lastKey << "(" << runMgr->altKey(lastKey) << ")";
//   //  cout <<  MusEcal::historyVarName[var] << " n/nok/norm/min/max " << nrun << "/" << ngood << "/" << norm << "/" << min <<  "/" << max << endl;
// }
