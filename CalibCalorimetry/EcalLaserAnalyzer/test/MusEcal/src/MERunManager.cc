#define MERunManager_cxx

#include <errno.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <string>
using namespace std;
#include "MERunManager.hh"
#include "../../interface/MEGeom.h"
#include "../../interface/MELaserPrim.h"
#include "MERun.hh"
#include "../../interface/MEChannel.h"
#include "MEVarVector.hh"

#include <TFile.h>
#include <TString.h>
#include <TObjString.h>

ClassImp( MERunManager )
 
MERunManager::MERunManager( unsigned int lmr, 
			    unsigned int type, 
			    unsigned int color )
: _lmr(lmr), _type(type), _color(color), _tree(0)
{  
  ME::regionAndSector( _lmr, _reg, _sm, _dcc, _side );  
  _lmdataPath = ME::lmdataPath( _lmr );
  _primPath   = ME::primPath( _lmr );

  updateRunList();
}

MERunManager::~MERunManager()
{
  // the run manager owns the runs
  // here delete all the runs...
  while( !_runs.empty() ) 
  {    
    MERun*   aRun = _runs.begin()->second;
    //    cout << ".... Deleting run "  << endl;
    //    aRun->print( cout );
    delete aRun;
    _runs.erase( _runs.begin() );  
  }
}

void
MERunManager::updateRunList()
{
  TString runlistfile = ME::runListName( _lmr, _type, _color );

  ifstream fin;
  fin.open(runlistfile);

  MERun* aRun(0);
  ME::Time key(0);
  bool store(false);
  do
    {
      if( store )  
	{
	  _runs[key] = aRun;
	}
      string rundir;
      long long tsb, tse;
      int rr, lb, mpga , mem, pp, ff, dd, evts;      
      fin >> rundir;
      fin >> rr >> lb >> evts >> tsb >> tse >> mpga >> mem >> pp >> ff >> dd;      
      if( rr<MusEcal::firstRun || rr>MusEcal::lastRun ) continue;
      
      key = ME::time_high(tsb);

      //      cout << "DEBUG MERunManager::updateRunList after key: "<< tsb<< " "<< key << endl;

      ME::Header header;
      header.rundir = rundir;
      header.dcc    = _dcc;
      header.side   = _side;
      header.run=rr;
      header.lb=lb;
      header.ts_beg=tsb;
      header.ts_end=tse;
      header.events=evts;
      
      ME::Settings settings;    
      settings.type = _type;
      settings.wavelength = _color;
      settings.mgpagain=mpga;
      settings.memgain=mem;
      settings.power=pp;
      settings.filter=ff;
      settings.delay=dd;
      
      aRun=0;
      store = ( _runs.count( key )==0 );
      //      cout << "DEBUG MERunManager::updateRunList after store: "<< store << endl;
      if( store ) 
	{
	  TString fname = ME::rootFileName( header, settings );
	  FILE *test;
	  test = fopen( fname, "r" );
	  if(test)
	    {
	      cout << "File " << fname << " found." << endl;
	      fclose( test );
	    }
	  else
	    {
	      //	       cout << "File " << fname << " not found." << endl;
	      //	       cout << "Warning: file " << fname << " does not exist" << endl;
	      store = false;
	      TString path_ = _lmdataPath;
	      switch( _type )
		{
		case ME::iLaser:
		  path_ += "Laser/";     break;
		case ME::iTestPulse:
		  path_ += "TestPulse/"; break;
		  //		case ME::iPedestal:
		  //		  path_ += "Pedestal/";  break;
		default:
		  abort();
		};
	      path_ += "Analyzed/"; // FIXME!...
	      path_ += header.rundir;
	      path_ += "/";
	      //	      cout << path_ << endl;
	      //	      cout << _primPath << endl;
	      MELaserPrim prim( header, settings, path_, fname );
	      if( prim.init_ok )
		{
		  cout << "Primitives for DCC=" << header.dcc << " Side=" << header.side; 
		  cout << " Type=" << settings.type;
		  if( settings.type==ME::iLaser ) cout << " Color=" << settings.wavelength;
		  cout << " Run=" << header.rundir  << " TS=" << header.ts_beg << endl;
		  		  cout << "Fill histograms " << endl;
		  prim.fillHistograms();
		  		  cout << "Write histograms " << endl;
		  prim.writeHistograms();
		  // 
		  // OK try again
		  test = fopen( fname, "r" );
		  if(test)
		    {
		      store = true;
		      fclose( test );
		    }
		}
	    }	  
	  //	  cout << "DEBUG MERunManager::updateRunList after store2: "<< store << endl;
	  if( store ) aRun = new MERun( header, settings, fname );
	}
    }
  while( fin.peek() != EOF );
  fin.close();
  
  bool runlists_are_not_complete = false;
  if( runlists_are_not_complete )
    {

      vector< TString > files = vector< TString >();
      DIR *dp;
      struct dirent *dirp;
      if( ( dp  = opendir(_primPath.Data()) ) == 0 ) 
	{
	  cout << "Error(" << errno << ") opening " << _primPath << endl;
	  return;
	}
      else
	{
	       cout << "Opening " << _primPath << endl;
	}
  
      while( ( dirp = readdir(dp) ) != 0 ) 
	{
	  files.push_back( TString(dirp->d_name) );
	        cout << "Taking " << TString(dirp->d_name) << endl;
	}
      closedir(dp);

      int count(0);
      int count_notfound(0);
      for( unsigned ii=0; ii<files.size(); ii++ ) 
	{      
	  TString fname_ = files[ii];
	  if( !fname_.Contains(".root") ) continue;
	  if( _type==ME::iLaser )
	    {
	      if( !fname_.Contains("Laser") ) continue;
	      if( _color==ME::iBlue && !fname_.Contains("Blue") ) continue; 
	    }
	  else if( _type==ME::iTestPulse && !fname_.Contains("TestPulse") ) continue; 

	  TString file_ = fname_;
	  TObjArray* array_ = file_.Tokenize(".");
	  TObjString* token_ = (TObjString*)array_->operator[](0);
	  TString str_ = token_->GetString();
	  //      cout << str_ << endl;
	  array_ = str_.Tokenize("_");
	  int nTokens_= array_->GetEntries();
	  if( nTokens_==0 ) continue;
	  //       for( int iToken=0; iToken<nTokens_; iToken++ )
	  // 	{
	  // 	  token_ = (TObjString*)array_->operator[](iToken);
	  // 	  cout << iToken << "->" << token_->GetString() << endl;
	  // 	}       
	  TString lmrName_ = ((TObjString*)array_->operator[](1))->GetString();
	  if( lmrName_!=ME::smName( _lmr ) ) continue;
	        cout << "Sector=" << lmrName_ << endl;
      
	  int side_ = ((TObjString*)array_->operator[](2))->GetString().Atoi();
	  if( side_!=_side ) continue;
	        cout << "Side=" << side_ << endl;
      
	  TString rundir_ = ((TObjString*)array_->operator[](4))->GetString();
	  rundir_+="_";
	  rundir_+= ((TObjString*)array_->operator[](5))->GetString();
	        cout << "Rundir=" << rundir_ << endl;
      
	  int run_= TString(rundir_(3,5)).Atoi();
	        cout << "run=" << run_ << endl;

	  int lb_= TString(rundir_(11,4)).Atoi();
	        cout << "LB=" << lb_ << endl;

	  long long ts_ = TString( ((TObjString*)array_->operator[](6))->GetString().operator()(2,32)).Atoll();
	        cout << "TS=" << ts_ << endl;

	  ME::Time key = ME::time_high(ts_);

	  count++;
      
	  if( _runs.count(key)!=0 ) continue;

	  count_notfound++;

	   cout << lmrName_ << " Adding run not found in the runlist file " << fname_ << endl;

	  ME::Header header;
	  header.rundir = rundir_;
	  header.dcc    = _dcc;
	  header.side   = _side;
	  header.run=run_;
	  header.lb=lb_;
	  header.ts_beg=ts_;
	  header.ts_end=ts_;
	  header.events=600;
      
	  ME::Settings settings;    
	  settings.type = _type;
	  settings.wavelength = _color;
	  settings.mgpagain=0;
	  settings.memgain=0;
	  settings.power=0;
	  settings.filter=0;
	  settings.delay=0;

	  TString fname = ME::rootFileName( header, settings );     
	  MERun* aRun = new MERun( header, settings, fname );      
	  _runs[key] = aRun;
	}
  //       << " (not found in runlist=" << count_notfound << ")" << endl;
    }

  cout << "LMR=" << _lmr << "---> Number of primitive files: " << _runs.size() << endl; 
  MusEcal::RunIterator p;
  _first = 0;
  for( p=_runs.begin(); p!=_runs.end(); ++p )
    {
      ME::Time key = p->first;
      if( _first==0 ) _first = key;
      _last = key;
    }
  _current   = _first;
  _normFirst = _first;
  _normLast  = _last;
}

MusEcal::RunIterator
MERunManager::it( ME::Time key )
{
  return _runs.find( key );
}

MusEcal::RunIterator
MERunManager::it()
{
  return _runs.begin();
}

MusEcal::RunIterator 
MERunManager::cur()
{
  return _runs.find( _current );
}

MusEcal::RunIterator 
MERunManager::first()
{
  return _runs.equal_range( _first ).first;
}

MusEcal::RunIterator 
MERunManager::last()
{
  return _runs.equal_range( _last ).second;
}

MusEcal::RunIterator 
MERunManager::from( ME::Time key )
{
  return _runs.equal_range( key ).first;
}

MusEcal::RunIterator 
MERunManager::to( ME::Time key )
{
  return _runs.equal_range( key ).second;
}

MusEcal::RunIterator 
MERunManager::begin()
{
  return _runs.begin();
}

MusEcal::RunIterator 
MERunManager::end()
{
  return _runs.end();
}

ME::Time
MERunManager::beginKey() const
{
  return _runs.begin()->first;
}

ME::Time
MERunManager::endKey() const
{
  return _runs.rbegin()->first;
}

MERun* 
MERunManager::beginRun() 
{
  return _runs.begin()->second;
}

MERun* 
MERunManager::endRun() 
{
  return _runs.rbegin()->second;
}

MERun*   
MERunManager::curRun() 
{
  return run( _current );
}

MERun* 
MERunManager::firstRun()
{
  return run( _first );
}

MERun* 
MERunManager::lastRun() 
{
  return run( _last );
}

MERun* 
MERunManager::run( ME::Time key )
{
  if( key==0 ) return 0;
  MusEcal::RunIterator p = _runs.find( key );
  if( p!=_runs.end() ) return p->second;
  return 0;
}

void 
MERunManager::print()
{
  
  cout << "Number of runs to analyze : " << _runs.size() << endl;
  if( _runs.size()>0 )
    {
      //      cout << "First: " << _first << endl;
      //      firstRun()-> print( cout );
      cout << "Current: " << _current << endl;
      curRun()  -> print( cout );
      //      cout << "Last: " << _last << endl;
      //      lastRun() -> print( cout );
//       for( int ixb=0; ixb<=84; ixb++ )
// 	{
// 	  for( int iyb=0; iyb<=19; iyb++ )
// 	    {
// 	      //	      int ix= (int) ax->GetBinLowEdge( ixb );
// 	      //	      int iy= (int) ay->GetBinLowEdge( iyb );
// 	      cout << "ix=" << ixb << "\tiy=" << iyb;
// 	      cout << "\tID     " 
// 		   << "\t" << 
// 		(int) curRun()->getVal( ME::iLmfLaserPrim,"LOGIC_ID",ixb, iyb )
// 		   << "\tAPD     " 
// 		   << "\t" << 
// 		curRun()->getVal( ME::iLmfLaserPrim,"MEAN",ixb, iyb )
// 		   << "\t" << 
// 		curRun()->getVal( ME::iLmfLaserPrim,"RMS",ixb, iyb )
// 		   << "\t" << 
// 		curRun()->getVal( ME::iLmfLaserPrim,"M3",ixb, iyb )
// 		   << endl;
// 	    }
// 	}
    } 
}

bool
MERunManager::setCurrentRun( ME::Time key )
{
  // first close the file of the current run
  MERun* aRun = curRun();
  if( aRun ) aRun->closeLaserPrimFile();

  key = closestKeyInFuture( key );
  _current = key;
  MusEcal::RunIterator p = _runs.find( key );
  if( p==_runs.end() ) return false;
  aRun = p->second;
  assert( aRun!=0 );
  return true;
}

void
MERunManager::setNoCurrent()
{
  _current=0;
}

ME::Time
MERunManager::closestKey( ME::Time key )
{
  if( _runs.size()==0 ) return 0;
  MusEcal::RunIterator p = _runs.find( key );
  if( p==_runs.end() )
    {
      cout << "**** Key is not found: find the closest key " << endl;
      p = _runs.lower_bound( key );
      if( p==_runs.end() )
	{
	  cout << " ----> Closest is last entry " << endl;
	  p--;
	}
      else if( p==_runs.begin() )
	{
	  cout << " ----> Closest is first entry " << endl;
	}
      else
	{
	  MusEcal::RunIterator p0 = p;
	  p0--;
	  cout << " ----> Key " << key << " is between " << p0->first << " and " << p->first << endl;
	  if( key-(p0->first) < (p->first)-key ) p--; 
	}
      key = p->first;
      cout << "**** Closest key is " << key << endl;
    }
  return key;
}

ME::Time
MERunManager::closestKeyInFuture( ME::Time key )
{
  if( _runs.size()==0 ) return 0;
  MusEcal::RunIterator p = _runs.find( key );
  if( p==_runs.end() )
    {
      p = _runs.lower_bound( key );
      if( p==_runs.end() )
	{
	  p--;
	}
      key = p->first;
    }
  return key;
}

bool
MERunManager::setPlotRange( ME::Time key1, ME::Time key2, bool verbose )
{
  if( key2<=key1 ) 
    {
      key2  = lastKey(); 
      key1  = firstKey(); 
    }
  if( key1>key2 ) return false;
  MusEcal::RunIterator p1 = _runs.find( key1 );
  MusEcal::RunIterator p2 = _runs.find( key2 );
  if( p1==_runs.end() || p2==_runs.end() ) return false;
  _first = key1;
  _last  = key2;
  if( _current<_first ) setCurrentRun( _first );
  if( _current>_last )  setCurrentRun( _last );
  _normFirst = _first;
  _normLast  = _last;
  return true;
}

bool
MERunManager::setNormRange( ME::Time key1, ME::Time key2, bool verbose )
{
  if( key1>key2 ) return false;
  MusEcal::RunIterator p1 = _runs.find( key1 );
  MusEcal::RunIterator p2 = _runs.find( key2 );
  if( p1==_runs.end() || p2==_runs.end() ) return false;
  _normFirst = key1;
  _normLast  = key2;
  return true;
}

void
MERunManager::setBadRun()
{
  _badRuns[ curKey() ] = false;
}

void
MERunManager::setBadRange( ME::Time key1, ME::Time key2, bool verbose )
{
  MusEcal::RunIterator it;
  for( it=from( key1 ); it!=to( key2 ); it++ )
    {
      ME::Time key = it->first;
      _badRuns[ key ]=false;
    }
}

void
MERunManager::refreshBadRuns()
{
  setPlotRange( beginKey(), endKey(), true );
  _badRuns.clear();
}

bool 
MERunManager::isGood( ME::Time key )
{
  if( _badRuns.count( key ) ) return _badRuns[ key ];
  if( _runs.count( key ) ) return true;
  return false;
}

MEChannel* 
MERunManager::tree()
{
  if( _tree!=0 ) return _tree;
  _tree = ME::regTree(_reg)->getDescendant( ME::iLMRegion, _lmr );
  assert( _tree!=0 );
  return _tree;
}

void
MERunManager::fillMaps()
{
  //
  // First, fill "APD" map
  //

  // cleanup
  while( !_apdMap.empty() ) 
  {    
    delete _apdMap.begin()->second;
    _apdMap.erase( _apdMap.begin() );  
  }

  for( int ipn=0; ipn<2; ipn++ )
    {
      while( !_pnMap[ipn].empty() ) 
	{    
	  delete _pnMap[ipn].begin()->second;
	  _pnMap[ipn].erase( _pnMap[ipn].begin() );  
	}
    }

  while( !_mtqMap.empty() ) 
  {    
    delete _mtqMap.begin()->second;
    _mtqMap.erase( _mtqMap.begin() );  
  }

  unsigned int size_(0);
  unsigned int table_(0);
  if( _type==ME::iLaser )
    {
      table_=ME::iLmfLaserPrim;
      size_=ME::iSizeAPD;
    }
  else if( _type==ME::iTestPulse )
    {
      table_=ME::iLmfTestPulsePrim;
      size_=ME::iSizeTPAPD;
    }
  
  vector< MEChannel* > listOfChan_;
  //
  // at the crystal level
  //
  listOfChan_.clear();
  tree()->getListOfDescendants( ME::iCrystal, listOfChan_ );

  int reg_ = tree()->getAncestor(0)->id(); // fixme
  cout << "id=" << reg_ << endl;

  cout << "Filling APD Maps (number of C=" << listOfChan_.size() << ")" << endl;
  for( unsigned int ichan=0; ichan<listOfChan_.size(); ichan++ )
    {
      //      if( ichan%25==0 ) cout << "." << flush;
      MEChannel* leaf_ = listOfChan_[ichan];
      int ix = leaf_->ix();
      int iy = leaf_->iy();
      if( reg_==ME::iEBM || reg_==ME::iEBP )
	{
	  int ieta = leaf_->ix();
	  int iphi = leaf_->iy();
	  MEEBGeom::XYCoord ixy = MEEBGeom::localCoord( ieta, iphi );
	  ix = ixy.first;
	  iy = ixy.second;
	}

      MEVarVector* varVector_ = new MEVarVector( size_ );
      for( MusEcal::RunIterator p=_runs.begin(); p!=_runs.end(); ++p )
	{
	  MERun* run_ = p->second;
	  ME::Time time = run_->time();
	  varVector_->addTime( time );
	  for( unsigned ii=0; ii<size_; ii++ )
	    {
	      float val = run_->getVal( table_, ii, ix, iy );
	      varVector_->setVal( time, ii, val );
	    }
	}
      _apdMap[leaf_] = varVector_;
    }
  cout << "...done." << endl;

  //
  // At higher levels
  //
  for( int ig=ME::iSuperCrystal; ig>=ME::iLMRegion; ig-- )
    {
      listOfChan_.clear();
      if( ig==ME::iLMRegion ) listOfChan_.push_back( tree() );
      else tree()->getListOfDescendants( ig, listOfChan_ );
      cout << "Filling APD Maps (number of " << ME::granularity[ig] << "=" 
      	   << listOfChan_.size() << ")" << endl;
      for( unsigned int ichan=0; ichan<listOfChan_.size(); ichan++ )
	{
	  //	  if( ichan ) cout << "." << flush;
	  MEChannel* leaf_ = listOfChan_[ichan];
	  MEVarVector* varVector_ = new MEVarVector( size_ );
	  for( MusEcal::RunIterator p=_runs.begin(); p!=_runs.end(); ++p )
	    {
	      MERun* run_ = p->second;
	      ME::Time time = run_->time();
	      varVector_->addTime( time );
	      for( unsigned ii=0; ii<size_; ii++ )
		{
		  float val=0;
		  float n=0;
		  // loop on daughters
		  for( unsigned int idau=0; idau<leaf_->n(); idau++ )
		    {
		      float val_(0);
		      bool flag_=true;
		      assert( _apdMap[leaf_->d(idau)]
			      ->getValByTime( time, ii, val_, flag_ ) );
		      if( val_>0 )
			{
			  n++;
			  val+=val_;
			}
		    }
		  if( n!=0 ) val/=n; 
		  varVector_->setVal( time, ii, val );
		}
	    }
	  _apdMap[leaf_] = varVector_;
	}
      cout << "...done." << endl;
    }
  
  //
  // Second, fill "PN" map
  // 
  

  if( _type==ME::iLaser )
    {
      table_=ME::iLmfLaserPnPrim;
      size_=ME::iSizePN;
    }
  else if( _type==ME::iTestPulse )
    {
      table_=ME::iLmfTestPulsePnPrim;
      size_=ME::iSizeTPPN;
    }

  listOfChan_.clear();
  tree()->getListOfDescendants( ME::iLMModule, listOfChan_ );


  cout << "Filling PN Maps (number of LMM=" << listOfChan_.size() << ")" << endl;


  for( unsigned int ichan=0; ichan<listOfChan_.size(); ichan++ )
    {
      //      if( ichan ) cout << "." << flush;
      MEChannel* leaf_ = listOfChan_[ichan];
      int id_ = leaf_->id();

      for( int ipn=0; ipn<2; ipn++ )
	{
	  MEVarVector* varVector_ = new MEVarVector( size_ );
	  for( MusEcal::RunIterator p=_runs.begin(); p!=_runs.end(); ++p )
	    {
	      MERun* run_ = p->second;
	      ME::Time time = run_->time();
	      varVector_->addTime( time );
	      for( unsigned jj=0; jj<size_; jj++ )
		{
		  //float val=0;
		  float val = run_->getVal( table_, jj, id_, ipn );
		  varVector_->setVal( time, jj, val );
		}
	    }
	  _pnMap[ipn][leaf_] = varVector_;
	}
    }
  cout << "...done." << endl; 


  //
  // At higher levels
  //
  listOfChan_.clear();
  int ig=ME::iLMRegion;
  tree()->getListOfDescendants( ig, listOfChan_ );
  cout << "Filling PN Maps (number of " << ME::granularity[ig] << "=" 
       << listOfChan_.size() << ")" << endl;
  for( unsigned int ichan=0; ichan<listOfChan_.size(); ichan++ )
    {
      MEChannel* leaf_ = listOfChan_[ichan];

      for( int ipn=0; ipn<2; ipn++ ){
	
	MEVarVector* varVector_ = new MEVarVector( size_ );
	for( MusEcal::RunIterator p=_runs.begin(); p!=_runs.end(); ++p )
	  {
	    MERun* run_ = p->second;
	    ME::Time time = run_->time();
	    varVector_->addTime( time );
	    for( unsigned ii=0; ii<size_; ii++ )
	      {
		float val=0;
		float n=0;
		
		// loop on daughters
		for( unsigned int idau=0; idau<leaf_->n(); idau++ )
		  {
		    float val_(0);
		    // bool flag_=true;
		    // FIXME: add flag
		    MEChannel* dau=leaf_->d(idau);
		    int id_=dau->id();
		    val_ = run_->getVal( table_, ii, id_, ipn );
		    
		    if( val_>0 )
		      {
			n++;
			val+=val_;
		      }
		  }
		if( n!=0 ) val/=n; 
		varVector_->setVal( time, ii, val );
	      }
	  }
	_pnMap[ipn][leaf_] = varVector_;
      }
    }  
  cout << "...done." << endl;
  
  //
  // Third, fill "MTQ" map
  // 

  if( _type==ME::iLaser )
    {
      table_=ME::iLmfLaserPulse;
      size_=ME::iSizeMTQ;
    }
  else 
    {
      table_=ME::iLmfLaserPulse;
      size_=0;
    }
  listOfChan_.clear();
  listOfChan_.push_back( tree() );
  
  cout << "Filling MTQ Maps: (number of LMR=" << listOfChan_.size() << ") "<< endl;
  
  for( unsigned int ichan=0; ichan<listOfChan_.size(); ichan++ )
    {
      MEChannel* leaf_ = listOfChan_[ichan];
      int id_ = leaf_->id();
      
      MEVarVector* varVector_ = new MEVarVector( size_ );
      for( MusEcal::RunIterator p=_runs.begin(); p!=_runs.end(); ++p )
	{
	  MERun* run_ = p->second;
	  ME::Time time = run_->time();
	  varVector_->addTime( time );
	  for( unsigned jj=0; jj<size_; jj++ )
	    {
	      float val = run_->getVal( table_, jj, id_ );
	      varVector_->setVal( time, jj, val );
	    }
	}
      _mtqMap[leaf_] = varVector_;
      
    }
  cout << "...done." << endl; 

  setFlags();
}

void
MERunManager::setFlags()
{
  if( _type==ME::iLaser )
    {
      setLaserFlags();
    }
  else if( _type==ME::iTestPulse )
    {
      setTestPulseFlags();
    }
}

void
MERunManager::setLaserFlags()
{
  vector< MEChannel* > listOfChan_;
  //
  // at the crystal level
  //
  listOfChan_.clear();
  tree()->getListOfDescendants( ME::iCrystal, listOfChan_ );

  cout << "Setting APD Quality Flags " << endl;

  for( unsigned int ichan=0; ichan<listOfChan_.size(); ichan++ )
    {
      MEChannel* leaf_ = listOfChan_[ichan];
      MEVarVector* varVector_ = apdVector( leaf_ );
      vector< ME::Time > time;
      varVector_->getTime( time );
      for( unsigned int itime=0; itime<time.size(); itime++ )
	{
	  ME::Time t_ = time[itime];
	  float apd_;
	  float apd_rms_;
	  float apd_time_;
	  float apd_time_nevt_;
	  bool flag_;
	  varVector_->getValByTime( t_, ME::iAPD_MEAN, apd_, flag_ );
	  varVector_->getValByTime( t_, ME::iAPD_RMS, apd_rms_, flag_ );
	  varVector_->getValByTime( t_, ME::iAPD_TIME_MEAN, apd_time_, flag_ );
	  varVector_->getValByTime( t_, ME::iAPD_TIME_NEVT, apd_time_nevt_, flag_ );
	 
	  if( apd_<100  
	      || (apd_rms_<10 || apd_rms_>500) 
	      || (apd_time_<5 || apd_time_>8.5)
	      || (apd_time_nevt_<400 || apd_time_nevt_>650) )
	    {
	      varVector_->setFlag( t_, ME::iAPD_MEAN, false );
	      varVector_->setFlag( t_, ME::iAPD_RMS, false );
	      varVector_->setFlag( t_, ME::iAPD_M3, false );
	      varVector_->setFlag( t_, ME::iAPD_OVER_PNA_MEAN, false );
	      varVector_->setFlag( t_, ME::iAPD_OVER_PNA_RMS, false );
	      varVector_->setFlag( t_, ME::iAPD_OVER_PNA_M3, false );
	      varVector_->setFlag( t_, ME::iAPD_OVER_PNB_MEAN, false );
	      varVector_->setFlag( t_, ME::iAPD_OVER_PNB_RMS, false );
	      varVector_->setFlag( t_, ME::iAPD_OVER_PNB_M3, false );
	      varVector_->setFlag( t_, ME::iAPD_OVER_PN_MEAN, false );
	      varVector_->setFlag( t_, ME::iAPD_OVER_PN_RMS, false );
	      varVector_->setFlag( t_, ME::iAPD_OVER_PN_M3, false );
	      varVector_->setFlag( t_, ME::iAPD_OVER_APDA_MEAN, false ); // JM
	      varVector_->setFlag( t_, ME::iAPD_OVER_APDA_RMS, false ); // JM
	      varVector_->setFlag( t_, ME::iAPD_OVER_APDA_M3, false ); // JM
	      varVector_->setFlag( t_, ME::iAPD_OVER_APDB_MEAN, false ); // JM
	      varVector_->setFlag( t_, ME::iAPD_OVER_APDB_RMS, false ); // JM
	      varVector_->setFlag( t_, ME::iAPD_OVER_APDB_M3, false ); // JM
	      varVector_->setFlag( t_, ME::iAPD_TIME_MEAN, false );
	      varVector_->setFlag( t_, ME::iAPD_TIME_RMS, false );
	      varVector_->setFlag( t_, ME::iAPD_TIME_M3, false );
	      varVector_->setFlag( t_, ME::iAPD_TIME_M3, false ); 
	    }
	}
    }
  
  listOfChan_.clear();
  tree()->getListOfDescendants( ME::iLMModule, listOfChan_ );

  cout << "Setting PN Quality Flags " << endl;
  for( unsigned int ichan=0; ichan<listOfChan_.size(); ichan++ )
    {
      MEChannel* leaf_ = listOfChan_[ichan];
      for( int ipn=0; ipn<2; ipn++ )
	{
	  MEVarVector* varVector_ = pnVector( leaf_, ipn );
	  vector< ME::Time > time;
	  varVector_->getTime( time );
	  for( unsigned int itime=0; itime<time.size(); itime++ )
	    {
	      ME::Time t_ = time[itime];
	      float pn_;
	      bool flag_;
	      varVector_->getValByTime( t_, ME::iPN_MEAN, pn_, flag_ );
	      if( pn_<100 )  
		{
		  varVector_->setFlag( t_, ME::iPN_MEAN, false );
		  varVector_->setFlag( t_, ME::iPN_RMS, false );
		  varVector_->setFlag( t_, ME::iPN_M3, false );
		  if( ipn==0 )
		    {
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNA_MEAN, false );
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNA_RMS, false );
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNA_M3, false );
		    }
		  if( ipn==1 )
		    {
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNB_MEAN, false );
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNB_RMS, false );
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNB_M3, false );
		    }
		  setFlag( leaf_, t_, ME::iAPD_OVER_PN_MEAN, false );
		  setFlag( leaf_, t_, ME::iAPD_OVER_PN_RMS, false );
		  setFlag( leaf_, t_, ME::iAPD_OVER_PN_M3, false );
		}
	    }
	}
    }

  listOfChan_.clear();
  listOfChan_.push_back( tree() );

  cout << "Setting MTQ Quality Flags " << endl;
  
  for( unsigned int ichan=0; ichan<listOfChan_.size(); ichan++ )
    {
      MEChannel* leaf_ = listOfChan_[ichan];
      MEVarVector* varVectorMtq_ = mtqVector( leaf_ );

      vector< ME::Time > time;
      varVectorMtq_->getTime( time );

      for( unsigned int itime=0; itime<time.size(); itime++ )
	{
	  ME::Time t_ = time[itime];
	
	  float mtqampl_;
	  float mtqfwhm_;
	  bool flagmtq_; 
	  varVectorMtq_->getValByTime( t_, ME::iMTQ_AMPL, mtqampl_, flagmtq_ );  
	  varVectorMtq_->getValByTime( t_, ME::iMTQ_FWHM, mtqfwhm_, flagmtq_ );  

	  if( mtqampl_< 100. ||  
	      (mtqfwhm_ <10.0 ||  mtqfwhm_ >100.0) ){
	    varVectorMtq_->setFlag( t_, ME::iMTQ_AMPL, false );
	    varVectorMtq_->setFlag( t_, ME::iMTQ_RISE, false );
	    varVectorMtq_->setFlag( t_, ME::iMTQ_FIT_METHOD, false );
	    varVectorMtq_->setFlag( t_, ME::iMTQ_FWHM, false );
	    varVectorMtq_->setFlag( t_, ME::iMTQ_FW20, false );
	    varVectorMtq_->setFlag( t_, ME::iMTQ_FW80, false );
	    varVectorMtq_->setFlag( t_, ME::iMTQ_SLIDING, false );
	    varVectorMtq_->setFlag( t_, ME::iMTQ_TIME, false );
	  } 

	}
    }
}

void
MERunManager::setTestPulseFlags()
{
  vector< MEChannel* > listOfChan_;
  //
  // at the crystal level
  //
  listOfChan_.clear();
  tree()->getListOfDescendants( ME::iCrystal, listOfChan_ );
  //  cout << "Setting TPAPD Quality Flags " << endl;
  for( unsigned int ichan=0; ichan<listOfChan_.size(); ichan++ )
    {
      MEChannel* leaf_ = listOfChan_[ichan];
      MEVarVector* varVector_ = apdVector( leaf_ );
      vector< ME::Time > time;
      varVector_->getTime( time );
      for( unsigned int itime=0; itime<time.size(); itime++ )
	{
	  ME::Time t_ = time[itime];
	  float apd_;
	  float apd_rms_;
	  float apd_nevt_;
	  bool flag_;
	  varVector_->getValByTime( t_, ME::iTPAPD_MEAN, apd_, flag_ );
	  varVector_->getValByTime( t_, ME::iTPAPD_RMS,  apd_rms_, flag_ );
	  varVector_->getValByTime( t_, ME::iTPAPD_NEVT, apd_nevt_, flag_ );
	  if( apd_<100  
	      || (apd_rms_==0 || apd_rms_>500) 
	      || (apd_nevt_<400 || apd_nevt_>2000) )
	    {
	      varVector_->setFlag( t_, ME::iTPAPD_MEAN, false );
	      varVector_->setFlag( t_, ME::iTPAPD_RMS, false );
	      varVector_->setFlag( t_, ME::iTPAPD_M3, false );
	      varVector_->setFlag( t_, ME::iTPAPD_NEVT, false );
	    }
	}
    } 
}

void
MERunManager::setFlag( MEChannel* leaf_, ME::Time t_, int ivar_, bool flag_ )
{
  if( leaf_==0 ) return;
  MEVarVector* varVector_ = apdVector( leaf_ );
  varVector_->setFlag( t_, ivar_, flag_ );
  for( unsigned int idau=0; idau<leaf_->n(); idau++ )
    {
      setFlag( leaf_->d(idau), t_, ivar_, flag_ );
    }
}

MEVarVector*
MERunManager::apdVector( MEChannel* leaf )
{
  if( _apdMap.count( leaf ) !=0 ) return _apdMap[leaf];
  else
    return 0;
} 

MEVarVector*
MERunManager::mtqVector( MEChannel* leaf )
{
  if( _mtqMap.count( leaf ) !=0 ) return _mtqMap[leaf];
  else
    {
      cout<< "-- debug -- MERunManager::mtqVector empty"<< endl; 
      return 0;
    }
} 

MEVarVector*
MERunManager::pnVector( MEChannel* leaf, int ipn )
{
  assert( ipn>=0 && ipn<2 );
  if( _pnMap[ipn].count( leaf ) !=0 ) return _pnMap[ipn][leaf];
  else {
    cout<< "-- debug -- MERunManager::pnVector empty"<< endl; 
    return 0;
  }
}


void
MERunManager::refresh()
{
  for( MusEcal::RunIterator p=_runs.begin(); p!=_runs.end(); ++p )
    {
      p->second->closeLaserPrimFile();
    }
}
