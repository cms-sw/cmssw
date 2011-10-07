#define MERunManager_cxx

#include <errno.h>
#include <dirent.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <string>
using namespace std;
#include "MERunManager.hh"
#include "MERun.hh"
#include "MEVarVector.hh"
#include "../../interface/MEGeom.h"
#include "../../interface/MELaserPrim.h"
#include "../../interface/MEChannel.h"

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

      //cout << "DEBUG MERunManager::updateRunList after key: "<< rr<<"  "<<lb<<"  "<<tsb<< " "<< key << endl;

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
      // cout << "DEBUG MERunManager::updateRunList after store: "<< store << endl;
     

      if( store ) 
	{
	  
	  TString fname = ME::rootFileName( header, settings ); 
	  FILE *test;
	  
	  test = fopen( fname, "r" );
	  bool fileok=false;

	  if(test)
	    {
	      // cout << "File " << fname << " found." << endl;
	      
	      TFile testtfile( fname );
	      if(! testtfile.IsZombie()){ // JM
		fileok = true;
		store = true;
		//cout<<" file ok "<< endl;
	      }else{ 
		//Delete file (in case of corrupted streamer info, file will be regenerated)
		stringstream del;
		//cout<<" deleting " <<fname<< endl;
		del << "rm " <<fname;
		system(del.str().c_str());
		store = false;
	      }
	      testtfile.Close();
	      
	      fclose( test );
	      
	    }else{

	    cout<<" FILE NOT FOUND "<< fname<< " ! "<<endl;
	    
	    store = false;
	    TString path_ = _lmdataPath;
	    switch( _type )
	      {
	      case ME::iLaser:
		path_ += "Laser/";     break;
	      case ME::iTestPulse:
		path_ += "TestPulse/"; break;
	      case ME::iLED:
		path_ += "LED/";     break;
		//		case ME::iPedestal:
		//		  path_ += "Pedestal/";  break;
	      default:
		abort();
	      };

	    path_ += "Analyzed/"; // FIXME!...
	    path_ += header.rundir;
	    path_ += "/";
	    cout << path_ << endl;
	    cout << _primPath << endl;
	    MELaserPrim prim( header, settings, path_, fname );
	    cout<<" here ok "<<  endl;
	    if( prim.init_ok )
	      {
		cout << "Primitives for DCC=" << header.dcc << " Side=" << header.side; 
		cout << " Type=" << settings.type;
		if( settings.type==ME::iLaser || settings.type==ME::iLED) cout << " Color=" << settings.wavelength;
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
		    TFile testtfile( fname );
		    if(! testtfile.IsZombie()){ // JM
		      store = true;
		      cout<<" file is not zombie "<< endl;
		    }else{
		      cout<<" file is zombie "<< endl;
		      stringstream del;
		      del << "rm " <<fname;
		      //system(del.str().c_str());
		      store = false;
		    }
		    fclose( test );
		  }else{
		  cout << " file does not exists!"<< endl;
		}
		//cout<<" Hello1 "<< fname<<endl;
	      }else{
	      cout<< " prim_init_ok =false" << endl;
	    }
	    //cout<<" Hello2 "<< fname<<endl;
	  }
	  //cout<<" Before MERun "<< fname<<endl;
	  if( store ) aRun = new MERun( header, settings, fname );
	  //cout<<" After MERun "<< endl;
	
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
	  else if( _type==ME::iLED )
	    {
	      if( !fname_.Contains("LED") ) continue;
	      if( _color==ME::iBlue && !fname_.Contains("Blue") ) continue; 
	    }

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



  //
  // First, fill "APD" map
  // 
  
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
  else if( _type==ME::iLED )
    {
      table_=ME::iLmfLEDPrim;
      size_=ME::iSizeAPD;
    }
  
  vector< MEChannel* > listOfChan_;
  //
  // at the crystal level
  //
  listOfChan_.clear();
  tree()->getListOfDescendants( ME::iCrystal, listOfChan_ );

  int reg_ = tree()->getAncestor(0)->id(); // fixme
  
  cout << "Filling APD Maps (number of C=" << listOfChan_.size() << ")" << endl;
  
  for( MusEcal::RunIterator p=_runs.begin(); p!=_runs.end(); ++p )
    {

      MERun* run_ = p->second;
      ME::Time time = run_->time();
      
      for( unsigned int ichan=0; ichan<listOfChan_.size(); ichan++ )
	{

	  //if( ichan%25==0 ) cout << "." << flush;
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

	  if( _apdMap.count(leaf_)==0 )
	    {
	      _apdMap[leaf_] = new MEVarVector( size_ ); 
	    }

	  MEVarVector* varVector_ = _apdMap[leaf_];
	  
	  varVector_->addTime( time );
	  for( unsigned ii=0; ii<size_; ii++ )
	    {
	      float val = run_->getVal( table_, ii, ix, iy );
	      varVector_->setVal( time, ii, val );
	    }
	}
      
      run_->closeLaserPrimFile();
    }
  
  cout << "...done." << endl;


  //
  // At higher levels
  
  cout << "Filling APD Maps for other granularities" << endl;
  //cout << "  ( number of SC=" << listOfChan_.size() << ")"<< endl;

  for( MusEcal::RunIterator p=_runs.begin(); p!=_runs.end(); ++p )
    {
      MERun* run_ = p->second;
      ME::Time time = run_->time();

      for( int ig=ME::iSuperCrystal; ig>=ME::iLMRegion; ig-- )
	{
	  listOfChan_.clear();
	  if( ig==ME::iLMRegion ) listOfChan_.push_back( tree() );
	  else tree()->getListOfDescendants( ig, listOfChan_ );
	  
	  for( unsigned int ichan=0; ichan<listOfChan_.size(); ichan++ )
	    {
	      //	      cout << "." << flush;
	      MEChannel* leaf_ = listOfChan_[ichan];
	      if( _apdMap.count(leaf_)==0 )
		{
		  _apdMap[leaf_] = new MEVarVector( size_ ); 
		}
	      MEVarVector* varVector_ = _apdMap[leaf_];

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
	}
      run_->closeLaserPrimFile();
      
    }

  cout << "...done." << endl;

  
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
  else if( _type==ME::iLED )
    {
      table_=ME::iLmfLEDPnPrim;
      size_=ME::iSizePN;
    }

  listOfChan_.clear();
  tree()->getListOfDescendants( ME::iLMModule, listOfChan_ );


  cout << "Filling PN Maps (number of LMM=" << listOfChan_.size() << ")" << endl;
  
  for( MusEcal::RunIterator p=_runs.begin(); p!=_runs.end(); ++p )
    {
      MERun* run_ = p->second;
      ME::Time time = run_->time();
      for( unsigned int ichan=0; ichan<listOfChan_.size(); ichan++ )
	{
	  //      if( ichan ) cout << "." << flush;
	  MEChannel* leaf_ = listOfChan_[ichan];
	  int id_ = leaf_->id();
	  
	  for( int ipn=0; ipn<2; ipn++ )
	    {
	      
	      if( _pnMap[ipn].count(leaf_)==0 )
		{
		  _pnMap[ipn][leaf_] = new MEVarVector( size_ ); 
		}
	      MEVarVector* varVector_ = _pnMap[ipn][leaf_];
	      
	      varVector_->addTime( time );
	      for( unsigned jj=0; jj<size_; jj++ ) // loop on variables
		{
		  //float val=0;
		  float val = run_->getVal( table_, jj, id_, ipn );
		  varVector_->setVal( time, jj, val );
		}
	    }
	}
      run_->closeLaserPrimFile();
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


  for( MusEcal::RunIterator p=_runs.begin(); p!=_runs.end(); ++p )
    {
      MERun* run_ = p->second;
      ME::Time time = run_->time();
      for( unsigned int ichan=0; ichan<listOfChan_.size(); ichan++ )
	{
	  MEChannel* leaf_ = listOfChan_[ichan];
	  
	  for( int ipn=0; ipn<2; ipn++ ){
	    
	    if( _pnMap[ipn].count(leaf_)==0 )
	      {
		_pnMap[ipn][leaf_] = new MEVarVector( size_ ); 
	      }
	    MEVarVector* varVector_ = _pnMap[ipn][leaf_];
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
	  //_pnMap[ipn][leaf_] = varVector_;
	}
      
      run_->closeLaserPrimFile();
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
  for( MusEcal::RunIterator p=_runs.begin(); p!=_runs.end(); ++p )
    {
      MERun* run_ = p->second;
      ME::Time time = run_->time();
      for( unsigned int ichan=0; ichan<listOfChan_.size(); ichan++ )
	{
	  MEChannel* leaf_ = listOfChan_[ichan];
	  int id_ = leaf_->id();
	  
	  if( _mtqMap.count(leaf_)==0 )
	    {
	      _mtqMap[leaf_] = new MEVarVector( size_ ); 
	    }
	  MEVarVector* varVector_ = _mtqMap[leaf_];
	  varVector_->addTime( time );
	  for( unsigned jj=0; jj<size_; jj++ )
	    {
	      float val = run_->getVal( table_, jj, id_ );
	      varVector_->setVal( time, jj, val );
	    }
	}
      
      run_->closeLaserPrimFile();
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
  else if( _type==ME::iLED )
    {
      setLEDFlags(); // LEDFIXME CHECK THIS
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

      MEChannel* mtqleaf_=leaf_;
     
      if(mtqleaf_->ig()>ME::iLMRegion){
	while( mtqleaf_->ig() != ME::iLMRegion){
	  mtqleaf_=mtqleaf_->m();
	}
      }

      MEChannel* pnleaf_=leaf_;
      if(pnleaf_->ig()>ME::iLMModule){
	while( pnleaf_->ig() != ME::iLMModule){
	  pnleaf_=pnleaf_->m();
	}
      }
      
      MEVarVector* varVectorMtq_ = mtqVector( mtqleaf_ );

      vector< ME::Time > time;
      varVector_->getTime( time );
       
      for( unsigned int itime=0; itime<time.size(); itime++ )
	{
	  ME::Time t_ = time[itime];

	  float apd_;
          float apd_rms_;
          float apd_m3_;
          float apdopn_rms_;
          float apdopn_mean_;
          float apdopn_m3_;
          float apd_time_;
          float apd_nevt_;
          //float alpha_;
          //float beta_;
          bool  flag_;
	  float apd_rms_norm_;
          float apdopn_rms_norm_;
	  float mtqfwhm_;
	  bool flagmtq_;
	  float scapd_;

          varVector_->getValByTime( t_, ME::iAPD_MEAN, apd_, flag_ );
          varVector_->getValByTime( t_, ME::iAPD_M3, apd_m3_, flag_ );
          varVector_->getValByTime( t_, ME::iAPD_OVER_PN_RMS, apdopn_rms_, flag_ );
          varVector_->getValByTime( t_, ME::iAPD_OVER_PN_MEAN, apdopn_mean_, flag_ );
          varVector_->getValByTime( t_, ME::iAPD_OVER_PN_M3, apdopn_m3_, flag_ );
          varVector_->getValByTime( t_, ME::iAPD_SHAPE_COR, scapd_, flag_ );

          varVector_->getValByTime( t_, ME::iAPD_RMS, apd_rms_, flag_ );
          varVector_->getValByTime( t_, ME::iAPD_TIME_MEAN, apd_time_, flag_ );
          varVector_->getValByTime( t_, ME::iAPD_NEVT, apd_nevt_, flag_ );

	  //varVector_->getValByTime( t_, ME::iAPD_ALPHA, alpha_, flag_);
          //varVector_->getValByTime( t_, ME::iAPD_BETA, beta_, flag_);


	  varVectorMtq_->getValByTime( t_, ME::iMTQ_FWHM, mtqfwhm_, flagmtq_ );  


	  if (apdopn_mean_==0) apdopn_rms_norm_=0;
	  else  apdopn_rms_norm_=apdopn_rms_/apdopn_mean_;
	  
	  if (apd_==0) apd_rms_norm_=0;
	  else  apd_rms_norm_=apd_rms_/apd_;
	  

          // FIXME: hardcoded cuts

          double Cut_apd_rms_norm[2];
          double Cut_apd_m3[2];
          double Cut_apdopn_m3[2];
          double Cut_apdopn_rms_norm;
          double Cut_apd[2];
          double Cut_apd_time[2];
          double Cut_apd_nevt[2];
	  // double Cut_ab[2];
          double Cut_fwhm[2];
          double Cut_sc_apd[2];


	  
	  
          Cut_apd[0]=200.0;
          Cut_apd[1]=4000.0; // gain prbs with high amplitudes
          //Cut_apd_rms[0]=5.0;
          //Cut_apd_rms[1]=200.0; //100
          //Cut_apdopn_rms=0.05; // 0.05
          Cut_apd_rms_norm[0]=0.005; // MIN 0.5%
          Cut_apd_rms_norm[1]=0.10;   // MAX 10%
          Cut_apdopn_rms_norm=0.015;  // MAX  3%

	  // different cuts for endcaps eventually:

	  if( ME::isBarrel(_lmr) ){

	    Cut_sc_apd[0]=0.8;
	    Cut_sc_apd[1]=0.9;

	  }else{
	  
	    Cut_sc_apd[0]=0.7;
	    Cut_sc_apd[1]=0.95;
	  }



	  if(_color==0){
	    Cut_apd_m3[0]=0.0; // MIN 0%
	    Cut_apd_m3[1]=0.4;   // MAX 40%
	    Cut_apdopn_m3[0]=0.0; // MIN 0%
	    Cut_apdopn_m3[1]=0.4;   // MAX 40%
	  }else{
	    Cut_apd_m3[0]=0.0; // MIN 0%
	    Cut_apd_m3[1]=0.8;   // MAX 40%
	    Cut_apdopn_m3[0]=0.0; // MIN 0%
	    Cut_apdopn_m3[1]=0.8;   // MAX 40%	    
	  }
          Cut_apd_time[0]=3.0;  // FIXME: put back 4.0 when time is fixed...
          Cut_apd_time[1]=8.0;  
          Cut_apd_nevt[0]=100.0;
          Cut_apd_nevt[1]=2000.0;
	  //Cut_ab[0]=1.5;
	  //Cut_ab[1]=3.5;
	  Cut_fwhm[0]=20.0;
	  Cut_fwhm[1]=45.0;

          if( ( apd_<  Cut_apd[0] || apd_>  Cut_apd[1])
              || (apdopn_rms_norm_>Cut_apdopn_rms_norm )
              || (apd_rms_norm_<Cut_apd_rms_norm[0] || apd_rms_norm_>Cut_apd_rms_norm[1])
              || (apd_time_<Cut_apd_time[0] || apd_time_>Cut_apd_time[1])
              || (apd_nevt_<Cut_apd_nevt[0] || apd_nevt_>Cut_apd_nevt[1]) 
	      // || (alpha_*beta_<Cut_ab[0] || alpha_*beta_>Cut_ab[1])
              || (TMath::Abs(apd_m3_)<Cut_apd_m3[0] || TMath::Abs(apd_m3_)>Cut_apd_m3[1]) 
              || (TMath::Abs(apdopn_m3_)<Cut_apdopn_m3[0] || TMath::Abs(apdopn_m3_)>Cut_apdopn_m3[1]) 
	      || ( mtqfwhm_<Cut_fwhm[0]|| mtqfwhm_>Cut_fwhm[1] )
	      || flag_==false 
	      )
	    {
	      
              varVector_->setFlag( t_, ME::iAPD_MEAN, false );
              varVector_->setFlag( t_, ME::iAPD_RMS, false );
              varVector_->setFlag( t_, ME::iAPD_M3, false );
              varVector_->setFlag( t_, ME::iAPD_NEVT, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNA_MEAN, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNA_RMS, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNA_M3, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNA_NEVT, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNB_MEAN, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNB_RMS, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNB_M3, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNB_NEVT, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PN_MEAN, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PN_RMS, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PN_M3, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PN_NEVT, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_APDA_MEAN, false ); 
              varVector_->setFlag( t_, ME::iAPD_OVER_APDA_RMS, false ); 
              varVector_->setFlag( t_, ME::iAPD_OVER_APDA_M3, false ); 
              varVector_->setFlag( t_, ME::iAPD_OVER_APDA_NEVT, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_APDB_MEAN, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_APDB_RMS, false ); 
              varVector_->setFlag( t_, ME::iAPD_OVER_APDB_M3, false ); 
              varVector_->setFlag( t_, ME::iAPD_OVER_APDB_NEVT, false );
              varVector_->setFlag( t_, ME::iAPD_TIME_MEAN, false );
              varVector_->setFlag( t_, ME::iAPD_TIME_RMS, false );
              varVector_->setFlag( t_, ME::iAPD_TIME_M3, false );
              varVector_->setFlag( t_, ME::iAPD_TIME_NEVT, false );
            
	      varVector_->setFlag( t_, ME::iAPD_OVER_PNACOR_MEAN, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNACOR_RMS, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNACOR_M3, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNACOR_NEVT, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNBCOR_MEAN, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNBCOR_RMS, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNBCOR_M3, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNBCOR_NEVT, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNCOR_MEAN, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNCOR_RMS, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNCOR_M3, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNCOR_NEVT, false );

//               varVector_->setFlag( t_, ME::iAPDABFIT_OVER_PNACOR_MEAN, false );
//               varVector_->setFlag( t_, ME::iAPDABFIT_OVER_PNACOR_RMS, false );
//               varVector_->setFlag( t_, ME::iAPDABFIT_OVER_PNACOR_M3, false );
//               varVector_->setFlag( t_, ME::iAPDABFIT_OVER_PNACOR_NEVT, false );
//               varVector_->setFlag( t_, ME::iAPDABFIT_OVER_PNBCOR_MEAN, false );
//               varVector_->setFlag( t_, ME::iAPDABFIT_OVER_PNBCOR_RMS, false );
//               varVector_->setFlag( t_, ME::iAPDABFIT_OVER_PNBCOR_M3, false );
//               varVector_->setFlag( t_, ME::iAPDABFIT_OVER_PNBCOR_NEVT, false );
//               varVector_->setFlag( t_, ME::iAPDABFIT_OVER_PNCOR_MEAN, false );
//               varVector_->setFlag( t_, ME::iAPDABFIT_OVER_PNCOR_RMS, false );
//               varVector_->setFlag( t_, ME::iAPDABFIT_OVER_PNCOR_M3, false );
//               varVector_->setFlag( t_, ME::iAPDABFIT_OVER_PNCOR_NEVT, false );

//               varVector_->setFlag( t_, ME::iAPDABFIX_OVER_PNACOR_MEAN, false );
//               varVector_->setFlag( t_, ME::iAPDABFIX_OVER_PNACOR_RMS, false );
//               varVector_->setFlag( t_, ME::iAPDABFIX_OVER_PNACOR_M3, false );
//               varVector_->setFlag( t_, ME::iAPDABFIX_OVER_PNACOR_NEVT, false );
//               varVector_->setFlag( t_, ME::iAPDABFIX_OVER_PNBCOR_MEAN, false );
//               varVector_->setFlag( t_, ME::iAPDABFIX_OVER_PNBCOR_RMS, false );
//               varVector_->setFlag( t_, ME::iAPDABFIX_OVER_PNBCOR_M3, false );
//               varVector_->setFlag( t_, ME::iAPDABFIX_OVER_PNBCOR_NEVT, false );
//               varVector_->setFlag( t_, ME::iAPDABFIX_OVER_PNCOR_MEAN, false );
//               varVector_->setFlag( t_, ME::iAPDABFIX_OVER_PNCOR_RMS, false );
//               varVector_->setFlag( t_, ME::iAPDABFIX_OVER_PNCOR_M3, false );
//               varVector_->setFlag( t_, ME::iAPDABFIX_OVER_PNCOR_NEVT, false );
		
	    }
	  
	  if( scapd_<  Cut_sc_apd[0] || scapd_>  Cut_sc_apd[1]){
      
	    varVector_->setFlag( t_, ME::iAPD_SHAPE_COR, false );
	  }

	  
         //  if ( ( alpha_*beta_<Cut_ab[0] || alpha_*beta_>Cut_ab[1] ) )	       
// 	    {
// 	      varVector_->setFlag( t_, ME::iAPD_ALPHA,false);
// 	      varVector_->setFlag( t_, ME::iAPD_BETA,false);
//               varVector_->setFlag( t_, ME::iAPDABFIT_OVER_PNACOR_MEAN, false );
//               varVector_->setFlag( t_, ME::iAPDABFIT_OVER_PNACOR_RMS, false );
//               varVector_->setFlag( t_, ME::iAPDABFIT_OVER_PNACOR_M3, false );
//               varVector_->setFlag( t_, ME::iAPDABFIT_OVER_PNACOR_NEVT, false );
//               varVector_->setFlag( t_, ME::iAPDABFIT_OVER_PNBCOR_MEAN, false );
//               varVector_->setFlag( t_, ME::iAPDABFIT_OVER_PNBCOR_RMS, false );
//               varVector_->setFlag( t_, ME::iAPDABFIT_OVER_PNBCOR_M3, false );
//               varVector_->setFlag( t_, ME::iAPDABFIT_OVER_PNBCOR_NEVT, false );
//               varVector_->setFlag( t_, ME::iAPDABFIT_OVER_PNCOR_MEAN, false );
//               varVector_->setFlag( t_, ME::iAPDABFIT_OVER_PNCOR_RMS, false );
//               varVector_->setFlag( t_, ME::iAPDABFIT_OVER_PNCOR_M3, false );
//               varVector_->setFlag( t_, ME::iAPDABFIT_OVER_PNCOR_NEVT, false );
	      	      
// 	    }
        }
    }
  
  listOfChan_.clear();
  tree()->getListOfDescendants( ME::iLMModule, listOfChan_ );

  double Cut_pn[2];
  double Cut_sc_pn[2];
  Cut_pn[0]=200.0;
  Cut_pn[1]=4000.0; 

  if( ME::isBarrel(_lmr) ){    
    Cut_sc_pn[0]=0.8;
    Cut_sc_pn[1]=1.0;    
  }else{
    Cut_sc_pn[0]=0.7;
    Cut_sc_pn[1]=1.0;
  }

  // FIXME: problem with EE+ scpn...  
  if(_lmr>=73 || _lmr<=82){
    Cut_sc_pn[0]=0.;
    Cut_sc_pn[1]=10.0;	    
  }
  

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
	      float scpn_;
	      bool flag_;
	      varVector_->getValByTime( t_, ME::iPN_SHAPE_COR, scpn_, flag_ );
	      varVector_->getValByTime( t_, ME::iPN_MEAN, pn_, flag_ );
	      
	      if( pn_<Cut_pn[0] || pn_>Cut_pn[1] )  
		{
		  varVector_->setFlag( t_, ME::iPN_MEAN, false );
		  varVector_->setFlag( t_, ME::iPN_RMS, false );
		  varVector_->setFlag( t_, ME::iPN_M3, false );
		  
		  if( ipn==0 )
		    {
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNA_MEAN, false );
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNA_RMS, false );
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNA_M3, false );
		      
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNACOR_MEAN, false );
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNACOR_RMS, false );
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNACOR_M3, false );
		      
		      // setFlag( leaf_, t_, ME::iAPDABFIT_OVER_PNACOR_MEAN, false );
// 		      setFlag( leaf_, t_, ME::iAPDABFIT_OVER_PNACOR_RMS, false );
// 		      setFlag( leaf_, t_, ME::iAPDABFIT_OVER_PNACOR_M3, false );
		      
// 		      setFlag( leaf_, t_, ME::iAPDABFIX_OVER_PNACOR_MEAN, false );
// 		      setFlag( leaf_, t_, ME::iAPDABFIX_OVER_PNACOR_RMS, false );
// 		      setFlag( leaf_, t_, ME::iAPDABFIX_OVER_PNACOR_M3, false );
		    }
		  if( ipn==1 )
		    {
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNB_MEAN, false );
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNB_RMS, false );
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNB_M3, false );
		      
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNBCOR_MEAN, false );
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNBCOR_RMS, false );
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNBCOR_M3, false );
		      
		     //  setFlag( leaf_, t_, ME::iAPDABFIT_OVER_PNBCOR_MEAN, false );
// 		      setFlag( leaf_, t_, ME::iAPDABFIT_OVER_PNBCOR_RMS, false );
// 		      setFlag( leaf_, t_, ME::iAPDABFIT_OVER_PNBCOR_M3, false );
		      
// 		      setFlag( leaf_, t_, ME::iAPDABFIX_OVER_PNBCOR_MEAN, false );
// 		      setFlag( leaf_, t_, ME::iAPDABFIX_OVER_PNBCOR_RMS, false );
// 		      setFlag( leaf_, t_, ME::iAPDABFIX_OVER_PNBCOR_M3, false );
		    }
		  
		  setFlag( leaf_, t_, ME::iAPD_OVER_PN_MEAN, false );
		  setFlag( leaf_, t_, ME::iAPD_OVER_PN_RMS, false );
		  setFlag( leaf_, t_, ME::iAPD_OVER_PN_M3, false );
		  
		  setFlag( leaf_, t_, ME::iAPD_OVER_PNCOR_MEAN, false );
		  setFlag( leaf_, t_, ME::iAPD_OVER_PNCOR_RMS, false );
		  setFlag( leaf_, t_, ME::iAPD_OVER_PNCOR_M3, false );
		  
		  // setFlag( leaf_, t_, ME::iAPDABFIT_OVER_PNCOR_MEAN, false );
// 		  setFlag( leaf_, t_, ME::iAPDABFIT_OVER_PNCOR_RMS, false );
// 		  setFlag( leaf_, t_, ME::iAPDABFIT_OVER_PNCOR_M3, false );
		  
// 		  setFlag( leaf_, t_, ME::iAPDABFIX_OVER_PNCOR_MEAN, false );
// 		  setFlag( leaf_, t_, ME::iAPDABFIX_OVER_PNCOR_RMS, false );
// 		  setFlag( leaf_, t_, ME::iAPDABFIX_OVER_PNCOR_M3, false );
	
		}
	      
	      if ( scpn_<  Cut_sc_pn[0] || scpn_>  Cut_sc_pn[1]){
		setFlag( leaf_, t_, ME::iPN_SHAPE_COR, false );
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
	    varVectorMtq_->setFlag( t_, ME::iMTQ_FW10, false );
	    varVectorMtq_->setFlag( t_, ME::iMTQ_FW05, false );
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
	      || (apd_nevt_<180. || apd_nevt_>2000) )
	    {
	      varVector_->setFlag( t_, ME::iTPAPD_MEAN, false );
	      varVector_->setFlag( t_, ME::iTPAPD_RMS, false );
	      varVector_->setFlag( t_, ME::iTPAPD_M3, false );
	      varVector_->setFlag( t_, ME::iTPAPD_NEVT, false );
	    }
	}
    } 
  
  listOfChan_.clear();
  tree()->getListOfDescendants( ME::iLMModule, listOfChan_ );
  
  //cout << "Setting TPPN Quality Flags " << endl;

  int n=8;
  double cutStep=0.001;
  double cutSlope=0.0001;

  for( unsigned int ichan=0; ichan<listOfChan_.size(); ichan++ )
    {
      MEChannel* leaf_ = listOfChan_[ichan];
      for( int ipn=0; ipn<2; ipn++ )
	{

	  MEVarVector* varVector_ = pnVector( leaf_, ipn );
	  vector< ME::Time > time;
	  vector< bool  > flagPN; 
	  vector< float > valPN;
	  
	  varVector_->getTime( time );
	  varVector_->getValAndFlag( ME::iTPPN_MEAN, time, valPN, flagPN );
	  
	  for (unsigned int i=0;i<valPN.size()-n;i++){
	    
	    std::pair<ME::Time, ME::Time> badInt;
	    vector<float> delta8;
	    vector<ME::Time> time8;
	    double slopes=0;
	    bool fourneg=true;
	    
	    for (int j=0;j<n;j++){
	      int k=i+j;
	      delta8.push_back( 2.0*(valPN[k+1]-valPN[k])/ (valPN[k+1]+valPN[k]) );
	      time8.push_back(time[k]);
	      if(j>0){
		slopes+=0.5*(valPN[k+1]-valPN[k])/(valPN[k+1]+valPN[k]);
		if(valPN[k+1]-valPN[k]>0) fourneg=false;
	      }
	    } 
	  
	    if( delta8[0] > cutStep && TMath::Abs(slopes)>cutSlope ){
	     
	      for (int ifl=0;ifl<n;ifl++){
		varVector_->setFlag( time[i+ifl],  ME::iTPPN_MEAN, false); 
		varVector_->setFlag( time[i+ifl],  ME::iTPPN_RMS, false); 
		varVector_->setFlag( time[i+ifl],  ME::iTPPN_M3, false); 

	      }
	      i+=n-1;      
	    }else if(valPN[i]<1. || flagPN[i]==false){
	      
	      varVector_->setFlag( time[i],  ME::iTPPN_MEAN, false); 
	      varVector_->setFlag( time[i],  ME::iTPPN_RMS, false); 
	      varVector_->setFlag( time[i],  ME::iTPPN_M3, false); 	      
	      
	    }
	    
	  }
	}      
    }
}

void
MERunManager::setLEDFlags()
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

      
      MEChannel* pnleaf_=leaf_;
      if(pnleaf_->ig()>ME::iLMModule){
	while( pnleaf_->ig() != ME::iLMModule){
	  pnleaf_=pnleaf_->m();
	}
      }
      

      vector< ME::Time > time;
      varVector_->getTime( time );
       
      for( unsigned int itime=0; itime<time.size(); itime++ )
	{
	  ME::Time t_ = time[itime];

	  float apd_;
          float apd_rms_;
          float apd_nevt_;
	  float apdopn_rms_;
          float apdopn_mean_;
	  bool  flag_;
	  float apd_rms_norm_;
          float apdopn_rms_norm_;
	 
          varVector_->getValByTime( t_, ME::iAPD_MEAN, apd_, flag_ );
          varVector_->getValByTime( t_, ME::iAPD_OVER_PN_RMS, apdopn_rms_, flag_ );
          varVector_->getValByTime( t_, ME::iAPD_OVER_PN_MEAN, apdopn_mean_, flag_ );

          varVector_->getValByTime( t_, ME::iAPD_RMS, apd_rms_, flag_ );
          varVector_->getValByTime( t_, ME::iAPD_NEVT, apd_nevt_, flag_ );

	  if (apdopn_mean_==0) apdopn_rms_norm_=0;
	  else  apdopn_rms_norm_=apdopn_rms_/apdopn_mean_;
	  
	  if (apd_==0) apd_rms_norm_=0;
	  else  apd_rms_norm_=apd_rms_/apd_;
	  

          // FIXME: hardcoded cuts

          double Cut_apd_rms_norm[2];
          double Cut_apdopn_rms_norm;
          double Cut_apd[2];
	  double Cut_apd_nevt[2];

	  Cut_apd_nevt[0]=100;
	  Cut_apd_nevt[1]=1200;
	  
	  
          Cut_apd[0]=10.0;
          Cut_apd[1]=4000.0; 
          Cut_apd_rms_norm[0]=0.0; // MIN 0.5%
          Cut_apd_rms_norm[1]=0.5;   // MAX 50%
          Cut_apdopn_rms_norm=0.5;  // MAX  50%

	  // different cuts for endcaps eventually:


          if( ( apd_<  Cut_apd[0] || apd_>  Cut_apd[1])
              || (apdopn_rms_norm_>Cut_apdopn_rms_norm )
              || (apd_rms_norm_<Cut_apd_rms_norm[0] || apd_rms_norm_>Cut_apd_rms_norm[1])
	      || (apd_nevt_<Cut_apd_nevt[0] || apd_nevt_>Cut_apd_nevt[1]) 
	      || flag_==false )
	    {
	      
              varVector_->setFlag( t_, ME::iAPD_MEAN, false );
              varVector_->setFlag( t_, ME::iAPD_RMS, false );
              varVector_->setFlag( t_, ME::iAPD_M3, false );
              varVector_->setFlag( t_, ME::iAPD_NEVT, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNA_MEAN, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNA_RMS, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNA_M3, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNA_NEVT, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNB_MEAN, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNB_RMS, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNB_M3, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNB_NEVT, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PN_MEAN, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PN_RMS, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PN_M3, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PN_NEVT, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_APDA_MEAN, false ); 
              varVector_->setFlag( t_, ME::iAPD_OVER_APDA_RMS, false ); 
              varVector_->setFlag( t_, ME::iAPD_OVER_APDA_M3, false ); 
              varVector_->setFlag( t_, ME::iAPD_OVER_APDA_NEVT, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_APDB_MEAN, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_APDB_RMS, false ); 
              varVector_->setFlag( t_, ME::iAPD_OVER_APDB_M3, false ); 
              varVector_->setFlag( t_, ME::iAPD_OVER_APDB_NEVT, false );
              varVector_->setFlag( t_, ME::iAPD_TIME_MEAN, false );
              varVector_->setFlag( t_, ME::iAPD_TIME_RMS, false );
              varVector_->setFlag( t_, ME::iAPD_TIME_M3, false );
              varVector_->setFlag( t_, ME::iAPD_TIME_NEVT, false );
            
	      varVector_->setFlag( t_, ME::iAPD_OVER_PNACOR_MEAN, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNACOR_RMS, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNACOR_M3, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNACOR_NEVT, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNBCOR_MEAN, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNBCOR_RMS, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNBCOR_M3, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNBCOR_NEVT, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNCOR_MEAN, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNCOR_RMS, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNCOR_M3, false );
              varVector_->setFlag( t_, ME::iAPD_OVER_PNCOR_NEVT, false );

	    }
        }
    }
  
  listOfChan_.clear();
  tree()->getListOfDescendants( ME::iLMModule, listOfChan_ );

  double Cut_pn[2];
  double Cut_sc_pn[2];
  Cut_pn[0]=10.0;
  Cut_pn[1]=4000.0; 

  if( ME::isBarrel(_lmr) ){    
    Cut_sc_pn[0]=0.8;
    Cut_sc_pn[1]=1.0;    
  }else{
    Cut_sc_pn[0]=0.7;
    Cut_sc_pn[1]=1.0;
  }

  // FIXME: problem with EE+ scpn...  
  if(_lmr>=73 || _lmr<=82){
    Cut_sc_pn[0]=0.;
    Cut_sc_pn[1]=10.0;	    
  }
  

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
	      float scpn_;
	      bool flag_;
	      varVector_->getValByTime( t_, ME::iPN_SHAPE_COR, scpn_, flag_ );
	      varVector_->getValByTime( t_, ME::iPN_MEAN, pn_, flag_ );
	      
	      if( pn_<Cut_pn[0] || pn_>Cut_pn[1] )  
		{
		  varVector_->setFlag( t_, ME::iPN_MEAN, false );
		  varVector_->setFlag( t_, ME::iPN_RMS, false );
		  varVector_->setFlag( t_, ME::iPN_M3, false );
		  
		  if( ipn==0 )
		    {
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNA_MEAN, false );
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNA_RMS, false );
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNA_M3, false );
		      
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNACOR_MEAN, false );
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNACOR_RMS, false );
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNACOR_M3, false );
		      
// 		      setFlag( leaf_, t_, ME::iAPDABFIT_OVER_PNACOR_MEAN, false );
// 		      setFlag( leaf_, t_, ME::iAPDABFIT_OVER_PNACOR_RMS, false );
// 		      setFlag( leaf_, t_, ME::iAPDABFIT_OVER_PNACOR_M3, false );
		      
// 		      setFlag( leaf_, t_, ME::iAPDABFIX_OVER_PNACOR_MEAN, false );
// 		      setFlag( leaf_, t_, ME::iAPDABFIX_OVER_PNACOR_RMS, false );
// 		      setFlag( leaf_, t_, ME::iAPDABFIX_OVER_PNACOR_M3, false );
		    }
		  if( ipn==1 )
		    {
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNB_MEAN, false );
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNB_RMS, false );
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNB_M3, false );
		      
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNBCOR_MEAN, false );
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNBCOR_RMS, false );
		      setFlag( leaf_, t_, ME::iAPD_OVER_PNBCOR_M3, false );
		      
		    //   setFlag( leaf_, t_, ME::iAPDABFIT_OVER_PNBCOR_MEAN, false );
// 		      setFlag( leaf_, t_, ME::iAPDABFIT_OVER_PNBCOR_RMS, false );
// 		      setFlag( leaf_, t_, ME::iAPDABFIT_OVER_PNBCOR_M3, false );
		      
// 		      setFlag( leaf_, t_, ME::iAPDABFIX_OVER_PNBCOR_MEAN, false );
// 		      setFlag( leaf_, t_, ME::iAPDABFIX_OVER_PNBCOR_RMS, false );
// 		      setFlag( leaf_, t_, ME::iAPDABFIX_OVER_PNBCOR_M3, false );
		    }
		  
		  setFlag( leaf_, t_, ME::iAPD_OVER_PN_MEAN, false );
		  setFlag( leaf_, t_, ME::iAPD_OVER_PN_RMS, false );
		  setFlag( leaf_, t_, ME::iAPD_OVER_PN_M3, false );
		  
		  setFlag( leaf_, t_, ME::iAPD_OVER_PNCOR_MEAN, false );
		  setFlag( leaf_, t_, ME::iAPD_OVER_PNCOR_RMS, false );
		  setFlag( leaf_, t_, ME::iAPD_OVER_PNCOR_M3, false );
		  
		 //  setFlag( leaf_, t_, ME::iAPDABFIT_OVER_PNCOR_MEAN, false );
// 		  setFlag( leaf_, t_, ME::iAPDABFIT_OVER_PNCOR_RMS, false );
// 		  setFlag( leaf_, t_, ME::iAPDABFIT_OVER_PNCOR_M3, false );
		  
// 		  setFlag( leaf_, t_, ME::iAPDABFIX_OVER_PNCOR_MEAN, false );
// 		  setFlag( leaf_, t_, ME::iAPDABFIX_OVER_PNCOR_RMS, false );
// 		  setFlag( leaf_, t_, ME::iAPDABFIX_OVER_PNCOR_M3, false );
	
		}
	      
	      if ( scpn_<  Cut_sc_pn[0] || scpn_>  Cut_sc_pn[1]){
		setFlag( leaf_, t_, ME::iPN_SHAPE_COR, false );
	      }  
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
      //cout<< "-- debug -- MERunManager::mtqVector empty"<< endl; 
      return 0;
    }
} 

MEVarVector*
MERunManager::pnVector( MEChannel* leaf, int ipn )
{
  assert( ipn>=0 && ipn<2 );
  if( _pnMap[ipn].count( leaf ) !=0 ) return _pnMap[ipn][leaf];
  else {
    //cout<< "-- debug -- MERunManager::pnVector empty"<< endl; 
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

