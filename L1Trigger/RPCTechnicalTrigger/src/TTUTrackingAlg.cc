// $Id: TTUTrackingAlg.cc,v 1.10 2009/10/27 09:01:48 aosorio Exp $
// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUTrackingAlg.h"
#include <algorithm>

//-----------------------------------------------------------------------------
// Implementation file for class : TTUTrackingAlg
//
// 2008-10-18 : Andres Osorio
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
TTUTrackingAlg::TTUTrackingAlg(  ) {

  int StOrder[6]={6,5,4,3,2,1};
  int SeOrder[12]={4,5,3,6,2,7,1,8,12,9,11,10};
  
  for(int i=0; i < 6; ++i) 
  {
    m_STscanorder[i]       = StOrder[i];
    m_SEscanorder[i*2]     = SeOrder[i*2];
    m_SEscanorder[(i*2)+1] = SeOrder[(i*2)+1];
  }
  
  m_triggersignal = false;
  m_mintrklength = 4;

  m_debug = false;
  
}

//=============================================================================
// Destructor
//=============================================================================
TTUTrackingAlg::~TTUTrackingAlg() {

  TracksItr itr1;
  for (itr1=m_tracks.begin(); itr1!=m_tracks.end(); ++itr1)
    delete (*itr1);
  
  SeedsItr itr2;
  for (itr2=m_initialseeds.begin(); itr2!=m_initialseeds.end(); ++itr2)
    delete (*itr2);
  
  //m_tracks.clear();
  //m_initialseeds.clear();
  
} 

//=============================================================================
void TTUTrackingAlg::setBoardSpecs( const TTUBoardSpecs::TTUBoardConfig & boardspecs ) 
{
  
  m_mintrklength = boardspecs.m_TrackLength;
  
}

bool TTUTrackingAlg::process( const TTUInput & inmap )
{
  
  if( m_debug) std::cout << "TTUTrackingAlg>process() starts + bx= " << inmap.m_bx << std::endl;

  m_triggersignal = false;
  
  Track * initTrk = new Track();
  
  //.
  runSeedBuster( inmap );
  
  if ( m_initialseeds.size() > 0 && m_initialseeds.size() < 20 ) // if too much hits, then cannot process
    initTrk->add( m_initialseeds[0] );
  else {
    initTrk->addnone();
    if( m_debug) std::cout << "TTUTrackingAlg>process() ends: no initialseeds" << std::endl;
    return false;
  }
  




  m_tracks.push_back( initTrk );
  
  //..
  SeedsItr _seed = m_initialseeds.begin();
  std::vector<Seed*> neighbors;
  
  while ( _seed != m_initialseeds.end() ) {
    
    findNeighbors( (*_seed) , neighbors );
    filter( initTrk, neighbors );
    executeTracker( initTrk, neighbors );
    ghostBuster( initTrk );
    
    ++_seed;
    
    if ( _seed != m_initialseeds.end() ) {
      initTrk = new Track();
      initTrk->add((*_seed));
      m_tracks.push_back( initTrk );
      
    }

  }

  TracksItr itr;
  
  if( m_debug) { 
    std::cout << "Total tracks: " << m_tracks.size() << std::endl;
    for( itr = m_tracks.begin(); itr != m_tracks.end(); ++itr)
      std::cout << "length: " << (*itr)->length() << '\t';
    std::cout << std::endl;
  }
  
  //...
  alignTracks();
  
  //.... Look at the first track and compare its track length
  
  int tracklen(0);
  itr = m_tracks.begin();
  if ( itr != m_tracks.end() ) tracklen = (*itr)->length();
  
  if ( tracklen >= m_mintrklength )
    m_triggersignal = true;
  
  if( m_debug ) {
    std::cout << "TTUTrackingAlg> trk len= " 
              << tracklen << '\t' << "triggered: "
              << m_triggersignal << std::endl;
  }
  
  //..... Clean up for next run
  
  cleanUp();
    
  //.......................................................
  
  if( m_debug ) std::cout << "TTUTrackingAlg>process ends" << std::endl;
  
  return true;
  
}

void TTUTrackingAlg::runSeedBuster( const TTUInput & inmap )
{
  
  int idx(0);
  int idy(0);

  for(int i=0; i < 12; ++i) 
  {
    idx = (m_SEscanorder[i] - 1);
    std::bitset<6> station = inmap.input_sec[idx];
    
    if ( ! station.any() ) continue;
    
    for(int k=0; k < 6; ++k ) {
      
      idy = (m_STscanorder[k] - 1);
      bool _hit = station[idy];
      
      if ( _hit ) {
        Seed *_seed = new Seed( idx, idy, 0 );
        m_initialseeds.push_back(_seed);
      }
    }
  }
  
  //...
  if ( m_debug ) std::cout << "SeedBuster: " << m_initialseeds.size() << std::endl;
    
}

int TTUTrackingAlg::executeTracker( Track * _trk, std::vector<Seed*> & neighbors)
{
  
  if ( m_debug ) std::cout << "executeTracker: " << neighbors.size() << std::endl;
  
  //...
  
  SeedsItr _itr = neighbors.begin();
  
  while( _itr != neighbors.end() ) {
  
    _trk->add( (*_itr) );
    
    std::vector<Seed*> _nextneighbors;
    
    findNeighbors( (*_itr) , _nextneighbors );
    
    filter( _trk, _nextneighbors );
    
    if ( _nextneighbors.size() == 1 ) 
      executeTracker( _trk, _nextneighbors );
    
    //... bifurcation not considered at the moment
        
    ++_itr;
    
  }
  
  //...
  

  
  return 1;
  
}

void TTUTrackingAlg::findNeighbors( Seed  * _seed, std::vector<Seed*> & neighbors)
{
  
  neighbors.clear();
  
  int _xo = _seed->m_sectorId;
  int _yo = _seed->m_stationId;

  if( m_debug ) std::cout << "X: " << _xo+1 << " Y: " << _yo+1 << std::endl;
  
  SeedsItr _itr = m_initialseeds.begin();
  
  while( _itr != m_initialseeds.end() ) {
    
    int _difx    = std::abs( _xo - (*_itr)->m_sectorId );
    int _dify    = std::abs( _yo - (*_itr)->m_stationId );
    
    if (m_debug) std::cout << "difference (x,y): " << _difx << "," << _dify << "\t";
    
    if ( _difx == 11 ) _difx = 1;
    
    if ( ((_difx == 1) && (_dify == 1)) ||
         ((_difx == 1) && (_dify == 0)) ||
         ((_difx == 0) && (_dify == 1)) ) 
      
      neighbors.push_back( (*_itr) );
    
    ++_itr;
  }

  if (m_debug) std::cout << std::endl;
  
}

void TTUTrackingAlg::filter( Track * _trk, 
                             std::vector<Seed*> & _nbrs )
{
  
  //... filter: removes from neighbors list, seeds already present
  //...    in tracks

  SeedsItr _itr;
  
  for( _itr = _trk->m_seeds.begin();_itr != _trk->m_seeds.end(); ++_itr) 
  {
    SeedsItr _isalready = std::find( _nbrs.begin(),_nbrs.end(), (*_itr) );
    
    if( _isalready != _nbrs.end() ) { 
      _nbrs.erase( _isalready ); 
      if( m_debug ) std::cout << "removing ..." << std::endl;
    }
    
    
  }
  
}

void TTUTrackingAlg::ghostBuster( Track * currentTrk )
{
  
  //...do a final check to make sure there are no repeated seeds in track
  
  std::vector<Seed*>::iterator seedItr;
  
  std::sort( currentTrk->m_seeds.begin(), currentTrk->m_seeds.end(), SortBySector() );
  std::sort( currentTrk->m_seeds.begin(), currentTrk->m_seeds.end(), SortByLayer() );
  
  seedItr = std::unique (currentTrk->m_seeds.begin(), currentTrk->m_seeds.end(), CompareSeeds() );
  
  currentTrk->m_seeds.resize(seedItr - currentTrk->m_seeds.begin());

  currentTrk->updateTrkLength();
    
}

void TTUTrackingAlg::alignTracks()
{

  TracksItr itr;
  CompareMechanism<Track> compare;
  
  std::sort( m_tracks.begin(), m_tracks.end(), compare );
  std::reverse( m_tracks.begin(), m_tracks.end() );
  
  if( m_debug ) {
    for( itr = m_tracks.begin(); itr != m_tracks.end(); ++itr )
      std::cout << "Align tracks> trk len: " << (*itr)->length() << " ";
    std::cout << std::endl;
  }
  
}

void TTUTrackingAlg::cleanUp()
{
  
  TracksItr itr1;
  for (itr1=m_tracks.begin(); itr1!=m_tracks.end(); ++itr1)
    delete (*itr1);
  
  SeedsItr itr2;
  for (itr2=m_initialseeds.begin(); itr2!=m_initialseeds.end(); ++itr2)
    delete (*itr2);
  
  m_tracks.clear();
  m_initialseeds.clear();
  
}
