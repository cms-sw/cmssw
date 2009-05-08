// $Id: TTUTrackingAlg.cc,v 1.1 2009/01/30 15:42:48 aosorio Exp $
// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/src/TTUTrackingAlg.h"
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
  
  m_mintrklength = 3;
  
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
bool TTUTrackingAlg::process( const TTUInput & _inmap )
{
  
  if( m_debug) std::cout << "TTUTrackingAlg::process starts" << std::endl;

  m_triggersignal = false;
  
  Track * _initTrk = new Track();
  
  //.
  runSeedBuster( _inmap );
  
  if ( m_initialseeds.size() > 0 ) 
    _initTrk->add( m_initialseeds[0] );
  else {
    _initTrk->addnone();
    return false;
  }
  
  m_tracks.push_back( _initTrk );
  
  //..
  SeedsItr _seed = m_initialseeds.begin();
  std::vector<Seed*> _neighbors;
  
  while ( _seed != m_initialseeds.end() ) {
  
    findNeighbors( (*_seed) , _neighbors );
    filter( _initTrk, _neighbors );
    executeTracker( _initTrk, _neighbors );
    
    ++_seed;
    
    if ( _seed != m_initialseeds.end() ) {
      _initTrk = new Track();
      _initTrk->add((*_seed));
      m_tracks.push_back( _initTrk );
      
    }
    
  }

  TracksItr itr;
  
  if( m_debug) { 
    std::cout << "Total tracks: " << m_tracks.size() << std::endl;
    for( itr = m_tracks.begin(); itr != m_tracks.end(); ++itr)
      std::cout << (*itr)->length() << '\t';
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
    std::cout << "TTUTrackingAlg " 
              << tracklen << '\t' 
              << m_triggersignal << std::endl;
  }
  
  //..... Clean up for next run
  
  cleanUp();
    
  //.......................................................
  
  if( m_debug ) std::cout << "TTUTrackingAlg>process ends" << std::endl;
  
  return false;
  
}

void TTUTrackingAlg::runSeedBuster( const TTUInput & _inmap )
{
  
  int _idx(0);
  int _idy(0);

  for(int i=0; i < 12; ++i) 
  {
    _idx = (m_SEscanorder[i] - 1);
    std::bitset<6> _station = _inmap.input_sec[_idx];
    
    if ( ! _station.any() ) continue;
    
    for(int k=0; k < 6; ++k ) {
      
      _idy = (m_STscanorder[k] - 1);
      bool _hit = _station[_idy];
      
      if ( _hit ) {
        Seed *_seed = new Seed( _idx, _idy, 0 );
        m_initialseeds.push_back(_seed);
      }
    }
  }
  
  //...

  
  
}

int TTUTrackingAlg::executeTracker( Track * _trk, 
                                    std::vector<Seed*> & _neighbors)
{
  
  if ( m_debug ) std::cout << "executeTracker: " << _neighbors.size() << std::endl;
  
  //...
  
  SeedsItr _itr = _neighbors.begin();
  
  while( _itr != _neighbors.end() ) {
  
    _trk->add( (*_itr) );
  
    std::vector<Seed*> _nextneighbors;
    
    findNeighbors( (*_itr) , _nextneighbors );
    
    filter( _trk, _nextneighbors );
    
    if ( _nextneighbors.size() == 1 ) 
      executeTracker( _trk, _nextneighbors );
    
    //... bifurcation not considered at the moment
    // if ( _itr != _neighbors.end() ) 
    //     {
    //       std::cout << "executeTracker> found a bifurcation" << std::endl;
    //       _trk = new Track( (*_trk) );
    //       m_tracks.push_back( _trk );
    //       std::vector<Seed*> _nextneighbors;
    //       findNeighbors( (*_itr) , _nextneighbors );
    //       filter( _trk, _nextneighbors );
    //       executeTracker( _trk, _nextneighbors );
    //     }
    
    ++_itr;
    
  }
  
  //...
  

  
  return 1;
  
}

void TTUTrackingAlg::findNeighbors( Seed  * _seed, 
                                    std::vector<Seed*> & _neighbors)
{
  
  _neighbors.clear();
  
  int _xo = _seed->m_sectorId;
  int _yo = _seed->m_stationId;

  if( m_debug ) std::cout << "X: " << _xo+1 << " Y: " << _yo+1 << std::endl;
  
  SeedsItr _itr = m_initialseeds.begin();
  
  while( _itr != m_initialseeds.end() ) {
    
    int _difx    = std::abs( _xo - (*_itr)->m_sectorId );
    int _dify    = std::abs( _yo - (*_itr)->m_stationId );
    
    if ( _difx == 11 ) _difx = 1;
    
    if ( (_difx == 1) && (_dify == 1) ||
         (_difx == 1) && (_dify == 0) ||
         (_difx == 0) && (_dify == 1)   ) 
      
      _neighbors.push_back( (*_itr) );
    
    ++_itr;
  }
  
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
    if( _isalready != _nbrs.end() ) _nbrs.erase( _isalready );
  }
  
}

void TTUTrackingAlg::alignTracks()
{

  TracksItr itr;
  CompareMechanism<Track> compare;
  
  std::sort( m_tracks.begin(), m_tracks.end(), compare );
  std::reverse( m_tracks.begin(), m_tracks.end() );
  
  if( m_debug ) {
    for( itr = m_tracks.begin(); itr != m_tracks.end(); ++itr )
      std::cout << (*itr)->length() << " ";
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
