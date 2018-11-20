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
  
  auto  initTrk =std::make_unique<Track>();
  
  //.
  runSeedBuster( inmap );
  
  if ( !m_initialseeds.empty() && m_initialseeds.size() < 20 ) // if too much hits, then cannot process
    initTrk->add( m_initialseeds[0].get() );
  else {
    initTrk->addnone();
    if( m_debug) std::cout << "TTUTrackingAlg>process() ends: no initialseeds" << std::endl;
    return false;
  }
  
  auto trk = m_tracks.emplace_back( std::move(initTrk) ).get();
  
  //..
  auto _seed = m_initialseeds.begin();
  std::vector<Seed*> neighbors;
  
  while ( _seed != m_initialseeds.end() ) {
    
    findNeighbors( (*_seed).get() , neighbors );
    filter( trk, neighbors );
    executeTracker( trk, neighbors );
    ghostBuster( trk );
    
    ++_seed;
    
    if ( _seed != m_initialseeds.end() ) {
      auto initTrk = std::make_unique<Track>();
      initTrk->add((*_seed).get());
      m_tracks.emplace_back( std::move(initTrk) );
      
    }

  }

  if( m_debug) { 
    std::cout << "Total tracks: " << m_tracks.size() << std::endl;
    for( auto& tr : m_tracks)
      std::cout << "length: " << tr->length() << '\t';
    std::cout << std::endl;
  }
  
  //...
  alignTracks();
  
  //.... Look at the first track and compare its track length
  
  int tracklen(0);
  auto itr = m_tracks.begin();
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
        m_initialseeds.emplace_back(std::make_unique<Seed>(idx, idy, 0) );
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
  
  auto _itr = neighbors.begin();
  
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
  
  auto _itr = m_initialseeds.begin();
  
  while( _itr != m_initialseeds.end() ) {
    
    int _difx    = std::abs( _xo - (*_itr)->m_sectorId );
    int _dify    = std::abs( _yo - (*_itr)->m_stationId );
    
    if (m_debug) std::cout << "difference (x,y): " << _difx << "," << _dify << "\t";
    
    if ( _difx == 11 ) _difx = 1;
    
    if ( ((_difx == 1) && (_dify == 1)) ||
         ((_difx == 1) && (_dify == 0)) ||
         ((_difx == 0) && (_dify == 1)) ) 
      
      neighbors.push_back( (*_itr).get() );
    
    ++_itr;
  }

  if (m_debug) std::cout << std::endl;
  
}

void TTUTrackingAlg::filter( Track * _trk, 
                             std::vector<Seed*> & _nbrs )
{
  
  //... filter: removes from neighbors list, seeds already present
  //...    in tracks

  for( auto _itr = _trk->m_seeds.begin();_itr != _trk->m_seeds.end(); ++_itr) 
  {
    auto _isalready = std::find( _nbrs.begin(),_nbrs.end(), (*_itr) );
    
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

  CompareMechanism<std::unique_ptr<Track>> compare;
  
  std::sort( m_tracks.begin(), m_tracks.end(), compare );
  std::reverse( m_tracks.begin(), m_tracks.end() );
  
  if( m_debug ) {
    for( auto& tr : m_tracks)
      std::cout << "Align tracks> trk len: " << tr->length() << " ";
    std::cout << std::endl;
  }
  
}

void TTUTrackingAlg::cleanUp()
{
  
  m_tracks.clear();
  m_initialseeds.clear();
  
}
