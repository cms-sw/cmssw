// $Id: TTUTrackingAlg.h,v 1.5 2009/08/09 11:11:36 aosorio Exp $
#ifndef TTUTRACKINGALG_H 
#define TTUTRACKINGALG_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/TTULogic.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUInput.h"

#include <iostream>
#include <vector>

/** @class TTUTrackingAlg TTUTrackingAlg.h
 *  
 *  Tracking Algorithm [ref 2]
 *
 *  ref 2: <EM>"A configurable Tracking Algorithm to detect cosmic muon
 *          tracks for the CMS-RPC based Technical Trigger", R.T.Rajan et al</EM>
 *
 *
 *  @author Andres Osorio
 *
 *  email: aosorio@uniandes.edu.co
 *
 *  @date   2008-10-18
 */
class TTUTrackingAlg : public TTULogic {
public: 
  /// Standard constructor
  TTUTrackingAlg( ); 
  
  virtual ~TTUTrackingAlg( ); ///< Destructor

  //... from TTULogic interface:
  
  bool process( const TTUInput & );
  
  void setBoardSpecs( const TTUBoardSpecs::TTUBoardConfig & );
  
  //...
  
  class Seed 
  {
  public:
    
    Seed() {m_sectorId = -9; m_stationId = -1; m_tkLength = 0;};
    Seed( int _seId, int _stId, int _tl)
    {
      m_sectorId  = _seId;
      m_stationId = _stId;
      m_tkLength  = _tl;
    };
  
    ~Seed() {};
    
    Seed( const Seed & _seed) 
    {
      m_sectorId  = _seed.m_sectorId;
      m_stationId = _seed.m_stationId;
      m_tkLength  = _seed.m_tkLength;
    };
    
    bool operator==(const Seed & rhs) 
    {
      return (m_sectorId == rhs.m_sectorId) 
        && (m_stationId == rhs.m_stationId);
    };
    
    int m_sectorId;
    int m_stationId;
    int m_tkLength;
    
  };
  
  class Track
  {
  public:
    
    Track() { m_tracklength = 0; };
    ~Track() { 
      if ( m_tracklength < 0 ) delete m_seeds[0];
      m_seeds.clear();
    };
    
    Track( const Track & trk ) 
    {
      m_seeds = trk.m_seeds;
      m_tracklength = trk.m_tracklength;
    };
    
    void add( Seed * sd ) { 
      m_seeds.push_back(sd); 
      ++m_tracklength;
    };
    
    void addnone() { 
      Seed *_sd = new Seed(0,0,0);
      m_seeds.push_back(_sd); 
      m_tracklength = -1;
    };

    void updateTrkLength() {
      m_tracklength = m_seeds.size();
    };
    
    bool operator<(const Track &rhs) {
      return m_tracklength < rhs.m_tracklength;
    };
    
    int length() { return m_tracklength;};
    
    std::vector<Seed*> m_seeds;
    
  private:
    
    int m_tracklength;
        
  };
  
  typedef std::vector<Seed*>::iterator SeedsItr;
  typedef std::vector<Track*>::iterator TracksItr;

  void setMinTrkLength( int val ) 
  {
    m_mintrklength = val;
  };
  
  template< class T>
  struct CompareMechanism
  {
    bool operator()( T* a, T* b ) { return (*a) < (*b) ; }
  };
    
protected:
  
private:
  
  void runSeedBuster( const TTUInput & );
  
  void findNeighbors( Seed * , std::vector<Seed*> & );
  
  int  executeTracker( Track *, std::vector<Seed*> & );
  
  void filter( Track * , std::vector<Seed*>  & );

  void ghostBuster( Track * );
    
  void alignTracks();

  void cleanUp();
    
  int m_STscanorder[6];
  
  int m_SEscanorder[12];

  int m_mintrklength;

  std::vector<Track*> m_tracks;
  
  std::vector<Seed*>  m_initialseeds;
  
  struct CompareSeeds {
    bool operator()( const Seed * a, const Seed * b )
    {
      //std::cout << (*a).m_sectorId << " " << (*b).m_sectorId << " " 
      //<< (*a).m_stationId << " " << (*b).m_stationId << std::endl;
      return ((*a).m_sectorId == (*b).m_sectorId ) && ((*a).m_stationId == (*b).m_stationId );
    }
  };
  
  struct SortBySector {
    bool operator()( const Seed * a, const Seed * b )
    {
      return ((*a).m_sectorId <= (*b).m_sectorId );
    }
  };
  
  struct SortByLayer {
    bool operator()( const Seed * a, const Seed * b )
    {
      return ((*a).m_stationId <= (*b).m_stationId );
      
    }
  };



  inline void print( const std::vector<Seed*> & seeds ) 
  {
    std::vector<Seed*>::const_iterator itr;
    for( itr = seeds.begin(); itr != seeds.end(); ++itr)
      std::cout << (*itr) << '\t';
    std::cout << '\n';
  };

  bool m_debug;
    
};
#endif // TTUTRACKINGALG_H
