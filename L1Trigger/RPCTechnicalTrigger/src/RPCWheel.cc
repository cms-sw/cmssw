// $Id: RPCWheel.cc,v 1.11 2009/08/09 11:11:37 aosorio Exp $
// Include files



// local
#include "L1Trigger/RPCTechnicalTrigger/interface/RPCWheel.h"

//-----------------------------------------------------------------------------
// Implementation file for class : RPCWheel
//
// 2008-10-15 : Andres Osorio
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
RPCWheel::RPCWheel() {
  
  m_id = 0; 
  
  m_maxrbc     = 6;
  m_maxlayers  = 6;
  m_maxsectors = 12;
  
  m_debug = false;

  m_sec1id.push_back(12);
  m_sec2id.push_back(1);
  m_sec1id.push_back(2);
  m_sec2id.push_back(3);
  m_sec1id.push_back(4);
  m_sec2id.push_back(5);
  m_sec1id.push_back(6);
  m_sec2id.push_back(7);
  m_sec1id.push_back(8);
  m_sec2id.push_back(9);
  m_sec1id.push_back(10);
  m_sec2id.push_back(11);

  m_wheelmap = new std::bitset<6>[12];
    
}

void RPCWheel::setProperties( int wid ) {
  
  m_id = wid;
  
  int bisector[2];
  
  for( int k=0; k < m_maxrbc; ++k )
  {
    bisector[0]= m_sec1id[k];
    bisector[1]= m_sec2id[k];
    m_RBCE.push_back( new RBCEmulator( ) ); 
    m_RBCE[k]->setid( wid, bisector );
  }

  for( int k=0; k < m_maxsectors; ++k)
    m_wheelmap[k].reset();
  
}


void RPCWheel::setProperties( int wid, const char * logic_type) {
  
  m_id = wid;
  
  int bisector[2];
  
  for( int k=0; k < m_maxrbc; ++k )
  {
    bisector[0]= m_sec1id[k];
    bisector[1]= m_sec2id[k];
    m_RBCE.push_back( new RBCEmulator( logic_type ) ); 
    m_RBCE[k]->setid( wid, bisector );
  }

  for( int k=0; k < m_maxsectors; ++k)
    m_wheelmap[k].reset();
  
}

void RPCWheel::setProperties( int wid, const char * f_name, const char * logic_type) {
  
  m_id = wid; 
  
  int bisector[2];
  
  for( int k=0; k < m_maxrbc; ++k )
  {
    bisector[0]= (k*2)+1;
    bisector[1]= (k*2)+2;
    m_RBCE.push_back( new RBCEmulator( f_name, logic_type ) ); 
    m_RBCE[k]->setid( wid, bisector );
  }
  
  for( int k=0; k < m_maxsectors; ++k)
    m_wheelmap[k].reset();
  
}

//=============================================================================
// Destructor
//=============================================================================
RPCWheel::~RPCWheel() {
  
  //destroy all rbc objects associated
  std::vector<RBCEmulator*>::iterator itr;
  for( itr = m_RBCE.begin(); itr != m_RBCE.end(); ++itr)
    if ( (*itr) ) delete (*itr);
  
  m_RBCE.clear();
  m_sec1id.clear();
  m_sec2id.clear();

  if ( m_wheelmap ) delete[] m_wheelmap;
    
} 

//=============================================================================
void RPCWheel::setSpecifications( const RBCBoardSpecs * rbcspecs )
{
  
  for( int k=0; k < m_maxrbc; ++k )
    m_RBCE[k]->setSpecifications( rbcspecs ); 
  
}

bool RPCWheel::initialise()
{

  bool status(false);
  for( int k=0; k < m_maxrbc; ++k )
    status = m_RBCE[k]->initialise();
  return status;
  
}

void RPCWheel::emulate() 
{
  //This is a test emulation
  for( int k=0; k < m_maxrbc; ++k )
  {
    m_RBCE[k]->emulate();
  }
  
}

bool RPCWheel::process( int bx, const std::map<int,RBCInput*> & data )
{
  
  int bxsign(1);
  bool status(false);
  
  std::map<int,RBCInput*>::const_iterator itr;
  
  if ( bx != 0 ) bxsign = ( bx / abs(bx) );
  else bxsign = 1;
  
  for(int k=0; k < m_maxrbc; ++k) {
    
    m_RBCE[k]->reset();
    
    int key = bxsign*( 1000000 * abs(bx)
                       + m_RBCE[k]->m_rbcinfo->wheelIdx()*10000 
                       + m_RBCE[k]->m_rbcinfo->sector(0)*100
                       + m_RBCE[k]->m_rbcinfo->sector(1) );
    
    itr = data.find( key );
    
    if ( itr != data.end() )  {
      
      if ( ! (*itr).second->hasData )  { 
        status |= false;
        continue;
      } else {
        if( m_debug ) std::cout << "RPCWheel::process> found data at: " 
                                <<  key << '\t' 
                                << ( itr->second ) << std::endl;
        m_RBCE[k]->emulate( ( itr->second ) );
        status |= true;
      }
      
    } else {
      //if( m_debug ) std::cout << "RPCWheel::process> position not found: " <<  key << std::endl;
      status |= false;
    }
    
  }
  
  return status;
  
}

bool RPCWheel::process( int bx, const std::map<int,TTUInput*> & data )
{
  
  int bxsign(1);
  bool status(false);
  
  std::map<int,TTUInput*>::const_iterator itr;
  
  if ( bx != 0 ) bxsign = ( bx / abs(bx) );
  else bxsign = 1;
  
  int key = bxsign*( 1000000 * abs(bx) + (m_id+2)*10000 );
  
  itr = data.find( key );
  
  if ( itr != data.end() )  {
    if( m_debug ) std::cout << "RPCWheel::process> found data at: " <<  key << '\t' 
                            << ( itr->second ) << std::endl;

    if ( ! (*itr).second->m_hasHits ) return false;
    
    for( int k=0; k < m_maxsectors; ++k ) {
      m_wheelmap[k]     = (*itr).second->input_sec[k];
      status = true;
    }

  } else {
    //if( m_debug ) std::cout << "RPCWheel::process> position not found: " <<  key << std::endl;
    status = false;
  }
  
  return status;
  
}


//.............................................................................

void RPCWheel::createWheelMap()
{
  
  m_rbcDecision.reset();
  
  std::bitset<6> layersignal;
  
  layersignal       = * m_RBCE[0]->getlayersignal( 0 );
  m_wheelmap[11]     = layersignal;
  
  m_rbcDecision.set( 11 , m_RBCE[0]->getdecision( 0 ) );
  
  for( int k=0; k < (m_maxrbc-1); ++k )
  {
    layersignal             = * m_RBCE[k+1]->getlayersignal( 0 );
    m_wheelmap[(k*2)+1]     = layersignal;
    layersignal             = * m_RBCE[k+1]->getlayersignal( 1 );
    m_wheelmap[(k*2)+2]     = layersignal;
    
    m_rbcDecision.set( (k*2)+1  , m_RBCE[k+1]->getdecision( 0 ) );
    m_rbcDecision.set( (k*2)+2  , m_RBCE[k+1]->getdecision( 1 ) );
    
  }
  
  layersignal       = * m_RBCE[0]->getlayersignal( 1 );
  m_wheelmap[0]     = layersignal;
  
  m_rbcDecision.set( 0 , m_RBCE[0]->getdecision( 1 ) );
  
  if( m_debug ) std::cout << "RPCWheel::createWheelMap done" << std::endl;
  
}

void RPCWheel::retrieveWheelMap( TTUInput & output ) 
{
  
  if( m_debug ) std::cout << "RPCWheel::retrieveWheelMap starts" << std::endl;
  output.reset();
  
  for(int i=0; i < m_maxsectors; ++i ) {
    for( int j=0; j < m_maxlayers; ++j ) 
    {
      output.input_sec[i].set(j, m_wheelmap[i][j]);
    }
  }
  
  output.m_wheelId = m_id;
  
  output.m_rbcDecision = m_rbcDecision;
    
  if( m_debug ) print_wheel( output );
  if( m_debug ) std::cout << "RPCWheel::retrieveWheelMap done" << std::endl;
  
}

//=============================================================================


void RPCWheel::printinfo() 
{
  
  std::cout << "Wheel -> " << m_id << '\n';
  for( int k=0; k < m_maxrbc; ++k )
    m_RBCE[k]->printinfo();
  
}

void RPCWheel::print_wheel(const TTUInput & wmap )
{

  std::cout << "RPCWheel::print_wheel> " << wmap.m_wheelId << '\t' << wmap.m_bx << std::endl;
  
  for( int i=0; i < m_maxsectors; ++i) std::cout << '\t' << (i+1);
  std::cout << std::endl;
  
  for( int k=0; k < m_maxlayers; ++k )
  {
    std::cout << (k+1) << '\t';
    for( int j=0; j < m_maxsectors; ++j)
      std::cout << wmap.input_sec[j][k] << '\t';
    std::cout << std::endl;
  }
  
}

//=============================================================================
