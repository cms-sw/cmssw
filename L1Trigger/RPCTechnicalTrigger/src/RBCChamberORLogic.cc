// Include files 


// local
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCChamberORLogic.h"

//-----------------------------------------------------------------------------
// Implementation file for class : RBCChamberORLogic
// 
// 2008-10-11 : Andres Osorio
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
RBCChamberORLogic::RBCChamberORLogic(  ) {
  m_rbname.reserve(13);
  m_rbname.emplace_back("RB1InFw");
  m_rbname.emplace_back("RB1OutFw");
  m_rbname.emplace_back("RB22Fw");
  m_rbname.emplace_back("RB23Fw");
  m_rbname.emplace_back("RB23M");
  m_rbname.emplace_back("RB3Fw");
  m_rbname.emplace_back("RB4Fw");
  m_rbname.emplace_back("RB1InBk");
  m_rbname.emplace_back("RB1OutBk");
  m_rbname.emplace_back("RB22Bk");
  m_rbname.emplace_back("RB23Bk");
  m_rbname.emplace_back("RB3Bk");
  m_rbname.emplace_back("RB4Bk");
  
  itr2names itr = m_rbname.begin();
  
  while ( itr != m_rbname.end() )
  {
    m_chamber.insert( make_pair( (*itr) , 0 ) );
    ++itr;
  }

  m_maxcb    = 13;
  m_maxlevel = 3; // 1 <= m <= 6

}

//=============================================================================

void RBCChamberORLogic::process( const RBCInput & _input, std::bitset<2> & _decision ) 
{
  
  bool status(false);
  //std::cout << "RBCChamberORLogic> Working with chambers OR logic ..." << '\n';

  m_layersignal[0].reset();
  m_layersignal[1].reset();

  for (int k=0; k < 2; ++k ) 
  {
    
    if( _input.needmapping )
      this->createmap( _input.input_sec[k] );
    else
      this->copymap  ( _input.input_sec[k] );
    
    status = this->evaluateLayerOR( "RB1InFw"  , "RB1InBk" );
    m_layersignal[k].set( 0 , status);
    
    status = this->evaluateLayerOR( "RB1OutFw" , "RB1OutBk" );
    m_layersignal[k].set( 1 , status);
    
    //... RB2
    //... wheel -2,+2 RB2IN divided in 2 eta partitions, RB2OUT in 3 eta
    //... wheel -1, 0, +1 RB2IN divided in 3 eta partitions, RB2OUT in 2 eta

    if ( abs( _input.wheelId() ) >= 2 ) {
      
      status = this->evaluateLayerOR( "RB22Fw"   , "RB22Bk" );
      m_layersignal[k].set( 2 , status);
      
      bool rb23FB = this->evaluateLayerOR( "RB23Fw"   , "RB23Bk" );
      bool rb23MF = this->evaluateLayerOR( "RB23Fw"   , "RB23M" );
      bool rb23MB = this->evaluateLayerOR( "RB23M"    , "RB23Bk" );
      
      status = rb23FB || rb23MF || rb23MB;
    
      m_layersignal[k].set( 3 , status );
      
    } else {
      
      status = this->evaluateLayerOR( "RB22Fw"   , "RB22Bk" );
      m_layersignal[k].set( 3 , status);
      
      bool rb23FB = this->evaluateLayerOR( "RB23Fw"   , "RB23Bk" );
      bool rb23MF = this->evaluateLayerOR( "RB23Fw"   , "RB23M" );
      bool rb23MB = this->evaluateLayerOR( "RB23M"    , "RB23Bk" );
      
      status = rb23FB || rb23MF || rb23MB;
    
      m_layersignal[k].set( 2 , status );
      
    }
    
    //.......
    
    status = this->evaluateLayerOR( "RB3Fw"    , "RB3Bk" );
    m_layersignal[k].set( 4 , status);
    
    status = this->evaluateLayerOR( "RB4Fw"    , "RB4Bk" );
    m_layersignal[k].set( 5 , status);
    
    reset();
    
    //... apply now majority level criteria:
    
    int _majority = int(m_layersignal[k].count());
    
    if ( _majority >= m_maxlevel) _decision[k] = true;
    else _decision[k] = false;
    
  }

  //...all done!
  
}

void RBCChamberORLogic::setBoardSpecs( const RBCBoardSpecs::RBCBoardConfig & specs )
{
  
  m_maxlevel = specs.m_MayorityLevel;
    
}

void RBCChamberORLogic::copymap( const std::bitset<15> & _input ) 
{
  
  m_chamber[m_rbname[0]]  = _input[0];
  m_chamber[m_rbname[1]]  = _input[1];
  m_chamber[m_rbname[2]]  = _input[2];
  m_chamber[m_rbname[3]]  = _input[3];
  m_chamber[m_rbname[4]]  = _input[4];
  m_chamber[m_rbname[5]]  = _input[5];
  m_chamber[m_rbname[6]]  = _input[6];
  m_chamber[m_rbname[7]]  = _input[7];
  m_chamber[m_rbname[8]]  = _input[8];
  m_chamber[m_rbname[9]]  = _input[9];
  m_chamber[m_rbname[10]] = _input[10];
  m_chamber[m_rbname[11]] = _input[11];
  m_chamber[m_rbname[12]] = _input[12];
  
}

void RBCChamberORLogic::createmap( const std::bitset<15> & _input ) 
{
  
  m_chamber[m_rbname[0]]  = _input[3];
  m_chamber[m_rbname[1]]  = _input[4];
  m_chamber[m_rbname[2]]  = _input[5];
  m_chamber[m_rbname[3]]  = _input[8];
  m_chamber[m_rbname[4]]  = _input[7];
  m_chamber[m_rbname[5]]  = _input[11];
  m_chamber[m_rbname[6]]  = _input[12] || _input[14];
  m_chamber[m_rbname[7]]  = _input[0];
  m_chamber[m_rbname[8]]  = _input[1];
  m_chamber[m_rbname[9]]  = _input[2];
  m_chamber[m_rbname[10]] = _input[6];
  m_chamber[m_rbname[11]] = _input[9];
  m_chamber[m_rbname[12]] = _input[10] || _input[13];
  
}

void RBCChamberORLogic::reset() 
{
  
  //... Reset map for next sector analysis
  m_chamber.clear();
  
  itr2names itr = m_rbname.begin();
  
  while ( itr != m_rbname.end() )
  {
    m_chamber.insert( make_pair( (*itr) , 0 ) );
    ++itr;
  }
  
  //m_layersignal[0].reset();
  //m_layersignal[1].reset();
  
}

bool RBCChamberORLogic::evaluateLayerOR(const char * _chA, const char *_chB )
{
  
  itr2chambers ptr1 = m_chamber.find( std::string(_chA) );
  itr2chambers ptr2 = m_chamber.find( std::string(_chB) );
  
  if ( ptr1 == m_chamber.end() || ptr2 == m_chamber.end() ) {
    //handle error...
    std::cout << "RBCChamberORLogic> Cannot find a chamber name" << '\n';
    return false;
  }
  
  return ( ptr1->second || ptr2->second );
  
}

