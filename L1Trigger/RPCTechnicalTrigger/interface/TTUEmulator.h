// $Id: TTUEmulator.h,v 1.7 2009/08/19 15:04:01 aosorio Exp $
#ifndef TTUEMULATOR_H 
#define TTUEMULATOR_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/RPCInputSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUConfiguration.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RPCWheel.h"

#include "CondFormats/RPCObjects/interface/RBCBoardSpecs.h"
#include "CondFormats/RPCObjects/interface/TTUBoardSpecs.h"

#include <map>
#include <bitset>

/** @class TTUEmulator TTUEmulator.h
 *  
 *  This class performs the following tasks [ref 2]:
 *
 *
 *    - find a single or multiple muon tracks
 *    - find length of each track
 *    - produce a wheel level trigger
 *
 *  The default algorithm is implemented is TrackingAlg [ref 2].
 *
 *  ref 2: <EM>"A configurable Tracking Algorithm to detect cosmic muon
 *          tracks for the CMS-RPC based Technical Trigger", R.T.Rajan et al</EM>
 *
 *
 *  @author Andres Osorio
 *
 *  email: aosorio@uniandes.edu.co
 *
 *  @date   2008-10-15
 */

class TTUEmulator {
public: 
  /// Standard constructor
  TTUEmulator( ) { }; 
  
  TTUEmulator( int, int );
  
  TTUEmulator( int, const char *, const char *, int );
  
  TTUEmulator( int, const char *, const char * , const char *, int );
  
  virtual ~TTUEmulator( ); ///< Destructor
  
  bool initialise();
  
  void emulate();
  
  void processTtu( RPCInputSignal * );
  
  void processTtu( RPCInputSignal * , int );
  
  void printinfo();
  
  void setSpecifications( const TTUBoardSpecs *, const RBCBoardSpecs *);
  
  void clearTriggerResponse();
  
  int mode() {
    return m_mode;
  };
  
  void setmode(int mode) {
    m_mode = mode;
  };
  
  int line() {
    return m_line;
  };
  
  void SetLineId( int );
  
  void setSpecs();
  
  int m_maxWheels;
  
  RPCWheel * m_Wheels;
  std::bitset<2> m_trigger;
  std::map<int, std::bitset<2> > m_triggerBx;

  class TriggerResponse 
  {
  public:
    
    TriggerResponse() { m_bx = 0; m_wedge = 0; m_trigger.reset(); };
    ~TriggerResponse() {;};
    
    void setTriggerBits( int bx , const std::bitset<2> & inbits )
    {
      m_bx = bx;
      m_trigger = inbits;
    };
    
    void setTriggerBits( int bx , int wdg, const std::bitset<2> & inbits )
    {
      m_bx = bx;
      m_wedge = wdg;
      m_trigger = inbits;
    };
    
    int m_bx;
    int m_wedge;
    std::bitset<2> m_trigger;
    
  };
  
  std::vector<TriggerResponse*> m_triggerBxVec;
  
protected:
  
private:
  
  int m_id;
  int m_bx;
  int m_mode;
  int m_line;
  
  int * m_wheelIds;
  
  std::string m_logtype;
  
  TTUInput         * m_ttuin;
  
  TTUConfiguration * m_ttuconf;
  
  bool m_debug;


  
};
#endif // TTUEMULATOR_H
