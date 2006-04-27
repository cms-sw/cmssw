#ifndef DQM_SiStripCommon_SiStripHistoNamingScheme_H
#define DQM_SiStripCommon_SiStripHistoNamingScheme_H

#include "DQM/SiStripCommon/interface/SiStripHistoNamingConstants.h"
#include "boost/cstdint.hpp"
#include <string>

using namespace std;

class SiStripHistoNamingScheme {
  
 public:

  // ----- STRUCTS AND ENUMS -----

  enum Task        { UNKNOWN_TASK,     NO_TASK,  PHYSICS, FED_CABLING, APV_TIMING, FED_TIMING, OPTO_SCAN, VPSP_SCAN, PEDESTALS, APV_LATENCY };
  enum Contents    { UNKNOWN_CONTENTS, COMBINED, SUM2, SUM, NUM };
  enum KeyType     { UNKNOWN_KEY,      NO_KEY,   FED, FEC, DET };
  enum Granularity { UNKNOWN_GRAN,     MODULE,   LLD_CHAN, APV_PAIR, APV };

  /** Simple struct to hold components of histogram title. */
  struct HistoTitle {
    Task        task_;
    Contents    contents_;
    KeyType     keyType_;
    uint32_t    keyValue_;
    Granularity granularity_;
    uint16_t    channel_;
    string      extraInfo_;
  };
  
  /** Simple struct to hold control path parameters. */
  struct ControlPath {
    uint16_t fecCrate_;
    uint16_t fecSlot_;
    uint16_t fecRing_;
    uint16_t ccuAddr_;
    uint16_t ccuChan_;
  };
  
  // ----- METHODS RETURNING SOME GENERIC STRINGS AND CONSTANTS -----

  inline static string top() { return sistrip::root_ + sistrip::top_; }
  inline static string controlView() { return sistrip::top_ + sistrip::dir_ + sistrip::controlView_; }
  inline static string readoutView() { return sistrip::top_ + sistrip::dir_ + sistrip::readoutView_; }

/*   /\** Returns reserved value (0xFFFF) to represent "NOT SPECIFIED". *\/ */
/*   inline static const uint16_t& all()   { return sistrip::all_; } */
/*   inline static const string& fedId()   { return sistrip::fedId_; } */
/*   inline static const string& fedCh()   { return sistrip::fedChannel_; } */
/*   inline static const string& gain()    { return sistrip::gain_; } */
/*   inline static const string& digital() { return sistrip::digital_; } */
  
  // ----- FORMULATION OF DIRECTORY PATHS -----

  /** Returns directory path in the form of a string, based on control
      params (FEC crate, slot and ring, CCU address and channel). */ 
  static string controlPath( uint16_t fec_crate = sistrip::all_, 
			     uint16_t fec_slot  = sistrip::all_, 
			     uint16_t fec_ring  = sistrip::all_, 
			     uint16_t ccu_addr  = sistrip::all_, 
			     uint16_t ccu_chan  = sistrip::all_ );

  /** Returns control parameters in the form of a "ControlPath" struct,
      based on directory path string of the form
      ControlView/FecCrateA/FecSlotA/FecRingC/CcuAddrD/CcuChanE/. */
  static ControlPath controlPath( string path );
  
  /** Returns directory path in the form of a string, based on readout
      parameters (FED id and channel). */ 
  static string readoutPath( uint16_t fed_id = sistrip::all_, 
			     uint16_t fed_channel = sistrip::all_ );

  /** Returns readout parameters in the form of a pair (FED
      id/channel), based on directory path string of the form
      ReadoutView/FedIdX/FedChannelY/. */
  static pair<uint16_t,uint16_t> readoutPath( string path ) { return pair<uint16_t,uint16_t>(0,0); } //@@ NO IMPLEMENTATION YET!
  
  // ----- FORMULATION OF HISTOGRAM TITLES -----

  /** Contructs histogram name based on a general histogram name,
      histogram contents, a histogram key and a channel id. */
  static string histoTitle( Task        task,
			    Contents    contents   = SiStripHistoNamingScheme::COMBINED, 
			    KeyType     key_type   = SiStripHistoNamingScheme::NO_KEY, 
			    uint32_t    key_value  = 0, 
			    Granularity granarity  = SiStripHistoNamingScheme::MODULE,
			    uint16_t    channel    = 0,
			    string      extra_info = "" );

  /** Extracts various parameters from histogram name and returns the
      values in the form of a HistoTitle struct. */
  static HistoTitle histoTitle( string histo_title );  

  static string task( Task task );
  static string contents( Contents contents );
  static string keyType( KeyType key_type );
  static string granularity( Granularity Granularity );
  static Task task( string task );
  static Contents contents( string contents );
  static KeyType keyType( string key_type );
  static Granularity granularity( string granularity );

};

#endif // DQM_SiStripCommon_SiStripHistoNamingScheme_H


