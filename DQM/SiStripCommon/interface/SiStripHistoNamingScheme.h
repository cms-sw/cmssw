#ifndef DQM_SiStripCommon_SiStripHistoNamingScheme_H
#define DQM_SiStripCommon_SiStripHistoNamingScheme_H

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

  inline static string top() { return root_ + top_; }
  inline static string controlView() { return top_ + dir_ + controlView_; }
  inline static string readoutView() { return top_ + dir_ + readoutView_; }

  /** Returns reserved value (0xFFFF) to represent "NOT SPECIFIED". */
  inline static const uint16_t& all()   { return all_; }
  inline static const string& fedId()   { return fedId_; }
  inline static const string& fedCh()   { return fedChannel_; }
  inline static const string& gain()    { return gain_; }
  inline static const string& digital() { return digital_; }
  
  // ----- FORMULATION OF DIRECTORY PATHS -----

  /** Returns directory path in the form of a string, based on control
      params (FEC crate, slot and ring, CCU address and channel). */ 
  static string controlPath( uint16_t fec_crate = all_, 
			     uint16_t fec_slot  = all_, 
			     uint16_t fec_ring  = all_, 
			     uint16_t ccu_addr  = all_, 
			     uint16_t ccu_chan  = all_ );

  /** Returns control parameters in the form of a "ControlPath" struct,
      based on directory path string of the form
      ControlView/FecCrateA/FecSlotA/FecRingC/CcuAddrD/CcuChanE/. */
  static ControlPath controlPath( string path );
  
  /** Returns directory path in the form of a string, based on readout
      parameters (FED id and channel). */ 
  static string readoutPath( uint16_t fed_id = all_, 
			     uint16_t fed_channel = all_ );

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

 private: // ----- private data members -----

  // misc
  static const string root_;
  static const string top_;
  static const string dir_;
  static const string sep_;
  static const uint16_t all_; // 0xFFFF

  // control view
  static const string controlView_;
  static const string fecCrate_;
  static const string fecSlot_;
  static const string fecRing_;
  static const string ccuAddr_;
  static const string ccuChan_;
  
  // readout view
  static const string readoutView_;
  static const string fedId_;
  static const string fedChannel_;

  // detector view
  static const string detectorView_; //@@ necessary?

  // histo title
  static const string fedCabling_;
  static const string apvTiming_;
  static const string fedTiming_;
  static const string optoScan_;
  static const string vpspScan_;
  static const string pedestals_;
  static const string apvLatency_;
  static const string unknownTask_;
  
  // histo contents
  static const string sum2_;
  static const string sum_;
  static const string num_;
  static const string unknownContents_;

  // histo keys
  static const string fedKey_;
  static const string fecKey_;
  static const string detKey_; //@@ necessary?
  static const string unknownKey_;

  // granularity
  static const string lldChan_;
  static const string apvPair_;
  static const string apv_;
  static const string unknownGranularity_;
  
  // extra info
  static const string gain_;       // Opto scan
  static const string digital_;    // Opto scan
  
};

#endif // DQM_SiStripCommon_SiStripHistoNamingScheme_H


