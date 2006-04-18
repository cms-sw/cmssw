#ifndef DQM_SiStripCommon_SiStripHistoNamingScheme_H
#define DQM_SiStripCommon_SiStripHistoNamingScheme_H

#include "boost/cstdint.hpp"
#include <string>

using namespace std;

class SiStripHistoNamingScheme {
  
 public: // ----- public interface -----
 
  enum HistoType   { UNKNOWN_TYPE=0, NO_TYPE=1, SUM2=2, SUM=3, NUM=4 };
  enum KeyType     { UNKNOWN_KEY=0, NO_KEY=1, FED=2, FEC=3, DET=4 };
  enum Granularity { UNKNOWN_GRAN=0, MODULE=1, LLD_CHAN=2, APV_PAIR=3, APV=4 };
  
  /** Simple struct to hold control path parameters. */
  struct ControlPath {
    uint16_t fecCrate_;
    uint16_t fecSlot_;
    uint16_t fecRing_;
    uint16_t ccuAddr_;
    uint16_t ccuChan_;
  };
  
  /** Simple struct to hold control path parameters. */
  struct HistoName {
    string      histoTitle_;
    HistoType   histoType_;
    KeyType     keyType_;
    uint32_t    histoKey_;
    Granularity granularity_;
    uint16_t    channel_;
  };
 
  inline static string top() { return root_ + top_; }
  inline static string controlView() { return top_ + dir_ + controlView_; }
  inline static string readoutView() { return top_ + dir_ + readoutView_; }
  
  /** Returns reserved value (0xFFFF) to represent "NOT SPECIFIED". */
  inline static const uint16_t& all() { return all_; }
  
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
  
  /** Contructs histogram name based on a general histogram name,
      histogram contents, a histogram key and a channel id. */
  static string histoName( string      his_name, 
			   HistoType   his_type = SiStripHistoNamingScheme::NO_TYPE, 
			   KeyType     key_type = SiStripHistoNamingScheme::NO_KEY, 
			   uint32_t    his_key  = 0, 
			   Granularity gran     = SiStripHistoNamingScheme::MODULE,
			   uint16_t    channel  = 0 );

  /** Extracts various parameters from histogram name and returns the
      values in the form of a HistoName struct. */
  static HistoName histoName( string histo_name );  

 private: // ----- data members -----

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

  // histo contents
  static const string sum2_;
  static const string sum_;
  static const string num_;
  static const string unknownType_;

  // histo keys
  static const string fecKey_;
  static const string fedKey_;
  static const string detKey_; //@@ necessary?
  static const string unknownKey_;

  // granularity
  static const string lldChan_;
  static const string apvPair_;
  static const string apv_;
  static const string unknownGranularity_;

};

#endif // DQM_SiStripCommon_SiStripHistoNamingScheme_H


