#ifndef RPCRecord_h
#define RPCRecord_h

/** \file
 *
 *  Class that finds whether record contains chamber data or a control word
 *
 *  $Date: 2005/10/21 11:00:18 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
  */

class RPCRecord {

public:
  
  /// Constructor
  RPCRecord(const unsigned char* index): 
    word_(reinterpret_cast<const unsigned int*>(index)) {};

  /// Destructor
  virtual ~RPCRecord() {};

  /// List of DT DDU Word Types
  enum recordTypes {
    	ChamberData = 1,
    	StartOfChannelData = 2,
    	EmptyWord = 3,
        RMBDiscardedData =4,
	SLinkDiscardedData=5,
        UndefinedType =6
  };


  /// Record type getter 
  enum recordTypes type(); 
   
   /// Record Unpacker 
  void recordUnpack(enum recordTypes); 
    
   /// Go to beginning of next Record (16 bits jump) 
  void next(); 
 

  /// Control bits definitions
  static const int  MaxChamberFlag  = 2;
  static const int  controlWordFlag = 3;
  
  static const int  StartOfChannelDataFlag        = 7;
  static const int  EmptyWordOrSLinkDiscardedFlag = 5;
  static const int  RMBDiscardedDataFlag          = 6;


  static const int  EmptyWordFlag                 = 0;
  static const int  SLinkDiscardedDataFlag        = 1;
 
 
  static const int RPC_RECORD_BIT_SIZE =16;

  static const int RECORD_TYPE_MASK = 0X3;
  static const int RECORD_TYPE_SHIFT =14;

  static const int CONTROL_TYPE_MASK = 0X7;
  static const int CONTROL_TYPE_SHIFT =11;

  static const int EMPTY_OR_SLDISCARDED_MASK  = 0X1;
  static const int EMPTY_OR_SLDISCARDED_SHIFT =0;


private:

  const unsigned int * word_;
  


};




#endif
