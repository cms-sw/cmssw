#ifndef RPCRecord_h
#define RPCRecord_h

/** \file
 *
 *  Class that finds whether record contains chamber data or a control word
 *
 *  $Date: 2005/10/21 16:45:41 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
  */

#define RPC_RECORD_BIT_SIZE =16

#define RECORD_TYPE_MASK = 0X3
#define RECORD_TYPE_SHIFT =14


/** \class RPCRecords
 * //scrivere cosa fa sta classe
 *
 *  $Date: 2005/10/21 16:43:07 $
 *  $Revision: 1.1 $
 * \author Ilaria Segoni - CERN
 */
class RPCRecord {

public:
  
  /// Constructor
  RPCRecord(const unsigned char* index);

  /// Destructor
  virtual ~RPCRecord() {};

  /// List of DT DDU Word Types
  enum recordTypes {
    	DataChamber = 1,
    	Control = 2,
        UndefinedType =3
  };


  /// Record type getter 
  enum recordTypes type(); 
    
   /// Record Unpacker 
  void recordUnpack(enum recordTypes); 
    
   /// Go to beginning of next Record (16 bits jump) 
  void next(); 
 

  /// Control bits definitions
  static const unsigned int  chamberZeroFlag = 0;
  static const unsigned int  chamberOneFlag  = 1;
  static const unsigned int  chamberTwoFlag  = 2;
  static const unsigned int  controlWordFlag = 3;



private:

  const unsigned int * word_;
  
};




#endif
