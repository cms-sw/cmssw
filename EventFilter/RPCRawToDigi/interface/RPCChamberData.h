#ifndef RPCChamberData_h
#define RPCChamberData_h


/** \class RPCChamberData
 *
 * Unpacks RPC Chamber Data Record (needs pointer to beginning of buffer)
 *
 *  $Date: 2005/11/24 18:04:46 $
 *  $Revision: 1.4 $
 * \author Ilaria Segoni - CERN
 */
class RPCChamberData {

public:
  
  /// Constructor
  RPCChamberData(const unsigned int* index);

  /// Destructor
  virtual ~RPCChamberData() {};

  /// unpacked data access methods
  int partitionData();
  int halfP();
  int eod();
  int partitionNumber();
  int chamberNumber();

  static const int PARTITION_DATA_MASK  = 0XFF;
  static const int PARTITION_DATA_SHIFT =0;

  static const int HALFP_MASK = 0X1;
  static const int HALFP_SHIFT =8;

  static const int EOD_MASK = 0X1;
  static const int EOD_SHIFT =9;

  static const int PARTITION_NUMBER_MASK = 0XF;
  static const int PARTITION_NUMBER_SHIFT =10;

  static const int CHAMBER_MASK = 0X3;
  static const int CHAMBER_SHIFT =14;



private:

  const unsigned int * word_;
 
  int partitionData_;
  int halfP_;
  int eod_;
  int partitionNumber_;
  int chamberNumber_;
  
};




#endif
