#ifndef ChamberData_h
#define ChamberData_h


/** \file
 * Unpacks RPC Chamber Data
 *
 *  $Date: 2005/10/21 11:00:01 $
 *  $Revision: 1.1 $
 * \author Ilaria Segoni - CERN
 */
class ChamberData {

public:
  
  /// Constructor
  ChamberData(const unsigned char* index);

  /// Destructor
  virtual ~ChamberData() {};

  /// unpacked data access methods
  int partitionData();
  int halfP();
  int eod();
  int partitionNumber();
  int chamberNumber();

  static const int PARTITION_DATA_MASK  = 0X7;
  static const int PARTITION_DATA_SHIFT =0;

  static const int HALFP_MASK = 0X100;
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
