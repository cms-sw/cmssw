#ifndef ChamberData_h
#define ChamberData_h

/** \file
 *
 *  $Date: 2005/10/21 16:45:41 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
  */

#define PARTITION_DATA_MASK  = 0X7
#define PARTITION_DATA_SHIFT =0

#define HALFP_MASK = 0X100
#define HALFP_SHIFT =8

#define EOD_MASK = 0X1
#define EOD_SHIFT =9

#define PARTITION_NUMBER_MASK = 0XF
#define PARTITION_NUMBER_SHIFT =10

#define CHAMBER_MASK = 0X3
#define CHAMBER_SHIFT =14


/** \class ChamberData
 * Unpacks RPC Chamber Data
 *
 *  $Date: 2005/10/21 16:43:07 $
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

private:

  const unsigned int * word_;
 
  int partitionData_;
  int halfP_;
  int eod_;
  int partitionNumber_;
  int chamberNumber_;
  
};




#endif
