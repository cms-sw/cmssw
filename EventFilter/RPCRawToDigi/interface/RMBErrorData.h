#ifndef RMBErrorData_h
#define RMBErrorData_h


/** \file
 * Unpacks Information of RMB Discarded or Corrupted Data
 *
 *  $Date: 2005/11/07 15:44:09 $
 *  $Revision: 1.1 $
 * \author Ilaria Segoni - CERN
 */
class RMBErrorData{

public:
  
  /// Constructor
  RMBErrorData(const unsigned char* index);

  /// Destructor
  virtual ~RMBErrorData() {};

  /// unpacked data access methods
  int channel();
  int tbRmb();
  int chamber();

  static const int CHANNEL_MASK  = 0X1F;
  static const int CHANNEL_SHIFT =0;

  static const int TB_RMB_MASK = 0X3F;
  static const int TB_RMB_SHIFT =5;

  static const int CHAMBER_MASK = 0X3;
  static const int CHAMBER_SHIFT =14;



private:

  const unsigned int * word_;
 
  int channel_;
  int tbRmb_;
  int chamber_;
  
};




#endif
