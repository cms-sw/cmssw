#ifndef StartOfBXData_h
#define StartOfBXData_h


/** \file
 * BX number
 *
 *  $Date: 2005/11/07 15:43:50 $
 *  $Revision: 1.1 $
 * \author Ilaria Segoni - CERN
 */
class StartOfBXData {

public:
  
  /// Constructor
  StartOfBXData(const unsigned char* index);

  /// Destructor
  virtual ~StartOfBXData() {};

  /// unpacked data access methods
  int bx();

  static const int BX_MASK  = 0XC;
  static const int BX_SHIFT =0;

private:

  const unsigned int * word_;
 
  int bx_;

};




#endif
