#ifndef DiscardedData_h
#define DiscardedData_h


/** \file
 * Unpacks Information of Discarded Data
 *
 *  $Date: 2005/10/21 11:00:01 $
 *  $Revision: 1.1 $
 * \author Ilaria Segoni - CERN
 */
class DiscardedData{

public:
  
  /// Constructor
  DiscardedData(const unsigned char* index);

  /// Destructor
  virtual ~DiscardedData() {};

  /// unpacked data access methods
  int channelDiscarded();
  int tbRmbDiscarded();
  int chamberDiscarded();

  static const int CHANNEL_DISCARDED_MASK  = 0X1F;
  static const int CHANNEL_DISCARDED_SHIFT =0;

  static const int TB_RMB_DISCARDED_MASK = 0X3F;
  static const int TB_RMB_DISCARDED_SHIFT =5;

  static const int CHAMBER_DISCARDED_MASK = 0X3;
  static const int CHAMBER_DISCARDED_SHIFT =14;



private:

  const unsigned int * word_;
 
  int channelDiscarded_;
  int tbRmbDiscarded_;
  int chamberDiscarded_;
  
};




#endif
