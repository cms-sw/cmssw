#ifndef DTROSErrorNotifier_h
#define DTROSErrorNotifier_h

/** \class DTROSErrorNotifier
 *
 *  $Date: 2007/04/24 12:08:20 $
 *  $Revision: 1.1 $
 *  \author M. Zanetti - INFN Padova
 */

#include <EventFilter/DTRawToDigi/interface/DTDDUWords.h>
//class DTROSErrorWord;

class DTROSErrorNotifier {

public:
  
  /// Constructor
  DTROSErrorNotifier(DTROSErrorWord error ); 

  /// Destructor
  virtual ~DTROSErrorNotifier(); 

  /// Print out the error information >>> FIXME: to be implemented
  void print(); 

  // >>> FIXME: Other methods to notify? to whom?

private:

  DTROSErrorWord error_;

};

#endif
