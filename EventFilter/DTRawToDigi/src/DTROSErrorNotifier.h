#ifndef DTROSErrorNotifier_h
#define DTROSErrorNotifier_h

/** \class DTROSErrorNotifier
 *
 *  $Date: 2005/11/09 13:20:48 $
 *  $Revision: 1.1.2.1 $
 *  \author M. Zanetti - INFN Padova
 */

#include <EventFilter/DTRawToDigi/src/DTDDUWords.h>
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
