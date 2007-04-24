#ifndef DTROSErrorNotifier_h
#define DTROSErrorNotifier_h

/** \class DTROSErrorNotifier
 *
 *  $Date: 2006/02/21 19:15:55 $
 *  $Revision: 1.3 $
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
