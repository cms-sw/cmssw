#ifndef DTTDCErrorNotifier_h
#define DTTDCErrorNotifier_h

/** \class DTTDCErrorNotifier
 *
 *  $Date: 2007/05/07 16:16:40 $
 *  $Revision: 1.3 $
 *  \author M. Zanetti - INFN Padova
 */

#include <EventFilter/DTRawToDigi/interface/DTDDUWords.h>
//class DTTDCErrorWord;

class DTTDCErrorNotifier {

public:
  
  /// Constructor
  DTTDCErrorNotifier(DTTDCErrorWord error ); 

  /// Destructor
  virtual ~DTTDCErrorNotifier(); 

  /// Print out the error information >>> FIXME: to be implemented
  void print(); 

  // >>> FIXME: Other methods to notify? to whom?

private:

  DTTDCErrorWord error_;

};

#endif
