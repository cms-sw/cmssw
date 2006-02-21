#ifndef DTTDCErrorNotifier_h
#define DTTDCErrorNotifier_h

/** \class DTTDCErrorNotifier
 *
 *  $Date: 2005/11/21 17:38:48 $
 *  $Revision: 1.2 $
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
