#ifndef DTTDCErrorNotifier_h
#define DTTDCErrorNotifier_h

/** \class DTTDCErrorNotifier
 *
 *  \author M. Zanetti - INFN Padova
 */

#include <DataFormats/DTDigi/interface/DTDDUWords.h>
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
