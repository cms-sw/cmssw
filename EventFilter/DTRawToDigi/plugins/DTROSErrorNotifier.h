#ifndef DTROSErrorNotifier_h
#define DTROSErrorNotifier_h

/** \class DTROSErrorNotifier
 *
 *  \author M. Zanetti - INFN Padova
 */

#include <DataFormats/DTDigi/interface/DTDDUWords.h>
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
