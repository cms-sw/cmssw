/* file 
 *
 *  \author  M. Zanetti  INFN Padova
 */

#include <EventFilter/DTRawToDigi/plugins/DTTDCErrorNotifier.h>

//using namespace edm;
using namespace std;

#include <iostream>

DTTDCErrorNotifier::DTTDCErrorNotifier(DTTDCErrorWord error) : error_(error) {}

DTTDCErrorNotifier::~DTTDCErrorNotifier() {}

void DTTDCErrorNotifier::print() {
  cout << "[DTUnpackingModule]: WARNING!! TDC Error of type " << error_.tdcError() << ", from TDC " << error_.tdcID()
       << endl;
}
