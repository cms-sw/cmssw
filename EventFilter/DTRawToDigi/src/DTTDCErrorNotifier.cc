/* file 
 *
 *  $Date: 2007/04/26 18:53:06 $
 *  $Revision: 1.5 $
 *  \author  M. Zanetti  INFN Padova
 */

#include <EventFilter/DTRawToDigi/src/DTTDCErrorNotifier.h>
#include <EventFilter/DTRawToDigi/interface/DTDDUWords.h>

//using namespace edm;
using namespace std;

#include <iostream>


DTTDCErrorNotifier::DTTDCErrorNotifier(DTTDCErrorWord error ): error_(error) {}


DTTDCErrorNotifier::~DTTDCErrorNotifier() {}


void DTTDCErrorNotifier::print() {

    cout<<"[DTUnpackingModule]: WARNING!! TDC Error of type "<<error_.tdcError()
	<<", from TDC "<<error_.tdcID()<<endl;

}
