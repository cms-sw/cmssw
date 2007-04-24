/* file 
 *
 *  $Date: 2006/02/21 19:15:55 $
 *  $Revision: 1.3 $
 *  \author  M. Zanetti  INFN Padova
 */

#include <EventFilter/DTRawToDigi/plugins/DTTDCErrorNotifier.h>
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
