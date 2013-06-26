/* file 
 *
 *  $Date: 2008/01/22 18:46:41 $
 *  $Revision: 1.4 $
 *  \author  M. Zanetti  INFN Padova
 */

#include <EventFilter/DTRawToDigi/plugins/DTTDCErrorNotifier.h>

//using namespace edm;
using namespace std;

#include <iostream>


DTTDCErrorNotifier::DTTDCErrorNotifier(DTTDCErrorWord error ): error_(error) {}


DTTDCErrorNotifier::~DTTDCErrorNotifier() {}


void DTTDCErrorNotifier::print() {

    cout<<"[DTUnpackingModule]: WARNING!! TDC Error of type "<<error_.tdcError()
	<<", from TDC "<<error_.tdcID()<<endl;

}
