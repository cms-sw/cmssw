/* file 
 *
 *  $Date: 2007/04/24 12:08:20 $
 *  $Revision: 1.1 $
 *  \author  M. Zanetti  INFN Padova
 */

#include <EventFilter/DTRawToDigi/plugins/DTROSErrorNotifier.h>
#include <EventFilter/DTRawToDigi/interface/DTDDUWords.h>

//using namespace edm;
using namespace std;

#include <iostream>


DTROSErrorNotifier::DTROSErrorNotifier(DTROSErrorWord error ): error_(error) {}


DTROSErrorNotifier::~DTROSErrorNotifier() {}


void DTROSErrorNotifier::print() {

    cout<<"[DTUnpackingModule]: WARNING!! ROS Error of type "<<error_.errorType()
	<<", from ROB "<<error_.robID()<<endl;

}
