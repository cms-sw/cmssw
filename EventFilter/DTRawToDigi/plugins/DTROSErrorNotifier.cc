/* file 
 *
 *  \author  M. Zanetti  INFN Padova
 */

#include <EventFilter/DTRawToDigi/plugins/DTROSErrorNotifier.h>

//using namespace edm;
using namespace std;

#include <iostream>


DTROSErrorNotifier::DTROSErrorNotifier(DTROSErrorWord error ): error_(error) {}


DTROSErrorNotifier::~DTROSErrorNotifier() {}


void DTROSErrorNotifier::print() {

    cout<<"[DTUnpackingModule]: WARNING!! ROS Error of type "<<error_.errorType()
	<<", from ROB "<<error_.robID()<<endl;

}
