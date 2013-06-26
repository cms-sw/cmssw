/* file 
 *
 *  $Date: 2008/01/22 18:46:41 $
 *  $Revision: 1.4 $
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
