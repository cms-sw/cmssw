/* file 
 *
 *  $Date: 2006/02/21 19:15:55 $
 *  $Revision: 1.3 $
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
