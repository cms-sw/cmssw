/* file 
 *
 *  $Date: 2005/11/21 17:38:48 $
 *  $Revision: 1.2 $
 *  \author  M. Zanetti  INFN Padova
 */

#include <EventFilter/DTRawToDigi/src/DTROSErrorNotifier.h>
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
