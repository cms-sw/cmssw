/*
 * msq_constants.h
 *
 *  Created on: Nov 23, 2011
 *      Author: aspataru : aspataru@cern.ch
 */

#ifndef MSQ_CONSTANTS_H_
#define MSQ_CONSTANTS_H_

namespace evf {

// DATA message types
static const unsigned int RAW_MESSAGE_TYPE = 100;
static const unsigned int RECO_MESSAGE_TYPE = 101;
static const unsigned int DQM_MESSAGE_TYPE = 102;

// CONTROL message types
static const unsigned int DISCARD_RAW_MESSAGE_TYPE = 200;

}

#endif /* MSQ_CONSTANTS_H_ */
