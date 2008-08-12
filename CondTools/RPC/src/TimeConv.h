#ifndef TIME_CONV_H
#define TIME_CONV_H

/*
 * \class TimeConv
 *  Converts time stamp to Unix int time and viceversa
 *
 *  $Date: 2008/02/1 15:36:08 $
 *  $Revision: 1.1 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */


#include "CoralBase/TimeStamp.h"
#include "RPCSourceHandler.h"


int TtoUT(coral::TimeStamp time);

coral::TimeStamp UTtoT(int utime);


#endif
