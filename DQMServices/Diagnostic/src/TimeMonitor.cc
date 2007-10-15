//<<<<<< INCLUDES                                                       >>>>>>

#include "DQMServices/Diagnostic/interface/TimeMonitor.h"
#include <iostream>

using std::string;
using std::cout; 
using std::endl;


//DEFINE_FWK_SERVICE(TimeMonitor)


//<<<<<< PRIVATE DEFINES                                                >>>>>>
//<<<<<< PRIVATE CONSTANTS                                              >>>>>>
//<<<<<< PRIVATE TYPES                                                  >>>>>>
//<<<<<< PRIVATE VARIABLE DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC VARIABLE DEFINITIONS                                    >>>>>>
//<<<<<< CLASS STRUCTURE INITIALIZATION                                 >>>>>>
//<<<<<< PRIVATE FUNCTION DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC FUNCTION DEFINITIONS                                    >>>>>>
//<<<<<< MEMBER FUNCTION DEFINITIONS                                    >>>>>>

TimeMonitor::TimeMonitor () 
    : tempTime (0), N_ITER (20)
{}


TimeMonitor::~TimeMonitor (void) 
{
    //FIXME CLOSE ALL OPENED FILES
//     std::map <const char*, TimeData*>::iterator i;
//     for (i = timeMap.begin (); i != timeMap.end (); ++i) 
//     {
// 	(*i).second->file.close ();
//     }
} 

void TimeMonitor::set (const char* sourceName, seal::TimeInfo::NanoSecs currentClock, int currSize) 
{
    if (!strcmp (sourceName,"Collector"))		//We are in the Client part
    {
	return;
    }
   
    //if there are no occurencies of that sourceName add it to the map
    if (timeMap.count (sourceName) == 0) 
    {
	TimeData *t = new TimeData ();
	t->file.open (sourceName);
	t->file << "ADDING: " << sourceName << " at time: " << currentClock << "\n";
	t->startTime = currentClock;
	t->doneTime = -1;
	t->count = 1;
	t->iterCounter = 1;
	t->messSize = currSize;
	
	timeMap [sourceName] = t;
	doneMap [sourceName] = false;
    } else
	if (doneMap [sourceName]) 
	{
	    timeMap [sourceName]->doneTime = currentClock;
	    timeMap [sourceName]->file << (timeMap [sourceName]->iterCounter) << "   Sorgente: " << sourceName << "   DeltaTime: " 
				       << (timeMap[sourceName]->doneTime) - (timeMap[sourceName]->startTime) 
				       << "   FIRST: " << (timeMap[sourceName]->startTime) << "   SECOND: " << (timeMap[sourceName]->doneTime)    
				       << "   NUMERO AGGIORNAMENTI: " << timeMap[sourceName]->count << "   SIZE: " 
				       << timeMap [sourceName]->messSize <<"\n";
	    
	    if (timeMap [sourceName]->iterCounter == N_ITER)
		timeMap [sourceName]->file.flush ();
	    else
		timeMap [sourceName]->iterCounter++;
	    
	    timeMap [sourceName]->count = 0;
	    timeMap [sourceName]->startTime = currentClock;
	    timeMap [sourceName]->messSize = 0;
	    
	    doneMap [sourceName] = false;
	}
	else
	{
	    timeMap [sourceName]->count++;
	    timeMap [sourceName]->messSize += currSize;
	}
    tempTime = currentClock;
}

void TimeMonitor::setDone (const char* source) 
{
    doneMap [source] = true;
}

