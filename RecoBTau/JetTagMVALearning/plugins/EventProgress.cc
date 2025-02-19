#include <time.h>

#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>

#include "EventProgress.h"

EventProgress::EventProgress() :
	nEvents(0), startTime(time(NULL)), lastUpdate(0), displayWidth(0)
{
}

EventProgress::EventProgress(unsigned long _nEvents) :
	nEvents(_nEvents), startTime(time(NULL)), lastUpdate(0),
	displayWidth(int(std::log(nEvents + 0.5) / M_LN10 + 1))
{
}

EventProgress::~EventProgress()
{
	std::cout << "                            \r" << std::flush;
}

void EventProgress::update(unsigned long event)
{
	time_t currentTime = time(0);
	if (currentTime == lastUpdate)
		return;

	lastUpdate = currentTime;

	if (displayWidth)
		std::cout << "Event " << std::setw(displayWidth) << event;
	else
		std::cout << "Event " << event;

	if (nEvents) {
		unsigned long eta = (event >= 10) ? ((currentTime - startTime) * (nEvents - event) / event)
		                                  : std::numeric_limits<unsigned long>::max();

		std::cout << " (" << std::setw(2) << (event * 100 / nEvents) << "%), ETA ";
		if (eta >= 6000)
			std::cout << "##:##\r";
		else
			std::cout << std::setw(2) << (eta / 60) << ":"
			          << std::setfill('0') << std::setw(2) << (eta % 60);
	}

	std::cout << std::setfill(' ')
	          << std::resetiosflags(std::ios::fixed)
	          << std::resetiosflags(std::ios::scientific)
	          << std::resetiosflags(std::ios::floatfield)
	          << std::setprecision(8)
	          << "\r" << std::flush;
}
