#ifndef __EventProgress_H__
#define __EventProgress_H__

#include <time.h>

class EventProgress {
    public:
	EventProgress();
	EventProgress(unsigned long nEvents);
	~EventProgress();

	void update(unsigned long event);

    private:
	unsigned long	nEvents;
	time_t		startTime;
	time_t		lastUpdate;
	unsigned int	displayWidth;
};

#endif // __EventProgress_H__
