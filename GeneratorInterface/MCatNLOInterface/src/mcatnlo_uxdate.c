#include <time.h>
#include <sys/times.h>
#include <unistd.h>
/* long	iutime() */
/*main()*/
void uxdate_(year,mon,day,hour,min)
int	*year, *mon, *day, *hour, *min;
{
	struct	tm q;
        struct  tm *localtime();
        time_t  tp;
        time_t  mktime();
        time_t  time();
        char    *ctime();
	char    *date;
        

        time(&tp);
        date = ctime(&tp);
        q = *localtime(&tp);
        *year = q.tm_year;
        *mon  = q.tm_mon + 1;
        *day  = q.tm_mday;
        *hour = q.tm_hour;
        *min  = q.tm_min;

	return;
}

float	uxtime_() 
{
	struct	tms q;
	long 	t,s;
	long    sysconf();
	long    ticks;
	float   uxtime;


        ticks = sysconf(_SC_CLK_TCK);
	times(&q);
        t = q.tms_utime + q.tms_cutime
           +q.tms_stime + q.tms_cstime;
	return (float) t/ (float) ticks;
}
