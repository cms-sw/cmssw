#ifndef DQM_SiStripMonitorClient_check_runcomplete_h
#define DQM_SiStripMonitorClient_check_runcomplete_h
#include <string>
int  read_runflag        ( int,std::string);
int  get_filename        ( int,std::string,std::string&);
void check_runcomplete ( int run , std::string repro_type );
#endif // DQM_SiStripMonitorClient_check_runcomplete_h
