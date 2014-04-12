#include <unistd.h> // usleep
#include <iostream> // cout
#include <math.h>   // exp
#include <stdio.h>  // printf
#include <errno.h>  // errno
#include <sys/wait.h> // waitpid

//To compile on lxplus: /afs/cern.ch/cms/sw/slc4_ia32_gcc345/external/gcc/3.4.5-CMS8/bin/g++ -o show show.cc -I../../../
// make sure you have the latest IORawData/CSCCommissioning first

/// For interactive controll
#include <sys/select.h>
int kbhit(void){
        struct timeval wait;
        int ret;
        fd_set readfd;
        FD_ZERO(&readfd);
        FD_SET(0, &readfd);
        wait.tv_sec  = 0;   /* Wait zero seconds */
        wait.tv_usec = 0;
        ret = select(1, &readfd, NULL, NULL, &wait);
        return(ret);
}
#include <termios.h>
struct termios stored_settings;
void set_keypress(void) {
        struct termios new_settings;
        tcgetattr(0,&stored_settings);
        new_settings = stored_settings;
        new_settings.c_lflag &= (~ICANON);
        new_settings.c_cc[VTIME] = 0;
        tcgetattr(0,&stored_settings);
        new_settings.c_cc[VMIN] = 1;
        tcsetattr(0,TCSANOW,&new_settings);
}
void reset_keypress(void) {
        tcsetattr(0,TCSANOW,&stored_settings);
}
int tty_echo(bool echo){
	struct termios tattr; int fd = 0;
	if(isatty(fd)){
		int rc  = tcgetattr(fd, &tattr);
		if (rc != 0) return rc;
		if(!echo)
			tattr.c_lflag &= ~(ICANON|ECHO);        //Clear ICANON and ECHO
		else
			tattr.c_lflag &=  (ICANON|ECHO);        //Set   ICANON and ECHO
		rc = tcsetattr(fd, TCSANOW, &tattr);
	}
	return 0;
}

__pid_t pid = 0;
int  pipedescr[2] = { 3, 4 };
// Signal handling
#define SIGNAL(s, handler)	{ \
	sa.sa_handler = handler; \
	if (sigaction(s, &sa, NULL) < 0) { \
	    fprintf(stderr, "Couldn't establish signal handler (%d): %m", s); \
	    exit(1); \
	} \
    }
static void term(int qwe){
        printf("Terminating\n"); 
        int stat_loc;
	if( write(pipedescr[1],".q\n",4) != 4 ){ printf("Can't write to pipe errno=%d\n",errno); exit(1); }
        waitpid(pid,&stat_loc,WCONTINUED);
        tty_echo(true);
        reset_keypress();
	system("cp ~/.root_hist.save ~/.root_hist");
	exit(0);
}
static void chld(int qwe){
        printf("root.exe terminated\n"); 
        tty_echo(true);
        reset_keypress();
	system("cp ~/.root_hist.save ~/.root_hist");
	exit(0);
}
static void hup (int qwe){
        printf("Hanging up\n");
        int stat_loc;
	if( write(pipedescr[1],".q\n",4) != 4 ){ printf("Can't write to pipe errno=%d\n",errno); exit(1); }
        waitpid(pid,&stat_loc,WCONTINUED);
        tty_echo(true);
        reset_keypress();
	system("cp ~/.root_hist.save ~/.root_hist");
	exit(0);
}
static void bad_signal(int qwe){
        printf("Some signal %d\n",qwe);
        int stat_loc;
	if( write(pipedescr[1],".q\n",4) != 4 ){ printf("Can't write to pipe errno=%d\n",errno); exit(1); }
        waitpid(pid,&stat_loc,WCONTINUED);
        tty_echo(true);
        reset_keypress();
	system("cp ~/.root_hist.save ~/.root_hist");
	exit(0);
}

/// Unpacker ///
#include "EventFilter/CSCTFRawToDigi/src/CSCTFEvent.cc"
#include "EventFilter/CSCTFRawToDigi/src/CSCSPEvent.cc"
#include "EventFilter/CSCTFRawToDigi/src/CSCSPRecord.cc"
#include "IORawData/CSCCommissioning/src/FileReaderDDU.cc"
////////////////

#include <iterator>
int main(int argc, char *argv[]){
	// Check the arguments
	if( argc<2 ){
		printf("Usage: ./show csc_00*RUI00*.raw\n"); exit(1);
	}

	// Disable terminal bufferization
	set_keypress(); 

	// DDU File Reader
	FileReaderDDU reader;
	try {
		reader.open(argv[1]);
	} catch (...){
		printf("Can't open file %s , errno=%d\n",argv[1],errno); exit(1);
	}

	// Starting root cint and binding its std input to the fd below:
	//__pid_t pid;
	//int  pipedescr[2] = { 3, 4 };
	if( pipe(pipedescr) ) { printf("Can't open pipe errno=%d\n",errno); exit(1); }
	if( ( pid = fork() ) == -1 ) { printf("Can't create process errno=%d\n",errno); exit(1); }
	if( pid == 0 ){
		close(0);
		dup(pipedescr[0]);
		close(pipedescr[0]);
		execl("/bin/sh", "sh", "-c", "root.exe", (char *)0);
	}

	// Save root history
	system("cp ~/.root_hist ~/.root_hist.save");

	// Signal handling
	struct sigaction sa;
	sigset_t mask;

	sigemptyset(&mask);
	sigaddset(&mask, SIGHUP);
	sigaddset(&mask, SIGINT);
	sigaddset(&mask, SIGTERM);
	sigaddset(&mask, SIGCHLD);

	sa.sa_mask = mask;
	sa.sa_flags = 0;
	SIGNAL(SIGHUP, hup);		// Hangup
	SIGNAL(SIGINT, term);		// Interrupt
	SIGNAL(SIGTERM, term);		// Terminate
	SIGNAL(SIGCHLD, chld);          // Child dies

	SIGNAL(SIGUSR1, bad_signal);	// Toggle debug flag
	SIGNAL(SIGUSR2, bad_signal);

	SIGNAL(SIGABRT, bad_signal);
	SIGNAL(SIGALRM, bad_signal);
	SIGNAL(SIGFPE, bad_signal);
	SIGNAL(SIGILL, bad_signal);
	SIGNAL(SIGPIPE, bad_signal);
	SIGNAL(SIGQUIT, bad_signal);
	SIGNAL(SIGSEGV, bad_signal);

#ifdef SIGBUS
	SIGNAL(SIGBUS, bad_signal);
#endif
#ifdef SIGEMT
	SIGNAL(SIGEMT, bad_signal);
#endif
#ifdef SIGPOLL
	SIGNAL(SIGPOLL, bad_signal);
#endif
#ifdef SIGPROF
	SIGNAL(SIGPROF, bad_signal);
#endif
#ifdef SIGSYS
	SIGNAL(SIGSYS, bad_signal);
#endif
#ifdef SIGTRAP
	SIGNAL(SIGTRAP, bad_signal);
#endif
#ifdef SIGVTALRM
	SIGNAL(SIGVTALRM, bad_signal);
#endif
#ifdef SIGXCPU
	SIGNAL(SIGXCPU, bad_signal);
#endif
#ifdef SIGXFSZ
	SIGNAL(SIGXFSZ, bad_signal);
#endif

	// Open TCanvas in root cint:
	char command[1024];
	sprintf(command,"TCanvas p(\"p\",\"Positive endcap\",0,0,600,600);\n");
	if( write(pipedescr[1],command,strlen(command)) != strlen(command) ){ printf("Can't write to pipe errno=%d\n",errno); exit(1); }
	sprintf(command,"TCanvas n(\"n\",\"Negative endcap\",650,0,600,600);\n");
	if( write(pipedescr[1],command,strlen(command)) != strlen(command) ){ printf("Can't write to pipe errno=%d\n",errno); exit(1); }

	using namespace std;

	// Event buffer
	size_t size, nevents=0;
	const unsigned short *buf=0;

	// Do not echo if key is pressed
	tty_echo(false);

	// Main cycle
	while( (size = reader.read(buf)) ){ //&& nevents<1 ){
		unsigned short event[size];

		// Swep out C-words
		unsigned int index1=12, index2=12;
		memcpy(event,buf,12*sizeof(unsigned short));
		while( index1 < size ){
			if( (buf[index1]&0xF000)!=0xC000 ){
				event[index2] = buf[index1];
				index1++;
				index2++;
			} else {
				index1++;
			}
		}

		// Unpack
		CSCTFEvent tfEvent;
		if(nevents%1000==0) cout<<"Event: "<<nevents<<endl;
		//cout<<" Unpack: "<<
		tfEvent.unpack(event,index2);
		//<<endl;

		// Skip empty events
		bool empty = true;
		vector<CSCSPEvent> SPs = tfEvent.SPs();
		for(vector<CSCSPEvent>::const_iterator spPtr=SPs.begin(); spPtr!=SPs.end(); spPtr++)
			for(unsigned int tbin=0; tbin<spPtr->header().nTBINs(); tbin++)
				if( spPtr->record(tbin).LCTs().size() ){ empty = false; break; }
		if( empty ) continue;

		// Set the scale
		sprintf(command,"p.cd();\n");
		if( write(pipedescr[1],command,strlen(command)) != strlen(command) ){ printf("Can't write to pipe errno=%d\n",errno); exit(1); }
		sprintf(command,"TGraph g(2); g.SetPoint(0,-12,-12); g.SetPoint(1,12,12); g.Draw(\"AP\"); g.SetMarkerColor(0); \n");
		if( write(pipedescr[1],command,strlen(command)) != strlen(command) ){ printf("Can't write to pipe errno=%d\n",errno); exit(1); }
		sprintf(command,"n.cd();\n");
		if( write(pipedescr[1],command,strlen(command)) != strlen(command) ){ printf("Can't write to pipe errno=%d\n",errno); exit(1); }
		sprintf(command,"g.Draw(\"AP\"); \n");
		if( write(pipedescr[1],command,strlen(command)) != strlen(command) ){ printf("Can't write to pipe errno=%d\n",errno); exit(1); }
		sprintf(command,"TLegend leg(0.7,0.7,0.99,0.99); \n");
		if( write(pipedescr[1],command,strlen(command)) != strlen(command) ){ printf("Can't write to pipe errno=%d\n",errno); exit(1); }
		sprintf(command,"TGraph _m6(1);   _m6.SetMarkerStyle(22);  _m6.SetMarkerColor(3); leg.AddEntry(&_m6,\"CSCTF track mode=6\",\"p\"); \n");
		if( write(pipedescr[1],command,strlen(command)) != strlen(command) ){ printf("Can't write to pipe errno=%d\n",errno); exit(1); }
		sprintf(command,"TGraph _m8(1);   _m8.SetMarkerStyle(22);  _m8.SetMarkerColor(4); leg.AddEntry(&_m8,\"CSCTF track mode=8\",\"p\"); \n");
		if( write(pipedescr[1],command,strlen(command)) != strlen(command) ){ printf("Can't write to pipe errno=%d\n",errno); exit(1); }
		sprintf(command,"TGraph _m11(1); _m11.SetMarkerStyle(22); _m11.SetMarkerColor(7); leg.AddEntry(&_m11,\"CSCTF track mode=11\",\"p\"); \n");
		if( write(pipedescr[1],command,strlen(command)) != strlen(command) ){ printf("Can't write to pipe errno=%d\n",errno); exit(1); }
		sprintf(command,"TGraph _m15(1); _m15.SetMarkerStyle(22); _m15.SetMarkerColor(6); leg.AddEntry(&_m15,\"CSCTF track mode=15\",\"p\"); \n");
		if( write(pipedescr[1],command,strlen(command)) != strlen(command) ){ printf("Can't write to pipe errno=%d\n",errno); exit(1); }
		sprintf(command,"TGraph _csc12(1); _csc12.SetMarkerStyle(4); _csc12.SetMarkerColor(3); leg.AddEntry(&_csc12,\"ME1 LCT\",\"p\"); \n");
		if( write(pipedescr[1],command,strlen(command)) != strlen(command) ){ printf("Can't write to pipe errno=%d\n",errno); exit(1); }
		sprintf(command,"TGraph  _csc2(1);  _csc2.SetMarkerStyle(4);  _csc2.SetMarkerColor(4); leg.AddEntry(&_csc2,\"ME2 LCT\",\"p\"); \n");
		if( write(pipedescr[1],command,strlen(command)) != strlen(command) ){ printf("Can't write to pipe errno=%d\n",errno); exit(1); }
		sprintf(command,"TGraph  _csc3(1);  _csc3.SetMarkerStyle(4);  _csc3.SetMarkerColor(7); leg.AddEntry(&_csc3,\"ME3 LCT\",\"p\"); \n");
		if( write(pipedescr[1],command,strlen(command)) != strlen(command) ){ printf("Can't write to pipe errno=%d\n",errno); exit(1); }
		sprintf(command,"TGraph  _csc4(1);  _csc4.SetMarkerStyle(4);  _csc4.SetMarkerColor(6); leg.AddEntry(&_csc4,\"ME2 LCT\",\"p\"); \n");
		if( write(pipedescr[1],command,strlen(command)) != strlen(command) ){ printf("Can't write to pipe errno=%d\n",errno); exit(1); }
		sprintf(command,"leg.Draw(); \n");
		if( write(pipedescr[1],command,strlen(command)) != strlen(command) ){ printf("Can't write to pipe errno=%d\n",errno); exit(1); }

		for(vector<CSCSPEvent>::const_iterator spPtr=SPs.begin(); spPtr!=SPs.end(); spPtr++){
///			cout<<" L1A="<<SPs[0].header().L1A()<<endl;
			if( !spPtr->header().endcap() ){
				sprintf(command,"p.cd();\n");
				if( write(pipedescr[1],command,strlen(command)) != strlen(command) ){ printf("Can't write to pipe errno=%d\n",errno); exit(1); }
			} else {
				sprintf(command,"n.cd();\n");
				if( write(pipedescr[1],command,strlen(command)) != strlen(command) ){ printf("Can't write to pipe errno=%d\n",errno); exit(1); }
			}

			for(unsigned int tbin=0; tbin<spPtr->header().nTBINs(); tbin++){
				int sector = spPtr->header().sector() + spPtr->header().endcap()*6;
				vector<CSCSP_MEblock> LCTs = spPtr->record(tbin).LCTs();
				if( LCTs.size() ){
					cout<<"Event: "<<nevents<<" Sector="<<spPtr->header().sector()<<" L1A="<<spPtr->header().L1A()<<endl;
					cout<<" Endcap: "<<(spPtr->header().endcap()?2:1)<<" sector: "<<spPtr->header().sector();
					cout<<"  tbin: "<<tbin<<"  nLCTs: "<<LCTs.size()<<" (";//<<endl;
				}
				for(std::vector<CSCSP_MEblock>::const_iterator lct=LCTs.begin(); lct!=LCTs.end(); lct++){
					cout<<" F"<<((lct->spInput()-1)/3+1)<<"/CSC"<<lct->csc()<<":{w="<<lct->wireGroup()<<",s="<<lct->strip()<<"} ";
					unsigned short mpc =(lct->spInput()-1)/3+1; // 1-5
					unsigned short csc = lct->csc();            // 1-9

					// Approximate LUTs' parameters:
					const double offsetEta[6][10] = {
						{-1,  -1,  -1,  -1, -1, -1, -1, -1, -1, -1},
						{-1, 2.4,  2.4,  2.4,  1.67, 1.67, 1.67, 1.13, 1.13, 1.13},
						{-1, 2.4,  2.4,  2.4,  1.67, 1.67, 1.67, 1.13, 1.13, 1.13},
						{-1, 2.48, 2.48, 2.48, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57},
						{-1, 2.47, 2.47, 2.47, 1.70, 1.70, 1.70, 1.70, 1.70, 1.70},
						{-1, 2.46, 2.46, 2.46, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78}
					};
					const double scaleEta[6][10] = {
						{-1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1},
						{-1, 0.75, 0.75, 0.75, 0.47, 0.47, 0.47, 0.24, 0.24, 0.24},
						{-1, 0.75, 0.75, 0.75, 0.47, 0.47, 0.47, 0.24, 0.24, 0.24},
						{-1, 0.88, 0.88, 0.88, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57},
						{-1, 0.74, 0.74, 0.74, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56},
						{-1, .645, .645, .645, .595, .595, .595, .595, .595, .595}
					};
					const double normEta[6][10] = {
						{-1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
						{-1, 48., 48., 48., 64., 64., 64., 32., 32., 32.},
						{-1, 48., 48., 48., 64., 64., 64., 32., 32., 32.},
						{-1,112.,112.,112., 64., 64., 64., 64., 64., 64.},
						{-1, 96., 96., 96., 64., 64., 64., 64., 64., 64.},
						{-1, 96., 96., 96., 64., 64., 64., 64., 64., 64.}
					};
					const double offsetPhi[6][10] = {
						{-1,   -1,   -1,   -1,   -1,  -1,    -1,   -1,   -1,   -1},
						{-1,    0, 1/6., 2/6.,    0, 1/6., 2/6.,    0, 1/6., 2/6.},
						{-1, 3/6., 4/6., 5/6., 3/6., 4/6., 5/6., 3/6., 4/6., 5/6.},
						{-1,    0, 1/3., 2/3.,    0, 1/6., 2/6., 3/6., 4/6., 5/6.},
						{-1,    0, 1/3., 2/3.,    0, 1/6., 2/6., 3/6., 4/6., 5/6.},
						{-1,    0, 1/3., 2/3.,    0, 1/6., 2/6., 3/6., 4/6., 5/6.}
					};
					const double normPhi[6][10] = {
						{-1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
						{-1, 158, 158, 158, 158, 158, 158, 126, 126, 126},
						{-1, 158, 158, 158, 158, 158, 158, 126, 126, 126},
						{-1, 158, 158, 158, 158, 158, 158, 158, 158, 158},
						{-1, 158, 158, 158, 158, 158, 158, 158, 158, 158},
						{-1, 158, 158, 158, 158, 158, 158, 158, 158, 158}
					};
					const double scalePhi[6][10] = {
						{-1,    -1,    -1,   -1,     -1,    -1,    -1,    -1,    -1,    -1},
						{-1, 1./6., 1./6., 1./6., 1./6., 1./6., 1./6., 1./6., 1./6., 1./6.},
						{-1, 1./6., 1./6., 1./6., 1./6., 1./6., 1./6., 1./6., 1./6., 1./6.},
						{-1, 1./3., 1./3., 1./3., 1./6., 1./6., 1./6., 1./6., 1./6., 1./6.},
						{-1, 1./3., 1./3., 1./3., 1./6., 1./6., 1./6., 1./6., 1./6., 1./6.},
						{-1, 1./3., 1./3., 1./3., 1./6., 1./6., 1./6., 1./6., 1./6., 1./6.}
					};
					const double emuZ[6][10] = {
						{-1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1},
						{-1,    6,    6,    6, 7.03, 7.03, 7.03, 7.03, 7.03, 7.03},
						{-1,    6,    6,    6, 7.03, 7.03, 7.03, 7.03, 7.03, 7.03},
						{-1,  8.2,  8.2,  8.2,  8.2,  8.2,  8.2,  8.2,  8.2,  8.2},
						{-1, 9.41, 9.41, 9.41, 9.41, 9.41, 9.41, 9.41, 9.41, 9.41},
						{-1,10.31,10.31,10.31,10.31,10.31,10.31,10.31,10.31,10.31}
					};

					if( mpc<6 && csc<10){
						double eta, phi;
						if( mpc<3 && csc<=3 && lct->strip()>=128 ){
							eta = 2.4 - lct->wireGroup()/48.* 0.3; // very approximate eta coordinate!!!
							phi = (lct->strip()-128)/32./6. + 1./6. + (mpc-1)*1./2.;
							phi = fmod((phi + spPtr->header().sector()-1.)/6.*2*3.1415927 + 3.1415927/12.,2*3.1415927);
						} else {
							eta = offsetEta[mpc][csc] - scaleEta[mpc][csc] * lct->wireGroup() / normEta[mpc][csc];
							phi = lct->strip() / normPhi[mpc][csc];
							if( (!spPtr->header().endcap() && mpc<4) || (spPtr->header().endcap() && mpc>=4) ) phi = 1. - phi;
							phi = fmod((phi*scalePhi[mpc][csc]+offsetPhi[mpc][csc] + spPtr->header().sector()-1.)/6.*2*3.1415927 + 3.1415927/12.,2*3.1415927);
						}

						double rho = 2.*emuZ[mpc][csc]/fabs(exp(eta)-exp(-eta));
						double x   = rho * cos(phi);
						double y   = rho * sin(phi);

						const int color[6] = { -1, 3, 3, 4, 7, 6 };
						sprintf(command,"TGraph lct_%d_%d_%d(1); lct_%d_%d_%d.SetPoint(0,%f,%f); lct_%d_%d_%d.SetMarkerStyle(4); lct_%d_%d_%d.SetMarkerColor(%d); lct_%d_%d_%d.Draw(\"P\"); \n",mpc,csc,sector,mpc,csc,sector,x,y,mpc,csc,sector,mpc,csc,sector,color[mpc],mpc,csc,sector);
						if( write(pipedescr[1],command,strlen(command)) != strlen(command) ){ printf("Can't write to pipe errno=%d\n",errno); exit(1); }
					}
				}
				if( LCTs.size() ) cout<<" )"<<endl;

				std::vector<CSCSP_SPblock> trks = spPtr->record(tbin).tracks();
				if( trks.size() ){ cout<<"  Track(s) at BX="<<spPtr->header().BXN()<<" :"; }
				for(std::vector<CSCSP_SPblock>::const_iterator trk=trks.begin(); trk!=trks.end(); trk++){
					cout<<" mode="<<trk->mode()<<" (eta="<<trk->eta()<<",phi="<<trk->phi()<<")";
					double eta = (2.5-0.9)*trk->eta()/32.+0.9;
					double phi = trk->phi() + ((spPtr->header().sector() - 1)*24) + 6;
					if(phi > 143) phi -= 143;
					phi /= 144./2./3.1415927;
					double rho2= 2.*8.2/fabs(exp(eta)-exp(-eta));
					double x2  = rho2 * cos(phi);
					double y2  = rho2 * sin(phi);
					const int color[16] = { -1, 1, 1, 1, 1, 1, 3, 1, 4, 1, 1, 7, 1, 1, 1, 6 };
					int i = distance(vector<CSCSP_SPblock>::const_iterator(trks.begin()),trk);
					sprintf(command,"TGraph trk_%d_%d(1); trk_%d_%d.SetPoint(0,%f,%f); trk_%d_%d.SetMarkerStyle(22); trk_%d_%d.SetMarkerColor(%d); trk_%d_%d.Draw(\"P\"); \n",i,sector,i,sector,x2,y2,i,sector,i,sector,color[trk->mode()],i,sector);
					if( write(pipedescr[1],command,strlen(command)) != strlen(command) ){ printf("Can't write to pipe errno=%d\n",errno); exit(1); }
				}
				if( trks.size() ){ cout<<endl; }

			}
		}

		// User interface block:
		while( !kbhit() ){
			int key1=0, key2=0, key3=0;
			if((key1=getchar())==27 && (key2=getchar())==91 && (key3=getchar())==68 ){
				std::cout<<"Pressed: "<<key1<<" "<<key2<<" "<<key3<<std::endl;
				//back=1;
				break;
			} // backward
			if( key3==67 ){
				//back=0;
				break;
			} // forward
			if( key3==65 ){
				//back=0;
				break;
			} // up
			if( key3==66 ){
				//back=1;
				break;
			} // down
			if( key1==32 ){
				//std::cout<<"Pause ... "<<std::flush;
				//while( !kbhit() || ( (key1=getchar())!=32 && (key1=getchar())!=113 ) ) usleep(100);
				//if( key1==113 ){
				//	std::cout<<"quit"<<std::endl;
				//	break;
				//} else {
				//	std::cout<<"continue"<<std::endl;
				//}
				break;
			} // space
			if( key1==113 ) { size = 0; break; }
			usleep(100);
		}
		sprintf(command,"p.Clear();\n");
		if( write(pipedescr[1],command,strlen(command)) != strlen(command) ){ printf("Can't write to pipe errno=%d\n",errno); exit(1); }
		sprintf(command,"n.Clear();\n");
		if( write(pipedescr[1],command,strlen(command)) != strlen(command) ){ printf("Can't write to pipe errno=%d\n",errno); exit(1); }
		nevents++;
		if( size==0 ) break;
	}
	if( write(pipedescr[1],".q\n",4) != 4 ){ printf("Can't write to pipe errno=%d\n",errno); exit(1); }
	int stat_loc;
	waitpid(pid,&stat_loc,WCONTINUED);
	tty_echo(true);
	reset_keypress();
	system("mv ~/.root_hist.save ~/.root_hist");
	return 0;
}
