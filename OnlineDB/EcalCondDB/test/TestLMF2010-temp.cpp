#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <time.h>
#include <cstdlib>
#include <limits.h>
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/all_lmf_types.h"

using namespace std;

class CondDBApp {
public:

  /**
   *   App constructor; Makes the database connection
   */
  CondDBApp(string sid, string user, string pass, run_t r)
  {
    try {
      cout << "Making connection..." << flush;
      econn = new EcalCondDBInterface( sid, user, pass );
      run = r;
      cout << "Done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
      exit(-1);
    }
  }

  /**
   *  App destructor;  Cleans up database connection
   */
  ~CondDBApp() 
  {
    delete econn;
  }

  int read_info(std::vector<RunIOV> & run_vec)
  {
    // READ DB
    //
    size_t nruns = run_vec.size();
    for( size_t irun = 0; irun < nruns; ++irun )
      {
	RunIOV & runiov = run_vec[irun];

	RunTag tag = runiov.getRunTag();
	Tm tm_start( runiov.getRunStart().microsTime() );
	Tm tm_end(   runiov.getRunEnd().microsTime() );

	LMFSeqDat seq(econn); //GHM
	std::map<int, LMFSeqDat> l = seq.fetchByRunIOV(runiov);

	//      if( l.size()<nSeqMin ) continue;
	//      std::cout << ii << "\n";
	std::cout << "\nRunIOV("
		  << runiov.getRunNumber() << "/"
		  << tag.getGeneralTag() << "/"
		  << tag.getLocationDef().getLocation() << "/"
		  << tag.getRunTypeDef().getRunType() << ") "
		  << tm_start.str() << " --> " 
		  << tm_end.str()
		  << " === " << l.size() << " Laser Monitoring Sequences"
		  << "\n";

	std::map<int, LMFSeqDat>::const_iterator b = l.begin();
	std::map<int, LMFSeqDat>::const_iterator e = l.end();

	// Loop on sequences for run ii:
	//-------------------------------

	int d = 0;
	while (b != e) {
	  LMFSeqDat seq = b->second;
	  seq.setConnection(econn->getEnv(), econn->getConn());
	  Tm seq_start( seq.getSequenceStart().microsTime() );
	  Tm seq_stop ( seq.getSequenceStop().microsTime()  );
	  std::cout << "\tSeq="
		    << seq.getSequenceNumber() << "\t"
		    << seq_start.str() << " --> " 
		    << seq_stop.str() 
		    << "\n";

	  LMFRunIOV lmfruniov_;
	  lmfruniov_.setConnection(econn->getEnv(), econn->getConn());
	  std::list<LMFRunIOV> l_ = lmfruniov_.fetchBySequence(seq);

	  std::list<LMFRunIOV>::const_iterator b_ = l_.begin();
	  std::list<LMFRunIOV>::const_iterator e_ = l_.end();
	  std::cout << "----------- Got " << l_.size() << " LMFRunIOV's" << "\n";

	  // Loop on LMFRunIOV for sequence b: ie loop on LMR:
	  //----------------------------------------------------
	  int c = 0;
	  while (b_ != e_ ) {

	    LMFClsDat laser_(econn);

	    // choose which primitive you want to read (LASER BLUE)
	    laser_.setSystem("LASER");
	    laser_.setColor("BLUE");
	    // assign the correct LMFRunIOV
	    laser_.setLMFRunIOV(*b_);//CRASH!!!!
	    
	    // Fetch the data from DB
	    laser_.fetch();
	    std::cout << "Run: " << irun << "/" << nruns << " Seq " << d << "/"
		      << l.size() << " RunIOV " << c++ << "/" << l_.size() 
		      << '\r';
	    b_++;
	  }
	  std::cout << std::endl;
	  d++;
	  b++;
	}
      }
    return 1;
  }

  void doRun() {
    LocationDef my_locdef;
    my_locdef.setLocation( "P5_Co" );
    
    RunTypeDef my_rundef;
    my_rundef.setRunType( "TEST" );
    
    RunTag  my_runtag;
    my_runtag.setLocationDef( my_locdef );
    my_runtag.setRunTypeDef(  my_rundef );
    my_runtag.setGeneralTag( "default" );
    
    LMFCorrCoefDat aCorrection(econn);
    RunIOV r2 = aCorrection.fetchLastInsertedRun();
    std::cout << "IOV ID: " << r2.getID() << " Run No. : " << r2.getRunNumber() << std::endl;
    
    // Look for new runs in the DB
    RunList runlist = econn->fetchRunList( my_runtag, 140000, 150000 );
    //RunList runlist = econn->fetchRunList( my_runtag,r2.getRunNumber() , p->max_run );
    std::vector<RunIOV> run_vec =  runlist.getRuns();
    
    
    read_info(run_vec);
  }

private:
  CondDBApp();  // hidden default constructor
  EcalCondDBInterface* econn;
  run_t run;
};

int main (int argc, char* argv[])
{
  string sid;
  string user;
  string pass;

  if (argc != 5) {
    cout << "Usage:" << endl;
    cout << "  " << argv[0] << " <SID> <user> <pass> <run>" << endl;
    exit(-1);
  }

  sid = argv[1];
  user = argv[2];
  pass = argv[3];
  int run = atoi(argv[4]);

  try {
    CondDBApp app(sid, user, pass, run);
    app.doRun();
  } catch (exception &e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }

  cout << "All Done." << endl;

  return 0;
}
