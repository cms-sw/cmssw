#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include <cstdlib>
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/all_od_types.h"


using namespace std;

class CondDBApp {
public:

  /**
   *   App constructor; Makes the database connection
   */
  CondDBApp(string sid, string user, string pass, run_t r)
  {
    try {
      cout << "Making connection to " << user << "@" << sid << "..." << flush;
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
  
  void testRead()  {
  //Get RunIOV info
  RunIOV iov = econn->fetchRunIOV(run);
  std::list<ODBadTTDat> badttlist = econn->fetchBadTTForRun(&iov);
  std::list<ODBadTTDat>::const_iterator i =  badttlist.begin();
  std::list<ODBadTTDat>::const_iterator e =  badttlist.end();
  while (i != e) {
    std::cout << i->getFedId() << " " << i->getTTId() << ": " 
	      << i->getStatus() 
	      << std::endl << std::flush;
    i++;
  }
};




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
  int run;

  std::cout << argc << std::endl;
  if (argc != 5) {
    cout << "Usage:" << endl;
    cout << "  " << argv[0] << " <SID> <user> <pass> <run>" 
	 << endl;
    exit(-1);
  }

  sid = argv[1];
  user = argv[2];
  pass = argv[3];
  run = atoi(argv[4]);

  try {
    std::cout << "Retrieving data for run " << run 
	      << std::endl << std::flush;
    CondDBApp app(sid, user, pass, run);

    app.testRead();

  } catch (exception &e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }

  cout << "All Done." << endl;

  return 0;
}
