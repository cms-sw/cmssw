//
// Scan the directory 
// V 2.0 Benigno 20051207
// V 3.0 Benigno 20070425

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cerrno>
#include <map>
#include <algorithm>
#include <ctime>
#include <sstream>
#include <iomanip>

#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>

unsigned int selrunge;
bool         flrunge  = false;
unsigned int selrunle;
bool         flrunle  = false;
std::string  seltype;
bool         fltype   = false;
std::string  sortmode = "datedown";
unsigned int from = 0;
unsigned int pagesize = 0;

const std::string filebeg = "ecal_local.";
const int         rundigi = 8;
const std::string dqmadd  = "0";
const bool        na      = true;
const std::string logpath = "/logs";
const std::string htmpath = "/html";
const std::string roopath = "/root";

// ------------------------------------------------------------------------------------
int getentries( std::string const &path, std::vector<std::string> &entrynames ) {

  DIR *dir = opendir( path.c_str() );
  if( dir == NULL ) return (-1);
  dirent *entry( NULL );
  do {
    entry = readdir( dir );
    if( entry != NULL ) {
      entrynames.push_back( entry->d_name );
    }
  } while( entry != NULL );

  if( closedir( dir ) ) return -1;

  return 0;

}

// ------------------------------------------------------------------------------------
void select( std::string const &path, std::vector<std::string> const entrynames ) {

  unsigned int runnumber = 0;
  std::string  type;

  std::string  status;

  std::map<int, std::string> out1;

  std::string htmlpath = path.substr( 0, path.find( logpath )) + htmpath; 

  std::string rootpath = path.substr( 0, path.find( logpath )) + roopath;

  // Loop on all entries of choosen directory...
  for( unsigned int i=0; i<entrynames.size(); i++ ) {

    std::string logfile = entrynames[i].c_str();
    unsigned int pos0;

    if( (pos0 = logfile.find( filebeg )) == 0 ) {

      std::string sdate = "?";
      std::string stime = "?";
      struct stat st;
      if( stat( (path+"/"+logfile).c_str(), &st ) != -1 ) {
	time_t filetime = st.st_mtime;
	char cdate[11], ctime[9];
	strftime( cdate, 11, "%Y/%m/%d", localtime( &filetime ));
	strftime( ctime, 9, "%T", localtime( &filetime ));
	sdate = cdate;
	stime = ctime;
      }

      unsigned int n;
      if ( ( n = logfile.find( ".gz" )) < logfile.size() ) { // remove .gz extention...
	logfile = logfile.substr( 0, n );    
      }

      std::string runNb = dqmadd + logfile.substr( pos0+filebeg.size(), rundigi );
      runnumber = atoi( runNb.c_str() );

      std::string output = "";
      
      std::string infile = htmlpath + "/" + runNb + "/index.html";
      std::ifstream f( infile.c_str() );
      
      bool infileOK = true;
      if( f.fail() ) {
	infileOK = false;
      }
      else {
	char s[256];
	while( ! f.eof() ) {
	  f.getline(s, 256);
	  if( strncmp( s, "<h2>Run type:&nbsp&nbsp&nbsp", 28 ) == 0 ) { 
	    f.getline(s, 256);
	    std::string st = s;
	    st = st.substr( st.find(">")+1 );
	    st = st.substr( 0, st.find("<") );
	    type = st;
	  }
	}
      }

      std::string rofile1 = rootpath + "/" + "DQM_EcalBarrel_R" + runNb + ".root";
      std::ifstream f1( rofile1.c_str() );
      std::string rofile2 = rootpath + "/" + "DQM_EcalEndcap_R" + runNb + ".root";
      std::ifstream f2( rofile2.c_str() );

      std::string rofile3 = rootpath + "/" + "DQM_V0001_EcalBarrel_R" + runNb + ".root";
      std::ifstream f3( rofile3.c_str() );
      std::string rofile4 = rootpath + "/" + "DQM_V0001_EcalEndcap_R" + runNb + ".root";
      std::ifstream f4( rofile4.c_str() );

      bool rofileOK = true;
      if( f1.fail() && f2.fail() && f3.fail() && f4.fail() ) {
        rofileOK = false;
      }

      bool sele = true; 

      if( flrunge && flrunle ) {
	if( runnumber < selrunge || runnumber > selrunle ) sele = false;
      }
      else if( flrunge ) {
	if( runnumber < selrunge ) sele = false;
      }
      else if( flrunle ) {
	if( runnumber > selrunle ) sele = false;
      }

      if( fltype && sele ) {
	if( type == seltype || ( seltype == "NOTAVAILABLE" && !infileOK ) ) sele = true;
	else sele = false;
      }

      if( sele ) {
	if( infileOK ) {
	  output = "<td class=entry><b>ENTRY</b></td>";
	  output += "<td class=type><nobr>" + type + "</nobr></td>";
	  output += "<td class=date>" + sdate + "</td>";
	  output += "<td class=time>" + stime + "</td>";
	  output += "<td class=runnb><a href=.." + htmpath + "/" + runNb + "/index.html>" + runNb + "</a></td>";
	  
	  output += "<td class=log><a href=showlogfile.php?filename=" + path + "/" + logfile + "><img src=blue_arrow.gif border=0></a></td>";
	  if( rofileOK ) {
	    std::stringstream s;
	    s << runnumber;
	    output += "<td class=gui><a href=http://ecalod-web01.cms:8030/dqm/ecal/start?runnr=" + s.str() + "><img src=blue_arrow.gif border=0></a></td>";
	  }
	  else {
	    output += "<td class=gui>-</td>";
	  }
	  out1[runnumber] = output;
	}
	else {
          if( na ) { 
	    output = "<td class=entry><b>ENTRY</b></td>";
	    output += "<td class=type> <font color=red><nobr>&lt;not available&gt;</nobr></font> </td>";
	    output += "<td class=date>" + sdate + "</td>";
	    output += "<td class=time>" + stime + "</td>";
	    output += "<td class=runnb>" + runNb + "</td>";
	    output += "<td class=log><a href=showlogfile.php?filename=" + path + "/" + logfile + "><img src=blue_arrow.gif border=0></a></td>";
	    if( rofileOK ) {
	      std::stringstream s;
	      s << runnumber;
	      output += "<td class=gui><a href=http://ecalod-web01.cms:8030/dqm/ecal/start?runnr=" + s.str() + "><img src=blue_arrow.gif border=0></a></td>";
	    }
	    else {
	      output += "<td class=gui>-</td>";
	    }
	    out1[runnumber] = output;
	  }
	}
      }
    }    
  }

  unsigned int entry=0;
  if( sortmode == "runup" ) {
    for( std::map<int,std::string>::iterator i=out1.begin(); i!=out1.end(); i++ ) {
      entry++;
      if( entry>=(from+1) && ( entry<(pagesize+from+1) || pagesize==0)) {
	std::ostringstream os; os << entry;
	std::string s = i->second;
	s = s.substr( 0, s.find("ENTRY", 0) ) + os.str() + s.substr( s.find("ENTRY",0)+5, s.size());
	std::cout << s << std::endl;
      }
    }
  }
  else if( sortmode == "rundown" ) {
    for( std::map<int,std::string>::reverse_iterator i=out1.rbegin(); i!=out1.rend(); i++ ) {
      entry++;
      if( entry>=(from+1) && ( entry<(pagesize+from+1) || pagesize==0)) {
	std::ostringstream os; os << entry;
	std::string s = i->second;
	s = s.substr( 0, s.find("ENTRY", 0) ) + os.str() + s.substr( s.find("ENTRY",0)+5, s.size());
	std::cout << s << std::endl;
      }
    }
  }

  std::cout << entry << std::endl;
}

// ------------------------------------------------------------------------------------
void usage( char* cp ) {

  std::cout << 
"\n\
usage: " << cp << " [OPTIONS] directory \n\n\
     -h             : print this help message \n\
     -f entry       : output starting from 'entry'-th entry \n\
     -o sortmode    : rundown (default), runup \n\
     -p pagesize    : output 'pagesize' entries \n\
     -r run         : select only files with run number less or equal runnb \n\
     -R run         : select only files with run number greater or equal runnb \n\
     -t type        : select only files of specific type \n\n";
}

// ------------------------------------------------------------------------------------
int main( int argc, char **argv ) {

  std::string outfile;
  std::string path;
  std::vector<std::string> entrynames; 

  int   rc;
  char* cp;

  if(( cp = (char*) strrchr( argv[0], '/' )) != NULL ) {
    ++cp;
  }
  else {
    cp = argv[0];
  }

  if( argc > 1 ) {
    while(( rc = getopt( argc, argv, "R:f:ho:p:r:t:" )) != EOF ) {
      switch( rc ) {
      case 'h':
        usage(cp);
	return(0);
	break;
      case 'f':
        from      = atoi(optarg);
        break;
      case 'o':
        sortmode  = optarg;
        break;
      case 'p':
        pagesize  = atoi(optarg);
        break;
      case 'R':
	flrunge   = true;
        selrunge  = atoi(optarg);
        break;
      case 'r':
	flrunle   = true;
        selrunle  = atoi(optarg);
        break;
      case 't':
	fltype    = true;
        seltype   = optarg;
        break;
      case '?':
        return(-1);
	break;
      default:
	break;
      }
    }  
  }
  else {
    usage( cp );
    return (-1);
  }
  
  if( optind < argc ) {
    path = argv[optind];
  }
  else {
    std::cerr << "No directory specified" << std::endl;
    return (-1);
  }

  if( getentries( path, entrynames ) == 0 ) {

    select( path, entrynames );

  }
  
  return 0;
}
