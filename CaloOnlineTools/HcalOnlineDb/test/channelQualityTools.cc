#include <iostream>
#include <unistd.h>
#include <getopt.h>
#include <string.h>
#include <fstream>
#include <boost/program_options.hpp>

#include "CaloOnlineTools/HcalOnlineDb/interface/HcalChannelQualityXml.h"

namespace po = boost::program_options;

// returns 0 if success,
// returns -1 in case of a problem
int main( int argc, char **argv )
{
  //cout << "Running xmlTools..." << endl;

  //
  //===> command line options parser using boost  
  //
  int run_number, iov_begin, iov_end;
  std::string comment;
  po::options_description general("General options");
  general.add_options()
    ("help", "produce help message")
    ("quicktest", "Quick feature testing")
    ("make-xml-from-stdin","reads channel status from stdin in the ASCII format, generates XML for loading to OMDS and outputs XML to stdout")
    ("get-baseline-from-omds-to-stdout","reads channel status from OMDS for a given tag and IOV begin and outputs it to stdout in the ASCII format")
    ("dump-tags","dumps available channel quality tags from OMDS to stdout, newest first")
    ("dump-iovs","dumps available IOVs for a given channel quality tag from OMDS to stdout, newest first")
    ("tag-name", po::value<string>(), "tag name")
    ("run-number", po::value<int>(&run_number)->default_value( 1 ), "run number")
    ("iov-begin", po::value<int>(&iov_begin)->default_value( 1 ), "beginning of the interval of validity, units: run numbers")
    ("iov-end", po::value<int>(&iov_end)->default_value( -1 ), "end of the interval of validity, units: run numbers")
    ("comment", po::value<string>(&comment)->default_value(""), "comment field. If left empty, it will be generated for you")
    ;

  try{
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, general), vm);
    po::notify(vm);
    
    if (vm.count("help")) {
      cout << general << "\n";
      return 1;
    }


    if (vm.count("quicktest")) {
      return 0;
    }

    if (vm.count("make-xml-from-stdin")){
      std::string _tag;
      if ( !vm.count("tag-name") ){
	cerr << "No tag name specified, impossible to proceed, exiting..." << endl;
	exit(-1);
      }
      else{
	_tag = vm["tag-name"].as<string>();
      }
      HcalChannelQualityXml cq;
      if (comment.size()<1){
	HcalAssistant ass;
	comment = ass.getRandomQuote();
      }
      cq.makeXmlFromAsciiStream(run_number,
				iov_begin,
				iov_end,
				_tag,
				comment
			     );
      return 0;
    }

    if (vm.count("get-baseline-from-omds-to-stdout")){
      std::string _tag;
      if ( !vm.count("tag-name") ){
	cerr << "No tag name specified, impossible to proceed, exiting..." << endl;
	exit(-1);
      }
      else{
	_tag = vm["tag-name"].as<string>();
      }
      HcalChannelQualityXml cq;
      cq.writeBaseLineFromOmdsToStdout(_tag, iov_begin);      
      return 0;
    }

    if (vm.count("dump-tags")) {
      HcalChannelQualityXml xml;
      xml.dumpTagsFromOmdsToStdout();
    }


    if (vm.count("dump-iovs")) {
      std::string _tag;
      if ( !vm.count("tag-name") ){
	cerr << "No tag name specified, impossible to proceed, exiting..." << endl;
	exit(-1);
      }
      else{
	_tag = vm["tag-name"].as<string>();
      }
      HcalChannelQualityXml xml;
      xml.dumpIovsFromOmdsToStdout(_tag);
    }


  } catch(boost::program_options::unknown_option) {
    cout << "Unknown option..." << endl;
    cout << general << "\n";
  }
  exit (0);  
}

