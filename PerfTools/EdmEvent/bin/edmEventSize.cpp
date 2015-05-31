/** measure branch sizes
 *
 *
 */

#include "PerfTools/EdmEvent/interface/EdmEventSize.h"


#include <boost/program_options.hpp>
#include <string>
#include <iostream>
#include <fstream>

#include <TROOT.h>
#include <TSystem.h>
#include <TError.h>
#include "FWCore/FWLite/interface/FWLiteEnabler.h"

static const char * const kHelpOpt = "help";
static const char * const kHelpCommandOpt = "help,h";
static const char * const kDataFileOpt = "data-file";
static const char * const kDataFileCommandOpt = "data-file,d";
static const char * const kTreeNameOpt = "tree-name";
static const char * const kTreeNameCommandOpt = "tree-name,n";
static const char * const kOutputOpt = "output";
static const char * const kOutputCommandOpt = "output,o";
static const char * const kAutoLoadOpt ="auto-loader";
static const char * const kAutoLoadCommandOpt ="auto-loader,a";
static const char * const kPlotOpt ="plot";
static const char * const kPlotCommandOpt ="plot,p";
static const char * const kSavePlotOpt ="save-plot";
static const char * const kSavePlotCommandOpt ="save-plot,s";
static const char * const kPlotTopOpt ="plot-top";
static const char * const kPlotTopCommandOpt ="plot-top,t";
static const char * const kVerboseOpt = "verbose";
static const char * const kVerboseCommandOpt = "verbose,v";
static const char * const kAlphabeticOrderOpt ="alphabetic-order";
static const char * const kAlphabeticOrderCommandOpt ="alphabetic-order,A";
static const char * const kFormatNamesOpt ="format-names";
static const char * const kFormatNamesCommandOpt ="format-names,F";

int main( int argc, char * argv[] ) {
  using namespace boost::program_options;
  using namespace std;

  string programName( argv[ 0 ] );
  string descString( programName );
  descString += " [options] ";
  descString += "data_file \nAllowed options";
  options_description desc( descString );

  desc.add_options()
    ( kHelpCommandOpt, "produce help message" )
    ( kAutoLoadCommandOpt, "automatic library loading (avoid root warnings)" )
    ( kDataFileCommandOpt, value<string>(), "data file" )
    ( kTreeNameCommandOpt, value<string>(), "tree name (default \"Events\")" )
    ( kOutputCommandOpt, value<string>(), "output file" )
    ( kAlphabeticOrderCommandOpt, "sort by alphabetic order (default: sort by size)" )
    ( kFormatNamesCommandOpt, "format product name as \"product:label (type)\" (default: use full branch name)" )
    ( kPlotCommandOpt, value<string>(), "produce a summary plot" )
    ( kPlotTopCommandOpt, value<int>(), "plot only the <arg> top size branches" )
    ( kSavePlotCommandOpt, value<string>(), "save plot into root file <arg>" )
    ( kVerboseCommandOpt, "verbose printout" );

  positional_options_description p;

  p.add( kDataFileOpt, -1 );

  variables_map vm;
  try {
    store( command_line_parser(argc,argv).options(desc).positional(p).run(), vm );
    notify( vm );
  } catch( const error& ) {
    return 7000;
  }

  if( vm.count( kHelpOpt ) ) {
    cout << desc <<std::endl;
    return 0;
  }

  if( ! vm.count( kDataFileOpt ) ) {
    cerr << programName << ": no data file given" << endl;
    return 7001;
  }

  gROOT->SetBatch();

  if( vm.count( kAutoLoadOpt ) != 0 ) {
    gSystem->Load( "libFWCoreFWLite" );
    FWLiteEnabler::enable();
  }
  else 
    gErrorIgnoreLevel = kError; 

  bool verbose = vm.count( kVerboseOpt ) > 0;


  std::string fileName = vm[kDataFileOpt].as<string>();

  std::string treeName = "Events";
  if ( vm.count( kTreeNameOpt) )
    treeName=vm[kTreeNameOpt].as<string>();

  perftools::EdmEventSize me;
  
  try {
    me.parseFile(fileName,treeName);
  } catch(perftools::EdmEventSize::Error const & error) {
    std::cerr <<  programName << ":" << error.descr << std::endl;
    return error.code;
  } 

  if ( vm.count( kFormatNamesOpt) )
    me.formatNames();

  if ( vm.count( kAlphabeticOrderOpt ) )
    me.sortAlpha();

  if (verbose) {
    std::cout << std::endl;
    me.dump(std::cout);
    std::cout << std::endl;
  }

  if (vm.count( kOutputOpt )) {
    std::ofstream of(vm[kOutputOpt].as<std::string>().c_str());
    me.dump(of); of << std::endl;
  }

  bool plot = ( vm.count( kPlotOpt ) > 0 );
  bool save = ( vm.count( kSavePlotOpt ) > 0 );
  if (plot||save) {

    std::string plotName;
    std::string histName; 
    if( plot ) plotName = vm[kPlotOpt].as<string>();
    if( save ) histName = vm[kSavePlotOpt].as<string>();
    int top=0;
    if( vm.count( kPlotTopOpt ) > 0 ) top = vm[ kPlotTopOpt ].as<int>();
    me.produceHistos(plotName,histName,top);
    

  }

  return 0;
}
