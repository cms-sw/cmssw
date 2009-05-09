// $Id: SprAddBaggersApp.cc,v 1.1 2007/10/30 18:56:09 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedBagger.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClassifierReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStringParser.hh"

#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <cassert>
#include <cstdio>

using namespace std;


void help(const char* prog) 
{
  cout << "Usage:  " << prog 
       << " list_of_bagger_configuration_files"
       << " output_file_for_overall_bagger" 
       << endl;
  cout << "Example: " << prog 
       << " \'bagger_1.spr,bagger_2.spr,bagger_3.spr\'" 
       << " bagger_total.spr" << endl;
  cout << "\t Options: " << endl;
  cout << "\t-h --- help                                        " << endl;
  cout << "\t-v verbosity level                                 " << endl;
}


int main(int argc, char ** argv)
{
  // check command line
  if( argc < 3 ) {
    help(argv[0]);
    return 1;
  }

  // init
  int verbose = 0;

  // decode command line
  int c;
  extern char* optarg;
  //  extern int optind;
  while((c = getopt(argc,argv,"hv:")) != EOF ) {
    switch( c )
      {
      case 'h' :
	help(argv[0]);
	return 1;
      case 'v' :
	verbose = (optarg==0 ? 0 : atoi(optarg));
	break;
      }
  }

  // Must have 2 arguments on the command line
  string inputFileList = argv[argc-2];
  string outputFile    = argv[argc-1];
  if( inputFileList.empty() ) {
    cerr << "No input Bagger configuration files are specified." << endl;
    return 1;
  }
  if( outputFile.empty() ) {
    cerr << "No output Bagger file is specified." << endl;
    return 1;
  }

  // get classifier files
  vector<vector<string> > inputFiles;
  SprStringParser::parseToStrings(inputFileList.c_str(),inputFiles);
  if( inputFiles.empty() || inputFiles[0].empty() ) {
    cerr << "Unable to parse input file list: " 
	 << inputFileList.c_str() << endl;
    return 1;
  }
  int nTrained = inputFiles[0].size();

  // read classifier configuration
  SprTrainedBagger* total = 0;
  for( int i=0;i<nTrained;i++ ) {

    // read
    SprAbsTrainedClassifier* absTrained 
      = SprClassifierReader::readTrained(inputFiles[0][i].c_str(),verbose);
    if( absTrained == 0 ) {
      cerr << "Unable to read classifier configuration from file "
	   << inputFiles[0][i].c_str() << endl;
      delete absTrained;
      delete total;
      return 2;
    }

    // check type and downcast
    SprTrainedBagger* current = 0;
    if( absTrained->name() == "Bagger" )
      current = static_cast<SprTrainedBagger*>(absTrained);
    else {
      cerr << "Fetched classifier is not Bagger. Cannot add." << endl;
      delete absTrained;
      delete total;
      return 2;
    }
    cout << "Read classifier " << current->name().c_str()
	 << " with dimensionality " << current->dim() 
	 << " and " << current->nClassifiers() << " classifiers." << endl;

    // add current to total
    if( i > 0 ) {
      *total += *current;
      delete current;
    }
    else
      total = current;
  }

  // show the result
  assert( total != 0 );
  cout << "Obtained final Bagger with " << total->nClassifiers() 
       << " classifiers." << endl;

  // store the final bagger
  if( !total->store(outputFile.c_str()) ) {
    cerr << "Unable to store final Bagger into file " 
	 << outputFile.c_str() << endl;
    return 3;
  }
  cout << "Final Bagger has been stored in file " 
       << outputFile.c_str() << endl;

  // cleanup
  delete total;

  // exit
  return 0;
}
