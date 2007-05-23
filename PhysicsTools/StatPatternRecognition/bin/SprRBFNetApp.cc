//$Id: SprRBFNetApp.cc,v 1.3 2006/11/26 02:04:31 narsky Exp $
/*
  Reads RBF net and stores its output into an output tuple.
*/

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedRBF.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprData.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprSimpleReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDataFeeder.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprMyWriter.hh"

#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>

using namespace std;


void help(const char* prog) 
{
  cout << "Usage:  " << prog 
       << " training_data_file " 
       << " net_configuration_file " << endl;
  cout << "\t Options: " << endl;
  cout << "\t-h --- help                                        " << endl;
  cout << "\t-o output Tuple file                                 " << endl;
  cout << "\t-a input ascii file mode (see SprSimpleReader.hh) " << endl;
}


int main(int argc, char ** argv)
{
  // check command line
  if( argc < 2 ) {
    help(argv[0]);
    return 1;
  }

  // init
  string hbkFile;
  int readMode = 1;
  
  // decode command line
  int c;
  extern char* optarg;
  //  extern int optind;
  while( (c = getopt(argc,argv,"ho:a:")) != EOF ) {
    switch( c )
      {
      case 'h' :
	help(argv[0]);
	return 1;
      case 'o' :
	hbkFile = optarg;
	break;
      case 'a' :
	readMode = (optarg==0 ? 1 : atoi(optarg));
	break;
      }
  }

  // There have to be 2 arguments after all options.
  string trFile = argv[argc-2];
  string netFile = argv[argc-1];
  if( trFile.empty() ) {
    cerr << "No training file is specified." << endl;
    return 1;
  }
  if( netFile.empty() ) {
    cerr << "No net file is specified." << endl;
    return 1;
  }

  // read training data from file
  SprSimpleReader reader(readMode);
  auto_ptr<SprAbsFilter> filter(reader.read(trFile.c_str()));
  if( filter.get() == 0 ) {
    cerr << "Unable to read data from file " << trFile.c_str() << endl;
    return 2;
  }
  vector<string> vars;
  filter->vars(vars);
  cout << "Read data from file " << trFile.c_str() 
       << " for variables";
  for( int i=0;i<vars.size();i++ ) 
    cout << " \"" << vars[i].c_str() << "\"";
  cout << endl;
  cout << "Total number of points read: " << filter->size() << endl;
  cout << "Points in class 0: " << filter->ptsInClass(0)
       << " 1: " << filter->ptsInClass(1) << endl;

  // read net
  SprTrainedRBF net;
  if( !net.readNet(netFile.c_str()) ) {
    cerr << "Unable to read net file " << netFile.c_str() << endl;
    return 3;
  }
  else {
    cout << "Read net configuration file:" << endl;
    net.print(cout);
  }

  // make histogram if requested
  if( hbkFile.empty() ) return 0;

  // make a writer
  SprMyWriter hbk("training");
  if( !hbk.init(hbkFile.c_str()) ) {
    cerr << "Unable to open output file " << hbkFile.c_str() << endl;
    return 4;
  }

  // feed
  SprDataFeeder feeder(filter.get(),&hbk);
  feeder.addClassifier(&net,"rbf");
  if( !feeder.feed(1000) ) {
    cerr << "Cannot feed data into file " << hbkFile.c_str() << endl;
    return 5;
  }

  // exit
  return 0;
}
