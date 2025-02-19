#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <map>
#include <cmath>
#include <getopt.h>
#include <sstream>
#include <cstdlib>

int main(int argc, char **argv) {
  // http://manpages.ubuntu.com/manpages/intrepid/fr/man3/getopt.html

  double factor = 0.995; 
  std::string file = "TimingInfo.txt";
  int opt;
  std::string oval="";
  while ((opt = getopt(argc, argv, "on:t:")) != -1) {
    switch (opt) {
    case 'n':
      file=std::string(optarg);
      break;
    case 't':
      {
	std::istringstream fraction(optarg);      
	fraction >> factor; 
	factor*=0.01;
	break	 ;
      }
    case 'o':
      oval="[OVAL]" ;
      break;
    default: 
      fprintf(stderr, "Usage: %s -t [fraction of CPU in percentage] [-n filename TimingInfo.txt] [-o for oval] \n",
	      argv[0]);
      exit(EXIT_FAILURE);
    }
  }

  std::cout << " Analyzing time report from " << file << " with CPU fraction correction " << factor << std::endl;

  std::map<std::string,double> timingPerModule, timingPerLabel;
  std::map<unsigned,double> timingPerEvent;
  std::ifstream myTimingFile(file.c_str(),std::ifstream::in);
  std::string dummy1, label, module;
  double timing;
  unsigned idummy1,evt;

  // If the machine is busy, the factor is not 100%.
  //  double factor = 0.995;

  if ( myTimingFile ) {
    while ( !myTimingFile.eof() ) { 
      myTimingFile >> dummy1 >> evt >> idummy1 >> label >> module >> timing ;
      // std::cout << evt << " " << module << " " << timing << std::endl;
      timingPerEvent[evt] += timing * factor * 1000.;	
      if ( evt != 1 ) {
	timingPerModule[module] += timing * factor * 1000.;
	timingPerLabel[module+":"+label] += timing * factor * 1000.;
      }	
    }
  } else {
    std::cout << "File " << file << " does not exist!" << std::endl;
  }

  std::map<std::string,double>::const_iterator modIt = timingPerModule.begin();
  std::map<std::string,double>::const_iterator labIt = timingPerLabel.begin();
  std::map<std::string,double>::const_iterator modEnd = timingPerModule.end();
  std::map<std::string,double>::const_iterator labEnd = timingPerLabel.end();
  std::map<double,std::string> modulePerTiming;
  std::map<double,std::string> labelPerTiming;

  for ( ; modIt != modEnd; ++modIt ) {
    double time = modIt->second/((double)evt-1.);
    std::string name = modIt->first;
    modulePerTiming[time] = name;
  }
    
  for ( ; labIt != labEnd; ++labIt ) {
    double time = labIt->second/((double)evt-1.);
    std::string name = labIt->first;
    labelPerTiming[time] = name;
  }
    
  std::map<double,std::string>::const_reverse_iterator timeIt = modulePerTiming.rbegin();
  std::map<double,std::string>::const_reverse_iterator timeEnd = modulePerTiming.rend();
  std::map<double,std::string>::const_reverse_iterator timeIt2 = labelPerTiming.rbegin();
  std::map<double,std::string>::const_reverse_iterator timeEnd2 = labelPerTiming.rend();

  std::cout << "Timing per module " << std::endl;
  std::cout << "================= " << std::endl;
  double totalTime = 0.;
  unsigned i=1;
  for ( ; timeIt != timeEnd; ++timeIt ) {

    totalTime += timeIt->first;
    std::cout << oval << " " << std::setw(3) << i++ 
	      << std::setw(50) << timeIt->second << " : " 
	      << std::setw(7) << std::setprecision(3) << timeIt-> first << " ms/event"
	      << std::endl;
  }
  std::cout << "Total time = " << totalTime << " ms/event " << std::endl;

  std::cout << "================= " << std::endl;
  std::cout << "Timing per label  " << std::endl;
  std::cout << "================= " << std::endl;
  totalTime = 0.;
  i = 1;
  for ( ; timeIt2 != timeEnd2; ++timeIt2 ) {

    totalTime += timeIt2->first;
    std::cout << oval << " " << std::setw(3) << i++ 
	      << std::setw(100) << timeIt2->second << " : " 
	      << std::setw(7) << std::setprecision(3) << timeIt2-> first << " ms/event"
	      << std::endl;
  }
  std::cout << "================= " << std::endl;
  std::cout << "Total time = " << totalTime << " ms/event " << std::endl;

  std::map<unsigned,double>::const_iterator eventIt = timingPerEvent.begin();
  std::map<unsigned,double>::const_iterator eventEnd = timingPerEvent.end();
  double minEv = 9999.;
  double maxEv = 0.;
  double rms = 0.;
  double mean = 0.;
  
  for ( ; eventIt != eventEnd; ++eventIt ) { 
    if ( eventIt->first == 1 ) continue;
    double timeEv = eventIt->second;
    if ( eventIt->second > maxEv ) maxEv = timeEv;
    if ( eventIt->second < minEv ) minEv = timeEv;
    mean += timeEv;
    rms += timeEv*timeEv;    
  }

  mean /= (double)evt-1.;
  rms /= (double)evt-1.;
  rms = std::sqrt(rms-mean*mean);
  std::cout << "Total time = " << mean << " +/- " << rms << " ms/event" << std::endl;
  std::cout << "Min.  time = " << minEv << " ms/event" << std::endl;
  std::cout << "Max.  time = " << maxEv << " ms/event" << std::endl;
}

