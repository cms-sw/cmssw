// -*- C++ -*-
//
// Package:    TestCompareDDDumpFiles
// Class:      TestCompareDDDumpFiles
// 
/**\class TestCompareDDDumpFiles TestCompareDDDumpFiles.cc test/TestCompareDDDumpFiles/src/TestCompareDDDumpFiles.cc

 Description: Compares two geoHistory dump files 

 Implementation:
     Read two files with a certain format and compare each line.  If lines are out of sync stop.
**/
//
// Original Author:  Michael Case
//         Created:  Thu Sep 10, 2009
// $Id: TestCompareDDDumpFiles.cc,v 1.2 2009/09/16 22:22:46 case Exp $
//
//

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

class TestCompareDDDumpFiles : public edm::EDAnalyzer {
public:
  explicit TestCompareDDDumpFiles( const edm::ParameterSet& );
  ~TestCompareDDDumpFiles();
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
private:
  std::string fname1_;
  std::string fname2_;
  double tol_;
  std::ifstream f1_;
  std::ifstream f2_;
};

TestCompareDDDumpFiles::TestCompareDDDumpFiles( const edm::ParameterSet& ps ) 
  :    fname1_(ps.getParameter<std::string>("dumpFile1"))
  ,fname2_(ps.getParameter<std::string>("dumpFile2"))
  ,tol_(ps.getUntrackedParameter<double>("tolerance", 0.000001))
  ,f1_(fname1_.c_str(), std::ios::in)
  ,f2_(fname2_.c_str(), std::ios::in)
{ 
  if (!f1_ || !f2_) {
    throw cms::Exception("MissingFileDDTest") << fname1_ << " and/or " << fname2_ << " do not exist." ;
  }
}

TestCompareDDDumpFiles::~TestCompareDDDumpFiles () {
  f1_.close();
  f2_.close();
}

void TestCompareDDDumpFiles::analyze( const edm::Event&, const edm::EventSetup& ) {
  
  std::string l1, l2, ts;
  std::vector<double> t1(3), t2(3), r1(9), r2(9);
  double diffv;
  while ( !f1_.eof() && !f2_.eof() ) {
    getline(f1_, l1, ',');
    getline(f2_, l2, ',');
    if ( l1 != l2 ) {
      std::cout << "Lines don't match or are out of synchronization."
		<< "  The difference is much bigger than just the numbers,"
		<< " actual parts are missing or added.  This program "
		<< " does not handle this at this time... use diff first."
		<< std::endl
		<< "["<<l1 <<"]"<< std::endl
		<< "["<<l2 <<"]"<< std::endl
		<< std::endl;
      break;
    }
//     std::cout << "1================================" << std::endl;
//     std::cout << "["<<l1 <<"]"<< std::endl
// 	      << "["<<l2 <<"]"<< std::endl;
//     std::cout << "================================" << std::endl;

    size_t i = 0;
    while ( i < 3 ) {
      ts.clear();
      getline(f1_,ts,',');
      //	std::cout << ts << std::endl;
      std::istringstream s1 (ts);
      s1>>t1[i++];
    }

    i=0;
    while ( i < 8 ) {
      ts.clear();
      getline(f1_,ts,',');
      //	std::cout << ts << std::endl;
      std::istringstream s1 (ts);
      s1>>r1[i++];
    }
    ts.clear();
    getline(f1_, ts);
    std::istringstream s2(ts);
    s2>>r1[8];

    i=0;
    while ( i < 3 ) {
      ts.clear();
      getline(f2_,ts,',');
      //	std::cout << ts << std::endl;
      std::istringstream s1 (ts);
      s1>>t2[i++];
    }

    i=0;
    while ( i < 8 ) {
      ts.clear();
      getline(f2_,ts,',');
      //	std::cout << ts << std::endl;
      std::istringstream s1 (ts);
      s1>>r2[i++];
    }
    ts.clear();
    getline(f2_, ts);
    std::istringstream s3(ts);
    s3>>r2[8];

//     std::cout << "2================================" << std::endl;
//     std::cout << "["<<l1 <<"]"<< std::endl
// 	      << "["<<l2 <<"]"<< std::endl;
//     std::cout << "ts = " << ts << std::endl;
//     std::cout << "================================" << std::endl;
    //      std::cout << "l1=" << l1 << std::endl;
    std::vector<bool> cerrorind;
    cerrorind.reserve(3);
    for (i=0; i < 3; i++) {
//       std::cout << std::setw(13)
// 		<< std::setprecision(7) << std::fixed 
// 		<< "t1[" << i << "] = " << t1[i]
// 		<< " t2[" << i << "] = " << t2[i] << std::endl;
      diffv = std::fabs(t1[i] - t2[i]);
      if ( diffv > tol_ ) cerrorind[i] = true;
      else cerrorind[i] = false;
    }
    if ( cerrorind[0] || cerrorind[1] || cerrorind[2] ) std::cout << l1;
    for ( i=0; i<3; ++i ) {
      diffv = std::fabs(t1[i] - t2[i]);
      if ( cerrorind[i] &&  diffv > tol_ ) {
	std::cout << " coordinate ";
	if ( i == 0 ) std::cout << "x ";
	else if ( i == 1 ) std::cout << "y ";
	else if ( i == 2 ) std::cout << "z ";
	std::cout << " is different by: " << std::setw(13)
		  << std::setprecision(7) << std::fixed << diffv
		  << " mm ";
      }
    }
    bool rerror(false);
    for (i=0; i < 9; i++) {
//       std::cout << "r1[" << i << "] = " << r1[i]
// 		<< " r2[" << i << "] = " << r2[i] << std::endl;
	diffv=std::fabs(r1[i] - r2[i]);
      if ( diffv > tol_ ) rerror = true;
    }
    if ( rerror && !cerrorind[0] && !cerrorind[1] && !cerrorind[2] ) {
      std::cout << l1 << " ";
    }
    if ( rerror ) {
      for (i=0; i < 9; i++) {
	// 	std::cout << "r1[" << i << "] = " << r1[i]
	// 		  << " r2[" << i << "] = " << r2[i] << std::endl;
	diffv=std::fabs(r1[i] - r2[i]);
	if ( diffv > tol_ ) {
	  //	  std::cout << std::endl;
	  std::cout << " index " << i << " of rotation matrix differs by " 
		    << std::setw(13) << std::setprecision(7) 
		    << std::fixed << r1[i] - r2[i];
	}
      }
      std::cout << std::endl;
    } else if ( cerrorind[0] || cerrorind[1] || cerrorind[2]) {
      std::cout << std::endl;
    }
//     std::cout << "3================================" << std::endl;
//     std::cout << "["<<l1 <<"]"<< std::endl
// 	      << "["<<l2 <<"]"<< std::endl;
//     std::cout << "================================" << std::endl;
    //       getline(f1_, l1, ',');
    //       getline(f2_, l2, ',');
//     if ( f1_.peek() != 10 ) {
//       std::cout << "not eol on f1_ doing an extra getline()" << std::endl;
//       getline(f1_,ts);
//       std::cout << "why this is left?" << ts << std::endl;
//     }
//     if ( f2_.peek() != 10 ) {
//       std::cout << "not eol on f2_ doing an extra getline()" << std::endl;
//       getline(f2_,ts);
//       std::cout << "why this is left?" << ts << std::endl;
//     }
  }

}
//define this as a plug-in
DEFINE_FWK_MODULE(TestCompareDDDumpFiles);
