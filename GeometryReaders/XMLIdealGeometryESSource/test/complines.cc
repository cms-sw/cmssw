
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>

int main (int argc, char *argv[]) {
  std::string fname1, fname2;
  // assume two arguments NO MATTER WHAT! crash otherwise... who cares?
  if ( argc == 3 ) {
    //    std::cout << "got two" << std::endl;
    fname1 = argv[1];
    fname2 = argv[2];
    //    std::cout << fname1 << " and " << fname2 << std::endl;
  } else {
    std::cout << "need two arguments (filenames) to compare dumpGeoHistory" 
	      << std::endl;
      return 1;
  }

  std::ifstream f1(fname1.c_str());
  std::ifstream f2(fname2.c_str());
  std::string l1, l2, ts;
  std::vector<double> t1(3), t2(3), r1(9), r2(9);

  while ( !f1.eof() && !f2.eof() ) {
    getline(f1, l1, ',');
    getline(f2, l2, ',');
    if ( l1 != l2 ) {
      std::cout << "Lines don't match or are out of synchronization."
		<< "  The difference is much bigger than just the numbers,"
		<< " actual parts are missing or added.  This program "
		<< " does not handle this at this time... use diff first."
		<< std::endl
		<< "["<<l1 <<"]"<< std::endl
		<< "["<<l2 <<"]"<< std::endl
		<< std::endl;
      return 1;
    }
//     std::cout << "1================================" << std::endl;
//     std::cout << "["<<l1 <<"]"<< std::endl
// 	      << "["<<l2 <<"]"<< std::endl;
//     std::cout << "================================" << std::endl;

    size_t i = 0;
    while ( i < 3 ) {
      ts.clear();
      getline(f1,ts,',');
      //	std::cout << ts << std::endl;
      std::istringstream s1 (ts);
      s1>>t1[i++];
    }

    i=0;
    while ( i < 8 ) {
      ts.clear();
      getline(f1,ts,',');
      //	std::cout << ts << std::endl;
      std::istringstream s1 (ts);
      s1>>r1[i++];
    }
    ts.clear();
    getline(f1, ts);
    std::istringstream s2(ts);
    s2>>r1[8];

    i=0;
    while ( i < 3 ) {
      ts.clear();
      getline(f2,ts,',');
      //	std::cout << ts << std::endl;
      std::istringstream s1 (ts);
      s1>>t2[i++];
    }

    i=0;
    while ( i < 8 ) {
      ts.clear();
      getline(f2,ts,',');
      //	std::cout << ts << std::endl;
      std::istringstream s1 (ts);
      s1>>r2[i++];
    }
    ts.clear();
    getline(f2, ts);
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
      if ( std::fabs(t1[i] - t2[i]) > 0.0000001 ) cerrorind[i] = true;
      else cerrorind[i] = false;
    }
    if ( cerrorind[0] || cerrorind[1] || cerrorind[2] ) std::cout << l1;
    for ( i=0; i<3; ++i ) {
      if ( cerrorind[i] ) {
	std::cout << " coordinate ";
	if ( i == 0 ) std::cout << "x ";
	else if ( i == 1 ) std::cout << "y ";
	else if ( i == 2 ) std::cout << "z ";
	std::cout << " is different by: " << std::setw(13)
		  << std::setprecision(7) << std::fixed << t1[i] - t2[i] 
		  << " mm ";
      }
    }
    bool rerror(false);
    for (i=0; i < 9; i++) {
//       std::cout << "r1[" << i << "] = " << r1[i]
// 		<< " r2[" << i << "] = " << r2[i] << std::endl;
      if ( std::fabs(r1[i] - r2[i]) > 0.0000001 ) rerror = true;
    }
    if ( rerror && !cerrorind[0] && !cerrorind[1] && !cerrorind[2] ) {
      std::cout << l1 << " ";
    }
    if ( rerror ) {
      for (i=0; i < 9; i++) {
	// 	std::cout << "r1[" << i << "] = " << r1[i]
	// 		  << " r2[" << i << "] = " << r2[i] << std::endl;
	if ( std::fabs(r1[i] - r2[i]) > 0.0000001 ) {
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
    //       getline(f1, l1, ',');
    //       getline(f2, l2, ',');
//     if ( f1.peek() != 10 ) {
//       std::cout << "not eol on f1 doing an extra getline()" << std::endl;
//       getline(f1,ts);
//       std::cout << "why this is left?" << ts << std::endl;
//     }
//     if ( f2.peek() != 10 ) {
//       std::cout << "not eol on f2 doing an extra getline()" << std::endl;
//       getline(f2,ts);
//       std::cout << "why this is left?" << ts << std::endl;
//     }
  }
  return 0;
}
