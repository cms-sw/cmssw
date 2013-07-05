#include "prepareMagneticFieldGrid.h"

#include "MagneticField/Interpolation/src/VectorFieldInterpolation.h"
#include "MagneticField/Interpolation/src/binary_ofstream.h"
#include "DataFormats/GeometryVector/interface/Pi.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <iostream>

using namespace std;

void prepareMagneticFieldGrid::countTrueNumberOfPoints(const std::string& name) const {

  int nLines = 0;
  int nPoints = 0;


  // define vectors of IndexedDoubleVectors
  IndexedDoubleVector         XBVector;
  std::vector<IndexedDoubleVector> XBValues;

  // read file and copy to a vector
  double epsilonRadius = 1e-8;
  std::ifstream file(name.c_str());
  string line;
  if (file.good()) {
    while (getline(file,line)) {
      double x1,x2,x3,Bx,By,Bz,perm;//,poten;
      stringstream linestr;
      linestr << line;      
      linestr >> x1 >> x2 >> x3 >> Bx >> By >> Bz >> perm;;// >> poten;
      if (file){
        XBVector.putV6(x1, x2, x3, Bx, By, Bz);
// 	if (nLines<5) {
// 	  double tx1, tx2, tx3, tBx, tBy, tBz;
// 	  XBVector.getV6(tx1, tx2, tx3, tBx, tBy, tBz);
// 	  cout << "Read: " <<  setprecision(12) << Bx << " " << tBx << endl;
// 	}
        XBValues.push_back(XBVector);
        ++nLines;
      }
    }
  }

  // compare point by point
  for (int iLine=0; iLine<nLines; ++iLine){
    double x1,x2,x3,Bx,By,Bz;
    XBVector = XBValues.operator[](iLine);
    XBVector.getV6(x1, x2, x3, Bx, By, Bz);
    double pnt[3] = {x1,x2,x3};
    double tmpBz = Bz;
    bool isSinglePoint = true;
    for (int i=0; i<nLines; ++i){
      if (i < iLine){
	XBVector = XBValues.operator[](i);
	XBVector.getV6(x1, x2, x3, Bx, By, Bz);
	double distPP = sqrt((pnt[0]-x1)*(pnt[0]-x1)+(pnt[1]-x2)*(pnt[1]-x2)+(pnt[2]-x3)*(pnt[2]-x3));
	if (distPP < epsilonRadius) {
	  isSinglePoint = false;
          cout << "same points(" << iLine << ") with dR = " << distPP << endl;
	  cout << "             point 1: " << iLine << " " <<  pnt[0] << " " << pnt[1] << " " << pnt[2] << " Bz: " << tmpBz << endl;
	  cout << "             point 2: " << i << " " << x1     << " " << x2     << " " << x3     << " Bz: " <<    Bz << endl;
	}
      }
    }
    if (isSinglePoint) ++nPoints;
  }

  if (PRINT) cout << "  file " << name << endl;
  if (PRINT) cout << "  # lines = " << nLines << "  # points = " << nPoints;
  if (nLines == nPoints) {
    if (PRINT) cout << "  -->  PASSED" << endl;
  }
  else {
    if (PRINT) cout << "  -->  FAILED" << endl;
  }

  return;
}


void prepareMagneticFieldGrid::fillFromFile(const std::string& name){

  double phi=(rotateSector-1)*Geom::pi()/6.;
  double sphi = sin(-phi);
  double cphi = cos(-phi);

  // check, whether coordinates are Cartesian or cylindrical
  std::string::size_type ibeg, iend;
  ibeg = name.find('-');  // first occurance of "-"
  iend = name.rfind('-'); // last  occurance of "-"
  if  ((name.substr(ibeg+1, iend-ibeg-1)) == "xyz") {
    XyzCoordinates = true;
  } else if  ((name.substr(ibeg+1, iend-ibeg-1)) == "rpz") {
    RpzCoordinates = true;
  } else {
    cout << " Unrecognized input file name: valid formats are *-xyz-* or *-rpz-*" << endl;
    abort();
  }

  // define vectors of IndexedDoubleVectors
  IndexedDoubleVector XBVector;
  std::vector<IndexedDoubleVector>  XBValues;
  std::vector<IndexedDoubleVector> IXBValues;
  std::vector<IndexedDoubleVector>  XBArray;

  // vectors for pre analysis
  std::vector<double> valX[3];
  std::vector<int>    numX[3];

  // copy ASCII file to the vector
  int nLines = 0;
  int     index[3] = {0,0,0};
  int    nSteps[3] = {0,0,0};
  double x1,x2,x3,Bx,By,Bz,perm,poten;
  std::ifstream file(name.c_str());
  if (file.good()) {
    while (file.good()){
      file >> x1 >> x2 >> x3 >> Bx >> By >> Bz >> perm >> poten;
      if (file){
	if (rotateSector>1) {
	  if (RpzCoordinates) {
	    x2 = x2-phi;
	    if (fabs(x2)>0.78) {
	      cout << "ERROR: Input coordinates do not belong to sector " << rotateSector 
		   << " : " << fabs(x2) << endl;
	      abort();
	    }
	  } else {
	    double x=x1;
	    double y=x2;	    
	    x1 = x*cphi-y*sphi;
	    x2 = x*sphi+y*cphi;
	    if (fabs(atan2(x2,x1))>0.78) {
	      cout << "ERROR: Input coordinates do not belong to sector " << rotateSector 
		   << " : " << atan2(x2,x1) << endl;
	      abort();
	    }
	  }
	  double Bx0 = Bx;
	  double By0 = By;	  
	  Bx = Bx0*cphi-By0*sphi;
	  By = Bx0*sphi+By0*cphi;
	  
	}
	
	//	cerr << fixed << x1 << " " <<  x2 << " " <<  x3 << " " <<  Bx << " " <<  By << " " <<  Bz << " " <<  perm << " " <<  poten << " " << endl;

        XBVector.putI3(index[0], index[1], index[2]);
        XBVector.putV6(x1, x2, x3, Bx, By, Bz);
        XBValues.push_back(XBVector);
        ++nLines;
        // pre analyze file content
        double pnt[3] = {x1,x2,x3};
        if (nLines == 1){
	  for (int i=0; i<3; ++i){
	    valX[i].push_back(pnt[i]);
	    numX[i].push_back(1);
	  }
        }
        else{
	  for (int i=0; i<3; ++i){
            bool knownValue = false;
	    for (int j=0; j<(nSteps[i]+1); ++j){
	      if (std::abs(valX[i].operator[](j) - pnt[i]) < EPSILON) knownValue = true;
	      if (std::abs(valX[i].operator[](j) - pnt[i]) < EPSILON) numX[i].operator[](j)+=1;
	    }
	    if (!knownValue) {
	      valX[i].push_back(pnt[i]);
	      nSteps[i]++;
	      numX[i].push_back(1);
	    }
	  }
        }
      }
    }
  }
  else return;

  // PART I: FOR SIMPLE COORDINATES

  // begin simple grid (EasyCoordinate) analysis
  for (int i=0; i<3; ++i){
    // sort the different values per one coordinate
    sort(valX[i].begin(), valX[i].end());
    // calculate number of points and constant step size
    EasyCoordinate[i] = true;
    NumberOfPoints[i] = nSteps[i] + 1;
    BasicDistance0[i] = 9e9;
    double newVal = 0.0;
    double oldVal = 0.0;
    for (int j=0; j<NumberOfPoints[i]; ++j){
      newVal = valX[i].operator[](j);
      if (j ==1) BasicDistance0[i] = newVal - oldVal;
      else if (j > 1){
	if (std::abs(newVal - oldVal - BasicDistance0[i]) > EPSILON) EasyCoordinate[i] = false;
      }
      oldVal = newVal;
    }
  }

  // define indices for easy coordinates
  for (int iLine=0; iLine<nLines; ++iLine){
    XBVector = XBValues.operator[](iLine);
    XBVector.getI3(index[0], index[1], index[2]);
    XBVector.getV6(x1, x2, x3, Bx, By, Bz);
    double pnt[3] = {x1,x2,x3};
    for (int i=0; i<3; ++i){
      for (int j=0; j<NumberOfPoints[i]; ++j){
	if (EasyCoordinate[i]){
	  if (std::abs(valX[i].operator[](j) - pnt[i]) < EPSILON) index[i] = j;
	}
      }
    }
    XBVector.putI3(index[0], index[1], index[2]);
    IXBValues.push_back(XBVector);
  }
  if (!XBValues.empty()) XBValues.clear();

  // sort according to defined indices
  sort(IXBValues.begin(), IXBValues.end());

  // check, if structure is known at this point
  bool systematicGrid[3] = {false,false,false};
  KnownStructure = true;
  if (NumberOfPoints[0]*NumberOfPoints[1]*NumberOfPoints[2] != nLines) KnownStructure = false;
  for (int i=0; i<3; ++i){
    systematicGrid[i] = EasyCoordinate[i];
      if (!systematicGrid[i]) KnownStructure = false;
  }

  if (KnownStructure == true){
    for (int iLine=0; iLine<nLines; ++iLine){
      XBVector = IXBValues.operator[](iLine);
      XBArray.push_back(XBVector);
    }
  }

  // get first point = ReferencePoint (and last point for structure checking)
  if (!XBArray.empty()) XBVector = XBArray.front();
  XBVector.getV6(x1, x2, x3, Bx, By, Bz);
  double firstListPoint[3] = {x1,x2,x3};
  if (!XBArray.empty()) XBVector = XBArray.back();
  XBVector.getV6(x1, x2, x3, Bx, By, Bz);
  double secondRefPoint[3] = {x1,x2,x3};

  // recheck all coordinates
  KnownStructure = true;
  for (int i=0; i<3; ++i){
    ReferencePoint[i] = firstListPoint[i];
    // calculate step size for maximal indices
    double stepSize = BasicDistance0[i];
    for (int j=0; j<3; ++j){
      stepSize += BasicDistance1[i][j]*nSteps[j];
    }
    double offset = 0.0;
    for (int j=0; j<3; ++j){
      offset += BasicDistance2[i][j]*nSteps[j];
    }
    // calculate difference between the two reference points minus the total difference (=0)
    double totDiff = secondRefPoint[i] - (ReferencePoint[i] + offset + stepSize*nSteps[i]);
    if (std::abs(totDiff) < EPSILON*10.) systematicGrid[i] = true;
    else                         systematicGrid[i] = false;
    if (!systematicGrid[i]) KnownStructure = false;
  }

  if (KnownStructure){
    if (XyzCoordinates) GridType = 1;
    if (RpzCoordinates) GridType = 3;
    if (PRINT){
      // print result
      cout << "  read " << nLines << " lines -> grid structure:" << endl;
      cout << "  # of points:   N1 = " << NumberOfPoints[0]
                         << "   N2 = " << NumberOfPoints[1]
                         << "   N3 = " << NumberOfPoints[2] << endl; 
      cout << "  ref. point :   X1 = " << ReferencePoint[0]
                         << "   X2 = " << ReferencePoint[1]
                         << "   X3 = " << ReferencePoint[2] << endl; 
      cout << "  step parm0 :   A1 = " << BasicDistance0[0]
                         << "   A2 = " << BasicDistance0[1]
                         << "   A3 = " << BasicDistance0[2] << endl; 
      cout << "  easy grid  :   E1 = " << EasyCoordinate[0]
                         << "   E2 = " << EasyCoordinate[1]
                         << "   E3 = " << EasyCoordinate[2] << endl; 
      cout << "  structure  :   S1 = " << systematicGrid[0]
                         << "   S2 = " << systematicGrid[1]
                         << "   S3 = " << systematicGrid[2]; 
      if (KnownStructure) cout << "  -->  VALID   (step I)" << endl;
      else                cout << "  -->  INVALID (step I)" << endl;
    }
  }

  // END OF PART I: FOR SIMPLE COORDINATES
  //
  // PART II: FOR MORE COMPLICATED COORDINATES
  bool goodIndex[3] = {true,true,true};

  if (!KnownStructure){
    // analyze structure of missing coordinates
    int nEasy = 0;
    for (int i=0; i<3; ++i){
      if (EasyCoordinate[i]) ++nEasy;
    }
    std::vector<double> misValX[3];

    // try to recover info of missing coordiates
    if (nEasy > 0){
      bool lastLine = false;
      int lowerLineLimit = 0;
      int upperLineLimit = 0;
      int oldIndex[3] = {0,0,0};
      for (int iLine=0; iLine<nLines; ++iLine){
	if (iLine == (nLines-1)) lastLine = true;
        XBVector = IXBValues.operator[](iLine);
	XBVector.getI3(index[0], index[1], index[2]);
	bool newIndices = false;
        for (int i=0; i<3; ++i){
	  if (oldIndex[i] != index[i]){
	    oldIndex[i] = index[i];
	    newIndices = true;
	  }
	}
	if (newIndices){
	  for (int i=0; i<3; ++i){
	    if (!EasyCoordinate[i]) sort(misValX[i].begin(), misValX[i].end());
	  }
	  lowerLineLimit = upperLineLimit;
	  upperLineLimit = iLine;
	  for (int jLine=lowerLineLimit; jLine<upperLineLimit; ++jLine){
	    XBVector = IXBValues.operator[](jLine);
	    XBVector.getI3(index[0], index[1], index[2]);
	    XBVector.getV6(x1, x2, x3, Bx, By, Bz);
	    double pnt[3] = {x1,x2,x3};
	    for (int i=0; i<3; ++i){
	      if (!EasyCoordinate[i]){
		double thisVal = 0.; double lastVal = 0.; double deltVal = 0.;
		for (int j=0; j<NumberOfPoints[i]; ++j){
                  thisVal = misValX[i].operator[](j);
		  if (std::abs(thisVal - pnt[i]) < EPSILON) index[i] = j;
		  if (j ==1) deltVal = thisVal - lastVal;
		  else if (j > 1){
		    if (std::abs(thisVal - lastVal - deltVal) > EPSILON) goodIndex[i] = false;
		  }
		  lastVal = thisVal;
		}
	      }
	    }
	    XBVector.putI3(index[0], index[1], index[2]);
	    XBArray.push_back(XBVector);
	    for (int i=0; i<3; ++i){
	      if (!misValX[i].empty()) misValX[i].clear();
	    }
	  }
	}
        XBVector = IXBValues.operator[](iLine);
        XBVector.getV6(x1, x2, x3, Bx, By, Bz);
        double pnt[3] = {x1,x2,x3};
        for (int i=0; i<3; ++i){
          if (!EasyCoordinate[i]){
	    if (iLine == 0){
	      NumberOfPoints[i] = 1;
	      misValX[i].push_back(pnt[i]);
	    }
	    else if (newIndices){
	      NumberOfPoints[i] = 1;
	      misValX[i].push_back(pnt[i]);
	    }
	    else{
	      bool knownValue = false;
	      for (int j=0; j<NumberOfPoints[i]; ++j){
		if (std::abs(misValX[i].operator[](j) - pnt[i]) < EPSILON) knownValue = true;
	      }
	      if (!knownValue){
		++NumberOfPoints[i];
		misValX[i].push_back(pnt[i]);
	      }
	    }
	  }
	}
	if (lastLine){
	  for (int i=0; i<3; ++i){
	    if (!EasyCoordinate[i]) sort(misValX[i].begin(), misValX[i].end());
	  }
	  lowerLineLimit = upperLineLimit;
	  upperLineLimit = nLines;
	  for (int jLine=lowerLineLimit; jLine<upperLineLimit; ++jLine){
	    XBVector = IXBValues.operator[](jLine);
	    XBVector.getI3(index[0], index[1], index[2]);
	    XBVector.getV6(x1, x2, x3, Bx, By, Bz);
	    double pnt[3] = {x1,x2,x3};
	    for (int i=0; i<3; ++i){
	      if (!EasyCoordinate[i]){
		double thisVal = 0.; double lastVal = 0.; double deltVal = 0.;
		for (int j=0; j<NumberOfPoints[i]; ++j){
                  thisVal = misValX[i].operator[](j);
		  if (std::abs(thisVal - pnt[i]) < EPSILON) index[i] = j;
		  if (j ==1) deltVal = thisVal - lastVal;
		  else if (j > 1){
		    if (std::abs(thisVal - lastVal - deltVal) > EPSILON) goodIndex[i] = false;
		  }
		  lastVal = thisVal;
		}
	      }
	    }
	    XBVector.putI3(index[0], index[1], index[2]);
	    XBArray.push_back(XBVector);
	    for (int i=0; i<3; ++i){
	      if (!misValX[i].empty()) misValX[i].clear();
	    }
	  }
	}
      }
    }
    int nGoodInd = 0;
    for (int i=0; i<3; ++i){
      if (goodIndex[i]) ++nGoodInd;
    }
    // try to recover info of last missing coordiates
    if (nGoodInd < 3){
      cout << endl;
      cout << "Neasy = 1: beginning of T E S T area!" << endl;
      cout << "# good indices = " << nGoodInd << " (" << goodIndex[0] << goodIndex[1] << goodIndex[2] << ") " << endl;
      // reset wrong indices and sort
      for (int iLine=0; iLine<nLines; ++iLine){
	XBVector = XBArray.operator[](iLine);
	XBVector.getI3(index[0], index[1], index[2]);
	for (int i=0; i<3; ++i){
	  if (!goodIndex[i]) index[i] = 0;
	}
	XBVector.putI3(index[0], index[1], index[2]);
      }
      sort(XBArray.begin(), XBArray.end());

      cout << "Neasy = 1: end of T E S T area!" << endl;
      cout << endl;
    }
    // common part of missing coordiate(s) recovery (1 or 2 missing)
    if (nEasy > 0){
      // sort without resetting of indices
      sort(XBArray.begin(), XBArray.end());

      double miniGrid[3][2][2][2] = {{{{0.,0.},{0.,0.}},{{0.,0.},{0.,0.}}},
                                     {{{0.,0.},{0.,0.}},{{0.,0.},{0.,0.}}},
                                     {{{0.,0.},{0.,0.}},{{0.,0.},{0.,0.}}}};
      for (int iLine=0; iLine<nLines; ++iLine){
	XBVector = XBArray.operator[](iLine);
	XBVector.getI3(index[0], index[1], index[2]);
	XBVector.getV6(x1, x2, x3, Bx, By, Bz);
	double pnt[3] = {x1,x2,x3};
        if (index[0] < 2 && index[1] < 2 && index[2] < 2){
       	for (int i=0; i<3; ++i){
            miniGrid[i][index[0]][index[1]][index[2]] = pnt[i];
	  }
	}
      }
      // basic distances (sorry, I have found no smarter solution so far)
      // recalculate constant terms for step size from miniGrid[3][2][2][2]
      BasicDistance0[0] = miniGrid[0][1][0][0] - miniGrid[0][0][0][0];
      BasicDistance0[1] = miniGrid[1][0][1][0] - miniGrid[1][0][0][0];
      BasicDistance0[2] = miniGrid[2][0][0][1] - miniGrid[2][0][0][0];
      // now calculate linear terms for step size from miniGrid[3][2][2][2]
      double disd10 = miniGrid[0][1][1][0] - miniGrid[0][0][1][0];
      double disd00 = BasicDistance0[0];
      double disd01 = miniGrid[0][1][0][1] - miniGrid[0][0][0][1];
      BasicDistance1[0][0] = 0.0;
      BasicDistance1[0][1] = disd10 - disd00;
      BasicDistance1[0][2] = disd01 - disd00;
      double dis1d0 = miniGrid[1][1][1][0] - miniGrid[1][1][0][0];
      double dis0d0 = BasicDistance0[1];
      double dis0d1 = miniGrid[1][0][1][1] - miniGrid[1][0][0][1];
      BasicDistance1[1][0] = dis1d0 - dis0d0;
      BasicDistance1[1][1] = 0.0;
      BasicDistance1[1][2] = dis0d1 - dis0d0;
      double dis10d = miniGrid[2][1][0][1] - miniGrid[2][1][0][0];
      double dis00d = BasicDistance0[2];
      double dis01d = miniGrid[2][0][1][1] - miniGrid[2][0][1][0];
      BasicDistance1[2][0] = dis10d - dis00d;
      BasicDistance1[2][1] = dis01d - dis00d;
      BasicDistance1[2][2] = 0.0;
      // now calculate linear terms offsets from miniGrid[3][2][2][2]
      BasicDistance2[0][0] = 0.0;
      BasicDistance2[0][1] = miniGrid[0][0][1][0] - miniGrid[0][0][0][0];
      BasicDistance2[0][2] = miniGrid[0][0][0][1] - miniGrid[0][0][0][0];
      BasicDistance2[1][0] = miniGrid[1][1][0][0] - miniGrid[1][0][0][0];
      BasicDistance2[1][1] = 0.0;
      BasicDistance2[1][2] = miniGrid[1][0][0][1] - miniGrid[1][0][0][0];
      BasicDistance2[2][0] = miniGrid[2][1][0][0] - miniGrid[2][0][0][0];
      BasicDistance2[2][1] = miniGrid[2][0][1][0] - miniGrid[2][0][0][0];
      BasicDistance2[2][2] = 0.0;
      // set default values of BasicDistance0 in case of < 2 points
      for (int i=0; i<3; ++i){
        if (NumberOfPoints[i] < 2) BasicDistance0[i] = 9e9;
      }
    }

    // get first point = ReferencePoint (and last point for structure checking)
    if (!XBArray.empty()) XBVector = XBArray.front();
    XBVector.getV6(x1, x2, x3, Bx, By, Bz);
    firstListPoint[0] = x1;
    firstListPoint[1] = x2;
    firstListPoint[2] = x3;
    if (!XBArray.empty()) XBVector = XBArray.back();
    XBVector.getV6(x1, x2, x3, Bx, By, Bz);
    secondRefPoint[0] = x1;
    secondRefPoint[1] = x2;
    secondRefPoint[2] = x3;

    // recheck all coordinates
    KnownStructure = true;
    for (int i=0; i<3; ++i){
      ReferencePoint[i] = firstListPoint[i];
      nSteps[i] = NumberOfPoints[i]-1;
      // calculate step size for maximal indices
      double stepSize = BasicDistance0[i];
      for (int j=0; j<3; ++j){
        stepSize += BasicDistance1[i][j]*nSteps[j];
      }
      double offset = 0.0;
      for (int j=0; j<3; ++j){
        offset += BasicDistance2[i][j]*nSteps[j];
      }
      // calculate difference between the two reference points minus the total difference (=0)
      double totDiff = secondRefPoint[i] - (ReferencePoint[i] + offset + stepSize*nSteps[i]);
      if (std::abs(totDiff) < EPSILON*10.) systematicGrid[i] = true;
      else                         systematicGrid[i] = false;
      if (!systematicGrid[i]) KnownStructure = false;
    }

    if (KnownStructure){
      if (XyzCoordinates) GridType = 2;
      if (RpzCoordinates) GridType = 4;
      if (PRINT){
        // print result
        cout << "  read " << nLines << " lines -> grid structure:" << endl;
        cout << "  # of points:   N1 = " << NumberOfPoints[0]
                           << "   N2 = " << NumberOfPoints[1]
                           << "   N3 = " << NumberOfPoints[2] << endl; 
        cout << "  ref. point :   X1 = " << ReferencePoint[0]
                           << "   X2 = " << ReferencePoint[1]
                           << "   X3 = " << ReferencePoint[2] << endl; 
        cout << "  step parm0 :   A1 = " << BasicDistance0[0]
                           << "   A2 = " << BasicDistance0[1]
                           << "   A3 = " << BasicDistance0[2] << endl; 
        cout << "  step parm1 :  B11 = " << BasicDistance1[0][0]
                           << "  B21 = " << BasicDistance1[1][0]
                           << "  B31 = " << BasicDistance1[2][0] << endl; 
        cout << "             :  B12 = " << BasicDistance1[0][1]
                           << "  B22 = " << BasicDistance1[1][1]
                           << "  B32 = " << BasicDistance1[2][1] << endl;
        cout << "             :  B13 = " << BasicDistance1[0][2]
                           << "  B23 = " << BasicDistance1[1][2]
                           << "  B33 = " << BasicDistance1[2][2] << endl; 
        cout << "  offset parm:  O11 = " << BasicDistance2[0][0]
                           << "  O21 = " << BasicDistance2[1][0]
                           << "  O31 = " << BasicDistance2[2][0] << endl; 
        cout << "             :  O12 = " << BasicDistance2[0][1]
                           << "  O22 = " << BasicDistance2[1][1]
                           << "  O32 = " << BasicDistance2[2][1] << endl;
        cout << "             :  O13 = " << BasicDistance2[0][2]
                           << "  O23 = " << BasicDistance2[1][2]
                           << "  O33 = " << BasicDistance2[2][2] << endl; 
        cout << "  easy grid  :   E1 = " << EasyCoordinate[0]
                           << "   E2 = " << EasyCoordinate[1]
                           << "   E3 = " << EasyCoordinate[2] << endl; 
        cout << "             :   I1 = " << goodIndex[0]
                           << "   I2 = " << goodIndex[1]
                           << "   I3 = " << goodIndex[2] << endl; 
        cout << "  structure  :   S1 = " << systematicGrid[0]
                           << "   S2 = " << systematicGrid[1]
                           << "   S3 = " << systematicGrid[2]; 
        if (KnownStructure) cout << "  -->  VALID   (step II)" << endl;
        else{               cout << "  -->  INVALID (step II)" << endl;
          cout << "  reason for error: ";
          if (NumberOfPoints[0]*NumberOfPoints[1]*NumberOfPoints[2] != nLines) {
            cout << endl;
            cout << "  N1*N2*N3 =/= N.lines  -->  exiting now ..." << endl;
            return;
          }
          else {
	    cout << "  no idea so far  -->  exiting now ..." << endl;
            return;
          }
        }
      }
    }
  }
  // END OF PART II: FOR MORE COMPLICATED COORDINATES

  // save coordinates and field values
  SixDPoint GridEntry;
  for (unsigned int iLine=0; iLine<XBArray.size(); ++iLine){
    XBVector = XBArray.operator[](iLine);
    XBVector.getV6(x1, x2, x3, Bx, By, Bz);
    GridEntry.putP6(x1, x2, x3, Bx, By, Bz);
    GridData.push_back(GridEntry);
  }
  if (!XBArray.empty()) XBArray.clear();

  return;
}


void prepareMagneticFieldGrid::fillFromFileSpecial(const std::string& name){

  // GridType = 5; // 1/sin(phi) coordinates. Obsolete, not used anymore.
  GridType = 6; // 1/cos(phi) coordinates.

  double phi=(rotateSector-1)*Geom::pi()/6.;
  double sphi = sin(-phi);
  double cphi = cos(-phi);

  // check, whether coordinates are Cartesian or cylindrical
  std::string::size_type ibeg, iend;
  ibeg = name.find('-');  // first occurance of "-"
  iend = name.rfind('-'); // last  occurance of "-"
  if  ((name.substr(ibeg+1, iend-ibeg-1)) == "xyz") {
    XyzCoordinates = true;
  } else if  ((name.substr(ibeg+1, iend-ibeg-1)) == "rpz") {
    RpzCoordinates = true;
  } else {
    cout << " Unrecognized input file name: valid formats are *-xyz-* or *-rpz-*" << endl;
    abort();
  }



  // define vectors of IndexedDoubleVectors
  IndexedDoubleVector XBVector;
  std::vector<IndexedDoubleVector>  XBValues;
  std::vector<IndexedDoubleVector> IXBValues;
  std::vector<IndexedDoubleVector>  XBArray;

  // vectors for pre analysis
  std::vector<double> valX[3];
  std::vector<int>    numX[3];

  // copy ASCII file to the vector
  int nLines = 0;
  int     index[3] = {0,0,0};
  int    nSteps[3] = {0,0,0};
  double x1,x2,x3,Bx,By,Bz,perm,poten;
  std::ifstream file(name.c_str());
  if (file.good()) {
    while (file.good()){
      file >> x1 >> x2 >> x3 >> Bx >> By >> Bz >> perm >> poten;
      if (file){
	if (rotateSector>1) {
	  if (RpzCoordinates) {
	    x2 = x2-phi;
	    if (fabs(x2)>0.78) {
	      cout << "ERROR: Input coordinates do not belong to sector " << rotateSector 
		   << " : " << x2 << endl;
	      abort();
	    }
	  } else {
	    double x=x1;
	    double y=x2;	    
	    x1 = x*cphi-y*sphi;
	    x2 = x*sphi+y*cphi;
	    if (fabs(atan2(x2,x1))>0.78) {
	      cout << "ERROR: Input coordinates do not belong to sector " << rotateSector 
		   << " : " << atan2(x2,x1) << endl;
	      abort();
	    }
	  }
	  double Bx0 = Bx;
	  double By0 = By;	  
	  Bx = Bx0*cphi-By0*sphi;
	  By = Bx0*sphi+By0*cphi;
	  
	}

        XBVector.putI3(index[0], index[1], index[2]);
        XBVector.putV6(x1, x2, x3, Bx, By, Bz);
        XBValues.push_back(XBVector);
        ++nLines;

        // pre analyze file content
        double pnt[3] = {x1,x2,x3};
        if (nLines == 1){
	  for (int i=0; i<3; ++i){
	    valX[i].push_back(pnt[i]);
	    numX[i].push_back(1);
	  }
        }
        else{
	  for (int i=0; i<3; ++i){
            bool knownValue = false;
	    for (int j=0; j<(nSteps[i]+1); ++j){
	      if (std::abs(valX[i].operator[](j) - pnt[i]) < EPSILON) knownValue = true;
	      if (std::abs(valX[i].operator[](j) - pnt[i]) < EPSILON) numX[i].operator[](j)+=1;
	    }
	    if (!knownValue) {
	      valX[i].push_back(pnt[i]);
	      nSteps[i]++;
	      numX[i].push_back(1);
	    }
	  }
        }
      }
    }
  }
  else return;

  // begin simple grid (EasyCoordinate) analysis
  for (int i=0; i<3; ++i){
    // sort the different values per one coordinate
    sort(valX[i].begin(), valX[i].end());
    // calculate number of points and constant step size
    EasyCoordinate[i] = true;
    NumberOfPoints[i] = nSteps[i] + 1;
    BasicDistance0[i] = 9e9;
    double newVal = 0.0;
    double oldVal = 0.0;
    for (int j=0; j<NumberOfPoints[i]; ++j){
      newVal = valX[i].operator[](j);
      if (j ==1) BasicDistance0[i] = newVal - oldVal;
      else if (j > 1){
	if (std::abs(newVal - oldVal - BasicDistance0[i]) > EPSILON) EasyCoordinate[i] = false;
      }
      oldVal = newVal;
    }
  }

  std::cout << std::endl;

  // define indices for easy coordinates
  for (int iLine=0; iLine<nLines; ++iLine){
    XBVector = XBValues.operator[](iLine);
    XBVector.getI3(index[0], index[1], index[2]);
    XBVector.getV6(x1, x2, x3, Bx, By, Bz);
    double pnt[3] = {x1,x2,x3};
    for (int i=0; i<3; ++i){
      for (int j=0; j<NumberOfPoints[i]; ++j){
	if (EasyCoordinate[i]){
	  if (std::abs(valX[i].operator[](j) - pnt[i]) < EPSILON) index[i] = j;
	}
      }
    }
    XBVector.putI3(index[0], index[1], index[2]);
    IXBValues.push_back(XBVector);
  }
  if (!XBValues.empty()) XBValues.clear();

  // sort according to defined indices
  sort(IXBValues.begin(), IXBValues.end());

  // check, if structure is known at this point
  bool systematicGrid[3] = {false,false,false};
  KnownStructure = true;
  if (NumberOfPoints[0]*NumberOfPoints[1]*NumberOfPoints[2] != nLines) KnownStructure = false;
  for (int i=0; i<3; ++i){
    systematicGrid[i] = EasyCoordinate[i];
      if (!systematicGrid[i]) KnownStructure = false;
  }

  if (KnownStructure == true){
    for (int iLine=0; iLine<nLines; ++iLine){
      XBVector = IXBValues.operator[](iLine);
      XBArray.push_back(XBVector);
    }
  }

  // get first point = ReferencePoint (and last point for structure checking)
  if (!XBArray.empty()) XBVector = XBArray.front();
  XBVector.getV6(x1, x2, x3, Bx, By, Bz);
  double firstListPoint[3] = {x1,x2,x3};
  if (!XBArray.empty()) XBVector = XBArray.back();
  XBVector.getV6(x1, x2, x3, Bx, By, Bz);
  double secondRefPoint[3] = {x1,x2,x3};

  // recheck all coordinates
  KnownStructure = true;
  for (int i=0; i<3; ++i){
    ReferencePoint[i] = firstListPoint[i];
    // calculate step size for maximal indices
    double stepSize = BasicDistance0[i];
    // calculate difference between the two reference points minus the total difference (=0)
    double totDiff = secondRefPoint[i] - (ReferencePoint[i] + stepSize*nSteps[i]);
    if (std::abs(totDiff) < EPSILON*10.) systematicGrid[i] = true;
    else                         systematicGrid[i] = false;
    if (!systematicGrid[i]) KnownStructure = false;
  }
  bool goodIndex[3] = {true,true,true};

  if (!KnownStructure){
    // analyze structure of missing coordinates
    int nEasy = 0;
    for (int i=0; i<3; ++i){
      if (EasyCoordinate[i]) ++nEasy;
    }
    std::vector<double> misValX[3];

    // try to recover info of missing coordiates
    if (nEasy > 0){
      bool lastLine = false;
      int lowerLineLimit = 0;
      int upperLineLimit = 0;
      int oldIndex[3] = {0,0,0};
      for (int iLine=0; iLine<nLines; ++iLine){
	if (iLine == (nLines-1)) lastLine = true;
        XBVector = IXBValues.operator[](iLine);
	XBVector.getI3(index[0], index[1], index[2]);
	bool newIndices = false;
        for (int i=0; i<3; ++i){
	  if (oldIndex[i] != index[i]){
	    oldIndex[i] = index[i];
	    newIndices = true;
	  }
	}
	if (newIndices){
	  for (int i=0; i<3; ++i){
	    if (!EasyCoordinate[i]) sort(misValX[i].begin(), misValX[i].end());
	  }
	  lowerLineLimit = upperLineLimit;
	  upperLineLimit = iLine;
	  for (int jLine=lowerLineLimit; jLine<upperLineLimit; ++jLine){
	    XBVector = IXBValues.operator[](jLine);
	    XBVector.getI3(index[0], index[1], index[2]);
	    XBVector.getV6(x1, x2, x3, Bx, By, Bz);
	    double pnt[3] = {x1,x2,x3};
	    for (int i=0; i<3; ++i){
	      if (!EasyCoordinate[i]){
		double thisVal = 0.; double lastVal = 0.; double deltVal = 0.;
		for (int j=0; j<NumberOfPoints[i]; ++j){
                  thisVal = misValX[i].operator[](j);
		  if (std::abs(thisVal - pnt[i]) < EPSILON) index[i] = j;
		  if (j ==1) deltVal = thisVal - lastVal;
		  else if (j > 1){
		    if (std::abs(thisVal - lastVal - deltVal) > EPSILON) goodIndex[i] = false;
		  }
		  lastVal = thisVal;
		}
	      }
	    }
	    XBVector.putI3(index[0], index[1], index[2]);
	    XBArray.push_back(XBVector);
	    for (int i=0; i<3; ++i){
	      if (!misValX[i].empty()) misValX[i].clear();
	    }
	  }
	}
        XBVector = IXBValues.operator[](iLine);
        XBVector.getV6(x1, x2, x3, Bx, By, Bz);
        double pnt[3] = {x1,x2,x3};
        for (int i=0; i<3; ++i){
          if (!EasyCoordinate[i]){
	    if (iLine == 0){
	      NumberOfPoints[i] = 1;
	      misValX[i].push_back(pnt[i]);
	    }
	    else if (newIndices){
	      NumberOfPoints[i] = 1;
	      misValX[i].push_back(pnt[i]);
	    }
	    else{
	      bool knownValue = false;
	      for (int j=0; j<NumberOfPoints[i]; ++j){
		if (std::abs(misValX[i].operator[](j) - pnt[i]) < EPSILON) knownValue = true;
	      }
	      if (!knownValue){
		++NumberOfPoints[i];
		misValX[i].push_back(pnt[i]);
	      }
	    }
	  }
	}
	if (lastLine){
	  for (int i=0; i<3; ++i){
	    if (!EasyCoordinate[i]) sort(misValX[i].begin(), misValX[i].end());
	  }
	  lowerLineLimit = upperLineLimit;
	  upperLineLimit = nLines;
	  for (int jLine=lowerLineLimit; jLine<upperLineLimit; ++jLine){
	    XBVector = IXBValues.operator[](jLine);
	    XBVector.getI3(index[0], index[1], index[2]);
	    XBVector.getV6(x1, x2, x3, Bx, By, Bz);
	    double pnt[3] = {x1,x2,x3};
	    for (int i=0; i<3; ++i){
	      if (!EasyCoordinate[i]){
		double thisVal = 0.; double lastVal = 0.; double deltVal = 0.;
		for (int j=0; j<NumberOfPoints[i]; ++j){
                  thisVal = misValX[i].operator[](j);
		  if (std::abs(thisVal - pnt[i]) < EPSILON) index[i] = j;
		  if (j ==1) deltVal = thisVal - lastVal;
		  else if (j > 1){
		    if (std::abs(thisVal - lastVal - deltVal) > EPSILON) goodIndex[i] = false;
		  }
		  lastVal = thisVal;
		}
	      }
	    }
	    XBVector.putI3(index[0], index[1], index[2]);
	    XBArray.push_back(XBVector);
	    for (int i=0; i<3; ++i){
	      if (!misValX[i].empty()) misValX[i].clear();
	    }
	  }
	}
      }
    }
    int nGoodInd = 0;
    for (int i=0; i<3; ++i){
      if (goodIndex[i]) ++nGoodInd;
    }
    // common part of missing coordiate(s) recovery (1 or 2 missing)
    if (nEasy > 0){
      // sort without resetting of indices
      sort(XBArray.begin(), XBArray.end());

      for (int i=0; i<3; ++i){
	nSteps[i] = NumberOfPoints[i]-1;
      }
      std::vector<double> rMinVec;
      std::vector<double> rMaxVec;
      std::vector<double> phiVec;
      for (int iLine=0; iLine<nLines; ++iLine){
	XBVector = XBArray.operator[](iLine);
	XBVector.getI3(index[0], index[1], index[2]);
	// in this case:r,phi,z       
	XBVector.getV6(x1, x2, x3, Bx, By, Bz);
	if (index[2] == 0){
	  if (index[0] == 0        ) rMinVec.push_back(x1);
	  if (index[0] == nSteps[0]) rMaxVec.push_back(x1);
	  if (index[0] == nSteps[0])  phiVec.push_back(x2);
	}
      }
      for (int j=0; j<NumberOfPoints[1]; ++j){
	double phi  =  phiVec.operator[](j);
	double rMin = rMinVec.operator[](j);
	double rMax = rMaxVec.operator[](j);
	double rhoMin, rhoMax;
	if (GridType==6) {
	  rhoMin = rMin*cos(phi);
	  rhoMax = rMax*cos(phi);
	} else if (GridType==5) {
	  rhoMin = rMin*sin(phi);
	  rhoMax = rMax*sin(phi);
	} else {
	  cout << "unknown grid type!" << endl;
	  abort();
	}
	
	
	if (j == 0){
	  RParAsFunOfPhi[0] = rMax;
	  RParAsFunOfPhi[1] = rhoMax;
	  RParAsFunOfPhi[2] = rMin;
	  RParAsFunOfPhi[3] = rhoMin;
	}
	else{
	  if (std::abs(rMax   - RParAsFunOfPhi[0]) > EPSILON) RParAsFunOfPhi[0] = 0.;
	  if (std::abs(rhoMax - RParAsFunOfPhi[1]) > EPSILON) RParAsFunOfPhi[1] = 0.;
	  if (std::abs(rMin   - RParAsFunOfPhi[2]) > EPSILON) RParAsFunOfPhi[2] = 0.;
	  if (std::abs(rhoMin - RParAsFunOfPhi[3]) > EPSILON) RParAsFunOfPhi[3] = 0.;
	}
      }

      // set default values of BasicDistance0 in case of < 2 points
      for (int i=0; i<3; ++i){
        if (NumberOfPoints[i] < 2) BasicDistance0[i] = 9e9;
      }
    }

    // get first point = ReferencePoint (and last point for structure checking)
    if (!XBArray.empty()) XBVector = XBArray.front();
    XBVector.getV6(x1, x2, x3, Bx, By, Bz);
    firstListPoint[0] = x1;
    firstListPoint[1] = x2;
    firstListPoint[2] = x3;
    if (!XBArray.empty()) XBVector = XBArray.back();
    XBVector.getV6(x1, x2, x3, Bx, By, Bz);
    secondRefPoint[0] = x1;
    secondRefPoint[1] = x2;
    secondRefPoint[2] = x3;

    // recheck all coordinates
    KnownStructure = true;
    for (int i=0; i<3; ++i){
      ReferencePoint[i] = firstListPoint[i];
      nSteps[i] = NumberOfPoints[i]-1;
    }

    double sinPhi; // either sin or cos depending on grid type
    if (GridType==6) {
      sinPhi = cos(secondRefPoint[1]);
    } else if (GridType==5) {
      sinPhi = sin(secondRefPoint[1]);
    } else {
      cout << "unknown GridType!" << endl;
      abort();
    }
	
    
    if (std::abs(sinPhi) < EPSILON) {
      cout << " WARNING: unexpected sinPhi parameter = 0" << endl;
      sinPhi = EPSILON;
    }
    
    double totStepSize = RParAsFunOfPhi[0] + RParAsFunOfPhi[1]/sinPhi - RParAsFunOfPhi[2] - RParAsFunOfPhi[3]/sinPhi;
    double startingPoint = RParAsFunOfPhi[2] + RParAsFunOfPhi[3]/sinPhi;
    double totDiff = secondRefPoint[0] - (startingPoint + totStepSize);
    if (std::abs(totDiff) < EPSILON*10.) systematicGrid[0] = true;
    else                         systematicGrid[0] = false;
    if (!systematicGrid[0]) KnownStructure = false;
    totDiff = secondRefPoint[1] - (ReferencePoint[1] + BasicDistance0[1]*nSteps[1]);
    if (std::abs(totDiff) < EPSILON*10.) systematicGrid[1] = true;
    else                         systematicGrid[1] = false;
    if (!systematicGrid[1]) KnownStructure = false;
    totDiff = secondRefPoint[2] - (ReferencePoint[2] + BasicDistance0[2]*nSteps[2]);
    if (std::abs(totDiff) < EPSILON*10.) systematicGrid[2] = true;
    else                         systematicGrid[2] = false;
    if (!systematicGrid[2]) KnownStructure = false;
    
    if (PRINT){
      // print result
      cout << "  read " << nLines << " lines -> grid structure:" << endl;
      cout << "  # of points:   N1 = " << NumberOfPoints[0]
	   << "   N2 = " << NumberOfPoints[1]
	   << "   N3 = " << NumberOfPoints[2] << endl;
      cout << "  ref. point :   X1 = " << ReferencePoint[0]
	   << "   X2 = " << ReferencePoint[1]
	   << "   X3 = " << ReferencePoint[2] << endl;
      cout << "  step parm0 :   A1 = " << BasicDistance0[0]
	   << "   A2 = " << BasicDistance0[1]
	   << "   A3 = " << BasicDistance0[2] << endl;
      cout << "  r param.s. :   R1 = " << RParAsFunOfPhi[0]
	   << " RHO1 = " << RParAsFunOfPhi[1]
	   << "   R2 = " << RParAsFunOfPhi[2]
	   << " RHO2 = " << RParAsFunOfPhi[3] << endl;
      cout << "  easy grid  :   E1 = " << EasyCoordinate[0]
	   << "   E2 = " << EasyCoordinate[1]
	   << "   E3 = " << EasyCoordinate[2] << endl;
      cout << "             :   I1 = " << goodIndex[0]
	   << "   I2 = " << goodIndex[1]
	   << "   I3 = " << goodIndex[2] << endl;
      cout << "  structure  :   S1 = " << systematicGrid[0]
	   << "   S2 = " << systematicGrid[1]
	   << "   S3 = " << systematicGrid[2]; 
      if (KnownStructure) cout << "  -->  VALID   (step II)" << endl;
      else{               cout << "  -->  INVALID (step II)" << endl;
	cout << "  reason for error: ";
	if (NumberOfPoints[0]*NumberOfPoints[1]*NumberOfPoints[2] != nLines) {
	  cout << endl;
	  cout << "  N1*N2*N3 =/= N.lines  -->  exiting now ..." << endl;
	  return;
	}
	else {
	  cout << "  no idea so far  -->  exiting now ..." << endl;
	  return;
        }
      }
    }
  }

  // save coordinates and field values
  SixDPoint GridEntry;
  for (unsigned int iLine=0; iLine<XBArray.size(); ++iLine){
    XBVector = XBArray.operator[](iLine);
    XBVector.getV6(x1, x2, x3, Bx, By, Bz);
    GridEntry.putP6(x1, x2, x3, Bx, By, Bz);
    GridData.push_back(GridEntry);
  }
  if (!XBArray.empty()) XBArray.clear();

  return;
}


int prepareMagneticFieldGrid::gridType(){
  int type = GridType;
  if (PRINT){
    if (type == 0) cout << "  grid type = " << type << "  -->  not determined" << endl;
    if (type == 1) cout << "  grid type = " << type << "  -->  (x,y,z) cube" << endl;
    if (type == 2) cout << "  grid type = " << type << "  -->  (x,y,z) trapezoid" << endl;
    if (type == 3) cout << "  grid type = " << type << "  -->  (r,phi,z) cube" << endl;
    if (type == 4) cout << "  grid type = " << type << "  -->  (r,phi,z) trapezoid" << endl;
    if (type == 5) cout << "  grid type = " << type << "  -->  (r,phi,z) 1/sin(phi)" << endl;
    if (type == 6) cout << "  grid type = " << type << "  -->  (r,phi,z) 1/cos(phi)" << endl;
  }
  return type;
}


void prepareMagneticFieldGrid::validateAllPoints(){

  if (!KnownStructure) return;

  double x1, x2, x3, Bx, By, Bz;
  int numberOfErrors = 0;

  // loop over three dimensions
  int index[3];

  if (GridType < 5){
    for (index[0]=0; index[0]<NumberOfPoints[0]; ++index[0]){
      for (index[1]=0; index[1]<NumberOfPoints[1]; ++index[1]){
        for (index[2]=0; index[2]<NumberOfPoints[2]; ++index[2]){
	  // get reference values
	  putIndicesGetXAndB(index[0], index[1], index[2], x1, x2, x3, Bx, By, Bz);

	  // first check: calculate indices from coordinates and compare with original indices
	  double pnt[3] = {x1,x2,x3};
	  int tmpIdx[3] = {999,999,999};
	  putCoordGetIndices((x1+EPSILON), (x2+EPSILON), (x3+EPSILON), tmpIdx[0], tmpIdx[1], tmpIdx[2]);
	  for (int i=0; i<3; ++i){
	    if (tmpIdx[i] != index[i]) ++numberOfErrors;
	  }

	  // second check: calculate coordinate from indices and compare with original values
          double tmpPnt[3] = {0.0,0.0,0.0};
	  putIndCalcXReturnB(index[0], index[1], index[2], tmpPnt[0], tmpPnt[1], tmpPnt[2], Bx, By, Bz);
	  for (int i=0; i<3; ++i){
	    if (std::abs(tmpPnt[i]-pnt[i]) > EPSILON) ++numberOfErrors;
	  }
        }
      }
    }
  }

  if (GridType == 5 || GridType == 6){
    for (index[0]=0; index[0]<NumberOfPoints[0]; ++index[0]){
      for (index[1]=0; index[1]<NumberOfPoints[1]; ++index[1]){
	for (index[2]=0; index[2]<NumberOfPoints[2]; ++index[2]){
	  // get reference values
	  putIndicesGetXAndB(index[0], index[1], index[2], x1, x2, x3, Bx, By, Bz);

	  // first check: calculate indices from coordinates and compare with original indices
	  double pnt[3] = {x1,x2,x3};
	  int tmpIdx[3] = {999,999,999};
	  putCoordGetIndices((x1+EPSILON), (x2+EPSILON), (x3+EPSILON), tmpIdx[0], tmpIdx[1], tmpIdx[2]);
	  for (int i=0; i<3; ++i){
	    if (tmpIdx[i] != index[i]){
	      putCoordGetIndices((x1-EPSILON), (x2+EPSILON), (x3+EPSILON), tmpIdx[0], tmpIdx[1], tmpIdx[2]);
	      if (tmpIdx[i] != index[i]){
		putCoordGetIndices((x1+EPSILON), (x2-EPSILON), (x3+EPSILON), tmpIdx[0], tmpIdx[1], tmpIdx[2]);
		if (tmpIdx[i] != index[i]){
		  putCoordGetIndices((x1-EPSILON), (x2-EPSILON), (x3+EPSILON), tmpIdx[0], tmpIdx[1], tmpIdx[2]);
		  if (tmpIdx[i] != index[i]) ++numberOfErrors;
		}
	      }
	    }
	  }

	  // second check: calculate coordinate from indices and compare with original values
	  double tmpPnt[3] = {0.0,0.0,0.0};
	  putIndCalcXReturnB(index[0], index[1], index[2], tmpPnt[0], tmpPnt[1], tmpPnt[2], Bx, By, Bz);
	  for (int i=0; i<3; ++i){
	    if (std::abs(tmpPnt[i]-pnt[i]) > EPSILON) ++numberOfErrors;
	  }
	}
      }
    }
  }

  if (numberOfErrors > 0) KnownStructure = false;
  if (PRINT) cout << "  grid validation of all points done  -->  # errors = " << numberOfErrors << endl;

  return;
}


void prepareMagneticFieldGrid::saveGridToFile(const std::string& outName){

  // open output file
  binary_ofstream outFile(outName);
  // write grid type
  outFile << GridType;
  // write header (depending on grid type)
  convertUnits(); // m->cm
  if (GridType == 1) {
    outFile << NumberOfPoints[0]    << NumberOfPoints[1]    << NumberOfPoints[2];
    outFile << ReferencePoint[0]    << ReferencePoint[1]    << ReferencePoint[2];
    outFile << BasicDistance0[0]    << BasicDistance0[1]    << BasicDistance0[2];
  }
  if (GridType == 2) {
    outFile << NumberOfPoints[0]    << NumberOfPoints[1]    << NumberOfPoints[2];
    outFile << ReferencePoint[0]    << ReferencePoint[1]    << ReferencePoint[2];
    outFile << BasicDistance0[0]    << BasicDistance0[1]    << BasicDistance0[2];
    outFile << BasicDistance1[0][0] << BasicDistance1[1][0] << BasicDistance1[2][0];
    outFile << BasicDistance1[0][1] << BasicDistance1[1][1] << BasicDistance1[2][1];
    outFile << BasicDistance1[0][2] << BasicDistance1[1][2] << BasicDistance1[2][2];
    outFile << BasicDistance2[0][0] << BasicDistance2[1][0] << BasicDistance2[2][0];
    outFile << BasicDistance2[0][1] << BasicDistance2[1][1] << BasicDistance2[2][1];
    outFile << BasicDistance2[0][2] << BasicDistance2[1][2] << BasicDistance2[2][2];
    outFile << EasyCoordinate[0]    << EasyCoordinate[1]    << EasyCoordinate[2];
  }
  if (GridType == 3) {
    outFile << NumberOfPoints[0]    << NumberOfPoints[1]    << NumberOfPoints[2];
    outFile << ReferencePoint[0]    << ReferencePoint[1]    << ReferencePoint[2];
    outFile << BasicDistance0[0]    << BasicDistance0[1]    << BasicDistance0[2];
  }
  if (GridType == 4) {
    outFile << NumberOfPoints[0]    << NumberOfPoints[1]    << NumberOfPoints[2];
    outFile << ReferencePoint[0]    << ReferencePoint[1]    << ReferencePoint[2];
    outFile << BasicDistance0[0]    << BasicDistance0[1]    << BasicDistance0[2];
    outFile << BasicDistance1[0][0] << BasicDistance1[1][0] << BasicDistance1[2][0];
    outFile << BasicDistance1[0][1] << BasicDistance1[1][1] << BasicDistance1[2][1];
    outFile << BasicDistance1[0][2] << BasicDistance1[1][2] << BasicDistance1[2][2];
    outFile << BasicDistance2[0][0] << BasicDistance2[1][0] << BasicDistance2[2][0];
    outFile << BasicDistance2[0][1] << BasicDistance2[1][1] << BasicDistance2[2][1];
    outFile << BasicDistance2[0][2] << BasicDistance2[1][2] << BasicDistance2[2][2];
    outFile << EasyCoordinate[0]    << EasyCoordinate[1]    << EasyCoordinate[2];
  }
  if (GridType == 5 || GridType == 6) {
    outFile << NumberOfPoints[0]    << NumberOfPoints[1]    << NumberOfPoints[2];
    outFile << ReferencePoint[0]    << ReferencePoint[1]    << ReferencePoint[2];
    outFile << BasicDistance0[0]    << BasicDistance0[1]    << BasicDistance0[2];
    outFile << RParAsFunOfPhi[0]    << RParAsFunOfPhi[1]    << RParAsFunOfPhi[2]    << RParAsFunOfPhi[3];
  }
  int nLines = NumberOfPoints[0]*NumberOfPoints[1]*NumberOfPoints[2];
  SixDPoint GridPoint;
  // write magnetic field values (3*nPoints)
  for (int iLine=0; iLine<nLines; ++iLine){
    GridPoint = GridData.operator[](iLine);
    float Bx = float(GridPoint.bx());
    float By = float(GridPoint.by());
    float Bz = float(GridPoint.bz());
    outFile << Bx << By << Bz;
    //    if (iLine<5) cout << setprecision(12) << Bx << " " << GridPoint.bx() << endl;
  }
  // make end and close output file
  const std::string lastEntry = "complete";
  outFile << lastEntry;
  outFile.close();
  if (PRINT) cout << "  output " << outName << endl;
  return;
}

void  prepareMagneticFieldGrid::convertUnits(){
  double cm = 100.; // m->cm (just multiply all lengths with 100)
  if (XyzCoordinates){
    for (int i=0;i<3; ++i) {ReferencePoint[i] *= cm;};
    for (int i=0;i<3; ++i) {BasicDistance0[i] *= cm;};
    for (int i=0;i<3; ++i) {for (int j=0;j<3; ++j) {BasicDistance1[i][j] *= cm;};};
    for (int i=0;i<3; ++i) {for (int j=0;j<3; ++j) {BasicDistance2[i][j] *= cm;};};
    for (int i=0;i<4; ++i) {RParAsFunOfPhi[i] *= cm;};
  }
  double du[3] = {100.,1.,100.}; // m->cm ; rad->rad (unchanged)
  if (RpzCoordinates){
    for (int i=0;i<3; ++i) {ReferencePoint[i] *= du[i];};
    for (int i=0;i<3; ++i) {BasicDistance0[i] *= du[i];};
    for (int i=0;i<3; ++i) {for (int j=0;j<3; ++j) {BasicDistance1[i][j] *= du[i];};};
    for (int i=0;i<3; ++i) {for (int j=0;j<3; ++j) {BasicDistance2[i][j] *= du[i];};};
    for (int i=0;i<4; ++i) {RParAsFunOfPhi[i] *= cm;};
  }
  return;
}

bool prepareMagneticFieldGrid::isReady(){  return KnownStructure; }

void prepareMagneticFieldGrid::interpolateAtPoint(double X1, double X2, double X3, double &Bx, double &By, double &Bz){

  Bx = By = Bz = 0.0;
  if (KnownStructure){
    // define interpolation object
    VectorFieldInterpolation MagInterpol;
    // calculate indices for "CellPoint000"
    int index[3];
    putCoordGetIndices(X1,X2,X3,index[0],index[1],index[2]);
    int index0[3] = {0,0,0};
    int index1[3] = {0,0,0};
    for (int i=0; i<3; ++i){
      if (NumberOfPoints[i] > 1){
	                                     index0[i] = std::max(0,index[i]);
	if (index0[i] > NumberOfPoints[i]-2) index0[i] = NumberOfPoints[i]-2;;
	                                     index1[i] = std::max(1,index[i]+1);;
	if (index1[i] > NumberOfPoints[i]-1) index1[i] = NumberOfPoints[i]-1;
      }
    }
    double tmpX[3];
    double tmpB[3];
    // define the corners of interpolation volume
    putIndicesGetXAndB(index0[0],index0[1],index0[2],tmpX[0],tmpX[1],tmpX[2],tmpB[0],tmpB[1],tmpB[2]);
    MagInterpol.defineCellPoint000(tmpX[0],tmpX[1],tmpX[2],tmpB[0],tmpB[1],tmpB[2]);
    putIndicesGetXAndB(index1[0],index0[1],index0[2],tmpX[0],tmpX[1],tmpX[2],tmpB[0],tmpB[1],tmpB[2]);
    MagInterpol.defineCellPoint100(tmpX[0],tmpX[1],tmpX[2],tmpB[0],tmpB[1],tmpB[2]);
    putIndicesGetXAndB(index0[0],index1[1],index0[2],tmpX[0],tmpX[1],tmpX[2],tmpB[0],tmpB[1],tmpB[2]);
    MagInterpol.defineCellPoint010(tmpX[0],tmpX[1],tmpX[2],tmpB[0],tmpB[1],tmpB[2]);
    putIndicesGetXAndB(index1[0],index1[1],index0[2],tmpX[0],tmpX[1],tmpX[2],tmpB[0],tmpB[1],tmpB[2]);
    MagInterpol.defineCellPoint110(tmpX[0],tmpX[1],tmpX[2],tmpB[0],tmpB[1],tmpB[2]);
    putIndicesGetXAndB(index0[0],index0[1],index1[2],tmpX[0],tmpX[1],tmpX[2],tmpB[0],tmpB[1],tmpB[2]);
    MagInterpol.defineCellPoint001(tmpX[0],tmpX[1],tmpX[2],tmpB[0],tmpB[1],tmpB[2]);
    putIndicesGetXAndB(index1[0],index0[1],index1[2],tmpX[0],tmpX[1],tmpX[2],tmpB[0],tmpB[1],tmpB[2]);
    MagInterpol.defineCellPoint101(tmpX[0],tmpX[1],tmpX[2],tmpB[0],tmpB[1],tmpB[2]);
    putIndicesGetXAndB(index0[0],index1[1],index1[2],tmpX[0],tmpX[1],tmpX[2],tmpB[0],tmpB[1],tmpB[2]);
    MagInterpol.defineCellPoint011(tmpX[0],tmpX[1],tmpX[2],tmpB[0],tmpB[1],tmpB[2]);
    putIndicesGetXAndB(index1[0],index1[1],index1[2],tmpX[0],tmpX[1],tmpX[2],tmpB[0],tmpB[1],tmpB[2]);
    MagInterpol.defineCellPoint111(tmpX[0],tmpX[1],tmpX[2],tmpB[0],tmpB[1],tmpB[2]);
    // interpolate
    MagInterpol.putSCoordGetVField(X1,X2,X3,Bx,By,Bz);
  }

  return;
}

void prepareMagneticFieldGrid::putCoordGetIndices(double X1, double X2, double X3, int &Index1, int &Index2, int &Index3){

  double pnt[3] = {X1,X2,X3};
  int index[3];
	
  if (GridType < 5){
    for (int i=0; i<3; ++i){
      if (EasyCoordinate[i]){
	index[i] = int((pnt[i]-ReferencePoint[i])/BasicDistance0[i]);
      }
    }
    for (int i=0; i<3; ++i){
      if (!EasyCoordinate[i]){
	double stepSize = BasicDistance0[i];
	double offset   = 0.0;
	for (int j=0; j<3; ++j){
	  stepSize += BasicDistance1[i][j]*index[j];
	  offset   += BasicDistance2[i][j]*index[j];
	}
	index[i] = int((pnt[i]-(ReferencePoint[i] + offset))/stepSize);
      }
    }
  }
  if (GridType == 5 || GridType == 6){

    double sinPhi; // either sin or cos depending on grid type
    if (GridType==6) {
      sinPhi = cos(pnt[1]);
    } else if (GridType==5) {
      sinPhi = sin(pnt[1]);
    } else {
      abort();
    } 
    
    if (std::abs(sinPhi) < EPSILON){
      sinPhi = EPSILON;
      cout << "ERROR DIVISION BY ZERO" << endl;
    }
    double stepSize = RParAsFunOfPhi[0] + RParAsFunOfPhi[1]/sinPhi - RParAsFunOfPhi[2] - RParAsFunOfPhi[3]/sinPhi;
    stepSize =  stepSize/(NumberOfPoints[0]-1);
    double startingPoint = RParAsFunOfPhi[2] + RParAsFunOfPhi[3]/sinPhi;
    index[0] = int((pnt[0]-startingPoint)/stepSize);
    index[1] = int((pnt[1]-ReferencePoint[1])/BasicDistance0[1]);
    index[2] = int((pnt[2]-ReferencePoint[2])/BasicDistance0[2]);
  }

  Index1 = index[0];
  Index2 = index[1];
  Index3 = index[2];

  return;
}

void prepareMagneticFieldGrid::putIndicesGetXAndB(int Index1, int Index2, int Index3, double &X1, double &X2, double &X3, double &Bx, double &By, double &Bz){

  SixDPoint GridPoint;
  GridPoint = GridData.operator[](lineNumber(Index1, Index2, Index3));
  X1 = GridPoint.x1();
  X2 = GridPoint.x2();
  X3 = GridPoint.x3();
  Bx = GridPoint.bx();
  By = GridPoint.by();
  Bz = GridPoint.bz();

  return;
}

void prepareMagneticFieldGrid::putIndCalcXReturnB(int Index1, int Index2, int Index3, double &X1, double &X2, double &X3, double &Bx, double &By, double &Bz){

  int index[3] = {Index1, Index2, Index3};
  double pnt[3];

  if (GridType < 5){
    for (int i=0; i<3; ++i){
      if (EasyCoordinate[i]){
	pnt[i] = ReferencePoint[i] + BasicDistance0[i]*index[i];
      }
      else {
	double stepSize = BasicDistance0[i];
	double offset   = 0.0;
	for (int j=0; j<3; ++j){
	  stepSize += BasicDistance1[i][j]*index[j];
	  offset   += BasicDistance2[i][j]*index[j];
	}
	pnt[i] = ReferencePoint[i] + offset + stepSize*index[i];
      }
    }
  }
  if (GridType == 5 || GridType == 6){
    pnt[2] = ReferencePoint[2] + BasicDistance0[2]*index[2];
    pnt[1] = ReferencePoint[1] + BasicDistance0[1]*index[1];

    double sinPhi; // Set either cos or sin depending on grid type
    if (GridType==6) {
      sinPhi = cos(pnt[1]);
    } else if (GridType ==5) {
      sinPhi = sin(pnt[1]);
    } else {
      cout << "unknown GridType!" << endl;
      abort();
    }
    
    if (std::abs(sinPhi) < EPSILON){
      sinPhi = EPSILON;
      cout << "ERROR DIVISION BY ZERO" << endl;
    }
    double stepSize = RParAsFunOfPhi[0] + RParAsFunOfPhi[1]/sinPhi - RParAsFunOfPhi[2] - RParAsFunOfPhi[3]/sinPhi;
    stepSize =  stepSize/(NumberOfPoints[0]-1);
    double startingPoint = RParAsFunOfPhi[2] + RParAsFunOfPhi[3]/sinPhi;
    pnt[0] = startingPoint + stepSize*index[0];
  }

  X1 = pnt[0];
  X2 = pnt[1];
  X3 = pnt[2];

  SixDPoint GridPoint;
  GridPoint = GridData.operator[](lineNumber(Index1, Index2, Index3));
  Bx = GridPoint.bx();
  By = GridPoint.by();
  Bz = GridPoint.bz();

  return;
}

int prepareMagneticFieldGrid::lineNumber(int Index1, int Index2, int Index3){
  return Index1*NumberOfPoints[1]*NumberOfPoints[2] + Index2*NumberOfPoints[2] + Index3;
}


bool prepareMagneticFieldGrid::IndexedDoubleVector::operator<(const IndexedDoubleVector& x) const{

  if (I3[0] < x.I3[0]) return true;
  else if (I3[0] == x.I3[0]){
    if (I3[1] < x.I3[1]) return true;
    else if (I3[1] == x.I3[1]){
      if (I3[2] < x.I3[2]) return true;
      else                 return false;
    }
    else                 return false;
  }
  else                 return false;
}

void prepareMagneticFieldGrid::IndexedDoubleVector::putI3(int  index1, int  index2, int  index3){
  I3[0] = index1;
  I3[1] = index2;
  I3[2] = index3;
  return;
}

void prepareMagneticFieldGrid::IndexedDoubleVector::getI3(int &index1, int &index2, int &index3){
  index1 = I3[0];
  index2 = I3[1];
  index3 = I3[2];
  return;
}

void prepareMagneticFieldGrid::IndexedDoubleVector::putV6(double  X1, double  X2, double  X3, double  Bx, double  By, double  Bz){
  V6[0] = X1;
  V6[1] = X2;
  V6[2] = X3;
  V6[3] = Bx;
  V6[4] = By;
  V6[5] = Bz;
  return;
}

void prepareMagneticFieldGrid::IndexedDoubleVector::getV6(double &X1, double &X2, double &X3, double &Bx, double &By, double &Bz){
  X1 = V6[0];
  X2 = V6[1];
  X3 = V6[2];
  Bx = V6[3];
  By = V6[4];
  Bz = V6[5];
  return;
}


void prepareMagneticFieldGrid::SixDPoint::putP6(double  X1, double  X2, double  X3, double  Bx, double  By, double  Bz){
  P6[0] = X1;
  P6[1] = X2;
  P6[2] = X3;
  P6[3] = Bx;
  P6[4] = By;
  P6[5] = Bz;
  return;
}

double prepareMagneticFieldGrid::SixDPoint::x1(){  return P6[0]; }

double prepareMagneticFieldGrid::SixDPoint::x2(){  return P6[1]; }

double prepareMagneticFieldGrid::SixDPoint::x3(){  return P6[2]; }

double prepareMagneticFieldGrid::SixDPoint::bx(){  return P6[3]; }

double prepareMagneticFieldGrid::SixDPoint::by(){  return P6[4]; }

double prepareMagneticFieldGrid::SixDPoint::bz(){  return P6[5]; }
