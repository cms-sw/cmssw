//   COCOA class implementation file
//Id: DeviationsFromFileSensor2D.cc
//CAT: Model
//
//   History: v1.0
//   Pedro Arce

#include "Alignment/CocoaModel/interface/DeviationsFromFileSensor2D.h"
#include "Alignment/CocoaUtilities/interface/ALIFileIn.h"
#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
#include <cstdlib>
#include <cmath>		// include floating-point std::abs functions
#include <memory>

enum directions{ xdir = 0, ydir = 1};

ALIbool DeviationsFromFileSensor2D::theApply = true;


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void DeviationsFromFileSensor2D::readFile( ALIFileIn& ifdevi )
{
  verbose = ALIUtils::debug;
  //  verbose = 4;
  if( verbose >= 3) std::cout << "DeviationsFromFileSensor2D::readFile " << this << ifdevi.name() << std::endl;

  theScanSenseX = 0;
  theScanSenseY = 0;

  ALIuint nl = 1;

  ALIdouble oldposX=0, oldposY=0;
  vd vcol;
  std::vector<ALIstring> wl;
  /*  //------ first line with dimension factors //now with global option
  ifdevi.getWordsInLine( wl );
  if( wl[0] != "DIMFACTOR:" || wl.size() != 3) {
    std::cerr << "Error reading sensor2D deviation file " << ifdevi.name() << std::endl
	 << " first line has to be DIMFACTOR: 'posDimFactor' 'angDimFactor' " << std::endl;
    ALIUtils::dumpVS( wl, " ");
    exit(1);
  }
  ALIdouble posDimFactor = atof(wl[1].c_str());
  ALIdouble angDimFactor = atof(wl[2].c_str());
  */

  for(;;) {
    ifdevi.getWordsInLine( wl );
    if(ifdevi.eof() ) break;

    DeviationSensor2D* dev = new DeviationSensor2D();
    dev->fillData( wl );

    if(verbose >= 5) {
      ALIUtils::dumpVS( wl, "deviation sensor2D", std::cout );
    }

    //--- line 2 of data
    if(nl == 2) {
      //--------- get if scan is first in Y or X
      firstScanDir = ydir;
      if(verbose >= 3) std::cout << "firstScanDir " << firstScanDir << " " <<  dev->posX() <<  " " << oldposX << " " <<  dev->posY() <<  " " << oldposY << std::endl;
      if( std::abs( dev->posX() - oldposX ) >  std::abs( dev->posY() - oldposY ) ) {
	std::cerr << "!!!EXITING first scan direction has to be Y for the moment " << ifdevi.name() << std::endl;
	firstScanDir = xdir;
	exit(1);
      }
      //-------- get sense of first scan direction
      if(firstScanDir == ydir ){
	if( dev->posY() > oldposY ) {
	  theScanSenseY = +1;
	} else {
	  theScanSenseY = -1;
	}
	if( verbose >= 3 ) std::cout << " theScanSenseY " << theScanSenseY << std::endl;
      }else {
	if( dev->posX() > oldposX ) {
	  theScanSenseX = +1;
	} else {
	  theScanSenseX = -1;
	}
	if( verbose >= 3 ) std::cout << " theScanSenseX " << theScanSenseX << std::endl;
      }
    }

    //-      std::cout << "firstScanDir " << firstScanDir << " " <<  dev->posX() <<  " " << oldposX << " " <<  dev->posY() <<  " " << oldposY << std::endl;

    //------- check if change of row: clear current std::vector of column values
    if( ( firstScanDir == ydir && ( dev->posY() - oldposY)*theScanSenseY < 0 )
	|| ( firstScanDir == xdir && ( dev->posX() - oldposX)*theScanSenseX < 0 )) {
      theDeviations.push_back( vcol );
      vcol.clear();

      //-      std::cout << " theDevi size " << theDeviations.size() << " " << ifdevi.name()  << std::endl;
      //-------- get sense of second scan direction
      if( theScanSenseY == 0 ) {
	if( dev->posY() > oldposY ) {
	  theScanSenseY = +1;
	} else {
	  theScanSenseY = -1;
	}
      }
      if( theScanSenseX == 0 ) {
	if( dev->posX() > oldposX ) {
	  theScanSenseX = +1;
	} else {
	  theScanSenseX = -1;
	}
      }
      if( verbose >= 3 ) std::cout << " theScanSenseX " << theScanSenseX << " theScanSenseY " << theScanSenseY << std::endl;
    }

    oldposX = dev->posX();
    oldposY = dev->posY();

    //--- fill deviation std::vectors
    vcol.push_back( dev );
    nl++;
  }
  theDeviations.push_back( vcol );

  //----- Calculate std::vector size
  ALIdouble dvsiz = ( sqrt( ALIdouble(nl-1) ) );
  //----- Check that it is a square of points (<=> vsiz is integer)
  ALIuint vsiz = ALIuint(dvsiz);
  if( vsiz != dvsiz ) {
    if( ALIUtils::debug >= 0) std::cerr << "!!WARNING: error reading deviation file: number of X points <> number of Y points : Number of points in X " << dvsiz << " nl " << nl-1 << " file " << ifdevi.name() << std::endl;
    //    exit(1);
  }
  theNPoints = vsiz;

  if(verbose >= 4 ) {
    std::cout << " Filling deviation from file: " << ifdevi.name() << " theNPoints " <<  theNPoints << std::endl;
  }

}


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
std::pair< ALIdouble, ALIdouble > DeviationsFromFileSensor2D::getDevis( ALIdouble intersX, ALIdouble intersY )
{
  //intersX += 10000;
  //intersY += 10000;
  if(verbose >= 4) std::cout << " entering getdevis " << intersX << " " <<intersY << " " << this << std::endl;
  vvd::iterator vvdite;
  vd::iterator vdite;

  // assume scan is in Y first, else revers intersection coordinates
/*  if( !firstScanDirY ) {
    ALIdouble tt = intersY;
    intersY = intersX;
    intersX = tt;
  }
*/

  //---------- look which point in the deviation matrices correspond to intersX,Y
  //----- Look for each column, between which rows intersY is
  //assume first dir is Y
  auto yrows = std::make_unique<unsigned int[]>(theNPoints);

  unsigned int ii = 0;
  ALIbool insideMatrix = false;
  for( vvdite = theDeviations.begin(); vvdite != (theDeviations.end()-1); ++vvdite ){
    for( vdite = (*vvdite).begin(); vdite != ((*vvdite).end()-1); ++vdite ){
     if( verbose >= 5 ) std::cout << " check posy " << (*(vdite))->posY()  << " " <<  (*(vdite+1))->posY()  << " " <<  (*(vdite)) << std::endl;
      // if posy is between this point and previous one

     //-     std::cout << "intersy" << intersY << " " <<  (*(vdite))->posY() << std::endl;

      if( (intersY - (*(vdite))->posY() )*theScanSenseY > 0
	  && (intersY - (*(vdite+1))->posY() )*theScanSenseY < 0 ) {
	//-std::cout << " ii " << ii << std::endl;
	yrows[ii] = vdite - (*vvdite).begin();
	if( verbose >= 3 ) std::cout << intersY << " DeviationsFromFileSensor2D yrows " << ii << " " << yrows[ii] << " : " << (*(vdite))->posY() << std::endl;
	insideMatrix = true;
	break;
      }
    }
    ii++;
  }
  if(insideMatrix == 0) {
    std::cerr << "!!EXITING intersection in Y outside matrix of deviations from file " << intersY << std::endl;
    exit(1);
  }
  insideMatrix = false;

  vd thePoints;
  thePoints.clear();
  //----- For each row in 'yrows' look between which columns intersX is
  unsigned int rn;
  DeviationSensor2D *dev1,*dev2;
  for( ii = 0; ii < theNPoints-1; ii++) {
    rn = yrows[ii];
    //-    std::cout << ii << " rn " << rn << std::endl;
    dev1 = (*(theDeviations.begin()+ii))[ rn ]; // column ii, row yrows[ii]
    rn = yrows[ii+1];
    dev2 = (*(theDeviations.begin()+ii+1))[ rn ]; // column ii+1, row yrows[ii+1]
    if( (intersX - dev1->posX() )*theScanSenseX > 0
	&& (intersX - dev2->posX() )*theScanSenseX < 0) {
      thePoints.push_back( dev1 );
      thePoints.push_back( dev2 );
      insideMatrix = true;
      if( verbose >= 3 ) std::cout << " column up " << ii << " " << dev1->posX()  << " " << dev2->posX() << " : " << intersX << std::endl;
    }

    rn = yrows[ii] + 1;
    if(rn == theNPoints) rn = theNPoints-1;
    dev1 = (*(theDeviations.begin()+ii))[ rn ]; // column ii, row yrows[ii]+1
    rn = yrows[ii+1] + 1;
    if(rn == theNPoints) rn = theNPoints-1;
    dev2 = (*(theDeviations.begin()+ii+1))[ rn ]; // column ii+1, row yrows[ii+1]+1
    if( (intersX - dev1->posX() )*theScanSenseX > 0
	&& (intersX - dev2->posX() )*theScanSenseX < 0) {
      thePoints.push_back( dev1 );
      thePoints.push_back( dev2 );
      if( verbose >= 3 ) std::cout << " column down " << ii << " " <<  dev1->posX()  << " " << dev2->posX() << " : " << intersX << std::endl;
    }

    if( thePoints.size() == 4 ) break;

  }

  if(insideMatrix == 0) {
    std::cerr << "!!EXITING intersection in X outside matrix of deviations from file " << intersX << std::endl;
    exit(1);
  }

  //----------- If loop finished and not 4 points, point is outside scan bounds

  //----- calculate deviation in x and y interpolating between four points
  ALIdouble dist, disttot=0, deviX=0, deviY=0;

   if( verbose >= 4) std::cout << " thepoints size " << thePoints.size() << std::endl;

  for( ii = 0; ii < 4; ii++) {
    dist = sqrt( pow(thePoints[ii]->posX() - intersX, 2 ) +  pow(thePoints[ii]->posY() - intersY, 2 ) );
    disttot += 1./dist;
    deviX += thePoints[ii]->devX()/dist;
    deviY += thePoints[ii]->devY()/dist;
    if( verbose >= 4 ) {
      //t      std::cout << ii << " point " << *thePoints[ii] << std::endl;
      std::cout << ii << " distances: " << dist << " " << deviX << " " << deviY << " devX " << thePoints[ii]->devX() << " devY " << thePoints[ii]->devY() << std::endl;
    }
  }
  deviX /= disttot;
  deviY /= disttot;

  // add offset
  deviX += theOffsetX;
  deviY += theOffsetY;
  if( verbose >= 4 ) {
    std::cout << " devisX/Y: " << deviX << " " << deviY
	 << " intersX/Y " << intersX  << " " << intersY << std::endl;
  }

  //change sign!!!!!?!?!?!?!?!
   deviX *= -1;
   deviY *= -1;
  return std::pair< ALIdouble, ALIdouble>( deviX*1.e-6, deviY*1e-6 );  // matrix is in microrad

}

