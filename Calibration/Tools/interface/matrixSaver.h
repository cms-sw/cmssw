/** \class matrixSaver

    \brief save (read) CLHEP::HepMatrix to (from) text files 

    $Date: 2008/02/25 17:40:25 $
    $Revision: 1.2 $
    $Id: matrixSaver.h,v 1.2 2008/02/25 17:40:25 malberti Exp $ 
    \author $Author: malberti $
*/

#ifndef __CINT__
#ifndef matrixSaver_h
#define matrixSaver_h

//#include <memory>
#include <vector>

#include<string>
#include<fstream>
#include<iostream>

#include "CLHEP/Matrix/GenMatrix.h"
#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/Vector.h"

class matrixSaver 
{
public:

  matrixSaver () ;
  ~matrixSaver () ;
  
  int saveMatrix (std::string outputFileName, 
                  const CLHEP::HepGenMatrix * saveMe) ;

  int saveMatrixVector (std::string outputFileName, 
                       const std::vector<CLHEP::HepGenMatrix*> &saveMe) ;

  int touch (std::string inputFileName) ;

  CLHEP::HepGenMatrix* getMatrix (std::string inputFileName) ;
  
  std::vector<CLHEP::HepGenMatrix*> * 
  getMatrixVector (std::string inputFileName) ;
  
  std::vector<CLHEP::HepMatrix> 
  getConcreteMatrixVector (std::string inputFileName) ;

private:


} ;

  std::istream &
  operator>> (std::istream& input, CLHEP::HepGenMatrix &matrix) ;
  
  std::ostream &
  operator<< (std::ostream& outputFile, const CLHEP::HepGenMatrix &saveMe) ;

#endif
#endif
