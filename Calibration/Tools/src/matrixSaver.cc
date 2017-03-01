#include <fstream>
#include <iostream> // for any std::cout
#include <iomanip>  // to print nicely formatted
#include <string>
#include <map>
#include <memory>
//#include <iterator>
#include <cassert>

#include "CLHEP/Matrix/GenMatrix.h"
#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/Vector.h"

#include "Calibration/Tools/interface/matrixSaver.h"

matrixSaver::matrixSaver ()
{
//   std::cout << "[matrixSaver][ctor] matrixSaver instance" << std::endl ;
}


matrixSaver::~matrixSaver ()
{
//   std::cout << "[matrixSaver][dtor] destroyed" << std::endl ;
}


std::ostream &
operator<< (std::ostream& outputFile,
	    const CLHEP::HepGenMatrix *saveMe)
{

  int numRow = saveMe->num_row () ;
  int numCol = saveMe->num_col () ;

  // write out the matrix dimensions
  outputFile << numRow << '\t'
	         << numCol << '\n' ;

  // write the elements in the file
  for (int row=0 ; row<numRow ; ++row)
    {
	    for (int col=0 ; col<numCol ; ++col)
        {
						assert (row < numRow) ;
            assert (col < numCol) ;
	          outputFile << (*saveMe)[row][col] << '\t' ;
        }
	    outputFile << '\n' ;
    }

   return outputFile ;
}



int
matrixSaver::saveMatrix (std::string outputFileName,
			const CLHEP::HepGenMatrix *saveMe)
{
   // open the output file
   std::fstream outputFile (outputFileName.c_str (), std::ios::out) ;
   assert (outputFile) ;
 
   int numRow = saveMe->num_row () ;
   int numCol = saveMe->num_col () ;

   // write out the matrix dimensions
   outputFile << numRow << '\t'
	          << numCol << '\n' ;

   outputFile << saveMe ;

   return 0 ;

}


int
matrixSaver::saveMatrixVector (std::string filename,
			       const std::vector<CLHEP::HepGenMatrix *> &saveMe)
{
     typedef std::vector<CLHEP::HepGenMatrix*>::const_iterator const_iterator ;
     // open the output file
     std::fstream outputFile (filename.c_str (), std::ios::out) ;
     assert(outputFile.fail() == false) ;

     // save the number of elements of the vector
     outputFile << saveMe.size ()	
                << '\n' ;

     // save the matrix sizes
     outputFile << (*saveMe.begin ())->num_row ()
                << '\t'
                << (*saveMe.begin ())->num_col ()
                << '\n' ;

     // loop over the vector
     for (const_iterator it = saveMe.begin () ;
	       it != saveMe.end () ;
	       ++it)
       {
	       outputFile << (*it) ;
       } // loop over the vecor

     return 0 ;
}


std::istream &
operator>> (std::istream& input, CLHEP::HepGenMatrix &matrix)
{
  int numRow = 0 ;
  int numCol = 0 ;
  
  //PG read the matrix dimension
  input >> numRow ;
  input >> numCol ;
  
  //PG check whether the matrices have the right dimension
  assert ( numRow == matrix.num_row () ) ;
  assert ( numCol == matrix.num_col () ) ;

  //PG get the matrix elements from the file
  for (int row=0 ; row<numRow ; ++row)
    {
      for (int col=0 ; col<numCol ; ++col)
	      {
	        input >> matrix[row][col] ;
            assert (col*row < numRow*numCol) ;
	      }	
    }	

  return input ;
}

bool matrixSaver::touch (std::string inputFileName)
{
   std::fstream inputFile (inputFileName.c_str (), std::ios::in) ;
   return !inputFile.fail();
}



CLHEP::HepGenMatrix *
matrixSaver::getMatrix (std::string inputFileName)
{
     //PG open the output file
     std::fstream inputFile (inputFileName.c_str (), std::ios::in) ;
     if (inputFile.fail()) std::cerr << "file: " << inputFileName << std::endl ;
     assert(inputFile.fail() == false);

     //PG get the matrix dimensions
     int numRow = 0 ;
     int numCol = 0 ;
     inputFile >> numRow ;
     inputFile >> numCol ;

     //PG instantiate the matrix
     CLHEP::HepGenMatrix * matrix ;
     if (numCol > 1)
	     matrix = new CLHEP::HepMatrix (numRow, numCol, 0) ;
     else
	     matrix = new CLHEP::HepVector (numRow, 0) ;

     inputFile >> *matrix ;

     return matrix ;
}


std::vector<CLHEP::HepGenMatrix*> *
matrixSaver::getMatrixVector (std::string inputFileName)
{
     // open the output file
     std::fstream inputFile (inputFileName.c_str (), std::ios::in) ;
     assert(inputFile.fail() == false);

     // get the vector length
     int numElem = 0 ;
     inputFile >> numElem ;
     
     // get the matrix dimensions
     int numRow = 0 ;
     int numCol = 0 ;
     inputFile >> numRow ;
     inputFile >> numCol ;

     //PG prepara il vector
     std::vector<CLHEP::HepGenMatrix*>* matrixVector = 
       new std::vector<CLHEP::HepGenMatrix*> (numElem) ;
          
     //PG loop sugli elementi del vettore
     for (int i=0 ; i<numElem ; ++i)
       {
          //PG definisce il puntatore
          CLHEP::HepGenMatrix * matrix ;
          //PG attribuisce un oggetto concreto

          if (numCol > 1)
	          matrix = new CLHEP::HepMatrix (numRow, numCol, 0) ;
          else
	          matrix = new CLHEP::HepVector (numRow, 0) ;
          
          //PG scarica su un oggetto concreto
          inputFile >> *matrix ;

          //PG riempie il vettore
          (*matrixVector)[i] = matrix ;
       }

     return matrixVector ;
}


std::vector<CLHEP::HepMatrix> 
matrixSaver::getConcreteMatrixVector (std::string inputFileName)
{
     // open the output file
     std::fstream inputFile (inputFileName.c_str (), std::ios::in) ;
     assert(inputFile.fail() == false);

     // get the vector length
     int numElem = 0 ;
     inputFile >> numElem ;
     
     // get the matrix dimensions
     int numRow = 0 ;
     int numCol = 0 ;
     inputFile >> numRow ;
     inputFile >> numCol ;

     //PG prepara il vector
     std::vector<CLHEP::HepMatrix> matrixVector (
         numElem,
         CLHEP::HepMatrix (numRow,numCol,0)
       ) ;
     
     //PG loop sugli elementi del vettore
     for (int i=0 ; i<numElem ; ++i)
       {
          
          //PG scarica su un oggetto concreto
          inputFile >> matrixVector[i] ;

       }

     return matrixVector ;
}
