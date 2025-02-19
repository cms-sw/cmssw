/** 
    $Date: 2011/06/30 10:10:52 $
    $Revision: 1.1 $
    $Id: BlockSolver.cc,v 1.1 2011/06/30 10:10:52 muzaffar Exp $ 
    \author $Author: muzaffar $
*/

#include "Calibration/Tools/interface/BlockSolver.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
int 
BlockSolver::operator () (const CLHEP::HepMatrix & matrix, 
                          const CLHEP::HepVector & vector,
                          CLHEP::HepVector & result) 
{
  double threshold = matrix.trace () / 
                     static_cast<double> (matrix.num_row ()) ;
  std::vector<int> holes ;
  // loop over the matrix rows
  for (int row = 0 ; row < matrix.num_row () ; ++row)
    {
      double sumAbs = 0. ;
      for (int col = 0 ; col < matrix.num_col () ; ++col)
        sumAbs += fabs (matrix[row][col]) ;
      if (sumAbs < threshold) holes.push_back (row) ;  
    } // loop over the matrix rows

  int dim = matrix.num_col () - holes.size () ;

  if (holes.size () == 0) //PG exceptional case!
    {
      for (int i = 0 ; i < result.num_row () ; ++i)
        result[i] = 1. ;
    }    
  else if (dim > 0) 
    {
      CLHEP::HepMatrix solution (dim, dim, 0) ;
      CLHEP::HepVector input (dim, 0) ;
      shrink (matrix, solution, vector, input, holes) ;
      CLHEP::HepVector output = solve (solution,input) ;
      pour (result, output, holes) ;
    } 
  return holes.size () ;
}


// ------------------------------------------------------------


void 
BlockSolver::shrink (const CLHEP::HepMatrix & matrix,
                     CLHEP::HepMatrix & solution,
                     const CLHEP::HepVector & result,
                     CLHEP::HepVector & input,
                     const std::vector<int> & where)
{
  
  int offsetRow = 0 ;
  std::vector<int>::const_iterator whereRows = where.begin () ;
  // loop over rows
  for (int row = 0 ; row < matrix.num_row () ; ++row)
    {
      if (row == *whereRows) 
        {
//          std::cerr << "        DEBUG shr hole found " << std::endl ;
          ++offsetRow ;
          ++whereRows ;
          continue ;
        } 
      input[row-offsetRow] = result[row] ;        
      int offsetCol = 0 ;
      std::vector<int>::const_iterator whereCols = where.begin () ;
      // loop over columns
      for (int col = 0 ; col < matrix.num_col () ; ++col)
        {
          if (col == *whereCols) 
            {
              ++offsetCol ;
              ++whereCols ;
              continue ;
            }
          solution[row-offsetRow][col-offsetCol] = matrix[row][col] ;    
        }
    } // loop over rows
  return ;  
}                      


// ------------------------------------------------------------


void
BlockSolver::pour (CLHEP::HepVector & result,
                   const CLHEP::HepVector & output,
                   const std::vector<int> & where) 
{
  std::vector<int>::const_iterator whereCols = where.begin () ;
  int readingIndex = 0 ;
  //PG loop over the output crystals
  for (int xtal = 0 ; xtal < result.num_row () ; ++xtal)
    {
      if (xtal == *whereCols)
        {
          result[xtal] = 1. ;
          ++whereCols ;        
        }
      else
        {
          result[xtal] = output[readingIndex] ;
          ++readingIndex ;
        }    
    } //PG loop over the output crystals

  return ;  
}                

