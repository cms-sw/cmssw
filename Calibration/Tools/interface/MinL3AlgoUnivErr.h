#ifndef MinL3AlgoUnivErr_H
#define MinL3AlgoUnivErr_H

//=============================================================================

/** class MinL3AlgoUnivErr

 * $Date: 2009/10/20 12:56:44 $
 * $Revision: 1.1 $
 * \author R.Ofierzynski, CERN, 2007/08/23
 *                              under class name MinL3AlgoUniv
 *  Modified by A.Fedotov :
 *           24.07.09: a calculation of statistical errors implemented
 *                     on top of revision 1.2 2007/08/23 12:38:02 ;
 *                     the code remains backward compatible
 *           20.10.09: class name changed to MinL3AlgoUnivErr in order to
 *                     exclude any effect on older applications that make use
 *                     of the MinL3AlgoUniv class
 *
 *=============================================================================
 *
 *  General purpose
 *  ~~~~~~~~~~~~~~~
 *  Implementation of the L3 Collaboration algorithm to solve a system
 *                        Ax = B
 *  by minimization of |Ax-B| using an iterative linear approach
 *
 *  This class should be universal, i.e. working with DetIds or whatever else 
 *  will be invented to identify Subdetector parts
 *
 *  The bookkeeping of the cluster size and its elements
 *  has to be done by the user.
 *
 *  Calculation of statistical errors
 *  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *  The solution whose errors have to be obtained, is found by a call to
 *  the `iterate' function. The errors are also found within that procedure
 *  in a general phenomenological approach consisting in
 *    - splitting the full sample to a certain number n of equal subsamples
 *                           (an optional argument nSubsamples of `iterate'
 *                            determines n; n = 10  by default)    
 *    - solving the problem for every part separately
 *    - then the spread of partial solutions is a measure of the error:
 *                    error = rms / sqrt (n - 1).                          (1)
 *  The relative precision of such estimate is believed to be 
 *          1 / sqrt[2(n - 1]            which yields 24% for n = 10.
 *  The user can fetch the errors by calling a `getError' function, and
 *  the average partial solution -- via `getMeanPartialSolution'.
 *
 *  Known PROBLEMS:
 *     1. If the event statistics for a particular cell is low enough, then
 *  the number of subsamples where the cell is present, n_cell, can be less
 *  than n (e.g, it is always so if a cell is active in 5 events while we
 *  split the full sample into n=10 parts). Then the error of this cell
 *  becomes wrong because a part of the full statistics gets lost for the cell
 *  (b.t.w., n_cell is actually used in eq.(1) ).
 *  The user can check the presence of such cells via function
 *  `numberOfWrongErrors' and check which cells have wrong errors with the
 *  functions `getErrorQuality' which gives the ratio n_cell/n -- the fraction
 *  of the full statistics used for the error estimation.
 *    2. Cases have been observed where the total solution converged nicely 
 *  with the increasing no. of iterations, while some of partial solutions 
 *  did not. Apparently, the errors can explode in such cases and do not
 *  reflect the real stat. errors of the total solution. This seems to be
 *  a problem of instabilities of the L3 method itself.
 *
 */

//=============================================================================

#include <vector>
#include <iostream>
#include <map>
#include <math.h>

//=============================================================================
template <class IDdet>
class MinL3AlgoUnivErr
{
public:
  typedef          std::map<IDdet,float>  IDmapF;
  typedef typename IDmapF::value_type     IDmapFvalue;
  typedef typename IDmapF::iterator       iter_IDmapF;
  typedef typename IDmapF::const_iterator citer_IDmapF;
  typedef          std::map<IDdet,int>    IDmapI;
  typedef typename IDmapI::value_type     IDmapIvalue;
  typedef typename IDmapI::iterator       iter_IDmapI;
  typedef typename IDmapI::const_iterator citer_IDmapI;

  //----------------------------------------------
  /// Default constructor
  /// kweight_ = event weight

  MinL3AlgoUnivErr(float kweight_ = 0.);


  //----------------------------------------------
  /// Destructor

  ~MinL3AlgoUnivErr();


  //----------------------------------------------
  ///  method doing the full calibration running nIter number of times, 
  ///  recalibrating the event matrix after each iteration with the
  ///  new solution
  ///  returns the vector of calibration coefficients built
  ///  from all iteration solutions
  ///
  ///  The calibration is also done for nSubsamples sub-samples in order
  ///  to be able to estimate statistical errors of the main solution.  
  ///
  ///     >> also to be used also as recipe on how to use the calibration
  ///     >> methods one-by-one with a re-selection of the events in between
  ///  the iterations<<

  IDmapF iterate( const std::vector<std::vector<float> > & eventMatrix,
                  const std::vector<std::vector<IDdet> > & idMatrix,
                  const std::vector<float>               & energyVector,
                  const int                              & nIter,
                  const bool                             & normalizeFlag
                                                                       = false,
                  const int                              & nSubsamples = 10
                );
  //----------------------------------------------
  /// method to get the stat. error on the correction factor for cell id
  /// (to be called after the `iterate')
  ///
  ///    special values: getError = -2. : no info for the cell
  ///                             = -1. : the cell was met in one partial
  ///                                     solution only => the error equals to
  ///                                     INFINITY 

  float getError( IDdet id ) const;

  //----------------------------------------------
  /// method to get the stat. errors on the correction factors
  /// for all cells together (to be called after the `iterate').
  /// A map (id,error) is returned containing all the cells
  /// for which the information is available
  /// 
  ///    special value: error = -1. : the cell was met in one partial
  ///                                 solution only => the error equals to
  ///                                 INFINITY 

  IDmapF getError() const;

  //----------------------------------------------
  /// method to get the number of cells where the errors are incorrect
  /// due to nSamples_cell < nSubsamples 
  /// (to be called after the `iterate') .
  /// 
  /// this can happen for a cell if the number of events nev_cell
  /// where the cell is active (i.e. energy > 0),  is small enough,
  /// e.g. it is always so if nev_cell < nSubsamples.

  int numberOfWrongErrors() const;

  //----------------------------------------------
  /// two methods to get the fraction of full statistics used in calculation
  /// of stat. errors (to be called after the `iterate').
  ///
  /// a fraction < 1 indicates that the error is incorrect.
  /// 
  ///     version with one argument: the fraction for a particular cells
  ///                                (special returned value: = -1. if no
  ///                                info for the cell is available =>
  ///                                a wrong id has been given)
  ///             w/o arguments    : the fractions for all cells together  

  float  getErrorQuality( IDdet id ) const;

  IDmapF getErrorQuality()           const;

  //----------------------------------------------
  /// method to get
  /// the mean partial solution for the correction factor for cell id
  /// (to be called after the `iterate')
  ///
  ///    special return value: 999. : no info for the cell

  float getMeanPartialSolution( IDdet id ) const;

  //----------------------------------------------
  /// method to get the mean partial solution for all cells together
  /// (to be called after the `iterate')
  /// A map (id,mean) is returned containing all the cells
  /// for which the information is available

  IDmapF getMeanPartialSolution() const;

  //----------------------------------------------
  /// add event to the calculation of the calibration vector

  void addEvent( const std::vector<float> & myCluster,
                 const std::vector<IDdet> & idCluster,
                 const float              & energy
               );

  //----------------------------------------------
  /// recalibrate before next iteration:
  /// give previous solution vector as argument

  std::vector<float>
  recalibrateEvent( const std::vector<float> & myCluster,
                    const std::vector<IDdet> & idCluster,
                    const IDmapF             & newCalibration,
                    const int                & isol = -1,
                    const int                & iter = -1
                ); 

  //----------------------------------------------
  /// get the solution at the end of the calibration as a map between
  /// DetIds and calibration constant

  IDmapF getSolution( const bool resetsolution=true );

  //----------------------------------------------
  /// reset for new iteration

  void resetSolution(); 

  //----------------------------------------------

private:

  float   kweight;
  int     countEvents;
  IDmapF  wsum;
  IDmapF  Ewsum;
  IDmapI                   idToIndex;    // map: cell id -> index of cell info
                                         // in sumPartSolu... vectors
  std::vector<int>         sumPartSolu0; // number of partial solutions
  std::vector<float>       sumPartSolu1; // sum of partial solutions
  std::vector<float>       sumPartSolu2; // sum of squared partial solutions

  int nOfSubsamples ; //  a copy of an input argument of `iterate' function

  //----------------------------------------------
  // register a partial solution in data members
  void registerPartialSolution( const IDmapF & partialSolution );

  //----------------------------------------------

};                                 //end of class MinL3AlgoUnivErr prototype


//=============================================================================


template<class IDdet>
MinL3AlgoUnivErr<IDdet>::MinL3AlgoUnivErr(float kweight_)
  :kweight(kweight_), countEvents(0)
{
  std::cout << "MinL3AlgoUnivErr : L3 algo with a calculation"
            << " of stat.errors working..." << std::endl;
  resetSolution();
}

//=============================================================================

template<class IDdet>
MinL3AlgoUnivErr<IDdet>::~MinL3AlgoUnivErr()
{
}

//=============================================================================

template<class IDdet>
typename MinL3AlgoUnivErr<IDdet>::IDmapF
MinL3AlgoUnivErr<IDdet>::
iterate( const std::vector < std::vector<float> > & eventMatrix,
         const std::vector < std::vector<IDdet> > & idMatrix,
         const std::vector <float>                & energyVector,
         const int                                & nIter,
         const bool                               & normalizeFlag,
         const int                                & nSubsamples
       )
{
  // clear the data members which are filled inside the function
  //                               (in registerPartialSolution called from here)

  nOfSubsamples = nSubsamples; // keep the input argument for use in other
                               // functions

  idToIndex    .clear();
  sumPartSolu0 .clear();
  sumPartSolu1 .clear();
  sumPartSolu2 .clear();

  IDmapF totalSolution;

  // Loop over samples/solutions:
  //    isol = 0                 : all events with the solution stored in
  //                               totalSolution
  //    isol = 1,...,nSubsamples : partial solutions are found for sub-samples
  //                               with the info on the solutions stored in the
  //                               data members
  //                                    idToIndex, sumPartSolu...
  //                               in order to be able to estimate the stat.
  //                               errors later

  for (int isol = 0 ; isol <= nSubsamples; isol++) {

    IDmapF sampleSolution ;  // solution for the sample
    IDmapF iterSolution   ;  // intermediate solution after an iteration
    std::vector < std::vector<float> > myEventMatrix  ;
    std::vector < std::vector<IDdet> > myIdMatrix     ;
    std::vector <float>                myEnergyVector ;

    // Select the sample.
    // Fill myEventMatrix, myIdMatrix and myEnergyVector 
    // either with all evs or with independent event subsamples

    if (isol == 0)  // total solution
      {
        myEventMatrix  = eventMatrix  ;
        myIdMatrix     = idMatrix     ;
        myEnergyVector = energyVector ;
      }
    else            // partial solution # isol
      {
        // clear containers filled for the previous sample
        sampleSolution .clear() ;
        myEventMatrix  .clear() ;
        myIdMatrix     .clear() ;
        myEnergyVector .clear() ;

        for (int i = 0; i < static_cast<int>( eventMatrix.size() ); i++)
          {
            // select every nSubsamples'th event to the subsample
            if ( i % nSubsamples + 1  ==  isol )
              {
                myEventMatrix  .push_back (eventMatrix  [i]) ;
                myIdMatrix     .push_back (idMatrix     [i]) ;
                myEnergyVector .push_back (energyVector [i]) ;
              }
          }
      }

    int Nevents = myEventMatrix.size(); // Number of events to calibrate with
    countEvents = 0;

    int i;

    // Iterate the correction
    for (int iter=1 ; iter <= nIter ; iter++) 
      {

        // if normalization flag is set, normalize energies
        float sumOverEnergy;
        if (normalizeFlag)
          {
            float scale = 0.;
            
            for (i=0; i<Nevents; i++)
              {
                sumOverEnergy = 0.;
                for (unsigned j=0 ; j < myEventMatrix[i].size() ; j++)
                                       {sumOverEnergy += myEventMatrix[i][j];}
                sumOverEnergy /= myEnergyVector[i];
                scale += sumOverEnergy;
              }
            scale /= Nevents;
          
            for (i=0; i<Nevents; i++) {myEnergyVector[i] *= scale;}       
          } // end normalize energies

        // now the real work starts:
        for (int iEvt=0; iEvt < Nevents; iEvt++)
          {
            addEvent( myEventMatrix[iEvt], myIdMatrix[iEvt]
                                         , myEnergyVector[iEvt] );
          }
        iterSolution = getSolution();
        if (iterSolution.empty())
          { sampleSolution.clear();
            break;    // exit the iteration loop leaving sampleSolution empty 
          }

        // re-calibrate eventMatrix with solution
        for (int ievent = 0; ievent<Nevents; ievent++)
          {
            myEventMatrix[ievent] = recalibrateEvent (myEventMatrix[ievent],
                                                      myIdMatrix[ievent],
                                                      iterSolution,
                                                      isol,iter);
          }

        // save solution into the sampleSolution map
        for (iter_IDmapF i = iterSolution.begin(); i != iterSolution.end(); i++)
          {
            iter_IDmapF itotal = sampleSolution.find(i->first);
            if (itotal == sampleSolution.end())
              {
                sampleSolution.insert(IDmapFvalue(i->first,i->second));
              }
            else
              {
                itotal->second *= i->second;
              }
          }

        //      resetSolution(); // reset for new iteration,
        //               now: getSolution does it automatically if not vetoed
      } // end iterate correction
    
    if (isol == 0) // total solution
      {
        totalSolution = sampleSolution;
      }
    else           // partial solution => register it in sumPartSolu...
      {
        registerPartialSolution( sampleSolution );
      }

  }  // end of the loop over solutions/samples

  return totalSolution;
}


//=============================================================================

template<class IDdet>
void MinL3AlgoUnivErr<IDdet>::
addEvent( const std::vector<float> & myCluster,
          const std::vector<IDdet> & idCluster,
          const float              & energy
        )
{
  countEvents++;

  float w, invsumXmatrix;
  float eventw;

  // Loop over the crystal matrix to find the sum
  float sumXmatrix=0.;
  for (unsigned i=0; i<myCluster.size(); i++) { sumXmatrix += myCluster[i]; }
      
  // event weighting
  eventw = 1 - fabs(1 - sumXmatrix/energy);
  eventw = pow(eventw,kweight);
      
  if (sumXmatrix != 0.)
    {
      invsumXmatrix = 1/sumXmatrix;
      // Loop over the crystal matrix (3x3,5x5,7x7) again
      // and calculate the weights for each xtal
      for (unsigned i=0; i<myCluster.size(); i++) 
        {               
          w = myCluster[i] * invsumXmatrix;

          // include the weights into wsum, Ewsum
          iter_IDmapF iwsum = wsum.find(idCluster[i]);
          if (iwsum == wsum.end())
                               wsum.insert(IDmapFvalue(idCluster[i],w*eventw));
          else iwsum->second += w*eventw;

          iter_IDmapF iEwsum = Ewsum.find(idCluster[i]);
          if (iEwsum == Ewsum.end())
                Ewsum.insert(IDmapFvalue (idCluster[i],
                                          (w*eventw * energy * invsumXmatrix)
                                         ));
          else iEwsum->second += (w*eventw * energy * invsumXmatrix);
        }
    }
  //  else {std::cout << " Debug: dropping null event: " << countEvents << std::endl;}
}

//=============================================================================

template<class IDdet>
typename MinL3AlgoUnivErr<IDdet>::IDmapF
MinL3AlgoUnivErr<IDdet>::
getSolution( const bool resetsolution )
{
  IDmapF solution;

  for (iter_IDmapF i = wsum.begin(); i != wsum.end(); i++)
    {
      iter_IDmapF iEwsum = Ewsum.find(i->first);
      float myValue = 1;
      if (i->second != 0) myValue = iEwsum->second / i->second;

      solution.insert( IDmapFvalue(i->first,myValue) );
    }
  
  if (resetsolution) resetSolution();

  return solution;
}

//=============================================================================

template<class IDdet>
void MinL3AlgoUnivErr<IDdet>::resetSolution()
{
  wsum.clear();
  Ewsum.clear();
}

//=============================================================================

template<class IDdet>
std::vector<float>
MinL3AlgoUnivErr<IDdet>::
recalibrateEvent( const std::vector <float> & myCluster,
                  const std::vector <IDdet> & idCluster,
                  const IDmapF              & newCalibration,
                  const int                 & isol,     // for a printout only
                  const int                 & iter      // for a printout only
                )
{
  std::vector<float> newCluster(myCluster);

  for (unsigned i=0; i<myCluster.size(); i++) 
    {
      citer_IDmapF icalib = newCalibration.find(idCluster[i]);
      if (icalib != newCalibration.end())
        {
          newCluster[i] *= icalib->second;
        }
      else
        {
          std::cout << "No calibration available for this element."
                    << std::endl;
          std::cout  << "   isol = " << isol
                     << "   iter = " << iter
                     << "   idCluster[i] = " << idCluster[i] << "\n";
        }

    }

  return newCluster;
}

//=============================================================================

template<class IDdet>
void
MinL3AlgoUnivErr<IDdet>::
registerPartialSolution( const IDmapF & partialSolution )

{
  int lastIndex = sumPartSolu0.size() - 1; // index of the last element
                                           // of the parallel vectors

  for (citer_IDmapF cell  = partialSolution.begin() ;
                    cell != partialSolution.  end() ; ++cell)
    {
      IDdet id   = cell->first ;
      float corr = cell->second;

      // where is the cell in another map?
      iter_IDmapI cell2 =  idToIndex.find( id );

      if ( cell2 == idToIndex.end() )
        {
          // the cell is met for the first time in patial solutions
          // => insert the info to the end of the vectors

          sumPartSolu0.push_back( 1 );
          sumPartSolu1.push_back( corr );
          sumPartSolu2.push_back( corr * corr );
          idToIndex.insert( IDmapIvalue( id, ++lastIndex ) );
        }
      else
        {
          // add the info to the already registered cell

          int index = cell2 -> second;
          sumPartSolu0[ index ] +=           1;
          sumPartSolu1[ index ] +=        corr;
          sumPartSolu2[ index ] += corr * corr;
        }

    }

}

//=============================================================================

template<class IDdet>
float
MinL3AlgoUnivErr<IDdet>::
getError( IDdet id ) const

{
  float error;
  citer_IDmapI cell =  idToIndex.find( id );
  if ( cell == idToIndex.end() ) error = -2.;     // no info for the cell
  else
    {
      int i = cell->second;
      int n = sumPartSolu0[ i ];
      if (n <= 1 ) error = -1.;     // 1 entry => error estimate impossible
      else
        {
          float meanX  = sumPartSolu1[ i ] / n;
          float meanX2 = sumPartSolu2[ i ] / n;

          error = sqrt( fabs(meanX2 - meanX * meanX) / (n - 1.)  ) ;
        }
        
    }
  return error;
}

//=============================================================================

template<class IDdet>
typename MinL3AlgoUnivErr<IDdet>::IDmapF
MinL3AlgoUnivErr<IDdet>::
getError() const

{
  IDmapF errors;
  float  error;

  for (citer_IDmapI cell  = idToIndex.begin();
                    cell != idToIndex.  end(); ++cell )
    {
      int i = cell->second;
      int n = sumPartSolu0[ i ];
      if (n <= 1 ) error = -1.;     // 1 entry => error estimate impossible
      else
        {
          float meanX  = sumPartSolu1[ i ] / n;
          float meanX2 = sumPartSolu2[ i ] / n;

          error = sqrt( fabs(meanX2 - meanX * meanX) / (n - 1)  ) ;
        }

      errors.insert( IDmapFvalue( cell->first , error ) );
    }
  return errors;
}
//=============================================================================

template<class IDdet>
float
MinL3AlgoUnivErr<IDdet>::
getErrorQuality( IDdet id ) const

{
  float fraction;
  citer_IDmapI cell =  idToIndex.find( id );
  if ( cell == idToIndex.end() ) fraction = -1.;   // no info for the cell
                                                   // => return a special value
  else
    {
      int i = cell->second;
      int n = sumPartSolu0[ i ];
      if (n < nOfSubsamples ) fraction = float( n ) / nOfSubsamples; 
      else                    fraction = 1.;
    }
  return fraction;
}

//=============================================================================

template<class IDdet>
typename MinL3AlgoUnivErr<IDdet>::IDmapF
MinL3AlgoUnivErr<IDdet>::
getErrorQuality()           const

{
  IDmapF fractions;
  float  fraction;

  for (citer_IDmapI cell  = idToIndex.begin();
                    cell != idToIndex.  end(); ++cell )
    {
      int i = cell->second;
      int n = sumPartSolu0[ i ];

      if (n < nOfSubsamples ) fraction = float( n ) / nOfSubsamples; 
      else                    fraction = 1.;

      fractions.insert( IDmapFvalue( cell->first , fraction ) );
    }

  return fractions;
}


//=============================================================================

template<class IDdet>
int
MinL3AlgoUnivErr<IDdet>::
numberOfWrongErrors() const

{
  int nWrong = 0 ;

  for (citer_IDmapI cell  = idToIndex.begin();
                    cell != idToIndex.  end(); ++cell )
    {
      int i = cell->second;
      int n = sumPartSolu0[ i ];

      if (n < nOfSubsamples ) nWrong++;
    }

  return nWrong;
}

//=============================================================================

template<class IDdet>
float
MinL3AlgoUnivErr<IDdet>::
getMeanPartialSolution( IDdet id ) const

{
  float meanX;
  citer_IDmapI cell =  idToIndex.find( id );
  if ( cell == idToIndex.end() ) meanX = 999.;     // no info for the cell
  else
    {
      int i = cell->second;
      int   n      = sumPartSolu0[ i ];
      float meanX  = sumPartSolu1[ i ] / n;
    }
  return meanX;

}

//=============================================================================

template<class IDdet>
typename MinL3AlgoUnivErr<IDdet>::IDmapF
MinL3AlgoUnivErr<IDdet>::
getMeanPartialSolution () const

{
  IDmapF solution;

  for (citer_IDmapI cell  = idToIndex.begin();
                    cell != idToIndex.  end(); ++cell )
    {
      int i = cell->second;
      int   n      = sumPartSolu0[ i ];
      float meanX  = sumPartSolu1[ i ] / n;
      solution.insert( IDmapFvalue( cell->first , meanX ) );
    }
  return solution;
}

//=============================================================================

#endif // MinL3AlgoUnivErr_H
