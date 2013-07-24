#ifndef MinL3AlgoUniv_H
#define MinL3AlgoUniv_H

/** \class MinL3AlgoUniv
 *  Implementation of the L3 Collaboration algorithm to solve a system Ax = B
 *  by minimization of |Ax-B| using an iterative linear approach
 *  This class should be universal, i.e. working with DetIds or whatever else 
 *    will be invented to identify Subdetector parts
 *  The bookkeeping of the cluster size and its elements has to be done by the user.
 *
 * $Date: 2007/08/23 12:38:02 $
 * $Revision: 1.2 $
 * \author R.Ofierzynski, CERN
 */

#include <vector>
#include <iostream>
#include <map>
#include <math.h>

template<class IDdet>
class MinL3AlgoUniv
{
public:
  typedef std::map<IDdet,float> IDmap;
  typedef typename IDmap::value_type IDmapvalue;
  typedef typename IDmap::iterator iter_IDmap;
  typedef typename IDmap::const_iterator citer_IDmap;

  /// Default constructor
  /// kweight_ = event weight
  MinL3AlgoUniv(float kweight_ = 0.);

  /// Destructor
  ~MinL3AlgoUniv();

  /// method doing the full calibration running nIter number of times, 
  ///   recalibrating the event matrix after each iteration with the new solution
  /// returns the vector of calibration coefficients built from all iteration solutions
  /// >> also to be used also as recipe on how to use the calibration methods <<
  /// >> one-by-one with a re-selection of the events in between the iterations<<
  IDmap iterate(const std::vector<std::vector<float> >& eventMatrix, const std::vector<std::vector<IDdet> >& idMatrix, const std::vector<float>& energyVector, const int& nIter, const bool& normalizeFlag = false);

  /// add event to the calculation of the calibration vector
  void addEvent(const std::vector<float>& myCluster, const std::vector<IDdet>& idCluster, const float& energy);

  /// recalibrate before next iteration: give previous solution vector as argument
  std::vector<float> recalibrateEvent(const std::vector<float>& myCluster, const std::vector<IDdet>& idCluster, const IDmap& newCalibration); 

  /// get the solution at the end of the calibration as a map between
  /// DetIds and calibration constant
  IDmap getSolution(const bool resetsolution=true);

  /// reset for new iteration
  void resetSolution(); 

private:

  float kweight;
  int countEvents;
  IDmap wsum;
  IDmap Ewsum;

};



template<class IDdet>
MinL3AlgoUniv<IDdet>::MinL3AlgoUniv(float kweight_)
  :kweight(kweight_), countEvents(0)
{
  resetSolution();
}


template<class IDdet>
MinL3AlgoUniv<IDdet>::~MinL3AlgoUniv()
{
}


template<class IDdet>
typename MinL3AlgoUniv<IDdet>::IDmap MinL3AlgoUniv<IDdet>::iterate(const std::vector<std::vector<float> >& eventMatrix, const std::vector<std::vector<IDdet> >& idMatrix, const std::vector<float>& energyVector, const int& nIter, const bool& normalizeFlag)
{
  int Nevents = eventMatrix.size(); // Number of events to calibrate with

  IDmap totalSolution;
  IDmap iterSolution;
  std::vector<std::vector<float> > myEventMatrix(eventMatrix);
  std::vector<float> myEnergyVector(energyVector);

  int i;

  // Iterate the correction
  for (int iter=1;iter<=nIter;iter++) 
    {

      // if normalization flag is set, normalize energies
      float sumOverEnergy;
      if (normalizeFlag)
	{
	  float scale = 0.;
	  
	  for (i=0; i<Nevents; i++)
	    {
	      sumOverEnergy = 0.;
	      for (unsigned j=0; j<myEventMatrix[i].size(); j++) {sumOverEnergy += myEventMatrix[i][j];}
	      sumOverEnergy /= myEnergyVector[i];
	      scale += sumOverEnergy;
	    }
	  scale /= Nevents;
	  
	  for (i=0; i<Nevents; i++) {myEnergyVector[i] *= scale;}	  
	} // end normalize energies

      // now the real work starts:
      for (int iEvt=0; iEvt < Nevents; iEvt++)
	{
	  addEvent(myEventMatrix[iEvt], idMatrix[iEvt], myEnergyVector[iEvt]);
	}
      iterSolution = getSolution();
      if (iterSolution.empty()) return iterSolution;

      // re-calibrate eventMatrix with solution
      for (int ievent = 0; ievent<Nevents; ievent++)
	{
	  myEventMatrix[ievent] = recalibrateEvent(myEventMatrix[ievent], idMatrix[ievent], iterSolution);
	}

      // save solution into theCalibVector
      for (iter_IDmap i = iterSolution.begin(); i != iterSolution.end(); i++)
	{
	  iter_IDmap itotal = totalSolution.find(i->first);
	  if (itotal == totalSolution.end())
	    {
	      totalSolution.insert(IDmapvalue(i->first,i->second));
	    }
	  else
	    {
	      itotal->second *= i->second;
	    }
	}

      //      resetSolution(); // reset for new iteration, now: getSolution does it automatically if not vetoed
    } // end iterate correction

  return totalSolution;
}


template<class IDdet>
void MinL3AlgoUniv<IDdet>::addEvent(const std::vector<float>& myCluster, const std::vector<IDdet>& idCluster, const float& energy)
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
      // Loop over the crystal matrix (3x3,5x5,7x7) again and calculate the weights for each xtal
      for (unsigned i=0; i<myCluster.size(); i++) 
	{		
	  w = myCluster[i] * invsumXmatrix;

	  // include the weights into wsum, Ewsum
	  iter_IDmap iwsum = wsum.find(idCluster[i]);
	  if (iwsum == wsum.end()) wsum.insert(IDmapvalue(idCluster[i],w*eventw));
	  else iwsum->second += w*eventw;

	  iter_IDmap iEwsum = Ewsum.find(idCluster[i]);
	  if (iEwsum == Ewsum.end()) Ewsum.insert(IDmapvalue(idCluster[i], (w*eventw * energy * invsumXmatrix) ));
	  else iEwsum->second += (w*eventw * energy * invsumXmatrix);
	}
    }
  //  else {std::cout << " Debug: dropping null event: " << countEvents << std::endl;}
}


template<class IDdet>
typename MinL3AlgoUniv<IDdet>::IDmap MinL3AlgoUniv<IDdet>::getSolution(const bool resetsolution)
{
  IDmap solution;

  for (iter_IDmap i = wsum.begin(); i != wsum.end(); i++)
    {
      iter_IDmap iEwsum = Ewsum.find(i->first);
      float myValue = 1;
      if (i->second != 0) myValue = iEwsum->second / i->second;

      solution.insert(IDmapvalue(i->first,myValue));
    }
  
  if (resetsolution) resetSolution();

  return solution;
}


template<class IDdet>
void MinL3AlgoUniv<IDdet>::resetSolution()
{
  wsum.clear();
  Ewsum.clear();
}


template<class IDdet>
std::vector<float> MinL3AlgoUniv<IDdet>::recalibrateEvent(const std::vector<float>& myCluster, const std::vector<IDdet> &idCluster, const IDmap& newCalibration)
{
  std::vector<float> newCluster(myCluster);

  for (unsigned i=0; i<myCluster.size(); i++) 
    {
      citer_IDmap icalib = newCalibration.find(idCluster[i]);
      if (icalib != newCalibration.end())
	{
	  newCluster[i] *= icalib->second;
	}
      else
	{
	  std::cout << "No calibration available for this element." << std::endl;
	}

    }

  return newCluster;
}


#endif // MinL3AlgoUniv_H
