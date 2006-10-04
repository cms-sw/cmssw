
/** \file MinL3Algorithm.cc
 *
 * $Date: 2006/08/25 $
 * $Revision: 1.2 $
 *
 * \author R.Ofierzynski, CERN
 */

#include "Calibration/Tools/interface/MinL3Algorithm.h"


MinL3Algorithm::MinL3Algorithm(bool normalise)
  :normaliseFlag(normalise)
// if normalize=true: for all events sum the e25ovTRP. then take the mean value and rescale all TrP
{
}


MinL3Algorithm::~MinL3Algorithm()
{
}


vector<float> MinL3Algorithm::iterate(const vector<vector<float> >& eventMatrix, const vector<float>& energyVector, const int nIter)
{
  vector<float> solution;
  vector<float> theCalibVector(energyVector.size(),1.);
  vector<vector<float> > myEventMatrix(eventMatrix);
  int Nevents = eventMatrix.size(); // Number of events to calibrate with
  int Nchannels = eventMatrix[0].size(); // Number of channel coefficients

  // Iterate the correction
  for (int iter=1;iter<=nIter;iter++) 
    {
      // make one iteration
      solution = iterate(myEventMatrix, energyVector);

      if (solution.empty()) return solution;
      // R.O.: or throw an exception, what's the standard CMS way ?

      // re-calibrate eventMatrix with solution
      for (int i=0; i<Nchannels; i++) 
	{
	  for (int ievent = 0; ievent<Nevents; ievent++)
	    {
	      myEventMatrix[ievent][i] *= solution[i];
	    }
	  // save solution into theCalibVector
	  theCalibVector[i] *= solution[i];
	}

    } // end iterate the correction

  return theCalibVector;
}


vector<float> MinL3Algorithm::iterate(const vector<vector<float> >& eventMatrix, const vector<float>& energyVector)
{
  vector<float> solution;
  vector<float> myEnergyVector(energyVector);

  int Nevents = eventMatrix.size(); // Number of events to calibrate with
  int Nchannels = eventMatrix[0].size(); // Number of channel coefficients

  // Sanity check
  if (Nevents != myEnergyVector.size()) 
    {
      cout << "MinL3Algorithm::iterate(): Error: bad matrix dimensions. Dropping out." << endl;
      return solution; // empty vector !
    }

  // initialize the solution vector with 1.
  solution.assign(Nchannels,1.);

  int ievent, i, j;

  // if normalization flag is set, normalize energies
  float sumOverEnergy;
  if (normaliseFlag)
    {
      float scale = 0.;
      
      cout << "MinL3Algorithm::iterate(): Normalising event data" << endl;

      for (i=0; i<Nevents; i++)
	{
	  sumOverEnergy = 0.;
	  for (j=0; j<Nchannels; j++) {sumOverEnergy += eventMatrix[i][j];}
	  sumOverEnergy /= myEnergyVector[i];
	  scale += sumOverEnergy;
	}
      scale /= Nevents;
      cout << "  Normalisation = " << scale << endl;
      
      for (i=0; i<Nevents; i++) {myEnergyVector[i] *= scale;}	  
    } // end normalize energies


  // This is where the real work goes on...
  float sum25, invsum25;
  float w; // weight for event
  vector<float> wsum(Nchannels,0.); // sum of weights for a crystal
  vector<float> Ewsum(Nchannels,0.); // sum of products of weight*Etrue/E25

  // Loop over events
  for (ievent = 0; ievent<Nevents; ievent++)
    {
      // Loop over the 5x5 to find the sum
      sum25=0.;
      
      for (i=0; i<Nchannels; i++) { sum25+=eventMatrix[ievent][i]; } //*calibs[i];
      
      if (sum25 != 0.)
	{
	  invsum25 = 1/sum25;
	  // Loop over the 5x5 again and calculate the weights for each xtal
	  for (i=0; i<Nchannels; i++) 
	    {		
	      w = eventMatrix[ievent][i] * invsum25; // * calibs[i]
	      wsum[i] += w;
	      Ewsum[i] += (w * myEnergyVector[ievent] * invsum25);	
	    }
	}
      else {cout << " Debug: dropping null event: " << ievent << endl;}
      
    } // end Loop over events
  
  // Apply correction factors to all channels not in the margin
  for (i=0; i<Nchannels; i++) 
    {
      if (wsum[i] != 0.) 
	{ solution[i]*=Ewsum[i]/wsum[i]; }
      else 
	{ cout << "warning - no event data for crystal index " << i << endl; }
    }

  return solution;
}
