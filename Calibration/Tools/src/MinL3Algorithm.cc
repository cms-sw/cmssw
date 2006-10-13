
/** \file GenericMinL3Algorithm.cc
 *
 * $Date: 2006/10/13 $
 * $Revision: 2.0 $
 * \author R.Ofierzynski, CERN
 */

#include "Calibration/Tools/interface/MinL3Algorithm.h"


MinL3Algorithm::MinL3Algorithm(int squareMode_, int mineta_, int maxeta_, int minphi_, int maxphi_)
  :squareMode(squareMode_), mineta(mineta_), maxeta(maxeta_), minphi(minphi_), maxphi(maxphi_), countEvents(0)
{
  int Neta = maxeta - mineta + 1;
  if (mineta * maxeta < 0) Neta--; // there's no eta index = 0
  int Nphi = maxphi - minphi + 1;
  if (Nphi <0) Nphi += 360;
  
  Nchannels = Neta * Nphi; // no. of channels, get it from edges of the region

  Nxtals = squareMode*squareMode; // no. of xtals in one event

  wsum.assign(Nchannels,0.);
  Ewsum.assign(Nchannels,0.);
}


MinL3Algorithm::~MinL3Algorithm()
{
}


vector<float> MinL3Algorithm::iterate(const vector<vector<float> >& eventMatrix, const vector<int>& VmaxCeta, const vector<int>& VmaxCphi, const vector<float>& energyVector, const int& nIter, const bool& normalizeFlag)
{
  int Nevents = eventMatrix.size(); // Number of events to calibrate with

  vector<float> totalSolution(Nchannels,1.);
  vector<float> iterSolution;
  vector<vector<float> > myEventMatrix(eventMatrix);
  vector<float> myEnergyVector(energyVector);

  int i, j;

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
	      for (j=0; j<Nxtals; j++) {sumOverEnergy += myEventMatrix[i][j];}
	      sumOverEnergy /= myEnergyVector[i];
	      scale += sumOverEnergy;
	    }
	  scale /= Nevents;
	  
	  for (i=0; i<Nevents; i++) {myEnergyVector[i] *= scale;}	  
	} // end normalize energies

      // now the real work starts:
      for (int iEvt=0; iEvt < Nevents; iEvt++)
	{
	  addEvent(myEventMatrix[iEvt], VmaxCeta[iEvt], VmaxCphi[iEvt], myEnergyVector[iEvt]);
	}
      iterSolution = getSolution();
      if (iterSolution.empty()) return iterSolution;

      // re-calibrate eventMatrix with solution
      for (int ievent = 0; ievent<Nevents; ievent++)
	{
	  myEventMatrix[ievent] = recalibrateEvent(myEventMatrix[ievent], VmaxCeta[ievent], VmaxCphi[ievent], iterSolution);
	}

      for (int i=0; i<Nchannels; i++) 
	{
	  // save solution into theCalibVector
	  totalSolution[i] *= iterSolution[i];
	}
      //      resetSolution(); // reset for new iteration, now: getSolution does it automatically if not vetoed
    } // end iterate correction

  return totalSolution;
}


void MinL3Algorithm::addEvent(const vector<float>& eventSquare, const int& maxCeta, const int& maxCphi, const float& energy)
{
  countEvents++;

  float w, invsumXmatrix;
  int iFull, i;
  // Loop over the crystal matrix to find the sum
  float sumXmatrix=0.;
      
  for (i=0; i<Nxtals; i++) { sumXmatrix+=eventSquare[i]; }
      
  if (sumXmatrix != 0.)
    {
      invsumXmatrix = 1/sumXmatrix;
      // Loop over the crystal matrix (3x3,5x5,7x7) again and calculate the weights for each xtal
      for (i=0; i<Nxtals; i++) 
	{		
	  w = eventSquare[i] * invsumXmatrix;

	  iFull = indexSqr2Reg(i, maxCeta, maxCphi);
	  if (iFull >= 0)
	    {
	      wsum[iFull] += w;
	      Ewsum[iFull] += (w * energy * invsumXmatrix);
	    }
	}
    }
  //  else {cout << " Debug: dropping null event: " << countEvents << endl;}
}


vector<float> MinL3Algorithm::getSolution(bool resetsolution)
{
  vector<float> solution(Nchannels,1.);

  for (int i=0; i<Nchannels; i++) 
    {
      if (wsum[i] != 0.) 
	{ solution[i]*=Ewsum[i]/wsum[i];}
      //      else 
      //	{ cout << "warning - no event data for crystal index (reduced region) " << i << endl; }
    }
  
  if (resetsolution) resetSolution();

  return solution;
}


void MinL3Algorithm::resetSolution()
{
  wsum.assign(Nchannels,0.);
  Ewsum.assign(Nchannels,0.);
}


vector<float> MinL3Algorithm::recalibrateEvent(const vector<float>& eventSquare, const int& maxCeta, const int& maxCphi, const vector<float>& recalibrateVector)
{
  vector<float> newEventSquare(eventSquare);
  int iFull;

  for (int i=0; i<Nxtals; i++) 
    {
      iFull = indexSqr2Reg(i, maxCeta, maxCphi);
      if (iFull >=0)
	newEventSquare[i] *= recalibrateVector[iFull];
    }
  return newEventSquare;
}


int MinL3Algorithm::indexSqr2Reg(const int& sqrIndex, const int& maxCeta, const int& maxCphi)
{
  int regionIndex;

  // get the current eta, phi indices
  int curr_eta = maxCeta - squareMode/2 + sqrIndex%squareMode;
  if (curr_eta * maxCeta <= 0) {if (maxCeta > 0) curr_eta--; else curr_eta++; }  // JUMP over 0

  int curr_phi = maxCphi - squareMode/2 + sqrIndex/squareMode;
  if (curr_phi < 1) curr_phi += 360;
  if (curr_phi > 360) curr_phi -= 360;

  bool negPhiDirection = (maxphi < minphi);
  int iFullphi;

  regionIndex = -1;

  if (curr_eta >= mineta && curr_eta <= maxeta)
    if ( (!negPhiDirection && (curr_phi >= minphi && curr_phi <= maxphi)) ||
	 (negPhiDirection && !(curr_phi >= minphi && curr_phi <= maxphi))      ) 
      {
	iFullphi = curr_phi - minphi;
	if (iFullphi < 0) iFullphi += 360;
	regionIndex = (curr_eta - mineta) * (maxphi - minphi + 1 + 360*negPhiDirection) + iFullphi;
      }

  return regionIndex;
}
