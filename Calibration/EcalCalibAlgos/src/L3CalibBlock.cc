/**
   $Date: 2008/11/14 11:40:34 $
    $Revision: 1.4 $
    $Id: L3CalibBlock.cc,v 1.4 2008/11/14 11:40:34 presotto Exp $ 
    \author $Author: presotto $
*/

#include "Calibration/EcalCalibAlgos/interface/L3CalibBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TH1F.h"
#include "TFile.h"


// -----------------------------------------------------


L3CalibBlock::L3CalibBlock (const int numberOfElements, 
                            const int keventweight):
  VEcalCalibBlock (numberOfElements), 
  m_L3AlgoUniv (new MinL3AlgoUniv<unsigned int>(keventweight))
{
  reset () ;
}


// -----------------------------------------------------


L3CalibBlock::~L3CalibBlock()
{
  delete m_L3AlgoUniv ;
}


// -----------------------------------------------------


void
L3CalibBlock::Fill (std::map<int,double>::const_iterator MapBegin,
                    std::map<int,double>::const_iterator MapEnd ,
                    double pTk,
                    double pSubtract,
                    double sigma)
{
  // to feed the L3 algo
  std::vector<float> energy ;
  std::vector<unsigned int> position ;
  // loop over the energies map
  for (std::map<int,double>::const_iterator itMap = MapBegin ; 
       itMap != MapEnd ; 
       ++itMap)
    {
      // translation into vectors for the L3 algo
      position.push_back (itMap->first) ;
      energy.push_back (itMap->second) ;
    } // loop over the energies map
  m_L3AlgoUniv->addEvent (energy, position, pTk-pSubtract) ;

  return ;
}


// ------------------------------------------------------------


int
L3CalibBlock::solve (int usingBlockSolver, double min, double max)
{
  m_coefficients = m_L3AlgoUniv->getSolution () ;
  return 0 ;
}


// ------------------------------------------------------------


void 
L3CalibBlock::reset () 
{
  //PG FIXME could it be it is not wanted to be reset?
  m_L3AlgoUniv->resetSolution () ;
  return ;
}


