/**
    $Date: 2012/12/06 09:46:42 $
    $Revision: 1.12 $
    $Id: IMACalibBlock.cc,v 1.12 2012/12/06 09:46:42 eulisse Exp $ 
    \author $Author: eulisse $
*/

#include "Calibration/EcalCalibAlgos/interface/IMACalibBlock.h"
#include "Calibration/Tools/interface/BlockSolver.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "TH1F.h"
#include "TFile.h"
#include <cassert>

// -----------------------------------------------------


IMACalibBlock::IMACalibBlock (const int numberOfElements):
  VEcalCalibBlock (numberOfElements) ,
  m_kaliVector (m_numberOfElements) , 
  m_kaliMatrix (evalX2Size ())
{
  reset () ;
}


// -----------------------------------------------------


IMACalibBlock::~IMACalibBlock()
{
}


// -----------------------------------------------------


void
IMACalibBlock::Fill (std::map<int,double>::const_iterator MapBegin,
                     std::map<int,double>::const_iterator MapEnd ,
                     double pTk,
                     double pSubtract,
                     double sigma)
{
//  std::cerr<<"\n\nfaccio il fill!\n";
  double inverror = 1./sigma ; 
  //LP fist loop over energies
  for (std::map<int,double>::const_iterator itMap1 =MapBegin ; 
       itMap1 !=MapEnd ; 
       ++itMap1)
    {
//      std::cerr<<"numero "<<itMap1->first<<" vale "<<itMap1->second<<"\t"; 
      for (std::map<int,double>::const_iterator itMap2 = itMap1 ; 
      itMap2 !=MapEnd ; 
      ++itMap2)  
        {
          //LP calculate the chi square value
          double dummy = itMap1->second * itMap2->second;
          dummy *= inverror ;
          //LP fill the calib matrix
          m_kaliMatrix.at(itMap1->first*m_numberOfElements +itMap2->first) += dummy ;            
        } //LP second loop over xtals

      //LP calculate the vector value
      double dummy = pTk ;
      dummy -= pSubtract ;                        //LP borders
      dummy *= itMap1->second ;
      dummy *= inverror ;
      //LP fill the calib vector
      m_kaliVector.at(itMap1->first) += dummy ;
    } //LP first loop over energies
  return ;
}


//------------------------------------------------------------


void 
IMACalibBlock::complete ()
{
 int bef;
 int aft;
 for (unsigned int i=0; i<m_numberOfElements;++i)
  {
    for (unsigned int j=i+1; j< m_numberOfElements; ++j) 
      {
         bef = (i*m_numberOfElements+j);
         aft =  (j*m_numberOfElements +i);
            m_kaliMatrix.at(aft) = m_kaliMatrix.at(bef) ;
          } //LP second loop over xtals
      } //LP first loop over xtals

    return ;
  }


// ------------------------------------------------------------


int
IMACalibBlock::solve (int usingBlockSolver, double min, double max)
{
 complete () ;
// TH1F vettore ("vettore","vettore",10,-0.1,9.9);
// TH1F matrice ("matrice","matrice",100,-0.1,99.9);
 CLHEP::HepMatrix kaliMatrix (m_numberOfElements,m_numberOfElements) ;
// for (std::vector<double>::iterator it = m_kaliVector.begin();
//		 it!= m_kaliVector.end();++it)
//	 vettore.Fill(it-m_kaliVector.begin(),*it);
 riempiMtr (m_kaliMatrix , kaliMatrix) ;
// for (std::vector<double>::iterator it = m_kaliMatrix.begin();
//		 it!= m_kaliMatrix.end();++it)
//	 matrice.Fill(it-m_kaliMatrix.begin(),*it);
//  TFile f ("fileInteressante.root","RECREATE");
//  vettore.Write();
// matrice.Write();
 CLHEP::HepVector kaliVector (m_numberOfElements) ;
 riempiVtr (m_kaliVector , kaliVector) ;
 //PG linear system solution
 CLHEP::HepVector result = CLHEP::solve (kaliMatrix,kaliVector) ;
  for (int i = 0 ; i < kaliVector.num_row () ; ++i)
 if (result.normsq () < min * kaliVector.num_row () ||
     result.normsq () > max * kaliVector.num_row ()) 
   {
   if (usingBlockSolver)  
     {
        edm::LogWarning ("IML") << "using  blocSlover " << std::endl ;
        BlockSolver() (kaliMatrix,kaliVector,result) ;
     }
   else 
     {
       edm::LogWarning ("IML") <<"coeff out of range " <<std::endl;
       for (int i = 0 ; i < kaliVector.num_row () ; ++i)
             result[i] = 1. ;
     }
   }
 fillMap(result);
 return 0;
}


//------------------------------------------------------------


inline int 
IMACalibBlock::evalX2Size ()
  {

    return m_numberOfElements* m_numberOfElements;
  }


// ------------------------------------------------------------


void
IMACalibBlock::riempiMtr (const std::vector<double> & piena, 
                          CLHEP::HepMatrix & vuota) 
  {
    unsigned int max = m_numberOfElements ;

    assert (piena.size () == max * max) ; 
    assert (vuota.num_row () == int(max)) ;
    assert (vuota.num_col () == int(max)) ;
    for (unsigned int i = 0 ; i < max ; ++i)
     for (unsigned int j = 0 ; j < max ; ++j)
       if (edm::isNotFinite (piena[i*max + j])) vuota[i][j] = 0. ;
         else vuota[i][j] = piena[i*max + j] ; 

    return ;
  }


// ------------------------------------------------------------

void
IMACalibBlock::riempiVtr (const std::vector<double> & pieno, 
                          CLHEP::HepVector & vuoto) 
  {

    int max = m_numberOfElements ;
    assert (vuoto.num_row () == max) ;
    for (int i = 0 ; i < max ; ++i)
      if (edm::isNotFinite (pieno[i])) vuoto[i] = 0. ;
      else vuoto[i] = pieno[i] ; 

    return ;
  }


// ------------------------------------------------------------


void 
IMACalibBlock::reset () 
{

  for (std::vector<double>::iterator vecIt = m_kaliVector.begin () ;
       vecIt != m_kaliVector.end ();
       ++vecIt)
    {
      *vecIt = 0. ;
    }  

  for (std::vector<double>::iterator vecIt = m_kaliMatrix.begin () ;
       vecIt != m_kaliMatrix.end ();
       ++vecIt)
    {
      *vecIt = 0. ;
    }  

}


// ------------------------------------------------------------
//LP sta provando a fare i seguenti due metodi come locali. Speriamo di non fare stronzate.


void 
IMACalibBlock::fillMap (const CLHEP::HepVector & result) 
{

   for (unsigned int i=0; i < m_numberOfElements; ++i)
      {
         m_coefficients[i] = result[i] ;
      } 

   return ;
}

