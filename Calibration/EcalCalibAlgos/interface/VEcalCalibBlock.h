#ifndef __CINT__
#ifndef VEcalCalibBlock_H
#define VEcalCalibBlock_H

#include <map>
#include <string>
#include <vector>

/** \class VEcalCalibBlock
 
    \brief element for the single ECAL block intercalibration  
*/
class VEcalCalibBlock
{
  public :
  
    //! ctor
    VEcalCalibBlock (int numberOfElements) :
      m_numberOfElements (numberOfElements) {} ;
    //! dtor
    virtual ~VEcalCalibBlock () {} ;
    
    //! insert an entry
    virtual void Fill (std::map<int,double>::const_iterator,
                       std::map<int,double>::const_iterator,
                       double pTk,
                       double pSubtract,
                       double sigma = 1.) = 0 ;

    //! reset the chi2 matrices
    virtual void reset () = 0 ;
    //! retrieve the coefficients
    float at (const unsigned int index) { return m_coefficients[index] ; }
    //! solve the chi2 linear system
    virtual int solve (int usingBlockSolver, double min, double max) = 0 ;

  protected :  

    //! The only parameter!
    unsigned int m_numberOfElements ;
    //! map of coefficients   
    std::map<unsigned int, float> m_coefficients ;
} ;


#endif
#endif

