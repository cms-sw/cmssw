#ifndef __CINT__
#ifndef CalibCoeff_H
#define CalibCoeff_H

/** \class CalibCoeff
 
    \brief intercalibration coefficient

    $Date: 2008/01/23 10:59:53 $
    $Revision: 1.1.2.1 $
    $Id: CalibCoeff.h,v 1.1.2.1 2008/01/23 10:59:53 govoni Exp $ 
    \author $Author: govoni $
*/
class CalibCoeff
{
  public :
    //! ctor
    CalibCoeff (const double & value = 1., 
                const bool & isGood = 0) ;
    //! dtor
    ~CalibCoeff () ;
    
    //! its value
    double value () const ;
    //! the abs difference wrt prev value
    double difference () const ;
    //! its status
    bool status () const ;
    //! set its value and turn into good the coefficient
    void setValue (const double & val) ;
    //! set its value and turn into good the coefficient
    void setStatus (const bool & stat) ;
    //! update the value and turn into good the coefficient
    double operator *= (const double & var) ; 
    
  private :  
  
    //! the actual value
    double m_value ;
    //! if it is good
    bool m_isGood ;
    //! the difference with the previous value
    double m_difference ;
    
} ;





#endif
#endif
