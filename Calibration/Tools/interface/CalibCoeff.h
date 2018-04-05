#ifndef __CINT__
#ifndef CalibCoeff_H
#define CalibCoeff_H

/** \class CalibCoeff
 
    \brief intercalibration coefficient

*/
class CalibCoeff
{
  public :
    //! ctor
    CalibCoeff (const double & value = 1., 
                const bool & isGood = false) ;
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
