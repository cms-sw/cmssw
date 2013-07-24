/** 
    $Date: 2008/02/25 17:41:51 $
    $Revision: 1.2 $
    $Id: CalibCoeff.cc,v 1.2 2008/02/25 17:41:51 malberti Exp $ 
    \author $Author: malberti $
*/

#include "Calibration/Tools/interface/CalibCoeff.h"
#include <cmath>

CalibCoeff::CalibCoeff (const double & value , 
                        const bool & isGood) :
  m_value (value),
  m_isGood (isGood),
  m_difference (m_value)
  {}


// ------------------------------------------------------------


CalibCoeff::~CalibCoeff () {}


// ------------------------------------------------------------


double 
CalibCoeff::value () const 
{
  return m_isGood * m_value + !m_isGood ;
}


// ------------------------------------------------------------


double
CalibCoeff::difference () const
{
  return m_isGood * m_difference - !m_isGood ;
}


// ------------------------------------------------------------


bool 
CalibCoeff::status () const 
{
  return m_isGood ;
}


// ------------------------------------------------------------


void 
CalibCoeff::setValue (const double & val)  
{
  m_value = val ;
  m_difference = m_value ;
  m_isGood = true ;
  return ;
}


// ------------------------------------------------------------


void 
CalibCoeff::setStatus (const bool & stat)  
{
  m_isGood = stat ;
  return ;
}


// ------------------------------------------------------------


double 
CalibCoeff::operator *= (const double & var) 
{
  double oldval = m_value ;
  m_value *= var ;
  m_difference = fabs (m_value - oldval) ;
  m_isGood = true ;
  return m_value ;
}

