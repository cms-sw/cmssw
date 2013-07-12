/* 
    $Date: 2008/01/23 10:59:54 $
    $Revision: 1.1.2.1 $
    $Id: smartSelector.h,v 1.1.2.1 2008/01/23 10:59:54 govoni Exp $ 
    $Author: govoni $
*/

#ifndef smartSelector_h
#define smartSelector_h

#include <iostream>

class smartSelector
{
  public :
  
    //! ctor
    smartSelector (int smallestPart = 0) : 
      m_smallestPart (smallestPart) {}

    //! set the smallest part only if was set to 0
    int setSmallestPart (int smallestPart)
      {
        if (!m_smallestPart) 
          {
            m_smallestPart = smallestPart ;
            return 0 ;
          }
        else return 1 ;          
      }

    //! actual selector
    int accept (int eventNb, const int numberOfFractions) const
      {      
        if (!m_smallestPart) return 1 ;
	if (m_smallestPart == numberOfFractions) return 1 ;
        int position = eventNb % m_smallestPart ;
        int sum = 0 ; 
        for (int i=1 ; i<numberOfFractions ; i *= 2) sum += i ;
//        std::cout << "debug f_" << numberOfFractions
//                  << "  " << sum  
//                  << "  " << sum+numberOfFractions << "\n" ;
        if (position >= sum &&
            position < sum + numberOfFractions)
          return 1 ;  
        return 0 ;
      }
    
  private :

    int m_smallestPart ;

} ;


#endif
