#ifndef DataFormats_PatCandidates_UserData_h
#define DataFormats_PatCandidates_UserData_h

/** \class    pat::UserData UserData.h "DataFormats/PatCandidates/interface/UserData.h"
 *
 *  \brief    Base class for data that users can add to pat objects
 *
 *  
 *
 *  \author   Sal Rappoccio
 *
 *  \version  $Id: UserData.h,v 1.01 
 *
 */

#include <string>
#include <vector>


namespace pat {

  class UserData {
  public:
    UserData() {}
    virtual ~UserData() {}

    virtual UserData * clone() const { return new UserData(*this); }
    
  protected:
  };

  typedef std::vector<pat::UserData>   UserDataCollection;
}


#endif
