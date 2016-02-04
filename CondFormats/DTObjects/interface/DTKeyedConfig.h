#ifndef DTKeyedConfig_H
#define DTKeyedConfig_H
/** \class DTKeyedConfig
 *
 *  Description: 
 *
 *
 *  $Date: 2010/05/14 11:42:55 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "CondFormats/Common/interface/BaseKeyed.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------


//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTKeyedConfig: public cond::BaseKeyed {

 public:

  /** Constructor
   */
  DTKeyedConfig();
  DTKeyedConfig( const DTKeyedConfig& obj );

  /** Destructor
   */
  virtual ~DTKeyedConfig();

  /** Operations
   */
  ///
  int getId() const;
  void setId( int id );
  void add( const std::string& data );
  void add( int id );

  typedef std::vector<std::string>::const_iterator data_iterator;
  typedef std::vector<        int>::const_iterator link_iterator;
  data_iterator dataBegin() const;
  data_iterator dataEnd() const;
  link_iterator linkBegin() const;
  link_iterator linkEnd() const;

 private:

  int cfgId;
  std::vector<std::string> dataList; 
  std::vector<int>         linkList; 

};


#endif // DTKeyedConfig_H






