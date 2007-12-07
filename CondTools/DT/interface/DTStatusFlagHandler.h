#ifndef DTStatusFlagHandler_H
#define DTStatusFlagHandler_H
/** \class DTStatusFlagHandler
 *
 *  Description: 
 *
 *
 *  $Date: 2007/11/24 12:29:52 $
 *  $Revision: 1.1.2.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "CondCore/PopCon/interface/PopConSourceHandler.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
class DTStatusFlag;

//---------------
// C++ Headers --
//---------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTStatusFlagHandler: public popcon::PopConSourceHandler<DTStatusFlag> {

 public:

  /** Constructor
   */
  DTStatusFlagHandler( std::string name,
                       std::string connect_string,
                       const edm::Event& evt,
                       const edm::EventSetup& est,
                       const std::string& tag,
                       const std::string& file );
  void getNewObjects();

  /** Destructor
   */
  virtual ~DTStatusFlagHandler();

  /** Operations
   */
  /// 

 private:

  std::string dataTag;
  std::string fileName;

};


#endif // DTStatusFlagHandler_H






