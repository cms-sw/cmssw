#ifndef DTDeadFlagHandler_H
#define DTDeadFlagHandler_H
/** \class DTDeadFlagHandler
 *
 *  Description: 
 *
 *
 *  $Date: 2007/11/24 12:29:51 $
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
class DTDeadFlag;

//---------------
// C++ Headers --
//---------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTDeadFlagHandler: public popcon::PopConSourceHandler<DTDeadFlag> {

 public:

  /** Constructor
   */
  DTDeadFlagHandler( std::string name,
                     std::string connect_string,
                     const edm::Event& evt,
                     const edm::EventSetup& est,
                     const std::string& tag,
                     const std::string& file );
  void getNewObjects();

  /** Destructor
   */
  virtual ~DTDeadFlagHandler();

  /** Operations
   */
  /// 

 private:

  std::string dataTag;
  std::string fileName;

};


#endif // DTDeadFlagHandler_H

