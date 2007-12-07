#ifndef DTRangeT0Handler_H
#define DTRangeT0Handler_H
/** \class DTRangeT0Handler
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
class DTRangeT0;

//---------------
// C++ Headers --
//---------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTRangeT0Handler: public popcon::PopConSourceHandler<DTRangeT0> {

 public:

  /** Constructor
   */
  DTRangeT0Handler( std::string name,
                    std::string connect_string,
                    const edm::Event& evt,
                    const edm::EventSetup& est,
                    const std::string& tag,
                    const std::string& file );
  void getNewObjects();

  /** Destructor
   */
  virtual ~DTRangeT0Handler();

  /** Operations
   */
  /// 

 private:

  std::string dataTag;
  std::string fileName;

};


#endif // DTRangeT0Handler_H






