#ifndef DTDataBuffer_H
#define DTDataBuffer_H
/** \class DTDataBuffer
 *
 *  Description:
 *       Class to hold drift tubes T0s
 *
 *  $Date: 2007/10/30 17:30:20 $
 *  $Revision: 1.4.6.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "CondFormats/DTObjects/interface/DTBufferTree.h"

//---------------
// C++ Headers --
//---------------
#include <string>
#include <map>

//              ---------------------
//              -- Class Interface --
//              ---------------------

template <class Key, class Content>
class DTDataBuffer {

 public:

  /** Constructor
   */
  DTDataBuffer();

  /** Destructor
   */
  ~DTDataBuffer();

  /** Operations
   */
  /// access internal buffer
  static DTBufferTree<Key,Content>* openBuffer( const std::string& name );
  static DTBufferTree<Key,Content>* findBuffer( const std::string& name );
  static void                       dropBuffer( const std::string& name );

 private:

  static std::map<std::string,DTBufferTree<Key,Content>*> bufferMap;

};

#include "DTDataBuffer.icc"

#endif // DTDataBuffer_H

