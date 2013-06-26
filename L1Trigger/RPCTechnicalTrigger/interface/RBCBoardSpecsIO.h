// $Id: RBCBoardSpecsIO.h,v 1.2 2009/09/26 13:39:16 elmer Exp $
#ifndef CONFIGCODE_RBCBOARDSPECSIO_H 
#define CONFIGCODE_RBCBOARDSPECSIO_H 1

// Include files
#include "CondFormats/RPCObjects/interface/RBCBoardSpecs.h"

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <ios>

/** @class RBCBoardSpecsIO RBCBoardSpecsIO.h ConfigCode/RBCBoardSpecsIO.h
 *  
 *
 *  @author Andres Osorio
 *  @date   2008-12-16
 */
class RBCBoardSpecsIO : public RBCBoardSpecs {
public: 
  /// Standard constructor
  RBCBoardSpecsIO( );

  virtual ~RBCBoardSpecsIO( ); ///< Destructor
  
  friend std::istream& operator>>(std::istream &, RBCBoardConfig &);
  
protected:
  
private:
  
};

std::istream& operator>>(std::istream &, RBCBoardSpecsIO::RBCBoardConfig &);

#endif // CONFIGCODE_RBCBOARDSPECSIO_H
