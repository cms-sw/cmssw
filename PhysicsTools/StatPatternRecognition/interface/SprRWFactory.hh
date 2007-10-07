//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprRWFactory.hh,v 1.1 2007/10/05 20:03:09 narsky Exp $
//
// Description:
//      Class SprRWFactory :
//         Makes writers and readers of the requested type.
//
// Environment:
//      Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2007              California Institute of Technology
//
//------------------------------------------------------------------------
 
#ifndef _SprRWFactory_HH
#define _SprRWFactory_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsWriter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsReader.hh"

#include "PhysicsTools/StatPatternRecognition/interface/SprAsciiWriter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprSimpleReader.hh"

#include "PhysicsTools/StatPatternRecognition/interface/SprRootWriter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprRootReader.hh"

#include <iostream>

class SprPreFilter;


struct SprRWFactory
{
  enum DataType { Ascii, Root };

  static SprAbsReader* makeReader(DataType dataType, 
				  int mode,
				  SprPreFilter* filter=0) {
    if(      dataType == Ascii )
      return new SprSimpleReader(mode,filter);
    else if( dataType == Root )
      return new SprRootReader(filter);
    else {
      std::cerr << "Unknown reader type requested." << std::endl;
      return 0;
    }
  }


  static SprAbsWriter* makeWriter(DataType dataType, const char* label) {
    if(      dataType == Ascii )
      return new SprAsciiWriter(label);
    else if( dataType == Root )
      return new SprRootWriter(label);
    else {
      std::cout << "Unknown writer type requested. " 
		<< "Returning Ascii writer by default." << std::endl;
      return new SprAsciiWriter(label);
    }
  }

};

#endif
