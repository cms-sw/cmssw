// File and Version Information:
//      $Id: SprNNDefs.hh,v 1.1 2006/11/26 02:04:30 narsky Exp $
//
// Description:
//      Class SprNNDefs :
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2006              California Institute of Technology
//
//------------------------------------------------------------------------
 
#ifndef _SprNNDefs_HH
#define _SprNNDefs_HH

struct SprNNDefs
{
  enum NodeType { INPUT=1, HIDDEN, OUTPUT };
  enum ActFun   { ID=1, LOGISTIC }; 
  enum OutFun { OUTID=1 };
};

#endif
