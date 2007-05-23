// File and Version Information:
//      $Id: SprAbsTwoClassCriterion.hh,v 1.2 2006/10/19 21:27:52 narsky Exp $
//
// Description:
//      Class SprAbsTwoClassCriterion :
//          Criterion for two-class figure-of-merit, for example:
//            - S/sqrt(S+B)
//            - fraction of correctly classified events
//            - etc.
//          Larger values of FOM must correspond to a higher (better)
//          level of signal optimization. FOM must be between min() and max().
//
// Environment:
//      Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2005              California Institute of Technology
//
//------------------------------------------------------------------------
 
#ifndef _SprAbsTwoClassCriterion_HH
#define _SprAbsTwoClassCriterion_HH


class SprAbsTwoClassCriterion
{
public:
  virtual ~SprAbsTwoClassCriterion() {}

  SprAbsTwoClassCriterion() {}

  /*
    Return figure of merit.
    wcor0 - correctly classified weighted fraction of background
    wmis0 - misclassified weighted fraction of background
    wcor1 - correctly classified weighted fraction of signal
    wmis1 - misclassified weighted fraction of signal
  */
  virtual double fom(double wcor0, double wmis0, 
		     double wcor1, double wmis1) const = 0;

  /*
    Symmetric FOM or not? Returns true if fom(a,b,c,d)=fom(b,a,d,c).
  */
  virtual bool symmetric() const = 0;

  /*
    Return minimal and maximal values of acceptable FOM.
  */
  virtual double min() const = 0;
  virtual double max() const = 0;

  /*
    Returns derivatives.
  */
  virtual double dfom_dwmis0(double wcor0, double wmis0, 
			     double wcor1, double wmis1) const = 0;
  virtual double dfom_dwcor1(double wcor0, double wmis0, 
			     double wcor1, double wmis1) const = 0;
};

#endif
