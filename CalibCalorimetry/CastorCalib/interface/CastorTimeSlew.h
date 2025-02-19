#ifndef CALIBCALORIMETRY_CASTORCALIB_CASTORTIMESLEW_H
#define CALIBCALORIMETRY_CASTORCALIB_CASTORTIMESLEW_H 1


/** \class CastorTimeSlew
  * 
  * copy from HCAL (author: J. Mans)
  *
  * Provides pulse delay as a function of amplitude for three choices
  * of QIE bias setting.  The "Medium" setting is used in HB and HE,
  * while the "Slow" (and lower noise) setting is used in HO.  All
  * data taken from bench measurements of the QIE and plotted in
  * Physics TDR Vol 1.
  *
  * Not to be used for HF at this time (unlikely to have much effect, however)
  *
  */
class CastorTimeSlew {
public:
  enum BiasSetting { Slow=0, Medium=1, Fast=2 };

  /** \brief Returns the amount (ns) by which a pulse of the given
   number of fC will be delayed by the timeslew effect, for the
   specified bias setting. */
  static double delay(double fC, BiasSetting bias=Medium);
};

#endif
