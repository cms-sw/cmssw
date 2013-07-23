#ifndef CALIBCALORIMETRY_HCALALGOS_HCALTIMESLEW_H
#define CALIBCALORIMETRY_HCALALGOS_HCALTIMESLEW_H 1


/** \class HcalTimeSlew
  * 
  * Provides pulse delay as a function of amplitude for three choices
  * of QIE bias setting.  The "Medium" setting is used in HB and HE,
  * while the "Slow" (and lower noise) setting is used in HO.  All
  * data taken from bench measurements of the QIE and plotted in
  * Physics TDR Vol 1.
  *
  * Not to be used for HF at this time (unlikely to have much effect, however)
  *
  * $Date: 2006/02/07 23:02:36 $
  * $Revision: 1.1 $
  * \author J. Mans - Minnesota
  */
class HcalTimeSlew {
public:
  enum BiasSetting { Slow=0, Medium=1, Fast=2 };

  /** \brief Returns the amount (ns) by which a pulse of the given
   number of fC will be delayed by the timeslew effect, for the
   specified bias setting. */
  static double delay(double fC, BiasSetting bias=Medium);
};

#endif
