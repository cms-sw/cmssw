#ifndef Alignment_CommonAlignment_Hierarchy_h
#define Alignment_CommonAlignment_Hierarchy_h

/** \class Hierarchy
 *
 *  Class to define the detector's hierarchy.
 *
 *  Also provides tree info for a sub-detector or an Alignable.
 *  Basically gives the structure names and positions of Alignables.
 *
 *  $Date: 2007/10/23 08:55:14 $
 *  $Revision: 1.2 $
 *  \author Chung Khim Lae
 */

#include "Alignment/CommonAlignment/interface/Counters.h"

class Hierarchy
{
  public:

  typedef std::pair<std::string, align::Counter> NameCounter;
	typedef std::vector<NameCounter> NameCounters;

  /// Init the detector and sub-detector IDs for an Alignable given its ID.
	/// Throw exception if ID is invalid.
  Hierarchy(
      			align::ID
            );

  /// Get ID of detector to which Alignable belongs.
  inline unsigned int det() const;

  /// Get ID of sub-detector to which Alignable belongs.
  inline unsigned int subdet() const;

  /// Name of sub-detector to which Alignable belongs.
	inline const std::string& subdetector() const;

  /// Get the NameCounters of a sub-detector to which Alignable belongs.
  inline const NameCounters& nameCounters() const;

  /// Name of sub-detector. Does not check if IDs are valid.
	inline static const std::string& subdetector(
                                               unsigned int det,
                                               unsigned int subdet
                                               );

  /// Get the NameCounters of a sub-detector. Does not check if IDs are valid.
  inline static const NameCounters& nameCounters(
                                                 unsigned int det,
                                                 unsigned int subdet
                                                 );

  private:

  /// Init structure names and counters.
  Hierarchy();

  static const unsigned int maxDetector    = 6;
  static const unsigned int maxSubdetector = 7;

	static std::string theDetectors[maxDetector];
  static std::vector<std::string> theSubdetectors[maxDetector];

  static NameCounters theNameCounters[maxDetector][maxSubdetector];

  unsigned int theDet;
  unsigned int theSubdet;
};

unsigned int Hierarchy::det() const
{
  return theDet;
}

unsigned int Hierarchy::subdet() const
{
  return theSubdet;
}

const std::string& Hierarchy::subdetector() const
{
  return theSubdetectors[theDet][theSubdet];
}

const Hierarchy::NameCounters& Hierarchy::nameCounters() const
{
  return theNameCounters[theDet][theSubdet];
}

const std::string& Hierarchy::subdetector(unsigned int det,
                                          unsigned int subdet)
{
  static Hierarchy hierarchy; // init names and counters in hierarchy

  return theSubdetectors[det][subdet];
}

const Hierarchy::NameCounters& Hierarchy::nameCounters(unsigned int det,
                                                       unsigned int subdet)
{
  static Hierarchy hierarchy; // init names and counters in hierarchy

  return theNameCounters[det][subdet];
}

#endif
