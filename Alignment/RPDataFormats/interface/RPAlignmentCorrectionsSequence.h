/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors:
 *	Jan Kašpar (jan.kaspar@gmail.com)
 *
 ****************************************************************************/

#ifndef Alignment_RPDataFormats_RPAlignmentCorrectionsSequence
#define Alignment_RPDataFormats_RPAlignmentCorrectionsSequence

#include <map>
#include <string>

#include "DataFormats/Provenance/interface/Timestamp.h"

#include "Alignment/RPDataFormats/interface/RPAlignmentCorrections.h"

/**
 *\brief Validity interval in timestamps.
 */
struct TimeValidityInterval
{
    /// the boundaries (included) of the interval expressed as UNIX timestamps
    edm::TimeValue_t first, last;

    TimeValidityInterval(edm::TimeValue_t _f = 0, edm::TimeValue_t _l = 0) : first(_f), last(_l) {}

    static const edm::TimeValue_t BeginOfTime()
    {
      return edm::Timestamp::beginOfTime().value();
    }

    static const edm::TimeValue_t EndOfTime()
    {
      return edm::Timestamp::endOfTime().value();
    }
    
    void SetInfinite()
    {
      first = BeginOfTime();
      last = EndOfTime();
    }

    static std::string ValueToUNIXString(const edm::TimeValue_t &v)
    {
      if (v == BeginOfTime())
        return "-inf";

      if (v == EndOfTime())
        return "+inf";

      // see: src/framework/DataFormats/Provenance/interface/Timestamp.h
      char buf[50];
      sprintf(buf, "%llu", v >> 32);
      return buf;
    }
    
    static edm::TimeValue_t UNIXStringToValue(const std::string &s)
    {
      if (s.compare("-inf") == 0)
        return BeginOfTime();

      if (s.compare("+inf") == 0)
        return EndOfTime();

      // see: src/framework/DataFormats/Provenance/interface/Timestamp.h
      edm::TimeValue_t v = atoi(s.c_str());
      return v << 32;
    }

    bool operator< (const TimeValidityInterval &o) const
    {
        if (first < o.first)
          return true;
        if (first > o.first)
          return false;
        if (last < o.last)
          return true;
        return false;
    }
};

//----------------------------------------------------------------------------------------------------

/**
 *\brief Time sequence of alignment corrections.
 */
class RPAlignmentCorrectionsSequence : public std::map<TimeValidityInterval, RPAlignmentCorrections>
{
  public:
    RPAlignmentCorrectionsSequence() {}

    RPAlignmentCorrectionsSequence(const std::string &fileName)
    {
      LoadXMLFile(fileName);
    }
    
    /// inserts a set of corrections with validity interval [first, last]
    void Insert(edm::TimeValue_t first, edm::TimeValue_t last, const RPAlignmentCorrections& corr)
    {
      insert(std::pair<TimeValidityInterval, RPAlignmentCorrections>(TimeValidityInterval(first, last), corr));
    }

    /// loads data from an alignment file
    void LoadXMLFile(const std::string &fileName);

    /// saves data to an alignment file
    void WriteXMLFile(const std::string &fileName, bool precise=false, bool wrErrors=true,
      bool wrSh_r=true, bool wrSh_xy=true, bool wrSh_z=true, bool wrRot_z=true) const;

};

#endif

