/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kaspar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/RPAlignmentCorrectionsDataSequence.h"
#include "CondFormats/AlignmentRecord/interface/RPMeasuredAlignmentRecord.h"
#include "CondFormats/AlignmentRecord/interface/RPRealAlignmentRecord.h"
#include "CondFormats/AlignmentRecord/interface/RPMisalignedAlignmentRecord.h"

#include <vector>
#include <string>
#include <map>
#include <set>


/**
 * \ingroup TotemRPGeometry
 * \brief A class adding (mis)alignments to geometry (both real and misaligned).
 *
 * See schema of \ref TotemRPGeometry "TOTEM RP geometry classes"
 **/
class  TotemRPIncludeAlignments : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder
{
  public:
    TotemRPIncludeAlignments(const edm::ParameterSet &p);
    virtual ~TotemRPIncludeAlignments(); 

    std::unique_ptr<RPAlignmentCorrectionsData> produceMeasured(const RPMeasuredAlignmentRecord &);
    std::unique_ptr<RPAlignmentCorrectionsData> produceReal(const RPRealAlignmentRecord &);
    std::unique_ptr<RPAlignmentCorrectionsData> produceMisaligned(const RPMisalignedAlignmentRecord &);

  protected:
    unsigned int verbosity;
    RPAlignmentCorrectionsDataSequence acsMeasured, acsReal, acsMisaligned;
    RPAlignmentCorrectionsData acMeasured, acReal, acMisaligned;

    virtual void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue&, edm::ValidityInterval&);

    /// merges an array of sequences to one
    RPAlignmentCorrectionsDataSequence Merge(const std::vector<RPAlignmentCorrectionsDataSequence>) const;

    /// builds a sequence of corrections from provided sources and runs a few checks
    void PrepareSequence(const std::string &label, RPAlignmentCorrectionsDataSequence &seq, const std::vector<std::string> &files) const;
};

//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

TotemRPIncludeAlignments::TotemRPIncludeAlignments(const edm::ParameterSet &pSet) :
  verbosity(pSet.getUntrackedParameter<unsigned int>("verbosity", 1))
{
  PrepareSequence("Measured", acsMeasured, pSet.getParameter< vector<string> >("MeasuredFiles"));
  PrepareSequence("Real", acsReal, pSet.getParameter< vector<string> >("RealFiles"));
  PrepareSequence("Misaligned", acsMisaligned, pSet.getParameter< vector<string> >("MisalignedFiles"));

  setWhatProduced(this, &TotemRPIncludeAlignments::produceMeasured);
  setWhatProduced(this, &TotemRPIncludeAlignments::produceReal);
  setWhatProduced(this, &TotemRPIncludeAlignments::produceMisaligned);
  
  findingRecord<RPMeasuredAlignmentRecord>();
  findingRecord<RPRealAlignmentRecord>();
  findingRecord<RPMisalignedAlignmentRecord>();
}

//----------------------------------------------------------------------------------------------------

TotemRPIncludeAlignments::~TotemRPIncludeAlignments()
{
}

//----------------------------------------------------------------------------------------------------

RPAlignmentCorrectionsDataSequence TotemRPIncludeAlignments::Merge(const vector<RPAlignmentCorrectionsDataSequence> files) const
{
  // find interval boundaries
  map< TimeValue_t, vector< pair<bool, const RPAlignmentCorrectionsData*> > > bounds;

  for (vector<RPAlignmentCorrectionsDataSequence>::const_iterator fit = files.begin(); fit != files.end(); ++fit)
  {
    for (RPAlignmentCorrectionsDataSequence::const_iterator iit = fit->begin(); iit != fit->end(); ++iit)
    {
      const TimeValidityInterval &tvi = iit->first;
      const RPAlignmentCorrectionsData *corr = & iit->second;

      bounds[tvi.first].push_back( pair<bool, const RPAlignmentCorrectionsData*>(true, corr) );

      TimeValue_t delta = (tvi.last != TimeValidityInterval::EndOfTime()) ? (1ULL << 32) : 0;  // input resolution is 1s
      bounds[tvi.last + delta].push_back( pair<bool, const RPAlignmentCorrectionsData*>(false, corr) );
    }
  }
  
  // build correction sums per interval
  set<const RPAlignmentCorrectionsData*> accumulator;
  RPAlignmentCorrectionsDataSequence result;
  //  bool gap_found = false;
  for (map< TimeValue_t, vector< pair<bool, const RPAlignmentCorrectionsData*> > >::const_iterator tit = bounds.begin(); tit != bounds.end(); ++tit)
  {
    for (vector< pair<bool, const RPAlignmentCorrectionsData*> >::const_iterator cit = tit->second.begin(); cit != tit->second.end(); ++cit)
    {
      bool add = cit->first;
      const RPAlignmentCorrectionsData *corr = cit->second;

      if (add)
        accumulator.insert(corr);
      else 
        accumulator.erase(corr);
    }
    
    map< TimeValue_t, vector< pair<bool, const RPAlignmentCorrectionsData*> > >::const_iterator tit_next = tit;
    tit_next++;
    if (tit_next == bounds.end())
      break;

    TimeValue_t delta = (tit_next->first != TimeValidityInterval::EndOfTime()) ? 1 : 0; // minimal step
    TimeValidityInterval tvi(tit->first, tit_next->first - delta);

    if (verbosity)
    {
      printf("\tfirst=%10s, last=%10s: alignment blocks=%li\n",
        TimeValidityInterval::ValueToUNIXString(tvi.first).c_str(),
        TimeValidityInterval::ValueToUNIXString(tvi.last).c_str(),
        accumulator.size()
      );
    }

    for (set<const RPAlignmentCorrectionsData*>::iterator sit = accumulator.begin(); sit != accumulator.end(); ++sit)
      result[tvi].AddCorrections(*(*sit));
  }

  return result;
}

//----------------------------------------------------------------------------------------------------

void TotemRPIncludeAlignments::PrepareSequence(const string &label, RPAlignmentCorrectionsDataSequence &seq, const vector<string> &files) const
{
  if (verbosity)
    printf(">> TotemRPIncludeAlignments::PrepareSequence(%s)\n", label.c_str());

  vector<RPAlignmentCorrectionsDataSequence> sequences;
  for (unsigned int i = 0; i < files.size(); i++)
    sequences.push_back(RPAlignmentCorrectionsDataSequence(files[i]));

  seq = Merge(sequences);
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<RPAlignmentCorrectionsData> TotemRPIncludeAlignments::produceMeasured(const RPMeasuredAlignmentRecord &iRecord)
{
  return std::make_unique<RPAlignmentCorrectionsData>(acMeasured);
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<RPAlignmentCorrectionsData> TotemRPIncludeAlignments::produceReal(const RPRealAlignmentRecord &iRecord)
{
  return std::make_unique<RPAlignmentCorrectionsData>(acReal);
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<RPAlignmentCorrectionsData> TotemRPIncludeAlignments::produceMisaligned(const RPMisalignedAlignmentRecord &iRecord)
{
  return std::make_unique<RPAlignmentCorrectionsData>(acMisaligned);
}

//----------------------------------------------------------------------------------------------------

void TotemRPIncludeAlignments::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &key,
    const IOVSyncValue& iosv, ValidityInterval& valInt) 
{
  if (verbosity)
  {
    LogVerbatim("TotemRPIncludeAlignments")
      << ">> TotemRPIncludeAlignments::setIntervalFor(" << key.name() << ")";

    time_t unixTime = iosv.time().unixTime();
    char timeStr[50];
    strftime(timeStr, 50, "%F %T", localtime(&unixTime));

    LogVerbatim("TotemRPIncludeAlignments")
      << "    run=" << iosv.eventID().run() << ", event=" << iosv.eventID().event() << ", UNIX timestamp=" << unixTime << " (" << timeStr << ")";
  }

  // determine what sequence and corrections should be used
  RPAlignmentCorrectionsDataSequence *seq = NULL;
  RPAlignmentCorrectionsData *corr = NULL;

  if (strcmp(key.name(), "RPMeasuredAlignmentRecord") == 0)
  {
    seq = &acsMeasured;
    corr = &acMeasured;
  }

  if (strcmp(key.name(), "RPRealAlignmentRecord") == 0)
  {
    seq = &acsReal;
    corr = &acReal;
  }

  if (strcmp(key.name(), "RPMisalignedAlignmentRecord") == 0)
  {
    seq = &acsMisaligned;
    corr = &acMisaligned;
  }

  if (seq == NULL)
    throw cms::Exception("TotemRPIncludeAlignments::setIntervalFor") << "Unknown record " << key.name();

  // find the corresponding time interval
  bool next_exists = false;
  TimeValue_t t = iosv.time().value(), next_start = TimeValidityInterval::EndOfTime();

  for (RPAlignmentCorrectionsDataSequence::iterator it = seq->begin(); it != seq->end(); ++it)
  {
    if (it->first.first <= t && it->first.last >= t)
    {
      valInt = ValidityInterval(IOVSyncValue(Timestamp(it->first.first)), IOVSyncValue(Timestamp(it->first.last)));
      *corr = it->second;

      if (verbosity)
      {
        LogVerbatim("TotemRPIncludeAlignments")
          << "    setting validity interval [" << TimeValidityInterval::ValueToUNIXString(valInt.first().time().value())
          << ", " << TimeValidityInterval::ValueToUNIXString(valInt.last().time().value()) << "]";
      }

      return;
    }

    if (t <= it->first.first)
    {
      next_exists = true;
      next_start = min(next_start, it->first.first);
    }
  }
    
  // no interval found, set empty corrections
  *corr = RPAlignmentCorrectionsData();

  if (!next_exists)
    valInt = ValidityInterval(iosv, iosv.endOfTime());
  else 
    valInt = ValidityInterval(iosv, IOVSyncValue(Timestamp(next_start - 1)));
  
  if (verbosity)
  {
    LogVerbatim("TotemRPIncludeAlignments")
      << "    setting validity interval [" << TimeValidityInterval::ValueToUNIXString(valInt.first().time().value())
      << ", " << TimeValidityInterval::ValueToUNIXString(valInt.last().time().value()) << "]";
  }
}

DEFINE_FWK_EVENTSETUP_SOURCE(TotemRPIncludeAlignments);
