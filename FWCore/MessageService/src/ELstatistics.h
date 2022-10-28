#ifndef FWCore_MessageService_ELstatistics_h
#define FWCore_MessageService_ELstatistics_h

// ----------------------------------------------------------------------
//
// ELstatistics	is a subclass of ELdestination representing the
//		provided statistics (for summary) keeping.
//
// 7/8/98 mf	Created file.
// 7/2/99 jvr   Added noTerminationSummary() function
// 12/20/99 mf  Added virtual destructor.
// 6/7/00 web	Reflect consolidation of ELdestination/X; consolidate
//		ELstatistics/X.
// 6/14/00 web	Declare classes before granting friendship.
// 10/4/00 mf   Add filterModule() and excludeModule()
// 1/15/00 mf   line length control: changed ELoutputLineLen to
//              the base class lineLen (no longer static const)
// 3/13/01 mf	statisticsMap()
//  4/4/01 mf   Removed moduleOfInterest and moduleToExclude, in favor
//              of using base class method.
// 1/17/06 mf	summary() for use in MessageLogger
// 8/16/07 mf	noteGroupedCategory(cat) to support grouping of modules in
//		specified categories.  Also, a static vector of such categories.
// 6/19/08 mf	summaryForJobReport() for use in CMS framework
//
// ----------------------------------------------------------------------

#include "FWCore/MessageLogger/interface/ELextendedID.h"
#include "FWCore/MessageLogger/interface/ELmap.h"
#include "FWCore/MessageService/src/ELdestination.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <set>

namespace edm {

  // ----------------------------------------------------------------------
  // prerequisite classes:
  // ----------------------------------------------------------------------

  class ErrorObj;
  namespace service {
    class ELadministrator;

    // ----------------------------------------------------------------------
    // ELstatistics:
    // ----------------------------------------------------------------------

    class ELstatistics : public ELdestination {
      friend class ELadministrator;

    public:
      // -----  constructor/destructor:
      ELstatistics();
      ELstatistics(std::ostream& osp);
      ELstatistics(int spaceLimit);
      ELstatistics(int spaceLimit, std::ostream& osp);
      ELstatistics(const ELstatistics& orig);
      ELstatistics& operator=(const ELstatistics& orig) = delete;  // verboten
      ~ELstatistics() override;

      // -----  Methods invoked by the ELadministrator:
      //
    public:
      // Used by attach() to put the destination on the ELadministrators list
      //-| There is a note in Design Notes about semantics
      //-| of copying a destination onto the list:  ofstream
      //-| ownership is passed to the new copy.

      bool log(const edm::ErrorObj& msg) override;

      // ----- Methods invoked by the MessageLoggerScribe, bypassing destControl
      //
    public:
      static void noteGroupedCategory(std::string const& cat);  // 8/16/07 mf

      void summary(unsigned long overfullWaitCount);
      void noTerminationSummary();
      void summaryForJobReport(std::map<std::string, double>& sm);
      void wipe() override;

    protected:
      void clearSummary();

      void zero() override;

      std::map<ELextendedID, StatsCount> statisticsMap() const;

    protected:
      int tableLimit;
      ELmap_stats stats;
      bool updatedStats;
      std::ostream& termStream;

      bool printAtTermination;

      CMS_THREAD_SAFE static std::set<std::string> groupedCategories;  // 8/16/07 mf
      static std::string formSummary(ELmap_stats& stats);              // 8/16/07 mf

      // ----  Helper methods specific to MessageLogger applicaton
      //
    private:
      std::string dualLogName(std::string const& s);

      void summary(std::ostream& os, std::string_view title);

    };  // ELstatistics

    // ----------------------------------------------------------------------

  }  // end of namespace service
}  // end of namespace edm

#endif  // FWCore_MessageService_ELstatistics_h
