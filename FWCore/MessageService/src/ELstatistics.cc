//  ---------------------------------------------------------------------
//
// ELstatistics.cc
//
// History:
//   7/8/98     mf      Created
//   7/2/99     jv      Added noTerminationSummary() function
//   6/7/00     web     Reflect consolidation of ELdestination/X;
//                      consolidate ELstatistics/X
//   6/14/00    web     Remove GNU relic code
//   6/15/00    web     using -> USING
//   10/4/00    mf      filterModule() and excludeModule()
//   3/13/00    mf      statisticsMap()
//    4/4/01    mf      Simplify filter/exclude logic by useing base class
//                      method thisShouldBeIgnored().  Eliminate
//                      moduleOfinterest and moduleToexclude.
//  11/01/01    web     Remove last vestige of GNU relic code; reordered
//                      initializers to correspond to order of member
//                      declarations
//   1/17/06    mf	summary() for use in MessageLogger
//   8/16/07    mf	Changes to implement grouping of modules in specified 
//			categories
//
//  ---------------------------------------------------------------------


#include "FWCore/MessageService/interface/ELstatistics.h"
#include "FWCore/MessageService/interface/ELadministrator.h"
#include "FWCore/MessageService/interface/ELcontextSupplier.h"

#include "FWCore/MessageLogger/interface/ErrorObj.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <ios>


// Possible Traces:
// #define ELstatisticsCONSTRUCTOR_TRACE
// #define ELstatsLOG_TRACE


namespace edm {
namespace service {



// ----------------------------------------------------------------------
// Constructors
// ----------------------------------------------------------------------


ELstatistics::ELstatistics()
: ELdestination     (           )
, tableLimit        ( -1        )
, stats             (           )
, updatedStats      ( false     )
, termStream        ( std::cerr )
, printAtTermination( true      )
{

  #ifdef ELstatisticsCONSTRUCTOR_TRACE
    std::cerr << "Constructor for ELstatistics()\n";
  #endif

}  // ELstatistics()


ELstatistics::ELstatistics( std::ostream & osp )
: ELdestination     (       )
, tableLimit        ( -1    )
, stats             (       )
, updatedStats      ( false )
, termStream        ( osp   )
, printAtTermination( true  )
{

  #ifdef ELstatisticsCONSTRUCTOR_TRACE
    std::cerr << "Constructor for ELstatistics(osp)\n";
  #endif

}  // ELstatistics()


ELstatistics::ELstatistics( int spaceLimit )
: ELdestination     (            )
, tableLimit        ( spaceLimit )
, stats             (            )
, updatedStats      ( false      )
, termStream        ( std::cerr  )
, printAtTermination( true       )
{

  #ifdef ELstatisticsCONSTRUCTOR_TRACE
    std::cerr << "Constructor for ELstatistics(spaceLimit)\n";
  #endif

}  // ELstatistics()


ELstatistics::ELstatistics( int spaceLimit, std::ostream & osp )
: ELdestination     (            )
, tableLimit        ( spaceLimit )
, stats             (            )
, updatedStats      ( false      )
, termStream        ( osp        )
, printAtTermination( true       )
{

  #ifdef ELstatisticsCONSTRUCTOR_TRACE
    std::cerr << "Constructor for ELstatistics(spaceLimit,osp)\n";
  #endif

}  // ELstatistics()


ELstatistics::ELstatistics( const ELstatistics & orig)
: ELdestination     (                         )
, tableLimit        ( orig.tableLimit         )
, stats             ( orig.stats              )
, updatedStats      ( orig.updatedStats       )
, termStream        ( orig.termStream         )
, printAtTermination( orig.printAtTermination )
{

  #ifdef ELstatisticsCONSTRUCTOR_TRACE
    std::cerr << "Copy constructor for ELstatistics()\n";
  #endif

  ignoreMostModules    = orig.ignoreMostModules;
  respondToThese       = orig.respondToThese;
  respondToMostModules = orig.respondToMostModules;
  ignoreThese          = orig.ignoreThese;

}  // ELstatistics()


ELstatistics::~ELstatistics()  {

  #ifdef ELstatisticsCONSTRUCTOR_TRACE
    std::cerr << "Destructor for ELstatistics\n";
  #endif

  if ( updatedStats && printAtTermination )
    summary( termStream, "Termination Summary" );

}  // ~ELstatistics()


// ----------------------------------------------------------------------
// Methods invoked by the ELadministrator
// ----------------------------------------------------------------------

ELstatistics *
ELstatistics::clone() const  {

  return  new ELstatistics( *this );

}  // clone()


bool  ELstatistics::log( const edm::ErrorObj & msg )  {

  #ifdef ELstatsLOG_TRACE
    std::cerr << "  =:=:=: Log to an ELstatistics\n";
  #endif

  // See if this message is to be counted.

  if ( msg.xid().severity < threshold )        return false;
  if ( thisShouldBeIgnored(msg.xid().module) ) return false;

  // Account for this message, making a new table entry if needed:
  //
  ELmap_stats::iterator s = stats.find( msg.xid() );
  if ( s == stats.end() )  {
    if ( tableLimit < 0  ||  static_cast<int>(stats.size()) < tableLimit )  {
      stats[msg.xid()] = StatsCount();
      s = stats.find( msg.xid() );
    }
  }
  #ifdef ELstatsLOG_TRACE
    std::cerr << "    =:=:=: Message accounted for in stats \n";
  #endif
  if ( s != stats.end() )  {
    #ifdef ELstatsLOG_TRACE
        std::cerr << "    =:=:=: Message not last stats \n";
        std::cerr << "    =:=:=: getContextSupplier \n";
        const ELcontextSupplier & csup
          = ELadministrator::instance()->getContextSupplier();
        std::cerr << "    =:=:=: getContextSupplier \n";
        ELstring sumcon;
        std::cerr << "    =:=:=: summaryContext \n";
        sumcon = csup.summaryContext();
        std::cerr << "    =:=:=: summaryContext is: " << sumcon << "\n";
        (*s).second.add( sumcon, msg.reactedTo() );
        std::cerr << "    =:=:=: add worked. \n";
    #else
        (*s).second.add( ELadministrator::instance()->
                    getContextSupplier().summaryContext(), msg.reactedTo() );
    #endif

    updatedStats = true;
    #ifdef ELstatsLOG_TRACE
      std::cerr << "    =:=:=: Updated stats \n";
    #endif
  }


  // For the purposes of telling whether any log destination has reacted
  // to the message, the statistics destination does not count:
  //

  #ifdef ELstatsLOG_TRACE
    std::cerr << "  =:=:=: log(msg) done (stats) \n";
  #endif

  return false;


}  // log()


// ----------------------------------------------------------------------
// Methods invoked through the ELdestControl handle
// ----------------------------------------------------------------------

void  ELstatistics::clearSummary()  {

  limits.zero();
  ELmap_stats::iterator s;
  for ( s = stats.begin();  s != stats.end();  ++s )  {
    (*s).second.n = 0;
    (*s).second.context1 = (*s).second.context2 = (*s).second.contextLast = "";
  }

}  // clearSummary()


void  ELstatistics::wipe()  {

  limits.wipe();
  stats.erase( stats.begin(), stats.end() );  //stats.clear();

}  // wipe()


void  ELstatistics::zero()  {

  limits.zero();

}  // zero()


static ELstring  formSummary( ELmap_stats & stats )  {

  using std::ios;       /* _base ? */
  using std::setw;
  using std::right;
  using std::left;

  std::ostringstream          s;
  ELmap_stats::const_iterator i;  // traverse messages
  int                         n;

  // -----  Summary part I:
  //
  ELstring  lastProcess( "" );
  bool      ftnote( false );

  struct part3  {
    long n, t;
    part3() : n(0L), t(0L)  { ; }
  }  p3[ELseverityLevel::nLevels];

  for ( i = stats.begin(), n = 0;  i != stats.end();  ++i )  {

    // -----  Emit new process and part I header, if needed:
    //
    if ( n == 0  || ! eq(lastProcess, (*i).first.process) ) {
      s << "\n";
      lastProcess = (*i).first.process;
      if ( lastProcess.size() > 0) {
        s << "Process " << (*i).first.process << '\n';
      }
      s << " type     category        sev    module        "
             "subroutine        count    total\n"
        << " ---- -------------------- -- ---------------- "
             "----------------  -----    -----\n"
        ;
    }
    // -----  Emit detailed message information:
    //
    s << right << setw( 5) << ++n                                     << ' '
      << left  << setw(20) << (*i).first.id.substr(0,20)              << ' '
      << left  << setw( 2) << (*i).first.severity.getSymbol()         << ' '
      << left  << setw(16) << (*i).first.module.substr(0,16)          << ' '
      << left  << setw(16) << (*i).first.subroutine.substr(0,16)
      << right << setw( 7) << (*i).second.n
      << left  << setw( 1) << ( (*i).second.ignoredFlag ? '*' : ' ' )
      << right << setw( 8) << (*i).second.aggregateN                  << '\n'
      ;
    ftnote = ftnote || (*i).second.ignoredFlag;

    // -----  Obtain information for Part III, below:
    //
    p3[(*i).first.severity.getLevel()].n += (*i).second.n;
    p3[(*i).first.severity.getLevel()].t += (*i).second.aggregateN;
  }  // for

  // -----  Provide footnote to part I, if needed:
  //
  if ( ftnote )
    s << "\n* Some occurrences of this message"
         " were suppressed in all logs, due to limits.\n"
      ;

  // -----  Summary part II:
  //
  for ( i = stats.begin(), n = 0;  i != stats.end();  ++i )  {
    if ( n ==  0 ) {
      s << '\n'
	<< " type    category    Examples: "
	   "run/evt        run/evt          run/evt\n"
	<< " ---- -------------------- ----"
	   "------------ ---------------- ----------------\n"
	;
    }
    s << right << setw( 5) << ++n                             << ' '
      << left  << setw(20) << (*i).first.id.c_str()           << ' '
      << left  << setw(16) << (*i).second.context1.c_str()    << ' '
      << left  << setw(16) << (*i).second.context2.c_str()    << ' '
                           << (*i).second.contextLast.c_str() << '\n'
      ;
  }  // for

  // -----  Summary part III:
  //
  s << "\nSeverity    # Occurrences   Total Occurrences\n"
    <<   "--------    -------------   -----------------\n";
  for ( int k = 0;  k < ELseverityLevel::nLevels;  ++k )  {
    if ( p3[k].n != 0  ||  p3[k].t != 0 )  {
      s << left  << setw( 8) << ELseverityLevel( ELseverityLevel::ELsev_(k) ).getName().c_str()
        << right << setw(17) << p3[k].n
        << right << setw(20) << p3[k].t
                             << '\n'
        ;
    }
  }  // for

  return s.str();

}  // formSummary()


void  ELstatistics::summary( ELdestControl & dest, const ELstring & title )  {

  dest.summarization( title, formSummary(stats) );
  updatedStats = false;

}  // summary()


void  ELstatistics::summary( std::ostream & os, const ELstring & title )  {

  os << title << std::endl << formSummary(stats) << std::flush;
  updatedStats = false;

}  // summary()

void  ELstatistics::summary( )  {

  termStream << "\n=============================================\n\n"
  	     << "MessageLogger Summary" << std::endl << formSummary(stats) 
             << std::flush;
  updatedStats = false;

}  // summary()


void  ELstatistics::summary( ELstring & s, const ELstring & title )  {

  s = title + '\n' + formSummary(stats);
  updatedStats = false;

}  // summary()


void  ELstatistics::noTerminationSummary()  { printAtTermination = false; }

std::map<ELextendedID , StatsCount> ELstatistics::statisticsMap() const {
  return std::map<ELextendedID , StatsCount> ( stats );
}

std::vector<std::string> ELstatistics::groupedCategories; // 8/16/07 mf 

void ELstatistics::noteGroupedCategory(std::string const & cat) {
  groupedCategories.push_back(cat);
}
  
} // end of namespace service  
} // end of namespace edm  
