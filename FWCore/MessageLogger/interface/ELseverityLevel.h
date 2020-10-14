#ifndef MessageLogger_ELseverityLevel_h
#define MessageLogger_ELseverityLevel_h

// ----------------------------------------------------------------------
//
// ELseverityLevel.h - declare objects that encode a message's urgency
//
//	Both frameworker and user will often pass one of the
//	instantiated severity levels to logger methods.
//
//	The only other methods of ELseverityLevel a frameworker
//	might use is to check the relative level of two severities
//	using operator< or the like.
//
// 30-Jun-1998 mf	Created file.
// 26-Aug-1998 WEB	Made ELseverityLevel object less weighty.
// 16-Jun-1999 mf	Added constructor from string.
// 23-Jun-1999 mf	Additional ELsev_noValueAssigned to allow constructor
//			from string to give ELunspecified when not found, while
//			still allowing finding zero severity.
// 23-Jun-1999 mf	Corrections for subtleties in initialization of
//			global symbols:
//				Added ELsevLevGlobals array
//				Changed extern consts of SLseverityLevels into
//				  const ELseverityLevel & 's
//				Inserted class ELinitializeGlobalSeverityObjects
//				  in place of the
//				  initializeGlobalSeverityObjects() function.
//				Changed globalSeverityObjectsGuarantor to an
//				  ELinitializeGlobalSeverityObjects instance.
// 30-Jun-1999 mf	Modifications to eliminate problems with order of
//                      globals initializations:
//				translate(), getInputStr(), getVarName()
// 12-Jun-2000 web	Final fix to global static initialization problem
// 14-Jun-2000 web	Declare classes before granting friendship.
// 27-Jun-2000 web	Fix order-of-static-destruction problem
//
// ----------------------------------------------------------------------

#include <string>
#include <string_view>

namespace edm {

  // ----------------------------------------------------------------------
  // ELseverityLevel:
  // ----------------------------------------------------------------------

  class ELseverityLevel {
  public:
    // ---  One ELseverityLevel is globally instantiated (see below)
    // ---  for each of the following levels:
    //
    enum ELsev_ {
      ELsev_noValueAssigned = 0  // default returned by map when not found
      ,
      ELsev_zeroSeverity  // threshold use only
      ,
      ELsev_success  // report reaching a milestone
      ,
      ELsev_info  // information
      ,
      ELsev_warning  // warning
      ,
      ELsev_error  // error detected
      ,
      ELsev_unspecified  // severity was not specified
      ,
      ELsev_severe  // future results are suspect
      ,
      ELsev_highestSeverity  // threshold use only
      // -----
      ,
      nLevels  // how many levels?
    };         // ELsev_

    // -----  Birth/death:
    //
    constexpr ELseverityLevel(ELsev_ lev = ELsev_unspecified) noexcept : myLevel(lev) {}
    ELseverityLevel(std::string_view str);
    // str may match getSymbol, getName, getInputStr,
    // or getVarName -- see accessors
    ~ELseverityLevel() noexcept = default;

    // -----  Comparator:
    //
    [[nodiscard]] constexpr int cmp(ELseverityLevel const& e) const noexcept { return myLevel - e.myLevel; }

    // -----  Accessors:
    //
    constexpr int getLevel() const noexcept { return myLevel; }
    const std::string& getSymbol() const;    // example: "-e"
    const std::string& getName() const;      // example: "Error"
    const std::string& getInputStr() const;  // example: "ERROR"
    const std::string& getVarName() const;   // example: "ELerror"

    // -----  Emitter:
    //
    friend std::ostream& operator<<(std::ostream& os, const ELseverityLevel& sev);

  private:
    // Data per ELseverityLevel object:
    //
    int myLevel;

  };  // ELseverityLevel

  // ----------------------------------------------------------------------
  // Declare the globally available severity objects,
  // one generator function and one proxy per non-default ELsev_:
  // ----------------------------------------------------------------------

  constexpr const ELseverityLevel ELzeroSeverity{ELseverityLevel::ELsev_zeroSeverity};

  constexpr const ELseverityLevel ELdebug{ELseverityLevel::ELsev_success};

  constexpr const ELseverityLevel ELinfo{ELseverityLevel::ELsev_info};

  constexpr const ELseverityLevel ELwarning{ELseverityLevel::ELsev_warning};

  constexpr const ELseverityLevel ELerror{ELseverityLevel::ELsev_error};

  constexpr const ELseverityLevel ELunspecified{ELseverityLevel::ELsev_unspecified};

  constexpr const ELseverityLevel ELsevere{ELseverityLevel::ELsev_severe};

  constexpr const ELseverityLevel ELhighestSeverity{ELseverityLevel::ELsev_highestSeverity};

  // ----------------------------------------------------------------------
  // Comparators:
  // ----------------------------------------------------------------------

  constexpr inline bool operator==(ELseverityLevel const& e1, ELseverityLevel const& e2) noexcept {
    return e1.cmp(e2) == 0;
  }
  constexpr inline bool operator!=(ELseverityLevel const& e1, ELseverityLevel const& e2) noexcept {
    return e1.cmp(e2) != 0;
  }
  constexpr inline bool operator<(ELseverityLevel const& e1, ELseverityLevel const& e2) noexcept {
    return e1.cmp(e2) < 0;
  }
  constexpr inline bool operator<=(ELseverityLevel const& e1, ELseverityLevel const& e2) noexcept {
    return e1.cmp(e2) <= 0;
  }
  constexpr inline bool operator>(ELseverityLevel const& e1, ELseverityLevel const& e2) noexcept {
    return e1.cmp(e2) > 0;
  }
  constexpr inline bool operator>=(ELseverityLevel const& e1, ELseverityLevel const& e2) noexcept {
    return e1.cmp(e2) >= 0;
  }

  // ----------------------------------------------------------------------

}  // end of namespace edm

#endif  // MessageLogger_ELseverityLevel_h
