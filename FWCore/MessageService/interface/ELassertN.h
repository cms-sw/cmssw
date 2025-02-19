// ELassertN.h
//
// Graded assert macros producing ErrorLogger messages
//
// There is no code guard in this file, since assert.h files can be
// included multiple times.  That is, one can (in the same complilation unit)
// include ELassertN.h (or for that matter assert.h) and get meanings of the
// assert macros based on the current definition status of NDEBUG -- and later,
// one can alter NDEBUG (or NDEBUG1, NDEBUG2, NDBUG3) and re-include ELassertN.h
// so as to change to the new assertion behavior.  
//
// Usage:
//
// 	The variable errlog must be an ErrorLog which is in scope where the
//	ELassert is done.  The error message will go to that errlog.
//
//	ELassert  (condition) will behave like assert(condition) but if
//			      triggered will issue an error message with
//			      ELabort severity, rather than a cerr output
//			      and an abort.
//	ELassert1 (condition) will behave like ELassert(condition) but is
//	 		      disabled by either NDEBUG or NDEBUG1,
//	ELassert2 (condition) is disabled by those and by NDEBUG2.
//	ELassert3 (condition) is disabled by those and by NDEBUG2 or NDEBUG3.
//
//	ELsanityCheck(condition) will behave like ELassert(condition) but
//			         is disabled by NOSANCHECK rather than NDEBUG.
//
//	ELwarningAssert  (condition) same as the corresponding ELassert,
//      ELwarningAssert1 (condition) but will use ELwarning as the message
//      ELwarningAssert1 (condition) severity if triggered.
//      ELwarningAssert1 (condition)
//	ELwarningCheck   (condition) same as ELsanityCheck but will issue
//				     an error message with ELwarning severity
//				     (rather than ELabort) if triggered

// Clean up any earlier definitions, in case this is not the first time the
// file has been included:
#if defined ELassert3
     #undef ELassert3
#endif
#if defined ELwarningAssert3
     #undef ELwarningAssert3
#endif
#if defined ELassert2
     #undef ELassert2
#endif
#if defined ELwarningAssert2
     #undef ELwarningAssert2
#endif
#if defined ELassert1
     #undef ELassert1
#endif
#if defined ELwarningAssert1
     #undef ELwarningAssert1
#endif
#if defined ELassert
     #undef ELassert
#endif
#if defined ELwarningAssert
     #undef ELwarningAssert
#endif
#if defined ELsanityCheck
     #undef ELsanityCheck
#endif
#if defined ELwarningCheck
     #undef ELwarningCheck
#endif

#ifndef NDEBUG3
#define ELassert3(condition) ELassert2(condition)
#define ELwarningAssert3(condition) ELwarningAssert2(condition)
#else
#define ELassert3(condition) ((void)0)
#define ELwarningAssert3(condition) ((void)0)
#endif

#ifndef NDEBUG2
#define ELassert2(condition) ELassert1(condition)
#define ELwarningAssert2(condition) ELwarningAssert1(condition)
#else
#define ELassert2(condition) ((void)0)
#define ELwarningAssert2(condition) ((void)0)
#endif

#ifndef NDEBUG1
#define ELassert1(condition) ELassert(condition)
#define ELwarningAssert1(condition) ELwarningAssert(condition)
#else
#define ELassert1(condition) ((void)0)
#define ELwarningAssert1(condition) ((void)0)
#endif

#ifndef NDEBUG
#define ELassert(condition) \
  ((condition)?((void)(0)):((void)( \
  errlog(ELabort,"assert failure")<<# condition<<"in"<<__FILE__<<"at line"\
  <<__LINE__<<endmsg)))
#define ELwarningAssert(condition) \
  ((condition)?((void)(0)):((void)( \
  errlog(ELwarning,"assert warning")<<# condition<<"in"<<__FILE__<<"at line"\
  <<__LINE__<<endmsg)))
#else
#define ELassert(condition) ((void)0)
#endif

#ifndef NOSANCHECK
#define ELsanityCheck(condition) \
  ((condition)?((void)(0)):((void)( \
  errlog(ELabort,"assert failure")<<# condition<<"in"<<__FILE__<<"at line"\
  <<__LINE__<<endmsg)))
#define ELwarningCheck(condition) \
  ((condition)?((void)(0)):((void)( \
  errlog(ELwarning,"assert warning")<<# condition<<"in"<<__FILE__<<"at line"\
  <<__LINE__<<endmsg)))
#else
#define ELsanityCheck(condition) ((void)0)
#define ELwarningCheck(condition) ((void)0)
#endif


