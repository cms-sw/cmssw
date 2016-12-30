/**********************************************************************
 * This file is part of printMacros.
 *
 * Copyright (C) 2003 Philippe Gras
 * Copyright (C) 2004 Philippe Gras
 * Copyright (C) 2005 Philippe Gras
 *
 * printMacros is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * printMacros is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with printMacros; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 ************************************************************************
 * printMacros release 1.0
 *
 * This file defines some macros to display various messages.
 * Description of the macros:
 *
 * File Localisation:
 * -----------------
 *   __HERE__: expands to file_name:line_number where file_name and line_number
 *   are the name of the file and the number of the line where __HERE__ is 
 *   invoked.
 * 
 * Warning and errors:
 * ------------------
 *    Use the following macros to emit warning and error. File
 *    name, the line number and error level (Warning, error, fatal error) are
 *    included in the displayed message. The argument 'stream' can be a 
 *    literal string or a STL stream, e.g.:
 *       "This is a warning."
 *       "This is an error. The number of cells must be less than " << cellMax
 *
 *    The macros to display warning/error messages are:
 *
 *    WARNING(stream)    Displays a warning message 
 *    ERROR(stream)      Displays a non-fatal error message
 *    FATALERROR(stream) Displays a fatal error message and abort
 *    IMPLERROR(steam)   Displays an implementation error and abort. An 
 * implementation error is defined as an error which by designe should never 
 * happen: for instance a default case of a switch which should never be 
 * reached. This macro ask also to the user to send a bug report to the adress
 * 'supportEmail' defined in this file.
 *
 * Messages:
 * --------
 *   In order to edit informative/debugging message use the macros:
 * MESS(verbosity,stream) or MESSN(verbosity,stream):
 * 
 *     MESS(verbosity, stream) sends 'stream' to cout (see above 
 * Warning and errors for usage of stream) if the verbosity (see gVerbosity)
 * is set to a value greater or equal to 'verbosity'.
 *
 *     MESSN(verbosity,stream) is identical to MESS but it appends a 
 * newline to the message and flushes the stream (std::endl).
 *
 * Temporary Debugging message:
 * ---------------------------
 *   For debugging purpose it is often conveninent to display some messages,
 * variable values, etc. Those messages must be absent in release, whatever
 * is the verbosity. For this purpose use the following macros. File name
 * and line number where macro is invoked are added to the message. 'stream'
 * usage is the same as for the Warning and error macros (see above).
 * 
 *  DPRINT(stream)  sends stream to cerr
 *  DPRINTN(stream) sends stream and a std::endl to cerr. The message is 
 * therefore terminated with a new line and the stream is flushed.
 *  PRINTVAL(variable) displays the name and the value of the variable 
 * 'variable'. For instance, assuming i=5, PRINTVAL(i) will display:
 *                            foo.cxx:24: i = 5
 *  PRINTVALN(variable) is like PRINTVAL(variable) but it appends a std::endl
 *  VAL(variable) expands to:
 *                          "variable" << variable
 *  It can then be used to display several variables on the same line,
 *                      DEBUGN(VAL(i) << ", " << VAL(imax))
 *  will display
 *                         foo.cxx:24: i = 5, i = 6
 *  PRETURN(ret_value) to use insted of "return". Displays ret_value 
 *  and returns ret_value. 
 *
 *
 * Assertion
 * ----------
 * 
 *   The VASSSERT macro ("Verbose ASSERTion) can be used for assertion of type
 *                    assert(a.op.b) with .op.:=<.>,-,etc...
 * In addition to the standard assert message it gives the value of a and b.
 * The message given by VASSERT(a.op.b) has the following format:
 *   <program_invocation_name>: <source_file_name>:<line_number>: <function>:
 *  Assertion `a.op.b' (<value_of_a>.op.<value_of_b>) failed.
 *
 * The VASSERT, DPRINT, DPRINTN, PRINTVAL, PRINTVALN, PRETURN are desactivated 
 * (defined as empty macros) if the NDEBUG is defined.
 */

#if not defined(ENDL)
# define ENDL "\n"
#endif //ENDL not defined

#ifndef __CINT__
#ifndef PRINTMACROS
#define PRINTMACROS

#include <iostream>
#include <cstdlib> //needed by abort()
#include <csignal>

#ifdef __GNUC__
extern "C" char *program_invocation_name;
#define PROGNAME program_invocation_name
#else //not GNU C
#define PROGNAME ""
#endif //GNU C

#ifndef __PRETTY_FUNCTION__ //GNU C
# ifdef __FUNCTION__        //GNU C
#   define __PRETTY_FUNCTION__ __FUNCTION__
# else //__FUNCTION__ not defined
#   define __PRETTY_FUNCTION__ __func__ //C99 standard
# endif //__FUNCTION__ defined
#endif //__PRETTY_FUNCTION__ not defined

const static char* supportEmail __attribute__ ((unused)) = "cms-orca-developers@cern.ch";
static std::ostream& printMacroOut = std::cout;


#ifndef NDEBUG
/*static*/ sig_atomic_t gVerbosity __attribute__ ((unused,weak)) = 1;
#else
/*static*/ sig_atomic_t gVerbosity __attribute__ ((unused,weak)) = 0;
#endif //NDEBUG not defined

//number of verbosity level (including 0):
static const int nVerbosityLevels = 10;

//macro which expands to <file>, line <line>:
#define __HERE__  __FILE__ ":" QUOTE2(__LINE__)
#define QUOTE(arg) #arg
#define QUOTE2(arg) QUOTE(arg)


// Warning and error messages:
/**
 */
#define WARNING(stream) printMacroOut << "Warning " << __HERE__ << ": " \
<< stream << ENDL;

/** Emits a warning. Source code file name and line number are added 
 * to the message.
 * @param stream error message string or a std stream, see WARNING
 */
#define ERROR(stream) printMacroOut << "Error " << __HERE__ << ": " \
<< stream << ENDL;


/** Emits an fatal error and abort. Source code file name and 
 * line number are added to the message.
 * @param stream error message string or a std stream, see WARNING
 */
#define FATALERROR(stream) printMacroOut << "Fatal error " << __HERE__ << ": " \
<< stream << ENDL;\
abort()

/** Emits an implementation error and abort. An impementation error is defined
 * as an error we should never happened according to the program design: for 
 * instance a default case of a switch which shoulld never be reached.
 * Source code file name and line number are added to the message.
 * @param stream error message string or a std stream, see WARNING
 */

#define IMPLERROR(stream) printMacroOut << "Implementation error " << __HERE__ \
<< ": " << stream << ". Please send a bug report to " << supportEmail \
<< ENDL;\
abort()

#ifdef LOCATE_MESS
#define MESS_(verbosity,stream) if(gVerbosity>=verbosity) std::cout << \
__HERE__ << ": " << stream
#else //LOCATE_MESS not defined
#define MESS_(verbosity,stream) if(gVerbosity>=verbosity) std::cout \
<< stream
#endif //LOCATE_MESS defined

#define MESS(verbosity,stream) MESS_(verbosity,stream) ;
#define MESSN(verbosity,stream) MESS_(verbosity,stream) << ENDL;


#define VAL(variable) #variable << " = " << variable


#ifdef NDEBUG

#define DPRINT(stream)
#define DPRINTN(stream)
#define PRINTVAL(variable)
#define PRINTVALN(variable)
#define PRETURN(ret_value)
#define VASSERT(a, op, b) (void)0
#define VFUNC(exp) exp
#define VFUNC1(f,args) f args

#else //NDEBUG not defined

#define DPRINT_(stream, suff) \
(gVerbosity?\
    (std::cout << __HERE__ << ": " << stream << suff,(void)0):\
    (void)0)
#define DPRINT(stream) DPRINT_(stream, "")
#define DPRINTN(stream) DPRINT_(stream, ENDL)

#define PRINTVAL_(variable, suff) (gVerbosity?(std::cout << __HERE__ << ": " \
<< VAL(variable) << suff,(void)0):(void)0)
#define PRINTVAL(variable) PRINTVAL_(variable,"")
#define PRINTVALN(variable) PRINTVAL_(variable,ENDL)

template<class T>
T& _printMacros_preturn(const char* here, T& ret_value){
  std::cout << __HERE__ << ": returns "
            << VAL(ret_value)
            << ENDL;
  return ret_value;
}

#define PRETURN(ret_value) return (gVerbosity?(_printMacros_preturn(__HERE__, (ret_value))):ret_value)

#define VASSERT(a, op, b)\
((!((a) op (b)))?\
  (printMacroOut << PROGNAME << (PROGNAME[0]?": ":"") << __HERE__ << ": " \
            << __PRETTY_FUNCTION__ << ": Assertion '" << #a << #op << #b \
            << "' (" << (a) << #op << (b) <<") failed." << ENDL,\
   abort()):(void)0)

#define VFUNC(exp) (gVerbosity?(std::cout << __HERE__ ": " #exp ": \n", (exp)):exp)
template<class T>
void print_args(const T& a){
  std::cout << a;
}
template<class T0, class T1>
void print_args(const T0& a, const T1& b){
  std::cout << a << "," << b;
}
template<class T0, class T1, class T2>
void print_args(const T0& a, const T1& b, const T2& c){
  std::cout << a << "," << b << "," << c;
}
template<class T0, class T1, class T2, class T3>
void print_args(const T0& a, const T1& b, const T2& c, const T3& d){
  std::cout << a << "," << b << "," << c << "," << d;
}
template<class T0, class T1, class T2, class T3, class T4>
void print_args(const T0& a, const T1& b, const T2& c, const T3& d,
		const T4& e){
  std::cout << a << "," << b << "," << c << "," << d << "," << e;
}
template<class T0, class T1, class T2, class T3, class T4, class T5>
void print_args(const T0& a, const T1& b, const T2& c, const T3& d,
		const T4& e, const T5& f){
  std::cout << a << "," << b << "," << c << "," << e << "," << f;
}

template<class T>
T _printMacros_vfunc1(T val){
  std::cout << val << ENDL;
  return val;
}
#define VFUNC1(f,args) (gVerbosity?(std::cout << __HERE__ ": " #f "(", print_args args, std::cout << ") = ", _printMacros_vfunc1((f args))):0)


#endif //NDEBUG defined
#endif // PRINTMACROS
#endif // __CINT__
	
