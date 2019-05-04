/*! \file ANSIColors.hh
 *  \brief Header file containing macros to highlight output lines in color.<BR>
 *  Check also <A
 * HREF="http://www.bluesock.org/~willg/dev/ansi.html">http://www.bluesock.org/~willg/dev/ansi.html</A>
 *  \image html separator.gif
 *  \image latex separator.pdf
 */

#ifndef _ANSICOLORS
#define _ANSICOLORS

#define ACBlack "\x1B[0;30m"
#define ACBlue "\x1B[0;34m"
#define ACGreen "\x1B[0;32m"
#define ACCyan "\x1B[0;36m"
#define ACRed "\x1B[0;31m"
#define ACPurple "\x1B[0;35m"
#define ACBrown "\x1B[0;33m"
#define ACGray "\x1B[0;37m"
#define ACDarkGray "\x1B[1;30m"
#define ACLightBlue "\x1B[1;34m"
#define ACLightGreen "\x1B[1;32m"
#define ACLightCyan "\x1B[1;36m"
#define ACLightRed "\x1B[1;31m"
#define ACLightPurple "\x1B[1;35m"
#define ACYellow "\x1B[1;33m"
#define ACWhite "\x1B[1;37m"

#define ACPlain "\x1B[0m"
#define ACBold "\x1B[1m"
#define ACUnderline "\x1B[4m"
#define ACBlink "\x1B[5m"
#define ACReverse "\x1B[7m"

#define ACClear "\x1B[2J"
#define ACClearL "\x1B[2K"

#define ACSave "\x1B[s"
#define ACRecall "\x1B[u"

#endif
