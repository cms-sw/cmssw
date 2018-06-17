#ifndef DDD_RegressionTest_h
#define DDD_RegressionTest_h

#include <string>
#include <DetectorDescription/Core/interface/DDCompactView.h>
// will define a small setup using not XML
// but only DDCore-calls

class ClhepEvaluator;

// constants and world-volume
/* world-volume is a box. It will be subdevided (conceptually) into
   8 corners. In each corner a test can be done. */   
/* adding a global file name */
void regressionTest_setup(ClhepEvaluator&);

// s
void regressionTest_first(ClhepEvaluator&);
void regressionTest_second();
void regressionTest_third();
void regressionTest_forth();

// parser test...
void testParser();

// misc
void testrot();

void output(std::string filename);
#endif
