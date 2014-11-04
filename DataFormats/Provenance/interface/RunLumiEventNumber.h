#ifndef DataFormats_Provenance_RunLumiEventNumber_h
#define DataFormats_Provenance_RunLumiEventNumber_h

/**

\author W. David Dagenhart, created 1 August, 2014

*/

namespace edm {

   typedef unsigned long long EventNumber_t;
   typedef unsigned int LuminosityBlockNumber_t;
   typedef unsigned int RunNumber_t;

   EventNumber_t const invalidEventNumber = 0U;
   LuminosityBlockNumber_t const invalidLuminosityBlockNumber = 0U;
   RunNumber_t const invalidRunNumber = 0U;
}
#endif
