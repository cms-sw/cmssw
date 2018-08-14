#include <iostream>
#include <stdlib.h>
#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"

using namespace tbb;
using namespace std;

class ArraySummer {

  int * p_array_a;
  int * p_array_b;
  int * p_array_sum;

public:
  // This empty constructor with an initialization list is used to setup calls to the function
  ArraySummer(int * p_a, int * p_b, int * p_sum) : p_array_a(p_a), p_array_b(p_b), p_array_sum(p_sum) { }

  void operator() ( const blocked_range<int>& r ) const {
    for ( int i = r.begin(); i != r.end(); i++ ) { // iterates over the entire chunk
      p_array_sum[i] = p_array_a[i] + p_array_b[i];
    }
  }

};

int main(int argc, char *argv[]) {
 int * p_A;
 int * p_B;
 int * p_SUM_1T;
 int * p_SUM_TBB;

 /* This is the TBB runtime... */
 task_scheduler_init init;

 constexpr int nElements = 10;

 p_A       = new int[nElements];
 p_B       = new int[nElements];
 p_SUM_1T  = new int[nElements];
 p_SUM_TBB = new int[nElements];

 /* 
  * Initialize the data sets ... could do this in parallel too, but 
  * serial is easier to read
  */
 p_A[0] = p_B[0] = 0;
 p_A[1] = p_B[1] = 1;
 for( int i=2;i<nElements;i++) {
   p_A[i]   = (p_A[i-1] + p_A[i-2]) % (INT_MAX/2);
   p_B[i]   = p_A[i];
   p_SUM_1T[i] = 0;
   p_SUM_TBB[i] = 0;
 }


 /*
  * Time how long it takes to sum the arrays using a single thread
  */
 for( int i=0;i<nElements;i++ ) {
   p_SUM_1T[i] = p_A[i] + p_B[i];
 }

 /*
  * Now sum the arrays again using TBB, again timing the execution
  */

 parallel_for(blocked_range<int>(0, nElements, 100),
       ArraySummer( p_A, p_B, p_SUM_TBB ) );

 /*
  * Verify the sums match
  */
 for(int i=0;i<nElements;i++) {
   if( p_SUM_1T[i] != p_SUM_TBB[i] ) {
     cout << p_A[i] << " + " << p_B[i] << " = " << p_SUM_1T[i] << " AND " << p_SUM_TBB[i] <<  endl;
   }
 }

 delete [] p_A;
 delete [] p_B;
 delete [] p_SUM_1T;
 delete [] p_SUM_TBB;

 return 0;
}

