#include "RecoLocalCalo/EcalRecAlgos/interface/TDecompCholFast.h"

void TDecompCholFast::SetMatrixFast(const TMatrixDSym& a, double *data) {
 
// Set the matrix to be decomposed, decomposition status is reset.
// Use external memory *data for internal matrix fU

   R__ASSERT(a.IsValid());

   ResetStatus();
   if (a.GetNrows() != a.GetNcols() || a.GetRowLwb() != a.GetColLwb()) {
      Error("SetMatrix(const TMatrixDSym &","matrix should be square");
      return;
   }

   SetBit(kMatrixSet);
   fCondition = -1.0;

   fRowLwb = a.GetRowLwb();
   fColLwb = a.GetColLwb();
   fU.Use(a.GetNrows(),a.GetNcols(),data);
   fU = a;
  
  
}