/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#include "Alignment/RPTrackBased/interface/MatrixTools.h"

#include <cmath>


void Print(TMatrixD& m, const char *label, bool mathematicaFormat)
{
  if (mathematicaFormat) {
    printf("{");
    for (int i = 0; i < m.GetNrows(); i++) {
      if (i > 0) printf(", ");
      printf("{");
      for (int j = 0; j < m.GetNcols(); j++) {
        if (j > 0) printf(", ");
        printf("%.3f", m[i][j]);
      }
      printf("}");
    }
    printf("}\n");
    return;
  }

  if (label)
    printf("\n%s\n", label);

  printf("    | ");
  for (int j = 0; j < m.GetNcols(); j++)
    printf(" %9i", j);
  printf("\n------");
  for (int j = 0; j < m.GetNcols(); j++)
    printf("----------");
  printf("\n");

  for (int i = 0; i < m.GetNrows(); i++) {
    printf("%3i | ", i);
    for (int j = 0; j < m.GetNcols(); j++) {
      double v = m[i][j];
      if
        (fabs(v) >= 1E4) printf(" %+9.2E", v);
      else
        if (fabs(v) > 1E-6)
          printf(" %+9.2E", v);
        else
          printf("         0");
    }
    printf("\n");
  }
}

