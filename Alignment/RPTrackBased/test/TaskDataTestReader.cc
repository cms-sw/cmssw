/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#include "Alignment/RPTrackBased/interface/AlignmentGeometry.h"

#include "TFile.h"

using namespace std;


//----------------------------------------------------------------------------------------------------

int main()
{
  printf("bla\n");

  TFile *f = new TFile("task_data.root");

  printf("f = %p\n", (void*) f);

  AlignmentGeometry *g = (AlignmentGeometry *) f->Get("geometry");

  printf("g = %p\n", (void*) g);

  g->Print();

  return 0;
}

