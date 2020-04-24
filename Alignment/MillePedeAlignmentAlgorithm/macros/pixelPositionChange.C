// Take two treeFile names as input,
// plot pixel module positions using PlotMillePede::DrawOrigPos(..)
// for BPIX and (BPIX+FPIX) and get the mean values of the histograms
// (i.e. centre of gravity).
// These are printed (in cm) and the difference is evaluated (in mum).
//
// Before executing this script you have to execute allMillePede.C.

#include <vector>
#include <iostream>
#include "GFUtils/GFHistManager.h"
#include "GFUtils/GFHistArray.h"
#include "PlotMillePede.h"

void pixelPositionChange(const char *treeFile1, const char *treeFile2)
{
  std::vector<PlotMillePede*> ps;
  ps.push_back(new PlotMillePede(treeFile1));
  ps.push_back(new PlotMillePede(treeFile2));

  std::vector<double> meanBPIX_X(ps.size());
  std::vector<double> meanBPIX_Y(ps.size());
  std::vector<double> meanBPIX_Z(ps.size());

  std::vector<double> meanPixel_X(ps.size());
  std::vector<double> meanPixel_Y(ps.size());
  std::vector<double> meanPixel_Z(ps.size());
  for (unsigned int i = 0; i < ps.size(); ++i) {
    ps[i]->GetHistManager()->SetBatch();
    ps[i]->SetSubDetId(1);
    ps[i]->DrawOrigPos();
    meanBPIX_X[i] = ps[i]->GetHistManager()->GetHistsOf(0,2)->First()->GetMean();
    meanBPIX_Y[i] = ps[i]->GetHistManager()->GetHistsOf(0,3)->First()->GetMean();
    meanBPIX_Z[i] = ps[i]->GetHistManager()->GetHistsOf(0,4)->First()->GetMean();
    ps[i]->SetSubDetIds(1,2);
    ps[i]->DrawOrigPos(true);
    meanPixel_X[i] = ps[i]->GetHistManager()->GetHistsOf(1,2)->First()->GetMean();
    meanPixel_Y[i] = ps[i]->GetHistManager()->GetHistsOf(1,3)->First()->GetMean();
    meanPixel_Z[i] = ps[i]->GetHistManager()->GetHistsOf(1,4)->First()->GetMean();

  }
  ps[0]->GetHistManager()->Overlay(ps[1]->GetHistManager(), 0, 0, "second");
  ps[0]->GetHistManager()->Overlay(ps[1]->GetHistManager(), 1, 1, "second");
  ps[0]->GetHistManager()->SetBatch(false);
  ps[0]->GetHistManager()->SameWithStats(true);
  ps[0]->GetHistManager()->Draw();

  const char separator[] = "====================================================\n";

  std::cout << separator << "BPIX:\n" << separator 
	    << "x1 x2: " << meanBPIX_X[0] << " " << meanBPIX_X[1]
	    << '\n'
	    // << ", diff (in mum) " 
	    // << (meanBPIX_X[1] - meanBPIX_X[0])* 10000. << '\n'

	    << "y1 y2: " << meanBPIX_Y[0] << " " << meanBPIX_Y[1]
	    << '\n'
	    // << ", diff (in mum) " 
	    // << (meanBPIX_Y[1] - meanBPIX_Y[0])* 10000. << '\n'

	    << "z1 z2: " << meanBPIX_Z[0] << " " << meanBPIX_Z[1]
	    << '\n'
	    // << ", diff (in mum) " 
	    // << (meanBPIX_Z[1] - meanBPIX_Z[0])* 10000.

	    << "Delta(x,y,z) = 2-1 = ("
	    << (meanBPIX_X[1] - meanBPIX_X[0])* 10000. << ','
	    << (meanBPIX_Y[1] - meanBPIX_Y[0])* 10000. << ','
	    << (meanBPIX_Z[1] - meanBPIX_Z[0])* 10000. << ") mum"
	    << std::endl;

  std::cout << separator << "Pixel:\n" << separator; 
  
  std::cout << "x1 x2: " << meanPixel_X[0] << " " << meanPixel_X[1]
	    << '\n'
	    // << ", diff (in mum) " 
	    // << (meanPixel_X[1] - meanPixel_X[0])* 10000. << '\n'

	    << "y1 y2 " << meanPixel_Y[0] << " " << meanPixel_Y[1]
	    << '\n'
	    // << ", diff (in mum) " 
	    // << (meanPixel_Y[1] - meanPixel_Y[0])* 10000. << '\n'

	    << "z1 z2: " << meanPixel_Z[0] << " " << meanPixel_Z[1]
	    << '\n'
	    // << ", diff (in mum) " 
	    // << (meanPixel_Z[1] - meanPixel_Z[0])* 10000.

	    << "Delta(x,y,z) = 2-1 = ("
	    << (meanPixel_X[1] - meanPixel_X[0])* 10000. << ','
	    << (meanPixel_Y[1] - meanPixel_Y[0])* 10000. << ','
	    << (meanPixel_Z[1] - meanPixel_Z[0])* 10000. << ") mum"
	    << std::endl;
  
  TString shortFileName1(treeFile1);
  if (shortFileName1.Contains("/")) { // remove all directory info
    shortFileName1.Remove(0, shortFileName1.Last('/') + 1);
  }
  TString shortFileName2(treeFile2);
  if (shortFileName2.Contains("/")) {
    shortFileName2.Remove(0, shortFileName2.Last('/') + 1);
  }

  std::cout << "|  | *" << shortFileName1 << "* || *"<< shortFileName1 << "*||\n"
	    << "| *Position* |  *BPIX* | *Full Pixel* | *BPIX* | *Full Pixel* |\n"
	    << "|  x  | " << meanBPIX_X[0] << " | " << meanPixel_X[0] << " | " << meanBPIX_X[1] << " | " << meanPixel_X[1] << " |\n"
	    << "|  y  | " << meanBPIX_Y[0] << " | " << meanPixel_Y[0] << " | " << meanBPIX_Y[1] << " | " << meanPixel_Y[1] << " |\n"
	    << "|  z  | " << meanBPIX_Z[0] << " | " << meanPixel_Z[0] << " | " << meanBPIX_Z[1] << " | " << meanPixel_Z[1] << " |\n"
	    << std::endl;

  std::cout << separator << separator << separator << std::endl; 

}

