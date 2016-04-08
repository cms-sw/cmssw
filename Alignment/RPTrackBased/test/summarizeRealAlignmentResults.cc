/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#include <stddef.h>
#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>
#include <cstring>
#include <unistd.h>
#include <map>
#include <string>
#include <cmath>

#include "TFile.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TH1D.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "Alignment/RPDataFormats/interface/RPAlignmentCorrections.h"

using namespace std;

//----------------------------------------------------------------------------------------------------

TDirectory* mkdir(TDirectory *parent, const char *child)
{
  TDirectory *dir = (TDirectory *) parent->Get(child);
  if (!dir)
    dir = parent->mkdir(child);
  return dir;
}

//----------------------------------------------------------------------------------------------------

struct PlotSet {
  TGraphErrors *shr, *shx, *shy, *rotz;
  PlotSet();
  void Write();
};

//----------------------------------------------------------------------------------------------------

PlotSet::PlotSet()
{
  shr = new TGraphErrors(); shr->SetName("shr"); shr->SetTitle(";iteration;shr   (um)");
  shx = new TGraphErrors(); shx->SetName("shx"); shx->SetTitle(";iteration;shx   (um)");
  shy = new TGraphErrors(); shy->SetName("shy"); shy->SetTitle(";iteration;shy   (um)");
  rotz = new TGraphErrors(); rotz->SetName("rotz"); rotz->SetTitle(";iteration;rotz   (mrad)");
}

//----------------------------------------------------------------------------------------------------

void PlotSet::Write()
{
  shr->Write();
  shx->Write();
  shy->Write();
  rotz->Write();
}

//----------------------------------------------------------------------------------------------------

typedef map<unsigned int, PlotSet> PlotMap;

PlotMap rpPlots, detPlots;

TGraph *eventsTotal, *eventsSelected;

//----------------------------------------------------------------------------------------------------

void PreparePlots(TFile *sf, const char *d1, const char *d2, const char *d3)
{
  gDirectory = mkdir(mkdir(mkdir(sf, d1), d2), d3);

  eventsTotal = new TGraph(); eventsTotal->SetName("eventsTotal");
  eventsSelected = new TGraph(); eventsSelected->SetName("eventsSelected");

  rpPlots.clear();
  detPlots.clear();

  for (unsigned int a = 0; a < 2; a++) {
    for (unsigned int s = 0; s < 3; s++) {
        if (s != 2)
          continue;
        for (unsigned int r = 0; r < 6; r++) {
          rpPlots[100*a+10*s+r] = PlotSet();

          for (unsigned int d = 0; d < 10; d++) {
            detPlots[1000*a+100*s+10*r+d] = PlotSet();
          }
        }
    }
  }
}

//----------------------------------------------------------------------------------------------------

void UpdatePlots(unsigned int i, RPAlignmentCorrections &r)
{
  //printf("* %u\n", i);
  
  // update RP plots
  for (PlotMap::iterator it = rpPlots.begin(); it != rpPlots.end(); ++it) {
    unsigned int idx = it->second.shr->GetN();
    const RPAlignmentCorrection &ac = r.GetRPCorrection(it->first);

    it->second.shr->SetPoint(idx, i, ac.sh_r()*1E3);
    it->second.shr->SetPointError(idx, 0., ac.sh_r_e()*1E3);
    
    it->second.shx->SetPoint(idx, i, ac.sh_x()*1E3);
    it->second.shx->SetPointError(idx, 0., ac.sh_x_e()*1E3);
    
    it->second.shy->SetPoint(idx, i, ac.sh_y()*1E3);
    it->second.shy->SetPointError(idx, 0., ac.sh_y_e()*1E3);
    
    it->second.rotz->SetPoint(idx, i, ac.rot_z()*1E3);
    it->second.rotz->SetPointError(idx, 0., ac.rot_z_e()*1E3);
  }
  
  // update det plots
  for (PlotMap::iterator it = detPlots.begin(); it != detPlots.end(); ++it) {
    unsigned int idx = it->second.shr->GetN();
    const RPAlignmentCorrection &ac = r.GetSensorCorrection(it->first);

    it->second.shr->SetPoint(idx, i, ac.sh_r()*1E3);
    it->second.shr->SetPointError(idx, 0., ac.sh_r_e()*1E3);
    
    it->second.shx->SetPoint(idx, i, ac.sh_x()*1E3);
    it->second.shx->SetPointError(idx, 0., ac.sh_x_e()*1E3);
    
    it->second.shy->SetPoint(idx, i, ac.sh_y()*1E3);
    it->second.shy->SetPointError(idx, 0., ac.sh_y_e()*1E3);
    
    it->second.rotz->SetPoint(idx, i, ac.rot_z()*1E3);
    it->second.rotz->SetPointError(idx, 0., ac.rot_z_e()*1E3);
  }
}

//----------------------------------------------------------------------------------------------------

void WritePlots()
{
  //printf(">> WritePlots\n");

  TDirectory *orig = gDirectory;
  char buf[100];

  eventsTotal->Write();
  eventsSelected->Write();

  // write RP plots
  TDirectory *base = mkdir(orig, "RP");
  for (PlotMap::iterator it = rpPlots.begin(); it != rpPlots.end(); ++it) {
    sprintf(buf, "%u", it->first);
    gDirectory = mkdir(base, buf);
    it->second.Write();
  }
  
  // write detector plots
  base = mkdir(orig, "det");
  for (PlotMap::iterator it = detPlots.begin(); it != detPlots.end(); ++it) {
    sprintf(buf, "%u", it->first);
    gDirectory = mkdir(base, buf);
    it->second.Write();
  }

  gDirectory = orig;
}

//----------------------------------------------------------------------------------------------------

void ProcessLog(unsigned int i)
{
  FILE *f = fopen("log", "r");
  if (!f) {
    printf("      iteration %u: cannot process log file.\n", i);
    return;
  }

  char buf[201];
  while (!feof(f)) {
    fgets(buf, 200, f);
    
    char *p = NULL;
    if ((p = strstr(buf, "events total"))) {
      eventsTotal->SetPoint(eventsTotal->GetN(), i, atoi(p + 15));
    }
    
    if ((p = strstr(buf, "events selected"))) {
      eventsSelected->SetPoint(eventsSelected->GetN(), i, atoi(p + 18));
    }
  }

  fclose(f);
}

//----------------------------------------------------------------------------------------------------

bool IsRegDir(const dirent *de)
{
  if (de->d_type != DT_DIR)
    return false;

  if (!strcmp(de->d_name, "."))
    return false;

  if (!strcmp(de->d_name, ".."))
    return false;

  return true;
}

//----------------------------------------------------------------------------------------------------

int main(void)
{
  // open output file
  TFile *sf = new TFile("result_summary.root", "recreate");

  // traverse directory structure
  try {
    DIR *dp_t = opendir(".");
    dirent *de_t;
    while ((de_t = readdir(dp_t))) {
      if (!IsRegDir(de_t))
        continue;
      printf("%s\n", de_t->d_name);

      chdir(de_t->d_name);
      DIR *dp_a = opendir(".");
      dirent *de_a;
      while ((de_a = readdir(dp_a))) {
        if (!IsRegDir(de_a))
          continue;
        printf("  %s\n", de_a->d_name);

        chdir(de_a->d_name);
        DIR *dp_s = opendir(".");
        dirent *de_s;
        while ((de_s = readdir(dp_s))) {
          if (!IsRegDir(de_s))
            continue;
          printf("    %s\n", de_s->d_name);
          
          PreparePlots(sf, de_t->d_name, de_a->d_name, de_s->d_name);

          chdir(de_s->d_name);
          DIR *dp_i = opendir(".");
          dirent *de_i;
          while ((de_i = readdir(dp_i))) {
            if (!IsRegDir(de_i))
              continue;
            //printf("      %s\n", de_i->d_name);

            chdir(de_i->d_name);
            unsigned int iteration = atoi(de_i->d_name+9);
            ProcessLog(iteration);

            try {
              RPAlignmentCorrections r("./cumulative_factored_results_Jan.xml");
              UpdatePlots(iteration, r);
            }
            catch (cms::Exception e) {
              printf("      %s: cannot process XML file\n", de_i->d_name);
            }

            chdir("..");

          }
          closedir(dp_i);
          chdir("..");
        
          WritePlots();
        }
        closedir(dp_s);
        chdir("..");
      }
      closedir(dp_a);
      chdir("..");
    }
    closedir(dp_t);
  }

  catch (cms::Exception e) {
    printf("ERROR: A CMS exception has been caught:\n%s\nStopping.\n", e.what());
  }
  
  catch (std::exception e) {
    printf("ERROR: A std::exception has been caught:\n%s\nStopping.\n", e.what());
  }

  catch (...) {
    printf("ERROR: An exception has been caught, stopping.\n");
  }

  printf(">> CloseFile\n");
  delete sf;

  return 0;
}

