/* EDAnalyzer to study property of gains.
 *
 * \author Dominique Fortin
 */

#include <memory>
#include <iostream>
#include <stdexcept>

// user include files
#include <CondFormats/CSCObjects/test/CSCGainsStudy.h>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "TFile.h"

using namespace std;
using namespace edm;

// Constructor
CSCGainsStudy::CSCGainsStudy(const ParameterSet& pset) {
  // Get the various input parameters
  debug = pset.getUntrackedParameter<bool>("debug");
  rootFileName = pset.getUntrackedParameter<string>("rootFileName");

  if (debug)
    cout << "[CSCGainsStudy] Constructor called" << endl;

  // Create the root file
  theFile = new TFile(rootFileName.c_str(), "RECREATE");
  theFile->cd();

  // Book the histograms
  All_CSC = new HCSCGains("All_CSC");
  // ME+1/1
  ME_11_27 = new HCSCGains("ME_11_27");
  ME_11_28 = new HCSCGains("ME_11_28");
  ME_11_29 = new HCSCGains("ME_11_29");
  ME_11_30 = new HCSCGains("ME_11_30");
  ME_11_31 = new HCSCGains("ME_11_31");
  ME_11_32 = new HCSCGains("ME_11_32");
  // ME+1/2
  ME_12_27 = new HCSCGains("ME_12_27");
  ME_12_28 = new HCSCGains("ME_12_28");
  ME_12_29 = new HCSCGains("ME_12_29");
  ME_12_30 = new HCSCGains("ME_12_30");
  ME_12_31 = new HCSCGains("ME_12_31");
  ME_12_32 = new HCSCGains("ME_12_32");
  // ME+1/3
  ME_13_27 = new HCSCGains("ME_13_27");
  ME_13_28 = new HCSCGains("ME_13_28");
  ME_13_29 = new HCSCGains("ME_13_29");
  ME_13_30 = new HCSCGains("ME_13_30");
  ME_13_31 = new HCSCGains("ME_13_31");
  ME_13_32 = new HCSCGains("ME_13_32");
  // ME+2/1
  ME_21_14 = new HCSCGains("ME_21_14");
  ME_21_15 = new HCSCGains("ME_21_15");
  ME_21_16 = new HCSCGains("ME_21_16");
  // ME+2/2
  ME_22_27 = new HCSCGains("ME_22_27");
  ME_22_28 = new HCSCGains("ME_22_28");
  ME_22_29 = new HCSCGains("ME_22_29");
  ME_22_30 = new HCSCGains("ME_22_30");
  ME_22_31 = new HCSCGains("ME_22_31");
  ME_22_32 = new HCSCGains("ME_22_32");
  // ME+3/1
  ME_31_14 = new HCSCGains("ME_31_14");
  ME_31_15 = new HCSCGains("ME_31_15");
  ME_31_16 = new HCSCGains("ME_31_16");
  // ME+3/2
  ME_32_27 = new HCSCGains("ME_32_27");
  ME_32_28 = new HCSCGains("ME_32_28");
  ME_32_29 = new HCSCGains("ME_32_29");
  ME_32_30 = new HCSCGains("ME_32_30");
  ME_32_31 = new HCSCGains("ME_32_31");
  ME_32_32 = new HCSCGains("ME_32_32");
}

// Destructor
CSCGainsStudy::~CSCGainsStudy() {
  if (debug)
    cout << "[CSCGainsStudy] Destructor called" << endl;

  // Write the histos to file
  theFile->cd();
  All_CSC->Write();
  // ME+1/1
  ME_11_27->Write();
  ME_11_28->Write();
  ME_11_29->Write();
  ME_11_30->Write();
  ME_11_31->Write();
  ME_11_32->Write();
  // ME+1/2
  ME_12_27->Write();
  ME_12_28->Write();
  ME_12_29->Write();
  ME_12_30->Write();
  ME_12_31->Write();
  ME_12_32->Write();
  // ME+1/3
  ME_13_27->Write();
  ME_13_28->Write();
  ME_13_29->Write();
  ME_13_30->Write();
  ME_13_31->Write();
  ME_13_32->Write();
  // ME+2/1
  ME_21_14->Write();
  ME_21_15->Write();
  ME_21_16->Write();
  // ME+2/2
  ME_22_27->Write();
  ME_22_28->Write();
  ME_22_29->Write();
  ME_22_30->Write();
  ME_22_31->Write();
  ME_22_32->Write();
  // ME+3/1
  ME_31_14->Write();
  ME_31_15->Write();
  ME_31_16->Write();
  // ME+3/2
  ME_32_27->Write();
  ME_32_28->Write();
  ME_32_29->Write();
  ME_32_30->Write();
  ME_32_31->Write();
  ME_32_32->Write();

  // Close file
  theFile->Close();
  if (debug)
    cout << "************* Finished writing histograms to file" << endl;
}

/* analyze
 *
 */
void CSCGainsStudy::analyze(const Event& event, const EventSetup& eventSetup) {
  // Get the gains and compute global gain average to store for later use in strip calibration
  edm::ESHandle<CSCGains> hGains;
  eventSetup.get<CSCGainsRcd>().get(hGains);
  const CSCGains* hGains_ = &*hGains.product();

  // Store so it can be used in member functions
  pGains = hGains_;

  // Get global gain average:
  float AvgStripGain = getStripGainAvg();

  if (debug)
    std::cout << "Global average gain is " << AvgStripGain << std::endl;

  HCSCGains* histo = nullptr;

  TString prefix = "ME_";
  float thegain = -10.;
  float weight = -10.;
  float weight0 = 1.;

  // Build iterator which loops on all layer id:
  map<int, vector<CSCGains::Item> >::const_iterator it;

  for (it = pGains->gains.begin(); it != pGains->gains.end(); ++it) {
    // Channel id used for retrieving gains from database is
    // chId=220000000 + ec*100000 + st*10000 + rg*1000 + ch*10 + la;

    const unsigned ChId = it->first;
    unsigned offset = (ChId - 220000000);
    unsigned ec = offset / 100000;                                            // endcap
    unsigned st = (offset - ec * 100000) / 10000;                             // station
    unsigned rg = (offset - ec * 100000 - st * 10000) / 1000;                 // ring
    unsigned ch = (offset - ec * 100000 - st * 10000 - rg * 1000) / 10;       // chamber
    unsigned la = (offset - ec * 100000 - st * 10000 - rg * 1000 - ch * 10);  // layer

    if (la == 0)
      continue;  // layer == 0 means whole chamber...

    int channel = 1;

    // Build iterator which loops on all channels:
    vector<CSCGains::Item>::const_iterator gain_i;

    for (gain_i = it->second.begin(); gain_i != it->second.end(); ++gain_i) {
      thegain = gain_i->gain_slope;
      weight = AvgStripGain / thegain;

      //      if (debug) std::cout << "the weight is " << weight << std::endl;

      if (weight <= 0.)
        weight = -10.;  // ignore strips with no gain computed
      if (channel % 100 == 1)
        weight0 = weight;  // can't make comparison with first strip, so set diff to zero

      // Fill corrections factor for all strip at once
      histo = All_CSC;
      histo->Fill(weight, weight0, channel, la);

      // Now fill correction factor for given chamber
      // Get station
      if (st == 1) {  // station 1

        if (rg == 1) {  // ring 1
          if (ch == 27) {
            histo = ME_11_27;
          } else if (ch == 28) {
            histo = ME_11_28;
          } else if (ch == 29) {
            histo = ME_11_29;
          } else if (ch == 30) {
            histo = ME_11_30;
          } else if (ch == 31) {
            histo = ME_11_31;
          } else {
            histo = ME_11_32;
          }
        } else if (rg == 2) {  // ring 2
          if (ch == 27) {
            histo = ME_12_27;
          } else if (ch == 28) {
            histo = ME_12_28;
          } else if (ch == 29) {
            histo = ME_12_29;
          } else if (ch == 30) {
            histo = ME_12_30;
          } else if (ch == 31) {
            histo = ME_12_31;
          } else {
            histo = ME_12_32;
          }
        } else {  // ring 3
          if (ch == 27) {
            histo = ME_13_27;
          } else if (ch == 28) {
            histo = ME_13_28;
          } else if (ch == 29) {
            histo = ME_13_29;
          } else if (ch == 30) {
            histo = ME_13_30;
          } else if (ch == 31) {
            histo = ME_13_31;
          } else {
            histo = ME_13_32;
          }
        }

      } else if (st == 2) {  // station 2

        if (rg == 1) {  // ring 1
          if (ch == 14) {
            histo = ME_21_14;
          } else if (ch == 15) {
            histo = ME_21_15;
          } else {
            histo = ME_21_16;
          }
        } else {  // ring 2
          if (ch == 27) {
            histo = ME_22_27;
          } else if (ch == 28) {
            histo = ME_22_28;
          } else if (ch == 29) {
            histo = ME_22_29;
          } else if (ch == 30) {
            histo = ME_22_30;
          } else if (ch == 31) {
            histo = ME_22_31;
          } else {
            histo = ME_22_32;
          }
        }

      } else {  // station 3

        if (rg == 1) {  // ring 1
          if (ch == 14) {
            histo = ME_31_14;
          } else if (ch == 15) {
            histo = ME_31_15;
          } else {
            histo = ME_31_16;
          }
        } else {  // ring 2
          if (ch == 27) {
            histo = ME_32_27;
          } else if (ch == 28) {
            histo = ME_32_28;
          } else if (ch == 29) {
            histo = ME_32_29;
          } else if (ch == 30) {
            histo = ME_32_30;
          } else if (ch == 31) {
            histo = ME_32_31;
          } else {
            histo = ME_32_32;
          }
        }
      }
      histo->Fill(weight, weight0, channel, la);
      weight0 = weight;
      channel++;
    }
  }
}

/* getStripGainAvg
 *
 */
float CSCGainsStudy::getStripGainAvg() {
  int n_strip = 0;
  float gain_tot = 0.;
  float gain_avg = -1.;

  // Build iterator which loops on all layer id:
  map<int, vector<CSCGains::Item> >::const_iterator it;

  for (it = pGains->gains.begin(); it != pGains->gains.end(); ++it) {
    const unsigned ChId = it->first;
    unsigned offset = (ChId - 220000000);
    unsigned ec = offset / 100000;                                            // endcap
    unsigned st = (offset - ec * 100000) / 10000;                             // station
    unsigned rg = (offset - ec * 100000 - st * 10000) / 1000;                 // ring
    unsigned ch = (offset - ec * 100000 - st * 10000 - rg * 1000) / 10;       // chamber
    unsigned la = (offset - ec * 100000 - st * 10000 - rg * 1000 - ch * 10);  // layer

    if (la == 0)
      continue;  // layer == 0 means whole chamber...

    // Build iterator which loops on all channels:
    vector<CSCGains::Item>::const_iterator gain_i;

    for (gain_i = it->second.begin(); gain_i != it->second.end(); ++gain_i) {
      // Make sure channel isn't dead, otherwise don't include in average !
      if (gain_i->gain_slope > 0.) {
        gain_tot += gain_i->gain_slope;
        n_strip++;
      }
    }
  }
  // Average gain
  if (n_strip > 0)
    gain_avg = gain_tot / n_strip;

  return gain_avg;
}

DEFINE_FWK_MODULE(CSCGainsStudy);
