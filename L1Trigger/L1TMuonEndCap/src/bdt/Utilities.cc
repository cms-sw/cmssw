//                            Utilities.cxx                             //
// =====================================================================//
//                                                                      //
//                     Various helpful functions.                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////
// _______________________Includes_______________________________________//
///////////////////////////////////////////////////////////////////////////

#include "L1Trigger/L1TMuonEndCap/interface/bdt/Utilities.h"

#include "TRandom3.h"
#include "TStopwatch.h"
#include "TTree.h"
#include "TNtuple.h"
#include "TFile.h"
#include "TChain.h"
#include "TMath.h"

using namespace emtf;

//////////////////////////////////////////////////////////////////////////
// ------------------Some Helpful Arrays----------------------------------
//////////////////////////////////////////////////////////////////////////

// Array of GeV values for error calculation.
const double emtf::ptscale[31] =  { 0,
                 1.5,   2.0,   2.5,   3.0,   3.5,   4.0,
                 4.5,   5.0,   6.0,   7.0,   8.0,  10.0,  12.0,  14.0,
                 16.0,  18.0,  20.0,  25.0,  30.0,  35.0,  40.0,  45.0,
                 50.0,  60.0,  70.0,  80.0,  90.0, 100.0, 120.0, 140.0 };

const std::vector<double> ptScale = std::vector<double>(ptscale, ptscale + sizeof ptscale / sizeof ptscale[0]);

// Array of counts for error calculation.
const double emtf::twoJets_scale[16] =  { 0,
                       0.5,   1.0,   2.0,   3.0,   4.0,   5.0, 10.0, 20.0, 50.0,
                       100,   500,   1000,   5000,   7500,  50000};

const std::vector<double> emtf::twoJetsScale = std::vector<double>(twoJets_scale, twoJets_scale + sizeof twoJets_scale / sizeof twoJets_scale[0]);



//////////////////////////////////////////////////////////////////////////
// ------------------Some Helpful Functions-------------------------------
//////////////////////////////////////////////////////////////////////////

float processPrediction(float BDTPt, int Quality, float PrelimFit)
{
// Discretize and scale the BDTPt prediction


    // Fix terrible predictions
    if(BDTPt < 0) BDTPt = PrelimFit;
    if(BDTPt > 250) BDTPt = PrelimFit;

    float BDTPt1 = BDTPt;
    float scaleF = 1.0;

    // Scale based upon quality
    if (Quality == 3) scaleF = 1.15;
    if (Quality == 2) scaleF = 1.3;
    if (Quality == 1) scaleF = 1.7;

    BDTPt1 = scaleF*BDTPt1;


    // Discretize based upon ptscale
    for (int pts=0; pts<31; pts++)
    {
      if (ptscale[pts]<=BDTPt1 && ptscale[pts+1]>BDTPt1)
      {
        BDTPt1 = ptscale[pts];
        break;
      }
    }

    if (BDTPt1 > 140) BDTPt1 = 140;
    if (BDTPt1 < 0) BDTPt1 = 0;

    return BDTPt1;
}

/////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////

void mergeNtuples(const char* ntuplename, const char* filestomerge, const char* outputfile)
{
   TChain chain(ntuplename);
   chain.Add(filestomerge);
   chain.Merge(outputfile);
}

//////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////

void sortNtupleByEvent(const char* ntuplename, const char* filenametosort, const char* outputfile)
{
        //TFile f("../../all_test_redux_post.root");
        TFile f(filenametosort);
        TNtuple *tree = (TNtuple*)f.Get(ntuplename);
        int nentries = (int)tree->GetEntries();
        //Drawing variable pz with no graphics option.
        //variable pz stored in array fV1 (see TTree::Draw)
        tree->Draw("Event","","goff");
        int *index = new int[nentries];
        //sort array containing pz in decreasing order
        //The array index contains the entry numbers in decreasing order
        TMath::Sort(nentries,tree->GetV1(),index);

        //open new file to store the sorted Tree
        //TFile f2("../../test_events_sorted.root","recreate");
        TFile f2(outputfile,"recreate");

        //Create an empty clone of the original tree
        TTree *tsorted = (TTree*)tree->CloneTree(0);
        for (int i=0;i<nentries;i++) {
                tree->GetEntry(index[i]);
                tsorted->Fill();
        }
        tsorted->Write();
        delete [] index;
}
