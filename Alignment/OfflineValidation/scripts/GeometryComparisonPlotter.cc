#include "GeometryComparisonPlotter.h"

/***********************************************************************************/
/*                            GEOMETRY COMPARISON PLOTTER                          */
/* See the talk of 15 January 2015 for short documentation and the example script. */
/* This code is highly commented if need be to upgrade it.                         */
/* Any further question is to be asked to Patrick Connor (patrick.connor@desy.de). */
/*                                             Thanks a million <3                 */
/***********************************************************************************/

// NOTE: look for "TO DO" as a keyword to now what should be upgraded in later versions.... 


// modes
#define TALKATIVE
//#define DEBUG

// MACROS
#define INSIDE_VECTOR(vector) \
    cout << #vector << "={"; for (unsigned int i = 0 ; i < vector.size()-1 ; i++) cout << vector[i] << ","; cout << vector.back() << "}";
#define CHECK_MAP_CONTENT(m,type) \
    for (map<TString,type>::iterator it = m.begin() ; it != m.end() ; it++) \
        cout << __FILE__ << ":" << __LINE__ << ":Info:\t" << #m << "[" << it->first << "]=" << it->second << endl;

// CONSTRUCTOR AND DESTRUCTOR
GeometryComparisonPlotter::GeometryComparisonPlotter (TString tree_file_name,
                                                      TString output_directory) :
    _output_directory(output_directory + TString(output_directory.EndsWith("/") ? "" : "/")),
    _output_filename("comparison.root"),
    _print_option("pdf"),
    _print(true),       // print the graphs in a file (e.g. pdf)
    _legend(true),      // print the graphs in a file (e.g. pdf)
    _write(true),       // write the graphs in a root file
    _batchMode(
#ifdef DEBUG
            false        // false = display canvases (very time- and resource-consuming)
#else
            true         // true = no canvases
#endif
            ),           
    _1dModule(true),    // cut on 1d modules
    _2dModule(true),    // cut on 2d modules
    _levelCut (DEFAULT_LEVEL),      // module level (see branch of same name)
    _window_width(DEFAULT_WINDOW_WIDTH),
    _window_height(DEFAULT_WINDOW_HEIGHT)
{
#ifdef TALKATIVE
    cout << ">>> TALKATIVE MODE ACTIVATED <<<" << endl;
#endif
#ifdef DEBUG
    cout << ">>> DEBUG MODE ACTIVATED <<<" << endl;
    cout << __FILE__ << ":"<< __LINE__ << ":Info: inside constructor of GeometryComparisonPlotter"<< endl;
#endif

    //_sublevel_names = {"PXB", "PXF", "TIB", "TID", "TOB", "TEC"}; // C++11
    _sublevel_names[0] = TString("PXB");
    _sublevel_names[1] = TString("PXF");
    _sublevel_names[2] = TString("TIB");
    _sublevel_names[3] = TString("TID");
    _sublevel_names[4] = TString("TOB");
    _sublevel_names[5] = TString("TEC");
    // TO DO: handle other structures

    // read tree
    tree_file = new TFile(tree_file_name, "UPDATE");
    data = (TTree*) tree_file->Get("alignTree");
    // int branches
    data->SetBranchAddress("id"         ,&branch_i["id"]);      
    data->SetBranchAddress("mid"        ,&branch_i["mid"]);     
    data->SetBranchAddress("level"      ,&branch_i["level"]);   
    data->SetBranchAddress("mlevel"     ,&branch_i["mlevel"]);  
    data->SetBranchAddress("sublevel"   ,&branch_i["sublevel"]);
    data->SetBranchAddress("useDetId"   ,&branch_i["useDetId"]);
    data->SetBranchAddress("detDim"     ,&branch_i["detDim"]);  
    // float branches
    data->SetBranchAddress("x"          ,&branch_f["x"]);       
    data->SetBranchAddress("y"          ,&branch_f["y"]);       
    data->SetBranchAddress("z"          ,&branch_f["z"]);       
    data->SetBranchAddress("alpha"      ,&branch_f["alpha"]);   
    data->SetBranchAddress("beta"       ,&branch_f["beta"]);    
    data->SetBranchAddress("gamma"      ,&branch_f["gamma"]);   
    data->SetBranchAddress("phi"        ,&branch_f["phi"]);     
    data->SetBranchAddress("eta"        ,&branch_f["eta"]);     
    data->SetBranchAddress("r"          ,&branch_f["r"]);       
    data->SetBranchAddress("dx"         ,&branch_f["dx"]);      
    data->SetBranchAddress("dy"         ,&branch_f["dy"]);      
    data->SetBranchAddress("dz"         ,&branch_f["dz"]);      
    data->SetBranchAddress("dphi"       ,&branch_f["dphi"]);    
    data->SetBranchAddress("dr"         ,&branch_f["dr"]);	    
    data->SetBranchAddress("dalpha"     ,&branch_f["dalpha"]);  
    data->SetBranchAddress("dbeta"      ,&branch_f["dbeta"]);   
    data->SetBranchAddress("dgamma"     ,&branch_f["dgamma"]);  
    if (data->GetBranch("rdphi") == 0x0) // in the case of rdphi branch not existing, it is created from r and dphi branches
    { 
#ifdef TALKATIVE
        cout << __FILE__ << ":" << __LINE__ << ":Info:\tComputing the rdphi branch from r and dphi branches (assuming they exist...)" << endl;
#endif
        TBranch * br_rdphi = data->Branch("rdphi", &branch_f["rdphi"], "rdphi/F");
        for (unsigned int ientry = 0 ; ientry < data->GetEntries() ; ientry++)
        {
            data->GetEntry(ientry);
            branch_f["rdphi"] = branch_f["r"]*branch_f["dphi"];
            br_rdphi->Fill();
        }
    }
    else
        data->SetBranchAddress("rdphi",&branch_f["rdphi"]);

#ifdef DEBUG
    cout << __FILE__ << ":" << __LINE__ << ":Info: branch addresses set" << endl;
#endif

    // style
    gROOT->Reset();

    data->SetMarkerSize(0.5);
    data->SetMarkerStyle(6);

    gStyle->SetOptStat("emr");
    gStyle->SetTitleAlign(22);
    gStyle->SetTitleX(0.5);
    gStyle->SetTitleY(0.97);
    gStyle->SetTitleFont(62); 
    //gStyle->SetOptTitle(0);

    gStyle->SetTextFont(132);
    gStyle->SetTextSize(0.08);
    gStyle->SetLabelFont(132,"x");
    gStyle->SetLabelFont(132,"y");
    gStyle->SetLabelFont(132,"z");
    gStyle->SetTitleSize(0.08,"x");
    gStyle->SetTitleSize(0.08,"y");
    gStyle->SetTitleSize(0.08,"z");
    gStyle->SetLabelSize(0.08,"x");
    gStyle->SetLabelSize(0.08,"y");
    gStyle->SetLabelSize(0.08,"z");

    gStyle->SetMarkerStyle(8);
    gStyle->SetHistLineWidth(2); 
    gStyle->SetLineStyleString(2,"[12 12]"); // postscript dashes

    gStyle->SetFrameBorderMode(0);
    gStyle->SetCanvasBorderMode(0);
    gStyle->SetPadBorderMode(0);
    gStyle->SetPadColor(0);
    gStyle->SetCanvasColor(0);
    gStyle->SetTitleColor(1);
    gStyle->SetStatColor(0);
    gStyle->SetStatBorderSize(1);
    gStyle->SetFrameFillColor(0);

    gStyle->SetPadTickX(1);
    gStyle->SetPadTickY(1);

    gStyle->SetPadTopMargin(0.1);
    gStyle->SetPadRightMargin(0.05);
    gStyle->SetPadBottomMargin(0.16);
    gStyle->SetPadLeftMargin(0.18);

#ifdef DEBUG
    cout << __FILE__ << ":" << __LINE__ << ":Info: end of constructor" << endl;
#endif
}

GeometryComparisonPlotter::~GeometryComparisonPlotter ()
{
#ifdef DEBUG
    cout << __FILE__ << ":" << __LINE__ << endl;
#endif
    tree_file->Close();
#ifdef DEBUG
    cout << __FILE__ << ":" << __LINE__ << endl;
#endif
}

// MAIN METHOD
void GeometryComparisonPlotter::MakePlots (vector<TString> x, // axes to combine to plot
                                           vector<TString> y) // every combination (except the ones such that x=y) will be perfomed
{
    /// -1) check that only existing branches are called 
    // (we use a macro to avoid copy/paste)
#define CHECK_BRANCHES(branchname_vector) \
    for (unsigned int i = 0 ; i < branchname_vector.size() ; i++) \
    {   \
        if (branch_f.find(branchname_vector[i]) == branch_f.end()) \
        {   \
            cout << __FILE__ << ":" << __LINE__ << ":Error: The branch " << branchname_vector[i] << " is not recognised." << endl; \
            return; \
        }   \
    }
    CHECK_BRANCHES(x);
    CHECK_BRANCHES(y);

    const unsigned int nentries = data->GetEntries();

#ifdef TALKATIVE
    cout << __FILE__ << ":" << __LINE__ << ":Info:\t";    INSIDE_VECTOR(x);   cout << endl
         << __FILE__ << ":" << __LINE__ << ":Info:\t";    INSIDE_VECTOR(y);   cout << endl;
#endif

    /// 0) min and max values
    // the max and min of the graphs are computed from the tree if they have not been manually input yet
    // (we use a macro to avoid copy/paste)
#define LIMITS(axes_vector) \
    for (unsigned int i = 0 ; i < axes_vector.size() ; i++) \
    {   \
        if ( _SF.find(axes_vector[i]) == _SF.end() )  _SF[axes_vector[i]] = 1.; \
        if (_min.find(axes_vector[i]) == _min.end()) _min[axes_vector[i]] = _SF[axes_vector[i]]*data->GetMinimum(axes_vector[i]); \
        if (_max.find(axes_vector[i]) == _max.end()) _max[axes_vector[i]] = _SF[axes_vector[i]]*data->GetMaximum(axes_vector[i]); \
    }
    LIMITS(x);
    LIMITS(y);

#ifdef TALKATIVE 
    CHECK_MAP_CONTENT(_min,float);
    CHECK_MAP_CONTENT(_max,float);
    CHECK_MAP_CONTENT(_SF ,float);
#endif

    /// 1) declare TGraphs
    // the idea is to produce at the end a table of 7 TMultiGraphs:
    // - 0=Tracker, with color code for the different sublevels
    // - 1..6=different sublevels, with color code for z < or > 0
    // (convention: the six first (resp. last) correspond to z>0 (resp. z<0))
    // This means that 2*6 TGraphs will be filled during the loop on the TTree,
    // and will be arranged differently with different color codes in the TMultiGraphs
#ifndef NB_SUBLEVELS
#define NB_SUBLEVELS 6
#endif
#define NB_Z_SLICES 2
    TGraph * graphs[x.size()][y.size()][NB_SUBLEVELS*NB_Z_SLICES];
    int ipoint[x.size()][y.size()][NB_SUBLEVELS*NB_Z_SLICES];
    for (unsigned int ix = 0 ; ix < x.size() ; ix++)
    { 
        for (unsigned int iy = 0 ; iy < y.size() ; iy++)
        {
            //if (x[ix] == y[iy]) continue;       // do not plot graphs like (r,r) or (phi,phi)
            for (unsigned int igraph = 0 ; igraph < NB_SUBLEVELS*NB_Z_SLICES ; igraph++)
            {
                // declaring
                ipoint[ix][iy][igraph] = 0; // the purpose of an index for every graph is to avoid thousands of points at the origin of each
                graphs[ix][iy][igraph] = new TGraph ();
#define COLOR_CODE(icolor) int(icolor/4)+icolor+1
                graphs[ix][iy][igraph]->SetMarkerColor(COLOR_CODE(igraph));
                graphs[ix][iy][igraph]->SetMarkerStyle(6);
                // pimping
                graphs[ix][iy][igraph]->SetName (x[ix]+y[iy]+_sublevel_names[igraph%NB_SUBLEVELS]+TString(igraph>NB_SUBLEVELS ? "n"      : "p"       ));
                graphs[ix][iy][igraph]->SetTitle(            _sublevel_names[igraph%NB_SUBLEVELS]+TString(igraph>NB_SUBLEVELS ? " at z<0": " at z>=0")
                                                + TString (";") + LateXstyle(x[ix]) + " /" + _units[x[ix]]
                                                + TString (";") + LateXstyle(y[iy]) + " /" + _units[y[iy]]);
                graphs[ix][iy][igraph]->SetMarkerStyle(6);
                // setting ranges
                graphs[ix][iy][igraph]->GetXaxis()->SetLimits   (_min[x[ix]],_max[x[ix]]);
                graphs[ix][iy][igraph]->GetYaxis()->SetRangeUser(_min[y[iy]],_max[y[iy]]);
            }
        }
    }
#ifdef DEBUG
    cout << __FILE__ << ":" << __LINE__ << ":Info: Creation of the TGraph[" << x.size() << "][" << y.size() << "][" << NB_SUBLEVELS*NB_Z_SLICES << "] ended." << endl;
#endif

    /// 2) loop on the TTree data
#ifdef DEBUG
    cout << __FILE__ << ":" << __LINE__ << ":Info: Looping on the TTree" << endl;
#endif
#ifdef TALKATIVE
    unsigned int progress = 0;
    cout << __FILE__ << ":" << __LINE__ << ":Info: 0%" << endl;
#endif
    for (unsigned int ientry = 0 ; ientry < nentries ; ientry++)
    {
#ifdef  TALKATIVE
        if (10*ientry/nentries != progress)
        {
            progress = 10*ientry/nentries;
            cout << __FILE__ << ":" << __LINE__ << ":Info: " << 10*progress << "%" << endl;
        }
#endif
        // load current tree entry
        data->GetEntry(ientry);

        // CUTS on entry
        if (branch_i["level"] != _levelCut)        continue;
        if (!_1dModule && branch_i["detDim"] == 1) continue;
        if (!_2dModule && branch_i["detDim"] == 2) continue;

        // loop on the different couples of variables to plot in a graph
        for (unsigned int ix = 0 ; ix < x.size() ; ix++)
        {
            // CUTS on x[ix]
            if (branch_f[x[ix]] > _max[x[ix]] || branch_f[x[ix]] < _min[x[ix]]) continue;

            for (unsigned int iy = 0 ; iy < y.size() ; iy++)
            {
                // CUTS on y[iy]
                //if (x[ix] == y[iy])                                                   continue; // TO DO: handle display when such a case occurs
                if (branch_i["sublevel"] <= 0 || branch_i["sublevel"] > NB_SUBLEVELS) continue;
                if (branch_f[y[iy]] > _max[y[iy]] || branch_f[y[iy]] < _min[y[iy]])   continue;

                // FILLING GRAPH
                const int igraph = (branch_i["sublevel"]-1) + (branch_f["z"]>=0?0:NB_SUBLEVELS);
                graphs[ix][iy][igraph]->SetPoint(ipoint[ix][iy][igraph],
                                                 _SF[x[ix]]*branch_f[x[ix]],
                                                 _SF[y[iy]]*branch_f[y[iy]]);
                ipoint[ix][iy][igraph]++;
            }
        }
    }
#ifdef TALKATIVE
    cout << __FILE__ << ":" << __LINE__ << ":Info: 100%\tLoop ended" << endl;
#endif

    /// 3) merge TGraph objects into TMultiGraph objects, then draw, print and write (according to the options _batchMode, _print and _write respectively)
    gROOT->SetBatch(_batchMode); // if true, then equivalent to "root -b", i.e. no canvas
    if (_write)
    {   // opening the file to write the graphs
        output = new TFile(_output_directory+TString(_output_filename), "UPDATE"); // possibly existing file will be updated, otherwise created
        if (output->IsZombie())
        {
            cout << __FILE__ << ":" << __LINE__ << ":Error: Opening of " << _output_directory+TString(_output_filename) << " failed" << endl;
            exit(-1);
        }
#ifdef TALKATIVE
        cout << __FILE__ << ":"<< __LINE__ << ":Info: output file is " << _output_directory+TString(_output_filename) << endl;
#endif
    }
    // declaring TMultiGraphs and TCanvas
    TMultiGraph * mgraphs[x.size()][y.size()][1+NB_SUBLEVELS];
    TCanvas * c[x.size()][y.size()][1+NB_SUBLEVELS],
            * c_global[1+NB_SUBLEVELS];
    canvas_index++; // this index is a safety used in case the MakePlots method is used several times to avoid overloading
    // declaration of the c_global canvases
    c_global[0] = new TCanvas (TString::Format("global_tracker_%d", canvas_index),
                               "Global overview of the Tracker variables",
                               _window_width,
                               _window_height);
    c_global[0]->Divide(x.size(),y.size());
    for (unsigned int ic = 1 ; ic <= NB_SUBLEVELS ; ic++)
    {
        c_global[ic] = new TCanvas (TString("global") + _sublevel_names[ic-1] + TString::Format("_%d", canvas_index),
                                    TString("Global overview of the ") + _sublevel_names[ic-1] + TString(" variables"),
                                   _window_width,
                                   _window_height);
        c_global[ic]->Divide(x.size(),y.size());
    }
    // creating TLegend
    TLegend * legend = MakeLegend(.1,.92,.9,1.);
    if (_write) legend->Write();
    // running on the TGraphs to produce the TMultiGraph and draw/print them
    for (unsigned int ix = 0 ; ix < x.size() ; ix++)
    {
#ifdef DEBUG
        cout << __FILE__ << ":" << __LINE__ << ":Info: x[" << ix << "]="<< x[ix] << endl;
#endif
        for (unsigned int iy = 0 ; iy < y.size() ; iy++)
        {
#ifdef DEBUG
            cout << __FILE__ << ":" << __LINE__ << ":Info: x[" << ix << "]=" << x[ix]
                                                <<   " and y[" << iy << "]=" << y[iy] 
                                                <<   "\t-> creating TMultiGraph" << endl;
#endif
            mgraphs[ix][iy][0] = new TMultiGraph (x[ix] + y[iy] + TString::Format("tracker_%d", canvas_index),                  // name
                                                  //LateXstyle(x[ix]) + TString(" vs. ") + LateXstyle(y[iy]) + TString(" for Tracker") // graph title
                                                    TString (";") + LateXstyle(x[ix]) + " /" + _units[x[ix]]                     // x axis title
                                                  + TString (";") + LateXstyle(y[iy]) + " /" + _units[y[iy]]);                   // y axis title
            // TO DO: see title in canvases

            /// TRACKER
            // filling TMultiGraph
            //for (unsigned int jgraph = 0 ; jgraph < NB_SUBLEVELS*NB_Z_SLICES ; jgraph++)
            for (unsigned int jgraph = NB_SUBLEVELS*NB_Z_SLICES ; jgraph > 0 ; jgraph--) // reverse order has been chosen for humane readability
            {
                if (_write) graphs[ix][iy][jgraph-1]->Write();
                TGraph * gr = (TGraph *) graphs[ix][iy][jgraph-1]->Clone();
                int colorjgraph = jgraph>NB_SUBLEVELS ? jgraph-NB_SUBLEVELS : jgraph;
                gr->SetMarkerColor((int(colorjgraph/4)+colorjgraph+1)-1);
                mgraphs[ix][iy][0]->Add(gr, "p");
            }
            // writing the 6-colour tracker canvas into root file
            if (_write) mgraphs[ix][iy][0]->Write();

            // drawing the standalone tracker canvas 
            c[ix][iy][0] = new TCanvas ("c_" + x[ix] + y[iy] + TString::Format("tracker_%d", canvas_index),
                                        "",
                                        _window_width,
                                        _window_height);
            mgraphs[ix][iy][0]->Draw("a");
            if (_legend) legend->Draw();

            // drawing the global tracker canvas
#define INDEX_IN_GLOBAL_CANVAS(i1,i2) 1 + i1 + i2*x.size()
            c_global[0]->GetPad(INDEX_IN_GLOBAL_CANVAS(ix,iy))->SetFillStyle(4000); //  make the pad transparent
            c_global[0]->cd(INDEX_IN_GLOBAL_CANVAS(ix,iy)); // see TCanvas::Divide() to understand this formula
            mgraphs[ix][iy][0]->Draw("a");

            if (_print) c[ix][iy][0]->Print(_output_directory + mgraphs[ix][iy][0]->GetName() + ExtensionFromPrintOption(_print_option), _print_option);
            
            /// SUBLEVELS (1..6)
            for (unsigned int isublevel = 1 ; isublevel <= NB_SUBLEVELS ; isublevel++)
            {
                mgraphs[ix][iy][isublevel] = new TMultiGraph (x[ix] + y[iy] + _sublevel_names[isublevel-1]  + TString::Format("_%d", canvas_index),  // name
                                                             //LateXstyle(x[ix]) + TString(" vs. ") + LateXstyle(y[iy]) + TString(" for ") +
                                                              _sublevel_names[isublevel-1]                                 // graph title
                                                              + TString (";") + LateXstyle(x[ix]) + " /" + _units[x[ix]]   // x axis title
                                                              + TString (";") + LateXstyle(y[iy]) + " /" + _units[y[iy]]); // y axis title
                 graphs[ix][iy][             isublevel-1]->SetMarkerColor(kBlack);
                 graphs[ix][iy][NB_SUBLEVELS+isublevel-1]->SetMarkerColor(kRed);
                mgraphs[ix][iy][isublevel]->Add(graphs[ix][iy][             isublevel-1], "p"); // z>0
                mgraphs[ix][iy][isublevel]->Add(graphs[ix][iy][NB_SUBLEVELS+isublevel-1], "p"); // z<0
#if NB_Z_SLICES!=2
                cout << __FILE__ << ":" << __LINE__ << ":Warning: color code incomplete for Z slices..." << endl;
#endif
            }
            // draw and write the sublevels (bicolour plots)
            for (unsigned int isublevel = 1 ; isublevel <= NB_SUBLEVELS ; isublevel++)
            {
                // writing into root file
                if (_write) mgraphs[ix][iy][isublevel]->Write();

                // drawing standalone canvas for current sublevel
                c[ix][iy][isublevel] = new TCanvas ("c_" + x[ix] + y[iy] + TString::Format("%u_%d", isublevel, canvas_index),
                                                    "",
                                                    _window_width,
                                                    _window_height);
                mgraphs[ix][iy][isublevel]->Draw("a");
                // here no legend is necessary, as there are only two colours: black for z postive and red for x negative

                // drawing into the global canvas for current sublevel
                c_global[isublevel]->cd(INDEX_IN_GLOBAL_CANVAS(ix,iy));
                mgraphs[ix][iy][isublevel]->Draw("a");

                // printing pdf
                if (_print) c[ix][iy][isublevel]->Print(_output_directory + mgraphs[ix][iy][isublevel]->GetName() + ExtensionFromPrintOption(_print_option), _print_option);
            }
        } // end of loop on y
    }     // end of loop on x
    // setting legend to tracker global canvas
    gStyle->SetOptTitle(0); // otherwise, the title is repeated in every pad of the global canvases -> instead, we will write it in the upper part
    for (unsigned int ic = 0 ; ic <= NB_SUBLEVELS ; ic++)
    {
        c_global[ic]->Draw();
        TCanvas * c_temp = (TCanvas *) c_global[ic]->Clone(c_global[ic]->GetTitle() + TString("_sub"));
        c_temp->Draw();
        c_global[ic] = new TCanvas (c_temp->GetName() + TString("_final"), c_temp->GetTitle(), c_temp->GetWindowWidth(), c_temp->GetWindowHeight());
        c_global[ic]->Draw();
        TPad * p_up = new TPad (TString("legend_") + c_temp->GetName(), "",
                                    0., 0.9, 1., 1., // relative position
                                    -1, 0, 0),       // display options
             * p_down = new TPad (TString("main_") + c_temp->GetName(), "",
                                    0., 0., 1., 0.9,
                                    -1, 0, 0);
        // in the lower part, draw the plots
        p_down->Draw();
        p_down->cd();
        c_temp->DrawClonePad();
        c_global[ic]->cd();
        // in the upper part, pimp the canvas :p
        p_up->Draw();
        p_up->cd();
        if (ic == 0) // tracker
        {
            if (_legend)
            {
                TLegend * global_legend = MakeLegend(.05,.1,.95,.8);//, "brNDC");
                global_legend->Draw();
            }
            else
            {
            TPaveText * pt = new TPaveText(.05,.1,.95,.8, "NB");
            pt->SetLineWidth(0);
            pt->SetLineColor(c_temp->GetFillColor());
            pt->AddText("Tracker");
            pt->Draw();
            }
        }
        else // sublevels
        {
            TPaveText * pt = new TPaveText(.05,.1,.95,.8, "NB");
            pt->SetFillColor(0);
            pt->AddText(_sublevel_names[ic-1]);
            pt->Draw();
        }
        // printing
        if (_print) c_global[ic]->Print(_output_directory + c_global[ic]->GetName() + ExtensionFromPrintOption(_print_option), _print_option);
        if (_write) c_global[ic]->Write();
    }
    
    // printing global canvases
    if (_write) output->Close();
#ifdef TALKATIVE
    cout << __FILE__ << ":" << __LINE__ << ":Info: End of MakePlots method" << endl;
#endif

}

// OPTION METHODS
void GeometryComparisonPlotter::SetPrint               (const bool kPrint)             { _print            = kPrint               ; }
void GeometryComparisonPlotter::SetLegend              (const bool kLegend)            { _legend           = kLegend              ; }
void GeometryComparisonPlotter::SetWrite               (const bool kWrite)             { _write            = kWrite               ; }
void GeometryComparisonPlotter::Set1dModule            (const bool k1dModule)          { _1dModule         = k1dModule            ; }
void GeometryComparisonPlotter::Set2dModule            (const bool k2dModule)          { _2dModule         = k2dModule            ; }
void GeometryComparisonPlotter::SetLevelCut            (const int  kLevelCut)          { _levelCut         = kLevelCut            ; }
void GeometryComparisonPlotter::SetBatchMode           (const bool kBatchMode)         { _batchMode        = kBatchMode           ; }
void GeometryComparisonPlotter::SetBranchMax           (const TString branchname,
                                                        const float max)               { _max[branchname]   = max                  ; }
void GeometryComparisonPlotter::SetBranchMin           (const TString branchname,
                                                        const float min)               { _min[branchname]   = min                  ; }
void GeometryComparisonPlotter::SetBranchSF            (const TString branchname,
                                                        const float SF)                { _SF[branchname]    = SF                   ; }
void GeometryComparisonPlotter::SetBranchUnits         (const TString branchname,
                                                        const TString units)           { _units[branchname] = units                ; }
void GeometryComparisonPlotter::SetPrintOption         (const Option_t * print_option) { _print_option      = print_option         ; }
void GeometryComparisonPlotter::SetCanvasSize          (const int window_width,
                                                        const int window_height)       { _window_width      = window_width         ; 
                                                                                         _window_height     = window_height        ; }
void GeometryComparisonPlotter::SetOutputFileName      (const TString name)            { _output_filename   = name                 ; }
void GeometryComparisonPlotter::SetOutputDirectoryName (const TString name)            { _output_directory  = name                
                                                                                          + TString(name.EndsWith("/") ? "" : "/") ; }

// PRIVATE METHODS
TString GeometryComparisonPlotter::LateXstyle (TString word)
{
    word.ToLower();
    if (word.BeginsWith("d"))          word.ReplaceAll("d", "#Delta");
    if      (word == TString("rdphi")) word = "r#Delta#phi";            // TO DO: find something less ad hoc...
    else if (word.EndsWith("phi"))     word.ReplaceAll("phi", "#phi");
    else if (word.EndsWith("alpha"))   word.ReplaceAll("alpha", "#alpha");
    else if (word.EndsWith("beta"))    word.ReplaceAll("beta" , "#beta");
    else if (word.EndsWith("gamma"))   word.ReplaceAll("gamma", "#gamma");
    else if (word.EndsWith("eta"))     word.ReplaceAll("eta", "#eta");
    return word;
}

TString GeometryComparisonPlotter::ExtensionFromPrintOption (TString print_option)
{
         if (print_option.Contains("pdf" ))  return TString(".pdf" );
    else if (print_option.Contains("eps" ))  return TString(".eps" );
    else if (print_option.Contains("ps"  ))  return TString(".ps"  );
    else if (print_option.Contains("svg" ))  return TString(".svg" );
    else if (print_option.Contains("tex" ))  return TString(".tex" );
    else if (print_option.Contains("gif" ))  return TString(".gif" );
    else if (print_option.Contains("xpm" ))  return TString(".xpm" );
    else if (print_option.Contains("png" ))  return TString(".png" );
    else if (print_option.Contains("jpg" ))  return TString(".jpg" );
    else if (print_option.Contains("tiff"))  return TString(".tiff");
    else if (print_option.Contains("cxx" ))  return TString(".cxx" );
    else if (print_option.Contains("xml" ))  return TString(".xml" );
    else if (print_option.Contains("root"))  return TString(".root");
    else 
    {
        cout << __FILE__ << ":" << __LINE__ << ":Warning: unknown format. Returning .pdf, but possibly wrong..." << endl;
                                             return TString(".pdf");
    }
}

TLegend * GeometryComparisonPlotter::MakeLegend (double x1,
                                                 double y1,
                                                 double x2,
                                                 double y2,
                                                 const TString title)
{
    TLegend * legend = new TLegend (x1, y1, x2, y2, title.Data(), "NBNDC");
    legend->SetNColumns(NB_SUBLEVELS);
    legend->SetFillColor(0);
    legend->SetLineColor(0); // redundant with option
    legend->SetLineWidth(0); // redundant with option
    for (unsigned int i = 0 ; i < NB_SUBLEVELS ; i++)
    {
        TGraph * g = new TGraph (0);
        g->SetMarkerColor(COLOR_CODE(i));
        g->SetFillColor(COLOR_CODE(i));
        g->SetMarkerStyle(kFullSquare);
        legend->AddEntry(g,_sublevel_names[i], "p");
    }
    return legend;
}
