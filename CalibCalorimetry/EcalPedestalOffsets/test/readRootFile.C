// trivial ROOT macro to read the detailed output 
// produced by the pedoffset scan when plotting 
// the trends of the pedestals versus DAC values
// the file name is details.root, 
// in this case SM1 is studied and crystal 1698, gain 2
// are plotted 

{
    TFile * detail = new TFile ("../details.root") ;
    detail->cd ("SM1") ;
    TGraphErrors * graph ;
    gDirectory->GetObject ("XTL1698_GAIN2;1",graph) ;
    graph->Draw ("AP*") ;
}
