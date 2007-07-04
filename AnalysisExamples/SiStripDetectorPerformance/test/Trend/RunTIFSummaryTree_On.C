RunTIFSummaryTree_On(char* input, char* output, bool run=false)
{
  if(!gROOT->LoadMacro("TIFSummaryTree_On.C+"))
    TIFSummaryTree(input,output);
}
