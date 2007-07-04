RunTIFSummaryTree_Off(char* input, char* output, bool run=false)
{
  if(!gROOT->LoadMacro("TIFSummaryTree_Off.C+"))
    TIFSummaryTree(input,output);
}
