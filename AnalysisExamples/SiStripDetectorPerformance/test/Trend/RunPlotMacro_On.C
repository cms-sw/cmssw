RunPlotMacro_On(char* input, char *output, bool run=false)
{
if(!gROOT->LoadMacro("plotMacro_On.C"))
    PlotMacro(input,output);
}
