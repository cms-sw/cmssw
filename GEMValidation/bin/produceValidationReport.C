#include "iostream"
#include "fstream"
#include "string"

using namespace std;

int main( int argc, char* argv[] )
{
  string identifier("20130130");
  string fileName( "GEMValidationReport_" + identifier + ".tex");
  ofstream file;
  file.open(fileName.c_str());

  file << "\\documentclass[11pt]{article}" << endl
       << "\\usepackage{a4wide}" << endl
       << "\\usepackage[affil-it]{authblk}" << endl
       << "\\usepackage{amsmath}" << endl
       << "\\usepackage{amsfonts}" << endl
       << "\\usepackage{amssymb}" << endl
       << "\\usepackage{makeidx}" << endl
       << "\\usepackage{graphicx}" << endl
       << "\\usepackage[T1]{fontenc}" << endl
       << "\\usepackage[utf8]{inputenc}" << endl
       << "\\usepackage{hyperref}" << endl
       << endl
       << "\\title{\\LARGE\\textbf{CMS GEM Collaboration} \\\\[0.2cm] \\Large (GEMs for CMS) \\\\[0.2cm] \\LARGE\\textbf{Validation Report I}}" << endl
       << endl
       << "\\author[1]{Yasser~Assran}" << endl
       << "\\author[2]{Othmane~Bouhali}" << endl
       << "\\author[3]{Sven~Dildick}" << endl
       << "\\author[4]{Will~Flanagan}" << endl
       << "\\author[4]{Vadim~Khotilovich}" << endl
       << "\\author[4]{Alexei~Safonov}" << endl
       << endl
       << "\\affil[1]{ASRT-ENHEP (Egypt)}"  << endl
       << "\\affil[2]{ITS Research Computing, Texas A\\&M University at Qatar (Qatar)}" << endl
       << "\\affil[2]{Department of Physics and Astronomy, Ghent University (Belgium)}" << endl
       << "\\affil[4]{Department of Experimental High Energy Physics, Texas A\\&M University (USA)}" << endl
       << endl
       << "\\date{\\today}" << endl
       << endl
       << "\\begin{document}" << endl
       << "\\maketitle" << endl
       << endl
       << "\\begin{center}" << endl
       << "Contact GEM Validation group: \\href{mailto:gem-sim-validation@cern.ch}{gem-sim-validation@cern.ch}" << endl
       << "\\end{center}" << endl
       << endl
       << "\\section{Introduction}" << endl
       << "\\begin{itemize}" << endl
       << "\\item Data set path for simhit" << endl
       << "\\item data set path for digi" << endl
       << "\\item digitizer model (CVS version)" << endl
       << "\\end{itemize}" << endl
       << "\\section{SimHit validation plots}" << endl
       << "\\section{Digi validation plots}" << endl
       << "\\section{Efficiency plots}" << endl
       << endl
       << "\\end{document}" << endl;
  
  file.close();

  return 0;
}
