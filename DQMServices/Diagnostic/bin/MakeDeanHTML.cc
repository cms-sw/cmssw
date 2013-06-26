////////////////////////////////////////////////////////////////////
//
// Dean Andrew Hidas <dhidas@fnal.gov>
//
// Created on: Thu Aug 27 11:57:16 CEST 2009
//
// Usage:
//   MakeDeanHTML [InFile] [OutFile.html]
//
// Input Format:
//  The input file format has the following formatting:
//    line 1: specify col widths (the number of widths specified also
//            indicates the number of columns
//    The next lines if beginning with a "#' are printed directly
//    After any '#'s:
//      All lines are of the format: PlotName: Caption
//      There must be exactly the number of entries corresponding
//      to the number of cols followed by a blank line.  Lines can be
//      blank where you don't want a plot (for formatting reasons)
//
//    Example of valid input file:
//      320 320 320
//      #<h1>MyTitle</h1>
//      #<p>maybe some links here or whatever</p>
//      fig1a.gif: This is cap1a
//      fig1b.gif: This is cap1b
//      fig1c.gif: This is cap1c
//     
//      fig2a.gif: This is cap2a
//     
//      fig2c.gif: This is cap2c
//     
//      fig3a.gif: This is cap3a
//      fig3b.gif: This is cap3b
//      fig3c.gif: This is cap3c
//
//
////////////////////////////////////////////////////////////////////


#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>





int MakeDeanHTML (std::string const& InFileName, std::string const& OutFileName)
{
  // open the input file
  std::ifstream InFile(InFileName.c_str());
  if (!InFile.is_open()) {
    std::cerr << "ERROR: cannot open input file" << std::endl;
    return 1;
  }

  // open the output file
  std::ofstream OutFile(OutFileName.c_str());
  if (!OutFile.is_open()) {
    std::cerr << "ERROR: cannot open output file" << std::endl;
    return 1;
  }

  // print header
  OutFile << "<html>\n<body>\n\n";

  // Read first 3 line which contains widths (and is the number of cols)
  std::string FirstLine;
  std::getline(InFile, FirstLine);
  std::istringstream LineStream;
  LineStream.str(FirstLine);
  std::string Width;
  std::vector<std::string> Widths;
  while (LineStream >> Width) {
    Widths.push_back(Width);
  }
  while (InFile.peek() == '#') {
    std::getline(InFile, FirstLine);
    OutFile << std::string(FirstLine.begin()+1, FirstLine.end()) << std::endl;
  }

  OutFile << "<hr>\n<table>\n\n";

  while (!InFile.eof()) {
    OutFile << "  <tr>\n";
    std::string OneLine;
    for (size_t iCol = 0; iCol != Widths.size(); ++iCol) {
      std::getline(InFile, OneLine);
      bool const BlankLine = OneLine == "" ? true : false;

      OutFile << "    <td width=\"" << Widths[iCol] << "\">\n";
      if (!BlankLine) {
        std::string const PlotName(OneLine.begin(), OneLine.begin()+OneLine.find(":"));
        std::string const ThumbName = std::string(PlotName.begin(), PlotName.begin() + PlotName.find(".gif")) + "_small.gif";
        std::string const Caption(OneLine.begin()+OneLine.find(":")+1, OneLine.end());
        OutFile << "      <center>\n";
        OutFile << "        <a href=\"" << PlotName << "\">\n";
        OutFile << "        <img src=\"" << ThumbName << "\">\n";
        OutFile << "        </a><br>\n";
        OutFile << "        " << Caption << "\n";
        OutFile << "      </center>\n";
      }
      OutFile << "    </td>\n";
    }
    OutFile << "  </tr>\n\n";
    // Get a blank line
    std::getline(InFile, OneLine);
  }

  OutFile << "</table>\n\n</body>\n</html>";

  OutFile.close();
  InFile.close();

  return 0;
}


int main (int argc, char* argv[])
{
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " [InFile] [OutFile]" << std::endl;
    return 1;
  }

  MakeDeanHTML(argv[1], argv[2]);

  return 0;
}
